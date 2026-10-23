"""Tool definitions and executors for the recovery agent.

Holds the raw JSON-Schema tool specs (`RAW_TOOLS`), the Thought-augmented specs
actually sent to the provider (`TOOLS`), and the executor for each tool.

`build_tools(disabled=...)` returns that same list with named tools removed —
what the tool-ablation runs are (`--disable-tool grep`, `--disable-tool search`).
A disabled tool is *absent*, not stubbed: it is missing from the schemas sent to
the provider and from the prompt's tool list, so the model never learns it
existed and cannot spend a step discovering it errors.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable

from ..kb.builder import KBIndex
from ..llm.base import ToolUse
from ..repo_access import RepoAccess

# ── Tool definitions ──────────────────────────────────────────────────────────

RAW_TOOLS: list[dict] = [
    {
        "name": "search",
        "description": (
            "Search the indexed codebase by meaning. Returns the line-numbered "
            "source of each strongly matching symbol; weaker matches come back "
            "as locations only, without code."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Natural-language description of what you are looking "
                        "for, e.g. 'JWT token validation middleware'."
                    ),
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "grep",
        "description": (
            "Find an exact string anywhere in the repository. Returns matching "
            "'path:line: content' lines; when there are more than fit, the true "
            "total and a per-file breakdown come with them."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": (
                        "Literal text to find, e.g. 'SESSION_TTL'. Treated as an "
                        "exact substring unless regex is true."
                    ),
                },
                "glob": {
                    "type": "string",
                    "description": (
                        "Optional glob narrowing the search. One containing '/' "
                        "is anchored at the repo root ('src/**/*.py'); a bare "
                        "'*.py' matches at any depth. Omit to search everything."
                    ),
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Case-sensitive matching. Defaults to true.",
                },
                "regex": {
                    "type": "boolean",
                    "description": ("Treat pattern as a regular expression. Defaults to false."),
                },
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "read_file",
        "description": (
            "Read a file from the codebase by repo-relative path, with 1-based "
            "line numbers. Optionally restrict to a line range. At most 400 lines "
            "are returned per call."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Repo-relative file path, e.g. 'src/auth/service.py'.",
                },
                "start_line": {
                    "type": "integer",
                    "description": "First line to return (1-based). Defaults to 1.",
                },
                "end_line": {
                    "type": "integer",
                    "description": "Last line to return (inclusive). Defaults to end of file.",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "list_dir",
        "description": (
            "List a directory recursively as an indented tree of every "
            "subdirectory and file beneath it. A large tree is trimmed to the "
            "deepest level that fits and says so; call again on a subdirectory "
            "to expand that branch."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Repo-relative directory path, e.g. 'src/auth'. "
                        "Defaults to '.' (the codebase root)."
                    ),
                },
                "max_depth": {
                    "type": "integer",
                    "description": (
                        "Optional cap on how many levels below `path` to show; "
                        "0 = immediate contents only."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "submit_mapping",
        "description": (
            "Submit the final role mapping and end the run. Call this once every "
            "pattern role is accounted for."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "role_mappings": {
                    "type": "array",
                    "description": "One entry per pattern role.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "patternRole": {
                                "type": "string",
                                "description": "Role name from the pattern.",
                            },
                            "status": {
                                "type": "string",
                                "enum": [
                                    "present",
                                    "external",
                                    "missing",
                                ],
                                "description": (
                                    "'present' — in-repo code. "
                                    "'external' — a library, framework, or "
                                    "service. 'missing' — not found."
                                ),
                            },
                            "entity": {
                                "type": "string",
                                "description": (
                                    "Bare name of the entity as declared, with no "
                                    "enclosing class/module qualifier — e.g. "
                                    "'authenticate_user', or 'hashpw' for a "
                                    "library symbol. Prefer a method over its "
                                    "class. Empty only if missing."
                                ),
                            },
                            "file": {
                                "type": "string",
                                "description": "Repo-relative path; empty if missing.",
                            },
                            "lines": {
                                "type": "string",
                                "description": (
                                    "The entity's single declaration line, e.g. "
                                    "'12'. Empty if missing."
                                ),
                            },
                            "notes": {
                                "type": "string",
                                "description": "Why this code represents the role.",
                            },
                        },
                        "required": [
                            "patternRole",
                            "status",
                            "entity",
                            "file",
                            "lines",
                            "notes",
                        ],
                    },
                },
            },
            "required": [
                "role_mappings",
            ],
        },
    },
]

# ── Strict ReAct: Thought is a required argument of every Action ──────────────

_THOUGHT_PROPERTY = {
    "type": "string",
    "description": (
        "REQUIRED. Your Thought for this step: the reasoning that led to this "
        "call, written before it and starting from what the last Observation "
        "established. Two or three sentences, not a restatement of the arguments."
    ),
}


def with_thought(tools: list[dict]) -> list[dict]:
    """Add a required `thought` argument to every tool's input schema.

    This is what makes the loop strictly ReAct rather than ReAct-shaped. A native
    tool-use API lets the model emit a bare tool call with no accompanying text,
    so a prompt asking it to "think first" is only advisory — and models routinely
    skip it. Promoting Thought to a required schema field moves the constraint to
    the API's own validation: an Action literally cannot be emitted without its
    Thought, on either provider.
    """
    out: list[dict] = []
    for tool in tools:
        schema = copy.deepcopy(tool["input_schema"])
        # Thought first so it is generated before the arguments it justifies.
        schema["properties"] = {
            "thought": _THOUGHT_PROPERTY,
            **schema.get("properties", {}),
        }
        schema["required"] = ["thought", *schema.get("required", [])]
        out.append({**tool, "input_schema": schema})
    return out


TOOLS: list[dict] = with_thought(RAW_TOOLS)

# ── Tool ablation ─────────────────────────────────────────────────────────────

#: Every tool the agent can be given, in the order they are offered.
TOOL_NAMES: list[str] = [t["name"] for t in RAW_TOOLS]

#: The tools an ablation may remove. `submit_mapping` is not one of them: it is
#: how the loop terminates, so removing it ablates the run, not a capability.
ABLATABLE_TOOLS: list[str] = [n for n in TOOL_NAMES if n != "submit_mapping"]


def build_tools(disabled: Iterable[str] = ()) -> list[dict]:
    """The Thought-augmented tool schemas, minus every tool in `disabled`.

    Raises `ValueError` on an unknown or non-ablatable name — a typo'd
    `--disable-tool` must fail the run, not silently produce a control arm
    indistinguishable from the real one in the results.
    """
    names = [n.strip() for n in disabled if n and n.strip()]
    unknown = [n for n in names if n not in ABLATABLE_TOOLS]
    if unknown:
        raise ValueError(
            f"Cannot disable {', '.join(repr(n) for n in unknown)}. "
            f"Ablatable tools: {', '.join(ABLATABLE_TOOLS)}."
        )
    drop = set(names)
    kept = [t for t in RAW_TOOLS if t["name"] not in drop]
    return with_thought(kept)


def submission_problems(args: dict) -> list[str]:
    """Required fields a `submit_mapping` call left out, one line per gap.

    The schema's `required` lists are checked here as well as by the provider,
    because not every provider enforces them: a Responses API tool sent with
    `strict: False` (OpenAI, Meta's Muse) is accepted with fields missing. An
    empty string is a value, not a gap — a missing role's `entity` is empty by
    design.
    """
    schema = next(t for t in RAW_TOOLS if t["name"] == "submit_mapping")["input_schema"]
    problems = [f"missing `{k}`" for k in schema["required"] if k not in args]
    rows = args.get("role_mappings")
    if "role_mappings" in args and not isinstance(rows, list):
        problems.append("`role_mappings` is not a list")
        rows = []
    row_required = schema["properties"]["role_mappings"]["items"]["required"]
    for i, row in enumerate(rows or []):
        if not isinstance(row, dict):
            problems.append(f"role_mappings[{i}] is not an object")
            continue
        missing = [k for k in row_required if k not in row]
        if missing:
            label = row.get("patternRole") or row.get("entity") or "?"
            problems.append(
                f"role_mappings[{i}] ({label}): missing {', '.join(f'`{k}`' for k in missing)}"
            )
    return problems


def split_thought(tu: ToolUse) -> tuple[str, dict]:
    """Separate the ReAct Thought from the Action's real arguments."""
    args = dict(tu.input)
    thought = str(args.pop("thought", "") or "").strip()
    return thought, args


# ── Tool execution ────────────────────────────────────────────────────────────


# Search returns code, not chunk ids, so its caps bound the observation size:
# only confident matches are shown, at most a handful, each trimmed to a
# readable excerpt. Anything longer is a read_file call away.
_MIN_SCORE = 0.7
_MAX_HITS = 5
_MAX_CODE_LINES = 80


def _numbered(code: str, start_line: int) -> tuple[str, bool]:
    """Render a chunk's code with 1-based file line numbers; trim if long."""
    lines = code.splitlines()
    truncated = len(lines) > _MAX_CODE_LINES
    shown = lines[:_MAX_CODE_LINES]
    width = len(str(start_line + len(shown) - 1))
    body = "\n".join(f"{start_line + i:>{width}}| {text}" for i, text in enumerate(shown))
    return body, truncated


def _follow_ups(enabled: Iterable[str] | None) -> tuple[str, str, bool]:
    """The three follow-up hints a search observation carries, for this tool set.

    Search results point the agent at its *other* tools — grep this identifier,
    read_file that path — so an ablation arm would otherwise be told, over and
    over mid-run, to use the tool it does not have. `enabled` None means the full
    set and reproduces the wording byte for byte, so a control run's observations
    are unchanged.
    """
    have = None if enabled is None else set(enabled)
    has_grep = have is None or "grep" in have
    has_read = have is None or "read_file" in have

    leads = ["Rephrase the query in the codebase's own vocabulary"]
    if has_grep:
        leads.append("grep for a literal identifier")
    if has_read:
        leads.append("read_file one of the paths above")
    lead_hint = ", ".join(leads[:-1]) + f", or {leads[-1]}." if len(leads) > 1 else f"{leads[0]}."

    tail = (
        "\nThese are symbol-sized excerpts. Before grounding a role, read_file the "
        "surrounding file for full context and to confirm the declaration line."
        if has_read
        else "\nThese are symbol-sized excerpts. Before grounding a role, confirm the "
        "declaration line against the excerpt above."
    )
    return lead_hint, tail, has_read


def exec_search(kb: KBIndex, query: str, enabled: Iterable[str] | None = None) -> str:
    lead_hint, tail_hint, has_read = _follow_ups(enabled)
    results = kb.search(query, top_k=10)
    if not results:
        return "No results found. The knowledge base returned nothing for this query."

    hits = [r for r in results if r["score"] > _MIN_SCORE][:_MAX_HITS]
    if not hits:
        # Nothing confident enough to read. Show the near misses as leads only —
        # location and shape, no code — so the next move can be a grep or a
        # read_file rather than a reworded search.
        leads = [
            f"  {r['symbol']} ({r['kind']}) — {r['path']}:{r['start_line']}  score={r['score']}"
            for r in results[:5]
        ]
        return "\n".join(
            [
                f"No chunk scored above the {_MIN_SCORE} relevance threshold for "
                f"{query!r}, so no code is shown.",
                "",
                "Closest (weak) matches, as leads only:",
                *leads,
                "",
                lead_hint,
            ]
        )

    out = [
        f"{len(hits)} match(es) above the {_MIN_SCORE} relevance threshold for "
        f"{query!r} (of {len(results)} candidates), with source:"
    ]
    for r in hits:
        out.append(
            f"\n─── {r['symbol']} ({r['kind']}) — {r['path']}"
            f"  lines {r['start_line']}-{r['end_line']}  score={r['score']}"
        )
        if r.get("docstring"):
            out.append(f"docstring: {r['docstring']}")
        if r.get("calls"):
            out.append(f"calls: {', '.join(r['calls'][:5])}")
        body, truncated = _numbered(r.get("code", ""), r["start_line"])
        out.append(body)
        if truncated:
            out.append(
                f"... [trimmed at {_MAX_CODE_LINES} lines — "
                f"read_file('{r['path']}', {r['start_line']}, {r['end_line']}) "
                "for the rest]"
                if has_read
                else f"... [trimmed at {_MAX_CODE_LINES} lines]"
            )
    out.append(tail_hint)
    return "\n".join(out)


def exec_grep(repo: RepoAccess | None, args: dict) -> str:
    if repo is None:
        return "[ERROR] grep unavailable: no codebase root configured for this run."
    return repo.grep(
        str(args.get("pattern", "")),
        glob=args.get("glob") or None,
        case_sensitive=bool(args.get("case_sensitive", True)),
        is_regex=bool(args.get("regex", False)),
    )


def exec_list_dir(repo: RepoAccess | None, args: dict) -> str:
    if repo is None:
        return "[ERROR] list_dir unavailable: no codebase root configured for this run."
    depth = args.get("max_depth")
    return repo.list_dir(
        str(args.get("path") or "."),
        max_depth=int(depth) if depth is not None else None,
    )


def exec_read_file(repo: RepoAccess | None, args: dict) -> str:
    if repo is None:
        return "[ERROR] read_file unavailable: no codebase root configured for this run."
    return repo.read_file(
        str(args.get("path", "")),
        start_line=args.get("start_line"),
        end_line=args.get("end_line"),
    )


def fmt_input(tu: ToolUse) -> str:
    """One-line rendering of an Action's arguments, for the progress log."""
    if tu.name == "search":
        return f"query={tu.input.get('query', '')!r}"
    if tu.name == "grep":
        glob = tu.input.get("glob")
        scope = f", glob={glob!r}" if glob else ""
        return f"pattern={tu.input.get('pattern', '')!r}{scope}"
    if tu.name == "list_dir":
        depth = tu.input.get("max_depth")
        extra = f", max_depth={depth}" if depth is not None else ""
        return f"path={tu.input.get('path', '.')!r}{extra}"
    if tu.name == "read_file":
        span = ""
        if tu.input.get("start_line") or tu.input.get("end_line"):
            span = f", lines={tu.input.get('start_line', '')}-{tu.input.get('end_line', '')}"
        return f"path={tu.input.get('path', '')!r}{span}"
    return ""


__all__ = [
    "RAW_TOOLS",
    "TOOLS",
    "TOOL_NAMES",
    "ABLATABLE_TOOLS",
    "build_tools",
    "with_thought",
    "split_thought",
    "exec_search",
    "exec_grep",
    "exec_list_dir",
    "exec_read_file",
    "fmt_input",
]
