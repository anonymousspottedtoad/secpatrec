"""The one-shot baseline: pack, prompt, one call, parse — and its batch runner.

See the package docstring for what the baseline is and why it is built this way.
`secpat baseline` (in ``cli.py``) is the command that drives it.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import tiktoken

from ..llm import make_agent
from .pack import build_pack, iter_files

# Provider credentials come from the repo's .env, loaded by this package's
# `__init__` before anything here runs — the same .env `secpat recover` reads.

PROVIDER = "openai"
#: A thinking model bills its reasoning as output alongside the answer, so the
#: cap has to cover both — 8K left one record 700 tokens of headroom.
MAX_TOKENS = 32_000
#: Refuse to send a pack larger than this. The window is 1M and the model still
#: has to answer inside it, so the prompt cannot have all of it.
MAX_INPUT_TOKENS = 900_000

#: The whole prompt: what to do, and the exact JSON to answer with. Every value
#: is a placeholder describing the field, so the schema needs no comments the
#: model could copy into its answer and no second document to keep in step.
SYSTEM = """You are a security-pattern recovery analyst.

You are given one security pattern as JSON and the full source of one repository.
For every role in the pattern, decide whether the repository implements it and
ground it to the exact code, then give the order those roles interact in when the
repository runs.

Answer with one JSON object of exactly this shape, and nothing else:

{
  "role_mappings": [
    {
      "patternRole": "the pattern's role name, verbatim",
      "status": "present | external | missing",
      "entity": "the code entity's bare declared name",
      "file": "repo-relative path",
      "lines": "the entity's declaration line",
      "notes": "why this code represents the role"
    }
  ],
  "interaction_order": [
    {
      "from": "role that invokes",
      "to": "role it invokes",
      "file": "repo-relative file holding the call site",
      "line": 0,
      "notes": "the call, in a few words"
    }
  ]
}

"role_mappings" holds one entry per role in the pattern, none left out:

  status   "present" for in-repo code; "external" for a library,
           framework or service the repo only calls or configures; "missing"
           if not found.
  entity   The bare name as declared, with no enclosing class or module
           qualifier - "authenticate_user", or "hashpw" for a library symbol.
           Logic with no symbol of its own is cited by the innermost named
           entity containing it. Empty only for a role not found.
  file     Where the entity is declared; for a library symbol, where the repo
           imports or calls it. Empty if the role is missing.
  lines    The entity's single declaration line, e.g. "12". Empty if missing.

"interaction_order" lists the interactions in execution order, one entry each,
its "line" the call site's line number or 0 where that cannot be pinpointed.
It is [] if the order cannot be told."""


def parse_answer(text: str) -> dict:
    """The model's reply, as the answer object.

    The reply is asked for as bare JSON, but a model that fences it in
    ```json … ``` or writes a sentence around it has still answered; the object
    runs from the first `{` to the last `}`. Anything else raises, and the record
    fails — a half-parsed recovery is not this baseline's result.
    """
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        head = " ".join(text.split())[:200]
        raise ValueError(f"model answered with no JSON object: {head or '<empty>'}")
    try:
        answer = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"model answered with invalid JSON: {exc}") from exc
    if not isinstance(answer, dict) or "role_mappings" not in answer:
        raise ValueError("model's JSON has no 'role_mappings'")
    return answer


def as_envelope(rows: list[dict], repo: str) -> dict:
    """The model's claimed order, in the envelope `secpat recover` writes.

    Field for field the pipeline's `interaction_order`, so a baseline run and a
    pipeline run can be diffed directly. `backend: "model"` and
    `mechanism: "model-claim"` are what keep the two distinguishable: the
    pipeline's entries are call-graph edges carrying the line that evidences
    them, these are the model's assertion about the same code.
    """
    return {
        "available": bool(rows),
        "backend": "model",
        "requestedBackend": "model",
        "language": "",
        "repo": repo,
        "reason": "" if rows else "the model returned no interactions",
        "entryRole": rows[0]["from"] if rows else "",
        "entryRule": "model",
        "entryCandidates": [],
        "interactions": [
            {
                "rank": i,
                "interaction": f"{row['from']} \u2192 {row['to']}",
                "from": row["from"],
                "to": row["to"],
                # A model that omits a field (Muse dropped "line" on one row)
                # gets the schema's "not pinpointed" value, not a lost record.
                "file": row.get("file", ""),
                "line": row.get("line", 0),
                "via": [],
                "mechanism": "model-claim",
                "notes": row.get("notes", ""),
            }
            for i, row in enumerate(rows, 1)
        ],
        "anchoredRoles": [],
        "skippedRoles": [],
        "coLocatedRoles": [],
    }


def build_prompt(pattern_json: str, pack_text: str) -> str:
    """The one user message: the pattern, then the repository. The instruction
    and the output schema are the prompt above it, `SYSTEM`."""
    return f"<pattern>\n{pattern_json}\n</pattern>\n\n<repository>\n{pack_text}\n</repository>"


def prompt_tokens(prompt: str, provider_name: str) -> tuple[int, bool]:
    """`(tokens, exact)` for the pre-flight budget check.

    Every provider is counted with GPT's `o200k_base`, the one base tokenizer
    the fit guard uses: exact for `openai` (it matches GPT-6.1-sol's billed
    input to within 0.4%), an estimate for the rest, which publish no tokenizer
    (Muse) or are not counted with their own (DeepSeek, GLM).
    """
    tokens = len(tiktoken.get_encoding("o200k_base").encode(prompt, disallowed_special=()))
    return tokens, provider_name == "openai"


def recover(
    codebase: Path,
    pattern_file: Path,
    *,
    provider_name: str,
    model: str | None,
    all_text: bool,
    max_file_bytes: int,
    max_input_tokens: int,
    max_tokens: int,
) -> dict:
    """Pack the repo, make one call, return the run record."""
    provider = make_agent(provider_name, model=model, max_tokens=max_tokens)

    pack = build_pack(
        codebase, all_text=all_text, line_numbers=False, max_file_bytes=max_file_bytes
    )
    prompt = build_prompt(pattern_file.read_text(encoding="utf-8"), pack.text)

    tokens, exact = prompt_tokens(SYSTEM + prompt, provider_name)
    if tokens > max_input_tokens:
        raise ValueError(
            f"pack is {tokens:,}{'' if exact else ' (estimated)'} tokens, over the "
            f"{max_input_tokens:,} limit — this repo does not fit a one-shot call"
        )

    started = time.time()
    text = provider.response(SYSTEM, prompt)
    seconds = round(time.time() - started, 1)

    answer = parse_answer(text or "")
    return {
        "role_mappings": answer.get("role_mappings", []),
        "interaction_order": as_envelope(answer.get("interaction_order", []), codebase.name),
        "baseline": {
            "provider": provider_name,
            "model": provider.model_id,
            "codebase": str(codebase),
            "pattern_file": str(pattern_file),
            "pack_files": pack.files,
            "pack_chars": pack.chars,
            "prompt_tokens": tokens,
            "prompt_tokens_estimated": not exact,
            # The provider's own billed counts for the one call (USAGE_KEYS);
            # prompt_tokens above is only the pre-flight budget estimate.
            "usage": provider.last_usage,
            "seconds": seconds,
        },
    }


def write_run(out_dir: Path, result: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "recovery.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


# ── planning (no API calls) ──────────────────────────────────────────────────

#: Chars per token, measured over the 30 annotated repos (2.5–2.9 across the set).
#: Only ever used for planning — a run counts its own prompt exactly before sending.
_CHARS_PER_TOKEN = 2.8
#: A reference list price (Claude Opus 5's), USD per input token: the dry run's
#: rough cost column, the same for every provider.
_INPUT_USD = 5 / 1_000_000


def pack_bytes(root: Path, *, all_text: bool, max_file_bytes: int) -> tuple[int, int]:
    """`(files, bytes)` for a repo's pack, from `stat` alone — no file is read.

    Ranking repos by size must not cost what packing them costs: reading all 30
    to find the 5 smallest reads the 25 that are then discarded, and this set
    includes a 230 MB one. File sizes are within a rounding error of the pack's
    own length (the markers add ~60 bytes per file) and rank identically.
    """
    packed, _ = iter_files(root, all_text=all_text, max_file_bytes=max_file_bytes)
    return len(packed), sum(p.stat().st_size for p in packed)


def plan(
    records: list[dict], *, all_text: bool, max_file_bytes: int
) -> list[tuple[dict, int, int]]:
    """`(record, files, bytes)` per record, smallest first."""
    sized = [
        (
            rec,
            *pack_bytes(
                Path(rec["codebase_path"]), all_text=all_text, max_file_bytes=max_file_bytes
            ),
        )
        for rec in records
    ]
    return sorted(sized, key=lambda row: row[2])


def print_plan(rows: list[tuple[dict, int, int]]) -> None:
    total = 0
    print(f"{'record':<62}{'files':>7}{'~tokens':>12}{'~USD':>8}")
    for rec, files, size in rows:
        tokens = int(size / _CHARS_PER_TOKEN)
        total += tokens
        print(
            f"{rec['pattern'] + '__' + rec['repo']:<62}{files:>7}"
            f"{tokens:>12,}{tokens * _INPUT_USD:>8.2f}"
        )
    print(f"{'total':<62}{'':>7}{total:>12,}{total * _INPUT_USD:>8.2f}")
    print(
        "\nEstimates only (input tokens at list price, output excluded);"
        " each run counts its own prompt exactly before sending."
    )


# ── batch ────────────────────────────────────────────────────────────────────


def run_batch(records: list[dict], batch_dir: Path, **options) -> int:
    """Recover every record into ``batch_dir``; ``options`` are :func:`recover`'s keywords."""
    failures = 0
    for i, rec in enumerate(records, 1):
        slug = f"{rec['pattern']}__{rec['repo']}"
        print(f"[{i}/{len(records)}] {slug}", flush=True)
        out_dir = batch_dir / slug
        if (out_dir / "recovery.json").is_file():
            # Resuming: this record cost a call already, and re-running it would
            # buy a different answer to the same question rather than a missing one.
            print("    already recovered, skipped", flush=True)
            continue
        try:
            result = recover(Path(rec["codebase_path"]), Path(rec["pattern_file"]), **options)
        except (ValueError, OSError) as exc:
            failures += 1
            print(f"    skipped: {exc}", file=sys.stderr, flush=True)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "error.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
            continue
        write_run(out_dir, result)
        meta = result["baseline"]
        print(
            f"    {len(result['role_mappings'])} roles ·"
            f" {meta['prompt_tokens']:,} prompt tokens ·"
            f" {meta['seconds']}s",
            flush=True,
        )
    print(f"\nbatch: {batch_dir}")
    if failures:
        print(f"{failures}/{len(records)} record(s) produced no recovery.json", file=sys.stderr)
    return 1 if failures == len(records) else 0


