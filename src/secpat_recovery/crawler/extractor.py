from __future__ import annotations

import json
import logging
import re

from ..llm.base import LLMProvider
from .fetcher import DiagramSubflow, PatternPage, clean_page_text

log = logging.getLogger(__name__)

_FULL_SYSTEM = """You are a diagram transcription expert. You are given a single diagram \
image from the DistriNet (KU Leuven) security-pattern catalogue together with the pattern's \
description text. The image is the authoritative depiction of the pattern.

These images are box-and-arrow collaboration diagrams (roles drawn as boxes, connected by \
numbered message arrows) — NOT lifeline sequence diagrams. Transcribe the WHOLE diagram into \
ONE Mermaid `flowchart` (graph) that reproduces the image completely — every element, \
nothing added, nothing left out. Do NOT split it into separate diagrams and do NOT use \
`sequenceDiagram`.

Represent the image as a graph:
- One NODE per role / participant / box, with a valid single-token id and a bracketed label \
holding the real (possibly multi-word) name: `Enforcer["Enforcer"]`, \
`PasswordStore["Password store"]`.
- One directed EDGE per arrow, labelled with the arrow's EXACT message text INCLUDING its \
full step prefix/number (`1.`, `Reg.2`, `Res.4`, …): \
`Subject -->|"1. action(id, pwd[])"| Enforcer`.

Follow this METHOD so that no edge is missed and every direction is correct:
1. First, list every box (role) in the image. Create one node for each.
2. Then walk the arrows IN STEP-NUMBER ORDER (1, 2, 3 … then `Reg.1`, `Reg.2` … then \
`Res.1`, `Res.2` …). Every step number visible in the image MUST become exactly one edge — \
if you see labels up to 15, you must output 15 numbered edges, with none skipped and none \
duplicated.
3. For EACH arrow determine its direction by the ARROWHEAD: the edge goes FROM the box at \
the arrow's tail TO the box the arrowhead points at. Write `Source -->|"label"| Target` in \
exactly that direction. A reply/return arrow points back the other way — keep it as its own \
edge in its own direction. Do not guess direction from left/right position; read the \
arrowhead. When unsure, cross-check the numbered step against the PATTERN DESCRIPTION, which \
states who sends each message to whom.
4. Include return/response arrows, guards/conditions (e.g. `[ok]`, `[error]`), and any notes \
(add a note as its own node or fold it into the relevant edge label).
5. Before finishing, re-scan the image and confirm every arrow you can see has a matching \
edge and that each edge's endpoints and direction match the arrowhead.

Preserve label wording exactly as shown; do NOT summarise, merge two arrows into one, rename, \
or invent anything. If some text is partly illegible, transcribe your best reading.

Mermaid rules: start with `flowchart LR` (or `flowchart TD`). Node ids must be single tokens \
with NO spaces — always put the readable name in a bracketed `["..."]` label. ALWAYS wrap \
edge labels in double quotes because they contain digits, dots, parentheses, and commas: \
`A -->|"..."| B`.

Output ONLY the Mermaid source, starting with `flowchart`. No prose, no code fences."""

_FULL_USER = """Pattern: {name}.

--- PATTERN DESCRIPTION ---
{text}
--- END PATTERN DESCRIPTION ---

Transcribe the ENTIRE attached diagram into one complete Mermaid `flowchart` now: one node \
per role/box, one directed edge per arrow, every edge label (with its step prefix/number) \
quoted and exact. Capture every node, every arrow, and every label — do not miss any \
detail."""


_SUBFLOW_SYSTEM = """You are given the full Mermaid `flowchart` transcription of a security \
pattern's diagram (roles as nodes, numbered messages as directed edges) plus the pattern's \
description text. This is a TEXT task — work only from the flowchart; there is no image.

Every edge label starts with a step PREFIX — a number or tag such as `1.`, `2.`, `Reg.2`, or \
`Res.4`. Edges that share a prefix family (`1..N`, `Reg.*`, `Res.*`, …) form ONE sub-flow.

A family is decided by the prefix ALONE, never by topic. Plainly numbered edges `1`, `2`, \
`3`, … `N` are ONE single family however many they are and however many different roles they \
touch — they are the consecutive steps of one walkthrough, so NEVER cut a plain numeric chain \
into thematic groups ("verification", "enforcement", "update", …). A new family starts only \
at a distinct non-numeric tag (`Reg.*`, `Res.*`, `A.*`, …) or a restart of the numbering. \
Branch conditions and repeats do not start a family either: `[allow] 9: …` and `[block] 9: \
Error` are both step 9 of the numeric family. An edge with NO step prefix at all (e.g. a \
dotted `Policy changes` annotation) belongs to no family — leave it out entirely.

Do the following:
1. Group the flowchart's edges into sub-flows by their prefix family.
2. For each sub-flow emit a Mermaid `sequenceDiagram` containing exactly that family's edges, \
ordered by step number. Convert each directed edge `A -->|"n: msg"| B` into a message \
`A->>B: n: msg`, preserving the flowchart's direction EXACTLY (source → target). Use `-->>` \
when the label is clearly a reply/return/response, otherwise `->>`. Keep EVERY edge of the \
family — none may be dropped, merged, or duplicated — with its label verbatim.
3. Give each sub-flow a short descriptive NAME grounded in the description (e.g. "Runtime \
authentication", "Subject registration", "Password reset") — never a generic "Flow 1".

Participant ids must be single Mermaid-valid tokens with NO spaces; for a multi-word node \
name use an alias — `participant PasswordStore as Password store` — and reference the id in \
every message. Do NOT rename roles or invent edges. If the flowchart has only one prefix \
family, return a single sub-flow.

Output ONLY a JSON object of exactly this shape — no prose, no code fences:
{"subflows": [{"name": "<human name>", "prefix": "<edge prefix>", \
"mermaid": "<Mermaid source>"}]}"""

_SUBFLOW_USER = """--- PATTERN DESCRIPTION ---
{text}
--- END PATTERN DESCRIPTION ---

--- FULL DIAGRAM (Mermaid flowchart) ---
{mermaid}
--- END FULL DIAGRAM ---

Split the flowchart into sub-flows by edge-prefix family, name each from the description, and \
return the JSON now. Every PREFIXED edge in the flowchart must appear in exactly one \
sub-flow, with its label and direction unchanged."""


_MAIN_SYSTEM = """You are a security-pattern analyst. You are given a pattern's description \
text and a numbered list of candidate sub-flows extracted from its diagram. Exactly ONE of \
them is the MAIN flow — the core interaction the pattern is fundamentally about, i.e. the \
primary runtime behaviour laid out in the description's main 'Solution' / 'Behaviour' \
narrative (the numbered steps). The others are auxiliary or lifecycle flows (e.g. \
registration, password change / reset, key rotation, provisioning, setup).

Choose the single sub-flow that best corresponds to that main behaviour. Judge by the \
sub-flow names and edge prefixes against the description — the main flow is the one whose \
messages match the description's principal numbered walkthrough.

Output ONLY JSON, no prose, no code fences: {"index": <zero-based index of the main sub-flow>}"""

_MAIN_USER = """--- PATTERN DESCRIPTION ---
{text}
--- END PATTERN DESCRIPTION ---

Candidate sub-flows (index. name — edge prefix):
{catalogue}

Which one is the MAIN flow described by the text? Return {{"index": n}}."""


def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t[3:]
        if t.startswith("mermaid"):
            t = t[len("mermaid") :]
    if t.endswith("```"):
        t = t[: t.rfind("```")]
    return t.strip()


def _parse_subflows(raw: str) -> list[DiagramSubflow]:
    """Parse the vision model's JSON reply into sub-flows, leniently."""
    text = _strip_code_fences(raw)
    # Tolerate stray prose around the JSON object.
    lo, hi = text.find("{"), text.rfind("}")
    if lo != -1 and hi > lo:
        text = text[lo : hi + 1]
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return []

    subflows: list[DiagramSubflow] = []
    for item in data.get("subflows", []) if isinstance(data, dict) else []:
        if not isinstance(item, dict):
            continue
        mermaid = _strip_code_fences(str(item.get("mermaid", ""))).strip()
        if not mermaid:
            continue
        subflows.append(
            DiagramSubflow(
                name=str(item.get("name") or "Sub-flow").strip(),
                prefix=str(item.get("prefix") or "").strip(),
                mermaid=mermaid,
            )
        )
    return subflows


def _diagram_images(page: PatternPage) -> list[dict]:
    return [{"data": page.diagram.data_b64, "media_type": page.diagram.media_type}]


def diagram_to_full_mermaid(
    provider: LLMProvider,
    page: PatternPage,
    max_text_chars: int = 12000,
) -> str | None:
    """Transcribe the WHOLE diagram image into one complete Mermaid diagram.

    Reproduces every element of the image as a single diagram (not split).
    Returns None if the page has no diagram. Distinct from ``extract_mermaid``
    (which builds the pattern's abstract classDiagram from the page text): this
    is a literal, exhaustive transcription of the image itself.
    """
    if page.diagram is None:
        return None
    user = _FULL_USER.format(name=page.slug, text=page.text[:max_text_chars])
    raw = provider.vision_response(_FULL_SYSTEM, user, _diagram_images(page))
    return _strip_code_fences(raw)


def split_subflows(
    provider: LLMProvider,
    page: PatternPage,
    full_mermaid: str | None,
    max_text_chars: int = 12000,
) -> list[DiagramSubflow]:
    """Split the full-diagram Mermaid into one `sequenceDiagram` per sub-flow.

    This is a TEXT task: it works from ``full_mermaid`` (the flowchart produced
    by ``diagram_to_full_mermaid``), grouping its edges by shared step-prefix
    family (same prefix → same sub-flow) and naming each from the description —
    no second read of the image. Returns [] if there is no full diagram.
    """
    if not full_mermaid or not full_mermaid.strip():
        return []
    user = _SUBFLOW_USER.format(text=page.text[:max_text_chars], mermaid=full_mermaid)
    raw = provider.response(_SUBFLOW_SYSTEM, user)
    return _parse_subflows(raw)


def _parse_main_index(raw: str, count: int) -> int:
    """Parse the selector's ``{"index": n}`` reply, clamped to [0, count)."""
    text = _strip_code_fences(raw)
    lo, hi = text.find("{"), text.rfind("}")
    if lo != -1 and hi > lo:
        text = text[lo : hi + 1]
    try:
        idx = int(json.loads(text).get("index"))
    except (ValueError, TypeError, AttributeError):
        return 0
    return idx if 0 <= idx < count else 0


def select_main_subflow(
    provider: LLMProvider,
    page: PatternPage,
    subflows: list[DiagramSubflow],
    max_text_chars: int = 12000,
) -> DiagramSubflow | None:
    """Pick the single MAIN sub-flow described by the pattern text.

    With 0 or 1 sub-flows there is nothing to choose. With several, a separate
    text-only LLM call decides which one is the pattern's primary runtime flow
    (as opposed to auxiliary flows like registration or password reset); the
    others are dropped so the reference page focuses on the main flow.
    """
    if len(subflows) <= 1:
        return subflows[0] if subflows else None
    catalogue = "\n".join(
        f"{i}. {sf.name} — edge prefix `{sf.prefix or '(none)'}`" for i, sf in enumerate(subflows)
    )
    user = _MAIN_USER.format(text=page.text[:max_text_chars], catalogue=catalogue)
    raw = provider.response(_MAIN_SYSTEM, user)
    return subflows[_parse_main_index(raw, len(subflows))]


_PATTERN_SYSTEM = """You are a security-pattern analyst. From a DistriNet security pattern's \
cleaned text (`Roles`, `Actions`, `Data elements`, `Behaviour`) and the Mermaid diagram of its \
MAIN flow, produce the pattern's canonical JSON. Copy names and wording from the text; never \
invent anything.

An entry in `behaviour` is an INTERACTION, not an arrow. The request edge `A → B` and the \
response edge `B → A` answering it share ONE entry — the request's, in the requester's \
direction — with the response's sentence appended to its `description`, so the entry states \
what was asked and what came back. The response gets no entry and no field of its own. Pair \
them like a call stack: a response answers the most recent unanswered request between those \
two roles, so it may be several steps later. Example — arrows `1 Subject→Enforcer: \
action(token)`, `2 Enforcer→Verifier: verify(token)`, `3 Verifier→Store: get_principal(token)`, \
`4 Store→Verifier: principal/error`, `5 Verifier→Enforcer: principal/error`, `6 \
Enforcer→System: action(principal)` → four entries (1, 2, 3, 6), with 4 folded into 3 and 5 \
into 2. This includes the flow's closing `reply`/`error` back to the initiator. Only a \
returning edge whose target never made the request keeps its own entry (`kind: "data"`).

Output ONLY a JSON object:
- "name": the pattern's name; "summary": its `Summary` paragraph (else "").
- "roles": one entry `{"name", "type", "required", "description"}` per role — `type` is the \
metamodel type \
in parentheses ("Entity", "Storage", …). Include ONLY roles used as some interaction's `from` \
or `to`: drop the `Roles` section's auxiliary-flow roles (registration, reset, provisioning) \
the main flow never touches, and conversely add a main-flow participant the section omits \
(often the protected `System`), describing it from the text. "required" is false for the \
`Subject` and the `System` — the pattern's environment rather than its machinery — and true \
for every other role.
- "behaviour": the main flow's interactions, each `{"step", "from", "to", "kind", "element", \
"description"}`. `step` numbers the entries sequentially from "1" — not the diagram's edge \
numbers, whose sequence the merging breaks. `from`/`to` are the requester and responder, read \
off the request arrow's direction and named exactly as in `roles`. `kind` is "action" \
(a call) or "data" (an unfoldable returning edge). `element` is the request arrow's CONCRETE \
label as instantiated in the diagram — `hash(salt, pwd)`, not the `Actions`/`Data elements` \
section's placeholder `hash(x, y)`; "" if nothing matches. Every numbered `Behaviour` step is \
accounted for: a request as an entry, a response inside the description of the entry it \
answers."""

_PATTERN_USER = """--- PATTERN TEXT ---
{text}
--- END PATTERN TEXT ---

--- MAIN FLOW (Mermaid) ---
{mermaid}
--- END MAIN FLOW ---

Return the pattern JSON now — name, summary, roles[], behaviour[] — merging each request with \
the response that answers it into a single entry, numbered sequentially."""


def _loads_object(raw: str) -> dict:
    """Parse a JSON object from the model reply, leniently."""
    text = _strip_code_fences(raw)
    lo, hi = text.find("{"), text.rfind("}")
    if lo != -1 and hi > lo:
        text = text[lo : hi + 1]
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def _role_key(name: object) -> str:
    """Normalise a role name for participation matching (case/space/punctuation-insensitive)."""
    return re.sub(r"[^a-z0-9]+", "", str(name or "").lower())


def _normalise_behaviour_fields(data: dict) -> None:
    """Put every `behaviour` entry on the `{step, from, to, kind, element, description}` schema.

    Asked to merge a request with its response, a model readily invents fields
    to hold the response half (`action`/`response`/`response_step`). The schema
    deliberately has no such fields: an interaction is named by its call, and
    what came back is stated in the entry's prose. Those keys are therefore
    normalised away — `action` becomes `element`, and a `response` label is
    dropped once its sentence is in `description`.
    """
    behaviour = data.get("behaviour")
    if not isinstance(behaviour, list):
        return
    for entry in behaviour:
        if not isinstance(entry, dict):
            continue
        if not entry.get("element") and entry.get("action"):
            entry["element"] = entry["action"]
        if not entry.get("kind"):
            entry["kind"] = "action"
        for stray in ("action", "response", "response_step"):
            entry.pop(stray, None)


def _fold_stray_responses(data: dict) -> None:
    """Fold a response left as its own entry back into the request it answers.

    The prompt asks for merged interactions, and the model does it reliably for
    a flow's inner calls — but the terminal branches back to the initiator
    (`error`, `reply`) are often still emitted as standalone entries pointing
    the opposite way. This applies the prompt's own call-stack rule to the
    returned list: an entry is folded into the nearest earlier entry that it
    mirrors (its `from`/`to` reversed) and that is still awaiting a response.

    Only the response's prose survives the fold — its sentence is appended to
    the request's `description` — since the entry already names the exchange by
    its call. A `kind` of `"data"`, or a bare data label where a call would
    carry an argument list (`error` vs `verify(token)`), is the signal that an
    entry is a response rather than a genuine later call in the other
    direction; entries whose `element` has an argument list are left alone, so
    a role legitimately calling back is never swallowed.
    """
    behaviour = data.get("behaviour")
    if not isinstance(behaviour, list):
        return
    merged: list[dict] = []
    answered: set[int] = set()
    for entry in behaviour:
        if not isinstance(entry, dict):
            merged.append(entry)
            continue
        element = str(entry.get("element") or "").strip()
        is_response_shaped = str(entry.get("kind") or "").strip().lower() == "data" or (
            element and "(" not in element
        )
        target = None
        if is_response_shaped:
            for open_req in reversed(merged):
                if not isinstance(open_req, dict) or id(open_req) in answered:
                    continue
                if _role_key(open_req.get("from")) == _role_key(entry.get("to")) and _role_key(
                    open_req.get("to")
                ) == _role_key(entry.get("from")):
                    target = open_req
                    break
        if target is None:
            merged.append(entry)
            continue
        answered.add(id(target))
        desc = str(entry.get("description") or "").strip()
        if desc:
            target["description"] = f"{str(target.get('description') or '').strip()} {desc}".strip()
        log.debug(
            "folded step %s (%s → %s) into step %s as its response",
            entry.get("step"),
            entry.get("from"),
            entry.get("to"),
            target.get("step"),
        )
    data["behaviour"] = merged


_OPTIONAL_ROLES = {"subject", "system"}


def _mark_required_roles(data: dict) -> None:
    """Tag each role `required`, keeping the field's position after `type`.

    `Subject` and `System` are the pattern's environment — the caller that
    initiates the flow and the business logic being protected. Every codebase
    has them in some form, and they are not what the pattern contributes, so
    they are marked optional; every other role is machinery the pattern
    prescribes and is required. The split is currently by name rather than
    per-pattern judgement.
    """
    roles = data.get("roles")
    if not isinstance(roles, list):
        return
    for i, role in enumerate(roles):
        if not isinstance(role, dict):
            continue
        required = _role_key(role.get("name")) not in _OPTIONAL_ROLES
        ordered = {k: v for k, v in role.items() if k != "required"}
        ordered["required"] = required
        # Re-insert so `required` sits with the other metadata, before the prose.
        roles[i] = {
            **{k: ordered[k] for k in ("name", "type") if k in ordered},
            "required": required,
            **{k: v for k, v in ordered.items() if k not in ("name", "type", "required")},
        }


def _renumber_steps(data: dict) -> None:
    """Renumber the merged `behaviour` list `1..N`.

    The model numbers each entry after the diagram edge it came from, and
    merging removes the response edges — leaving gaps (1, 2, 3, 6) and, where
    the diagram labels a branch with its predecessor's number, duplicates. The
    diagram is not part of the output, so those numbers no longer identify
    anything; what the recovery agent needs is the walkthrough's order, which a
    contiguous sequence states plainly.
    """
    behaviour = data.get("behaviour")
    if not isinstance(behaviour, list):
        return
    for i, entry in enumerate(behaviour, start=1):
        if isinstance(entry, dict):
            entry["step"] = str(i)


def _keep_mainflow_roles(data: dict) -> None:
    """Drop roles that no `behaviour` step uses as its `from` or `to`.

    The page's `Roles` section documents every role of the pattern, including
    ones belonging to auxiliary flows (registration, reset, …) that the chosen
    main flow never touches. The pattern's roles are what `secpat recover` is
    asked to ground in a codebase, so a role with no step to justify it is an
    unrecoverable role by construction. The prompt asks for this already; this
    enforces it on the returned JSON. If nothing would survive (no behaviour
    steps, or no name overlap at all), the roles are left untouched.
    """
    roles = data.get("roles")
    behaviour = data.get("behaviour")
    if not isinstance(roles, list) or not isinstance(behaviour, list):
        return
    used = {
        _role_key(step.get(end))
        for step in behaviour
        if isinstance(step, dict)
        for end in ("from", "to")
    }
    used.discard("")
    if not used:
        return
    kept = [r for r in roles if isinstance(r, dict) and _role_key(r.get("name")) in used]
    if not kept:
        return
    dropped = [str(r.get("name")) for r in roles if r not in kept]
    if dropped:
        log.debug("dropped %d non-main-flow role(s): %s", len(dropped), ", ".join(dropped))
    data["roles"] = kept


def synthesize_pattern(
    provider: LLMProvider,
    page: PatternPage,
    mermaid: str | None,
    max_text_chars: int = 24000,
) -> dict:
    """Derive the final structured pattern (JSON) from the cleaned text + main flow.

    A text-only agent reads the pattern's `Roles`, `Actions`, `Data elements`,
    and `Behaviour` sections and the chosen main-flow Mermaid, and returns
    `{name, summary, roles[], behaviour[]}`. `behaviour[]` is the main flow's
    walkthrough as INTERACTIONS, each `{step, from, to, kind, element,
    description}`: a request edge `A → B` and the response edge `B → A` that
    answers it share the request's entry, the response contributing its
    sentence to `description` rather than a mirror-image entry of its own.
    `element` is the CONCRETE diagram edge label instantiated for this pattern
    (e.g. `hash(salt, pwd)`), not the generic section placeholder
    (`hash(x, y)`).
    `roles[]` is restricted to the roles the main flow actually uses — those
    named as some step's `from`/`to` — so roles belonging only to auxiliary
    flows (registration, reset, …) are dropped by `_keep_mainflow_roles`. The
    main-flow Mermaid is read only to recover edge direction and concrete
    labels; it is NOT emitted in the output. This is the final, canonical output
    derived from the raw text pattern.
    """
    text = clean_page_text(page)[:max_text_chars]
    user = _PATTERN_USER.format(text=text, mermaid=mermaid or "(none)")
    data = _loads_object(provider.response(_PATTERN_SYSTEM, user))
    data.setdefault("name", _page_title_from_text(text))
    data.setdefault("summary", "")
    data.setdefault("roles", [])
    data.setdefault("behaviour", [])
    _normalise_behaviour_fields(data)
    _fold_stray_responses(data)
    _renumber_steps(data)
    _keep_mainflow_roles(data)
    _mark_required_roles(data)
    return data


def _page_title_from_text(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""
