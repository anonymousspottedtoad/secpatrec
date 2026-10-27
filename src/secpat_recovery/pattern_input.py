from __future__ import annotations

import json
from pathlib import Path

# Heads the pattern text, and travels with it: read next to the JSON it
# describes, "`behaviour[]` — the pattern flow in order" is self-evidently about
# the array directly beneath it. In the system prompt the same sentences are a
# forward reference to a payload the model has to hold separately.
_INTRO = """\
## The pattern

Below is the pattern, as raw JSON with four fields:

- `name`, `summary` — what the pattern is and what it protects against.
- `roles[]` — elements involved in the pattern's logic. Each role has a `name` and a `description`.
- `behaviour[]` — the pattern flow in order. Each entry is one interaction: `from`
  triggers `to`, and the `description` explains what happens.
"""


def load_pattern(path: Path) -> str:
    """Load a pattern file as the pattern content of the agent's first message.

    The pattern file itself is passed through verbatim — a pattern's
    ``04_pattern.json`` reaches the agent as raw JSON, and a raw ``.mmd``/``.md``
    classDiagram likewise unchanged. Nothing is reformatted; the only addition is
    ``_INTRO`` above it, naming the JSON's fields where that JSON appears.
    """
    return f"{_INTRO.rstrip()}\n\n{path.read_text(encoding='utf-8').strip()}\n"


def parse_pattern(path: Path) -> dict | None:
    """The pattern as a document, for the stages that read its *fields*.

    The agent gets the pattern as text and never needs it parsed — that is the
    whole point of ``load_pattern``. The interaction-order stage does: it orders
    its walk by ``roles[]`` and picks the entry role from ``behaviour[]``. Both
    read the same file, so parsing it lives beside the loader rather than in the
    stage that consumes it.

    Returns ``None`` for a pattern that has no fields to read — a raw Mermaid
    ``.mmd``/``.md`` classDiagram, or JSON that will not parse. A stage that gets
    ``None`` degrades (the order comes from the call graph alone); none of them
    fail, so a malformed pattern is not worth raising over here when
    ``load_pattern`` has already handed the same bytes to the agent.
    """
    if path.suffix.lower() not in (".json",):
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return document if isinstance(document, dict) else None
