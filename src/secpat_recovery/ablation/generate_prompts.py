#!/usr/bin/env python3
"""Write each ablation arm's system prompt: the canonical one, tool edited out.

`src/secpat_recovery/recovery_agent/prompt.py` is not touched by the ablation.
It holds `SYSTEM`, the prompt every normal run has always used, and rewriting it
to run an experiment would put the control arm on a prompt no earlier batch ran
on — the one thing an ablation's control may not do. So each arm gets its **own
copy** on disk instead, and `secpat recover --system-prompt-file` runs on it.

This script produces those copies. It is the only place the edits are written
down, and it is deliberately dumb: take `SYSTEM`, delete the withheld tool's
bullet from the tool list, restate the sentences elsewhere that prescribe the
tool's capability, and write the result. Everything else is copied through
untouched, so `diff`-ing an arm's prompt against the canonical one shows exactly
what the arm changed and nothing else.

Dropping the bullet alone is not enough. The method section also prescribes
capabilities in prose ("the only role you may find by searching"), and an
instruction to work a way the run cannot work is worse than no instruction: the
model either spends steps looking for the tool or reads the sentence as
permission it does not have. `REWRITES` covers those sentences.

    secpat ablation prompts           # write the prompts
    secpat ablation prompts --check   # verify, write nothing

`--check` is what `tests/test_tool_ablation.py` runs: if `SYSTEM` is edited and
the arms' copies are not regenerated, the test fails rather than letting an
experiment run on a stale prompt.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..recovery_agent import SYSTEM

HERE = Path(__file__).resolve().parent
#: Where the arms' prompt copies are written, one ``<arm>.md`` each.
PROMPTS = HERE / "prompts"
#: The engine checkout, for printing paths relative to it.
ROOT = HERE.parents[2]

#: The tool-list bullet each tool owns in `SYSTEM`, verbatim. A mismatch means
#: the prompt's tool list was reworded — regenerate rather than guess.
BULLETS: dict[str, str] = {
    "list_dir": "- `list_dir` — recursive tree of a directory.\n",
    "search": (
        "- `search` — semantic search of the indexed code, returning the source of every\n"
        "  strong match.\n"
    ),
    "grep": "- `grep` — exact text search over the whole repo.\n",
    "read_file": "- `read_file` — a real file by path, with line numbers.\n",
}

#: Sentences outside the tool list that prescribe a tool's capability, and what
#: the arm without that tool is told instead.
#:
#: Only `search` needs any. The method section names the entry point as the one
#: role the agent may find by searching rather than by tracing — the single
#: instruction that stops being followable once the semantic index is gone. The
#: replacement keeps the permission (the entry point is still found by looking it
#: up) and drops only the naming of how.
REWRITES: dict[str, tuple[tuple[str, str], ...]] = {
    "search": (
        (
            "This is the\n  only role you may find by searching.",
            "This is the\n  only role you may find by looking it up directly.",
        ),
        (
            "A name-search hit you never saw",
            "A name-match hit you never saw",
        ),
    ),
}

#: arm name -> the tools that arm withholds. Matches `run.sh`'s arms.
ARMS: dict[str, tuple[str, ...]] = {
    "no-grep": ("grep",),
    "no-search": ("search",),
    "no-read": ("read_file",),
    "no-list": ("list_dir",),
}


def build(disabled: tuple[str, ...]) -> str:
    """`SYSTEM` with `disabled`'s bullets deleted and its prose restated."""
    text = SYSTEM
    for tool in disabled:
        bullet = BULLETS[tool]
        if bullet not in text:
            raise SystemExit(
                f"error: the '{tool}' bullet is not in SYSTEM verbatim — the prompt's "
                f"tool list was reworded. Update BULLETS in {Path(__file__).name}."
            )
        text = text.replace(bullet, "")
        for before, after in REWRITES.get(tool, ()):
            if before not in text:
                raise SystemExit(
                    f"error: a '{tool}' rewrite no longer matches SYSTEM. Update "
                    f"REWRITES in {Path(__file__).name}."
                )
            text = text.replace(before, after)
        # The prompt writes a tool's name in backticks and nothing else does. A
        # leftover means a sentence was added to SYSTEM that reintroduces the
        # tool this arm exists to withhold.
        if f"`{tool}`" in text:
            raise SystemExit(
                f"error: the arm's prompt still names `{tool}`. Add the sentence to "
                f"REWRITES[{tool!r}] in {Path(__file__).name}."
            )
    return text


def path_for(arm: str) -> Path:
    return PROMPTS / f"{arm}.md"


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    ap = argparse.ArgumentParser(prog=prog, description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="Exit non-zero if a written prompt differs from what this "
                         "script would generate; write nothing")
    ap.add_argument("--arm", action="append", default=[], choices=sorted(ARMS),
                    help="Only this arm (repeatable; default: all)")
    args = ap.parse_args(argv)

    arms = args.arm or sorted(ARMS)
    stale: list[str] = []
    for arm in arms:
        want = build(ARMS[arm])
        dest = path_for(arm)
        if args.check:
            have = dest.read_text(encoding="utf-8") if dest.is_file() else None
            if have != want:
                stale.append(arm)
                print(f"STALE  {dest.relative_to(ROOT)}"
                      + ("" if have is not None else "  (missing)"))
            else:
                print(f"ok     {dest.relative_to(ROOT)}")
            continue
        dest.write_text(want, encoding="utf-8")
        removed = ", ".join(ARMS[arm])
        print(f"wrote  {dest.relative_to(ROOT)}  (without: {removed})")

    if stale:
        print(
            f"\n{len(stale)} arm prompt(s) out of date with SYSTEM. Regenerate:\n"
            f"    secpat ablation prompts",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
