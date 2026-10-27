#!/usr/bin/env python3
"""`secpat recover --manifest-path`: the recovery pipeline over every record of a manifest.

A manifest lists repo × pattern records (``runs``, each with ``codebase_path`` and
``pattern_file`` relative to the manifest's own folder — see
``secpat_recovery.manifest``). Each record runs ``secpat recover`` in a subprocess
of its own, into one batch folder, one record at a time in manifest order.

Each record gets its own subfolder in ``--batch-path``, holding that run's
``recovery.json``, ``transcript.md`` and the raw CLI log::

    <batch-path>/
    ├── <pattern>__<repo>/
    │   ├── recovery.json
    │   ├── transcript.md
    │   └── run.log
    └── ...

Each subfolder is complete the moment its run finishes, so an interrupted
batch still leaves usable results for everything that did finish.

``--disable-tool NAME`` (repeatable) withholds one of the agent's tools from
every record in the batch, and ``--system-prompt-file`` runs that arm on its own
copy of the prompt with the tool edited out (``ablation/prompts/<arm>.md``) —
together, one arm of a tool ablation. The canonical prompt is never rewritten to
run an experiment. The batch folder records what it was run with in
``batch_config.json``, so an arm can never be told from its control by folder
name alone; ``secpat ablation run`` drives the arms and pairs the two flags.

A record whose subfolder already holds a ``recovery.json`` is still rerun; to
resume a batch, pass a manifest of the records still pending.

Usage (paths relative to where you run it):
    secpat recover --manifest-path <manifest.json> --batch-path <batch folder> --provider openai
    ... --only <substring> --only <substring>
    ... --disable-tool grep --system-prompt-file src/secpat_recovery/ablation/prompts/no-grep.md
    ... --dry-run
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from ..manifest import load_records
from ..recovery_agent import ABLATABLE_TOOLS

ROOT = Path(__file__).resolve().parents[3]

# --------------------------------------------------------------------------
# records
# --------------------------------------------------------------------------


def slug(rec: dict) -> str:
    """Subfolder name for a record — unique per (pattern, repo)."""
    return f"{rec['pattern']}__{rec['repo']}"


def check_inputs(rec: dict) -> str | None:
    """Return an error message if this record's inputs are missing."""
    pattern_file = Path(rec["pattern_file"])
    codebase = Path(rec["codebase_path"])
    if not pattern_file.is_file():
        return f"pattern file not found: {rec['pattern_file']}"
    if not codebase.is_dir():
        return f"codebase not found: {rec['codebase_path']}"
    return None


# --------------------------------------------------------------------------
# running
# --------------------------------------------------------------------------


def drop_half_built_kb(kb_dir: Path) -> bool:
    """Delete ``kb_dir`` if it holds an interrupted, unusable KB build.

    ``build_kb`` is not atomic: it writes ``source.json`` and ``struct.json``
    first, then embeds and commits the Chroma collection. Kill it in between —
    or let it die — and the directory is left existing, recording the right
    source codebase, but with no collection in it. ``PatternRecoveryEngine``
    only rebuilds when the KB dir is *absent* or was built from a *different*
    codebase, so neither guard fires and every later run dies with
    ``NotFoundError: Collection [codebase] does not exist``, forever.

    A committed collection has its own uuid-named subdirectory alongside
    ``chroma.sqlite3``; a half-built one does not. Checking that structurally
    avoids opening the sqlite file here and holding a lock the recovery
    subprocess would then contend with.
    """
    if not kb_dir.exists():
        return False
    if any(p.is_dir() for p in kb_dir.iterdir()):
        return False
    shutil.rmtree(kb_dir, ignore_errors=True)
    return True


def run_one(rec: dict, run_dir: Path, kb_dir: Path, args) -> dict:
    """Run the recovery CLI for one record. Returns a result dict."""
    run_dir.mkdir(parents=True, exist_ok=True)
    if drop_half_built_kb(kb_dir):
        print(f"    (discarded half-built KB at {kb_dir.name}, rebuilding)", flush=True)

    cmd = [
        sys.executable, "-m", "secpat_recovery.cli", "recover",
        rec["codebase_path"],
        "--pattern-file", rec["pattern_file"],
        "--output-dir", str(run_dir),
        "--provider", args.provider,
        "--kb-dir", str(kb_dir),
    ]
    if args.no_order:
        cmd += ["--no-order"]
    if args.model:
        cmd += ["--model", args.model]
    if args.rebuild_kb:
        cmd += ["--rebuild-kb"]
    for tool in args.disable_tool:
        cmd += ["--disable-tool", tool]
    if args.system_prompt_file:
        cmd += ["--system-prompt-file", str(Path(args.system_prompt_file).resolve())]

    log_path = run_dir / "run.log"
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(" ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.time() - started

    result = {
        "seconds": round(elapsed, 1),
        "roles": 0,
        "interactions": None,
        "badge": None,
        "error": None,
    }

    if proc.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
        result["error"] = tail[-1] if tail else f"exit {proc.returncode}"
        return result

    recovery = run_dir / "recovery.json"
    if not recovery.is_file():
        result["error"] = "run exited 0 but wrote no recovery.json"
        return result

    recovered = json.loads(recovery.read_text(encoding="utf-8"))
    mappings = recovered.get("role_mappings", [])
    result["roles"] = len(mappings)
    # Stage 3 cannot fail the run (a missing Docker daemon must not discard a
    # completed agent run), so its outcome is reported per record instead: a
    # count when the order was derived, None when it was not.
    order = recovered.get("interaction_order") or {}
    result["interactions"] = len(order.get("interactions", [])) if order.get("available") else None
    if order and not order.get("available"):
        print(f"    note: no interaction order — {order.get('reason', '')}", flush=True)
    statuses = {m.get("status", "missing") for m in mappings}
    result["badge"] = (
        "UNKNOWN" if not statuses
        else "FULL" if statuses == {"present"}
        else "NONE" if statuses == {"missing"}
        else "MIXED"
    )
    return result


def tools_line(disabled: list[str]) -> str:
    """One line naming the arm — which tools this batch's agent had."""
    if not disabled:
        return "all (control)"
    kept = [t for t in ABLATABLE_TOOLS if t not in set(disabled)]
    return f"{', '.join(kept)} (disabled: {', '.join(disabled)})"


def write_batch_config(batch_dir: Path, args, records: list[dict]) -> None:
    """Record what this batch was run with, beside its results.

    An ablation arm and its control differ only in a flag, and the results they
    write are shaped identically — so without this the arm is recoverable only
    from the first line of some record's run.log, and only for as long as the
    logs are kept. It goes in before the first run, so an interrupted batch is
    still labelled.
    """
    (batch_dir / "batch_config.json").write_text(
        json.dumps(
            {
                "batch": batch_dir.name,
                "started_at": datetime.now().isoformat(timespec="seconds"),
                "manifest": str(args.manifest_path),
                "provider": args.provider,
                "model": args.model,
                "deriveOrder": not args.no_order,
                "disabled_tools": list(args.disable_tool),
                "system_prompt_file": str(args.system_prompt_file or "") or None,
                "ablation": bool(args.disable_tool),
                "records": [slug(r) for r in records],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def run_record(n: int, total: int, rec: dict, batch_dir: Path, kb_root: Path, args) -> dict:
    """Run one record, then report it."""
    name = slug(rec)
    run_dir = batch_dir / name
    kb_dir = kb_root / rec["repo"]

    print(f"\n=== [{n}/{total}] {name} — started ===", flush=True)
    result = run_one(rec, run_dir, kb_dir, args)

    if result["error"]:
        print(f"    [{n}/{total}] {name}: FAILED ({result['seconds']}s): {result['error']}",
              flush=True)
    else:
        order = (
            f"{result['interactions']} interactions"
            if result["interactions"] is not None
            else "no order"
        )
        print(f"    [{n}/{total}] {name}: {result['badge']} — {result['roles']} roles, {order} "
              f"({result['seconds']}s) -> {run_dir}", flush=True)
    return result


# --------------------------------------------------------------------------


def run(args) -> int:
    """Run a batch. ``args`` carries ``secpat recover``'s batch options by name:
    ``manifest_path``, ``batch_path``, ``provider``, ``model``, ``kb_root``,
    ``rebuild_kb``, ``no_order``, ``disable_tool``, ``system_prompt_file``,
    ``only``, ``skip``, ``sleep``, ``dry_run``."""
    if args.system_prompt_file and not Path(args.system_prompt_file).is_file():
        print(f"error: system prompt file not found: {args.system_prompt_file}",
              file=sys.stderr)
        return 2
    if args.disable_tool and not args.system_prompt_file:
        # Not fatal — a deliberate "schemas only" arm is a legitimate variant —
        # but it is far more often a forgotten flag, and it leaves the prompt
        # advertising the tool the run withheld.
        print(f"warning: withholding {', '.join(args.disable_tool)} but running on the "
              f"built-in prompt, which still lists it. Pair --disable-tool with "
              f"--system-prompt-file src/secpat_recovery/ablation/prompts/<arm>.md.",
              file=sys.stderr)

    manifest_path = Path(args.manifest_path)
    if not manifest_path.is_file():
        print(f"error: manifest not found: {manifest_path}", file=sys.stderr)
        return 2
    records = load_records(manifest_path, args.only, args.skip)
    if not records:
        print("error: no records matched", file=sys.stderr)
        return 2

    problems = [(slug(r), msg) for r in records if (msg := check_inputs(r))]
    if problems:
        for name, msg in problems:
            print(f"error: {name}: {msg}", file=sys.stderr)
        return 2

    # Absolute: each record's run is a subprocess started in the engine root, so
    # a path relative to where this command was run would land somewhere else.
    batch_dir = Path(args.batch_path).absolute()

    if args.dry_run:
        print(f"Batch folder : {batch_dir}")
        print(f"Provider     : {args.provider}" + (f" ({args.model})" if args.model else ""))
        print(f"Stage 3      : {'skipped' if args.no_order else 'interaction order'}")
        print(f"Agent tools  : {tools_line(args.disable_tool)}")
        print(f"Prompt       : {args.system_prompt_file or 'built-in (prompt.SYSTEM)'}")
        print(f"Records      : {len(records)}\n")
        for n, rec in enumerate(records, 1):
            print(f"  {n}. {slug(rec)}")
            print(f"       pattern  {rec['pattern_file']}")
            print(f"       codebase {rec['codebase_path']}")
        return 0

    batch_dir.mkdir(parents=True, exist_ok=True)
    write_batch_config(batch_dir, args, records)
    if args.disable_tool:
        print(f"Ablation: {tools_line(args.disable_tool)}")
        print(f"Prompt:   {args.system_prompt_file or 'built-in (prompt.SYSTEM)'}\n")

    kb_root = Path(args.kb_root).absolute() if args.kb_root else ROOT / ".kb-batch"
    total = len(records)
    results = []
    for n, rec in enumerate(records, 1):
        results.append(run_record(n, total, rec, batch_dir, kb_root, args))
        if args.sleep and n < total:
            time.sleep(args.sleep)

    failed = sum(1 for r in results if r["error"])
    print(f"\nDone: {len(results) - failed} ok, {failed} failed -> {batch_dir}")
    return 1 if failed else 0

