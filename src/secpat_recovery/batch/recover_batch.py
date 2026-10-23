#!/usr/bin/env python3
"""`secpat recover --manifest-path`: the recovery pipeline over every record of a manifest.

A manifest lists repo × pattern records (``runs``, each with ``codebase_path`` and
``pattern_file`` relative to the manifest's own folder — see
``secpat_recovery.manifest``). Each record runs ``secpat recover`` in a subprocess
of its own, into one batch folder.

**How many run at once is per provider.** ``SECPAT_PARALLEL`` in ``.env`` sets it
for every provider in one place — ``SECPAT_PARALLEL=openai=4,claude=2,muse=8,qwen=4``
— and a provider it does not name runs one record at a time. ``--parallel N``
overrides it for one batch. Hosted providers bill against a per-account token
quota, so past some width more runs in flight only means more 429s; the client
waits each one out and retries (``llm/rate_limit.py``), so a too-wide setting is
slower, not lost. Self-hosted servers batch concurrent requests themselves, so
there width is close to an N-fold throughput win.

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
    ... --parallel 4
    ... --only <substring> --only <substring>
    ... --disable-tool grep --system-prompt-file src/secpat_recovery/ablation/prompts/no-grep.md
    ... --dry-run
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
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
# parallel runs
# --------------------------------------------------------------------------


def parallel_for(provider: str) -> int:
    """How many records run at once on ``provider``, from ``SECPAT_PARALLEL``.

    ``SECPAT_PARALLEL=openai=4,claude=2,muse=8`` — one entry per provider; one
    the setting does not name runs a record at a time.
    """
    for entry in os.getenv("SECPAT_PARALLEL", "").split(","):
        name, _, width = entry.partition("=")
        if name.strip() == provider and width.strip():
            return int(width)
    return 1

_print_lock = threading.Lock()


def say(*parts: str) -> None:
    """Print without interleaving two workers' lines mid-sentence."""
    with _print_lock:
        print(*parts, flush=True)


class KBLocks:
    """One lock per KB directory, so no two runs share a knowledge base.

    Records are keyed by repo and each repo gets its own ``.kb-batch/<repo>/``,
    so distinct repos never contend and a fully-distinct manifest runs at full
    width. Two records on the *same* repo are a different matter: ``build_kb``
    is not atomic and a ChromaDB collection is a sqlite file, so a concurrent
    first-run pair would race to create the same index and leave it in the
    half-built state ``drop_half_built_kb`` exists to clean up after. Holding
    the lock for the whole run — not just the build — keeps that impossible
    without the parent having to see inside the subprocess.
    """

    def __init__(self) -> None:
        self._locks: dict[Path, threading.Lock] = defaultdict(threading.Lock)
        self._guard = threading.Lock()

    def get(self, kb_dir: Path) -> threading.Lock:
        with self._guard:
            return self._locks[kb_dir]


class Stagger:
    """Space consecutive run *launches* by at least ``--sleep`` seconds.

    Parallel runs only. It stops N workers hitting a cold server — or starting N
    KB builds — at the same instant. Sequentially ``--sleep`` keeps its original
    meaning of a pause *after* each run (real rate-limit headroom, which a
    minimum launch spacing would not give: a gap shorter than the run itself
    elapses while the run is still going and never pauses anything).
    """

    def __init__(self, seconds: float) -> None:
        self._seconds = seconds
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        if not self._seconds:
            return
        with self._lock:
            delay = self._last + self._seconds - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self._last = time.monotonic()


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
        say(f"    (discarded half-built KB at {kb_dir.name}, rebuilding)")

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
        say(f"    note: no interaction order — {order.get('reason', '')}")
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
                "parallel": args.parallel,
                "records": [slug(r) for r in records],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def run_record(
    n: int,
    total: int,
    rec: dict,
    batch_dir: Path,
    kb_root: Path,
    args,
    kb_locks: KBLocks,
    stagger: Stagger,
) -> dict:
    """Run one record, then report it. Safe to call from several threads."""
    name = slug(rec)
    run_dir = batch_dir / name
    kb_dir = kb_root / rec["repo"]

    stagger.wait()
    lock = kb_locks.get(kb_dir)
    if not lock.acquire(blocking=False):
        # Another record on this repo holds the KB; say so rather than
        # announcing a start that has not happened.
        say(f"    [{n}/{total}] {name}: waiting for {kb_dir.name}'s KB")
        lock.acquire()
    try:
        say(f"\n=== [{n}/{total}] {name} — started ===")
        result = run_one(rec, run_dir, kb_dir, args)
    finally:
        lock.release()

    if result["error"]:
        say(f"    [{n}/{total}] {name}: FAILED ({result['seconds']}s): {result['error']}")
    else:
        order = (
            f"{result['interactions']} interactions"
            if result["interactions"] is not None
            else "no order"
        )
        say(f"    [{n}/{total}] {name}: {result['badge']} — {result['roles']} roles, {order} "
            f"({result['seconds']}s) -> {run_dir}")
    return result


# --------------------------------------------------------------------------


def run(args) -> int:
    """Run a batch. ``args`` carries ``secpat recover``'s batch options by name:
    ``manifest_path``, ``batch_path``, ``provider``, ``model``, ``kb_root``,
    ``rebuild_kb``, ``no_order``, ``disable_tool``, ``system_prompt_file``,
    ``only``, ``skip``, ``parallel`` (None: from ``SECPAT_PARALLEL``), ``sleep``,
    ``dry_run``."""
    if args.parallel is None:
        args.parallel = parallel_for(args.provider)
    if args.parallel < 1:
        print("error: --parallel must be at least 1", file=sys.stderr)
        return 2

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
        print(f"Parallel     : {args.parallel} run(s) at once")
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
    kb_locks = KBLocks()
    total = len(records)
    concurrent = args.parallel > 1

    def task(numbered: tuple[int, dict], stagger: Stagger) -> dict:
        n, rec = numbered
        return run_record(n, total, rec, batch_dir, kb_root, args, kb_locks, stagger)

    if concurrent:
        width = min(args.parallel, total)
        stagger = Stagger(args.sleep)
        say(f"Running {total} record(s), {width} in flight on {args.provider}.")
        with ThreadPoolExecutor(max_workers=width) as pool:
            # `map` keeps manifest order in the results, whatever order they finish in.
            results = list(pool.map(lambda nr: task(nr, stagger), enumerate(records, 1)))
    else:
        results = []
        for numbered in enumerate(records, 1):
            results.append(task(numbered, Stagger(0)))
            if args.sleep and numbered[0] < total:
                time.sleep(args.sleep)

    failed = sum(1 for r in results if r["error"])
    print(f"\nDone: {len(results) - failed} ok, {failed} failed -> {batch_dir}")
    return 1 if failed else 0

