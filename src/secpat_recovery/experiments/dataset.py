"""The records the experiments run, and stage 3 for a batch of existing recoveries.

Each part of the dataset is named by its own flag, so the parts can live anywhere:

    --manifests         the records: each run's pattern, repo and language
    --repos-path        <language>/<repo>/
    --cpg-path          <repo>__<JAVASRC|PYTHONSRC>.cpg
    --patterns-path     <pattern>/04_pattern.json
    --groundtruth-path  <language>/<pattern>__<repo>.json
    --recoveries-path   <experiment>/<model>/<language>/<pattern>__<repo>/recovery.json
"""

from __future__ import annotations

import json
from pathlib import Path

FRONTEND = {"java": "JAVASRC", "python": "PYTHONSRC"}


def load_runs(manifests: list[Path], repos=(), patterns=()) -> list[dict]:
    """The manifests' records, kept by exact repo / pattern names."""
    runs = []
    for manifest in manifests:
        data = json.loads(Path(manifest).read_text(encoding="utf-8"))
        runs += data["runs"] if isinstance(data, dict) else data
    return [
        r for r in runs
        if (not repos or r["repo"] in repos) and (not patterns or r["pattern"] in patterns)
    ]


def cpg_file(run: dict, cpg_path: Path) -> Path:
    return cpg_path / f"{run['repo']}__{FRONTEND[run['language']]}.cpg"


def write_manifest(runs: list[dict], repos_path: Path, patterns_path: Path, path: Path) -> Path:
    """A manifest of these records with absolute paths, so it can live anywhere."""
    full = [
        {**r, "codebase_path": str(repos_path / r["language"] / r["repo"]),
         "pattern_file": str(patterns_path / r["pattern"] / "04_pattern.json")}
        for r in runs
    ]
    path.write_text(json.dumps({"runs": full}, indent=2), encoding="utf-8")
    return path


def derive_orders(records: list[dict], batch_path: Path) -> int:
    """Stage 3 for every record of the batch that has a recovery.json.

    Derives each recovery's interaction order from its role_mappings, as
    `secpat recover` does after the agent, and writes it into that recovery.json.
    `records` come from `load_records`, so their paths are absolute.
    """
    from ..interaction.order import detect_action_sequence
    from ..output.mapping_report import write_json

    failed = 0
    for n, rec in enumerate(records, 1):
        slug = f"{rec['pattern']}__{rec['repo']}"
        recovery = batch_path / slug / "recovery.json"
        if not recovery.is_file():
            print(f"[{n}/{len(records)}] {slug}: no recovery.json, skipped", flush=True)
            continue
        result = json.loads(recovery.read_text(encoding="utf-8"))
        pattern = json.loads(Path(rec["pattern_file"]).read_text(encoding="utf-8"))
        try:
            order = detect_action_sequence(
                Path(rec["codebase_path"]), result.get("role_mappings", []), pattern
            )
        except Exception as exc:  # KotlinCitation and the like: keep going
            failed += 1
            print(f"[{n}/{len(records)}] {slug}: {type(exc).__name__}: {exc}", flush=True)
            continue
        result["interaction_order"] = order
        write_json(result, recovery)
        found = (f"{len(order['interactions'])} interactions" if order.get("available")
                 else f"no order ({order.get('reason')})")
        print(f"[{n}/{len(records)}] {slug}: {found}", flush=True)
    return 1 if failed else 0
