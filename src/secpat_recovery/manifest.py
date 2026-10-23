"""Batch manifests: one record per repo × pattern pair.

A manifest is a JSON object whose ``runs`` list holds the records (a bare list is
accepted too). Each record names its ``codebase_path`` and ``pattern_file``
**relative to the manifest's own folder**, so a manifest and the files it lists
move together — copy the folder anywhere and the paths still resolve.

:func:`load_records` is the one reader. It returns each record with those two
paths made absolute, so nothing downstream needs to know where the manifest was.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path

#: The record fields that are paths, relative to the manifest's folder.
PATH_FIELDS = ("codebase_path", "pattern_file")


def load_records(
    manifest_path: Path, only: Iterable[str] = (), skip: Iterable[str] = ()
) -> list[dict]:
    """The manifest's records, paths absolute, filtered by ``only`` / ``skip``.

    ``only`` keeps a record whose id, repo or pattern contains any of the given
    substrings (case-insensitive); ``skip`` drops one that does.
    """
    manifest_path = Path(manifest_path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    runs = data["runs"] if isinstance(data, dict) else data
    base = manifest_path.absolute().parent
    records = []
    for run in runs:
        rec = dict(run)
        for field in PATH_FIELDS:
            if rec.get(field):
                rec[field] = os.path.normpath(base / rec[field])
        records.append(rec)

    only, skip = [n.lower() for n in only], [n.lower() for n in skip]

    def matches(rec: dict, needles: list[str]) -> bool:
        hay = f"{rec.get('id', '')} {rec['repo']} {rec['pattern']}".lower()
        return any(n in hay for n in needles)

    if only:
        records = [r for r in records if matches(r, only)]
    if skip:
        records = [r for r in records if not matches(r, skip)]
    return records
