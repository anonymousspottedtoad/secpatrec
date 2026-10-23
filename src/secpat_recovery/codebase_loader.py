from __future__ import annotations

import sys
from pathlib import Path

from secpat_scoring.workspace import BLOCKED_DIRS as _BLOCKED_DIRS  # noqa: F401  (re-exported)
from secpat_scoring.workspace import is_blocked as _is_blocked

_ALLOWED_EXTENSIONS: frozenset[str] = frozenset(
    ".py .js .ts .jsx .tsx .java .kt .go .rs .c .cpp .h .hpp .cs .rb .php"
    " .swift .scala .sh .yaml .yml .json .toml .xml .tf".split()
)

_SECURITY_KEYWORDS = ("auth", "token", "security", "middleware", "permission", "secret", "cred")

# `_is_blocked` is the shared package's `secpat_scoring.workspace.is_blocked`, not a
# local copy. The KB builder and `RepoAccess` import it from here, so semantic
# search and grep see one file set — and because the interaction-order walk and
# the batch scorer index with the same default, that file set is also the one the
# order is derived and scored over. Hidden directories are excluded wholesale:
# they hold tooling state, not the codebase under analysis — including `.kb/`,
# this engine's own ChromaDB index, which would otherwise be indexed and grepped.


def _security_score(path: Path) -> int:
    name = path.as_posix().lower()
    return sum(1 for kw in _SECURITY_KEYWORDS if kw in name)


def load_codebase(
    root: Path,
    *,
    max_files: int = 150,
    max_file_bytes: int = 65536,
    extra_extensions: set[str] | None = None,
) -> list[tuple[Path, str]]:
    allowed = _ALLOWED_EXTENSIONS | (extra_extensions or set())
    candidates: list[Path] = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if _is_blocked(path, root):
            continue
        if path.suffix.lower() not in allowed:
            continue
        candidates.append(path)

    # Priority sort: security keywords desc, then path depth asc
    candidates.sort(key=lambda p: (-_security_score(p), len(p.parts)))

    if len(candidates) > max_files:
        candidates = candidates[:max_files]

    truncated = 0
    results: list[tuple[Path, str]] = []
    for path in candidates:
        raw = path.read_bytes()
        if len(raw) > max_file_bytes:
            raw = raw[:max_file_bytes]
            text = raw.decode("utf-8", errors="replace") + "\n# [TRUNCATED at 64KB]"
            truncated += 1
        else:
            text = raw.decode("utf-8", errors="replace")

        rel = path.relative_to(root)
        formatted = f"### FILE: {rel}\n{text}\n### END FILE"
        results.append((rel, formatted))

    print(
        f"Loaded {len(results)} files ({truncated} truncated) from {root}",
        file=sys.stderr,
    )
    return results
