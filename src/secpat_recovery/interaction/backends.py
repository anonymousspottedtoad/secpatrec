"""Which language to analyse a codebase as, and what name the service knows it by.

Building the order is not done here: :func:`secpat_interaction_order.detect` sends
the role mapping to the Joern query service and gets the ordered interactions
back. The batch scorer calls the same function, so the order this pipeline emits
and the order it is scored against are produced by one implementation.

What is left here is the part that is this engine's own: **which language to
analyse the codebase as**, read off the recovered citations rather than off a
file count (see :func:`detect_language`), and :func:`repo_key`, the name the
service resolves its own checkout by.

This module used to own ``make_backend`` and a ``BACKENDS`` choice — a Joern
export built or fetched here, wrapped in the indirect-invocation backend, walked
in Python. All of that moved into the query.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

from ..codebase_loader import _is_blocked

logger = logging.getLogger(__name__)

#: Source extension -> the language name the Joern frontend map is keyed by.
#: Only extensions a Joern frontend actually handles: a repo's ``.json`` and
#: ``.yaml`` say nothing about what language its logic is written in, and letting
#: them vote would name a language no frontend can read.
_LANGUAGE_BY_EXTENSION = {
    ".py": "python",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".php": "php",
    ".rb": "ruby",
}

__all__ = ["detect_language", "repo_key"]


def detect_language(codebase: Path, cited_files: list[str] | None = None) -> str:
    """The language to analyse this codebase as, or "" if none can be named.

    **The recovered citations decide it, when there are any.** A repo is often
    written in more than one language — a Python service with a TypeScript
    front-end is the common shape here, and by file count the front-end wins
    easily — but the walk only ever has to reach the files the roles were
    grounded in. Analysing the repo as its most numerous language instead would
    build a call graph that does not contain a single anchor, and report an empty
    order for a mapping that was perfectly good.

    Failing that (a mapping with no usable paths), it falls back to a file count
    over the same file set the KB indexes — ``_is_blocked`` skips vendored and
    hidden directories, so a bundled ``node_modules`` cannot outvote the repo's
    own source. Either way the answer names the Joern frontend, which is what
    stops one stray ``.js`` file from getting a Python repo analysed as
    JavaScript.
    """
    cited: Counter[str] = Counter()
    for file in cited_files or []:
        language = _LANGUAGE_BY_EXTENSION.get(Path((file or "").strip()).suffix.lower())
        if language:
            cited[language] += 1
    if cited:
        return cited.most_common(1)[0][0]

    counts: Counter[str] = Counter()
    for path in codebase.rglob("*"):
        if not path.is_file() or _is_blocked(path, codebase):
            continue
        language = _LANGUAGE_BY_EXTENSION.get(path.suffix.lower())
        if language:
            counts[language] += 1
    return counts.most_common(1)[0][0] if counts else ""


def repo_key(codebase: Path) -> str:
    """The cache key a codebase's Joern export is stored under.

    The directory's own name, which is also what ``manifest_<user>.json`` calls
    the repo — so a codebase already exported by an evaluation run is reused here
    rather than rebuilt, and vice versa. Two different codebases checked out under
    the same directory name would collide; the export is keyed by name because
    that is the identity every other stage of this project uses for a repo.
    """
    return codebase.resolve().name
