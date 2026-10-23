from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

# The `regex` module is a drop-in superset of `re` whose match calls accept a
# `timeout=`. The agent authors its own patterns, so an unbounded backtracking
# engine would let one bad pattern (e.g. "(a+)+b") hang the whole run.
import regex

from .codebase_loader import _ALLOWED_EXTENSIONS, _BLOCKED_DIRS, _is_blocked

# Caps keep a single tool observation small enough to stay in the agent's context.
_MAX_MATCHES = 60
# Per-file cap: with a global cap alone, one crowded file hides the whole repo.
_MAX_MATCHES_PER_FILE = 5
_MAX_FILE_SUMMARY = 25
_MAX_LINE_CHARS = 300
_MAX_READ_LINES = 400
_MAX_ENTRIES = 400
# Hard stop on recursion, independent of the display budget.
_MAX_TREE_DEPTH = 12

# Wall-clock budget for one grep call, across all files.
_GREP_TIMEOUT_S = 5.0


class RepoAccessError(Exception):
    pass


def _clip(line: str) -> str:
    line = line.rstrip("\n")
    return line if len(line) <= _MAX_LINE_CHARS else line[:_MAX_LINE_CHARS] + " …[clipped]"


@dataclass
class RepoAccess:
    """Direct, un-indexed access to the codebase on disk.

    The KB answers "what is semantically similar"; this answers "where does this
    exact string appear" and "show me the actual file". Both are scoped to the
    codebase root — no path may escape it.
    """

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root).resolve()

    # ── path resolution ──────────────────────────────────────────────────────

    def _resolve(self, rel_path: str) -> Path:
        candidate = (self.root / rel_path).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise RepoAccessError(f"Path {rel_path!r} is outside the codebase root.")
        if not candidate.is_file():
            raise RepoAccessError(f"No such file: {rel_path!r}")
        return candidate

    def _resolve_dir(self, rel_path: str) -> Path:
        candidate = (self.root / rel_path).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise RepoAccessError(f"Path {rel_path!r} is outside the codebase root.")
        if not candidate.is_dir():
            raise RepoAccessError(f"No such directory: {rel_path!r}")
        return candidate

    def _source_files(self, glob: str | None = None) -> list[Path]:
        # A glob containing a separator is anchored at the repo root, so
        # 'src/**/*.py' means the top-level src/ — not every nested src/ in the
        # tree, which is what rglob's implicit '**/' prefix would give. A bare
        # pattern like '*.py' stays recursive, which is what it is normally for.
        if glob and "/" in glob:
            candidates = self.root.glob(glob)
        else:
            candidates = self.root.rglob(glob or "*")

        files: list[Path] = []
        for p in sorted(candidates):
            if not p.is_file() or _is_blocked(p, self.root):
                continue
            if glob is None and p.suffix.lower() not in _ALLOWED_EXTENSIONS:
                continue
            files.append(p)
        return files

    # ── tools ────────────────────────────────────────────────────────────────

    def grep(
        self,
        pattern: str,
        *,
        glob: str | None = None,
        case_sensitive: bool = True,
        is_regex: bool = False,
    ) -> str:
        """Exact (or regex) text search over the codebase; ripgrep-style output."""
        if not pattern:
            return "[ERROR] Empty search pattern."

        # A literal search needs no regex engine: str.__contains__ is SIMD-fast in
        # CPython and cannot backtrack, so the common case is both quicker and
        # immune to the ReDoS risk the timeout below guards against.
        spent = 0.0
        if is_regex:
            flags = 0 if case_sensitive else regex.IGNORECASE
            try:
                needle = regex.compile(pattern, flags)
            except regex.error as e:
                return f"[ERROR] Invalid regex {pattern!r}: {e}"

            def _search(text: str) -> bool:
                # Budget only time spent *inside* the regex engine — that is where
                # a catastrophically backtracking pattern hangs. Directory walking
                # and file I/O are excluded, so a large tree cannot trip the guard.
                nonlocal spent
                remaining = _GREP_TIMEOUT_S - spent
                if remaining <= 0:
                    raise TimeoutError
                started = time.monotonic()
                try:
                    return needle.search(text, timeout=remaining) is not None
                finally:
                    spent += time.monotonic() - started
        else:
            literal = pattern if case_sensitive else pattern.lower()

            def _search(text: str) -> bool:
                return literal in (text if case_sensitive else text.lower())

        # The whole repo is always scanned — the caps limit what is *shown*, never
        # how far the search reaches.
        matches: dict[str, list[tuple[int, str]]] = {}
        per_file: dict[str, int] = {}
        total = 0
        timed_out = False
        try:
            for path in self._source_files(glob):
                try:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                if not _search(text):
                    continue
                rel = path.relative_to(self.root).as_posix()
                for lineno, line in enumerate(text.splitlines(), start=1):
                    if not _search(line):
                        continue
                    total += 1
                    per_file[rel] = per_file.get(rel, 0) + 1
                    kept = matches.setdefault(rel, [])
                    if len(kept) < _MAX_MATCHES_PER_FILE:
                        kept.append((lineno, _clip(line)))
        except TimeoutError:
            timed_out = True

        # Fill the display budget round-robin — one match from every matching file
        # before any file gets a second. Taking the first 60 in walk order would
        # spend the whole budget on a handful of alphabetically-early files (a
        # vendored subtree, say) and hide the rest of the repo entirely.
        picked: list[tuple[str, int, str]] = []
        for rank in range(_MAX_MATCHES_PER_FILE):
            for rel, kept in matches.items():
                if len(picked) >= _MAX_MATCHES:
                    break
                if rank < len(kept):
                    picked.append((rel, kept[rank][0], kept[rank][1]))
            if len(picked) >= _MAX_MATCHES:
                break
        # Group by file for readability once breadth is secured.
        picked.sort(key=lambda m: (m[0], m[1]))
        hits = [f"{rel}:{lineno}: {line}" for rel, lineno, line in picked]

        if timed_out and not hits:
            return (
                f"[ERROR] Search for {pattern!r} exceeded {_GREP_TIMEOUT_S:g}s. "
                "The pattern is too expensive to evaluate — simplify it, avoid "
                "nested quantifiers, or narrow the search with a glob."
            )

        if not hits:
            scope = f" in {glob!r}" if glob else ""
            return f"No matches for {pattern!r}{scope}."

        scope = f" in {glob!r}" if glob else ""
        header = (
            f"{total} match(es) for {pattern!r}{scope} "
            f"across {len(per_file)} file(s)"
        )
        if len(hits) < total:
            header += f" — showing {len(hits)}"
        if timed_out:
            header += f" (hit the {_GREP_TIMEOUT_S:g}s budget; counts are partial)"
        header += ":"

        out = [header, *hits]

        # When matches were withheld, tell the agent exactly where they are so it
        # can go straight to the right file instead of guessing at a narrower query.
        if len(hits) < total:
            ranked = sorted(per_file.items(), key=lambda kv: (-kv[1], kv[0]))
            out += [
                "",
                f"Matches per file (top {min(len(ranked), _MAX_FILE_SUMMARY)}):",
                *(
                    f"  {count:>5}  {rel}"
                    for rel, count in ranked[:_MAX_FILE_SUMMARY]
                ),
                "Re-run with glob= to see the withheld matches in a specific file.",
            ]
        return "\n".join(out)

    def _walk(self, root: Path) -> list[tuple[int, Path, bool]]:
        """Depth-first walk yielding (depth, path, is_dir), KB filters applied.

        Directories come before files at each level, both alphabetical, so the
        rendered tree reads like `tree(1)`. Symlinked directories are not
        descended into — a link pointing at an ancestor would loop forever.
        """
        out: list[tuple[int, Path, bool]] = []

        def rec(directory: Path, depth: int) -> None:
            if depth > _MAX_TREE_DEPTH:
                return
            try:
                entries = sorted(directory.iterdir(), key=lambda p: p.name.lower())
            except OSError:
                return
            dirs = [
                e
                for e in entries
                if e.is_dir()
                and not e.is_symlink()
                and not e.name.startswith(".")
                and e.name not in _BLOCKED_DIRS
                and not e.name.endswith(".egg-info")
            ]
            files = [e for e in entries if e.is_file() and not e.name.startswith(".")]
            for d in dirs:
                out.append((depth, d, True))
                rec(d, depth + 1)
            for f in files:
                out.append((depth, f, False))

        rec(root, 0)
        return out

    def list_dir(self, rel_path: str = ".", *, max_depth: int | None = None) -> str:
        """List a directory *recursively* as an indented tree, KB filters applied.

        The whole subtree is always walked. When it does not fit the entry budget
        the tree is trimmed by *depth* rather than cut off part-way through: a
        truncated listing then shows every entry down to some complete level,
        instead of one arbitrarily deep branch with the rest of the repo missing.
        """
        try:
            root = self._resolve_dir(rel_path)
        except RepoAccessError as e:
            return f"[ERROR] {e}"

        entries = self._walk(root)
        if not entries:
            return f"{rel_path} is empty (or contains only filtered entries)."

        total_dirs = sum(1 for _, _, is_dir in entries if is_dir)
        total_files = len(entries) - total_dirs
        deepest = max(depth for depth, _, _ in entries)

        # Pick the deepest level that still fits the budget (at least the top one).
        limit = _MAX_TREE_DEPTH if max_depth is None else max(0, max_depth)
        shown_depth = min(limit, deepest)
        while shown_depth > 0:
            if sum(1 for d, _, _ in entries if d <= shown_depth) <= _MAX_ENTRIES:
                break
            shown_depth -= 1

        visible = [e for e in entries if e[0] <= shown_depth]
        clipped = len(visible) > _MAX_ENTRIES
        if clipped:
            visible = visible[:_MAX_ENTRIES]

        lines: list[str] = []
        for depth, path, is_dir in visible:
            indent = "  " * (depth + 1)
            if is_dir:
                lines.append(f"{indent}{path.name}/")
            else:
                indexed = path.suffix.lower() in _ALLOWED_EXTENSIONS
                mark = "" if indexed else "  (not indexed)"
                lines.append(f"{indent}{path.name}{mark}")

        header = (
            f"{rel_path} (recursive) — {total_dirs} dir(s), {total_files} file(s)"
        )
        omitted = len(entries) - len(visible)
        if omitted:
            header += f"; showing {len(visible)}"
            if shown_depth < deepest:
                header += f", depth ≤ {shown_depth} of {deepest}"
        header += ":"

        out = [header, *lines]
        if omitted:
            out += [
                "",
                f"{omitted} deeper entr(ies) not shown — list_dir a subdirectory "
                "to expand it, or pass max_depth.",
            ]
        return "\n".join(out)

    def read_file(
        self,
        rel_path: str,
        *,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> str:
        """Read a file (or a line range) with 1-based line numbers."""
        try:
            path = self._resolve(rel_path)
        except RepoAccessError as e:
            return f"[ERROR] {e}"

        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError as e:
            return f"[ERROR] Could not read {rel_path!r}: {e}"

        total = len(lines)
        start = max(1, start_line or 1)
        end = min(total, end_line or total)
        if start > total:
            return f"[ERROR] {rel_path!r} has {total} lines; start_line={start} is past the end."

        truncated = end - start + 1 > _MAX_READ_LINES
        if truncated:
            end = start + _MAX_READ_LINES - 1

        body = [f"{n:>5}\t{_clip(lines[n - 1])}" for n in range(start, end + 1)]
        header = f"{rel_path} (lines {start}-{end} of {total})"
        if truncated:
            header += " — truncated; request a further range to continue"
        return "\n".join([header, *body])
