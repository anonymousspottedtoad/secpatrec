"""Concatenate a whole codebase into one file, and measure whether it fits a model's context window.

    secpat pack codebase/contextfs_contextfs -o packs/contextfs.txt
    secpat pack --measure codebase/duffy codebase/keyshield
    secpat pack --measure --manifest-path <manifest.json>

The pack is one plain-text file holding the repo's directory tree followed by
every source file, each between explicit BEGIN/END markers carrying its
repo-relative path — the same path the annotations and `recovery.json` cite, so a
model reading the pack can ground a role without a second lookup.

The file set is the engine's own: `_ALLOWED_EXTENSIONS` and `_is_blocked` from
`codebase_loader`, so a pack contains exactly what `search` and `grep` see.
`--all-text` widens it to every file that decodes as UTF-8 (docs, configs,
Dockerfiles), which is what "the whole repo" usually means to a reader.

Token counts come from the Anthropic `count_tokens` endpoint (never a local
approximation of another vendor's tokenizer). A pack too large to send is
counted by sampling: slices are counted exactly and the repo's own
chars-per-token ratio extrapolates the rest; those figures are marked `~`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from ..codebase_loader import (
    _ALLOWED_EXTENSIONS,
    _BLOCKED_DIRS,
    _is_blocked,
)
from ..manifest import load_records

#: The engine checkout: where `.env` lives.
ROOT = Path(__file__).resolve().parents[3]

#: Context windows, in tokens. Overridable with `--model NAME=TOKENS`.
_MODELS: dict[str, int] = {
    "opus-5": 1_000_000,  # documented: 1M context, 128K max output
    "gpt-5.4": 400_000,  # assumption — override if your deployment differs
    "qwen3-30b": 262_144,  # Qwen3-30B-A3B's 256K window; a 32K serve is far smaller
}
#: A pack has to share the window with the system prompt, the tools and the
#: model's own output, so "fits" is judged against a fraction of the window.
_USABLE = 0.80
#: Largest pack sent to `count_tokens` whole; above this, sample and extrapolate.
_EXACT_LIMIT_CHARS = 2_000_000
_SAMPLE_SLICES = 4
_SAMPLE_CHARS = 120_000


@dataclass
class Pack:
    root: Path
    text: str
    files: int
    skipped: int
    chars: int


# ── file selection ───────────────────────────────────────────────────────────


def _decodes(path: Path) -> bool:
    """True if the file is UTF-8 text — the only thing a pack can carry."""
    try:
        with path.open("rb") as fh:
            head = fh.read(8192)
    except OSError:
        return False
    if b"\0" in head:
        return False
    try:
        head.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def iter_files(
    root: Path, *, all_text: bool, max_file_bytes: int = 0
) -> tuple[list[Path], list[Path]]:
    """`(packed, skipped)` — every non-blocked file, split by whether it is packable.

    `max_file_bytes` drops individual files above a size. The extension filter
    admits `.json` and `.xml`, which in a research repo are as often a bundled
    dataset as configuration — `iris-vector-rag` is 224 MB of PMC article XML
    against 5 MB of Python — and a pack is for reading code, so a cap that
    excludes one dataset file is worth more than every chunk of it.
    """
    packed: list[Path] = []
    skipped: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink() or _is_blocked(path, root):
            continue
        if path.name.startswith("."):
            skipped.append(path)
            continue
        if max_file_bytes and path.stat().st_size > max_file_bytes:
            skipped.append(path)
            continue
        if all_text:
            (packed if _decodes(path) else skipped).append(path)
        else:
            (packed if path.suffix.lower() in _ALLOWED_EXTENSIONS else skipped).append(path)
    return packed, skipped


def render_tree(root: Path, packed: set[Path]) -> str:
    """Indented tree of every non-blocked directory and file, packed ones marked."""
    lines = [f"{root.name}/"]

    def walk(directory: Path, depth: int) -> None:
        try:
            entries = sorted(directory.iterdir(), key=lambda p: (p.is_file(), p.name))
        except OSError:
            return
        indent = "  " * (depth + 1)
        for entry in entries:
            if entry.is_symlink():
                continue
            if entry.is_dir():
                if (
                    entry.name in _BLOCKED_DIRS
                    or entry.name.startswith(".")
                    or entry.name.endswith(".egg-info")
                ):
                    continue
                lines.append(f"{indent}{entry.name}/")
                walk(entry, depth + 1)
            else:
                mark = "" if entry in packed else "  (not packed)"
                lines.append(f"{indent}{entry.name}{mark}")

    walk(root, 0)
    return "\n".join(lines)


def build_pack(root: Path, *, all_text: bool, line_numbers: bool, max_file_bytes: int = 0) -> Pack:
    packed, skipped = iter_files(root, all_text=all_text, max_file_bytes=max_file_bytes)
    parts = [
        f"# Repo pack: {root.name}",
        f"# Files: {len(packed)} packed, {len(skipped)} not packed"
        f" | filter: {'all-text' if all_text else 'source-extensions'}"
        + (f", max {max_file_bytes:,} bytes/file" if max_file_bytes else ""),
        "",
        "===== DIRECTORY TREE =====",
        render_tree(root, set(packed)),
        "",
        "===== FILES =====",
        "",
    ]
    for path in packed:
        rel = path.relative_to(root).as_posix()
        try:
            body = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:  # unreadable mid-walk; record it rather than dying
            parts.append(
                f"--- BEGIN FILE: {rel} (unreadable: {exc}) ---\n--- END FILE: {rel} ---\n"
            )
            continue
        lines = body.splitlines()
        if line_numbers:
            body = "\n".join(f"{i:>5}| {line}" for i, line in enumerate(lines, 1))
        parts.append(f"--- BEGIN FILE: {rel} ({len(lines)} lines) ---")
        parts.append(body)
        parts.append(f"--- END FILE: {rel} ---")
        parts.append("")
    text = "\n".join(parts)
    return Pack(root=root, text=text, files=len(packed), skipped=len(skipped), chars=len(text))


# ── token counting ───────────────────────────────────────────────────────────


def _load_key() -> str | None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    env = ROOT / ".env"
    if not env.is_file():
        return None
    for line in env.read_text(encoding="utf-8").splitlines():
        name, _, value = line.partition("=")
        if name.strip() == "ANTHROPIC_API_KEY":
            return value.strip() or None
    return None


class Counter:
    """Exact token counts from the API, with sampling for packs too large to send."""

    def __init__(self, model: str = "claude-opus-5") -> None:
        import anthropic

        key = _load_key()
        if not key:
            raise SystemExit("error: ANTHROPIC_API_KEY not set and not found in .env")
        self._client = anthropic.Anthropic(api_key=key)
        self._model = model

    def _count(self, text: str) -> int:
        resp = self._client.messages.count_tokens(
            model=self._model, messages=[{"role": "user", "content": text}]
        )
        return resp.input_tokens

    def count(self, text: str) -> tuple[int, bool]:
        """`(tokens, exact)` — sampled and extrapolated when the pack is too large to send."""
        if len(text) <= _EXACT_LIMIT_CHARS:
            return self._count(text), True
        step = len(text) // _SAMPLE_SLICES
        sampled_chars = sampled_tokens = 0
        for i in range(_SAMPLE_SLICES):
            slice_ = text[i * step : i * step + _SAMPLE_CHARS]
            sampled_chars += len(slice_)
            sampled_tokens += self._count(slice_)
        return round(len(text) * sampled_tokens / sampled_chars), False


# ── commands ─────────────────────────────────────────────────────────────────


def _fits(tokens: int, models: dict[str, int]) -> str:
    return "  ".join(
        f"{name}:{'yes' if tokens <= window * _USABLE else 'no ':>4}"
        for name, window in models.items()
    )


def measure(
    roots: list[Path],
    models: dict[str, int],
    *,
    all_text: bool,
    as_json: bool,
    max_file_bytes: int = 0,
) -> int:
    counter = Counter()
    rows = []
    for root in roots:
        pack = build_pack(
            root, all_text=all_text, line_numbers=False, max_file_bytes=max_file_bytes
        )
        tokens, exact = counter.count(pack.text)
        rows.append(
            {
                "repo": root.name,
                "path": str(root),
                "files": pack.files,
                "chars": pack.chars,
                "tokens": tokens,
                "exact": exact,
                "fits": {n: tokens <= w * _USABLE for n, w in models.items()},
            }
        )
        if not as_json:
            print(f"  measured {root.name}: {tokens:,} tokens", file=sys.stderr)
    rows.sort(key=lambda r: r["tokens"])

    if as_json:
        print(json.dumps({"usable_fraction": _USABLE, "models": models, "repos": rows}, indent=2))
        return 0

    print(f"\n{'repo':<40} {'files':>6} {'tokens':>12}   fits (at {_USABLE:.0%} of window)")
    print("-" * 100)
    for row in rows:
        tok = f"{'~' if not row['exact'] else ''}{row['tokens']:,}"
        print(f"{row['repo']:<40} {row['files']:>6} {tok:>12}   {_fits(row['tokens'], models)}")
    print("-" * 100)
    for name, window in models.items():
        n = sum(1 for r in rows if r["fits"][name])
        print(
            f"{name:<12} window {window:>9,}  usable {int(window * _USABLE):>9,}"
            f"  fits: {n}/{len(rows)} repos"
        )
    return 0


def _manifest_roots(manifest: Path) -> list[Path]:
    """The distinct codebases named by an evaluation manifest, in repo order."""
    seen: dict[str, Path] = {}
    for run in load_records(manifest):
        path = Path(run["codebase_path"])
        if path.is_dir():
            seen.setdefault(run["repo"], path)
        else:
            print(f"warning: {run['repo']}: {path} missing, skipped", file=sys.stderr)
    return list(seen.values())


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog=prog,
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("roots", nargs="*", type=Path, help="Codebase directories to pack or measure")
    ap.add_argument("-o", "--out", type=Path, help="Write the pack here (default: stdout)")
    ap.add_argument("--measure", action="store_true", help="Report token counts instead of packing")
    ap.add_argument(
        "--manifest-path", type=Path, help="Take the repo list from an evaluation manifest"
    )
    ap.add_argument(
        "--all-text",
        action="store_true",
        help="Pack every UTF-8 file, not just the KB's source extensions",
    )
    ap.add_argument(
        "--line-numbers",
        action="store_true",
        help="Prefix each line with its number (costs ~15%% more tokens)",
    )
    ap.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="NAME=TOKENS",
        help="Add or override a context window for the fit table",
    )
    ap.add_argument(
        "--max-file-bytes",
        type=int,
        default=0,
        metavar="N",
        help="Skip files larger than N bytes (0 = no cap); excludes bundled datasets",
    )
    ap.add_argument("--json", action="store_true", help="Machine-readable measure output")
    args = ap.parse_args(argv)

    models = dict(_MODELS)
    for spec in args.model:
        name, _, window = spec.partition("=")
        models[name.strip()] = int(window)

    roots = list(args.roots)
    if args.manifest_path:
        roots += _manifest_roots(args.manifest_path)
    if not roots:
        ap.error("give at least one codebase directory, or --manifest-path")
    for root in roots:
        if not root.is_dir():
            ap.error(f"not a directory: {root}")

    if args.measure:
        return measure(
            roots,
            models,
            all_text=args.all_text,
            as_json=args.json,
            max_file_bytes=args.max_file_bytes,
        )

    if len(roots) > 1:
        ap.error("packing takes one codebase; use --measure for several")
    pack = build_pack(
        roots[0],
        all_text=args.all_text,
        line_numbers=args.line_numbers,
        max_file_bytes=args.max_file_bytes,
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(pack.text, encoding="utf-8")
        print(f"{pack.files} files, {pack.chars:,} chars -> {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(pack.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
