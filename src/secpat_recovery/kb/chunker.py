from __future__ import annotations

import sys
from pathlib import Path

from secpat_scoring.containment.ast_index import FileTree, SymbolNode

# ── Chunk text assembly ───────────────────────────────────────────────────────


def _build_text(chunk: dict) -> str:
    parts = [f"File:\n{chunk['path']}"]
    if chunk.get("class_name"):
        parts.append(f"Class:\n{chunk['class_name']}")
    kind = chunk.get("kind", "function")
    if kind in ("method", "function"):
        label = "Method" if kind == "method" else "Function"
        parts.append(f"{label}:\n{chunk.get('function_name', chunk.get('symbol', ''))}")
        if chunk.get("arguments"):
            parts.append("Arguments:\n" + "\n".join(chunk["arguments"]))
        if chunk.get("calls"):
            parts.append("Calls:\n" + "\n".join(chunk["calls"]))
    elif kind in ("class", "enum"):
        parts.append(f"{'Enum' if kind == 'enum' else 'Class'}:\n{chunk.get('symbol', '')}")
    parts.append(f"Code:\n{chunk.get('code', '')}")
    return "\n\n".join(parts)


# ── FileTree → chunk dicts ────────────────────────────────────────────────────


def _sym_to_chunk(sym: SymbolNode, path: str, class_name: str | None = None) -> dict:
    is_class = sym.kind in ("class", "enum")
    effective_class = sym.name if is_class else class_name
    code = "\n".join(sym.code.splitlines()[:40]) if is_class else sym.code

    chunk = {
        "path": path,
        "symbol": sym.qualified_name,
        "kind": sym.kind,
        "class_name": effective_class,
        "function_name": sym.name if not is_class else None,
        "start_line": sym.start_line,
        "end_line": sym.end_line,
        "arguments": sym.params,
        "calls": sym.calls,
        "decorators": sym.decorators,
        "docstring": sym.docstring,
        "code": code,
    }
    chunk["text"] = _build_text(chunk)
    return chunk


def _sym_to_chunks(sym: SymbolNode, path: str, class_name: str | None = None) -> list[dict]:
    """One chunk per symbol, recursing so a Java nested type's methods are chunked too."""
    chunks = [_sym_to_chunk(sym, path, class_name)]
    owner = sym.name if sym.kind in ("class", "enum") else class_name
    for child in sym.children:
        chunks.extend(_sym_to_chunks(child, path, owner))
    return chunks


def _tree_to_chunks(tree: FileTree) -> list[dict]:
    return [c for sym in tree.symbols for c in _sym_to_chunks(sym, tree.path)]


def _file_fallback(path: Path, rel_path: str) -> list[dict]:
    try:
        lines = path.read_bytes().decode("utf-8", errors="replace").splitlines()[:60]
    except OSError as e:
        print(f"[KB WARN] cannot read {rel_path}: {e}", file=sys.stderr)
        return []
    chunk = {
        "path": rel_path,
        "symbol": path.stem,
        "kind": "file",
        "class_name": None,
        "function_name": None,
        "start_line": 1,
        "end_line": len(lines),
        "arguments": [],
        "calls": [],
        "decorators": [],
        "docstring": "",
        "code": "\n".join(lines),
    }
    chunk["text"] = _build_text(chunk)
    return [chunk]


# ── Public API ────────────────────────────────────────────────────────────────


def chunk_file(path: Path, tree: FileTree) -> list[dict]:
    """Slice a parsed file into semantic chunks; one file-level chunk if it has no symbols."""
    chunks = _tree_to_chunks(tree)
    return chunks if chunks else _file_fallback(path, tree.path)
