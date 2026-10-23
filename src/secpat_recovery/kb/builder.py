from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import chromadb
from secpat_scoring.containment.ast_index import parse_source, tree_to_summary

from ..codebase_loader import _ALLOWED_EXTENSIONS, _is_blocked
from .chunker import chunk_file
from .embedder import embed_texts

_COLLECTION = "codebase"
_STRUCT_FILE = "struct.json"
_SOURCE_FILE = "source.json"
_BATCH = 512


def kb_source(kb_dir: Path) -> str | None:
    """Return the absolute codebase root a KB was built from, or None if unknown."""
    source_path = kb_dir / _SOURCE_FILE
    if not source_path.exists():
        return None
    try:
        return json.loads(source_path.read_text(encoding="utf-8")).get("root")
    except (json.JSONDecodeError, OSError):
        return None


def _to_meta(chunk: dict) -> dict:
    return {
        "path": chunk["path"],
        "symbol": chunk["symbol"],
        "kind": chunk["kind"],
        "class_name": chunk.get("class_name") or "",
        "function_name": chunk.get("function_name") or "",
        "start_line": chunk.get("start_line", 0),
        "end_line": chunk.get("end_line", 0),
        "docstring": chunk.get("docstring") or "",
        "arguments": json.dumps(chunk.get("arguments", [])),
        "calls": json.dumps(chunk.get("calls", [])),
        "decorators": json.dumps(chunk.get("decorators", [])),
        "code": chunk.get("code", ""),
    }


def build_kb(root: Path, kb_dir: Path = Path(".kb"), *, force: bool = False) -> None:
    """Walk root, build AST trees, chunk, embed, and store in ChromaDB + struct index."""
    if force and kb_dir.exists():
        shutil.rmtree(kb_dir)
    kb_dir.mkdir(parents=True, exist_ok=True)

    # Record which codebase this KB was built from, so a later run against a
    # different codebase can detect the mismatch and rebuild instead of silently
    # reusing a stale index.
    (kb_dir / _SOURCE_FILE).write_text(json.dumps({"root": str(root.resolve())}), encoding="utf-8")

    # Collect source files
    all_paths: list[Path] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if _is_blocked(p, root):
            continue
        if p.suffix.lower() not in _ALLOWED_EXTENSIONS:
            continue
        all_paths.append(p)

    # Build AST trees and structural index
    struct_index: dict[str, dict] = {}
    chunks: list[dict] = []

    # The parser is the scoring package's, read the way its `build_index` reads,
    # so this `struct.json` is the index a recovery is later scored over.
    for p in all_paths:
        rel = str(p.relative_to(root))
        try:
            source = p.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            print(f"[KB WARN] cannot read {rel}: {e}", file=sys.stderr)
            continue
        tree = parse_source(source, rel)
        if tree is None:
            continue
        struct_index[rel] = tree_to_summary(tree)
        chunks.extend(chunk_file(p, tree))

    # Persist structural index
    struct_path = kb_dir / _STRUCT_FILE
    struct_path.write_text(json.dumps(struct_index, indent=2), encoding="utf-8")
    print(
        f"[KB] AST index: {len(struct_index)} files → {struct_path}",
        file=sys.stderr,
    )

    if not chunks:
        print("[KB] No chunks extracted.", file=sys.stderr)
        return

    print(
        f"[KB] {len(chunks)} chunks from {len(all_paths)} files. Embedding...",
        file=sys.stderr,
    )

    for i, c in enumerate(chunks):
        c["id"] = str(i)

    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

    client = chromadb.PersistentClient(path=str(kb_dir))
    try:
        client.delete_collection(_COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(_COLLECTION, metadata={"hnsw:space": "cosine"})

    for start in range(0, len(chunks), _BATCH):
        batch = chunks[start : start + _BATCH]
        batch_emb = embeddings[start : start + _BATCH]
        collection.add(
            ids=[c["id"] for c in batch],
            embeddings=batch_emb.tolist(),
            documents=[c["text"] for c in batch],
            metadatas=[_to_meta(c) for c in batch],
        )

    print(f"[KB] Stored in {kb_dir}", file=sys.stderr)


@dataclass
class KBIndex:
    _collection: chromadb.Collection
    _struct: dict[str, dict] = field(default_factory=dict)

    def search(self, query_text: str, top_k: int = 10, *, min_score: float = 0.0) -> list[dict]:
        """Semantic embedding search.

        Returns hit dicts carrying the chunk's full `code`, so a search is a
        find-and-read in one step. `min_score` drops hits whose cosine
        similarity does not exceed it (results are score-ordered, so this
        truncates the tail).
        """
        q_vec = embed_texts([query_text])[0].tolist()
        results = self._collection.query(
            query_embeddings=[q_vec],
            n_results=min(top_k, self._collection.count()),
            include=["metadatas", "distances"],
        )
        out = []
        for chunk_id, meta, dist in zip(
            results["ids"][0], results["metadatas"][0], results["distances"][0]
        ):
            score = round(1 - dist, 4)
            if score <= min_score:
                continue
            out.append(
                {
                    "id": chunk_id,
                    "path": meta["path"],
                    "symbol": meta["symbol"],
                    "kind": meta["kind"],
                    "class_name": meta["class_name"] or None,
                    "function_name": meta["function_name"] or None,
                    "start_line": meta["start_line"],
                    "end_line": meta["end_line"],
                    "docstring": meta["docstring"],
                    "calls": json.loads(meta["calls"]),
                    "code": meta["code"],
                    "score": score,
                }
            )
        return out

    def get(self, chunk_id: str) -> dict | None:
        """Exact lookup by id. Returns full dict including code."""
        result = self._collection.get(ids=[chunk_id], include=["metadatas", "documents"])
        if not result["ids"]:
            return None
        meta = result["metadatas"][0]
        doc = result["documents"][0]
        return {
            "id": chunk_id,
            "path": meta["path"],
            "symbol": meta["symbol"],
            "kind": meta["kind"],
            "class_name": meta["class_name"] or None,
            "function_name": meta["function_name"] or None,
            "start_line": meta["start_line"],
            "end_line": meta["end_line"],
            "docstring": meta["docstring"],
            "arguments": json.loads(meta["arguments"]),
            "calls": json.loads(meta["calls"]),
            "decorators": json.loads(meta["decorators"]),
            "code": meta["code"],
            "text": doc,
        }

    def get_tree(self, path: str) -> dict | None:
        """Return the AST symbol tree summary for a file path, or None if not found."""
        return self._struct.get(path)

    def count(self) -> int:
        return self._collection.count()


def load_kb(kb_dir: Path) -> KBIndex:
    """Load an existing ChromaDB KB and structural index from kb_dir."""
    client = chromadb.PersistentClient(path=str(kb_dir))
    collection = client.get_collection(_COLLECTION)

    struct: dict[str, dict] = {}
    struct_path = kb_dir / _STRUCT_FILE
    if struct_path.exists():
        try:
            struct = json.loads(struct_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[KB WARN] could not load struct index: {e}", file=sys.stderr)

    return KBIndex(_collection=collection, _struct=struct)
