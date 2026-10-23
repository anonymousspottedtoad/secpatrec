from __future__ import annotations

import numpy as np

_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of strings. Returns shape (N, D) float32 array, L2-normalized."""
    model = _get_model()
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
