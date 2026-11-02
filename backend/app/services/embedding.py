"""Query embedding with fastembed (ONNX, no PyTorch), the same MiniLM model as the documents."""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from app import settings


@lru_cache(maxsize=1)
def _model():
    """Load the fastembed model once (downloads about 90 MB on first use)."""
    from fastembed import TextEmbedding

    return TextEmbedding(settings.EMBEDDING_MODEL)


def embed(text: str) -> np.ndarray:
    """L2-normalised 384-dimension vector for `text`."""
    v = np.asarray(next(iter(_model().embed([text]))), dtype=np.float32)
    return v / np.linalg.norm(v)
