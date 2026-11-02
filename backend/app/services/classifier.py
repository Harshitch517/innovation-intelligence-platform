"""Domain classifier for /classify: Logistic Regression weights exported to JSON, applied with NumPy."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.services.embedding import embed

MODEL_FILE = Path(__file__).resolve().parents[1] / "models" / "classifier.json"


@lru_cache(maxsize=1)
def _weights() -> tuple[list[str], np.ndarray, np.ndarray, str]:
    """Classes, coefficient matrix, intercepts and model name."""
    m = json.loads(MODEL_FILE.read_text(encoding="utf-8"))
    return m["classes"], np.array(m["coef"], dtype=np.float32), np.array(m["intercept"], dtype=np.float32), m["model"]


def classify(title: str, abstract: str) -> dict:
    """Predicted domain and class probabilities for a new title + abstract."""
    classes, coef, intercept, name = _weights()
    z = coef @ embed(f"{title.strip()}. {abstract.strip()}") + intercept
    p = np.exp(z - z.max())
    p /= p.sum()
    order = np.argsort(-p)
    return {"domain": classes[order[0]], "probabilities": {classes[i]: round(float(p[i]), 4) for i in order}, "model": name}
