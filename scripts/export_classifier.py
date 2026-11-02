"""Export the served classifier (MiniLM embeddings + Logistic Regression) to plain JSON.

The backend then predicts with a NumPy softmax over these weights, so it needs neither
scikit-learn nor the pickle file at runtime. Output: backend/app/models/classifier.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ROOT, path  # noqa: E402


def main() -> None:
    """Write classes, weights and intercepts, after checking the softmax matches predict_proba."""
    served = joblib.load(path("models") / "served_classifier.joblib")
    lr = served["model"]
    x = np.random.default_rng(0).normal(size=(5, lr.coef_.shape[1]))
    z = x @ lr.coef_.T + lr.intercept_
    p = np.exp(z - z.max(axis=1, keepdims=True))
    assert np.allclose(p / p.sum(axis=1, keepdims=True), lr.predict_proba(x), atol=1e-6)

    out = ROOT / "backend" / "app" / "models" / "classifier.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": "MiniLM embeddings + Logistic Regression",
        "embedding": served["embedding"],
        "classes": [str(c) for c in lr.classes_],
        "coef": lr.coef_.round(7).tolist(),
        "intercept": lr.intercept_.round(7).tolist(),
    }), encoding="utf-8")
    print(f"Wrote {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
