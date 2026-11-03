"""Load every served document: the 30,000-document main set plus the India supplement.

Evaluation (classifier, similarity, clustering metrics, trend back-test) uses the main set only;
institution profiles and the database use everything returned here.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import path


def load_all(include_extra: bool = True) -> tuple[pd.DataFrame, np.ndarray]:
    """Documents with split and cluster columns, and their MiniLM vectors in the same row order."""
    docs = pd.read_parquet(path("documents"))
    ids = json.loads((path("processed") / "embedding_ids.json").read_text(encoding="utf-8"))
    assert ids == docs["doc_id"].tolist(), "embeddings are not aligned with documents.parquet"
    vectors = np.load(path("processed") / "embeddings_minilm.npy")
    docs = docs.merge(pd.read_parquet(path("processed") / "splits.parquet"), on="doc_id", how="left", validate="one_to_one")
    docs = docs.merge(pd.read_parquet(path("outputs") / "doc_clusters.parquet"), on="doc_id", how="left", validate="one_to_one")
    docs["lpu"] = docs["institution_raw"].fillna("").str.contains("Lovely Professional", case=False)

    extra_file = path("processed") / "documents_extra.parquet"
    if include_extra and extra_file.exists():
        extra = pd.read_parquet(extra_file).merge(pd.read_parquet(path("outputs") / "doc_clusters_extra.parquet"), on="doc_id")
        docs = pd.concat([docs, extra.reindex(columns=docs.columns)], ignore_index=True)  # patent-only columns stay empty
        vectors = np.vstack([vectors, np.load(path("processed") / "embeddings_minilm_extra.npy")])
    assert len(docs) == len(vectors) and docs["doc_id"].is_unique
    return docs, vectors
