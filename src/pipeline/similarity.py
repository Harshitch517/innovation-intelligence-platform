"""Phase 5: similarity detection and semantic search over the document embeddings.

- SimilarityIndex: a FAISS inner-product index over L2-normalised vectors (inner product =
  cosine similarity) with `similar(doc_id, k)` and `search(query, k, filters)`.
- Retrieval evaluation (PLAN.md Section 7): for each query document the relevant documents
  are those sharing its primary CPC subgroup (patents) or linked to it by a citation in either
  direction. Reports Precision@10, Recall@10 and MRR for MiniLM, SPECTER2 and a TF-IDF baseline.
- Semantic search evaluation: Precision@10 on 20 hand-written queries (4 per domain), where a
  result counts as correct if its domain matches the query's intended domain.

Outputs: models/faiss_minilm.index, reports/similarity_eval.json, reports/search_eval.json

Usage:
    python -m src.pipeline.similarity
"""
from __future__ import annotations

import json
from collections import defaultdict

import faiss
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import load_config, path

EVAL_QUERIES: dict[str, list[str]] = {
    "Artificial Intelligence": [
        "neural network training method with reduced memory",
        "large language model for question answering",
        "reinforcement learning agent for decision making",
        "graph neural network for node classification",
    ],
    "Healthcare": [
        "wearable sensor for continuous heart rate monitoring",
        "MRI image reconstruction for diagnosis",
        "electronic health record system for patient data",
        "minimally invasive surgical instrument",
    ],
    "Robotics": [
        "robot arm grasping objects with a gripper",
        "path planning for autonomous mobile robots",
        "soft robotic actuator",
        "human robot collaboration in manufacturing",
    ],
    "Cybersecurity": [
        "intrusion detection system for network traffic",
        "encryption scheme for secure data sharing",
        "malware detection using machine learning",
        "user authentication protocol",
    ],
    "Renewable Energy": [
        "perovskite solar cell efficiency",
        "wind turbine blade control",
        "maximum power point tracking for photovoltaic systems",
        "hydrogen production by water electrolysis",
    ],
}


class SimilarityIndex:
    """FAISS cosine-similarity index over the document embeddings."""

    def __init__(self, docs: pd.DataFrame, vectors: np.ndarray, model_name: str | None = None) -> None:
        self.docs = docs.reset_index(drop=True)
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.row = {d: i for i, d in enumerate(self.docs["doc_id"])}
        self.index = faiss.IndexFlatIP(self.vectors.shape[1])
        self.index.add(self.vectors)
        self.model_name = model_name
        self._encoder = None

    def _encode(self, text: str) -> np.ndarray:
        """Embed a query with the same sentence-transformers model as the documents."""
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.model_name, device="cpu")
        return self._encoder.encode([text], normalize_embeddings=True).astype(np.float32)

    def _results(self, scores: np.ndarray, rows: np.ndarray, k: int, keep=None) -> pd.DataFrame:
        """Turn FAISS output into a result table, applying an optional row filter."""
        out = []
        for s, r in zip(scores, rows):
            if r < 0 or (keep is not None and not keep[r]):
                continue
            out.append((int(r), float(s)))
            if len(out) == k:
                break
        res = self.docs.iloc[[r for r, _ in out]][["doc_id", "source", "title", "institution_name", "year", "domain"]].copy()
        res["score"] = [round(s, 4) for _, s in out]
        return res.reset_index(drop=True)

    def similar(self, doc_id: str, k: int = 10) -> pd.DataFrame:
        """Top-k most similar documents to `doc_id` (excluding itself), with cosine scores."""
        r = self.row[doc_id]
        scores, rows = self.index.search(self.vectors[r : r + 1], k + 1)
        keep = np.ones(len(self.docs), dtype=bool)
        keep[r] = False
        return self._results(scores[0], rows[0], k, keep)

    def search(self, query: str, k: int = 10, domain: str | None = None, year_from: int | None = None,
               year_to: int | None = None, source: str | None = None) -> pd.DataFrame:
        """Top-k documents for a natural-language query, with optional filters."""
        keep = np.ones(len(self.docs), dtype=bool)
        d = self.docs
        if domain:
            keep &= (d["domain"] == domain).to_numpy()
        if year_from:
            keep &= (d["year"] >= year_from).to_numpy()
        if year_to:
            keep &= (d["year"] <= year_to).to_numpy()
        if source:
            keep &= (d["source"] == source).to_numpy()
        n = len(d) if not keep.all() else k
        scores, rows = self.index.search(self._encode(query), n)
        return self._results(scores[0], rows[0], k, keep)


# ---------------------------------------------------------------- evaluation

def relevance_sets(docs: pd.DataFrame) -> dict[int, set[int]]:
    """Row -> set of relevant rows: same primary CPC subgroup (patents) or a citation link."""
    row = {d: i for i, d in enumerate(docs["doc_id"])}
    rel: dict[int, set[int]] = defaultdict(set)
    for i, cited in enumerate(docs["cited_ids"]):
        for c in cited:
            j = row.get(c)
            if j is not None and j != i:
                rel[i].add(j)
                rel[j].add(i)
    patents = docs[docs["source"] == "patent"]
    groups = patents.groupby(patents["raw_codes"].map(lambda codes: codes[0])).indices
    patent_rows = patents.index.to_numpy()
    for members in groups.values():
        rows = set(patent_rows[members].tolist())
        if 1 < len(rows) <= 200:  # very large subgroups say little about similarity
            for i in rows:
                rel[i] |= rows - {i}
    return rel


def retrieval_metrics(vectors: np.ndarray, rel: dict[int, set[int]], k: int = 10) -> dict:
    """Mean Precision@k, Recall@k and MRR over all documents that have relevant documents."""
    queries = np.array(sorted(rel))
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(np.ascontiguousarray(vectors, dtype=np.float32))
    _, rows = index.search(np.ascontiguousarray(vectors[queries], dtype=np.float32), k + 1)
    p, r, mrr = [], [], []
    for q, ranked in zip(queries, rows):
        ranked = [x for x in ranked if x != q][:k]
        hits = [x in rel[q] for x in ranked]
        p.append(sum(hits) / k)
        r.append(sum(hits) / len(rel[q]))
        mrr.append(next((1 / (i + 1) for i, h in enumerate(hits) if h), 0.0))
    return {"queries": int(len(queries)), f"precision@{k}": round(float(np.mean(p)), 4),
            f"recall@{k}": round(float(np.mean(r)), 4), "mrr": round(float(np.mean(mrr)), 4)}


def tfidf_vectors(docs: pd.DataFrame) -> np.ndarray:
    """L2-normalised TF-IDF vectors reduced to 384 dimensions (baseline for retrieval)."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize

    x = TfidfVectorizer(min_df=2, max_df=0.9, sublinear_tf=True, stop_words="english").fit_transform(docs["text"])
    return normalize(TruncatedSVD(384, random_state=42).fit_transform(x)).astype(np.float32)


def search_eval(index: SimilarityIndex, k: int = 10) -> dict:
    """Precision@k of the 20 hand-written queries (correct = same domain as intended)."""
    per_query, by_domain = [], defaultdict(list)
    for domain, queries in EVAL_QUERIES.items():
        for q in queries:
            res = index.search(q, k)
            p = float((res["domain"] == domain).mean())
            per_query.append({"query": q, "domain": domain, "precision@10": round(p, 2),
                              "top3": res["title"].head(3).tolist()})
            by_domain[domain].append(p)
    return {"queries": len(per_query), "precision@10": round(float(np.mean([q["precision@10"] for q in per_query])), 4),
            "by_domain": {d: round(float(np.mean(v)), 4) for d, v in by_domain.items()}, "per_query": per_query}


def main() -> None:
    """Build the MiniLM index and run every evaluation."""
    cfg = load_config()
    docs = pd.read_parquet(path("documents"))
    proc = path("processed")
    ids = json.loads((proc / "embedding_ids.json").read_text(encoding="utf-8"))
    assert ids == docs["doc_id"].tolist()
    vec = {n: np.load(proc / f"embeddings_{n}.npy") for n in ("minilm", "specter2")}

    index = SimilarityIndex(docs, vec["minilm"], cfg["models"]["embedding"])
    faiss.write_index(index.index, str(path("models") / "faiss_minilm.index"))

    rel = relevance_sets(docs)
    print(f"{len(rel):,} documents have at least one relevant document")
    sim = {"ground_truth": "same primary CPC subgroup (patents, subgroups of 2-200 docs) or a citation link",
           "k": 10, "minilm": retrieval_metrics(vec["minilm"], rel), "specter2": retrieval_metrics(vec["specter2"], rel),
           "tfidf_svd_baseline": retrieval_metrics(tfidf_vectors(docs), rel)}
    rng = np.random.default_rng(cfg["seed"])
    random_vec = rng.normal(size=(len(docs), 384)).astype(np.float32)
    sim["random_baseline"] = retrieval_metrics(random_vec / np.linalg.norm(random_vec, axis=1, keepdims=True), rel)
    for name in ("minilm", "specter2", "tfidf_svd_baseline", "random_baseline"):
        print(f"  {name:<20} {sim[name]}")


    search = search_eval(index)
    print(f"Search Precision@10 over 20 queries: {search['precision@10']} {search['by_domain']}")
    reports = path("reports")
    (reports / "similarity_eval.json").write_text(json.dumps(sim, indent=2), encoding="utf-8")
    (reports / "search_eval.json").write_text(json.dumps(search, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
