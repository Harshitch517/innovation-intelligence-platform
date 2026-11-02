"""Phase 6: cluster innovations into sub-areas with BERTopic (UMAP + HDBSCAN).

Two clusterings on the MiniLM embeddings:
  global  all 30,000 documents together (compared with the domain labels via NMI / ARI)
  domain  each domain separately, giving the named sub-areas used by trends, the API and UI

HDBSCAN leaves some documents as outliers (topic -1). Metrics are computed on the raw result;
afterwards outliers are assigned to their nearest cluster (by embedding) so every document has
a sub-area for trend counting. Cluster keywords come from BERTopic's class-based TF-IDF.

Outputs: data/outputs/clusters.parquet       one row per cluster (id, scope, domain, keywords, size, sample titles)
         data/outputs/doc_clusters.parquet   doc_id -> global_cluster, domain_cluster (+ outlier flags)
         reports/clustering_eval.json        silhouette, NMI, ARI, outlier shares, cluster counts

Usage:
    python -m src.pipeline.cluster
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from umap import UMAP

from src.config import load_config, path

DOMAIN_CODES = {
    "Artificial Intelligence": "AI",
    "Healthcare": "HC",
    "Robotics": "RB",
    "Cybersecurity": "CS",
    "Renewable Energy": "RE",
}


def make_model(min_cluster_size: int, seed: int) -> BERTopic:
    """BERTopic with fixed-seed UMAP, HDBSCAN and an English bigram vectorizer."""
    return BERTopic(
        umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=seed),
        hdbscan_model=HDBSCAN(min_cluster_size=min_cluster_size, min_samples=10, metric="euclidean",
                              cluster_selection_method="eom", prediction_data=True),
        vectorizer_model=CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=5, max_df=0.5),
        top_n_words=10,
        calculate_probabilities=False,
        verbose=False,
    )


def fit(texts: list[str], vectors: np.ndarray, min_cluster_size: int, seed: int) -> tuple[BERTopic, np.ndarray, np.ndarray]:
    """Fit BERTopic; return the model, raw topics (with -1 outliers) and outlier-free topics."""
    model = make_model(min_cluster_size, seed)
    raw = np.array(model.fit_transform(texts, embeddings=vectors)[0])
    if (raw == -1).any():
        reduced = np.array(model.reduce_outliers(texts, raw, strategy="embeddings", embeddings=vectors))
    else:
        reduced = raw.copy()
    return model, raw, reduced


def silhouette(vectors: np.ndarray, labels: np.ndarray, seed: int) -> float | None:
    """Cosine silhouette on the original embeddings, outliers excluded (sampled for speed)."""
    keep = labels != -1
    if len(set(labels[keep])) < 2:
        return None
    return round(float(silhouette_score(vectors[keep], labels[keep], metric="cosine",
                                        sample_size=min(10_000, int(keep.sum())), random_state=seed)), 4)


def cluster_rows(model: BERTopic, topics: np.ndarray, titles: pd.Series, prefix: str, scope: str, domain: str | None) -> list[dict]:
    """One record per cluster: id, keywords, size and five sample titles."""
    rows = []
    for t in sorted(set(topics)):
        if t == -1:
            continue
        members = np.flatnonzero(topics == t)
        words = [w for w, _ in (model.get_topic(t) or [])]
        rows.append({
            "cluster_id": f"{prefix}-{t}",
            "scope": scope,
            "domain": domain,
            "keywords": words,
            "size": int(len(members)),
            "sample_titles": titles.iloc[members[:5]].tolist(),
        })
    return rows


def main() -> None:
    """Run the global and per-domain clusterings and save clusters, assignments and metrics."""
    cfg = load_config()
    seed = cfg["seed"]
    docs = pd.read_parquet(path("documents"))
    proc = path("processed")
    assert json.loads((proc / "embedding_ids.json").read_text(encoding="utf-8")) == docs["doc_id"].tolist()
    vectors = np.load(proc / "embeddings_minilm.npy")
    texts = docs["text"].tolist()

    print("Global clustering ...")
    start = time.perf_counter()
    g_model, g_raw, g_red = fit(texts, vectors, min_cluster_size=60, seed=seed)
    clusters = cluster_rows(g_model, g_red, docs["title"], "G", "global", None)
    assign = pd.DataFrame({"doc_id": docs["doc_id"], "global_cluster": [f"G-{t}" for t in g_red],
                           "global_outlier": g_raw == -1})
    domain_ids = docs["domain"].astype("category").cat.codes.to_numpy()
    evaluation = {
        "embedding": cfg["models"]["embedding"],
        "global": {
            "clusters": int(len(set(g_red))),
            "outlier_share": round(float((g_raw == -1).mean()), 4),
            "silhouette_cosine_excl_outliers": silhouette(vectors, g_raw, seed),
            "nmi_vs_domain_excl_outliers": round(float(normalized_mutual_info_score(domain_ids[g_raw != -1], g_raw[g_raw != -1])), 4),
            "ari_vs_domain_excl_outliers": round(float(adjusted_rand_score(domain_ids[g_raw != -1], g_raw[g_raw != -1])), 4),
            "nmi_vs_domain_all_docs": round(float(normalized_mutual_info_score(domain_ids, g_red)), 4),
            "ari_vs_domain_all_docs": round(float(adjusted_rand_score(domain_ids, g_red)), 4),
            "seconds": round(time.perf_counter() - start, 1),
        },
        "domain": {},
    }
    print(f"  {evaluation['global']}")

    assign["domain_cluster"] = ""
    assign["domain_outlier"] = False
    for domain, code in DOMAIN_CODES.items():
        mask = (docs["domain"] == domain).to_numpy()
        print(f"Clustering {domain} ...")
        model, raw, red = fit([texts[i] for i in np.flatnonzero(mask)], vectors[mask], min_cluster_size=30, seed=seed)
        clusters += cluster_rows(model, red, docs.loc[mask, "title"].reset_index(drop=True), code, "domain", domain)
        assign.loc[mask, "domain_cluster"] = [f"{code}-{t}" for t in red]
        assign.loc[mask, "domain_outlier"] = raw == -1
        evaluation["domain"][domain] = {
            "clusters": int(len(set(red))),
            "outlier_share": round(float((raw == -1).mean()), 4),
            "silhouette_cosine_excl_outliers": silhouette(vectors[mask], raw, seed),
        }
        print(f"  {evaluation['domain'][domain]}")

    out = path("outputs")
    pd.DataFrame(clusters).to_parquet(out / "clusters.parquet", index=False)
    assign.to_parquet(out / "doc_clusters.parquet", index=False)
    (path("reports") / "clustering_eval.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    print(f"Saved {len(clusters)} clusters")


if __name__ == "__main__":
    main()
