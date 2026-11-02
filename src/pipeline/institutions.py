"""Phase 7b: institution profiles, institution similarity and the relationship network.

For every institution with at least MIN_DOCS documents:
  - documents by domain and source, top sub-areas, countries, years
  - a profile vector: the mean MiniLM embedding of its documents, L2-normalised
  - cosine similarity between all profiles, and its TOP_SIMILAR most similar institutions

Network: nodes are those institutions; an edge joins two institutions whose profile similarity
is at least EDGE_THRESHOLD (the 99th percentile of all pairwise similarities) or that share at
least MIN_SHARED_CLUSTERS sub-areas. Each node keeps at most MAX_EDGES strongest edges so the
graph stays readable.

Outputs: data/outputs/institutions.parquet, data/outputs/institution_similarity.parquet,
         data/outputs/institution_network.json, reports/institutions_summary.json

Usage:
    python -m src.pipeline.institutions
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd

from src.config import path

MIN_DOCS = 10
TOP_SIMILAR = 10
MIN_SHARED_CLUSTERS = 3
MAX_EDGES = 8


def profiles(docs: pd.DataFrame, vectors: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """One row per institution with >= MIN_DOCS documents, plus its normalised mean embedding."""
    rows, vecs = [], []
    for inst, group in docs.groupby("institution"):
        if len(group) < MIN_DOCS:
            continue
        v = vectors[group.index].mean(axis=0)
        vecs.append(v / np.linalg.norm(v))
        by_domain = group["domain"].value_counts()
        rows.append({
            "institution": inst,
            "name": group["institution_name"].iloc[0],
            "country": Counter(group["country"]).most_common(1)[0][0],
            "documents": len(group),
            "patents": int((group["source"] == "patent").sum()),
            "papers": int((group["source"] == "paper").sum()),
            "top_domain": by_domain.index[0],
            "by_domain": {d: int(n) for d, n in by_domain.items()},
            "by_year": {int(y): int(n) for y, n in group["year"].value_counts().sort_index().items()},
            "top_clusters": [c for c, _ in Counter(group["domain_cluster"]).most_common(5)],
            "clusters": sorted(set(group["domain_cluster"])),
        })
    return pd.DataFrame(rows), np.vstack(vecs).astype(np.float32)


def main() -> None:
    """Build profiles, the similarity table and the network, and save them."""
    docs = pd.read_parquet(path("documents")).merge(pd.read_parquet(path("outputs") / "doc_clusters.parquet"), on="doc_id")
    ids = json.loads((path("processed") / "embedding_ids.json").read_text(encoding="utf-8"))
    assert ids == docs["doc_id"].tolist()
    vectors = np.load(path("processed") / "embeddings_minilm.npy")

    inst, vecs = profiles(docs, vectors)
    sim = vecs @ vecs.T
    np.fill_diagonal(sim, -1.0)
    n = len(inst)
    upper = sim[np.triu_indices(n, 1)]
    edge_threshold = round(float(np.quantile(upper, 0.99)), 4)

    names = inst["institution"].to_numpy()
    top = []
    for i in range(n):
        for j in np.argsort(-sim[i])[:TOP_SIMILAR]:
            top.append({"institution": names[i], "similar_institution": names[j], "similarity": round(float(sim[i, j]), 4)})
    similar = pd.DataFrame(top)
    inst["similar"] = [similar.loc[similar["institution"] == x, "similar_institution"].head(5).tolist() for x in names]

    cluster_sets = [set(c) for c in inst["clusters"]]
    candidates = []
    for i in range(n):
        for j in range(i + 1, n):
            shared = len(cluster_sets[i] & cluster_sets[j])
            if sim[i, j] >= edge_threshold or shared >= MIN_SHARED_CLUSTERS:
                candidates.append((i, j, float(sim[i, j]), shared))
    # keep each node's strongest edges (by similarity, then shared clusters)
    candidates.sort(key=lambda e: (e[2], e[3]), reverse=True)
    degree: Counter = Counter()
    edges = []
    for i, j, s, shared in candidates:
        if degree[i] < MAX_EDGES and degree[j] < MAX_EDGES:
            edges.append({"source": names[i], "target": names[j], "similarity": round(s, 4), "shared_clusters": shared})
            degree[i] += 1
            degree[j] += 1

    nodes = [{"id": r["institution"], "name": r["name"], "country": r["country"], "documents": int(r["documents"]),
              "top_domain": r["top_domain"], "by_domain": r["by_domain"]} for _, r in inst.iterrows()]
    out = path("outputs")
    # dict columns are stored as JSON strings (parquet maps need string keys)
    inst.drop(columns=["clusters"]).assign(
        by_domain=inst["by_domain"].map(json.dumps), by_year=inst["by_year"].map(json.dumps)
    ).to_parquet(out / "institutions.parquet", index=False)
    similar.to_parquet(out / "institution_similarity.parquet", index=False)
    (out / "institution_network.json").write_text(
        json.dumps({"nodes": nodes, "edges": edges, "edge_threshold": edge_threshold,
                    "rule": f"similarity >= {edge_threshold} (99th percentile) or >= {MIN_SHARED_CLUSTERS} shared sub-areas; "
                            f"max {MAX_EDGES} edges per node"}),
        encoding="utf-8")
    summary = {"institutions_profiled": n, "min_docs": MIN_DOCS, "edges": len(edges), "edge_threshold": edge_threshold,
               "isolated_nodes": int(n - len({e['source'] for e in edges} | {e['target'] for e in edges})),
               "most_connected": [names[i] for i, _ in degree.most_common(5)],
               "by_country": inst["country"].value_counts().head(10).to_dict()}
    (path("reports") / "institutions_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
