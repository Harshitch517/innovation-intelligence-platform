"""Phase 8: load every precomputed result into PostgreSQL + pgvector.

Runs backend/schema.sql (drops and recreates the tables), loads clusters, documents with their
MiniLM embeddings, trends, institutions, institution similarity and network edges, then builds
an HNSW index on the embeddings (cosine distance).

Check (PLAN.md Phase 8): for test documents, the database's top-10 nearest neighbours are
compared with the FAISS index from Phase 5.

The connection string comes from DATABASE_URL in .env (local Docker by default, Supabase later).

Usage:
    python -m src.pipeline.load_db            # load + check
    python -m src.pipeline.load_db --check    # check only
    python -m src.pipeline.load_db --trim     # online copy: abstracts trimmed to 1,500 characters
"""
from __future__ import annotations

import argparse
import json
import time

import faiss
import numpy as np
import pandas as pd
import psycopg
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb

from src.config import ROOT, get_key, load_config, path


def connect() -> psycopg.Connection:
    """Open a connection to DATABASE_URL with the pgvector type registered."""
    url = get_key("DATABASE_URL")
    if not url:
        raise SystemExit("DATABASE_URL is missing from .env")
    conn = psycopg.connect(url, autocommit=False)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def load(conn: psycopg.Connection, trim: bool) -> dict:
    """Create the schema and copy all tables in; return row counts."""
    out = path("outputs")
    docs = pd.read_parquet(path("documents"))
    ids = json.loads((path("processed") / "embedding_ids.json").read_text(encoding="utf-8"))
    assert ids == docs["doc_id"].tolist()
    vectors = np.load(path("processed") / "embeddings_minilm.npy")
    docs = docs.merge(pd.read_parquet(path("processed") / "splits.parquet"), on="doc_id")
    docs = docs.merge(pd.read_parquet(out / "doc_clusters.parquet"), on="doc_id")
    clusters = pd.read_parquet(out / "clusters.parquet")
    emerging = pd.read_parquet(out / "emerging.parquet").set_index("cluster_id")
    growth_all = pd.read_parquet(out / "trends.parquet")
    inst = pd.read_parquet(out / "institutions.parquet")
    sim = pd.read_parquet(out / "institution_similarity.parquet")
    network = json.loads((out / "institution_network.json").read_text(encoding="utf-8"))

    with conn.cursor() as cur:
        cur.execute((ROOT / "backend" / "schema.sql").read_text(encoding="utf-8"))
        with cur.copy("COPY clusters (cluster_id, scope, domain, keywords, size, sample_titles, recent_growth, cagr, emerging_rank) FROM STDIN") as cp:
            for r in clusters.itertuples():
                e = emerging.loc[r.cluster_id] if r.cluster_id in emerging.index else None
                cp.write_row((r.cluster_id, r.scope, r.domain, list(r.keywords), int(r.size), list(r.sample_titles),
                              None if e is None else float(e["recent_growth"]), None if e is None else float(e["cagr"]),
                              None if e is None else int(e["rank"])))
        with cur.copy("COPY documents (doc_id, source, title, abstract, institution, institution_name, country, year, domain, "
                      "raw_codes, split, global_cluster, cluster_id, embedding) FROM STDIN WITH (FORMAT BINARY)") as cp:
            cp.set_types(["text", "text", "text", "text", "text", "text", "text", "int4", "text", "text[]", "text", "text", "text", "vector"])
            for r, v in zip(docs.itertuples(), vectors):
                abstract = r.abstract[:1500] if trim else r.abstract
                cp.write_row((r.doc_id, r.source, r.title, abstract, r.institution, r.institution_name, r.country or None,
                              int(r.year), r.domain, list(r.raw_codes), r.split, r.global_cluster, r.domain_cluster, v))
        with cur.copy("COPY trends (cluster_id, year, count) FROM STDIN") as cp:
            for r in growth_all.itertuples():
                cp.write_row((r.cluster_id, int(r.year), int(r.count)))
        with cur.copy("COPY institutions (institution, name, country, documents, patents, papers, top_domain, by_domain, by_year, top_clusters, similar_institutions) FROM STDIN") as cp:
            for r in inst.itertuples():
                cp.write_row((r.institution, r.name, r.country or None, int(r.documents), int(r.patents), int(r.papers), r.top_domain,
                              Jsonb(json.loads(r.by_domain)), Jsonb(json.loads(r.by_year)), list(r.top_clusters), list(r.similar)))
        with cur.copy("COPY institution_similarity (institution, similar_institution, similarity) FROM STDIN") as cp:
            for r in sim.itertuples():
                cp.write_row((r.institution, r.similar_institution, float(r.similarity)))
        with cur.copy("COPY institution_edges (source, target, similarity, shared_clusters) FROM STDIN") as cp:
            for e in network["edges"]:
                cp.write_row((e["source"], e["target"], float(e["similarity"]), int(e["shared_clusters"])))
        start = time.perf_counter()
        cur.execute("CREATE INDEX documents_embedding_hnsw ON documents USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)")
        hnsw_seconds = round(time.perf_counter() - start, 1)
        cur.execute("ANALYZE")
    conn.commit()
    counts = {}
    for table in ("clusters", "documents", "trends", "institutions", "institution_similarity", "institution_edges"):
        counts[table] = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    counts["hnsw_build_seconds"] = hnsw_seconds
    return counts


def check(conn: psycopg.Connection, n: int = 100, seed: int = 42) -> dict:
    """Compare the database's top-10 neighbours with FAISS for `n` random test documents."""
    docs = pd.read_parquet(path("documents"))
    splits = pd.read_parquet(path("processed") / "splits.parquet")
    vectors = np.load(path("processed") / "embeddings_minilm.npy")
    index = faiss.read_index(str(path("models") / "faiss_minilm.index"))
    test_ids = splits.loc[splits["split"] == "test", "doc_id"].sample(n=n, random_state=seed)
    row = {d: i for i, d in enumerate(docs["doc_id"])}

    conn.execute("SET hnsw.ef_search = 200")
    overlap, exact, latencies = [], 0, []
    for doc_id in test_ids:
        q = vectors[row[doc_id]]
        _, faiss_rows = index.search(q[None, :], 11)
        faiss_top = [docs["doc_id"].iat[r] for r in faiss_rows[0] if docs["doc_id"].iat[r] != doc_id][:10]
        start = time.perf_counter()
        db_top = [r[0] for r in conn.execute(
            "SELECT doc_id FROM documents WHERE doc_id <> %s ORDER BY embedding <=> %s LIMIT 10", (doc_id, q)).fetchall()]
        latencies.append(time.perf_counter() - start)
        overlap.append(len(set(db_top) & set(faiss_top)) / 10)
        exact += db_top == faiss_top
    return {"test_documents": n, "mean_top10_overlap": round(float(np.mean(overlap)), 4),
            "identical_top10_share": round(exact / n, 4), "ef_search": 200,
            "median_query_ms": round(float(np.median(latencies)) * 1000, 1)}


def main() -> None:
    """Load the database (unless --check) and run the FAISS comparison."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="only run the FAISS comparison")
    parser.add_argument("--trim", action="store_true", help="trim abstracts to 1,500 characters (online copy)")
    args = parser.parse_args()
    load_config()
    with connect() as conn:
        report = {}
        if not args.check:
            report["rows"] = load(conn, args.trim)
            print("Loaded:", report["rows"])
        report["faiss_check"] = check(conn)
        print("FAISS check:", report["faiss_check"])
    (path("reports") / "db_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
