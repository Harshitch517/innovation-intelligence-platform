"""Supplement: extra research papers drawn at random from OpenAlex worldwide.

No country or institution filter is applied: the papers are a seeded random sample with the same
filters as fetch_papers (primary topic in a domain, 2020-2025, article, abstract, at least one
university/institute author), using a different seed so the draw is independent of the main set.

These documents are kept apart from the 30,000-document evaluation set, so the classifier's test
results and the trend back-test are unchanged. They are added to search, compare and clusters:

1. fetch  - OpenAlex random sample, FETCH_FACTOR x PER_DOMAIN per domain, shared across topic
            chunks in proportion to their size. Pages saved to data/raw/supplement/ (resumable).
2. build  - clean with the Phase 2 rules (abstract rebuild, >= 30 words, English only, duplicate
            titles also against the main dataset), then keep a seeded random PER_DOMAIN per domain.
3. embed  - MiniLM vectors (same model as the main dataset).
4. assign - each document joins the nearest existing sub-area of its domain and the nearest global
            cluster (by centroid), so the Gemini cluster names stay valid and no LLM is called.

Outputs: data/processed/documents_extra.parquet, data/processed/embeddings_minilm_extra.npy,
         data/outputs/doc_clusters_extra.parquet, reports/supplement.json

Usage:
    python -m src.pipeline.supplement
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from src.config import get_key, load_config, path
from src.pipeline.clean import (display_names, filter_and_dedupe, normalise_institution, normalise_title,
                                paper_record, topic_to_domain)
from src.pipeline.fetch_papers import MAX_TOPICS_PER_QUERY, SELECT_FIELDS, OpenAlexClient, build_filter, slug

PER_DOMAIN = 240          # final papers per domain (5 x 240 = 1,200)
FETCH_FACTOR = 1.5        # fetched on top, to cover cleaning and overlap with the main set
SEED_OFFSET = 1           # seed = config seed + 1, independent of the main sample


def raw_dir():
    return path("raw_papers").parent / "supplement"


def fetch(client: OpenAlexClient, cfg: dict) -> None:
    """Save a seeded random sample per domain to data/raw/supplement/ (resumable)."""
    out = raw_dir()
    out.mkdir(parents=True, exist_ok=True)
    y0, y1, seed = cfg["years"]["start"], cfg["years"]["end"], cfg["seed"] + SEED_OFFSET
    for domain, spec in cfg["domains"].items():
        topics = spec["openalex_topics"]
        chunks = [topics[i:i + MAX_TOPICS_PER_QUERY] for i in range(0, len(topics), MAX_TOPICS_PER_QUERY)]
        sizes = [client.works(filter=build_filter(c, y0, y1), per_page=1, select="id")["meta"]["count"] for c in chunks]
        for i, (chunk, size) in enumerate(zip(chunks, sizes)):
            wanted = min(size, math.ceil(PER_DOMAIN * FETCH_FACTOR * size / max(1, sum(sizes))))
            for page in range(1, math.ceil(wanted / 100) + 1):
                file = out / f"{slug(domain)}_c{i}_seed{seed}_n{wanted}_p{page:02d}.json"
                if not file.exists():
                    body = client.works(filter=build_filter(chunk, y0, y1), sample=wanted, seed=seed,
                                        per_page=100, page=page, select=SELECT_FIELDS)
                    file.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
        print(f"[{domain}] fetched (papers available: {sum(sizes):,})")


def build(cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Cleaned supplement documents in the unified schema (duplicates of the main set removed)."""
    topic_domain = topic_to_domain(cfg)
    rows: dict[str, dict] = {}
    for file in sorted(raw_dir().glob("*.json")):
        for work in json.loads(file.read_text(encoding="utf-8"))["results"]:
            rec = paper_record(work, topic_domain)
            if rec is not None:
                rows.setdefault(rec["doc_id"], rec)
    df = pd.DataFrame(list(rows.values()))
    log: dict = {"loaded": len(df)}

    main = pd.read_parquet(path("documents"))
    df = df[~df["doc_id"].isin(main["doc_id"])]
    df = df[~df["title"].map(normalise_title).isin(set(main["title"].map(normalise_title)))]
    log["dropped_already_in_main_dataset"] = log["loaded"] - len(df)
    df = filter_and_dedupe(df, cfg["cleaning"]["min_abstract_words"], log)
    df["institution"] = df["institution_raw"].map(lambda n: normalise_institution(n, cfg))
    df = df[df["institution"] != ""]

    # Equal random share per domain.
    seed = cfg["seed"] + SEED_OFFSET
    df = df.sort_values("doc_id").sample(frac=1, random_state=seed).groupby("domain").head(PER_DOMAIN)

    # Display names: reuse the main dataset's name for institutions it already has.
    names = dict(zip(main["institution"], main["institution_name"]))
    own = display_names(df, cfg)
    df["institution_name"] = df["institution"].map(lambda k: names.get(k) or own[k])
    df["text"] = df["title"] + ". " + df["abstract"]
    df["split"] = "extra"
    cols = ["doc_id", "source", "title", "abstract", "text", "institution", "institution_name", "institution_raw",
            "country", "year", "domain", "raw_codes", "cited_ids", "split"]
    df = df[cols].sort_values("doc_id").reset_index(drop=True)
    log["final"] = len(df)
    return df, log


def assign_clusters(df: pd.DataFrame, vec: np.ndarray) -> pd.DataFrame:
    """Nearest sub-area of the document's domain and nearest global cluster, by centroid."""
    main = pd.read_parquet(path("documents"))[["doc_id", "domain"]].merge(pd.read_parquet(path("outputs") / "doc_clusters.parquet"), on="doc_id")
    mvec = np.load(path("processed") / "embeddings_minilm.npy")

    def centroids(col: str) -> tuple[list[str], np.ndarray]:
        ids, cents = [], []
        for cid, idx in main.groupby(col).indices.items():
            c = mvec[idx].mean(axis=0)
            ids.append(cid)
            cents.append(c / np.linalg.norm(c))
        return ids, np.vstack(cents)

    g_ids, g_c = centroids("global_cluster")
    d_ids, d_c = centroids("domain_cluster")
    d_dom = main.drop_duplicates("domain_cluster").set_index("domain_cluster")["domain"]
    allowed_by_domain = {dom: np.array([d_dom[c] == dom for c in d_ids]) for dom in df["domain"].unique()}
    out = []
    for i, dom in enumerate(df["domain"]):
        sims_d = np.where(allowed_by_domain[dom], d_c @ vec[i], -9)
        out.append({"doc_id": df["doc_id"].iat[i], "global_cluster": g_ids[int(np.argmax(g_c @ vec[i]))],
                    "global_outlier": False, "domain_cluster": d_ids[int(np.argmax(sims_d))],
                    "domain_outlier": False})
    return pd.DataFrame(out)


def main() -> None:
    """Fetch, clean, embed and assign the supplement, then write a summary."""
    cfg = load_config()
    fetch(OpenAlexClient(cfg["apis"]["openalex_url"], get_key("OPENALEX_API_KEY")), cfg)
    df, log = build(cfg)

    from sentence_transformers import SentenceTransformer
    import torch

    model = SentenceTransformer(cfg["models"]["embedding"], device="cuda" if torch.cuda.is_available() else "cpu")
    vec = model.encode(df["text"].tolist(), batch_size=128, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    clusters = assign_clusters(df, vec)

    df.to_parquet(path("processed") / "documents_extra.parquet", index=False)
    np.save(path("processed") / "embeddings_minilm_extra.npy", vec)
    clusters.to_parquet(path("outputs") / "doc_clusters_extra.parquet", index=False)
    log |= {"seed": cfg["seed"] + SEED_OFFSET,
            "by_domain": df["domain"].value_counts().sort_index().to_dict(),
            "by_country": df["country"].replace("", "unknown").value_counts().head(15).to_dict(),
            "countries": int(df["country"].replace("", np.nan).nunique()),
            "institutions": int(df["institution"].nunique())}
    (path("reports") / "supplement.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
