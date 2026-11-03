"""India supplement: extra research papers from Indian universities and institutes, including
every matching Lovely Professional University (LPU) paper.

These documents are kept apart from the 30,000-document evaluation set, so the classifier's test
results and the trend back-test are unchanged. They are added to search, compare, clusters and
institution profiles:

1. fetch  - OpenAlex, same filters as fetch_papers (primary topic in a domain, 2020-2025, article,
            abstract): every LPU paper, plus a seeded random sample of papers with an Indian
            university or institute author (PER_DOMAIN per domain). Pages saved to data/raw/india/.
2. build  - clean with the Phase 2 rules (abstract rebuild, >= 30 words, English only, duplicate
            titles also against the main dataset). Each paper is credited to its LPU affiliation,
            else its first Indian university/institute, else its first university/institute.
3. embed  - MiniLM vectors (same model as the main dataset).
4. assign - each document joins the nearest existing sub-area of its domain and the nearest global
            cluster (by centroid), so the Gemini cluster names stay valid and no LLM is called.

Outputs: data/processed/documents_extra.parquet, data/processed/embeddings_minilm_extra.npy,
         data/outputs/doc_clusters_extra.parquet, reports/india_supplement.json

Indian patents are not included: Google Patents has only about 90 Indian university
publications for 2020-2025 (see PLAN.md Section 10).

Usage:
    python -m src.pipeline.supplement_india
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

LPU_ID = "I110360157"
PER_DOMAIN = 300


def fetch(client: OpenAlexClient, cfg: dict) -> None:
    """Save every LPU page and an Indian sample per domain to data/raw/india/ (resumable)."""
    out = path("raw_papers").parent / "india"
    out.mkdir(parents=True, exist_ok=True)
    y0, y1, seed = cfg["years"]["start"], cfg["years"]["end"], cfg["seed"]
    for domain, spec in cfg["domains"].items():
        topics = spec["openalex_topics"]
        chunks = [topics[i:i + MAX_TOPICS_PER_QUERY] for i in range(0, len(topics), MAX_TOPICS_PER_QUERY)]
        for i, chunk in enumerate(chunks):
            base = build_filter(chunk, y0, y1)
            # every LPU paper (cursor paging)
            cursor, page = "*", 1
            while cursor:
                file = out / f"lpu_{slug(domain)}_c{i}_p{page:02d}.json"
                if file.exists():
                    body = json.loads(file.read_text(encoding="utf-8"))
                else:
                    body = client.works(filter=f"{base},institutions.id:{LPU_ID}", per_page=200, cursor=cursor, select=SELECT_FIELDS)
                    file.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
                cursor = body["meta"].get("next_cursor") if body["results"] else None
                page += 1
            # random sample of Indian papers, shared across chunks by size
        sizes = [client.works(filter=f"{build_filter(c, y0, y1)},institutions.country_code:IN", per_page=1, select="id")["meta"]["count"]
                 for c in chunks]
        for i, (chunk, size) in enumerate(zip(chunks, sizes)):
            wanted = min(size, math.ceil(PER_DOMAIN * 1.3 * size / max(1, sum(sizes))))
            for page in range(1, math.ceil(wanted / 100) + 1):
                file = out / f"in_{slug(domain)}_c{i}_seed{seed}_n{wanted}_p{page:02d}.json"
                if not file.exists():
                    body = client.works(filter=f"{build_filter(chunk, y0, y1)},institutions.country_code:IN",
                                        sample=wanted, seed=seed, per_page=100, page=page, select=SELECT_FIELDS)
                    file.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
        print(f"[{domain}] fetched (Indian papers available: {sum(sizes):,})")


def indian_affiliation(work: dict) -> tuple[str, str] | None:
    """LPU if any author is affiliated with it, else the first Indian university/institute."""
    first_in = None
    for a in work.get("authorships", []):
        for inst in a.get("institutions", []):
            if inst.get("id", "").endswith(LPU_ID):
                return inst["display_name"], "IN"
            if first_in is None and inst.get("country_code") == "IN" and inst.get("type") in ("education", "facility"):
                first_in = (inst["display_name"], "IN")
    return first_in


def build(cfg: dict) -> pd.DataFrame:
    """Cleaned supplement documents in the unified schema (duplicates of the main set removed)."""
    topic_domain = topic_to_domain(cfg)
    rows: dict[str, dict] = {}
    for file in sorted((path("raw_papers").parent / "india").glob("*.json")):
        for work in json.loads(file.read_text(encoding="utf-8"))["results"]:
            rec = paper_record(work, topic_domain)
            if rec is None or rec["doc_id"] in rows:
                continue
            aff = indian_affiliation(work)
            if aff:
                rec["institution_raw"], rec["country"] = aff
            rec["lpu"] = file.name.startswith("lpu_") or (aff is not None and "Lovely" in aff[0])
            rows[rec["doc_id"]] = rec
    df = pd.DataFrame(list(rows.values()))
    log: dict = {"loaded": len(df)}

    main = pd.read_parquet(path("documents"))
    df = df[~df["doc_id"].isin(main["doc_id"])]
    df = df[~df["title"].map(normalise_title).isin(set(main["title"].map(normalise_title)))]
    log["dropped_already_in_main_dataset"] = log["loaded"] - len(df)
    df = filter_and_dedupe(df, cfg["cleaning"]["min_abstract_words"], log)
    df["institution"] = df["institution_raw"].map(lambda n: normalise_institution(n, cfg))
    df = df[df["institution"] != ""]

    # Display names: reuse the main dataset's name for institutions it already has.
    names = dict(zip(main["institution"], main["institution_name"]))
    own = display_names(df, cfg)
    df["institution_name"] = df["institution"].map(lambda k: names.get(k) or own[k])
    df["text"] = df["title"] + ". " + df["abstract"]
    df["split"] = "extra"
    cols = ["doc_id", "source", "title", "abstract", "text", "institution", "institution_name", "institution_raw",
            "country", "year", "domain", "raw_codes", "cited_ids", "lpu", "split"]
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
    out = []
    for i, dom in enumerate(df["domain"]):
        sims_d = d_c @ vec[i]
        allowed = np.array([d_dom[c] == dom for c in d_ids])
        out.append({"doc_id": df["doc_id"].iat[i], "global_cluster": g_ids[int(np.argmax(g_c @ vec[i]))],
                    "global_outlier": False, "domain_cluster": d_ids[int(np.argmax(np.where(allowed, sims_d, -9)))],
                    "domain_outlier": False})
    return pd.DataFrame(out)


def main() -> None:
    """Fetch, clean, embed and assign the India supplement, then write a summary."""
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
    lpu = df[df["lpu"]]
    log |= {"by_domain": df["domain"].value_counts().sort_index().to_dict(),
            "lpu_documents": len(lpu), "lpu_by_domain": lpu["domain"].value_counts().sort_index().to_dict(),
            "institutions": int(df["institution"].nunique()),
            "top_institutions": df["institution_name"].value_counts().head(10).to_dict()}
    (path("reports") / "india_supplement.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
