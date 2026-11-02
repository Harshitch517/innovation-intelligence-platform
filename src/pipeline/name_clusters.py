"""Phase 11: readable names and summaries for clusters with the Gemini API (offline, cached).

1. Every cluster: its keywords and sample titles are sent to Gemini, which returns a short name
   (at most 6 words) and a 2-sentence summary.
2. The top emerging sub-areas: Gemini writes a one-paragraph summary using the cluster's yearly
   counts, growth and sample titles.

Answers are cached in data/outputs/cluster_names.json, so the script is resumable and a rerun
only asks for what is missing. If Gemini is unavailable, a cluster keeps no name and the
platform falls back to its keywords. With --update-db the names are written to the database.

Usage:
    python -m src.pipeline.name_clusters               # name everything, then update the DB
    python -m src.pipeline.name_clusters --no-db       # cache only
"""
from __future__ import annotations

import argparse
import json
import time

import pandas as pd
import requests

from src.config import get_key, load_config, path

URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
TOP_EMERGING = 10
PAUSE_SECONDS = 1.0


def ask_json(prompt: str, model: str, key: str, retries: int = 5) -> dict | None:
    """Send a prompt that must be answered in JSON; return the parsed object or None."""
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 0.2}}
    for attempt in range(retries):
        try:
            r = requests.post(URL.format(model=model), json=body, headers={"x-goog-api-key": key}, timeout=60)
            if r.ok:
                text = r.json()["candidates"][0]["content"]["parts"][-1]["text"]
                return json.loads(text)
            if r.status_code not in (429, 500, 502, 503, 504):
                print(f"  Gemini error {r.status_code}: {r.text[:120]}")
                return None
        except (requests.RequestException, KeyError, IndexError, json.JSONDecodeError):
            pass
        time.sleep(5 * (attempt + 1))
    return None


def name_prompt(row: pd.Series) -> str:
    """Prompt for a cluster name and summary."""
    scope = f"a sub-area of {row['domain']}" if row["domain"] else "a cross-domain cluster"
    titles = "\n".join(f"- {t}" for t in list(row["sample_titles"])[:5])
    return (
        f"These university patents and research papers form {scope}.\n"
        f"Top keywords (class-based TF-IDF): {', '.join(list(row['keywords'])[:10])}\n"
        f"Sample titles:\n{titles}\n\n"
        "Give the technology area a precise, readable name of at most 6 words (Title Case, no quotes) "
        "and a 2-sentence summary of what the documents work on. Do not invent facts.\n"
        'Reply as JSON: {"name": "...", "summary": "..."}'
    )


def emerging_prompt(row: pd.Series, counts: dict[int, int], name: str) -> str:
    """Prompt for a one-paragraph emerging-area summary."""
    titles = "\n".join(f"- {t}" for t in list(row["sample_titles"])[:5])
    series = ", ".join(f"{y}: {n}" for y, n in sorted(counts.items()))
    return (
        f"Emerging technology area in {row['domain']}: {name}.\n"
        f"Documents per year in our sample of university patents and papers: {series}.\n"
        f"Growth 2024-25 vs 2022-23: {row['recent_growth']:+.0%}.\n"
        f"Keywords: {', '.join(list(row['keywords'])[:10])}\nSample titles:\n{titles}\n\n"
        "Write one paragraph (4-5 sentences) for a research dashboard: what this area is, what the "
        "documents focus on, and how fast it is growing in this sample. Do not invent facts or numbers.\n"
        'Reply as JSON: {"paragraph": "..."}'
    )


def main() -> None:
    """Fill the cache with names, summaries and emerging paragraphs, then update the database."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-db", action="store_true", help="do not write the names to the database")
    args = parser.parse_args()

    cfg = load_config()
    key, model = get_key("GEMINI_API_KEY"), cfg["apis"]["gemini_model"]
    if not key:
        raise SystemExit("GEMINI_API_KEY is missing from .env; clusters keep their keyword fallback")
    clusters = pd.read_parquet(path("outputs") / "clusters.parquet")
    emerging = pd.read_parquet(path("outputs") / "emerging.parquet").head(TOP_EMERGING).merge(
        clusters[["cluster_id", "sample_titles"]], on="cluster_id")
    trends = pd.read_parquet(path("outputs") / "trends.parquet")
    cache_file = path("outputs") / "cluster_names.json"
    cache = json.loads(cache_file.read_text(encoding="utf-8")) if cache_file.exists() else {}

    todo = clusters[~clusters["cluster_id"].isin([k for k, v in cache.items() if v.get("name")])]
    print(f"Naming {len(todo)} of {len(clusters)} clusters with {model} ...")
    for i, (_, row) in enumerate(todo.iterrows(), 1):
        ans = ask_json(name_prompt(row), model, key)
        if ans and ans.get("name"):
            cache.setdefault(row["cluster_id"], {}).update(
                {"name": ans["name"].strip().strip('"')[:80], "summary": (ans.get("summary") or "").strip(), "model": model})
        if i % 20 == 0 or i == len(todo):
            cache_file.write_text(json.dumps(cache, indent=1), encoding="utf-8")
            print(f"  {i}/{len(todo)}")
        time.sleep(PAUSE_SECONDS)

    for _, row in emerging.iterrows():
        cid = row["cluster_id"]
        if cache.get(cid, {}).get("emerging_summary"):
            continue
        counts = trends[trends["cluster_id"] == cid].set_index("year")["count"].to_dict()
        name = cache.get(cid, {}).get("name") or ", ".join(list(row["keywords"])[:4])
        ans = ask_json(emerging_prompt(row, counts, name), model, key)
        if ans and ans.get("paragraph"):
            cache.setdefault(cid, {})["emerging_summary"] = ans["paragraph"].strip()
        time.sleep(PAUSE_SECONDS)
    cache_file.write_text(json.dumps(cache, indent=1), encoding="utf-8")

    named = sum(1 for c in clusters["cluster_id"] if cache.get(c, {}).get("name"))
    para = sum(1 for c in emerging["cluster_id"] if cache.get(c, {}).get("emerging_summary"))
    print(f"Named {named}/{len(clusters)} clusters; {para}/{len(emerging)} emerging paragraphs")
    if not args.no_db:
        from src.pipeline.load_db import apply_cluster_names, connect

        with connect() as conn:
            print(f"Updated {apply_cluster_names(conn)} clusters in the database")


if __name__ == "__main__":
    main()
