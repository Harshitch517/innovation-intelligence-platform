"""Phase 1: fetch research papers from OpenAlex, one domain at a time.

For each domain we take a reproducible random sample (seed from config.yaml) of journal
articles whose *primary topic* is one of the domain's OpenAlex topics, published in the
configured years, with an abstract and at least one university / research-institute author.
Sampling across all years (instead of a fixed quota per year) keeps the real year-to-year
growth, which Phase 7 needs for trend detection.

Every API page is saved to data/raw/papers/<domain>/ as JSON. Pages already on disk are
skipped, so the script is resumable and a rerun fetches nothing new.

Usage:
    python -m src.pipeline.fetch_papers                    # all domains
    python -m src.pipeline.fetch_papers --domain Robotics --max 500
    python -m src.pipeline.fetch_papers --years 2020-2025
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import requests
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from src.config import get_key, load_config, path

# OpenAlex limits OR-filters on primary_topic.id to 10 values per request.
MAX_TOPICS_PER_QUERY = 10
# Extra records fetched on top of the target, to cover what cleaning drops later.
FETCH_BUFFER = 1.25
SELECT_FIELDS = (
    "id,doi,title,publication_year,type,abstract_inverted_index,"
    "primary_topic,topics,authorships,referenced_works"
)


def slug(name: str) -> str:
    """Folder-friendly version of a domain name ("Renewable Energy" -> "renewable_energy")."""
    return name.lower().replace(" ", "_")


def _retryable(exc: BaseException) -> bool:
    """Retry on network errors, rate limiting (429) and server errors (5xx)."""
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return isinstance(exc, requests.RequestException)


class OpenAlexClient:
    """Minimal OpenAlex /works client with polite pacing and retries."""

    def __init__(self, base_url: str, api_key: str | None, pause: float = 0.2) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.pause = pause
        self.session = requests.Session()

    @retry(
        retry=retry_if_exception(_retryable),
        wait=wait_exponential(multiplier=2, min=2, max=120),
        stop=stop_after_attempt(8),
        reraise=True,
    )
    def works(self, **params: str | int) -> dict:
        """GET /works with the given query parameters and return the JSON body."""
        if self.api_key:
            params["api_key"] = self.api_key
        r = self.session.get(f"{self.base_url}/works", params=params, timeout=90)
        r.raise_for_status()
        time.sleep(self.pause)
        return r.json()


def build_filter(topics: list[str], year_from: int, year_to: int) -> str:
    """OpenAlex filter string for one chunk of topics."""
    return ",".join(
        [
            "primary_topic.id:" + "|".join(topics),
            f"publication_year:{year_from}-{year_to}",
            "has_abstract:true",
            "type:article",
            "institutions.type:education|facility",
        ]
    )


def fetch_domain(
    client: OpenAlexClient,
    domain: str,
    topics: list[str],
    target: int,
    year_from: int,
    year_to: int,
    seed: int,
    per_page: int,
    out_dir: Path,
) -> Counter:
    """Fetch about `target` sampled papers for one domain; return counts per year."""
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks = [topics[i : i + MAX_TOPICS_PER_QUERY] for i in range(0, len(topics), MAX_TOPICS_PER_QUERY)]

    # Share the target between chunks in proportion to how many papers each chunk has.
    sizes = []
    for chunk in chunks:
        meta = client.works(filter=build_filter(chunk, year_from, year_to), per_page=1, select="id")["meta"]
        sizes.append(meta["count"])
    total = sum(sizes)
    print(f"[{domain}] {total:,} matching papers in OpenAlex across {len(chunks)} topic chunk(s)")

    years: Counter = Counter()
    wanted_total = math.ceil(target * FETCH_BUFFER)
    for i, (chunk, size) in enumerate(zip(chunks, sizes)):
        wanted = min(size, math.ceil(wanted_total * size / total)) if total else 0
        pages = math.ceil(wanted / per_page)
        for page in range(1, pages + 1):
            file = out_dir / f"chunk{i}_seed{seed}_n{wanted}_page{page:03d}.json"
            if file.exists():
                results = json.loads(file.read_text(encoding="utf-8"))["results"]
            else:
                body = client.works(
                    filter=build_filter(chunk, year_from, year_to),
                    sample=wanted,
                    seed=seed,
                    per_page=per_page,
                    page=page,
                    select=SELECT_FIELDS,
                )
                results = body["results"]
                file.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
                print(f"  saved {file.name} ({len(results)} records)")
            years.update(w["publication_year"] for w in results)
    return years


def main() -> None:
    """Parse arguments, fetch every requested domain and write a count log to reports/."""
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--domain", choices=list(cfg["domains"]), help="fetch one domain only")
    parser.add_argument("--years", default=f"{cfg['years']['start']}-{cfg['years']['end']}", help="e.g. 2020-2025")
    parser.add_argument("--max", type=int, default=cfg["target_per_source"], help="target papers per domain")
    args = parser.parse_args()

    year_from, year_to = (int(y) for y in args.years.split("-"))
    api = cfg["apis"]
    client = OpenAlexClient(api["openalex_url"], get_key("OPENALEX_API_KEY"))
    domains = [args.domain] if args.domain else list(cfg["domains"])

    log: dict[str, dict] = {}
    for domain in domains:
        years = fetch_domain(
            client,
            domain,
            cfg["domains"][domain]["openalex_topics"],
            args.max,
            year_from,
            year_to,
            cfg["seed"],
            api["openalex_per_page"],
            path("raw_papers") / slug(domain),
        )
        log[domain] = {"total": sum(years.values()), "by_year": dict(sorted(years.items()))}
        print(f"[{domain}] {log[domain]['total']} papers on disk: {log[domain]['by_year']}")

    report = path("reports") / "fetch_papers_counts.json"
    existing = json.loads(report.read_text(encoding="utf-8")) if report.exists() else {}
    existing.update(log)
    report.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"Counts written to {report}")


if __name__ == "__main__":
    main()
