"""Phase 1: fetch university / research-institute patents from Google Patents on BigQuery.

Source: `patents-public-data.patents.publications` (Google Patents Public Datasets), with the
English title and abstract from `patents-public-data.google_patents_research.publications`,
queried from the user's free BigQuery sandbox project (GCP_PROJECT_ID in .env, credentials
from `gcloud auth application-default login`).

Steps
1. ONE filtered query over the public tables (granted 2020-2025, US/IN, English title and
   abstract, primary CPC in one of our domains, a university/institute assignee, one
   publication per patent family). Its result is saved to a small staging table in the
   user's own dataset, so the expensive scan runs only once.
2. Per domain, a cheap query on the staging table draws a reproducible random sample
   (seed from config.yaml) and saves it to data/raw/patents/<domain>.parquet.

Quota safety: every query is dry-run first and the estimated bytes are printed. A query
whose estimate exceeds bigquery.max_bytes_billed_gb (200 GB) is not run; the script stops
so the user can decide. The same limit is set as maximum_bytes_billed on the real job.

Resumable: an existing staging table and existing domain files are reused, never re-queried
(use --refresh-staging to rebuild the staging table on purpose).

Usage:
    python -m src.pipeline.fetch_patents --dry-run        # only print the size estimate
    python -m src.pipeline.fetch_patents                  # all domains
    python -m src.pipeline.fetch_patents --domain Robotics --max 500
"""
from __future__ import annotations

import argparse
import json
import math
import sys

import pandas as pd
from google.api_core.exceptions import NotFound
from google.cloud import bigquery

from src.config import get_key, load_config, path

# Extra records fetched on top of the target, to cover what cleaning drops later
# (same buffer as fetch_papers.py).
FETCH_BUFFER = 1.25
GB = 1024**3


def slug(name: str) -> str:
    """File-friendly version of a domain name ("Renewable Energy" -> "renewable_energy")."""
    return name.lower().replace(" ", "_")


def cpc_condition(prefix: str, column: str = "primary_cpc") -> str:
    """SQL test that a CPC code column falls under a CPC prefix from config.yaml.

    BigQuery CPC codes look like "G06N3/08" or "G05B19/418". A subclass prefix ("G06N")
    matches by plain prefix; a main-group prefix ("G05B19") must be followed by "/" so that
    "H04L9" does not also match a different group such as "H04L93".
    """
    if len(prefix) > 4:
        return f"STARTS_WITH({column}, '{prefix}/')"
    return f"STARTS_WITH({column}, '{prefix}')"


def domain_case(domains: dict, column: str = "primary_cpc") -> str:
    """CASE expression that maps a CPC code column to a domain name (or NULL)."""
    whens = []
    for name, spec in domains.items():
        test = " OR ".join(cpc_condition(p, column) for p in spec["cpc_prefixes"])
        whens.append(f"WHEN {test} THEN '{name}'")
    return "CASE " + " ".join(whens) + " END"


def staging_sql(cfg: dict) -> str:
    """The one filtered query over the public patent tables.

    Only the needed columns are read (never SELECT *), which is what BigQuery bills for.
    Title and abstract come from `google_patents_research.publications` (one English text per
    patent, 110 GB for the abstract column) instead of `abstract_localized` in the main table
    (every language, 199 GB), which keeps the scan under the 200 GB cap. Machine-translated
    texts are excluded, so only patents originally written in English remain.

    Domain label, in order (the rule used is kept in `label_rule`):
      1. "primary":   the primary CPC code (flagged `first`) matches a domain prefix;
      2. "inventive": otherwise the first *inventive* CPC code that matches a domain prefix;
      3. "y02e_tag":  otherwise Renewable Energy if any code is a Y02E tag. Y-section codes are
                      supplementary tags and are never the primary code, so rule 1 alone would
                      find almost no renewable-energy patents.
    """
    start, end = cfg["years"]["start"], cfg["years"]["end"]
    return f"""
WITH base AS (
  SELECT
    p.publication_number,
    p.family_id,
    p.country_code,
    p.publication_date,
    p.grant_date,
    COALESCE(
      (SELECT c.code FROM UNNEST(p.cpc) c WHERE c.first LIMIT 1),
      p.cpc[SAFE_OFFSET(0)].code
    ) AS primary_cpc,
    ARRAY(SELECT c.code FROM UNNEST(p.cpc) c) AS cpc_codes,
    (SELECT {domain_case(cfg['domains'], 'c.code')} FROM UNNEST(p.cpc) c WITH OFFSET o
       WHERE c.inventive AND {domain_case(cfg['domains'], 'c.code')} IS NOT NULL
       ORDER BY o LIMIT 1) AS inventive_domain,
    EXISTS(SELECT 1 FROM UNNEST(p.cpc) c WHERE STARTS_WITH(c.code, 'Y02E')) AS has_y02e,
    -- first university/institute assignee; company names ("SAS INST INC") are skipped
    -- unless they also say UNIV or COLLEGE ("UNIV X RES FOUND", "X UNIV IND COOP CO")
    (SELECT AS STRUCT a.name, a.country_code FROM UNNEST(p.assignee_harmonized) a
       WHERE REGEXP_CONTAINS(UPPER(a.name), @assignee_pattern)
         AND NOT (REGEXP_CONTAINS(UPPER(a.name), @company_pattern)
                  AND NOT REGEXP_CONTAINS(UPPER(a.name), r'\\b(UNIV|UNIVERSITY|COLLEGE)\\b'))
       LIMIT 1) AS institution,
    ARRAY(SELECT c.publication_number FROM UNNEST(p.citation) c
            WHERE c.publication_number IS NOT NULL AND c.publication_number != '') AS cited_ids
  FROM `{cfg['bigquery']['source_table']}` AS p
  WHERE p.grant_date BETWEEN {start}0101 AND {end}1231
    AND p.country_code IN UNNEST(@countries)
),
labelled AS (
  SELECT *,
    COALESCE(primary_domain, inventive_domain, IF(has_y02e, 'Renewable Energy', NULL)) AS domain,
    CASE WHEN primary_domain IS NOT NULL THEN 'primary'
         WHEN inventive_domain IS NOT NULL THEN 'inventive'
         WHEN has_y02e THEN 'y02e_tag' END AS label_rule
  FROM (SELECT *, {domain_case(cfg['domains'])} AS primary_domain FROM base)
  WHERE institution IS NOT NULL
)
SELECT
  l.publication_number, l.family_id, l.country_code, l.publication_date, l.grant_date,
  r.title, r.abstract, l.primary_cpc, l.cpc_codes,
  l.institution.name AS institution, l.institution.country_code AS institution_country,
  l.cited_ids, l.domain, l.label_rule
FROM labelled AS l
JOIN `{cfg['bigquery']['text_table']}` AS r USING (publication_number)
WHERE l.domain IS NOT NULL
  AND r.title != '' AND r.abstract != ''
  AND NOT r.title_translated AND NOT r.abstract_translated
-- one publication per patent family: prefer the US grant, then the earliest grant
QUALIFY ROW_NUMBER() OVER (
  PARTITION BY l.family_id
  ORDER BY IF(l.country_code = 'US', 0, 1), l.grant_date, l.publication_number
) = 1
"""


class SafeBigQuery:
    """BigQuery client wrapper: dry-run every query and refuse anything over the byte cap."""

    def __init__(self, project: str, location: str, max_gb: float) -> None:
        self.client = bigquery.Client(project=project, location=location)
        self.max_bytes = int(max_gb * GB)

    def estimate(self, sql: str, params: list) -> int:
        """Dry-run `sql` and return the bytes BigQuery says it would process."""
        cfg = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False, query_parameters=params)
        return self.client.query(sql, job_config=cfg).total_bytes_processed

    def run(self, sql: str, params: list, label: str, destination: str | None = None) -> bigquery.QueryJob:
        """Dry-run, check the cap, then run the query (optionally into a destination table)."""
        est = self.estimate(sql, params)
        print(f"[dry run] {label}: {est / GB:,.2f} GB estimated (cap {self.max_bytes / GB:,.0f} GB)")
        if est > self.max_bytes:
            print(f"STOP: estimate exceeds the {self.max_bytes / GB:,.0f} GB cap; not running. Ask the user.")
            sys.exit(2)
        cfg = bigquery.QueryJobConfig(query_parameters=params, maximum_bytes_billed=self.max_bytes)
        if destination:
            cfg.destination = destination
            cfg.write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE
        job = self.client.query(sql, job_config=cfg)
        job.result()
        print(f"[billed]  {label}: {(job.total_bytes_billed or 0) / GB:,.2f} GB")
        return job

    def table_rows(self, table_id: str) -> int | None:
        """Row count of a table, or None if it does not exist."""
        try:
            return self.client.get_table(table_id).num_rows
        except NotFound:
            return None


def to_unified(df: pd.DataFrame) -> pd.DataFrame:
    """Map BigQuery columns to the unified schema of PLAN.md Section 5 (plus patent extras)."""
    out = pd.DataFrame(
        {
            "doc_id": "PAT-" + df["publication_number"],
            "source": "patent",
            "title": df["title"].str.strip(),
            "abstract": df["abstract"].str.strip(),
            "institution": df["institution"],
            "country": df["institution_country"].where(df["institution_country"].fillna("") != "", df["country_code"]),
            "year": (df["grant_date"] // 10000).astype(int),
            "domain": df["domain"],
            # primary CPC first; raw_codes are kept for reference and are never a model input
            "raw_codes": [[p] + [c for c in codes if c != p] for p, codes in zip(df["primary_cpc"], df["cpc_codes"])],
            "cited_ids": [["PAT-" + c for c in cited] for cited in df["cited_ids"]],
        }
    )
    out["label_rule"] = df["label_rule"]
    out["publication_number"] = df["publication_number"]
    out["family_id"] = df["family_id"]
    out["patent_office"] = df["country_code"]
    out["grant_date"] = df["grant_date"]
    return out


def main() -> None:
    """Build (or reuse) the staging table, sample each domain and write counts to reports/."""
    cfg = load_config()
    bq_cfg = cfg["bigquery"]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--domain", choices=list(cfg["domains"]), help="sample one domain only")
    parser.add_argument("--max", type=int, default=cfg["target_per_source"], help="target patents per domain")
    parser.add_argument("--dry-run", action="store_true", help="print the staging query estimate and exit")
    parser.add_argument("--refresh-staging", action="store_true", help="rebuild the staging table")
    args = parser.parse_args()

    project = get_key("GCP_PROJECT_ID")
    if not project:
        sys.exit("GCP_PROJECT_ID is missing from .env")
    bq = SafeBigQuery(project, bq_cfg["location"], bq_cfg["max_bytes_billed_gb"])
    params = [
        bigquery.ScalarQueryParameter("assignee_pattern", "STRING", bq_cfg["assignee_pattern"]),
        bigquery.ArrayQueryParameter("countries", "STRING", bq_cfg["countries"]),
        bigquery.ScalarQueryParameter("company_pattern", "STRING", bq_cfg["company_pattern"]),
    ]
    sql = staging_sql(cfg)

    if args.dry_run:
        est = bq.estimate(sql, params)
        print(f"[dry run] staging query: {est / GB:,.2f} GB ({est:,} bytes); cap {bq_cfg['max_bytes_billed_gb']} GB")
        return

    # 1. Staging table: the one expensive query, run once.
    dataset_id = f"{project}.{bq_cfg['dataset']}"
    staging_id = f"{dataset_id}.{bq_cfg['staging_table']}"
    bq.client.create_dataset(bigquery.Dataset(dataset_id), exists_ok=True)
    rows = bq.table_rows(staging_id)
    if rows is None or args.refresh_staging:
        bq.run(sql, params, "staging query", destination=staging_id)
        rows = bq.table_rows(staging_id)
    else:
        print(f"Reusing staging table {staging_id}")
    print(f"Staging table has {rows:,} patents (one per family)")

    available = bq.client.query(
        f"SELECT domain, COUNT(*) AS n FROM `{staging_id}` GROUP BY domain ORDER BY domain"
    ).to_dataframe()
    print("Available per domain:", dict(zip(available["domain"], available["n"])))

    # 2. Per-domain reproducible random sample from the small staging table.
    out_dir = path("raw_patents")
    wanted = math.ceil(args.max * FETCH_BUFFER)
    domains = [args.domain] if args.domain else list(cfg["domains"])
    log: dict[str, dict] = {}
    for domain in domains:
        file = out_dir / f"{slug(domain)}.parquet"
        if file.exists():
            df = pd.read_parquet(file)
            print(f"[{domain}] reusing {file.name} ({len(df)} patents)")
        else:
            sample_sql = f"""
SELECT * FROM `{staging_id}`
WHERE domain = @domain
ORDER BY FARM_FINGERPRINT(CONCAT(publication_number, CAST(@seed AS STRING)))
LIMIT @n"""
            sample_params = [
                bigquery.ScalarQueryParameter("domain", "STRING", domain),
                bigquery.ScalarQueryParameter("seed", "INT64", cfg["seed"]),
                bigquery.ScalarQueryParameter("n", "INT64", wanted),
            ]
            job = bq.run(sample_sql, sample_params, f"sample {domain}")
            df = to_unified(job.to_dataframe())
            df.to_parquet(file, index=False)
            print(f"[{domain}] saved {file.name} ({len(df)} patents)")
        log[domain] = {
            "total": len(df),
            "available_in_staging": int(available.set_index("domain")["n"].get(domain, 0)),
            "by_office": df["patent_office"].value_counts().sort_index().to_dict(),
            "by_label_rule": df["label_rule"].value_counts().to_dict(),
            "by_country": df["country"].value_counts().head(10).to_dict(),
            "by_year": {int(k): int(v) for k, v in df["year"].value_counts().sort_index().items()},
        }
        print(f"[{domain}] by year {log[domain]['by_year']} | by office {log[domain]['by_office']}")

    report = path("reports") / "fetch_patents_counts.json"
    existing = json.loads(report.read_text(encoding="utf-8")) if report.exists() else {}
    existing.update(log)
    report.write_text(json.dumps(existing, indent=2, default=int), encoding="utf-8")
    print(f"Counts written to {report}")


if __name__ == "__main__":
    main()
