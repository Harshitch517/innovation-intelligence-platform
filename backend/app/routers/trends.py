"""/stats/overview, /trends, /clusters and /clusters/{cluster_id}."""
from __future__ import annotations

from collections import defaultdict

from fastapi import APIRouter, HTTPException, Query

from app.schemas import ClusterDetail, ClusterSummary, OverviewResponse, TrendsResponse
from app.services.db import fetch_all, fetch_one

router = APIRouter(tags=["trends"])

CLUSTER_COLS = "cluster_id, scope, domain, name, summary, keywords, size, recent_growth, cagr, emerging_rank"


def _counts(rows: list[dict]) -> list[dict]:
    return [{"key": str(r["key"]), "count": r["count"]} for r in rows]


@router.get("/stats/overview", response_model=OverviewResponse)
def overview() -> dict:
    """Counts by domain, source and year; top institutions."""
    totals = fetch_one(
        "SELECT count(*) AS documents, count(*) FILTER (WHERE source = 'patent') AS patents, "
        "count(*) FILTER (WHERE source = 'paper') AS papers, count(DISTINCT institution) AS institutions FROM documents")
    by = {col: _counts(fetch_all(f"SELECT {col} AS key, count(*) AS count FROM documents GROUP BY 1 ORDER BY 1"))
          for col in ("domain", "source", "year")}
    domain_year = fetch_all("SELECT domain, year, source, count(*) AS count FROM documents GROUP BY 1, 2, 3 ORDER BY 1, 2, 3")
    top = fetch_all("SELECT institution, min(institution_name) AS name, count(*) AS count FROM documents "
                    "GROUP BY institution ORDER BY count DESC LIMIT 15")
    return {**totals, "by_domain": by["domain"], "by_source": by["source"], "by_year": by["year"],
            "by_domain_year": domain_year, "top_institutions": top}


def _with_counts(clusters: list[dict]) -> list[dict]:
    """Attach {year: count} to each cluster."""
    ids = [c["cluster_id"] for c in clusters]
    counts: dict[str, dict[int, int]] = defaultdict(dict)
    for r in fetch_all("SELECT cluster_id, year, count FROM trends WHERE cluster_id = ANY(%s)", (ids,)):
        counts[r["cluster_id"]][r["year"]] = r["count"]
    for c in clusters:
        c["counts"] = dict(sorted(counts[c["cluster_id"]].items()))
    return clusters


@router.get("/trends", response_model=TrendsResponse)
def trends(domain: str | None = None, emerging_limit: int = Query(10, ge=1, le=50)) -> dict:
    """Yearly counts and growth per sub-area, and the emerging list (optionally for one domain)."""
    filt, params = ("AND domain = %s", [domain]) if domain else ("", [])
    clusters = fetch_all(f"SELECT {CLUSTER_COLS} FROM clusters WHERE scope = 'domain' {filt} ORDER BY size DESC", params)
    emerging = fetch_all(f"SELECT {CLUSTER_COLS} FROM clusters WHERE emerging_rank IS NOT NULL {filt} "
                         "ORDER BY emerging_rank LIMIT %s", params + [emerging_limit])
    dc = fetch_all(f"SELECT year, count(*) AS count FROM documents WHERE true {filt} GROUP BY year ORDER BY year", params)
    return {"domain": domain, "domain_counts": {r["year"]: r["count"] for r in dc},
            "clusters": _with_counts(clusters), "emerging": _with_counts(emerging)}


@router.get("/clusters", response_model=list[ClusterSummary])
def clusters(scope: str | None = Query(None, pattern="^(global|domain)$"), domain: str | None = None) -> list[dict]:
    """Cluster names, keywords and sizes."""
    where, params = ["true"], []
    if scope:
        where.append("scope = %s"); params.append(scope)
    if domain:
        where.append("domain = %s"); params.append(domain)
    return fetch_all(f"SELECT {CLUSTER_COLS} FROM clusters WHERE {' AND '.join(where)} ORDER BY scope, size DESC", params)


@router.get("/clusters/{cluster_id}", response_model=ClusterDetail)
def cluster(cluster_id: str, limit: int = Query(20, ge=1, le=200)) -> dict:
    """One cluster with yearly counts, top institutions and member documents."""
    c = fetch_one(f"SELECT {CLUSTER_COLS}, sample_titles FROM clusters WHERE cluster_id = %s", (cluster_id,))
    if c is None:
        raise HTTPException(404, "cluster not found")
    col = "global_cluster" if c["scope"] == "global" else "cluster_id"
    c["members"] = fetch_all(
        "SELECT doc_id, source, title, institution, institution_name, country, year, domain, cluster_id "
        f"FROM documents WHERE {col} = %s ORDER BY year DESC, doc_id LIMIT %s", (cluster_id, limit))
    c["top_institutions"] = fetch_all(
        f"SELECT institution, min(institution_name) AS name, count(*) AS count FROM documents WHERE {col} = %s "
        "GROUP BY institution ORDER BY count DESC LIMIT 10", (cluster_id,))
    if c["scope"] == "domain":
        c = _with_counts([c])[0]
    else:
        rows = fetch_all("SELECT year, count(*) AS count FROM documents WHERE global_cluster = %s GROUP BY year ORDER BY year", (cluster_id,))
        c["counts"] = {r["year"]: r["count"] for r in rows}
    return c
