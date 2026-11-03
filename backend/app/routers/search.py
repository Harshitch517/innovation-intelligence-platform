"""/search, /compare, /explain and /classify."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app import settings
from app.schemas import ClassifyRequest, ClassifyResponse, CompareResponse, ExplainResponse, SearchResponse
from app.services import gemini
from app.services.classifier import classify as run_classifier
from app.services.db import execute, fetch_all, fetch_one
from app.services.embedding import embed

router = APIRouter(tags=["search"])

SUMMARY_COLS = "doc_id, source, title, institution, institution_name, country, year, domain, cluster_id"
# Short abstract preview for result cards and hover previews.
SNIPPET = "left(abstract, 320) AS snippet"


@router.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=2, max_length=500, description="natural-language query"),
    domain: str | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    source: str | None = Query(None, pattern="^(patent|paper)$"),
    institution: str | None = None,
    k: int = Query(10, ge=1, le=50),
) -> dict:
    """Top-k documents for a query, ranked by cosine similarity, with optional filters."""
    where, params = [], []
    for col, val, op in (("domain", domain, "="), ("year", year_from, ">="), ("year", year_to, "<="),
                         ("source", source, "="), ("institution", institution, "=")):
        if val is not None:
            where.append(f"{col} {op} %s")
            params.append(val)
    vec = embed(q)
    sql = (f"SELECT {SUMMARY_COLS}, {SNIPPET}, 1 - (embedding <=> %s) AS score FROM documents "
           f"{'WHERE ' + ' AND '.join(where) if where else ''} ORDER BY embedding <=> %s LIMIT %s")
    rows = fetch_all(sql, [vec, *params, vec, k])
    for r in rows:
        r["score"] = round(float(r["score"]), 4)
    return {"query": q, "count": len(rows), "results": rows}


def _pair(a: str, b: str) -> tuple[dict, dict, float]:
    """Both documents plus their cosine similarity (404 if either is missing)."""
    rows = fetch_all(
        "SELECT d.*, 1 - (d.embedding <=> (SELECT embedding FROM documents WHERE doc_id = %s)) AS sim "
        "FROM documents d WHERE doc_id IN (%s, %s)", (a, a, b))
    found = {r["doc_id"]: r for r in rows}
    missing = [x for x in (a, b) if x not in found]
    if missing:
        raise HTTPException(404, f"document not found: {', '.join(missing)}")
    return found[a], found[b], round(float(found[b]["sim"]), 4)


@router.get("/compare", response_model=CompareResponse)
def compare(a: str, b: str) -> dict:
    """Similarity score between two documents and what they share."""
    da, db_, sim = _pair(a, b)
    kw = fetch_all("SELECT cluster_id, keywords FROM clusters WHERE cluster_id = ANY(%s)",
                   ([da["cluster_id"], db_["cluster_id"], da["global_cluster"], db_["global_cluster"]],))
    words = {r["cluster_id"]: set(r["keywords"]) for r in kw}
    shared = (words.get(da["global_cluster"], set()) | words.get(da["cluster_id"], set())) & \
             (words.get(db_["global_cluster"], set()) | words.get(db_["cluster_id"], set()))
    return {
        "a": da, "b": db_, "similarity": sim,
        "same_domain": da["domain"] == db_["domain"],
        "same_cluster": da["cluster_id"] == db_["cluster_id"],
        "same_global_cluster": da["global_cluster"] == db_["global_cluster"],
        "shared_keywords": sorted(shared),
        "different_institutions": da["institution"] != db_["institution"],
    }


@router.get("/explain", response_model=ExplainResponse)
def explain(a: str, b: str) -> dict:
    """AI explanation of why two documents are similar, cached in explanations_cache."""
    key = tuple(sorted((a, b)))
    da, db_, sim = _pair(*key)
    cached = fetch_one("SELECT explanation, model FROM explanations_cache WHERE doc_a = %s AND doc_b = %s", key)
    if cached:
        return {"a": key[0], "b": key[1], "similarity": sim, "explanation": cached["explanation"],
                "cached": True, "model": None}
    text = gemini.explain_similarity(da, db_, sim)
    if text is None:
        return {"a": key[0], "b": key[1], "similarity": sim, "explanation": None, "cached": False, "model": None,
                "note": "Explanation unavailable right now. Please try again in a minute."}
    execute("INSERT INTO explanations_cache (doc_a, doc_b, explanation, model) VALUES (%s, %s, %s, %s) "
            "ON CONFLICT DO NOTHING", (*key, text, settings.GEMINI_MODEL))
    return {"a": key[0], "b": key[1], "similarity": sim, "explanation": text, "cached": False, "model": None}


@router.post("/classify", response_model=ClassifyResponse)
def classify(body: ClassifyRequest) -> dict:
    """Predicted domain and probabilities for a new title + abstract."""
    return run_classifier(body.title, body.abstract)
