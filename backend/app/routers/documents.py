"""/documents/{doc_id} and /documents/{doc_id}/similar."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.schemas import DocumentDetail, SimilarResponse
from app.services.db import fetch_all, fetch_one

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("/{doc_id}", response_model=DocumentDetail)
def get_document(doc_id: str) -> dict:
    """One document with its domain, sub-area, global cluster and institution."""
    doc = fetch_one(
        "SELECT doc_id, source, title, abstract, institution, institution_name, country, year, domain, raw_codes, "
        "cluster_id, global_cluster FROM documents WHERE doc_id = %s", (doc_id,))
    if doc is None:
        raise HTTPException(404, "document not found")
    clusters = {r["cluster_id"]: r for r in fetch_all(
        "SELECT cluster_id, name, keywords FROM clusters WHERE cluster_id = ANY(%s)",
        ([doc["cluster_id"], doc["global_cluster"]],))}
    doc["cluster"] = clusters.get(doc["cluster_id"])
    doc["global_cluster"] = clusters.get(doc["global_cluster"])
    return doc


@router.get("/{doc_id}/similar", response_model=SimilarResponse)
def similar(doc_id: str, k: int = Query(10, ge=1, le=50), other_institutions_only: bool = False) -> dict:
    """Most similar documents, marking those from other institutions."""
    doc = fetch_one("SELECT embedding, institution FROM documents WHERE doc_id = %s", (doc_id,))
    if doc is None:
        raise HTTPException(404, "document not found")
    extra = "AND institution <> %s" if other_institutions_only else ""
    params = [doc["embedding"], doc_id] + ([doc["institution"]] if other_institutions_only else []) + [doc["embedding"], k]
    rows = fetch_all(
        "SELECT doc_id, source, title, institution, institution_name, country, year, domain, cluster_id, "
        "left(abstract, 320) AS snippet, "
        f"1 - (embedding <=> %s) AS score FROM documents WHERE doc_id <> %s {extra} "
        "ORDER BY embedding <=> %s LIMIT %s", params)
    for r in rows:
        r["score"] = round(float(r["score"]), 4)
        r["other_institution"] = r["institution"] != doc["institution"]
    return {"doc_id": doc_id, "results": rows}
