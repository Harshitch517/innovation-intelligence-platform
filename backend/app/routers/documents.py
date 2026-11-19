"""/documents/{doc_id} and /documents/{doc_id}/similar."""
from __future__ import annotations

import requests
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from app.schemas import DocumentDetail, FullTextLinks, SimilarResponse
from app.services import fulltext
from app.services.db import fetch_all, fetch_one

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("/{doc_id}", response_model=DocumentDetail)
def get_document(doc_id: str) -> dict:
    """One document with its domain, sub-area and global cluster."""
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


def _source(doc_id: str) -> str:
    row = fetch_one("SELECT source FROM documents WHERE doc_id = %s", (doc_id,))
    if row is None:
        raise HTTPException(404, "document not found")
    return row["source"]


@router.get("/{doc_id}/fulltext", response_model=FullTextLinks)
def full_text(doc_id: str) -> dict:
    """Where to read the complete document, and whether its PDF can be shown inline."""
    if _source(doc_id) == "patent":
        return {"kind": "patent", "url": fulltext.patent_url(doc_id), "is_open_access": True, "embeddable": False}
    try:
        links = fulltext.paper_links(doc_id.removeprefix("PAP-"))
    except requests.RequestException as e:
        raise HTTPException(502, "Paper details are unavailable right now. Please try again.") from e
    return {"kind": "paper", **links}


@router.get("/{doc_id}/pdf")
def pdf(doc_id: str) -> StreamingResponse:
    """Stream a paper's open-access PDF (the URL comes from OpenAlex, never from the client)."""
    if _source(doc_id) != "paper":
        raise HTTPException(404, "no PDF for this document")
    try:
        url = fulltext.paper_links(doc_id.removeprefix("PAP-"))["pdf_url"]
    except requests.RequestException as e:
        raise HTTPException(502, "PDF not available") from e
    opened = fulltext.stream_pdf(url) if url and url.startswith("https://") else None
    if opened is None:
        raise HTTPException(404, "PDF not available")
    stream, length = opened
    headers = {"Content-Disposition": "inline"}
    if length:
        headers["Content-Length"] = str(length)  # lets the site show a download percentage
    return StreamingResponse(stream, media_type="application/pdf", headers=headers)


@router.get("/{doc_id}/similar", response_model=SimilarResponse)
def similar(doc_id: str, k: int = Query(10, ge=1, le=50)) -> dict:
    """Most similar documents, ranked by cosine similarity."""
    doc = fetch_one("SELECT embedding FROM documents WHERE doc_id = %s", (doc_id,))
    if doc is None:
        raise HTTPException(404, "document not found")
    rows = fetch_all(
        "SELECT doc_id, source, title, institution, institution_name, country, year, domain, cluster_id, "
        "left(abstract, 320) AS snippet, "
        "1 - (embedding <=> %s) AS score FROM documents WHERE doc_id <> %s "
        "ORDER BY embedding <=> %s LIMIT %s", (doc["embedding"], doc_id, doc["embedding"], k))
    for r in rows:
        r["score"] = round(float(r["score"]), 4)
    return {"doc_id": doc_id, "results": rows}
