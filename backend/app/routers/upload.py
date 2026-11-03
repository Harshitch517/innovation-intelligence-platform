"""Custom document upload: /analyze, /compare/upload, /explain/upload.

The upload is read locally (pypdf / python-docx) and embedded with fastembed, so analysing and
matching cost no LLM credits. Nothing is stored except cached explanations. The client keeps
the extracted title + abstract and sends them back for compare/explain (stateless server).
"""
from __future__ import annotations

import hashlib

import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app import settings
from app.schemas import AnalyzeResponse, ExplainResponse, UploadCompareRequest, UploadCompareResponse, UploadExplainRequest
from app.services import gemini
from app.services.classifier import classify
from app.services.db import execute, fetch_all, fetch_one
from app.services.embedding import embed
from app.services.extract import MAX_BYTES, ExtractionError, extract

router = APIRouter(tags=["upload"])

DOC_COLS = "doc_id, source, title, institution, institution_name, country, year, domain, cluster_id"


def _vector(title: str, abstract: str) -> np.ndarray:
    return embed(f"{title.strip()}. {abstract.strip()}")


def _upload_id(title: str, abstract: str) -> str:
    """Stable ID for an uploaded text, used as the cache key for explanations."""
    return "UP-" + hashlib.sha1(f"{title}\n{abstract}".encode()).hexdigest()[:16]


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...), k: int = Query(10, ge=1, le=30)) -> dict:
    """Read a .pdf/.docx/.txt, predict its domain and return the most similar database documents."""
    data = await file.read(MAX_BYTES + 1)
    try:
        info = extract(file.filename or "upload", data)
    except ExtractionError as e:
        raise HTTPException(422, str(e)) from e
    vec = _vector(info["title"], info["abstract"])
    rows = fetch_all(f"SELECT {DOC_COLS}, left(abstract, 320) AS snippet, 1 - (embedding <=> %s) AS score "
                     "FROM documents ORDER BY embedding <=> %s LIMIT %s", (vec, vec, k))
    for r in rows:
        r["score"] = round(float(r["score"]), 4)
    return {"filename": file.filename, **info, "prediction": classify(info["title"], info["abstract"]), "matches": rows}


def _doc(doc_id: str, cols: str = DOC_COLS) -> dict:
    row = fetch_one(f"SELECT {cols}, embedding FROM documents WHERE doc_id = %s", (doc_id,))
    if row is None:
        raise HTTPException(404, f"document not found: {doc_id}")
    return row


@router.post("/compare/upload", response_model=UploadCompareResponse)
def compare_upload(body: UploadCompareRequest) -> dict:
    """Similarity between an uploaded document and one database document."""
    b = _doc(body.doc_id)
    sim = float(np.dot(_vector(body.title, body.abstract), b.pop("embedding").to_numpy().astype(np.float32)))
    predicted = classify(body.title, body.abstract)["domain"]
    return {"b": b, "similarity": round(sim, 4), "predicted_domain": predicted, "same_domain": predicted == b["domain"]}


@router.post("/explain/upload", response_model=ExplainResponse)
def explain_upload(body: UploadExplainRequest) -> dict:
    """AI explanation for an uploaded document vs a database document (cached, short prompt)."""
    b = _doc(body.doc_id, DOC_COLS + ", abstract")
    sim = round(float(np.dot(_vector(body.title, body.abstract), b.pop("embedding").to_numpy().astype(np.float32))), 4)
    key = (_upload_id(body.title, body.abstract), body.doc_id)
    cached = fetch_one("SELECT explanation, model FROM explanations_cache WHERE doc_a = %s AND doc_b = %s", key)
    if cached:
        return {"a": key[0], "b": key[1], "similarity": sim, "explanation": cached["explanation"], "cached": True, "model": None}
    a = {"source": "uploaded document", "title": body.title, "abstract": body.abstract}
    text = gemini.explain_similarity(a, b, sim)
    if text is None:
        return {"a": key[0], "b": key[1], "similarity": sim, "explanation": None, "cached": False, "model": None,
                "note": "Explanation unavailable right now. Please try again in a minute."}
    execute("INSERT INTO explanations_cache (doc_a, doc_b, explanation, model) VALUES (%s, %s, %s, %s) "
            "ON CONFLICT DO NOTHING", (*key, text, settings.GEMINI_MODEL))
    return {"a": key[0], "b": key[1], "similarity": sim, "explanation": text, "cached": False, "model": None}
