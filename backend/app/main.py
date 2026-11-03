"""Innovation Intelligence Platform API (FastAPI).

Serves the precomputed results in PostgreSQL + pgvector. Only the user's search query (and a
/classify input) is embedded live, with fastembed.

Run locally:
    cd backend && uvicorn app.main:app --reload
"""
from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import settings
from app.routers import documents, institutions, search, trends, upload
from app.services.db import fetch_one

# Show warnings such as psycopg_pool connection errors in the server log.
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

app = FastAPI(
    title="Innovation Intelligence Platform API",
    description="Search, compare and explore university patents and research papers (2020–2025) "
                "across five technology domains.",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=settings.ALLOWED_ORIGINS, allow_methods=["GET", "POST"], allow_headers=["*"])

app.include_router(search.router)
app.include_router(documents.router)
app.include_router(trends.router)
app.include_router(institutions.router)
app.include_router(upload.router)


@app.get("/", tags=["health"])
def health() -> dict:
    """Health check: database reachable and document count."""
    row = fetch_one("SELECT count(*) AS documents FROM documents")
    return {"status": "ok", "documents": row["documents"], "domains": settings.DOMAINS}
