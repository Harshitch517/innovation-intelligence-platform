"""/institutions, /institutions/network and /institutions/{name}."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.schemas import InstitutionDetail, InstitutionSummary, NetworkResponse
from app.services.db import fetch_all, fetch_one

router = APIRouter(prefix="/institutions", tags=["institutions"])

SUMMARY_COLS = "institution, name, country, documents, patents, papers, top_domain"


@router.get("", response_model=list[InstitutionSummary])
def list_institutions(q: str | None = None, domain: str | None = None, country: str | None = None,
                      limit: int = Query(50, ge=1, le=700)) -> list[dict]:
    """Institution profiles (at least 10 documents), largest first; optional name search."""
    where, params = ["true"], []
    if q:
        where.append("(name ILIKE %s OR institution ILIKE %s)"); params += [f"%{q}%", f"%{q}%"]
    if domain:
        where.append("top_domain = %s"); params.append(domain)
    if country:
        where.append("country = %s"); params.append(country)
    return fetch_all(f"SELECT {SUMMARY_COLS} FROM institutions WHERE {' AND '.join(where)} "
                     "ORDER BY documents DESC LIMIT %s", params + [limit])


@router.get("/network", response_model=NetworkResponse)
def network(domain: str | None = None, min_similarity: float = Query(0.0, ge=0, le=1)) -> dict:
    """Network nodes and edges; with `domain`, only institutions active in that domain."""
    if domain:
        nodes = fetch_all("SELECT institution AS id, name, country, documents, top_domain FROM institutions "
                          "WHERE (by_domain ->> %s)::int > 0", (domain,))
    else:
        nodes = fetch_all("SELECT institution AS id, name, country, documents, top_domain FROM institutions")
    ids = [n["id"] for n in nodes]
    edges = fetch_all("SELECT source, target, similarity, shared_clusters FROM institution_edges "
                      "WHERE source = ANY(%s) AND target = ANY(%s) AND similarity >= %s", (ids, ids, min_similarity))
    return {"domain": domain, "nodes": nodes, "edges": edges}


@router.get("/{name}", response_model=InstitutionDetail)
def institution(name: str) -> dict:
    """Profile, domain mix, top sub-areas, similar institutions and recent documents."""
    inst = fetch_one(f"SELECT {SUMMARY_COLS}, by_domain, by_year, top_clusters FROM institutions WHERE institution = %s", (name,))
    if inst is None:
        raise HTTPException(404, "institution not found")
    inst["top_clusters"] = fetch_all(
        "SELECT cluster_id, name, keywords FROM clusters WHERE cluster_id = ANY(%s) "
        "ORDER BY array_position(%s, cluster_id)", (inst["top_clusters"], inst["top_clusters"]))
    inst["similar"] = fetch_all(
        "SELECT s.similar_institution AS institution, i.name, s.similarity FROM institution_similarity s "
        "JOIN institutions i ON i.institution = s.similar_institution WHERE s.institution = %s "
        "ORDER BY s.similarity DESC", (name,))
    inst["recent_documents"] = fetch_all(
        "SELECT doc_id, source, title, institution, institution_name, country, year, domain, cluster_id "
        "FROM documents WHERE institution = %s ORDER BY year DESC, doc_id LIMIT 10", (name,))
    return inst
