"""Where to read a document in full. The database holds only titles and abstracts.

Papers: OpenAlex lists the open-access copy (single-record lookups are free). If our server can
fetch that PDF, the site shows it inline through /documents/{id}/pdf; many publishers block
servers, and those PDFs open in a new tab from the reader's own browser instead.
Patents: the Google Patents page. No LLM is involved.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Iterator

import requests

OPENALEX = "https://api.openalex.org/works/"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; InnovationIntelligence/1.0)", "Accept": "application/pdf,*/*"}
MAX_PDF_BYTES = 40 * 1024 * 1024
CHUNK = 64 * 1024
MAX_AUTHORS = 15


def _open_pdf(url: str) -> requests.Response | None:
    """Open a PDF stream, or None if the server refuses or the response is not a PDF."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=15, stream=True, allow_redirects=True)
    except requests.RequestException:
        return None
    declared = int(r.headers.get("content-length") or 0)
    if not r.ok or declared > MAX_PDF_BYTES:
        r.close()
        return None
    return r


def _first_chunk(r: requests.Response) -> bytes:
    try:
        return next(r.iter_content(CHUNK), b"")
    except requests.RequestException:
        return b""


@lru_cache(maxsize=2048)
def paper_links(work_id: str) -> dict:
    """Details and open-access links for an OpenAlex work; `embeddable` means our server can fetch
    the PDF. Raises on a failed lookup, so a temporary failure is not cached."""
    params = {"select": "doi,open_access,best_oa_location,primary_location,authorships,publication_date,"
                        "cited_by_count,referenced_works_count,type"}
    if key := os.getenv("OPENALEX_API_KEY"):
        params["api_key"] = key
    r = requests.get(OPENALEX + work_id, params=params, timeout=15)
    r.raise_for_status()
    j = r.json()
    best = j.get("best_oa_location") or {}
    primary = j.get("primary_location") or {}
    pdf = best.get("pdf_url")
    url = pdf or best.get("landing_page_url") or j.get("doi") or primary.get("landing_page_url")
    embeddable = False
    if pdf and pdf.startswith("https://"):
        stream = _open_pdf(pdf)
        if stream is not None:
            embeddable = _first_chunk(stream).startswith(b"%PDF")
            stream.close()
    authors = [(a.get("author") or {}).get("display_name") for a in j.get("authorships") or []]
    return {
        "url": url, "pdf_url": pdf, "embeddable": embeddable,
        "is_open_access": bool((j.get("open_access") or {}).get("is_oa")),
        "oa_status": (j.get("open_access") or {}).get("oa_status"),
        "doi": j.get("doi"),
        "venue": (primary.get("source") or {}).get("display_name"),
        "publication_date": j.get("publication_date"),
        "work_type": j.get("type"),
        "authors": [a for a in authors if a][:MAX_AUTHORS],
        "author_count": len(authors),
        "cited_by_count": j.get("cited_by_count"),
        "references_count": j.get("referenced_works_count"),
    }


def patent_url(doc_id: str) -> str:
    """PAT-US-11234567-B2 -> the Google Patents page."""
    return f"https://patents.google.com/patent/{doc_id.removeprefix('PAT-').replace('-', '')}/en"


def stream_pdf(url: str) -> tuple[Iterator[bytes], int | None] | None:
    """(iterator over a PDF's bytes, length if known), or None if it cannot be fetched."""
    r = _open_pdf(url)
    if r is None:
        return None
    first = _first_chunk(r)
    if not first.startswith(b"%PDF"):
        r.close()
        return None

    def chunks() -> Iterator[bytes]:
        sent = len(first)
        try:
            yield first
            for chunk in r.iter_content(CHUNK):
                sent += len(chunk)
                if sent > MAX_PDF_BYTES:
                    break
                yield chunk
        finally:
            r.close()

    # A compressed response is decoded here, so its declared length would be wrong.
    length = None if r.headers.get("content-encoding") else int(r.headers.get("content-length") or 0) or None
    return chunks(), length
