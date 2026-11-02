"""Gemini API calls with a fallback: every function returns None when Gemini is unavailable,
so the platform keeps working without it."""
from __future__ import annotations

import time

import requests

from app import settings

URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def generate(prompt: str, max_tokens: int = 400, retries: int = 3) -> str | None:
    """Text reply from Gemini, or None if there is no key or every attempt fails."""
    if not settings.GEMINI_API_KEY:
        return None
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.3}}
    for attempt in range(retries):
        try:
            r = requests.post(URL.format(model=settings.GEMINI_MODEL), json=body,
                              headers={"x-goog-api-key": settings.GEMINI_API_KEY}, timeout=60)
            if r.ok:
                parts = r.json()["candidates"][0]["content"]["parts"]
                text = "".join(p.get("text", "") for p in parts).strip()
                return text or None
            if r.status_code not in (429, 500, 502, 503, 504):
                return None
        except (requests.RequestException, KeyError, IndexError):
            pass
        time.sleep(2 * (attempt + 1))
    return None


def explain_similarity(a: dict, b: dict, score: float) -> str | None:
    """One paragraph on why two documents are similar."""
    prompt = (
        "Two innovation records are semantically similar (cosine similarity "
        f"{score:.2f}). In 3-4 sentences, explain what technical ideas they share and how they differ. "
        "Be specific and do not invent facts beyond the abstracts.\n\n"
        f"A ({a['source']}, {a['institution_name']}, {a['year']}): {a['title']}\n{a['abstract'][:1500]}\n\n"
        f"B ({b['source']}, {b['institution_name']}, {b['year']}): {b['title']}\n{b['abstract'][:1500]}"
    )
    return generate(prompt)
