"""Check that GEMINI_API_KEY in .env works by sending one tiny request. Never prints the key."""
from __future__ import annotations

import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_key, load_config  # noqa: E402


def main() -> int:
    """Send a one-word prompt to the configured Gemini model; return 0 on success."""
    key = get_key("GEMINI_API_KEY")
    if key is None:
        print("GEMINI_API_KEY is missing from .env")
        return 1
    model = load_config()["apis"]["gemini_model"]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = {"contents": [{"parts": [{"text": "Reply with the single word: ok"}]}]}
    try:
        r = requests.post(url, json=body, headers={"x-goog-api-key": key}, timeout=180)
    except requests.RequestException as e:
        print(f"Gemini request did not complete: {type(e).__name__} (network issue, try again)")
        return 1
    if r.ok:
        text = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        print(f"Gemini key works (model {model}); reply: {text!r}")
        return 0
    message = r.json().get("error", {}).get("message", r.text[:200]) if r.content else ""
    print(f"Gemini request failed: HTTP {r.status_code} - {message}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
