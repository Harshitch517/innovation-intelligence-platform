"""Backend settings, read from environment variables (and the project .env when run locally)."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Locally the keys live in the repository's .env; on Render they are environment variables.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/innovation")
GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY") or None
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
ALLOWED_ORIGINS: list[str] = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",") if o.strip()]
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
DOMAINS: list[str] = ["Artificial Intelligence", "Healthcare", "Robotics", "Cybersecurity", "Renewable Energy"]
