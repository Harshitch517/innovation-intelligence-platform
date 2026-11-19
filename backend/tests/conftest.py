"""Make the backend's `app` package importable and provide a test client and sample IDs."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.services.db import fetch_one  # noqa: E402


@pytest.fixture(scope="session")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(scope="session")
def sample() -> dict:
    """A patent, a paper and a sub-area from the loaded database."""
    patent = fetch_one("SELECT doc_id, cluster_id FROM documents WHERE source = 'patent' ORDER BY doc_id LIMIT 1")
    paper = fetch_one("SELECT doc_id FROM documents WHERE source = 'paper' ORDER BY doc_id LIMIT 1")
    return {"patent": patent["doc_id"], "paper": paper["doc_id"], "cluster": patent["cluster_id"]}
