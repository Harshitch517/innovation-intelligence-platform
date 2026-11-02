"""PostgreSQL connection pool (psycopg 3) with the pgvector type registered."""
from __future__ import annotations

from typing import Any

from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from app import settings

_pool: ConnectionPool | None = None


def _configure(conn) -> None:
    """Per-connection setup: vector type, and HNSW iterative scans so filtered vector
    queries still return k rows."""
    register_vector(conn)
    conn.execute("SET hnsw.ef_search = 100")
    conn.execute("SET hnsw.iterative_scan = relaxed_order")
    conn.commit()


def pool() -> ConnectionPool:
    """The shared pool, opened on first use."""
    global _pool
    if _pool is None:
        _pool = ConnectionPool(settings.DATABASE_URL, min_size=1, max_size=5, configure=_configure,
                               kwargs={"row_factory": dict_row}, open=True)
    return _pool


def fetch_all(sql: str, params: Any = None) -> list[dict]:
    """Run a query and return every row as a dict."""
    with pool().connection() as conn:
        return conn.execute(sql, params).fetchall()


def fetch_one(sql: str, params: Any = None) -> dict | None:
    """Run a query and return the first row (or None)."""
    with pool().connection() as conn:
        return conn.execute(sql, params).fetchone()


def execute(sql: str, params: Any = None) -> None:
    """Run a statement that changes data and commit it."""
    with pool().connection() as conn:
        conn.execute(sql, params)
        conn.commit()
