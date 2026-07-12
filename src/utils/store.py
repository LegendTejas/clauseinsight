"""
ClauseInsight — Shared Store
==============================

Central module for all database connection management and shared
query helpers across the ClauseInsight pipeline.

WHY THIS MODULE EXISTS
-----------------------
ChromaDB and SQLite connections are needed by multiple modules:
  - embedder.py  — writes chunks + vectors
  - retriever.py — reads vectors for similarity search
  - scanner.py   — reads chunk text for risk classification
  - UI pages     — reads metadata for display + contract selector

Without this module, each of those would either duplicate the
connection + schema creation logic, or import from embedder.py
(wrong dependency direction — retriever shouldn't depend on embedder).

store.py is the single source of truth for:
  - Where data lives on disk (DEFAULT_CHROMA_DIR, DEFAULT_SQLITE_PATH)
  - The ChromaDB collection name and similarity metric
  - The SQLite schema (created idempotently via CREATE IF NOT EXISTS)
  - Read-side query helpers (get_chunk_text, list_ingested_contracts, etc.)
  - delete_contract() which touches both stores atomically

Write-side logic (embedding, batch inserts) stays in embedder.py —
that's the only module that should be writing to the stores.
"""

from __future__ import annotations

import logging
import sqlite3
import os
from pathlib import Path

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# Path + collection constants
# These are the single source of truth — import from here everywhere.
# ──────────────────────────────────────────────────────────────────

def get_embedding_model() -> str:
    return os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")

def get_chroma_collection_name() -> str:
    model_name = get_embedding_model().replace("-", "_")
    return f"contracts_{model_name}"

DEFAULT_CHROMA_DIR = Path("data/chroma")
DEFAULT_SQLITE_PATH = Path("data/metadata.db")


# ──────────────────────────────────────────────────────────────────
# Connection helpers
# ──────────────────────────────────────────────────────────────────

def get_chroma_collection(
    chroma_dir: Path = DEFAULT_CHROMA_DIR,
) -> chromadb.Collection:
    """
    Get (or create) the ChromaDB persistent collection.

    Uses cosine similarity — better than L2 for high-dimensional text
    embeddings because it normalises for document length differences
    between short sub-clauses and long section chunks.

    Safe to call multiple times — ChromaDB's get_or_create_collection
    is idempotent and returns the existing collection if it already exists.

    Args:
        chroma_dir: Directory for ChromaDB's persistent storage.
                    Created automatically if it doesn't exist.

    Returns:
        The ChromaDB Collection object ready for add/query/delete.
    """
    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(chroma_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    collection_name = get_chroma_collection_name()
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info(
        "ChromaDB collection '%s' ready (dir: %s, count: %d)",
        collection_name, chroma_dir, collection.count()
    )
    return collection


def get_sqlite_connection(
    db_path: Path = DEFAULT_SQLITE_PATH,
) -> sqlite3.Connection:
    """
    Open (or create) the SQLite metadata database.

    Creates all tables idempotently on first call. Safe to call
    multiple times across modules — will not overwrite existing data.

    WAL journal mode is enabled for safe concurrent reads from the
    Streamlit UI while embedder.py is writing in the background.

    Schema
    ------
    chunks table — one row per embedded chunk:
        id            TEXT PRIMARY KEY  — "<source_name>::<clause_id>"
                                          matches ChromaDB document ID
        source_name   TEXT              — original contract filename
        clause_id     TEXT              — "Section 4", "Clause 1(a)", etc.
        heading       TEXT              — clause title / first N words
        full_text     TEXT              — complete clause text
                                          (not stored in ChromaDB)
        page_start    INTEGER           — 1-indexed, matches PDF viewer
        page_end      INTEGER
        format_used   TEXT              — chunker strategy that produced this
        char_count    INTEGER
        embedded_at   TEXT              — ISO timestamp (UTC)

    ingestions table — audit log, one row per embed_and_store() call:
        source_name   TEXT
        total_chunks  INTEGER
        embedded      INTEGER
        skipped       INTEGER           — already existed (idempotency)
        failed        INTEGER
        ingested_at   TEXT

    Args:
        db_path: Path to the SQLite file.
                 Parent directory created automatically if needed.

    Returns:
        sqlite3.Connection with row_factory=sqlite3.Row set,
        so rows can be accessed by column name.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS chunks (
            id            TEXT PRIMARY KEY,
            source_name   TEXT NOT NULL,
            clause_id     TEXT NOT NULL,
            heading       TEXT,
            full_text     TEXT,
            page_start    INTEGER,
            page_end      INTEGER,
            format_used   TEXT,
            char_count    INTEGER,
            embedded_at   TEXT DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_chunks_source
            ON chunks(source_name);

        CREATE INDEX IF NOT EXISTS idx_chunks_clause
            ON chunks(source_name, clause_id);

        CREATE TABLE IF NOT EXISTS ingestions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            source_name   TEXT NOT NULL,
            total_chunks  INTEGER,
            embedded      INTEGER,
            skipped       INTEGER,
            failed        INTEGER,
            ingested_at   TEXT DEFAULT (datetime('now'))
        );
    """)
    conn.commit()
    logger.info("SQLite metadata store ready (path: %s)", db_path)
    return conn


# ──────────────────────────────────────────────────────────────────
# Read-side query helpers
# Used by retriever.py, scanner.py, and UI pages.
# Write-side logic (INSERT, batch operations) stays in embedder.py.
# ──────────────────────────────────────────────────────────────────

def get_chunk_text(
    clause_id: str,
    source_name: str,
    conn: sqlite3.Connection,
) -> str | None:
    """
    Fetch the full clause text for a specific clause by ID.

    Used by the Q&A engine to ground the LLM's answer in the exact
    clause text without doing a second vector search. SQLite lookup
    by primary key is O(log n) — much faster than a ChromaDB query.

    Args:
        clause_id:   e.g. "Section 4", "Clause 1(a)", "Para 7"
        source_name: original contract filename — needed because
                     clause_id is only unique within one contract.
        conn:        open SQLite connection.

    Returns:
        Full clause text string, or None if not found.
    """
    row = conn.execute(
        "SELECT full_text FROM chunks WHERE clause_id = ? AND source_name = ?",
        (clause_id, source_name),
    ).fetchone()
    return row["full_text"] if row else None


def get_chunk_by_id(
    chunk_id: str,
    conn: sqlite3.Connection,
) -> dict | None:
    """
    Fetch a complete chunk row by its store ID ("<source>::<clause_id>").

    Used when retriever.py returns a ChromaDB result with an ID and
    the caller needs the full text + all metadata in one lookup.

    Args:
        chunk_id: The composite ID, e.g. "contract.pdf::Section 4(a)"
        conn:     open SQLite connection.

    Returns:
        Dict with all chunk columns, or None if not found.
    """
    row = conn.execute(
        "SELECT * FROM chunks WHERE id = ?",
        (chunk_id,),
    ).fetchone()
    return dict(row) if row else None


def get_all_chunks_for_contract(
    source_name: str,
    conn: sqlite3.Connection,
    include_text: bool = False,
) -> list[dict]:
    """
    Return all chunk rows for a given contract, ordered by page then clause.

    Used by scanner.py to iterate over all clauses for risk classification.
    By default does NOT return full_text to avoid loading large strings into
    memory when only metadata is needed — pass include_text=True when you
    need the clause body (e.g. for LLM classification).

    Args:
        source_name:  original contract filename.
        conn:         open SQLite connection.
        include_text: if True, also return the full_text column.

    Returns:
        List of dicts, one per chunk, ordered by page_start then clause_id.
    """
    cols = (
        "id, clause_id, heading, page_start, page_end, format_used, char_count, embedded_at"
        + (", full_text" if include_text else "")
    )
    rows = conn.execute(
        f"SELECT {cols} FROM chunks WHERE source_name = ? ORDER BY page_start, clause_id",
        (source_name,),
    ).fetchall()
    return [dict(r) for r in rows]


def list_ingested_contracts(conn: sqlite3.Connection) -> list[dict]:
    """
    Return a summary row for every contract currently in the metadata store.

    Used by the Streamlit UI to populate the contract selector dropdown.
    Returns contracts ordered by most-recently ingested first.

    Returns:
        List of dicts with keys:
            source_name, chunk_count, first_page, last_page, last_ingested
    """
    rows = conn.execute(
        """
        SELECT
            source_name,
            COUNT(*)         AS chunk_count,
            MIN(page_start)  AS first_page,
            MAX(page_end)    AS last_page,
            MAX(embedded_at) AS last_ingested
        FROM chunks
        GROUP BY source_name
        ORDER BY last_ingested DESC
        """,
    ).fetchall()
    return [dict(r) for r in rows]


def delete_contract(
    source_name: str,
    collection: chromadb.Collection,
    conn: sqlite3.Connection,
) -> int:
    """
    Remove all chunks for a contract from BOTH ChromaDB and SQLite atomically.

    Call this before re-ingesting a contract to ensure a clean slate,
    or when a user explicitly deletes a contract from the UI.

    ChromaDB deletes happen in batches of 100 because the API has an
    implicit limit on bulk delete size.

    Args:
        source_name: original contract filename.
        collection:  open ChromaDB collection.
        conn:        open SQLite connection.

    Returns:
        Number of chunks deleted (same count removed from both stores).
    """
    rows = conn.execute(
        "SELECT id FROM chunks WHERE source_name = ?",
        (source_name,)
    ).fetchall()
    ids = [r["id"] for r in rows]

    if not ids:
        logger.info("delete_contract: no chunks found for '%s'", source_name)
        return 0

    # ChromaDB bulk delete in batches
    for i in range(0, len(ids), 100):
        collection.delete(ids=ids[i:i + 100])

    # SQLite delete
    conn.execute("DELETE FROM chunks WHERE source_name = ?", (source_name,))
    conn.commit()

    logger.info(
        "Deleted %d chunks for '%s' from both ChromaDB and SQLite.",
        len(ids), source_name
    )
    return len(ids)
