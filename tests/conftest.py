"""
Shared pytest fixtures used across all test modules.

Fixtures defined here are automatically available to every test file
without needing an explicit import — pytest discovers conftest.py
automatically.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
import chromadb

# ── Path fixtures ──────────────────────────────────────────────────

@pytest.fixture
def sample_pdf_path():
    """Path to the oneNDA sample contract — always present in the repo."""
    path = Path("legal_contracts/oneNDA_v2_1.pdf")
    if not path.exists():
        pytest.skip(f"Sample PDF not found at {path} — skipping PDF-dependent test")
    return path


@pytest.fixture
def agency_pdf_path():
    """Path to the Agency Agreement sample contract."""
    path = Path("legal_contracts/ATHENSBANCSHARESCORP_11_02_2009-EX-1_2-AGENCY_AGREEMENT___2009.PDF")
    if not path.exists():
        pytest.skip(f"Agency Agreement PDF not found at {path}")
    return path


@pytest.fixture
def affiliate_pdf_path():
    """Path to the Affiliate Agreement sample contract."""
    path = Path("legal_contracts/CreditcardscomInc_20070810_S-1_EX-10_33_362297_EX-10_33_Affiliate_Agreement.pdf")
    if not path.exists():
        pytest.skip(f"Affiliate Agreement PDF not found at {path}")
    return path


# ── Temporary storage fixtures ─────────────────────────────────────

@pytest.fixture
def tmp_dir():
    """Temporary directory that is cleaned up after each test."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def tmp_chroma(tmp_dir):
    """Temporary ChromaDB collection — isolated per test, auto-cleaned."""
    chroma_dir = tmp_dir / "chroma"
    chroma_dir.mkdir()
    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_or_create_collection(
        name="test_chunks",
        metadata={"hnsw:space": "cosine"},
    )
    yield collection


@pytest.fixture
def tmp_sqlite(tmp_dir):
    """Temporary SQLite connection with full schema — isolated per test."""
    import sqlite3
    db_path = tmp_dir / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY, source_name TEXT NOT NULL,
            clause_id TEXT NOT NULL, heading TEXT, full_text TEXT,
            page_start INTEGER, page_end INTEGER, format_used TEXT,
            char_count INTEGER, embedded_at TEXT DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_name);
        CREATE TABLE IF NOT EXISTS ingestions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_name TEXT NOT NULL, total_chunks INTEGER,
            embedded INTEGER, skipped INTEGER, failed INTEGER,
            ingested_at TEXT DEFAULT (datetime('now'))
        );
    """)
    conn.commit()
    yield conn
    conn.close()


# ── Sample data fixtures ───────────────────────────────────────────

@pytest.fixture
def sample_chunk():
    """A single realistic Chunk object for testing."""
    from src.pipeline.chunker import Chunk
    return Chunk(
        clause_id="Section 4",
        heading="Representations and Warranties",
        text=(
            "The Company and the Bank jointly and severally represent and "
            "warrant to and agree with the Agent as follows: (a) The "
            "Registration Statement has been declared effective."
        ),
        page_start=8,
        page_end=10,
        format_used="section_n",
    )


@pytest.fixture
def sample_chunks():
    """A list of realistic Chunk objects covering multiple formats."""
    from src.pipeline.chunker import Chunk
    return [
        Chunk(
            clause_id="Section 4",
            heading="Representations and Warranties",
            text="The Company represents and warrants to the Agent as follows.",
            page_start=8, page_end=10, format_used="section_n",
        ),
        Chunk(
            clause_id="Section 9",
            heading="Indemnification",
            text="The Company agrees to indemnify and hold harmless the Agent.",
            page_start=20, page_end=22, format_used="section_n",
        ),
        Chunk(
            clause_id="Clause 1",
            heading="What is Confidential Information?",
            text="Confidential Information means information that is disclosed.",
            page_start=2, page_end=2, format_used="onenda_table",
        ),
    ]


@pytest.fixture
def sample_retrieved_chunks():
    """A list of RetrievedChunk objects for testing retrieval modules."""
    from src.retrieval.retriever import RetrievedChunk
    return [
        RetrievedChunk(
            chunk_id="agency.pdf::Section 9",
            clause_id="Section 9",
            source_name="agency.pdf",
            heading="Indemnification",
            full_text="The Company agrees to indemnify and hold harmless the Agent from all losses.",
            page_start=20, page_end=22,
            format_used="section_n",
            similarity_score=0.87,
            mmr_rank=1,
        ),
        RetrievedChunk(
            chunk_id="oneNDA.pdf::Clause 3",
            clause_id="Clause 3",
            source_name="oneNDA.pdf",
            heading="What are my obligations?",
            full_text="The Receiver must only use the Confidential Information for the Purpose.",
            page_start=2, page_end=3,
            format_used="onenda_table",
            similarity_score=0.75,
            mmr_rank=2,
        ),
    ]


@pytest.fixture
def populated_sqlite(tmp_sqlite, sample_chunks):
    """SQLite connection pre-populated with sample chunk rows."""
    conn = tmp_sqlite
    for chunk in sample_chunks:
        conn.execute(
            """INSERT OR IGNORE INTO chunks
               (id, source_name, clause_id, heading, full_text,
                page_start, page_end, format_used, char_count)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                f"agency.pdf::{chunk.clause_id}",
                "agency.pdf",
                chunk.clause_id,
                chunk.heading,
                chunk.text,
                chunk.page_start,
                chunk.page_end,
                chunk.format_used,
                chunk.char_count,
            ),
        )
    conn.commit()
    return conn


# ── API key guard ──────────────────────────────────────────────────

@pytest.fixture
def require_api_key():
    """
    Skip integration tests if GOOGLE_API_KEY is not set.
    Use this fixture in any test decorated with @pytest.mark.integration.
    """
    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        pytest.skip("GOOGLE_API_KEY not set — skipping integration test")
    return key
