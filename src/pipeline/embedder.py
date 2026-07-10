"""
ClauseInsight — Embedder
=========================

Takes the list[Chunk] produced by chunker.py, embeds each chunk using
Google's text-embedding-004 model, and persists everything into two
synchronized stores via utils/store.py:

  1. ChromaDB  — vector store for similarity search at query time
  2. SQLite    — relational metadata store for structured lookups

WHY EMBEDDER ONLY WRITES
--------------------------
All connection management, schema creation, and read-side query helpers
live in utils/store.py. embedder.py is purely a write-side module:
  - Takes chunks → produces vectors → persists to both stores
  - Owns all Gemini API call logic (both RETRIEVAL_DOCUMENT and
    RETRIEVAL_QUERY task types live here so API calls stay in one place)
  - Owns retry/backoff/rate-limit handling

EMBEDDING DESIGN
-----------------
We embed: heading + full clause text as a single string.
  - heading gives the model context about what kind of clause this is
  - full text gives retrieval the semantic content to match against

task_type differentiation (Google's recommendation):
  - Indexing:  RETRIEVAL_DOCUMENT — optimises for being searched against
  - Querying:  RETRIEVAL_QUERY   — used in embed_query(), called by retriever.py

RATE LIMIT HANDLING
--------------------
text-embedding-004 free tier: 1,500 requests/minute.
We batch up to EMBED_BATCH_SIZE chunks per API call with inter-batch
sleep. Rate limit / transient errors get exponential backoff up to
MAX_RETRIES before a batch is marked failed and the run continues.

IDEMPOTENCY
------------
embed_and_store() checks ChromaDB for existing document IDs before
embedding. Re-uploading the same contract skips already-embedded chunks
and only processes new ones — safe to call multiple times.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import chromadb
import sqlite3
from google import genai
from google.genai import types

import sys as _sys
_root_dir = str(Path(__file__).resolve().parent.parent.parent)
if _root_dir not in _sys.path:
    _sys.path.insert(0, _root_dir)

from src.pipeline.chunker import Chunk

from src.utils.store import (
    CHROMA_COLLECTION,
    DEFAULT_CHROMA_DIR,
    DEFAULT_SQLITE_PATH,
    get_chroma_collection,
    get_sqlite_connection,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# Embedding constants
# ──────────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "gemini-embedding-001"

# gemini-embedding-001 native dimension is 3072.
# We use output_dimensionality=768 for efficiency — legal text works well at this size.
EMBEDDING_DIM = 768

# Google free tier: 1,500 req/min.
# Batch of 20 = ~8 batches per typical contract, well within limits.
EMBED_BATCH_SIZE = 20

# Seconds to sleep between batches.
# 0.5s * 8 batches = 4s overhead per contract — acceptable.
INTER_BATCH_SLEEP = 0.5

# Retry config for rate limit / transient API errors
MAX_RETRIES = 4
RETRY_BASE_DELAY = 2.0  # seconds, doubles each retry


# ──────────────────────────────────────────────────────────────────
# Result dataclass
# ──────────────────────────────────────────────────────────────────

@dataclass
class EmbedResult:
    """Summary of one embed_and_store() run — returned to the caller."""
    source_name: str
    total_chunks: int
    embedded_count: int    # newly embedded this run
    skipped_count: int     # already existed in ChromaDB (idempotency)
    failed_count: int      # chunks that errored after all retries
    elapsed_seconds: float

    @property
    def success(self) -> bool:
        return self.failed_count == 0

    def __str__(self) -> str:
        return (
            f"EmbedResult({self.source_name!r}: "
            f"{self.embedded_count} embedded, "
            f"{self.skipped_count} skipped, "
            f"{self.failed_count} failed, "
            f"{self.elapsed_seconds:.1f}s)"
        )


# ──────────────────────────────────────────────────────────────────
# Gemini client
# ──────────────────────────────────────────────────────────────────

def _make_gemini_client() -> genai.Client:
    """
    Create the Google GenAI client from GOOGLE_API_KEY env var.

    Loads .env file first (if python-dotenv is installed) so keys
    defined there are available via os.environ.

    Raises a clear EnvironmentError if the key is missing — better
    than letting the first API call fail with a cryptic auth error.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # python-dotenv not installed; rely on real env vars

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY environment variable is not set. "
            "Get a key at https://ai.google.dev and set it before running."
        )
    return genai.Client(api_key=api_key)


# ──────────────────────────────────────────────────────────────────
# Text preparation
# ──────────────────────────────────────────────────────────────────

def _build_embed_text(chunk: Chunk) -> str:
    """
    Combine heading + full text into the string we embed.

    Format: "<heading>\n\n<body>"
    Heading is prepended so the embedding captures clause type even
    if the body's opening tokens are boilerplate. If heading is empty
    or is already a prefix of the body, we embed body only to avoid
    redundant repetition in the token sequence.
    """
    heading = chunk.heading.strip()
    body = chunk.text.strip()

    if not heading or body.startswith(heading):
        return body
    return f"{heading}\n\n{body}"


def _make_chunk_id(source_name: str, clause_id: str) -> str:
    """
    Build a deterministic ChromaDB document ID for a chunk.

    Format: "<source_name>::<clause_id>"
    Determinism is what makes idempotency work — re-ingesting the
    same contract produces the same IDs, which ChromaDB can check cheaply.
    """
    safe_source = source_name.replace("\n", "_").replace(" ", "_")
    safe_clause = clause_id.replace("\n", "_")
    return f"{safe_source}::{safe_clause}"


# ──────────────────────────────────────────────────────────────────
# API call helpers (both task types live here)
# ──────────────────────────────────────────────────────────────────

def _embed_with_retry(
    client: genai.Client,
    texts: list[str],
    task_type: str,
) -> list[list[float]] | None:
    """
    Embed a batch of texts with exponential backoff on retriable errors.

    Handles both RETRIEVAL_DOCUMENT (indexing) and RETRIEVAL_QUERY
    (query time) — task_type is passed by the caller.

    Returns list of 768-dim float vectors, or None if all retries failed.
    """
    delay = RETRY_BASE_DELAY
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=EMBEDDING_DIM,
                ),
            )
            return [emb.values for emb in response.embeddings]

        except Exception as exc:
            err_str = str(exc).lower()
            is_retriable = any(
                x in err_str
                for x in ["429", "quota", "rate", "500", "503", "timeout"]
            )
            if is_retriable and attempt < MAX_RETRIES:
                logger.warning(
                    "Embed attempt %d/%d failed (%s). Retrying in %.1fs...",
                    attempt, MAX_RETRIES, exc, delay,
                )
                time.sleep(delay)
                delay *= 2
            else:
                logger.error("Embed failed after %d attempts: %s", attempt, exc)
                return None

    return None


# ──────────────────────────────────────────────────────────────────
# Query-time embedding — called by retriever.py
# ──────────────────────────────────────────────────────────────────

def embed_query(query_text: str) -> list[float]:
    """
    Embed a single user query for similarity search.

    Uses RETRIEVAL_QUERY task type — intentionally different from the
    RETRIEVAL_DOCUMENT task type used at index time. Google's model is
    trained on (query, document) pairs so the two types optimise the
    embedding space for the asymmetric retrieval direction.

    Args:
        query_text: The user's plain-English question.

    Returns:
        Embedding vector as a list of EMBEDDING_DIM floats.

    Raises:
        RuntimeError: If embedding fails after all retries.
    """
    client = _make_gemini_client()
    vectors = _embed_with_retry(client, [query_text], task_type="RETRIEVAL_QUERY")
    if vectors is None:
        raise RuntimeError(
            f"Failed to embed query after {MAX_RETRIES} retries. "
            "Check GOOGLE_API_KEY and network connectivity."
        )
    return vectors[0]


# ──────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────

def embed_and_store(
    chunks: list[Chunk],
    source_name: str,
    collection: Optional[chromadb.Collection] = None,
    conn: Optional[sqlite3.Connection] = None,
    chroma_dir: Path = DEFAULT_CHROMA_DIR,
    db_path: Path = DEFAULT_SQLITE_PATH,
) -> EmbedResult:
    """
    Embed all chunks and persist to ChromaDB + SQLite.

    Args:
        chunks:       Output of chunker.chunk_document()
        source_name:  Original contract filename — namespace key in both
                      stores so one collection holds many contracts.
        collection:   Optional pre-opened ChromaDB collection. If None,
                      one is opened via store.get_chroma_collection().
                      Pass one explicitly when calling in a loop to avoid
                      re-opening per contract.
        conn:         Optional pre-opened SQLite connection. Same rationale.
        chroma_dir:   Where to persist ChromaDB data.
        db_path:      Where to persist the SQLite metadata DB.

    Returns:
        EmbedResult with embedded/skipped/failed counts and elapsed time.
    """
    t_start = time.time()

    if collection is None:
        collection = get_chroma_collection(chroma_dir)
    if conn is None:
        conn = get_sqlite_connection(db_path)

    client = _make_gemini_client()

    # ── Idempotency: find which chunks already exist ─────────────
    # Append a suffix to duplicate clause_ids so every chunk gets a unique ID
    _id_counts: dict[str, int] = {}
    all_ids: list[str] = []
    for c in chunks:
        base_id = _make_chunk_id(source_name, c.clause_id)
        count = _id_counts.get(base_id, 0)
        _id_counts[base_id] = count + 1
        all_ids.append(f"{base_id}_{count}" if count > 0 else base_id)

    existing_ids: set[str] = set()
    unique_ids = list(dict.fromkeys(all_ids))  # preserve order, remove dupes
    for i in range(0, len(unique_ids), 100):
        result = collection.get(ids=unique_ids[i:i + 100], include=[])
        existing_ids.update(result["ids"])

    new_chunks = []
    new_ids = []
    for c, cid in zip(chunks, all_ids):
        if cid not in existing_ids:
            new_chunks.append(c)
            new_ids.append(cid)
    skipped = len(chunks) - len(new_chunks)

    if skipped:
        logger.info(
            "'%s': %d/%d chunks already embedded — skipping.",
            source_name, skipped, len(chunks)
        )

    if not new_chunks:
        elapsed = time.time() - t_start
        _log_ingestion(conn, source_name, len(chunks), 0, skipped, 0)
        return EmbedResult(
            source_name=source_name,
            total_chunks=len(chunks),
            embedded_count=0,
            skipped_count=skipped,
            failed_count=0,
            elapsed_seconds=elapsed,
        )

    # ── Batch embedding + persistence ────────────────────────────
    embedded_count = 0
    failed_count = 0

    for batch_start in range(0, len(new_chunks), EMBED_BATCH_SIZE):
        batch = new_chunks[batch_start: batch_start + EMBED_BATCH_SIZE]
        texts = [_build_embed_text(c) for c in batch]
        batch_ids = new_ids[batch_start: batch_start + EMBED_BATCH_SIZE]

        vectors = _embed_with_retry(client, texts, task_type="RETRIEVAL_DOCUMENT")

        if vectors is None:
            logger.error(
                "Batch %d-%d failed for '%s' — %d chunks NOT embedded.",
                batch_start + 1, batch_start + len(batch),
                source_name, len(batch),
            )
            failed_count += len(batch)
            time.sleep(INTER_BATCH_SLEEP * 4)
            continue

        # Write to ChromaDB
        collection.add(
            ids=batch_ids,
            embeddings=vectors,
            metadatas=[
                {
                    "source_name":  source_name,
                    "clause_id":    c.clause_id,
                    "heading":      c.heading,
                    "page_start":   c.page_start,
                    "page_end":     c.page_end,
                    "format_used":  c.format_used,
                    "char_count":   c.char_count,
                    # Short preview for display without a SQLite roundtrip
                    "text_preview": c.text[:200],
                }
                for c in batch
            ],
            # ChromaDB `documents` field: heading or clause_id for quick display.
            # Full text is authoritative in SQLite — not duplicated here.
            documents=[c.heading or c.clause_id for c in batch],
        )

        # Write to SQLite
        conn.executemany(
            """
            INSERT OR IGNORE INTO chunks
                (id, source_name, clause_id, heading, full_text,
                 page_start, page_end, format_used, char_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    cid,
                    source_name,
                    c.clause_id,
                    c.heading,
                    c.text,
                    c.page_start,
                    c.page_end,
                    c.format_used,
                    c.char_count,
                )
                for c, cid in zip(batch, batch_ids)
            ],
        )
        conn.commit()

        embedded_count += len(batch)
        logger.info(
            "'%s': embedded batch %d-%d (%d/%d total)",
            source_name,
            batch_start + 1,
            batch_start + len(batch),
            embedded_count,
            len(new_chunks),
        )

        if batch_start + EMBED_BATCH_SIZE < len(new_chunks):
            time.sleep(INTER_BATCH_SLEEP)

    elapsed = time.time() - t_start
    _log_ingestion(conn, source_name, len(chunks), embedded_count, skipped, failed_count)

    result = EmbedResult(
        source_name=source_name,
        total_chunks=len(chunks),
        embedded_count=embedded_count,
        skipped_count=skipped,
        failed_count=failed_count,
        elapsed_seconds=elapsed,
    )
    logger.info("embed_and_store complete: %s", result)
    return result


# ──────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────

def _log_ingestion(
    conn: sqlite3.Connection,
    source_name: str,
    total: int,
    embedded: int,
    skipped: int,
    failed: int,
) -> None:
    """Write one row to the ingestions audit log in SQLite."""
    conn.execute(
        """
        INSERT INTO ingestions (source_name, total_chunks, embedded, skipped, failed)
        VALUES (?, ?, ?, ?, ?)
        """,
        (source_name, total, embedded, skipped, failed),
    )
    conn.commit()


# ──────────────────────────────────────────────────────────────────
# Smoke test:
#   GOOGLE_API_KEY=your_key python src/pipeline/embedder.py path/to/contract.pdf
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from src.pipeline.parser import parse_pdf
    from src.pipeline.chunker import chunk_document

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if len(sys.argv) < 2:
        print("Usage: GOOGLE_API_KEY=<key> python src/pipeline/embedder.py <path_to_pdf>")
        sys.exit(1)

    path = sys.argv[1]

    print(f"\nParsing: {path}")
    parsed = parse_pdf(path)
    print(f"  -> {parsed.total_pages} pages, {parsed.total_word_count} words")

    print(f"\nChunking...")
    chunks = chunk_document(parsed, source=path)
    print(f"  -> {len(chunks)} chunks")

    print(f"\nEmbedding + storing...")
    result = embed_and_store(chunks, source_name=parsed.source_name)

    print(f"\nResult: {result}")
    print(f"  Success: {result.success}")

    from src.utils.store import get_sqlite_connection, list_ingested_contracts
    conn = get_sqlite_connection()
    contracts = list_ingested_contracts(conn)
    print(f"\nContracts in metadata store:")
    for c in contracts:
        print(f"  {c['source_name']}: {c['chunk_count']} chunks, ingested {c['last_ingested']}")
    conn.close()
