"""
ClauseInsight — Retriever
==========================

Given a plain-English user query, finds the most relevant contract
clauses from ChromaDB using vector similarity search, then reranks
the candidates using Maximal Marginal Relevance (MMR) to remove
redundant chunks before handing results to context_builder.py.

PIPELINE POSITION
------------------
    embedder.py  →  [ChromaDB]  →  retriever.py  →  context_builder.py
                                         ↑
                                    (this file)

RETRIEVAL STRATEGY: TWO-STAGE
-------------------------------
Stage 1 — Candidate fetch (pure similarity):
    Query ChromaDB with the embedded user query.
    Fetch MMR_CANDIDATE_POOL candidates (default 15) — 3× the final
    TOP_K (5) so MMR has enough diversity to select from.
    Candidates include their stored embeddings so MMR can compute
    inter-chunk similarity without extra API calls.

Stage 2 — MMR reranking:
    Maximal Marginal Relevance selects chunks that are:
      (a) relevant to the query, AND
      (b) not too similar to each other

    The MMR score for each remaining candidate at each selection step:
        MMR(c) = λ · sim(query, c) - (1 - λ) · max_sim(c, already_selected)

    where sim() is cosine similarity and λ (MMR_LAMBDA) controls the
    relevance/diversity trade-off:
        λ = 1.0 → pure similarity (same as no MMR)
        λ = 0.5 → equal weight to relevance and diversity
        λ = 0.7 → slightly relevance-favoured (our default)

    Why 0.7? Legal Q&A benefits more from relevance than diversity —
    if three chunks all mention "indemnification", they're probably
    all genuinely relevant to an indemnification question. But at 0.7
    (not 1.0) we still filter near-duplicate sub-clauses that add
    noise without adding information.

GLOBAL SEARCH
--------------
No source_name filter is applied — results can come from any ingested
contract. Each RetrievedChunk carries source_name + page metadata so
context_builder.py and the UI know which contract each clause came from.

This enables cross-contract queries like "which contract has the
strongest indemnification clause?" — something scoped retrieval
cannot answer.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import chromadb

from pathlib import Path
import sys as _sys
_src_dir = str(Path(__file__).resolve().parent.parent)
if _src_dir not in _sys.path:
    _sys.path.insert(0, _src_dir)

from utils.store import (
    DEFAULT_CHROMA_DIR,
    DEFAULT_SQLITE_PATH,
    get_chroma_collection,
    get_sqlite_connection,
    get_chunk_by_id,
)
from pipeline.embedder import embed_query

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────

# Final number of chunks returned to context_builder.py per query
TOP_K = 5

# Candidate pool size for MMR: fetch 3× TOP_K so MMR has diversity to work with.
# More candidates = better MMR diversity but slightly slower ChromaDB query.
MMR_CANDIDATE_POOL = TOP_K * 3  # 15

# MMR lambda: relevance vs diversity trade-off.
# 0.7 = slightly relevance-favoured, still filters near-duplicate sub-clauses.
# Range: 0.0 (pure diversity) → 1.0 (pure similarity, no MMR effect)
MMR_LAMBDA = 0.7

# Minimum similarity score (1 - cosine_distance) to include a chunk.
# Chunks below this threshold are too dissimilar to be useful even
# if they're in the top-K — better to return fewer, higher-quality results.
MIN_SIMILARITY_THRESHOLD = 0.3


# ──────────────────────────────────────────────────────────────────
# Output data structure
# ──────────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """
    A single clause chunk returned by the retriever.

    Carries everything context_builder.py and the UI need:
      - The clause text for LLM grounding
      - Citation metadata (source, clause ID, pages)
      - Similarity score for display / debugging
      - The MMR rank (1 = most relevant after reranking)
    """
    chunk_id: str           # "<source_name>::<clause_id>"
    clause_id: str          # e.g. "Section 4", "Clause 1(a)"
    source_name: str        # original contract filename
    heading: str
    full_text: str          # fetched from SQLite (authoritative)
    page_start: int
    page_end: int
    format_used: str
    similarity_score: float  # cosine similarity to query (0-1, higher = more relevant)
    mmr_rank: int            # 1-indexed rank after MMR reranking
    text_preview: str = field(default="")  # first 200 chars, for UI display

    def __post_init__(self) -> None:
        if not self.text_preview and self.full_text:
            self.text_preview = self.full_text[:200]

    @property
    def citation(self) -> str:
        """
        Human-readable citation string for display in the UI and LLM prompts.
        e.g. "Section 4 (agency_agreement.pdf, pp. 8-10)"
        """
        pages = (
            f"p. {self.page_start}"
            if self.page_start == self.page_end
            else f"pp. {self.page_start}–{self.page_end}"
        )
        return f"{self.clause_id} ({self.source_name}, {pages})"

    def __repr__(self) -> str:
        return (
            f"RetrievedChunk(rank={self.mmr_rank}, score={self.similarity_score:.3f}, "
            f"id={self.chunk_id!r}, pages={self.page_start}-{self.page_end})"
        )


# ──────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────

def retrieve(
    query: str,
    top_k: int = TOP_K,
    mmr_lambda: float = MMR_LAMBDA,
    source_name: Optional[str] = None,
    collection: Optional[chromadb.Collection] = None,
    chroma_dir=DEFAULT_CHROMA_DIR,
    db_path=DEFAULT_SQLITE_PATH,
) -> list[RetrievedChunk]:
    """
    Find the most relevant contract clauses for a plain-English query.

    Two-stage pipeline:
      1. Embed the query → fetch MMR_CANDIDATE_POOL chunks from ChromaDB
      2. MMR rerank → return top_k diverse, relevant chunks

    Full clause text is fetched from SQLite (not ChromaDB) to get the
    authoritative, complete text that was stored at ingestion time.

    Args:
        query:       Plain-English question from the user.
        top_k:       Number of chunks to return after MMR. Default 5.
        mmr_lambda:  MMR relevance/diversity trade-off. Default 0.7.
        source_name: Optional contract filename to scope search to one
                     contract. None (default) = global search across all.
        collection:  Pre-opened ChromaDB collection. If None, opens one
                     from chroma_dir. Pass explicitly in the UI to avoid
                     re-opening on every query.
        chroma_dir:  ChromaDB storage directory.
        db_path:     SQLite metadata DB path.

    Returns:
        List of RetrievedChunk, ordered by MMR rank (best first).
        May return fewer than top_k if the collection has fewer chunks
        or if candidates fall below MIN_SIMILARITY_THRESHOLD.
    """
    if not query or not query.strip():
        logger.warning("retrieve() called with empty query — returning []")
        return []

    if collection is None:
        collection = get_chroma_collection(chroma_dir)

    conn = get_sqlite_connection(db_path)

    try:
        # ── Stage 1: Embed query + fetch candidates ───────────────
        query_vector = embed_query(query.strip())

        candidate_pool = min(MMR_CANDIDATE_POOL, collection.count())
        if candidate_pool == 0:
            logger.warning("ChromaDB collection is empty — no chunks to retrieve.")
            return []

        # Build optional where filter for per-contract scoping
        where_filter = {"source_name": source_name} if source_name else None

        raw = collection.query(
            query_embeddings=[query_vector],
            n_results=candidate_pool,
            where=where_filter,
            include=["embeddings", "metadatas", "distances"],
        )

        # Unpack ChromaDB result (results are wrapped in a list because
        # ChromaDB supports batched queries — we sent one query, so [0])
        ids        = raw["ids"][0]
        embeddings = raw["embeddings"][0]   # list of 768-dim vectors
        metadatas  = raw["metadatas"][0]
        distances  = raw["distances"][0]    # cosine distances (0=identical, 2=opposite)

        if not ids:
            logger.info("No candidates returned for query: %r", query[:80])
            return []

        # Convert cosine distance → similarity score (0-1)
        # ChromaDB cosine distance = 1 - cosine_similarity, so:
        # similarity = 1 - distance
        similarities = [max(0.0, 1.0 - d) for d in distances]

        # Apply minimum similarity threshold — filter weak candidates
        # before MMR so we don't waste reranking slots on poor matches
        candidates = [
            (cid, emb, meta, sim)
            for cid, emb, meta, sim in zip(ids, embeddings, metadatas, similarities)
            if sim >= MIN_SIMILARITY_THRESHOLD
        ]

        if not candidates:
            logger.info(
                "All %d candidates fell below similarity threshold %.2f for query: %r",
                len(ids), MIN_SIMILARITY_THRESHOLD, query[:80]
            )
            return []

        logger.info(
            "Query: %r → %d candidates (threshold filtered: %d)",
            query[:60], len(candidates), len(ids) - len(candidates)
        )

        # ── Stage 2: MMR reranking ────────────────────────────────
        selected_ids = _mmr_select(
            query_vector=query_vector,
            candidates=candidates,
            top_k=min(top_k, len(candidates)),
            lambda_=mmr_lambda,
        )

        # ── Fetch full text from SQLite + build RetrievedChunk objects ──
        results: list[RetrievedChunk] = []
        for rank, (chunk_id, sim) in enumerate(selected_ids, start=1):
            row = get_chunk_by_id(chunk_id, conn)
            if row is None:
                # ChromaDB and SQLite are out of sync — log and skip
                logger.warning(
                    "Chunk ID %r found in ChromaDB but missing from SQLite. "
                    "Re-ingest the contract to fix this.",
                    chunk_id
                )
                continue

            results.append(RetrievedChunk(
                chunk_id=chunk_id,
                clause_id=row["clause_id"],
                source_name=row["source_name"],
                heading=row["heading"] or "",
                full_text=row["full_text"] or "",
                page_start=row["page_start"],
                page_end=row["page_end"],
                format_used=row["format_used"] or "",
                similarity_score=sim,
                mmr_rank=rank,
            ))

        logger.info(
            "retrieve() → %d results for query: %r",
            len(results), query[:60]
        )
        return results

    finally:
        conn.close()


# ──────────────────────────────────────────────────────────────────
# MMR implementation
# ──────────────────────────────────────────────────────────────────

def _mmr_select(
    query_vector: list[float],
    candidates: list[tuple[str, list[float], dict, float]],
    top_k: int,
    lambda_: float,
) -> list[tuple[str, float]]:
    """
    Maximal Marginal Relevance selection.

    Iteratively selects the chunk that maximises:
        MMR(c) = λ · sim(query, c) - (1 - λ) · max_sim(c, selected)

    At each step:
      - sim(query, c)       = cosine similarity to the query vector
                              (pre-computed as the `similarity` field)
      - max_sim(c, selected) = maximum cosine similarity between c
                               and any already-selected chunk
                               (computed from stored embeddings)

    This ensures each new chunk adds information not already covered
    by the previously selected chunks.

    Args:
        query_vector: The embedded query (768-dim float list).
        candidates:   List of (chunk_id, embedding, metadata, similarity).
                      similarity is pre-computed cosine sim to query.
        top_k:        Number of chunks to select.
        lambda_:      MMR lambda parameter (0.7 = relevance-favoured).

    Returns:
        List of (chunk_id, similarity_score) tuples, in MMR selection order.
    """
    if not candidates:
        return []

    selected: list[tuple[str, float]] = []          # (chunk_id, sim_to_query)
    selected_embeddings: list[list[float]] = []     # for inter-chunk sim computation
    remaining = list(candidates)                    # mutable working set

    while len(selected) < top_k and remaining:
        best_score = -math.inf
        best_idx = 0

        for i, (cid, emb, _meta, sim_to_query) in enumerate(remaining):
            # Relevance term: similarity to the query
            relevance = sim_to_query

            # Redundancy term: max similarity to any already-selected chunk
            if not selected_embeddings:
                # First selection — no redundancy penalty yet
                redundancy = 0.0
            else:
                redundancy = max(
                    _cosine_similarity(emb, sel_emb)
                    for sel_emb in selected_embeddings
                )

            mmr_score = lambda_ * relevance - (1 - lambda_) * redundancy

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i

        # Select the best candidate and remove from remaining pool
        chosen_id, chosen_emb, _, chosen_sim = remaining.pop(best_idx)
        selected.append((chosen_id, chosen_sim))
        selected_embeddings.append(chosen_emb)

        logger.debug(
            "MMR step %d: selected %r (sim=%.3f, mmr_score=%.3f)",
            len(selected), chosen_id, chosen_sim, best_score
        )

    return selected


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Cosine similarity between two vectors.

    Implemented from scratch rather than using numpy to keep this module
    dependency-light — retriever.py already depends on chromadb and
    google-genai, no need to add numpy just for this one operation.

    Returns a value in [-1, 1]. In practice for text embeddings from
    the same model, values are in [0, 1] since text embeddings are
    non-negative in the semantic similarity space.

    Returns 0.0 for zero vectors (avoids division by zero).
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (norm_a * norm_b)


# ──────────────────────────────────────────────────────────────────
# Convenience wrapper: retrieve for a single contract
# ──────────────────────────────────────────────────────────────────

def retrieve_for_contract(
    query: str,
    source_name: str,
    top_k: int = TOP_K,
    mmr_lambda: float = MMR_LAMBDA,
    collection: Optional[chromadb.Collection] = None,
    chroma_dir=DEFAULT_CHROMA_DIR,
    db_path=DEFAULT_SQLITE_PATH,
) -> list[RetrievedChunk]:
    """
    Retrieve chunks scoped to a single contract.

    Thin wrapper around retrieve() with source_name set — provided
    as a named function so the UI pages can call it without having
    to remember to pass source_name explicitly.

    Args:
        query:       Plain-English question.
        source_name: Contract filename to scope search to.
        top_k:       Number of results. Default 5.
        mmr_lambda:  MMR trade-off. Default 0.7.
        collection:  Pre-opened ChromaDB collection (optional).
        chroma_dir:  ChromaDB storage path.
        db_path:     SQLite metadata DB path.

    Returns:
        List of RetrievedChunk scoped to the given contract.
    """
    return retrieve(
        query=query,
        top_k=top_k,
        mmr_lambda=mmr_lambda,
        source_name=source_name,
        collection=collection,
        chroma_dir=chroma_dir,
        db_path=db_path,
    )


# ──────────────────────────────────────────────────────────────────
# Smoke test:
#   GOOGLE_API_KEY=your_key python src/retrieval/retriever.py
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    query = sys.argv[1] if len(sys.argv) > 1 else "What are the indemnification obligations?"

    print(f"\nQuery: {query!r}")
    print(f"Retrieving top {TOP_K} chunks with MMR (lambda={MMR_LAMBDA})...\n")

    results = retrieve(query)

    if not results:
        print("No results found. Make sure contracts have been ingested first.")
        sys.exit(0)

    for chunk in results:
        print(f"Rank {chunk.mmr_rank} | Score: {chunk.similarity_score:.3f}")
        print(f"  Citation : {chunk.citation}")
        print(f"  Format   : {chunk.format_used}")
        print(f"  Preview  : {chunk.text_preview[:120]}...")
        print()
