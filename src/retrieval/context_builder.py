"""
ClauseInsight — Context Builder
================================

Assembles the list[RetrievedChunk] from retriever.py into a clean,
structured BuiltContext object that both the Q&A engine and the risk
scanner can consume without knowing anything about retrieval internals.

PIPELINE POSITION
------------------
    retriever.py → context_builder.py → Q&A prompt (2_qa.py)
                                      → risk scanner (scanner.py)

WHY THIS MODULE EXISTS
-----------------------
retriever.py's job is finding the right chunks.
The Q&A page's job is generating an answer.
The scanner's job is classifying risk.

Without context_builder.py, both the Q&A page and the scanner would
need to know how to:
  - Deduplicate chunks that appear in multiple retrieval results
  - Compute token estimates to avoid blowing LLM context windows
  - Format citations consistently
  - Order chunks by page number for coherent reading order
  - Truncate chunks that are too long individually

context_builder.py owns all of that — one place, tested once.

OUTPUT: BuiltContext
---------------------
A BuiltContext carries:
  - chunks:           ordered list of RetrievedChunk (deduplicated)
  - context_text:     pre-formatted text block for direct LLM injection
  - citations:        list of citation strings for UI display
  - token_estimate:   rough token count (for context window checking)
  - source_names:     set of contract filenames represented
  - was_truncated:    True if any chunk was truncated to fit token budget

The LLM prompt itself is NOT built here — context_builder.py is
LLM-agnostic. The Q&A page and scanner each wrap context_text in
their own system/user prompt structure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .retriever import RetrievedChunk

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────

# Rough characters-per-token estimate for Gemini models.
# Gemini uses SentencePiece tokenisation — legal text averages ~4
# chars/token (slightly higher than general English due to defined terms).
CHARS_PER_TOKEN = 4

# Default token budget for the context block injected into an LLM prompt.
# gemini-2.0-flash context window: 1,048,576 tokens.
# We reserve a conservative 6,000 tokens for context — leaves plenty
# of room for the system prompt, user question, and LLM response.
DEFAULT_TOKEN_BUDGET = 6_000

# Separator between chunks in the formatted context_text block.
# Clear visual boundary helps the LLM distinguish clause boundaries.
CHUNK_SEPARATOR = "\n\n" + "─" * 60 + "\n\n"

# Max characters to show per chunk before truncation.
# At 4 chars/token, 2000 chars ≈ 500 tokens per chunk.
# With TOP_K=5 chunks: 5 × 500 = 2,500 tokens max for context.
MAX_CHARS_PER_CHUNK = 2_000


# ──────────────────────────────────────────────────────────────────
# Output data structure
# ──────────────────────────────────────────────────────────────────

@dataclass
class BuiltContext:
    """
    Structured context assembled from retrieved chunks.

    This is what flows into the Q&A engine and risk scanner —
    neither module needs to know anything about retrieval internals.

    Attributes
    ----------
    chunks:
        Deduplicated, page-ordered list of RetrievedChunk objects.
        Use these when you need per-chunk metadata (score, format, etc.)

    context_text:
        Pre-formatted string ready for LLM injection. Each chunk is
        presented with its citation header, full text, and separated
        by a visual divider. Inject this directly into your prompt.

    citations:
        Ordered list of citation strings (one per chunk) for display
        in the Streamlit UI alongside the LLM's answer.
        e.g. ["Section 4 (agency.pdf, pp. 8–10)", "Clause 1(a) (oneNDA.pdf, p. 2)"]

    token_estimate:
        Rough estimate of how many tokens context_text will consume.
        Use this to check against your LLM's context window before sending.

    source_names:
        Set of contract filenames represented in this context.
        Single-element for scoped queries, multi-element for global.

    was_truncated:
        True if one or more chunks were truncated to fit the token budget.
        The Q&A page can surface a warning to the user when this is True.

    query:
        The original user query that produced these chunks.
        Stored here so the Q&A page doesn't need to pass it separately.
    """

    chunks: list[RetrievedChunk]
    context_text: str
    citations: list[str]
    token_estimate: int
    source_names: set[str]
    was_truncated: bool
    query: str

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    @property
    def is_empty(self) -> bool:
        return len(self.chunks) == 0

    @property
    def is_multi_contract(self) -> bool:
        """True if chunks came from more than one contract."""
        return len(self.source_names) > 1

    def __repr__(self) -> str:
        return (
            f"BuiltContext("
            f"chunks={self.chunk_count}, "
            f"tokens≈{self.token_estimate}, "
            f"sources={self.source_names}, "
            f"truncated={self.was_truncated})"
        )


# ──────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────

def build_context(
    chunks: list[RetrievedChunk],
    query: str,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    max_chars_per_chunk: int = MAX_CHARS_PER_CHUNK,
) -> BuiltContext:
    """
    Assemble retrieved chunks into a structured BuiltContext.

    Steps:
      1. Deduplicate — remove chunks with identical clause_id + source
      2. Sort — by page number for coherent reading order
      3. Truncate — clip chunks that exceed max_chars_per_chunk
      4. Budget check — drop trailing chunks if total exceeds token_budget
      5. Format — build context_text and citations list

    Args:
        chunks:              Output of retriever.retrieve() or
                             retriever.retrieve_for_contract().
        query:               The original user query (stored in output).
        token_budget:        Max tokens for the context_text block.
                             Default 6,000 — safe for gemini-2.0-flash.
        max_chars_per_chunk: Max characters per chunk before truncation.
                             Default 2,000 chars ≈ 500 tokens.

    Returns:
        BuiltContext with all fields populated.
        Returns an empty BuiltContext (is_empty=True) if chunks is empty.
    """
    if not chunks:
        logger.info("build_context called with 0 chunks for query: %r", query[:60])
        return _empty_context(query)

    # ── Step 1: Deduplicate ──────────────────────────────────────
    # retriever.py already deduplicates by ChromaDB ID, but defensive
    # dedup here guards against callers passing chunks from multiple
    # retrieve() calls (e.g. the "compare two contracts" feature).
    seen: set[str] = set()
    unique_chunks: list[RetrievedChunk] = []
    for chunk in chunks:
        key = chunk.chunk_id
        if key not in seen:
            seen.add(key)
            unique_chunks.append(chunk)

    dupes_removed = len(chunks) - len(unique_chunks)
    if dupes_removed:
        logger.debug("build_context: removed %d duplicate chunks", dupes_removed)

    # ── Step 2: Sort by page number ──────────────────────────────
    # MMR ranking is relevance-ordered, but for LLM context it's
    # better to present clauses in document reading order so the
    # model can follow clause cross-references (e.g. "as defined in
    # Section 2 above"). Tie-break on clause_id for determinism.
    sorted_chunks = sorted(
        unique_chunks,
        key=lambda c: (c.source_name, c.page_start, c.clause_id)
    )

    # ── Step 3 + 4: Truncate + budget enforcement ────────────────
    budget_chars = token_budget * CHARS_PER_TOKEN
    was_truncated = False
    accepted: list[tuple[RetrievedChunk, str]] = []  # (chunk, text_to_use)
    chars_used = 0

    for chunk in sorted_chunks:
        body = chunk.full_text.strip()

        # Truncate oversized individual chunks
        if len(body) > max_chars_per_chunk:
            body = body[:max_chars_per_chunk] + "\n[... truncated for context window ...]"
            was_truncated = True

        # Build the formatted block for this chunk
        chunk_block = _format_chunk_block(chunk, body)
        block_chars = len(chunk_block)

        # Check if adding this chunk would exceed the token budget
        separator_chars = len(CHUNK_SEPARATOR) if accepted else 0
        if chars_used + separator_chars + block_chars > budget_chars:
            logger.info(
                "build_context: token budget reached after %d/%d chunks "
                "(budget: %d tokens, used: ~%d tokens)",
                len(accepted), len(sorted_chunks),
                token_budget, chars_used // CHARS_PER_TOKEN,
            )
            was_truncated = True
            break

        accepted.append((chunk, chunk_block))
        chars_used += separator_chars + block_chars

    if not accepted:
        # All chunks exceeded budget individually — take first chunk truncated hard
        chunk = sorted_chunks[0]
        body = chunk.full_text.strip()[:max_chars_per_chunk // 2]
        body += "\n[... truncated for context window ...]"
        chunk_block = _format_chunk_block(chunk, body)
        accepted = [(chunk, chunk_block)]
        was_truncated = True
        logger.warning(
            "build_context: all chunks exceeded budget — forcing first chunk truncated."
        )

    # ── Step 5: Assemble output ───────────────────────────────────
    final_chunks = [c for c, _ in accepted]
    blocks = [b for _, b in accepted]

    context_text = _build_context_header(query) + CHUNK_SEPARATOR.join(blocks)
    citations = [c.citation for c in final_chunks]
    source_names = {c.source_name for c in final_chunks}
    token_estimate = len(context_text) // CHARS_PER_TOKEN

    result = BuiltContext(
        chunks=final_chunks,
        context_text=context_text,
        citations=citations,
        token_estimate=token_estimate,
        source_names=source_names,
        was_truncated=was_truncated,
        query=query,
    )

    logger.info(
        "build_context: %d chunks, ~%d tokens, sources=%s, truncated=%s",
        result.chunk_count, result.token_estimate,
        result.source_names, result.was_truncated,
    )
    return result


# ──────────────────────────────────────────────────────────────────
# Convenience wrappers
# ──────────────────────────────────────────────────────────────────

def build_qa_context(
    chunks: list[RetrievedChunk],
    query: str,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
) -> BuiltContext:
    """
    Build context optimised for the Q&A engine.

    Identical to build_context() with Q&A-appropriate defaults.
    Named separately so 2_qa.py reads clearly without needing to
    remember to pass specific parameters.

    Args:
        chunks:       Retrieved chunks from retriever.retrieve().
        query:        User's plain-English question.
        token_budget: Token budget for context. Default 6,000.

    Returns:
        BuiltContext ready for Q&A prompt construction in 2_qa.py.
    """
    return build_context(chunks=chunks, query=query, token_budget=token_budget)


def build_scanner_context(
    chunks: list[RetrievedChunk],
    token_budget: int = DEFAULT_TOKEN_BUDGET,
) -> BuiltContext:
    """
    Build context optimised for the risk scanner.

    The scanner doesn't have a user query — it processes all clauses
    in a contract systematically. We pass an empty query string and
    a slightly larger token budget since scanner batches can be bigger.

    Args:
        chunks:       Chunks to classify — typically all chunks for
                      one contract from store.get_all_chunks_for_contract().
        token_budget: Token budget. Default 6,000.

    Returns:
        BuiltContext for scanner.py to wrap in its classification prompt.
    """
    return build_context(
        chunks=chunks,
        query="",   # scanner doesn't have a user query
        token_budget=token_budget,
    )


# ──────────────────────────────────────────────────────────────────
# Formatting helpers
# ──────────────────────────────────────────────────────────────────

def _format_chunk_block(chunk: RetrievedChunk, body: str) -> str:
    """
    Format a single chunk into the text block that goes into context_text.

    Format:
        [CLAUSE: Section 4 | agency_agreement.pdf | pp. 8-10 | score: 0.847]
        <clause text>

    The header line gives the LLM explicit provenance for each clause
    so it can cite correctly without hallucinating page numbers.
    """
    pages = (
        f"p. {chunk.page_start}"
        if chunk.page_start == chunk.page_end
        else f"pp. {chunk.page_start}–{chunk.page_end}"
    )
    header = (
        f"[CLAUSE: {chunk.clause_id} | "
        f"{chunk.source_name} | "
        f"{pages} | "
        f"relevance: {chunk.similarity_score:.3f}]"
    )
    return f"{header}\n{body}"


def _build_context_header(query: str) -> str:
    """
    Optional header block prepended to the full context_text.

    Tells the LLM what the context contains and what the user asked,
    so it has framing before reading the clauses.
    Omitted (empty string) when query is empty (scanner use case).
    """
    if not query:
        return ""
    return (
        f"The following contract clauses are relevant to the question:\n"
        f"QUESTION: {query}\n\n"
        f"CONTRACT CLAUSES:\n\n"
    )


def _empty_context(query: str) -> BuiltContext:
    """Return a well-formed empty BuiltContext when no chunks are available."""
    return BuiltContext(
        chunks=[],
        context_text="",
        citations=[],
        token_estimate=0,
        source_names=set(),
        was_truncated=False,
        query=query,
    )
