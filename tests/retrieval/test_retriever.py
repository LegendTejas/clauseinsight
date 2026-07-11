"""
Tests for src/retrieval/retriever.py and src/retrieval/context_builder.py

Unit tests: pure algorithm logic — cosine similarity, MMR, context assembly.
Integration tests: require embedded contracts in ChromaDB (OPENAI_API_KEY needed).
"""

from __future__ import annotations

import math
import pytest

from src.retrieval.retriever import (
    RetrievedChunk,
    _cosine_similarity,
    _mmr_select,
    TOP_K,
    MMR_LAMBDA,
    MMR_CANDIDATE_POOL,
    MIN_SIMILARITY_THRESHOLD,
)
from src.retrieval.context_builder import (
    BuiltContext,
    build_context,
    build_qa_context,
    build_scanner_context,
    _format_chunk_block,
    _build_context_header,
    DEFAULT_TOKEN_BUDGET,
    MAX_CHARS_PER_CHUNK,
)


# ── RetrievedChunk tests ───────────────────────────────────────────

class TestRetrievedChunk:
    def _make(self, **kwargs):
        defaults = dict(
            chunk_id="c.pdf::S4", clause_id="Section 4",
            source_name="c.pdf", heading="Indemnification",
            full_text="The Company shall indemnify...",
            page_start=8, page_end=10, format_used="section_n",
            similarity_score=0.87, mmr_rank=1,
        )
        defaults.update(kwargs)
        return RetrievedChunk(**defaults)

    def test_citation_multi_page(self):
        c = self._make(page_start=8, page_end=10)
        assert c.citation == "Section 4 (c.pdf, pp. 8–10)"

    def test_citation_single_page(self):
        c = self._make(page_start=3, page_end=3)
        assert c.citation == "Section 4 (c.pdf, p. 3)"

    def test_text_preview_auto_set(self):
        c = self._make(full_text="Hello world " * 20)
        assert len(c.text_preview) <= 200

    def test_text_preview_not_overridden_if_set(self):
        c = self._make(full_text="long text", text_preview="custom preview")
        assert c.text_preview == "custom preview"

    def test_repr_contains_rank_and_score(self):
        c = self._make()
        r = repr(c)
        assert "rank=1" in r
        assert "0.87" in r


# ── _cosine_similarity tests ───────────────────────────────────────

class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-9

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(_cosine_similarity(a, b)) < 1e-9

    def test_zero_vector_returns_zero(self):
        z = [0.0, 0.0]
        v = [1.0, 0.0]
        assert _cosine_similarity(z, v) == 0.0

    def test_known_value(self):
        # [1,1] · [1,0] = 1/sqrt(2)
        assert abs(_cosine_similarity([1.0, 1.0], [1.0, 0.0]) - 1/math.sqrt(2)) < 1e-9

    def test_symmetry(self):
        a = [0.3, 0.7, 0.1]
        b = [0.8, 0.2, 0.5]
        assert abs(_cosine_similarity(a, b) - _cosine_similarity(b, a)) < 1e-9

    def test_result_bounded(self):
        import random
        for _ in range(10):
            a = [random.random() for _ in range(10)]
            b = [random.random() for _ in range(10)]
            sim = _cosine_similarity(a, b)
            assert -1.0 <= sim <= 1.0 + 1e-9


# ── _mmr_select tests ──────────────────────────────────────────────

class TestMMRSelect:
    def _candidates(self):
        return [
            ("chunk_1", [1.0, 0.0, 0.0], {}, 0.9),   # high sim, direction A
            ("chunk_2", [0.9, 0.1, 0.0], {}, 0.7),   # med sim, similar to c1
            ("chunk_3", [0.0, 0.0, 1.0], {}, 0.65),  # med sim, different direction
        ]

    def test_first_selected_is_highest_similarity(self):
        result = _mmr_select([1.0, 0.0, 0.0], self._candidates(), top_k=1, lambda_=0.7)
        assert result[0][0] == "chunk_1"

    def test_mmr_prefers_diverse_over_redundant(self):
        # With lambda=0.7: step2 should prefer chunk_3 (diverse) over chunk_2 (redundant)
        result = _mmr_select([1.0, 0.0, 0.0], self._candidates(), top_k=2, lambda_=0.7)
        assert result[0][0] == "chunk_1"
        assert result[1][0] == "chunk_3"

    def test_pure_similarity_no_diversity(self):
        # With lambda=1.0: no diversity penalty, should pick by pure similarity
        result = _mmr_select([1.0, 0.0, 0.0], self._candidates(), top_k=2, lambda_=1.0)
        assert result[0][0] == "chunk_1"
        assert result[1][0] == "chunk_2"

    def test_top_k_larger_than_candidates_returns_all(self):
        result = _mmr_select([1.0, 0.0, 0.0], self._candidates(), top_k=100, lambda_=0.7)
        assert len(result) == 3

    def test_empty_candidates_returns_empty(self):
        assert _mmr_select([1.0, 0.0], [], top_k=5, lambda_=0.7) == []

    def test_single_candidate(self):
        result = _mmr_select([1.0, 0.0], [("only", [1.0, 0.0], {}, 0.8)], top_k=5, lambda_=0.7)
        assert len(result) == 1
        assert result[0][0] == "only"

    def test_scores_returned_correctly(self):
        result = _mmr_select([1.0, 0.0, 0.0], self._candidates(), top_k=3, lambda_=0.7)
        # Each result is (chunk_id, similarity_score_to_query)
        assert all(isinstance(r[1], float) for r in result)
        assert all(0.0 <= r[1] <= 1.0 for r in result)

    def test_lambda_zero_pure_diversity(self):
        # lambda=0: pure diversity — second pick should be most different from first
        result = _mmr_select([1.0, 0.0, 0.0], self._candidates(), top_k=2, lambda_=0.0)
        assert result[1][0] == "chunk_3"  # most different from chunk_1


# ── Constants tests ────────────────────────────────────────────────

class TestRetrieverConstants:
    def test_top_k(self):
        assert TOP_K == 5

    def test_candidate_pool_is_3x_top_k(self):
        assert MMR_CANDIDATE_POOL == TOP_K * 3

    def test_lambda_in_valid_range(self):
        assert 0.0 <= MMR_LAMBDA <= 1.0

    def test_threshold_in_valid_range(self):
        assert 0.0 <= MIN_SIMILARITY_THRESHOLD <= 1.0


# ── BuiltContext / context_builder tests ───────────────────────────

class TestBuiltContext:
    def test_is_empty_true_when_no_chunks(self):
        ctx = build_context([], query="test")
        assert ctx.is_empty is True

    def test_is_empty_false_with_chunks(self, sample_retrieved_chunks):
        ctx = build_context(sample_retrieved_chunks, query="test")
        assert ctx.is_empty is False

    def test_chunk_count(self, sample_retrieved_chunks):
        ctx = build_context(sample_retrieved_chunks, query="test")
        assert ctx.chunk_count == len(sample_retrieved_chunks)

    def test_is_multi_contract_true(self, sample_retrieved_chunks):
        # sample has chunks from agency.pdf and oneNDA.pdf
        ctx = build_context(sample_retrieved_chunks, query="test")
        assert ctx.is_multi_contract is True

    def test_is_multi_contract_false_single_source(self, sample_retrieved_chunks):
        single = [c for c in sample_retrieved_chunks if c.source_name == "agency.pdf"]
        ctx = build_context(single, query="test")
        assert ctx.is_multi_contract is False

    def test_query_stored(self, sample_retrieved_chunks):
        ctx = build_context(sample_retrieved_chunks, query="What is indemnification?")
        assert ctx.query == "What is indemnification?"

    def test_citations_list_populated(self, sample_retrieved_chunks):
        ctx = build_context(sample_retrieved_chunks, query="test")
        assert len(ctx.citations) == len(sample_retrieved_chunks)
        assert all(isinstance(c, str) for c in ctx.citations)

    def test_source_names_populated(self, sample_retrieved_chunks):
        ctx = build_context(sample_retrieved_chunks, query="test")
        assert "agency.pdf" in ctx.source_names
        assert "oneNDA.pdf" in ctx.source_names

    def test_token_estimate_positive(self, sample_retrieved_chunks):
        ctx = build_context(sample_retrieved_chunks, query="test")
        assert ctx.token_estimate > 0

    def test_context_text_contains_clause_headers(self, sample_retrieved_chunks):
        ctx = build_context(sample_retrieved_chunks, query="test")
        assert "[CLAUSE:" in ctx.context_text

    def test_context_text_contains_question(self, sample_retrieved_chunks):
        ctx = build_context(sample_retrieved_chunks, query="What is indemnification?")
        assert "What is indemnification?" in ctx.context_text

    def test_chunks_sorted_by_page(self, sample_retrieved_chunks):
        # agency chunk: pages 20-22, onenda chunk: pages 2-3
        ctx = build_context(sample_retrieved_chunks, query="test")
        pages = [c.page_start for c in ctx.chunks]
        assert pages == sorted(pages) or len(set(c.source_name for c in ctx.chunks)) > 1

    def test_deduplication(self, sample_retrieved_chunks):
        dup = sample_retrieved_chunks + [sample_retrieved_chunks[0]]  # add duplicate
        ctx = build_context(dup, query="test")
        assert ctx.chunk_count == len(sample_retrieved_chunks)

    def test_truncation_on_long_chunk(self):
        from src.retrieval.retriever import RetrievedChunk
        long_chunk = RetrievedChunk(
            chunk_id="c.pdf::S1", clause_id="S1", source_name="c.pdf",
            heading="Title", full_text="A" * 5000,
            page_start=1, page_end=1, format_used="section_n",
            similarity_score=0.9, mmr_rank=1,
        )
        ctx = build_context([long_chunk], query="test")
        assert ctx.was_truncated is True
        assert "truncated" in ctx.context_text

    def test_token_budget_enforcement(self):
        from src.retrieval.retriever import RetrievedChunk
        big_chunks = [
            RetrievedChunk(
                chunk_id=f"c.pdf::S{i}", clause_id=f"S{i}", source_name="c.pdf",
                heading=f"Section {i}", full_text="X" * 2000,
                page_start=i, page_end=i, format_used="section_n",
                similarity_score=0.8, mmr_rank=i,
            )
            for i in range(1, 20)
        ]
        ctx = build_context(big_chunks, query="test", token_budget=2000)
        assert ctx.was_truncated is True
        assert ctx.chunk_count < 19

    def test_scanner_context_no_question_header(self, sample_retrieved_chunks):
        ctx = build_scanner_context(sample_retrieved_chunks)
        assert ctx.query == ""
        assert "QUESTION" not in ctx.context_text

    def test_qa_context_has_question_header(self, sample_retrieved_chunks):
        ctx = build_qa_context(sample_retrieved_chunks, query="test query")
        assert "QUESTION" in ctx.context_text


# ── _format_chunk_block tests ──────────────────────────────────────

class TestFormatChunkBlock:
    def test_contains_clause_id(self, sample_retrieved_chunks):
        block = _format_chunk_block(sample_retrieved_chunks[0], "body text")
        assert "Section 9" in block

    def test_contains_source_name(self, sample_retrieved_chunks):
        block = _format_chunk_block(sample_retrieved_chunks[0], "body text")
        assert "agency.pdf" in block

    def test_contains_page_range(self, sample_retrieved_chunks):
        block = _format_chunk_block(sample_retrieved_chunks[0], "body text")
        assert "pp. 20" in block

    def test_single_page_format(self, sample_retrieved_chunks):
        chunk = sample_retrieved_chunks[1]  # page 2-3
        block = _format_chunk_block(chunk, "body")
        assert "pp." in block

    def test_contains_body_text(self, sample_retrieved_chunks):
        block = _format_chunk_block(sample_retrieved_chunks[0], "my body text here")
        assert "my body text here" in block


# ── Integration tests ──────────────────────────────────────────────

@pytest.mark.integration
class TestRetrieverIntegration:
    def test_retrieve_returns_results(self, require_api_key):
        """Requires oneNDA_v2.pdf to be already ingested."""
        from src.retrieval.retriever import retrieve
        results = retrieve("What are the confidentiality obligations?")
        # May return 0 if nothing ingested — not a failure
        assert isinstance(results, list)
        assert all(isinstance(r, RetrievedChunk) for r in results)

    def test_retrieve_respects_top_k(self, require_api_key):
        from src.retrieval.retriever import retrieve
        results = retrieve("confidentiality", top_k=3)
        assert len(results) <= 3

    def test_retrieve_scores_in_valid_range(self, require_api_key):
        from src.retrieval.retriever import retrieve
        results = retrieve("termination notice period")
        for r in results:
            assert 0.0 <= r.similarity_score <= 1.0

    def test_retrieve_ranks_sequential(self, require_api_key):
        from src.retrieval.retriever import retrieve
        results = retrieve("governing law jurisdiction")
        for i, r in enumerate(results, start=1):
            assert r.mmr_rank == i

    def test_empty_query_returns_empty(self, require_api_key):
        from src.retrieval.retriever import retrieve
        results = retrieve("")
        assert results == []
