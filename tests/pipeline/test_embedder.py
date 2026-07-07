"""
Tests for src/pipeline/embedder.py

Unit tests: test helpers that don't need API calls.
Integration tests: marked with @pytest.mark.integration, require GOOGLE_API_KEY.
"""

from __future__ import annotations

import pytest

from src.pipeline.embedder import (
    EmbedResult,
    _make_chunk_id,
    _build_embed_text,
    EMBEDDING_DIM,
    EMBED_BATCH_SIZE,
    MAX_RETRIES,
)


# ── _make_chunk_id tests ───────────────────────────────────────────

class TestMakeChunkId:
    def test_format_is_source_double_colon_clause(self):
        cid = _make_chunk_id("contract.pdf", "Section 4")
        assert cid == "contract.pdf::Section 4"

    def test_spaces_in_source_replaced(self):
        cid = _make_chunk_id("my contract.pdf", "Section 1")
        assert " " not in cid.split("::")[0]

    def test_sub_clause_id_preserved(self):
        cid = _make_chunk_id("contract.pdf", "Section 4(a)")
        assert "Section 4(a)" in cid

    def test_newlines_stripped(self):
        cid = _make_chunk_id("contract\n.pdf", "Section\n4")
        assert "\n" not in cid

    def test_deterministic(self):
        cid1 = _make_chunk_id("c.pdf", "S1")
        cid2 = _make_chunk_id("c.pdf", "S1")
        assert cid1 == cid2


# ── _build_embed_text tests ────────────────────────────────────────

class TestBuildEmbedText:
    def test_heading_prepended_to_body(self, sample_chunk):
        text = _build_embed_text(sample_chunk)
        assert sample_chunk.heading in text
        assert sample_chunk.text in text
        assert text.index(sample_chunk.heading) < text.index(sample_chunk.text[:10])

    def test_no_duplication_when_body_starts_with_heading(self):
        from src.pipeline.chunker import Chunk
        c = Chunk(
            clause_id="S1", heading="The Company represents",
            text="The Company represents that all statements are true.",
            page_start=1, page_end=1, format_used="section_n",
        )
        text = _build_embed_text(c)
        # Should not prefix again since body already starts with heading
        assert text == c.text

    def test_empty_heading_uses_body_only(self):
        from src.pipeline.chunker import Chunk
        c = Chunk(
            clause_id="Para 1", heading="",
            text="Some body text here.",
            page_start=1, page_end=1, format_used="fallback_prose",
        )
        assert _build_embed_text(c) == c.text

    def test_separator_between_heading_and_body(self, sample_chunk):
        text = _build_embed_text(sample_chunk)
        assert "\n\n" in text


# ── EmbedResult tests ──────────────────────────────────────────────

class TestEmbedResult:
    def test_success_when_no_failures(self):
        r = EmbedResult("c.pdf", total_chunks=10, embedded_count=10,
                        skipped_count=0, failed_count=0, elapsed_seconds=5.0)
        assert r.success is True

    def test_failure_when_any_failed(self):
        r = EmbedResult("c.pdf", total_chunks=10, embedded_count=9,
                        skipped_count=0, failed_count=1, elapsed_seconds=5.0)
        assert r.success is False

    def test_repr_contains_key_counts(self):
        r = EmbedResult("c.pdf", 10, 8, 2, 0, 3.5)
        s = str(r)
        assert "8 embedded" in s
        assert "2 skipped" in s
        assert "0 failed" in s

    def test_all_skipped_is_success(self):
        r = EmbedResult("c.pdf", 10, 0, 10, 0, 0.2)
        assert r.success is True


# ── Constants sanity checks ────────────────────────────────────────

class TestConstants:
    def test_embedding_dim(self):
        assert EMBEDDING_DIM == 768

    def test_batch_size_reasonable(self):
        assert 1 <= EMBED_BATCH_SIZE <= 50

    def test_max_retries_positive(self):
        assert MAX_RETRIES >= 1


# ── Integration tests ──────────────────────────────────────────────

@pytest.mark.integration
class TestEmbedAndStoreIntegration:
    def test_embed_single_chunk(self, require_api_key, sample_chunk, tmp_chroma, tmp_sqlite):
        from src.pipeline.embedder import embed_and_store
        result = embed_and_store(
            [sample_chunk],
            source_name="test_contract.pdf",
            collection=tmp_chroma,
            conn=tmp_sqlite,
        )
        assert result.embedded_count == 1
        assert result.failed_count == 0
        assert result.success is True

    def test_idempotency(self, require_api_key, sample_chunk, tmp_chroma, tmp_sqlite):
        from src.pipeline.embedder import embed_and_store
        # First run
        r1 = embed_and_store([sample_chunk], "test.pdf", collection=tmp_chroma, conn=tmp_sqlite)
        assert r1.embedded_count == 1

        # Second run — should skip
        r2 = embed_and_store([sample_chunk], "test.pdf", collection=tmp_chroma, conn=tmp_sqlite)
        assert r2.embedded_count == 0
        assert r2.skipped_count == 1

    def test_embed_multiple_chunks(self, require_api_key, sample_chunks, tmp_chroma, tmp_sqlite):
        from src.pipeline.embedder import embed_and_store
        result = embed_and_store(
            sample_chunks,
            source_name="test_contract.pdf",
            collection=tmp_chroma,
            conn=tmp_sqlite,
        )
        assert result.embedded_count == len(sample_chunks)
        assert result.failed_count == 0

    def test_vectors_stored_in_chroma(self, require_api_key, sample_chunk, tmp_chroma, tmp_sqlite):
        from src.pipeline.embedder import embed_and_store
        embed_and_store([sample_chunk], "test.pdf", collection=tmp_chroma, conn=tmp_sqlite)
        assert tmp_chroma.count() == 1

    def test_full_text_stored_in_sqlite(self, require_api_key, sample_chunk, tmp_sqlite, tmp_chroma):
        from src.pipeline.embedder import embed_and_store
        from src.utils.store import get_chunk_text
        embed_and_store([sample_chunk], "test.pdf", collection=tmp_chroma, conn=tmp_sqlite)
        text = get_chunk_text(sample_chunk.clause_id, "test.pdf", tmp_sqlite)
        assert text == sample_chunk.text

    @pytest.mark.slow
    def test_embed_query_returns_correct_dim(self, require_api_key):
        from src.pipeline.embedder import embed_query
        vector = embed_query("What are the indemnification obligations?")
        assert len(vector) == EMBEDDING_DIM
        assert all(isinstance(v, float) for v in vector)
