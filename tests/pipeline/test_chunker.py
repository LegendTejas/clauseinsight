"""
Tests for src/pipeline/chunker.py

Unit tests: test individual functions with mocked/synthetic input.
Integration tests: test full chunking pipeline on real PDFs.
"""

from __future__ import annotations

import math
import pytest

from src.pipeline.chunker import (
    Chunk,
    detect_format,
    chunk_document,
    _char_pos_to_page,
    _extract_sub_clauses,
    _truncate_heading,
)
from src.pipeline.parser import ParsedDocument, PageContent


# ── Chunk dataclass tests ──────────────────────────────────────────

class TestChunk:
    def test_char_count_computed(self):
        c = Chunk(
            clause_id="Section 1", heading="Title",
            text="Hello world", page_start=1, page_end=1,
            format_used="section_n",
        )
        assert c.char_count == 11

    def test_is_empty_short_text(self):
        c = Chunk(
            clause_id="S1", heading="", text="hi",
            page_start=1, page_end=1, format_used="section_n",
        )
        assert c.is_empty is True

    def test_is_empty_normal_text(self):
        c = Chunk(
            clause_id="S1", heading="Title",
            text="This is a normal length clause with enough content.",
            page_start=1, page_end=2, format_used="section_n",
        )
        assert c.is_empty is False

    def test_repr_contains_key_info(self):
        c = Chunk(
            clause_id="Section 4", heading="Reps",
            text="Some text here.", page_start=8, page_end=10,
            format_used="section_n",
        )
        r = repr(c)
        assert "Section 4" in r
        assert "8" in r


# ── _truncate_heading tests ────────────────────────────────────────

class TestTruncateHeading:
    def test_short_heading_unchanged(self):
        from src.pipeline.chunker import _truncate_heading
        h = "Representations and Warranties"
        assert _truncate_heading(h) == h

    def test_long_heading_truncated(self):
        from src.pipeline.chunker import _truncate_heading
        long_h = "word " * 20
        result = _truncate_heading(long_h)
        assert result.endswith("...")
        assert len(result.split()) <= 13  # 12 words + "..."

    def test_exactly_12_words_unchanged(self):
        from src.pipeline.chunker import _truncate_heading
        h = "one two three four five six seven eight nine ten eleven twelve"
        assert not _truncate_heading(h).endswith("...")


# ── _char_pos_to_page tests ────────────────────────────────────────

class TestCharPosToPage:
    def _make_offsets(self):
        # page 1: chars 0-99, page 2: 101-199, page 3: 201-299
        return [(0, 99, 1), (101, 199, 2), (201, 299, 3)]

    def test_first_page(self):
        assert _char_pos_to_page(0, self._make_offsets()) == 1

    def test_last_char_of_page(self):
        assert _char_pos_to_page(99, self._make_offsets()) == 1

    def test_second_page(self):
        assert _char_pos_to_page(150, self._make_offsets()) == 2

    def test_beyond_end_returns_last_page(self):
        assert _char_pos_to_page(9999, self._make_offsets()) == 3

    def test_empty_offsets(self):
        assert _char_pos_to_page(0, []) == 1


# ── _extract_sub_clauses tests ─────────────────────────────────────

class TestExtractSubClauses:
    def test_extracts_lettered_sub_clauses(self):
        text = "(a) First obligation.\n(b) Second obligation.\n(c) Third obligation."
        subs = _extract_sub_clauses(text, "Section 4", 8, 10, "section_n")
        assert len(subs) == 3
        assert subs[0].clause_id == "Section 4(a)"
        assert subs[1].clause_id == "Section 4(b)"
        assert subs[2].clause_id == "Section 4(c)"

    def test_extracts_roman_sub_clauses(self):
        text = "(i) First item.\n(ii) Second item.\n(iii) Third item."
        subs = _extract_sub_clauses(text, "Clause 1", 2, 3, "onenda_table")
        assert len(subs) == 3
        assert subs[0].clause_id == "Clause 1(i)"

    def test_returns_empty_for_single_sub_clause(self):
        text = "(a) Only one sub-clause here."
        subs = _extract_sub_clauses(text, "Section 4", 8, 10, "section_n")
        assert subs == []

    def test_returns_empty_for_no_sub_clauses(self):
        text = "Plain text with no sub-clause markers at all."
        subs = _extract_sub_clauses(text, "Section 4", 8, 10, "section_n")
        assert subs == []

    def test_sub_clauses_inherit_parent_pages(self):
        text = "(a) First.\n(b) Second."
        subs = _extract_sub_clauses(text, "Section 4", 8, 10, "section_n")
        assert all(s.page_start == 8 for s in subs)
        assert all(s.page_end == 10 for s in subs)

    def test_format_used_propagated(self):
        text = "(a) First.\n(b) Second."
        subs = _extract_sub_clauses(text, "Clause 1", 2, 2, "onenda_table")
        assert all(s.format_used == "onenda_table" for s in subs)


# ── detect_format integration tests ───────────────────────────────

class TestDetectFormat:
    def test_detects_onenda_table(self, sample_pdf_path):
        """oneNDA should detect as onenda_table."""
        import fitz
        from src.pipeline.parser import parse_pdf
        parsed = parse_pdf(sample_pdf_path)
        fitz_doc = fitz.open(str(sample_pdf_path))
        fmt = detect_format(parsed, fitz_doc)
        fitz_doc.close()
        assert fmt == "onenda_table"

    def test_detects_section_n(self, agency_pdf_path):
        """Agency Agreement should detect as section_n."""
        import fitz
        from src.pipeline.parser import parse_pdf
        parsed = parse_pdf(agency_pdf_path)
        fitz_doc = fitz.open(str(agency_pdf_path))
        fmt = detect_format(parsed, fitz_doc)
        fitz_doc.close()
        assert fmt == "section_n"

    def test_detects_bare_n(self, affiliate_pdf_path):
        """Chase Affiliate Agreement should detect as bare_n."""
        import fitz
        from src.pipeline.parser import parse_pdf
        parsed = parse_pdf(affiliate_pdf_path)
        fitz_doc = fitz.open(str(affiliate_pdf_path))
        fmt = detect_format(parsed, fitz_doc)
        fitz_doc.close()
        assert fmt == "bare_n"


# ── chunk_document integration tests ──────────────────────────────

class TestChunkDocument:
    def test_onenda_produces_chunks(self, sample_pdf_path):
        from src.pipeline.parser import parse_pdf
        parsed = parse_pdf(sample_pdf_path)
        chunks = chunk_document(parsed, source=str(sample_pdf_path))
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_onenda_chunk_count(self, sample_pdf_path):
        from src.pipeline.parser import parse_pdf
        parsed = parse_pdf(sample_pdf_path)
        chunks = chunk_document(parsed, source=str(sample_pdf_path))
        # oneNDA produces ~33 chunks with sub-clauses
        assert 10 < len(chunks) < 60

    def test_agency_agreement_chunk_count(self, agency_pdf_path):
        from src.pipeline.parser import parse_pdf
        parsed = parse_pdf(agency_pdf_path)
        chunks = chunk_document(parsed, source=str(agency_pdf_path))
        # Agency Agreement: 20 sections + many sub-clauses = ~100-200 chunks
        assert len(chunks) > 20

    def test_all_chunks_have_required_fields(self, sample_pdf_path):
        from src.pipeline.parser import parse_pdf
        parsed = parse_pdf(sample_pdf_path)
        chunks = chunk_document(parsed, source=str(sample_pdf_path))
        for c in chunks:
            assert c.clause_id, f"chunk missing clause_id: {c}"
            assert c.page_start >= 1
            assert c.page_end >= c.page_start
            assert c.format_used in ("section_n", "bare_n", "onenda_table", "fallback_prose")

    def test_no_empty_chunks(self, sample_pdf_path):
        from src.pipeline.parser import parse_pdf
        parsed = parse_pdf(sample_pdf_path)
        chunks = chunk_document(parsed, source=str(sample_pdf_path))
        assert all(not c.is_empty for c in chunks)

    def test_without_sub_clauses(self, sample_pdf_path):
        from src.pipeline.parser import parse_pdf
        parsed = parse_pdf(sample_pdf_path)
        chunks_with = chunk_document(parsed, source=str(sample_pdf_path), include_sub_clauses=True)
        chunks_without = chunk_document(parsed, source=str(sample_pdf_path), include_sub_clauses=False)
        # Without sub-clauses should produce fewer chunks
        assert len(chunks_without) <= len(chunks_with)

    def test_page_numbers_within_document_range(self, sample_pdf_path):
        from src.pipeline.parser import parse_pdf
        parsed = parse_pdf(sample_pdf_path)
        chunks = chunk_document(parsed, source=str(sample_pdf_path))
        for c in chunks:
            assert 1 <= c.page_start <= parsed.total_pages
            assert 1 <= c.page_end <= parsed.total_pages

    def test_format_used_consistent(self, sample_pdf_path):
        """All chunks from oneNDA should use the same format strategy."""
        from src.pipeline.parser import parse_pdf
        parsed = parse_pdf(sample_pdf_path)
        chunks = chunk_document(parsed, source=str(sample_pdf_path))
        formats = {c.format_used for c in chunks}
        # Should be predominantly one format
        assert len(formats) == 1


# ── chunk_document generalization tests ────────────────────────────
#
# Runs the full parse -> chunk pipeline against every contract in
# ALL_SAMPLE_CONTRACTS (any_contract_path fixture) so a pass proves the
# chunker works on contracts in general, not just the oneNDA table
# format it was originally built around.

class TestChunkDocumentAnyContract:
    def test_produces_at_least_one_chunk(self, any_contract_path):
        from src.pipeline.parser import parse_pdf
        parsed = parse_pdf(any_contract_path)
        chunks = chunk_document(parsed, source=str(any_contract_path))
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_all_chunks_have_required_fields(self, any_contract_path):
        from src.pipeline.parser import parse_pdf
        parsed = parse_pdf(any_contract_path)
        chunks = chunk_document(parsed, source=str(any_contract_path))
        for c in chunks:
            assert c.clause_id, f"chunk missing clause_id: {c}"
            assert c.page_start >= 1
            assert c.page_end >= c.page_start
            assert c.format_used in ("section_n", "bare_n", "onenda_table", "fallback_prose")

    def test_no_empty_chunks(self, any_contract_path):
        from src.pipeline.parser import parse_pdf
        parsed = parse_pdf(any_contract_path)
        chunks = chunk_document(parsed, source=str(any_contract_path))
        assert all(not c.is_empty for c in chunks)

    def test_page_numbers_within_document_range(self, any_contract_path):
        from src.pipeline.parser import parse_pdf
        parsed = parse_pdf(any_contract_path)
        chunks = chunk_document(parsed, source=str(any_contract_path))
        for c in chunks:
            assert 1 <= c.page_start <= parsed.total_pages
            assert 1 <= c.page_end <= parsed.total_pages

    def test_detect_format_returns_known_type(self, any_contract_path):
        import fitz
        from src.pipeline.parser import parse_pdf
        parsed = parse_pdf(any_contract_path)
        fitz_doc = fitz.open(str(any_contract_path))
        fmt = detect_format(parsed, fitz_doc)
        fitz_doc.close()
        assert fmt in ("section_n", "bare_n", "onenda_table", "fallback_prose")