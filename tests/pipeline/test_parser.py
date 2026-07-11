"""
Tests for src/pipeline/parser.py

Unit tests: all run offline, no API calls needed.
Integration tests: require a real PDF file in legal_contracts/.
"""

from __future__ import annotations

import io
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.pipeline.parser import (
    parse_pdf,
    ParsedDocument,
    PageContent,
    PasswordProtectedError,
    CorruptedPDFError,
    ParsingError,
    _clean_text,
    _empty_page_ratio,
    _detect_scanned,
)


# ── PageContent tests ──────────────────────────────────────────────

class TestPageContent:
    def test_word_count_computed(self):
        page = PageContent(page_number=1, text="hello world foo bar")
        assert page.word_count == 4

    def test_char_count_computed(self):
        page = PageContent(page_number=1, text="hello")
        assert page.char_count == 5

    def test_is_empty_true_when_few_words(self):
        page = PageContent(page_number=1, text="hi")
        assert page.is_empty is True

    def test_is_empty_false_when_enough_words(self):
        page = PageContent(page_number=1, text="hello world this is a page")
        assert page.is_empty is False

    def test_empty_text(self):
        page = PageContent(page_number=1, text="")
        assert page.word_count == 0
        assert page.is_empty is True


# ── ParsedDocument tests ───────────────────────────────────────────

class TestParsedDocument:
    def _make_doc(self, texts):
        pages = [PageContent(page_number=i+1, text=t) for i, t in enumerate(texts)]
        return ParsedDocument(
            source_name="test.pdf",
            pages=pages,
            total_pages=len(pages),
        )

    def test_full_text_joins_pages(self):
        doc = self._make_doc(["page one text", "page two text"])
        assert "page one text" in doc.full_text
        assert "page two text" in doc.full_text

    def test_total_word_count(self):
        doc = self._make_doc(["one two three", "four five"])
        assert doc.total_word_count == 5

    def test_get_page_by_number(self):
        doc = self._make_doc(["first", "second", "third"])
        assert doc.get_page(2).text == "second"

    def test_get_page_raises_on_missing(self):
        doc = self._make_doc(["only page"])
        with pytest.raises(IndexError):
            doc.get_page(99)

    def test_get_page_1indexed(self):
        doc = self._make_doc(["page one", "page two"])
        assert doc.get_page(1).page_number == 1
        assert doc.get_page(2).page_number == 2


# ── _clean_text tests ──────────────────────────────────────────────

class TestCleanText:
    def test_strips_trailing_whitespace_per_line(self):
        raw = "hello   \nworld   "
        result = _clean_text(raw)
        assert "hello" in result
        assert not any(line.endswith(" ") for line in result.splitlines())

    def test_collapses_triple_newlines(self):
        raw = "line one\n\n\n\nline two"
        result = _clean_text(raw)
        assert "\n\n\n" not in result

    def test_strips_outer_whitespace(self):
        raw = "\n\n  content  \n\n"
        assert _clean_text(raw) == "content"

    def test_preserves_double_newlines(self):
        raw = "para one\n\npara two"
        assert "\n\n" in _clean_text(raw)

    def test_empty_string(self):
        assert _clean_text("") == ""


# ── Scanned PDF detection tests ────────────────────────────────────

class TestScannedDetection:
    def _make_pages(self, texts):
        return [PageContent(page_number=i+1, text=t) for i, t in enumerate(texts)]

    def test_empty_page_ratio_all_empty(self):
        pages = self._make_pages(["", "", "hi"])
        # "hi" has 1 word < 3 → is_empty=True; all 3 are empty
        ratio = _empty_page_ratio(pages)
        assert ratio == 1.0

    def test_empty_page_ratio_none_empty(self):
        pages = self._make_pages(["hello world foo", "bar baz qux"])
        assert _empty_page_ratio(pages) == 0.0

    def test_empty_page_ratio_mixed(self):
        pages = self._make_pages(["hello world foo", "", ""])
        ratio = _empty_page_ratio(pages)
        assert abs(ratio - 2/3) < 1e-9

    def test_detect_scanned_true(self):
        # 9 out of 10 pages empty → ratio=0.9 >= threshold 0.8
        pages = self._make_pages([""] * 9 + ["hello world foo bar"])
        assert _detect_scanned(pages) is True

    def test_detect_scanned_false(self):
        pages = self._make_pages(["hello world foo"] * 10)
        assert _detect_scanned(pages) is False

    def test_empty_pages_list(self):
        assert _empty_page_ratio([]) == 0.0


# ── parse_pdf integration tests ────────────────────────────────────

class TestParsePDF:
    def test_parse_real_pdf(self, sample_pdf_path):
        """Integration: parse the oneNDA sample contract."""
        doc = parse_pdf(sample_pdf_path)
        assert isinstance(doc, ParsedDocument)
        assert doc.total_pages == 3
        assert doc.total_word_count > 500
        assert doc.likely_scanned is False
        assert doc.source_name == sample_pdf_path.name

    def test_parse_returns_1indexed_pages(self, sample_pdf_path):
        doc = parse_pdf(sample_pdf_path)
        assert doc.pages[0].page_number == 1
        assert doc.pages[-1].page_number == doc.total_pages

    def test_parse_bytes_input(self, sample_pdf_path):
        """parse_pdf should accept raw bytes (Streamlit upload scenario)."""
        pdf_bytes = sample_pdf_path.read_bytes()
        doc = parse_pdf(pdf_bytes, source_name="test_upload.pdf")
        assert doc.source_name == "test_upload.pdf"
        assert doc.total_pages == 3

    def test_parse_agency_agreement(self, agency_pdf_path):
        """Integration: agency agreement has 33 pages."""
        doc = parse_pdf(agency_pdf_path)
        assert doc.total_pages == 33
        assert doc.total_word_count > 5000

    def test_parse_affiliate_agreement(self, affiliate_pdf_path):
        """Integration: affiliate agreement is multi-page."""
        doc = parse_pdf(affiliate_pdf_path)
        assert doc.total_pages > 1
        assert doc.total_word_count > 1000

    def test_parse_corrupted_raises(self, tmp_dir):
        """Non-PDF bytes should raise CorruptedPDFError."""
        fake_pdf = tmp_dir / "fake.pdf"
        fake_pdf.write_bytes(b"this is not a pdf")
        with pytest.raises(CorruptedPDFError):
            parse_pdf(fake_pdf)

    def test_parse_nonexistent_raises(self):
        """Missing file should raise ParsingError."""
        with pytest.raises((ParsingError, FileNotFoundError)):
            parse_pdf(Path("nonexistent_contract.pdf"))

    def test_source_name_derived_from_path(self, sample_pdf_path):
        """source_name should default to filename when not provided."""
        doc = parse_pdf(sample_pdf_path)
        assert doc.source_name == sample_pdf_path.name

    def test_custom_source_name(self, sample_pdf_path):
        doc = parse_pdf(sample_pdf_path, source_name="my_contract.pdf")
        assert doc.source_name == "my_contract.pdf"

    def test_page_text_not_empty(self, sample_pdf_path):
        """Each page of a real contract should have some text."""
        doc = parse_pdf(sample_pdf_path)
        non_empty = [p for p in doc.pages if not p.is_empty]
        assert len(non_empty) > 0


# ── parse_pdf generalization tests ─────────────────────────────────
#
# Runs against every contract in ALL_SAMPLE_CONTRACTS (any_contract_path
# fixture in tests/conftest.py) instead of one hardcoded document, so a
# pass proves parse_pdf works on contracts in general, not just oneNDA.

class TestParsePDFAnyContract:
    def test_parses_without_error(self, any_contract_path):
        doc = parse_pdf(any_contract_path)
        assert isinstance(doc, ParsedDocument)

    def test_has_at_least_one_page(self, any_contract_path):
        doc = parse_pdf(any_contract_path)
        assert doc.total_pages >= 1

    def test_source_name_matches_filename(self, any_contract_path):
        doc = parse_pdf(any_contract_path)
        assert doc.source_name == any_contract_path.name

    def test_has_nonzero_word_count(self, any_contract_path):
        doc = parse_pdf(any_contract_path)
        assert doc.total_word_count > 0

    def test_pages_are_1indexed(self, any_contract_path):
        doc = parse_pdf(any_contract_path)
        assert doc.pages[0].page_number == 1
        assert doc.pages[-1].page_number == doc.total_pages

    def test_at_least_one_page_has_text(self, any_contract_path):
        """A real (non-scanned) contract should have extractable text
        on at least one page."""
        doc = parse_pdf(any_contract_path)
        non_empty = [p for p in doc.pages if not p.is_empty]
        assert len(non_empty) > 0