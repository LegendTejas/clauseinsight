"""
ClauseInsight — PDF Parser
============================

Extracts structured, page-aware text from legal contract PDFs.

This module is the entry point of the ingestion pipeline. Every other
module (chunker, embedder, risk_scanner) depends on the page-accurate
text this module produces, because clause citations downstream are
only as correct as the page numbers we attach here.

Design decisions (see also docs/adr/ADR-001.md):
    - PyMuPDF (fitz) is used over pypdf/pdfplumber for speed and because
      it gives reliable page-level metadata in one pass.
    - Accepts both a file path AND raw bytes, because Streamlit's
      `st.file_uploader` returns an in-memory buffer, not a path on disk.
    - Returns a ParsedDocument dataclass rather than a raw dict/list, so
      downstream code gets type-checked, IDE-autocompletable structure
      instead of guessing dict keys.
    - Detects scanned (image-only) PDFs by checking how many pages have
      near-zero extractable text — OCR is out of scope for this project,
      but the user needs to know *why* nothing came back, not just get
      an empty result.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf as fitz  # PyMuPDF 1.27+ uses pymupdf instead of fitz

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# Custom exceptions
# ──────────────────────────────────────────────────────────────────
# Why custom exceptions instead of letting fitz's raw errors bubble up:
# the Streamlit UI needs to show a human a useful message
# ("this PDF is password protected") rather than a stack trace.


class ParsingError(Exception):
    """Base exception for any failure in the parsing pipeline."""


class PasswordProtectedError(ParsingError):
    """Raised when a PDF is encrypted and cannot be opened without a password."""


class CorruptedPDFError(ParsingError):
    """Raised when the file is not a valid / readable PDF at all."""


class ScannedPDFWarning(UserWarning):
    """
    Not an error — raised as a warning when a PDF has little to no
    extractable text layer (i.e. it's likely a scanned image).
    OCR is a known limitation of this project (see README §9).
    """


# ──────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────


@dataclass
class PageContent:
    """Text and metadata for a single page of a parsed PDF."""

    page_number: int  # 1-indexed — matches what a human sees in a PDF viewer
    text: str
    word_count: int = field(init=False)
    char_count: int = field(init=False)

    def __post_init__(self) -> None:
        # Computed once at creation rather than recomputed every time
        # something downstream needs to check "is this page basically empty?"
        self.word_count = len(self.text.split())
        self.char_count = len(self.text)

    @property
    def is_empty(self) -> bool:
        """True if the page has effectively no extractable text."""
        return self.word_count < 3


@dataclass
class ParsedDocument:
    """
    The full structured output of parsing one contract PDF.

    This is what every downstream module (chunker.py, embedder.py)
    consumes. Keeping this as a dataclass — not a dict — means typos
    in field access fail at development time, not silently at runtime.
    """

    source_name: str
    pages: list[PageContent]
    total_pages: int
    title: str | None = None
    author: str | None = None
    file_size_bytes: int = 0
    likely_scanned: bool = False

    @property
    def full_text(self) -> str:
        """Concatenated text of all pages — useful for quick previews / debugging."""
        return "\n\n".join(p.text for p in self.pages)

    @property
    def total_word_count(self) -> int:
        return sum(p.word_count for p in self.pages)

    def get_page(self, page_number: int) -> PageContent:
        """1-indexed page lookup, matching how citations are shown to the user."""
        for page in self.pages:
            if page.page_number == page_number:
                return page
        raise IndexError(
            f"Page {page_number} not found. Document has {self.total_pages} pages."
        )


# ──────────────────────────────────────────────────────────────────
# Core parsing logic
# ──────────────────────────────────────────────────────────────────

# Threshold used to flag a document as "likely scanned": if more than
# this fraction of pages come back with essentially no text, the PDF
# almost certainly has no text layer (it's a raster scan).
_SCANNED_PAGE_RATIO_THRESHOLD = 0.8


def parse_pdf(source: str | Path | bytes, source_name: str | None = None) -> ParsedDocument:
    """
    Parse a contract PDF into a structured, page-aware ParsedDocument.

    Args:
        source: Either a file path (str/Path) or raw PDF bytes. Bytes
            input is what you get from Streamlit's `st.file_uploader`
            via `.getvalue()` — this function supports both so the same
            code path works whether you're testing from disk or running
            the live app.
        source_name: Human-readable name for logging/citations (e.g.
            the original uploaded filename). If not given, derived from
            the path, or set to "uploaded_document.pdf" for bytes input.

    Returns:
        ParsedDocument with page-by-page text and document metadata.

    Raises:
        PasswordProtectedError: PDF requires a password we don't have.
        CorruptedPDFError: File is not a valid/readable PDF.
        ParsingError: Any other unexpected failure during parsing.
    """
    doc, resolved_name, file_size = _open_pdf(source, source_name)

    if doc.needs_pass:
        doc.close()
        raise PasswordProtectedError(
            f"'{resolved_name}' is password-protected. "
            "ClauseInsight cannot read encrypted PDFs — please upload an unlocked copy."
        )

    try:
        pages = [_extract_page(doc, i) for i in range(doc.page_count)]
        metadata = doc.metadata or {}
        likely_scanned = _detect_scanned(pages)

        if likely_scanned:
            logger.warning(
                "'%s' appears to be a scanned PDF (%.0f%% of pages have "
                "near-zero extractable text). OCR is not implemented — "
                "see README known limitations.",
                resolved_name,
                _empty_page_ratio(pages) * 100,
            )

        return ParsedDocument(
            source_name=resolved_name,
            pages=pages,
            total_pages=doc.page_count,
            title=metadata.get("title") or None,
            author=metadata.get("author") or None,
            file_size_bytes=file_size,
            likely_scanned=likely_scanned,
        )
    finally:
        # Always close the document handle, even if extraction raised.
        doc.close()


def _open_pdf(
    source: str | Path | bytes, source_name: str | None
) -> tuple[fitz.Document, str, int]:
    """
    Opens the PDF regardless of whether the caller gave us a path or bytes.
    Returns the open document, a resolved display name, and the file size.
    """
    try:
        if isinstance(source, (str, Path)):
            path = Path(source)
            resolved_name = source_name or path.name
            file_size = path.stat().st_size
            doc = fitz.open(path)
        else:
            # bytes input — e.g. from Streamlit's uploader.getvalue()
            resolved_name = source_name or "uploaded_document.pdf"
            file_size = len(source)
            doc = fitz.open(stream=io.BytesIO(source), filetype="pdf")
    except fitz.FileDataError as exc:
        raise CorruptedPDFError(
            f"'{source_name or source}' could not be read as a valid PDF. "
            f"It may be corrupted or not actually a PDF file."
        ) from exc
    except Exception as exc:  # noqa: BLE001 — re-raised as our own typed error
        raise ParsingError(f"Unexpected error opening PDF: {exc}") from exc

    return doc, resolved_name, file_size


def _extract_page(doc: fitz.Document, index: int) -> PageContent:
    """Extract text from a single page. index is 0-based (PyMuPDF convention)."""
    page = doc.load_page(index)
    # sort=True respects column borders and prevents text jumbling in signature blocks or side-by-side structures.
    raw_text = page.get_text("text", sort=True)
    cleaned = _clean_text(raw_text)
    return PageContent(page_number=index + 1, text=cleaned)  # store as 1-indexed


def _clean_text(raw_text: str) -> str:
    """
    Light normalization only — we deliberately do NOT aggressively strip
    content here. Header/footer stripping and structural cleanup belongs
    in chunker.py, which has the section-level context to do it safely.
    """
    # Collapse runs of 3+ blank lines down to a double newline, and strip
    # trailing whitespace per line. PyMuPDF sometimes emits irregular
    # spacing around multi-column text.
    lines = [line.rstrip() for line in raw_text.splitlines()]
    text = "\n".join(lines)
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip()


def _empty_page_ratio(pages: list[PageContent]) -> float:
    if not pages:
        return 0.0
    return sum(1 for p in pages if p.is_empty) / len(pages)


def _detect_scanned(pages: list[PageContent]) -> bool:
    """
    Heuristic: if most pages have effectively no extractable text, the
    PDF almost certainly has no text layer (it's an image scan).
    This is intentionally simple — see pdf-reading skill's note that a
    `pdffonts` check would be the more rigorous version of this signal.
    """
    return _empty_page_ratio(pages) >= _SCANNED_PAGE_RATIO_THRESHOLD


# ──────────────────────────────────────────────────────────────────
# Manual smoke test — run this file directly during development:
#   python src/parser.py path/to/sample_contract.pdf
# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if len(sys.argv) != 2:
        print("Usage: python src/parser.py <path_to_pdf>")
        sys.exit(1)

    result = parse_pdf(sys.argv[1])

    print(f"\nParsed: {result.source_name}")
    print(f"  Pages: {result.total_pages}")
    print(f"  Title: {result.title or '(none)'}")
    print(f"  Author: {result.author or '(none)'}")
    print(f"  File size: {result.file_size_bytes:,} bytes")
    print(f"  Total words: {result.total_word_count:,}")
    print(f"  Likely scanned: {result.likely_scanned}")
    print(f"\n  Page 1 preview ({result.pages[0].word_count} words):")
    print(f"  {result.pages[0].text[:300]}...")