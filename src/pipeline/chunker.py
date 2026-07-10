"""
ClauseInsight — Contract Chunker
==================================

Splits a ParsedDocument into semantically meaningful, citation-ready
chunks — one chunk per clause or sub-clause.

WHY THIS MODULE EXISTS SEPARATELY FROM parser.py
-------------------------------------------------
parser.py is deliberately dumb: it extracts text page-by-page and
returns it faithfully. Chunker.py is where format-awareness lives.
Different contracts use completely different structural signals:
  - Bold "Section N. Title." lines          (Agency Agreement style)
  - Plain "N. Title" lines, same font/size  (Affiliate Agreement style)
  - Bold "N." number + title on next line   (oneNDA table-cell style)
  - Pure prose paragraphs, no numbering     (fallback)

DESIGN: FORMAT DETECTOR → STRATEGY DISPATCH
--------------------------------------------
Rather than one fragile regex that tries to cover all formats, this
module runs a lightweight format-detection step first (by scanning
a sample of the document for which heading signals fire), then
dispatches to a format-specific strategy. Each strategy:
  1. Identifies top-level clause boundaries
  2. Groups page text into clause-level chunks
  3. Attaches page numbers for downstream citations

CHUNKER ←→ PARSER INTERFACE
----------------------------
Primary input:  ParsedDocument (from parser.py)
Secondary input: original PDF path/bytes — needed ONLY for the
  oneNDA table-style format, where get_text("dict") block coordinates
  are required to stitch clause labels that land on separate lines
  from their content (a PyMuPDF layout artifact of table-cell PDFs).

OUTPUT: list[Chunk]
  Each Chunk carries:
    - clause_id:   human-readable "Section 4" / "3(a)" / "Clause 2"
    - heading:     the clause title if detectable, else first N words
    - text:        full clause text (heading + body)
    - page_start:  first page this clause appears on (1-indexed)
    - page_end:    last page this clause spans
    - format_used: which strategy produced this chunk (for debugging)
"""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pymupdf as fitz  # PyMuPDF 1.27+ uses pymupdf instead of fitz

# Import the dataclass that parser.py produces
# (chunker.py is always called after parser.py in the pipeline)
import sys as _sys
_root_dir = str(Path(__file__).resolve().parent.parent.parent)
if _root_dir not in _sys.path:
    _sys.path.insert(0, _root_dir)

from src.pipeline.parser import ParsedDocument, PageContent

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# Output data structure
# ──────────────────────────────────────────────────────────────────

FormatType = Literal["section_n", "bare_n", "onenda_table", "fallback_prose"]


@dataclass
class Chunk:
    """
    A single citation-ready clause chunk.

    clause_id examples:
        "Section 4"         — Agency Agreement style
        "Section 4(a)"      — Agency sub-clause
        "3"                 — Affiliate Agreement top-level
        "3. Referral Fee"   — when heading is embedded in the number line
        "Clause 1"          — oneNDA style
        "Clause 1(a)"       — oneNDA sub-clause
        "Para 7"            — fallback prose paragraph
    """

    clause_id: str
    heading: str
    text: str
    page_start: int
    page_end: int
    format_used: FormatType
    char_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.char_count = len(self.text)

    @property
    def is_empty(self) -> bool:
        return len(self.text.strip()) < 20

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return (
            f"Chunk(id={self.clause_id!r}, pages={self.page_start}-{self.page_end}, "
            f"chars={self.char_count}, preview={preview!r})"
        )


# ──────────────────────────────────────────────────────────────────
# Regex patterns — compiled once at module load
# ──────────────────────────────────────────────────────────────────

# Agency Agreement: "Section 4. Representations and Warranties..."
_RE_SECTION_N = re.compile(
    r'^(Section\s+\d+)\.\s*([A-Z][^\n]*)',
    re.MULTILINE
)

# Affiliate Agreement: "3. Referral Fee" or "5.Term of this Agreement"
# The \xa0 is a non-breaking space — the Chase PDF uses it between number and title
_RE_BARE_N = re.compile(
    r'^(\d+)[\.\xa0]\s*([A-Za-z][^\n]{0,80})',
    re.MULTILINE
)

# Sub-clause labels: (a), (b), (aa), (i), (ii), (iii), (iv), (v)
_RE_SUB_CLAUSE = re.compile(
    r'^\(([a-z]{1,2}|[ivxlc]+)\)',
    re.MULTILINE
)

# Footer patterns to strip — page numbers, source lines, running headers
_RE_FOOTER = re.compile(
    r'^\s*(\d{1,3}|Source:.*|Click to learn.*|Standard Mutual.*|Law Insider.*)\s*$',
    re.MULTILINE
)

# Detect SEC filing markup — used to reject obviously non-contract files
_RE_SEC_MARKUP = re.compile(r'<SEC-(?:DOCUMENT|HEADER)|<XBRL>', re.IGNORECASE)

# Minimum hits for a pattern to be considered the "dominant" format
_FORMAT_THRESHOLD = 3


# ──────────────────────────────────────────────────────────────────
# Format detection
# ──────────────────────────────────────────────────────────────────

def detect_format(doc: ParsedDocument, fitz_doc: fitz.Document) -> FormatType:
    """
    Scan a sample of the document to determine which chunking strategy
    to use. Checks in priority order — most specific first.

    Uses both the flat text (for regex patterns) and the fitz block
    structure (for the oneNDA bold+table signal).

    Args:
        doc:      ParsedDocument from parser.py (flat text, page-aware)
        fitz_doc: Open fitz.Document for layout/bold inspection

    Returns:
        One of: "section_n", "bare_n", "onenda_table", "fallback_prose"
    """
    # Sample first 10 pages (or all pages if short doc)
    sample_pages = doc.pages[:10]
    sample_text = "\n".join(p.text for p in sample_pages)

    # Reject: SEC EDGAR filing — not a contract at all
    if _RE_SEC_MARKUP.search(sample_text):
        logger.warning(
            "'%s' appears to be an SEC EDGAR filing, not a contract. "
            "Falling back to prose chunking but results may be meaningless.",
            doc.source_name
        )
        return "fallback_prose"

    # Strategy 1: "Section N." with bold headers (Agency Agreement style)
    # Signal: regex match + the matching line is bold in fitz block data
    section_hits = _count_bold_section_headers(fitz_doc, page_limit=8)
    if section_hits >= _FORMAT_THRESHOLD:
        logger.info("'%s' → format: section_n (%d bold Section headers found)",
                    doc.source_name, section_hits)
        return "section_n"

    # Strategy 2: Bare "N. Title" headings (Affiliate Agreement style)
    # Signal: regex fires enough times on flat text
    bare_hits = len(_RE_BARE_N.findall(sample_text))
    if bare_hits >= _FORMAT_THRESHOLD:
        logger.info("'%s' → format: bare_n (%d bare-number headings found)",
                    doc.source_name, bare_hits)
        return "bare_n"

    # Strategy 3: oneNDA table-cell style
    # Signal: bold isolated number blocks at consistent x0, separate from title text
    if _detect_onenda_style(fitz_doc, page_limit=5):
        logger.info("'%s' → format: onenda_table", doc.source_name)
        return "onenda_table"

    # Fallback: paragraph-based chunking
    logger.info("'%s' → format: fallback_prose (no numbered structure detected)",
                doc.source_name)
    return "fallback_prose"


def _count_bold_section_headers(fitz_doc: fitz.Document, page_limit: int) -> int:
    """Count lines matching 'Section N.' that are also bold in the fitz layout."""
    count = 0
    for page_idx in range(min(page_limit, fitz_doc.page_count)):
        page = fitz_doc[page_idx]
        for block in page.get_text("dict", sort=True)["blocks"]:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                txt = " ".join(s["text"] for s in line["spans"]).strip()
                if _RE_SECTION_N.match(txt):
                    # Verify it's actually bold — not just the regex firing on body text
                    is_bold = any(bool(s["flags"] & 2**4) for s in line["spans"])
                    if is_bold:
                        count += 1
    return count


def _detect_onenda_style(fitz_doc: fitz.Document, page_limit: int) -> bool:
    """
    Detect the oneNDA table-cell layout:
    - Bold, isolated single digit/number on its own line (e.g. "1.", "2.")
    - Followed immediately by a bold title on the NEXT line at the same x0
    - Sub-clause labels (a), (b) on their OWN lines, content on next line

    This pattern is a PyMuPDF artifact of table-cell PDFs where each
    cell renders as a separate block even when visually they're adjacent.
    """
    isolated_bold_numbers = 0
    for page_idx in range(min(page_limit, fitz_doc.page_count)):
        page = fitz_doc[page_idx]
        lines_data = []
        for block in page.get_text("dict", sort=True)["blocks"]:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                txt = " ".join(s["text"] for s in line["spans"]).strip()
                is_bold = any(bool(s["flags"] & 2**4) for s in line["spans"])
                x0 = round(block["bbox"][0], 1)
                lines_data.append((txt, is_bold, x0))

        for i, (txt, bold, x0) in enumerate(lines_data):
            # Look for: bold isolated number like "1." or "2." or "4."
            if bold and re.match(r'^\d+\.$', txt):
                # Check if next non-empty line is also bold at same x0
                for j in range(i + 1, min(i + 3, len(lines_data))):
                    ntxt, nbold, nx0 = lines_data[j]
                    if ntxt and nbold and abs(nx0 - x0) < 5:
                        isolated_bold_numbers += 1
                        break

    return isolated_bold_numbers >= 2


# ──────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────

def chunk_document(
    parsed: ParsedDocument,
    source: str | Path | bytes,
    include_sub_clauses: bool = True,
) -> list[Chunk]:
    """
    Chunk a parsed contract into citation-ready Chunk objects.

    Args:
        parsed:              Output of parser.parse_pdf()
        source:              Original file path or bytes — needed for
                             fitz layout inspection in format detection
                             and for the oneNDA table-stitching strategy.
        include_sub_clauses: If True, also produce sub-clause chunks
                             (e.g. Section 4(a), Section 4(b)) in
                             addition to top-level clause chunks.
                             Set False if you only need section-level
                             granularity (e.g. for a quick summary).

    Returns:
        List of Chunk objects, ordered by appearance in the document.
        Empty chunks (< 20 chars of text) are filtered out.
    """
    if isinstance(source, (str, Path)):
        fitz_doc = fitz.open(Path(source))
    else:
        fitz_doc = fitz.open(stream=io.BytesIO(source), filetype="pdf")

    try:
        fmt = detect_format(parsed, fitz_doc)

        if fmt == "section_n":
            chunks = _chunk_section_n(parsed, fitz_doc, include_sub_clauses)
        elif fmt == "bare_n":
            chunks = _chunk_bare_n(parsed, include_sub_clauses)
        elif fmt == "onenda_table":
            chunks = _chunk_onenda_table(parsed, fitz_doc, include_sub_clauses)
        else:
            chunks = _chunk_fallback_prose(parsed)

    finally:
        fitz_doc.close()

    # Filter empty chunks and return
    result = [c for c in chunks if not c.is_empty]
    logger.info(
        "'%s' → %d chunks produced (format: %s, sub_clauses: %s)",
        parsed.source_name, len(result), fmt, include_sub_clauses
    )
    return result


# ──────────────────────────────────────────────────────────────────
# Strategy 1: "Section N." with bold headers (Agency Agreement)
# ──────────────────────────────────────────────────────────────────

def _chunk_section_n(
    parsed: ParsedDocument,
    fitz_doc: fitz.Document,
    include_sub_clauses: bool,
) -> list[Chunk]:
    """
    Chunk by bold "Section N. Title" lines.

    The Agency Agreement has 20 bold Section headers across 33 pages.
    We collect ALL page text, find section boundaries using both the
    bold signal from fitz AND the regex, then split the full text at
    those boundaries. Page numbers are tracked by scanning which page
    each character offset falls on.

    Why we use fitz layout + regex together (not just regex):
    Body text sometimes contains "Section N" as a cross-reference
    (e.g. "as defined in Section 9(a)"). The bold check filters those
    out — cross-references are never bold.
    """
    # Build a page-offset index: for any char position in full_text,
    # which page is it on?
    page_offsets: list[tuple[int, int, int]] = []  # (start_char, end_char, page_num)
    cursor = 0
    for page in parsed.pages:
        start = cursor
        end = cursor + len(page.text)
        page_offsets.append((start, end, page.page_number))
        cursor = end + 2  # +2 for the "\n\n" join in full_text

    full_text = "\n\n".join(p.text for p in parsed.pages)
    full_text_clean = _RE_FOOTER.sub("", full_text)

    # Collect bold Section header positions from fitz
    bold_section_positions: set[str] = set()
    for page_idx in range(fitz_doc.page_count):
        page = fitz_doc[page_idx]
        for block in page.get_text("dict", sort=True)["blocks"]:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                txt = " ".join(s["text"] for s in line["spans"]).strip()
                m = _RE_SECTION_N.match(txt)
                if m:
                    is_bold = any(bool(s["flags"] & 2**4) for s in line["spans"])
                    if is_bold:
                        # Store "Section 4" as a key to validate regex hits in full_text
                        bold_section_positions.add(m.group(1).strip())

    # Find all section boundaries in full_text
    boundaries: list[tuple[int, str, str]] = []  # (char_pos, section_id, heading)
    for m in _RE_SECTION_N.finditer(full_text_clean):
        section_id = m.group(1).strip()        # "Section 4"
        heading_raw = m.group(2).strip()       # "Representations and Warranties..."
        # Validate: only count if this section ID was actually bold in fitz
        if section_id in bold_section_positions:
            boundaries.append((m.start(), section_id, heading_raw))

    if not boundaries:
        logger.warning("section_n strategy found 0 boundaries — falling back to prose")
        return _chunk_fallback_prose(parsed)

    chunks: list[Chunk] = []
    for i, (start_pos, section_id, heading) in enumerate(boundaries):
        end_pos = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(full_text_clean)
        clause_text = full_text_clean[start_pos:end_pos].strip()

        page_start = _char_pos_to_page(start_pos, page_offsets)
        page_end = _char_pos_to_page(end_pos - 1, page_offsets)

        # Top-level section chunk
        chunks.append(Chunk(
            clause_id=section_id,
            heading=_truncate_heading(heading),
            text=clause_text,
            page_start=page_start,
            page_end=page_end,
            format_used="section_n",
        ))

        # Sub-clause chunks within this section
        if include_sub_clauses:
            chunks.extend(
                _extract_sub_clauses(clause_text, section_id, page_start, page_end, "section_n")
            )

    return chunks


# ──────────────────────────────────────────────────────────────────
# Strategy 2: Bare "N. Title" headings (Affiliate Agreement)
# ──────────────────────────────────────────────────────────────────

def _chunk_bare_n(
    parsed: ParsedDocument,
    include_sub_clauses: bool,
) -> list[Chunk]:
    """
    Chunk by bare "N. Title" or "N.Title" lines.

    The Chase Affiliate Agreement uses no bold at all for headings —
    font size is identical to body. The ONLY signal is the numbering
    pattern. We can't use fitz bold validation here, so instead we
    filter false positives by requiring:
      1. The line starts at position 0 (not mid-paragraph)
      2. The number is sequential (gap of 1-2 from previous)
      3. The line is short (< 80 chars) — headings don't wrap

    Also handles bullet lists: bullets (•) belong to the current
    section and are NOT split into individual chunks (they're part
    of that section's clause text for embedding purposes).
    """
    page_offsets: list[tuple[int, int, int]] = []
    cursor = 0
    for page in parsed.pages:
        start = cursor
        end = cursor + len(page.text)
        page_offsets.append((start, end, page.page_number))
        cursor = end + 2

    full_text = "\n\n".join(p.text for p in parsed.pages)
    full_text_clean = _RE_FOOTER.sub("", full_text)

    # Replace non-breaking spaces with regular spaces
    full_text_clean = full_text_clean.replace("\xa0", " ")

    boundaries: list[tuple[int, str, str]] = []
    last_num = 0

    for m in _RE_BARE_N.finditer(full_text_clean):
        num = int(m.group(1))
        heading_raw = m.group(2).strip()

        # Sequence check: reject if number jumps by more than 3
        # (avoids matching "1,536 dollars" or similar mid-text numbers)
        if num < last_num or num > last_num + 3:
            continue

        # Short-line check: heading should end within 80 chars of the number
        line_start = full_text_clean.rfind("\n", 0, m.start()) + 1
        line_end = full_text_clean.find("\n", m.start())
        if line_end == -1:
            line_end = len(full_text_clean)
        line_len = line_end - line_start
        if line_len > 100:
            continue

        last_num = num
        boundaries.append((m.start(), str(num), heading_raw))

    if not boundaries:
        logger.warning("bare_n strategy found 0 boundaries — falling back to prose")
        return _chunk_fallback_prose(parsed)

    chunks: list[Chunk] = []
    for i, (start_pos, section_id, heading) in enumerate(boundaries):
        end_pos = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(full_text_clean)
        clause_text = full_text_clean[start_pos:end_pos].strip()

        page_start = _char_pos_to_page(start_pos, page_offsets)
        page_end = _char_pos_to_page(end_pos - 1, page_offsets)

        chunks.append(Chunk(
            clause_id=section_id,
            heading=_truncate_heading(heading),
            text=clause_text,
            page_start=page_start,
            page_end=page_end,
            format_used="bare_n",
        ))

        if include_sub_clauses:
            chunks.extend(
                _extract_sub_clauses(clause_text, section_id, page_start, page_end, "bare_n")
            )

    return chunks


# ──────────────────────────────────────────────────────────────────
# Strategy 3: oneNDA table-cell layout
# ──────────────────────────────────────────────────────────────────

def _chunk_onenda_table(
    parsed: ParsedDocument,
    fitz_doc: fitz.Document,
    include_sub_clauses: bool,
) -> list[Chunk]:
    """
    Chunk using fitz block layout directly, not flat text.

    WHY THIS STRATEGY IS DIFFERENT
    -------------------------------
    The oneNDA PDF is a table-based layout where:
      - Clause numbers ("1.", "2.") land as bold isolated blocks
      - Their titles ("What is Confidential Information?") are on the
        NEXT block at the same x0, also bold
      - Sub-clause labels "(a)", "(b)" are ALONE on their own lines
        with the actual content on the NEXT line at the same x0

    If we used flat text regex like the other strategies, we'd get:
      "(c)" as a standalone "clause" — because the text literally reads
      "(c)\nPermitted Receivers means..."

    Instead, we walk the fitz block stream and stitch adjacent blocks
    that belong together based on x0 alignment + bold continuity.

    FRONT MATTER HANDLING
    ----------------------
    The VARIABLES table (Purpose, Confidentiality period, Governing law)
    is extracted as a single "metadata" chunk rather than individual
    rows, because: (a) the values are often blank in templates, (b)
    they're contract-level metadata not clauses, (c) risk classification
    of "Governing law: [blank]" is meaningless.
    """
    chunks: list[Chunk] = []

    # Step 1: Extract the front-matter VARIABLES block as a metadata chunk
    meta_chunk = _extract_onenda_metadata(fitz_doc, parsed.pages[0])
    if meta_chunk:
        chunks.append(meta_chunk)

    # Step 2: Walk all pages collecting (text, bold, x0, page_num) tuples
    # for the TERMS section — which starts after the front matter page
    terms_lines: list[tuple[str, bool, float, int]] = []
    in_terms = False

    for page_idx in range(fitz_doc.page_count):
        page = fitz_doc[page_idx]
        page_num = page_idx + 1
        for block in page.get_text("dict", sort=True)["blocks"]:
            if block["type"] != 0:
                continue
            x0 = round(block["bbox"][0], 1)
            for line in block["lines"]:
                txt = " ".join(s["text"] for s in line["spans"]).strip()
                if not txt:
                    continue
                is_bold = any(bool(s["flags"] & 2**4) for s in line["spans"])

                # Mark start of TERMS section
                if txt == "TERMS" and is_bold:
                    in_terms = True
                    continue

                # Skip footer lines regardless
                if _RE_FOOTER.match(txt):
                    continue

                if in_terms:
                    terms_lines.append((txt, is_bold, x0, page_num))

    # Step 3: Walk terms_lines and stitch into clause chunks
    # A new top-level clause starts when: bold=True + text matches r'^\d+\.$'
    # and the NEXT non-empty line is also bold at same x0 (the title)
    current_clause_id: str | None = None
    current_heading: str = ""
    current_lines: list[str] = []
    current_page_start: int = 1
    current_page_end: int = 1

    i = 0
    while i < len(terms_lines):
        txt, bold, x0, pg = terms_lines[i]

        # Detect top-level clause boundary: isolated bold number "N."
        if bold and re.match(r'^\d+\.$', txt):
            # Save previous clause
            if current_clause_id is not None:
                clause_text = "\n".join(current_lines).strip()
                chunks.append(Chunk(
                    clause_id=f"Clause {current_clause_id}",
                    heading=current_heading,
                    text=clause_text,
                    page_start=current_page_start,
                    page_end=current_page_end,
                    format_used="onenda_table",
                ))

            clause_num = txt.rstrip(".")
            current_clause_id = clause_num
            current_page_start = pg
            current_page_end = pg

            # Try to grab the title from the next bold line at same x0
            heading_parts = [txt]  # start with "N."
            j = i + 1
            while j < len(terms_lines):
                ntxt, nbold, nx0, npg = terms_lines[j]
                if nbold and abs(nx0 - x0) < 5:
                    heading_parts.append(ntxt)
                    j += 1
                else:
                    break
            current_heading = " ".join(heading_parts[1:]) if len(heading_parts) > 1 else ""
            current_lines = [" ".join(heading_parts)]
            i = j
            continue

        # Stitch: sub-clause label alone on its line — peek forward for content
        if re.match(r'^\([a-z]{1,2}\)$', txt) or re.match(r'^\([ivxlc]+\)$', txt):
            label = txt
            # Collect following non-empty lines until next clause label or new top-level
            content_parts = []
            j = i + 1
            while j < len(terms_lines):
                ntxt, nbold, nx0, npg = terms_lines[j]
                # Stop if we hit another sub-clause label or top-level number
                if re.match(r'^\([a-z]{1,2}\)$', ntxt) or re.match(r'^\d+\.$', ntxt):
                    break
                # Stop if bold isolated number (new top-level)
                if nbold and re.match(r'^\d+\.$', ntxt):
                    break
                content_parts.append(ntxt)
                current_page_end = max(current_page_end, npg)
                j += 1
            stitched = label + " " + " ".join(content_parts)
            current_lines.append(stitched)
            i = j
            continue

        # Normal line: just append
        if current_clause_id is not None:
            current_lines.append(txt)
            current_page_end = max(current_page_end, pg)

        i += 1

    # Save the last clause
    if current_clause_id is not None:
        clause_text = "\n".join(current_lines).strip()
        chunks.append(Chunk(
            clause_id=f"Clause {current_clause_id}",
            heading=current_heading,
            text=clause_text,
            page_start=current_page_start,
            page_end=current_page_end,
            format_used="onenda_table",
        ))

    # Step 4: Optionally extract sub-clause chunks from each clause chunk
    if include_sub_clauses:
        sub_chunks = []
        for chunk in chunks:
            if chunk.clause_id.startswith("Clause "):
                sub_chunks.extend(
                    _extract_sub_clauses(
                        chunk.text, chunk.clause_id,
                        chunk.page_start, chunk.page_end,
                        "onenda_table"
                    )
                )
        chunks.extend(sub_chunks)

    return chunks


def _extract_onenda_metadata(fitz_doc: fitz.Document, page0: PageContent) -> Chunk | None:
    """
    Extract the VARIABLES table from oneNDA page 1 as a single metadata chunk.
    Returns None if the block can't be found (e.g. it's not a oneNDA-style doc).
    """
    page = fitz_doc[0]
    all_text = page.get_text("text", sort=True)

    var_start = all_text.find("VARIABLES")
    terms_start = all_text.find("TERMS")

    if var_start == -1:
        return None

    end = terms_start if terms_start > var_start else len(all_text)
    metadata_text = all_text[var_start:end].strip()
    metadata_text = _RE_FOOTER.sub("", metadata_text).strip()

    if not metadata_text:
        return None

    return Chunk(
        clause_id="Metadata",
        heading="Contract Variables (Parties, Purpose, Governing Law)",
        text=metadata_text,
        page_start=1,
        page_end=1,
        format_used="onenda_table",
    )


# ──────────────────────────────────────────────────────────────────
# Strategy 4: Fallback — paragraph-based chunking
# ──────────────────────────────────────────────────────────────────

def _chunk_fallback_prose(parsed: ParsedDocument) -> list[Chunk]:
    """
    Last-resort chunker: split on double newlines (paragraph breaks).

    Used when no numbered structure is detected — covers:
      - Pure-prose contracts with no section numbers
      - SEC filings / non-contract PDFs (with a warning logged earlier)
      - Contracts where numbering is too irregular to match

    Each paragraph becomes a chunk. Very short paragraphs (< 50 chars,
    e.g. page numbers, blank lines that slipped through) are merged
    into the previous chunk rather than emitted as their own chunk.

    Chunks get IDs like "Para 1", "Para 2" since there are no real
    clause IDs to extract.
    """
    full_text = "\n\n".join(p.text for p in parsed.pages)
    full_text_clean = _RE_FOOTER.sub("", full_text)

    paragraphs = [p.strip() for p in re.split(r'\n{2,}', full_text_clean)]
    paragraphs = [p for p in paragraphs if p]

    # Build page offset index for page attribution
    page_offsets: list[tuple[int, int, int]] = []
    cursor = 0
    for page in parsed.pages:
        start = cursor
        end = cursor + len(page.text)
        page_offsets.append((start, end, page.page_number))
        cursor = end + 2

    chunks: list[Chunk] = []
    para_num = 0
    carry = ""

    char_cursor = 0
    for para in paragraphs:
        if len(para) < 50:
            carry += " " + para
            char_cursor += len(para) + 2
            continue

        combined = (carry + " " + para).strip()
        carry = ""
        para_num += 1
        page_s = _char_pos_to_page(char_cursor, page_offsets)
        page_e = _char_pos_to_page(char_cursor + len(combined), page_offsets)

        heading = " ".join(combined.split()[:8])
        chunks.append(Chunk(
            clause_id=f"Para {para_num}",
            heading=heading,
            text=combined,
            page_start=page_s,
            page_end=page_e,
            format_used="fallback_prose",
        ))
        char_cursor += len(para) + 2

    return chunks


# ──────────────────────────────────────────────────────────────────
# Shared utilities
# ──────────────────────────────────────────────────────────────────

def _extract_sub_clauses(
    clause_text: str,
    parent_id: str,
    page_start: int,
    page_end: int,
    fmt: FormatType,
) -> list[Chunk]:
    """
    Split a top-level clause's text into sub-clause chunks at (a), (b)...
    boundaries. Each sub-clause gets an ID like "Section 4(a)".

    Sub-clauses all inherit the parent's page range because we can't
    cheaply recompute exact pages for mid-clause text — and for citation
    purposes "Section 4(a), page 8" (the section's start page) is
    accurate enough for the Q&A engine to direct a user.
    """
    sub_chunks: list[Chunk] = []
    boundaries: list[tuple[int, str]] = []

    for m in _RE_SUB_CLAUSE.finditer(clause_text):
        label = m.group(1)  # "a", "b", "i", "ii"
        boundaries.append((m.start(), label))

    if len(boundaries) < 2:
        # Only 0 or 1 sub-clause — not worth splitting
        return []

    for i, (start, label) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(clause_text)
        sub_text = clause_text[start:end].strip()
        sub_id = f"{parent_id}({label})"
        heading = " ".join(sub_text.split()[:10])

        sub_chunks.append(Chunk(
            clause_id=sub_id,
            heading=_truncate_heading(heading),
            text=sub_text,
            page_start=page_start,
            page_end=page_end,
            format_used=fmt,
        ))

    return sub_chunks


def _char_pos_to_page(pos: int, page_offsets: list[tuple[int, int, int]]) -> int:
    """
    Given a character position in the concatenated full_text, return
    the 1-indexed page number it falls on.
    Falls back to the last page if pos is beyond the end (rounding artifacts).
    """
    for start, end, pg in page_offsets:
        if start <= pos <= end:
            return pg
    return page_offsets[-1][2] if page_offsets else 1


def _truncate_heading(heading: str, max_words: int = 12) -> str:
    """Keep headings concise — long clause titles get truncated with '...'."""
    words = heading.split()
    if len(words) <= max_words:
        return heading
    return " ".join(words[:max_words]) + "..."


# ──────────────────────────────────────────────────────────────────
# Smoke test — run directly during development:
#   python src/chunker.py path/to/contract.pdf
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import logging

    # Allow running directly: add root/ to path so absolute imports work
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.pipeline.parser import parse_pdf

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python src/chunker.py <path_to_pdf>")
        sys.exit(1)

    path = sys.argv[1]
    parsed_doc = parse_pdf(path)
    chunks = chunk_document(parsed_doc, source=path)

    print(f"\nDocument: {parsed_doc.source_name}")
    print(f"Total chunks: {len(chunks)}")
    print(f"\nFirst 8 chunks:")
    for c in chunks[:8]:
        print(f"  {c}")
    print(f"\nLast 3 chunks:")
    for c in chunks[-3:]:
        print(f"  {c}")
