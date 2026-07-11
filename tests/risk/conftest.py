"""
Fixtures local to tests/risk/.

`ingested_contract` exists because TestScannerIntegration's tests were
calling scan_contract("oneNDA_v2.pdf") with no setup, silently relying
on a local data/metadata.db that only exists on a dev machine after a
manual ingest run. On a fresh CI checkout that file doesn't exist, so
scan_contract() found zero chunks and the assertion failed.

scan_contract() only reads `full_text` from SQLite — it never touches
ChromaDB/embeddings — so we can satisfy it cheaply by parsing + chunking
a real sample PDF straight into tmp_sqlite, with zero OpenAI API calls
spent on ingestion. The only OpenAI calls in these tests come from the
actual scan_contract()/scan_clauses() call itself.

The fixture is parametrized over two structurally different contracts
(a short table-style NDA and a long section-numbered agreement) rather
than one hardcoded oneNDA file, so passing tests demonstrate the scanner
works on contracts in general. It's capped at two — not all five in
ALL_SAMPLE_CONTRACTS — because each parametrization multiplies the
number of real OpenAI API calls these tests make, and that quota is
already tight (see the CI RESOURCE_EXHAUSTED warnings).
"""

from __future__ import annotations

from pathlib import Path

import pytest

SCANNER_TEST_CONTRACTS = [
    Path("legal_contracts/oneNDA_v2.pdf"),
    Path(
        "legal_contracts/CUAD_v1/full_contract_pdf/Part_II/"
        "Commercial Contracts (Part II-A)/Agency Agreements/"
        "ATHENSBANCSHARESCORP_AGENCY AGREEMENT.PDF"
    ),
]


@pytest.fixture(params=SCANNER_TEST_CONTRACTS, ids=lambda p: p.stem)
def ingested_contract(request, tmp_sqlite):
    """Parses + chunks a real sample contract and inserts the resulting
    chunks into tmp_sqlite, so scan_contract() has real clause text to
    read. Yields (conn, source_name) — tests run once per contract."""
    from src.pipeline.parser import parse_pdf
    from src.pipeline.chunker import chunk_document

    pdf_path = request.param
    if not pdf_path.exists():
        pytest.skip(f"Sample contract not found at {pdf_path}")

    source_name = pdf_path.name
    parsed = parse_pdf(pdf_path, source_name=source_name)
    chunks = chunk_document(parsed, source=pdf_path)

    for chunk in chunks:
        tmp_sqlite.execute(
            """INSERT OR IGNORE INTO chunks
               (id, source_name, clause_id, heading, full_text,
                page_start, page_end, format_used, char_count)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                f"{source_name}::{chunk.clause_id}",
                source_name,
                chunk.clause_id,
                chunk.heading,
                chunk.text,
                chunk.page_start,
                chunk.page_end,
                chunk.format_used,
                chunk.char_count,
            ),
        )
    tmp_sqlite.commit()
    return tmp_sqlite, source_name