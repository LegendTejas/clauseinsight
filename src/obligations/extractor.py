"""
ClauseInsight — Obligation & Deadline Extractor
==================================================

Scans every clause in a contract and extracts dated/deadline obligations
(renewal dates, notice periods, termination windows, payment deadlines,
auto-renewal terms) using GPT-4o-mini, producing structured Obligation
objects that the Streamlit dashboard displays.

PIPELINE POSITION
------------------
    SQLite (all chunks) → extractor.py → list[Obligation] → 4_obligations.py

This mirrors src/risk/scanner.py's shape deliberately: same batching
strategy, same retry/fallback handling, same in-memory result object
(no new SQLite table — like ScanResult, an ExtractionResult lives in
Streamlit session_state and is offered as a JSON download, not persisted).

EXTRACTION STRATEGY: BATCHED LLM CALLS, EVERY CLAUSE LABELLED
-----------------------------------------------------------------
Most clauses contain no obligation or deadline at all (governing law,
definitions, entire-agreement boilerplate, etc.) — expect roughly a
5-15% hit rate per contract. Rather than asking the LLM "find the
obligations" (which risks silently skipping clauses), each clause in
a batch is explicitly labelled `has_obligation: true/false`, exactly
like scanner.py labels every clause with a risk_level. This keeps the
same length-matching validation scanner.py relies on for retries, and
means a clause that's silently missing from the LLM's response is a
detectable parse failure, not indistinguishable from "no obligation".

PROMPT DESIGN
--------------
The system prompt injects:
  1. Obligation type definitions from obligation_labels.py
  2. Strict JSON output format with an example (has_obligation: true
     AND false examples, so the LLM sees both paths)

The user prompt injects the batch of clauses, each labelled with its
clause_id, same shape as scanner.py's _build_user_prompt.

ERROR HANDLING
---------------
Same three levels as scanner.py:
  1. JSON parse failure → retry the batch with a stricter prompt
  2. Individual field validation failure → coerce to nearest valid value
     (unknown obligation_type → OTHER_DEADLINE)
  3. Complete batch failure after retries → emit one Obligation per
     clause with confidence=0.0 so the UI can flag it for manual review
     rather than silently dropping the clause

RATE LIMITS & COST
-------------------
Same batching knobs as scanner.py (batch size 5, 4s inter-batch sleep) —
kept as a separate constant so the two features can be tuned independently
if one contract type needs a different batch size.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable

import openai

from pathlib import Path as _Path
import sys as _sys
_root_dir = str(_Path(__file__).resolve().parent.parent.parent)
if _root_dir not in _sys.path:
    _sys.path.insert(0, _root_dir)

from src.obligations.obligation_labels import (
    OBLIGATION_TYPE_DEFINITIONS,
    VALID_OBLIGATION_TYPES,
    Obligation,
    ObligationType,
)

from src.utils.store import (
    DEFAULT_SQLITE_PATH,
    get_all_chunks_for_contract,
    get_sqlite_connection,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────

LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.1"))

# Clauses per LLM call — separate knob from RISK_BATCH_SIZE so the two
# features can be tuned independently.
OBLIGATION_BATCH_SIZE = int(os.environ.get("OBLIGATION_BATCH_SIZE", "5"))

# Sleep between batches — same rate-limit guard as scanner.py
INTER_EXTRACT_SLEEP = 4.0   # seconds

# Retry config for failed batches
MAX_EXTRACT_RETRIES = 3
RETRY_DELAY = 5.0  # seconds


# ──────────────────────────────────────────────────────────────────
# Extraction result summary
# ──────────────────────────────────────────────────────────────────

@dataclass
class ExtractionResult:
    """Summary of one full contract obligation extraction."""
    source_name:         str
    total_clauses:        int
    obligations:          list[Obligation]
    failed_count:          int
    elapsed_seconds:        float

    @property
    def obligations_found(self) -> int:
        return len([o for o in self.obligations if not o.is_extraction_failure])

    @property
    def dated_count(self) -> int:
        return len([o for o in self.obligations if o.is_dated and not o.is_extraction_failure])

    @property
    def success(self) -> bool:
        return self.failed_count == 0

    def __str__(self) -> str:
        return (
            f"ExtractionResult({self.source_name!r}: "
            f"{self.obligations_found} obligations found "
            f"({self.dated_count} with fixed dates), "
            f"{self.failed_count} clauses failed extraction, "
            f"{self.elapsed_seconds:.1f}s)"
        )


# ──────────────────────────────────────────────────────────────────
# Prompt construction
# ──────────────────────────────────────────────────────────────────

def _build_system_prompt() -> str:
    """
    Build the system prompt injected into every extraction batch call.

    Includes obligation type definitions from obligation_labels.py so
    the LLM extracts consistently against our taxonomy — mirrors
    scanner.py's _build_system_prompt structure closely.
    """
    type_defs = "\n".join(
        f"- {t}: {defn}"
        for t, defn in OBLIGATION_TYPE_DEFINITIONS.items()
    )

    return f"""You are a legal contract analyser specialising in finding dates,
deadlines, and time-bound obligations. Your job is to read each clause and
determine whether it creates a dated or deadline-bearing obligation.

OBLIGATION TYPES:
{type_defs}

For each clause provided, return a JSON object with exactly these fields:
  - clause_id: the clause identifier as provided (string)
  - has_obligation: true if this clause contains a date, deadline, notice
            period, or renewal/payment term; false otherwise (boolean)
  - obligation_type: one of the type names listed above. Required if
            has_obligation is true, otherwise null.
  - description: 1-2 sentence plain-English summary of the obligation,
            written for a non-lawyer to understand. Required if
            has_obligation is true, otherwise null.
  - date_value: an ABSOLUTE calendar date in "YYYY-MM-DD" format, ONLY if
            the clause states one explicitly (e.g. "expires January 15,
            2027" -> "2027-01-15"). Otherwise null. Never guess or infer
            a date that isn't explicitly stated in the clause text.
  - period_value: a RELATIVE time period as plain text, ONLY if the clause
            specifies one instead of an absolute date (e.g. "30 days
            written notice", "net 30 from invoice date", "successive
            1-year terms"). Otherwise null.
  - confidence: your confidence in this extraction, 0.0 to 1.0 (number)

Most clauses do NOT contain obligations — governing law, definitions,
entire-agreement, and similar boilerplate clauses should be marked
has_obligation: false. Only mark true for clauses that genuinely create
a time-bound action or deadline for one of the parties.

A clause typically has EITHER date_value OR period_value, not both —
use whichever the clause text actually states. If neither a specific
date nor a specific period is stated, set has_obligation to false even
if the clause discusses a related topic (e.g. a clause that only
defines "Effective Date" as a term, without stating what it is, is
NOT an obligation).

Return ONLY a valid JSON array containing one object per clause.
Do NOT include markdown code fences, preamble, or explanation outside the JSON.
Do NOT skip any clause — every clause_id in the input must appear in the output.

Example output format:
[
  {{
    "clause_id": "Section 12",
    "has_obligation": true,
    "obligation_type": "Auto-Renewal",
    "description": "The agreement automatically renews for successive 1-year terms unless either party gives 60 days written notice before the renewal date.",
    "date_value": null,
    "period_value": "60 days written notice before renewal",
    "confidence": 0.95
  }},
  {{
    "clause_id": "Section 13",
    "has_obligation": false,
    "obligation_type": null,
    "description": null,
    "date_value": null,
    "period_value": null,
    "confidence": 0.9
  }}
]"""


def _build_user_prompt(batch: list[dict]) -> str:
    """
    Build the user prompt for one batch of clauses.

    Identical shape to scanner.py's _build_user_prompt — each clause
    is presented with its clause_id and full text.
    """
    lines = ["Analyse the following contract clauses for dated obligations:\n"]
    for i, chunk in enumerate(batch, start=1):
        lines.append(f"--- CLAUSE {i}: {chunk['clause_id']} ---")
        lines.append((chunk.get("full_text") or chunk.get("text") or "").strip())
        lines.append("")  # blank line between clauses
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────
# LLM call + response parsing
# ──────────────────────────────────────────────────────────────────

def _make_openai_client() -> openai.OpenAI:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable is not set."
        )
    return openai.OpenAI(api_key=api_key)


def _call_llm(
    client: openai.OpenAI,
    system_prompt: str,
    user_prompt: str,
) -> str:
    """
    Call OpenAI and return the raw response text.
    Raises on API errors — caller handles retries.
    """
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=LLM_TEMPERATURE,
        max_tokens=2048,
    )
    if not response.choices or not response.choices[0].message.content:
        raise ValueError("LLM returned an empty response (no text).")
    return response.choices[0].message.content


def _parse_llm_response(
    raw_text: str,
    batch: list[dict],
    source_name: str,
) -> list[dict]:
    """
    Parse the LLM's JSON array response into a list of per-clause result
    dicts, one per item the LLM returned.

    Unlike scanner.py's _parse_llm_response (which returns RiskLabel
    objects directly, one per clause, always), this returns raw dicts
    with a `matched` flag so the caller can both (a) check whether every
    clause_id got a response, for retry purposes, and (b) build Obligation
    objects only for has_obligation=true items.

    Handles the same failure modes as scanner.py:
      - Accidental markdown fences (```json ... ```)
      - Unknown obligation_type names → coerced to OTHER_DEADLINE
      - Missing clause_ids → matched by position in batch
      - Extra fields in LLM output → silently ignored

    Returns:
        List of dicts with keys: clause_id, obligation (Obligation | None)
        Length should equal len(batch) on success — caller checks this.
    """
    text = raw_text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*```$', '', text, flags=re.MULTILINE)
    text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.error("JSON parse failed: %s\nRaw response:\n%s", exc, raw_text[:500])
        return []

    if not isinstance(data, list):
        logger.error("LLM returned non-list JSON: %s", type(data))
        return []

    batch_lookup = {row["clause_id"]: row for row in batch}

    results: list[dict] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue

        clause_id = item.get("clause_id", "")

        # Fall back to positional match if clause_id missing or wrong
        if clause_id not in batch_lookup and i < len(batch):
            clause_id = batch[i]["clause_id"]
            logger.debug(
                "clause_id mismatch in LLM response — using positional fallback: %r",
                clause_id
            )

        row = batch_lookup.get(clause_id, batch[i] if i < len(batch) else {})

        has_obligation = bool(item.get("has_obligation", False))

        if not has_obligation:
            results.append({"clause_id": clause_id, "obligation": None})
            continue

        # Validate + coerce obligation_type
        raw_type = str(item.get("obligation_type", "")).strip()
        if raw_type in VALID_OBLIGATION_TYPES:
            obligation_type = ObligationType(raw_type)
        else:
            logger.warning(
                "Unknown obligation_type %r for %r — defaulting to Other Deadline",
                raw_type, clause_id
            )
            obligation_type = ObligationType.OTHER_DEADLINE

        description = str(item.get("description") or "No description provided.").strip()
        date_value = item.get("date_value") or None
        period_value = item.get("period_value") or None

        confidence_raw = item.get("confidence")
        try:
            confidence = float(confidence_raw) if confidence_raw is not None else None
        except (TypeError, ValueError):
            confidence = None

        obligation = Obligation(
            clause_id=clause_id,
            source_name=source_name,
            obligation_type=obligation_type,
            description=description,
            date_value=date_value,
            period_value=period_value,
            page_start=row.get("page_start", 0),
            page_end=row.get("page_end", 0),
            heading=row.get("heading", ""),
            confidence=confidence,
        )
        results.append({"clause_id": clause_id, "obligation": obligation})

    return results


def _fallback_results(batch: list[dict], source_name: str) -> list[dict]:
    """
    Generate fallback results for a batch that failed after all retries.

    Unlike scanner.py's UNKNOWN risk level (which is a valid label the UI
    displays), a failed obligation extraction is reported as a flagged
    Obligation with confidence=0.0 so the UI can surface "extraction
    failed — review manually" rather than silently reporting zero
    obligations for a clause that might genuinely have one.
    """
    return [
        {
            "clause_id": row["clause_id"],
            "obligation": Obligation(
                clause_id=row["clause_id"],
                source_name=source_name,
                obligation_type=ObligationType.OTHER_DEADLINE,
                description="Extraction failed — API error or response parse failure.",
                date_value=None,
                period_value=None,
                page_start=row.get("page_start", 0),
                page_end=row.get("page_end", 0),
                heading=row.get("heading", ""),
                confidence=0.0,
            ),
        }
        for row in batch
    ]


# ──────────────────────────────────────────────────────────────────
# Batch processing
# ──────────────────────────────────────────────────────────────────

def _extract_batch(
    client: openai.OpenAI,
    batch: list[dict],
    source_name: str,
    system_prompt: str,
) -> list[dict]:
    """
    Extract obligations from one batch of clauses, with retries.

    Returns a list of per-clause result dicts — falls back to flagged
    failure Obligations if all retries fail, mirroring scanner.py's
    _scan_batch structure.
    """
    user_prompt = _build_user_prompt(batch)

    for attempt in range(1, MAX_EXTRACT_RETRIES + 1):
        delay = RETRY_DELAY * (2 ** (attempt - 1))  # reset per-attempt: 5s, 10s, 20s
        try:
            raw = _call_llm(client, system_prompt, user_prompt)
            results = _parse_llm_response(raw, batch, source_name)

            if len(results) > 0:
                if len(results) != len(batch):
                    logger.info("LLM returned %d results for %d clauses. Accepting partial/merged parse.", len(results), len(batch))
                return results  # success — no sleep here, inter-batch sleep is in extract_obligations

            logger.warning(
                "Batch parse returned 0 results on attempt %d. Retrying...",
                attempt
            )
            time.sleep(delay)

        except Exception as exc:
            err_str = str(exc).lower()
            is_rate_limit = any(x in err_str for x in ["429", "quota", "rate"])
            logger.warning(
                "Extraction attempt %d/%d failed: %s. Retrying in %.1fs...",
                attempt, MAX_EXTRACT_RETRIES, exc, delay
            )
            if is_rate_limit:
                time.sleep(delay * 2)
            else:
                time.sleep(delay)

    logger.error(
        "Batch failed after %d attempts — emitting failure flags for: %s",
        MAX_EXTRACT_RETRIES,
        [r["clause_id"] for r in batch],
    )
    return _fallback_results(batch, source_name)


# ──────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────

def extract_obligations(
    source_name: str,
    conn: Optional[sqlite3.Connection] = None,
    db_path: Path = DEFAULT_SQLITE_PATH,
    batch_size: int = OBLIGATION_BATCH_SIZE,
    skip_sub_clauses: bool = True,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> ExtractionResult:
    """
    Extract all dated obligations/deadlines from a contract's clauses.

    Reads clause text from SQLite (where full_text is stored), sends
    batches to the LLM, and returns an ExtractionResult with all found
    Obligations. Mirrors scanner.py's scan_contract() shape closely.

    Args:
        source_name:      Contract filename — must match what was ingested.
        conn:             Pre-opened SQLite connection. If None, opens one.
        db_path:          SQLite path (used if conn is None).
        batch_size:       Clauses per LLM call. Default 5.
        skip_sub_clauses: If True, only scan top-level clauses (not sub-clauses
                          like Section 4(a)). Reduces API calls significantly.
                          Set False for maximum granularity.
        is_cancelled:     Optional function that returns True if extraction should abort early.

    Returns:
        ExtractionResult with all found Obligation objects and summary counts.
    """
    t_start = time.time()

    if conn is None:
        conn = get_sqlite_connection(db_path)

    all_chunks = get_all_chunks_for_contract(
        source_name, conn, include_text=True
    )

    if not all_chunks:
        logger.warning(
            "No chunks found for '%s'. Has it been ingested?", source_name
        )
        return ExtractionResult(
            source_name=source_name,
            total_clauses=0,
            obligations=[],
            failed_count=0,
            elapsed_seconds=0.0,
        )

    if skip_sub_clauses:
        chunks_to_scan = [
            c for c in all_chunks
            if not re.search(r'\([a-z]{1,2}\)$|\([ivxlc]+\)$', c["clause_id"])
        ]
        logger.info(
            "'%s': extracting from %d/%d top-level clauses (sub-clauses skipped)",
            source_name, len(chunks_to_scan), len(all_chunks)
        )
    else:
        chunks_to_scan = all_chunks
        logger.info(
            "'%s': extracting from all %d clauses", source_name, len(chunks_to_scan)
        )

    client = _make_openai_client()
    system_prompt = _build_system_prompt()
    all_obligations: list[Obligation] = []
    failed_count = 0

    total_batches = (len(chunks_to_scan) + batch_size - 1) // batch_size

    for batch_num, batch_start in enumerate(
        range(0, len(chunks_to_scan), batch_size), start=1
    ):
        if is_cancelled and is_cancelled():
            logger.info("Extraction cancelled for '%s'", source_name)
            break

        batch = chunks_to_scan[batch_start: batch_start + batch_size]

        logger.info(
            "'%s': extracting batch %d/%d (clauses: %s)",
            source_name, batch_num, total_batches,
            [c["clause_id"] for c in batch],
        )

        results = _extract_batch(client, batch, source_name, system_prompt)
        for r in results:
            obligation = r.get("obligation")
            if obligation is not None:
                all_obligations.append(obligation)
                if obligation.is_extraction_failure:
                    failed_count += 1

        if progress_callback:
            progress_callback(batch_num, total_batches)

        if batch_start + batch_size < len(chunks_to_scan):
            time.sleep(INTER_EXTRACT_SLEEP)

    elapsed = time.time() - t_start
    result = ExtractionResult(
        source_name=source_name,
        total_clauses=len(chunks_to_scan),
        obligations=all_obligations,
        failed_count=failed_count,
        elapsed_seconds=elapsed,
    )
    logger.info("extract_obligations complete: %s", result)
    return result


# ──────────────────────────────────────────────────────────────────
# Smoke test:
#   OPENAI_API_KEY=your_key python src/obligations/extractor.py oneNDA_v2.pdf
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import logging
    from dotenv import load_dotenv
    load_dotenv()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    source = sys.argv[1] if len(sys.argv) > 1 else "oneNDA_v2.pdf"

    print(f"\nExtracting obligations: {source!r}")
    print(f"Batch size: {OBLIGATION_BATCH_SIZE} clauses/call\n")

    result = extract_obligations(source)

    print(f"\n{'='*50}")
    print(f"EXTRACTION COMPLETE: {result}")
    print(f"{'='*50}")
    print(f"  Obligations found : {result.obligations_found}")
    print(f"  With fixed dates  : {result.dated_count}")
    print(f"  Failed clauses    : {result.failed_count}")
    print(f"  Time              : {result.elapsed_seconds:.1f}s")

    real = [o for o in result.obligations if not o.is_extraction_failure]
    if real:
        print(f"\nObligations found ({len(real)}):")
        for ob in sorted(real, key=lambda o: (o.date_value or "9999-99-99")):
            print(f"\n  {ob}")
            print(f"    Description: {ob.description}")
