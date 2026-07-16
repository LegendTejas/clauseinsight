"""
ClauseInsight — Risk Scanner
==============================

Classifies every clause in a contract as LOW / MEDIUM / HIGH risk
using GPT-4o-mini, producing structured RiskLabel objects that
the Streamlit dashboard displays.

PIPELINE POSITION
------------------
    SQLite (all chunks) → scanner.py → list[RiskLabel] → 3_scanner.py

CLASSIFICATION STRATEGY: BATCHED LLM CALLS
--------------------------------------------
Processing clauses one-at-a-time (one LLM call per clause) is the most
accurate approach but prohibitively slow for large contracts — a 161-chunk
agency agreement would take ~161 API calls and ~10 minutes.

Instead, we batch N clauses per LLM call (default SCAN_BATCH_SIZE=5).
Each batch is sent as a single structured prompt asking the LLM to return
a JSON array with one classification object per clause.

Trade-off: batch size vs. accuracy
  - Too large (20+): LLM attention dilutes across many clauses,
    later clauses in the batch get less careful analysis
  - Too small (1-2): defeats the purpose of batching
  - Sweet spot (5): each clause gets meaningful attention, 33 chunks
    from oneNDA = 7 API calls instead of 33

PROMPT DESIGN
--------------
The system prompt injects:
  1. Risk level definitions (LOW/MEDIUM/HIGH) from risk_labels.py
  2. Category definitions (all 20 types) from risk_labels.py
  3. Strict JSON output format with an example

The user prompt injects the batch of clauses, each labelled with its
clause_id for the LLM to reference in its output.

The LLM is instructed to return ONLY a JSON array — no markdown fences,
no preamble. The parser strips any accidental fences before parsing.

ERROR HANDLING
---------------
Three levels:
  1. JSON parse failure → retry the batch with a stricter prompt
  2. Individual field validation failure → coerce to nearest valid value
     (unknown category → GENERAL, unknown risk level → UNKNOWN)
  3. Complete batch failure after retries → emit RiskLabel with
     UNKNOWN risk level so the UI can flag it rather than silently skip

RATE LIMITS & COST
-------------------
OpenAI usage is billed per request/token — there's no free tier.
With SCAN_BATCH_SIZE=5 and INTER_SCAN_SLEEP=4.0s between batches,
a 33-chunk contract makes 7 API calls over ~28 seconds.
A 161-chunk contract makes ~33 calls over ~2 minutes. Keep contract
volume in mind since each batch call consumes billed tokens.
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

from src.risk.risk_labels import (
    CATEGORY_DEFINITIONS,
    RISK_LEVEL_DEFINITIONS,
    VALID_CATEGORIES,
    VALID_RISK_LEVELS,
    ClauseCategory,
    RiskLabel,
    RiskLevel,
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
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.2"))

# Clauses per LLM call — sweet spot between speed and accuracy
SCAN_BATCH_SIZE = int(os.environ.get("RISK_BATCH_SIZE", "5"))

# Sleep between batches — keeps us under 15 req/min free tier limit
INTER_SCAN_SLEEP = 4.0   # seconds

# Retry config for failed batches
MAX_SCAN_RETRIES = 3
RETRY_DELAY = 5.0  # seconds


# ──────────────────────────────────────────────────────────────────
# Scan result summary
# ──────────────────────────────────────────────────────────────────

@dataclass
class ScanResult:
    """Summary of one full contract scan."""
    source_name:   str
    total_clauses: int
    high_count:    int
    medium_count:  int
    low_count:     int
    unknown_count: int
    labels:        list[RiskLabel]
    elapsed_seconds: float

    @property
    def flagged_count(self) -> int:
        return self.high_count + self.medium_count

    @property
    def success(self) -> bool:
        return self.unknown_count == 0

    def __str__(self) -> str:
        return (
            f"ScanResult({self.source_name!r}: "
            f"{self.high_count} HIGH, {self.medium_count} MEDIUM, "
            f"{self.low_count} LOW, {self.unknown_count} UNKNOWN, "
            f"{self.elapsed_seconds:.1f}s)"
        )


# ──────────────────────────────────────────────────────────────────
# Prompt construction
# ──────────────────────────────────────────────────────────────────

def _build_system_prompt() -> str:
    """
    Build the system prompt injected into every scan batch call.

    Includes risk level definitions and category definitions from
    risk_labels.py so the LLM classifies consistently against our
    taxonomy — not its own internal notion of what 'HIGH' means.
    """
    risk_defs = "\n".join(
        f"- {level}: {defn}"
        for level, defn in RISK_LEVEL_DEFINITIONS.items()
    )

    category_defs = "\n".join(
        f"- {cat}: {defn}"
        for cat, defn in CATEGORY_DEFINITIONS.items()
    )

    return f"""You are a legal contract risk analyser. Your job is to classify
contract clauses by risk level and provide actionable guidance.

RISK LEVELS:
{risk_defs}

CLAUSE CATEGORIES:
{category_defs}

For each clause provided, return a JSON object with exactly these fields:
  - clause_id: the clause identifier as provided (string)
  - risk_level: one of "LOW", "MEDIUM", "HIGH" (string)
  - category: one of the category names listed above (string)
  - reason: 1-2 sentences explaining WHY this risk level was assigned,
            written for a non-lawyer to understand (string)
  - recommended_action: a concrete next step for the reader (string).
            For LOW risk, this can be "No action required."

Return ONLY a valid JSON array containing one object per clause.
Do NOT include markdown code fences, preamble, or explanation outside the JSON.
Do NOT skip any clause — every clause_id in the input must appear in the output.

Example output format:
[
  {{
    "clause_id": "Section 4",
    "risk_level": "HIGH",
    "category": "Indemnification",
    "reason": "This clause requires unlimited indemnification with no cap on liability.",
    "recommended_action": "Negotiate a liability cap tied to contract value."
  }}
]"""


def _build_user_prompt(batch: list[dict]) -> str:
    """
    Build the user prompt for one batch of clauses.

    Each clause is presented with its clause_id and full text so the
    LLM has the complete content to analyse, not just a preview.
    """
    lines = ["Classify the following contract clauses:\n"]
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
) -> list[RiskLabel]:
    """
    Parse the LLM's JSON array response into RiskLabel objects.

    Handles:
      - Accidental markdown fences (```json ... ```)
      - Unknown category names → coerced to GENERAL
      - Unknown risk level names → coerced to UNKNOWN
      - Missing clause_ids → matched by position in batch
      - Extra fields in LLM output → silently ignored
    """
    # Strip accidental markdown fences
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

    # Build a lookup from clause_id → batch row for metadata
    batch_lookup = {row["clause_id"]: row for row in batch}

    labels: list[RiskLabel] = []
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

        # Validate + coerce risk_level
        raw_risk = str(item.get("risk_level", "")).upper().strip()
        if raw_risk in VALID_RISK_LEVELS:
            risk_level = RiskLevel(raw_risk)
        else:
            logger.warning(
                "Unknown risk_level %r for %r — defaulting to UNKNOWN", raw_risk, clause_id
            )
            risk_level = RiskLevel.UNKNOWN

        # Validate + coerce category
        raw_cat = str(item.get("category", "")).strip()
        if raw_cat in VALID_CATEGORIES:
            category = ClauseCategory(raw_cat)
        else:
            logger.warning(
                "Unknown category %r for %r — defaulting to GENERAL", raw_cat, clause_id
            )
            category = ClauseCategory.GENERAL

        reason = str(item.get("reason", "No reason provided.")).strip()
        recommended_action = str(
            item.get("recommended_action", "No action specified.")
        ).strip()

        labels.append(RiskLabel(
            clause_id=clause_id,
            source_name=source_name,
            risk_level=risk_level,
            category=category,
            reason=reason,
            recommended_action=recommended_action,
            page_start=row.get("page_start", 0),
            page_end=row.get("page_end", 0),
            heading=row.get("heading", ""),
        ))

    return labels


def _fallback_labels(batch: list[dict], source_name: str) -> list[RiskLabel]:
    """
    Generate UNKNOWN RiskLabels for a batch that failed after all retries.
    Ensures every clause gets a label even on total API failure.
    """
    return [
        RiskLabel(
            clause_id=row["clause_id"],
            source_name=source_name,
            risk_level=RiskLevel.UNKNOWN,
            category=ClauseCategory.GENERAL,
            reason="Classification failed — API error or response parse failure.",
            recommended_action="Re-run the risk scanner or review this clause manually.",
            page_start=row.get("page_start", 0),
            page_end=row.get("page_end", 0),
            heading=row.get("heading", ""),
        )
        for row in batch
    ]


# ──────────────────────────────────────────────────────────────────
# Batch processing
# ──────────────────────────────────────────────────────────────────

def _scan_batch(
    client: openai.OpenAI,
    batch: list[dict],
    source_name: str,
    system_prompt: str,
) -> list[RiskLabel]:
    """
    Classify one batch of clauses with retries.

    Returns RiskLabel list — falls back to UNKNOWN labels if all retries fail.
    """
    user_prompt = _build_user_prompt(batch)

    for attempt in range(1, MAX_SCAN_RETRIES + 1):
        delay = RETRY_DELAY * (2 ** (attempt - 1))  # reset per-attempt: 5s, 10s, 20s
        try:
            raw = _call_llm(client, system_prompt, user_prompt)
            labels = _parse_llm_response(raw, batch, source_name)

            if len(labels) > 0:
                if len(labels) != len(batch):
                    logger.info("LLM returned %d labels for %d clauses. Accepting partial/merged parse.", len(labels), len(batch))
                return labels  # success — no sleep here, inter-batch sleep is in scan_contract

            logger.warning(
                "Batch parse returned 0 labels on attempt %d. Retrying...",
                attempt
            )
            time.sleep(delay)

        except Exception as exc:
            err_str = str(exc).lower()
            is_rate_limit = any(x in err_str for x in ["429", "quota", "rate"])
            logger.warning(
                "Scan attempt %d/%d failed: %s. Retrying in %.1fs...",
                attempt, MAX_SCAN_RETRIES, exc, delay
            )
            if is_rate_limit:
                time.sleep(delay * 2)
            else:
                time.sleep(delay)

    logger.error(
        "Batch failed after %d attempts — emitting UNKNOWN labels for: %s",
        MAX_SCAN_RETRIES,
        [r["clause_id"] for r in batch],
    )
    return _fallback_labels(batch, source_name)


# ──────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────

def scan_contract(
    source_name: str,
    conn: Optional[sqlite3.Connection] = None,
    db_path: Path = DEFAULT_SQLITE_PATH,
    batch_size: int = SCAN_BATCH_SIZE,
    skip_sub_clauses: bool = True,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> ScanResult:
    """
    Scan all clauses of a contract and classify each by risk level.

    Reads clause text from SQLite (where full_text is stored), sends
    batches to the LLM, and returns a ScanResult with all RiskLabels.

    Args:
        source_name:      Contract filename — must match what was ingested.
        conn:             Pre-opened SQLite connection. If None, opens one.
        db_path:          SQLite path (used if conn is None).
        batch_size:       Clauses per LLM call. Default 5.
        skip_sub_clauses: If True, only scan top-level clauses (not sub-clauses
                          like Section 4(a)). Reduces API calls significantly
                          and avoids redundant classifications — the scanner
                          classifies the parent which covers the sub-clause.
                          Set False for maximum granularity.
        is_cancelled:     Optional function that returns True if the scan should abort early.

    Returns:
        ScanResult with all RiskLabel objects and summary counts.
    """
    t_start = time.time()

    if conn is None:
        conn = get_sqlite_connection(db_path)

    # Fetch all chunks with full text
    all_chunks = get_all_chunks_for_contract(
        source_name, conn, include_text=True
    )

    if not all_chunks:
        logger.warning(
            "No chunks found for '%s'. Has it been ingested?", source_name
        )
        return ScanResult(
            source_name=source_name,
            total_clauses=0,
            high_count=0, medium_count=0, low_count=0, unknown_count=0,
            labels=[], elapsed_seconds=0.0,
        )

    # Optionally filter to top-level clauses only
    if skip_sub_clauses:
        chunks_to_scan = [
            c for c in all_chunks
            if not re.search(r'\([a-z]{1,2}\)$|\([ivxlc]+\)$', c["clause_id"])
        ]
        logger.info(
            "'%s': scanning %d/%d top-level clauses (sub-clauses skipped)",
            source_name, len(chunks_to_scan), len(all_chunks)
        )
    else:
        chunks_to_scan = all_chunks
        logger.info(
            "'%s': scanning all %d clauses", source_name, len(chunks_to_scan)
        )

    client = _make_openai_client()
    system_prompt = _build_system_prompt()
    all_labels: list[RiskLabel] = []

    total_batches = (len(chunks_to_scan) + batch_size - 1) // batch_size

    for batch_num, batch_start in enumerate(
        range(0, len(chunks_to_scan), batch_size), start=1
    ):
        if is_cancelled and is_cancelled():
            logger.info("Scan cancelled for '%s'", source_name)
            break

        batch = chunks_to_scan[batch_start: batch_start + batch_size]

        logger.info(
            "'%s': scanning batch %d/%d (clauses: %s)",
            source_name, batch_num, total_batches,
            [c["clause_id"] for c in batch],
        )

        labels = _scan_batch(client, batch, source_name, system_prompt)
        all_labels.extend(labels)

        if progress_callback:
            progress_callback(batch_num, total_batches)

        # Rate limit guard between batches
        if batch_start + batch_size < len(chunks_to_scan):
            time.sleep(INTER_SCAN_SLEEP)

    # Tally counts
    high   = sum(1 for l in all_labels if l.risk_level == RiskLevel.HIGH)
    medium = sum(1 for l in all_labels if l.risk_level == RiskLevel.MEDIUM)
    low    = sum(1 for l in all_labels if l.risk_level == RiskLevel.LOW)
    unknown = sum(1 for l in all_labels if l.risk_level == RiskLevel.UNKNOWN)

    elapsed = time.time() - t_start
    result = ScanResult(
        source_name=source_name,
        total_clauses=len(all_labels),
        high_count=high,
        medium_count=medium,
        low_count=low,
        unknown_count=unknown,
        labels=all_labels,
        elapsed_seconds=elapsed,
    )
    logger.info("scan_contract complete: %s", result)
    return result


# ──────────────────────────────────────────────────────────────────
# Convenience: scan a specific list of clause IDs only
# (used by the Q&A page to scan just the retrieved clauses)
# ──────────────────────────────────────────────────────────────────

def scan_clauses(
    clause_ids: list[str],
    source_name: str,
    conn: sqlite3.Connection,
    batch_size: int = SCAN_BATCH_SIZE,
) -> list[RiskLabel]:
    """
    Scan a specific subset of clauses by their clause_id.

    Used by the Q&A page (2_qa.py) to run risk classification on just
    the retrieved chunks so the user sees risk context alongside the answer.

    Args:
        clause_ids:  List of clause_id strings to classify.
        source_name: Contract filename.
        conn:        Open SQLite connection.
        batch_size:  Clauses per LLM call.

    Returns:
        List of RiskLabel, one per clause_id. Order matches input.
    """
    all_chunks = get_all_chunks_for_contract(source_name, conn, include_text=True)
    lookup = {c["clause_id"]: c for c in all_chunks}

    chunks_to_scan = [
        lookup[cid] for cid in clause_ids if cid in lookup
    ]

    if not chunks_to_scan:
        return []

    client = _make_openai_client()
    system_prompt = _build_system_prompt()
    labels: list[RiskLabel] = []

    for batch_start in range(0, len(chunks_to_scan), batch_size):
        batch = chunks_to_scan[batch_start: batch_start + batch_size]
        labels.extend(_scan_batch(client, batch, source_name, system_prompt))
        if batch_start + batch_size < len(chunks_to_scan):
            time.sleep(INTER_SCAN_SLEEP)

    return labels


# ──────────────────────────────────────────────────────────────────
# Smoke test:
#   OPENAI_API_KEY=your_key python src/risk/scanner.py oneNDA_v2.pdf
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import logging
    from dotenv import load_dotenv
    load_dotenv()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    source = sys.argv[1] if len(sys.argv) > 1 else "oneNDA_v2.pdf"

    print(f"\nScanning: {source!r}")
    print(f"Batch size: {SCAN_BATCH_SIZE} clauses/call\n")

    result = scan_contract(source)

    print(f"\n{'='*50}")
    print(f"SCAN COMPLETE: {result}")
    print(f"{'='*50}")
    print(f"  HIGH   : {result.high_count}")
    print(f"  MEDIUM : {result.medium_count}")
    print(f"  LOW    : {result.low_count}")
    print(f"  UNKNOWN: {result.unknown_count}")
    print(f"  Time   : {result.elapsed_seconds:.1f}s")

    if result.labels:
        print(f"\nFlagged clauses ({result.flagged_count}):")
        for label in sorted(result.labels, key=lambda l: l.risk_level.value):
            if label.is_flagged:
                print(f"\n  {label}")
                print(f"    Reason : {label.reason}")
                print(f"    Action : {label.recommended_action}")
