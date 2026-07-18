"""
ClauseInsight — Executive Summary
====================================

Combines an already-computed ScanResult and ExtractionResult into a
single "read this in 20 seconds" summary — deliberately WITHOUT a new
LLM call.

WHY NO LLM CALL
-----------------
By the time this runs, scanner.py has already produced a human-written
`reason` for every HIGH/MEDIUM risk clause, and extractor.py has already
produced a human-written `description` for every obligation — both
written by an LLM at classification/extraction time. Re-summarizing
that already-summarized text through a second LLM call would just add
cost, latency, and a fresh chance to hallucinate, for a result made
entirely from text this module could just select and format directly.

This module is pure aggregation: it counts, sorts, and picks — string
formatting, not generation. Same input always produces the same output,
there's no API call to fail or rate-limit, and it's instant.

WHY THIS DOESN'T TRIGGER SCANNING/EXTRACTION ITSELF
--------------------------------------------------------
This module only ever reads a ScanResult / ExtractionResult that some
other page already produced (scanner.py or extractor.py, both still
opt-in, user-triggered actions — see their own module docstrings for
why). If neither is available yet, build_executive_summary() returns a
summary in "teaser" mode (has_risk_data / has_obligation_data both
False) rather than running anything itself — the UI is responsible for
prompting the user to go run those pages, not this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from pathlib import Path as _Path
import sys as _sys
_root_dir = str(_Path(__file__).resolve().parent.parent.parent)
if _root_dir not in _sys.path:
    _sys.path.insert(0, _root_dir)

from src.risk.scanner import ScanResult
from src.risk.risk_labels import RiskLabel
from src.obligations.extractor import ExtractionResult
from src.obligations.obligation_labels import Obligation


# ──────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────

TOP_RISKS_COUNT = 3
UPCOMING_DEADLINES_COUNT = 3


# ──────────────────────────────────────────────────────────────────
# Result type
# ──────────────────────────────────────────────────────────────────

@dataclass
class ExecutiveSummary:
    """
    A 20-second read of a contract's risk + obligation picture.

    has_risk_data / has_obligation_data tell the UI whether each half
    of the summary is real (the user already ran that scan/extraction
    on its own page) or missing (should render as a teaser prompting
    the user to go run it) — a summary can be fully populated, half
    populated, or empty; the UI decides how to render each state.
    """
    source_name: str

    has_risk_data: bool
    total_clauses_scanned: int
    risk_counts: dict[str, int] = field(default_factory=dict)  # "HIGH"/"MEDIUM"/"LOW"/"UNKNOWN" -> count
    top_risks: list[RiskLabel] = field(default_factory=list)

    has_obligation_data: bool = False
    upcoming_deadlines: list[Obligation] = field(default_factory=list)

    verdict: str = ""
    verdict_level: str = "unknown"  # "urgent" | "caution" | "clear" | "unknown"

    @property
    def is_fully_available(self) -> bool:
        return self.has_risk_data and self.has_obligation_data

    @property
    def is_empty(self) -> bool:
        return not self.has_risk_data and not self.has_obligation_data


# ──────────────────────────────────────────────────────────────────
# Verdict logic — simple, explainable rules, not a model call
# ──────────────────────────────────────────────────────────────────

def _compute_verdict(risk_counts: dict[str, int]) -> tuple[str, str]:
    """
    Rule-based bottom-line verdict from risk counts alone.

    Deliberately simple and fully explainable — "any HIGH risk clause
    means review before signing" is a rule a non-lawyer can verify by
    eye against the risk counts shown right next to it, unlike a model
    output they'd have to just trust.
    """
    if risk_counts.get("HIGH", 0) > 0:
        return "⚠️ Requires legal review before signing", "urgent"
    if risk_counts.get("MEDIUM", 0) > 0:
        return "🟡 Review recommended before signing", "caution"
    return "✅ No major red flags detected", "clear"


# ──────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────

def build_executive_summary(
    source_name: str,
    scan_result: Optional[ScanResult],
    extraction_result: Optional[ExtractionResult],
) -> ExecutiveSummary:
    """
    Build an ExecutiveSummary from whatever results are already
    available. Never runs a scan or extraction itself — pass None for
    whichever hasn't been run yet, and the summary comes back in
    teaser mode for that half.

    Args:
        source_name: Contract filename — used for the summary header
                     and to keep this summary tied to one document even
                     if scan_result/extraction_result somehow mismatch.
        scan_result: Already-computed ScanResult from scanner.py, or
                     None if the user hasn't run the Risk Scanner yet.
        extraction_result: Already-computed ExtractionResult from
                     extractor.py, or None if the user hasn't run the
                     Obligations Extractor yet.

    Returns:
        ExecutiveSummary — check has_risk_data / has_obligation_data
        to know which parts are real vs. need a teaser in the UI.
    """
    has_risk_data = scan_result is not None
    has_obligation_data = extraction_result is not None

    risk_counts: dict[str, int] = {}
    top_risks: list[RiskLabel] = []
    total_clauses_scanned = 0

    if scan_result is not None:
        total_clauses_scanned = scan_result.total_clauses
        risk_counts = {
            "HIGH": scan_result.high_count,
            "MEDIUM": scan_result.medium_count,
            "LOW": scan_result.low_count,
            "UNKNOWN": scan_result.unknown_count,
        }
        # Document order (as scanner.py produced them) is already a
        # sensible "top" ordering — no re-sorting needed.
        top_risks = [l for l in scan_result.labels if l.is_high_risk][:TOP_RISKS_COUNT]

    upcoming_deadlines: list[Obligation] = []
    if extraction_result is not None:
        real_obligations = [
            o for o in extraction_result.obligations if not o.is_extraction_failure
        ]
        # Same ordering rule as the Obligations page: dated obligations
        # first (chronological), relative-period ones after.
        sorted_obligations = sorted(
            real_obligations,
            key=lambda o: (o.date_value is None, o.date_value or ""),
        )
        upcoming_deadlines = sorted_obligations[:UPCOMING_DEADLINES_COUNT]

    verdict, verdict_level = (
        _compute_verdict(risk_counts) if has_risk_data else ("", "unknown")
    )

    return ExecutiveSummary(
        source_name=source_name,
        has_risk_data=has_risk_data,
        total_clauses_scanned=total_clauses_scanned,
        risk_counts=risk_counts,
        top_risks=top_risks,
        has_obligation_data=has_obligation_data,
        upcoming_deadlines=upcoming_deadlines,
        verdict=verdict,
        verdict_level=verdict_level,
    )
