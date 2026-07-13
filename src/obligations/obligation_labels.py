"""
ClauseInsight — Obligation Labels
===================================

Single source of truth for all obligation/deadline extraction definitions
used by extractor.py and the Streamlit obligations dashboard (4_obligations.py).

WHY THIS IS A SEPARATE FILE
-----------------------------
Mirrors src/risk/risk_labels.py's separation of policy from logic:
  - extractor.py's extraction logic never needs to change when
    obligation type definitions are tuned
  - The Streamlit UI imports the same definitions for display
    (colors, icons) — no duplication
  - Internship reviewers can read the obligation taxonomy without
    digging through extractor logic

WHAT'S IN HERE
---------------
  1. ObligationType enum      — 6 categories of dated/deadline clauses
  2. Obligation dataclass     — structured output of one extracted obligation
  3. OBLIGATION_TYPE_DEFINITIONS — plain-English definition of each type
                                  (injected into the LLM prompt)
  4. UI display constants     — colors and icons for the Streamlit dashboard
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


# ──────────────────────────────────────────────────────────────────
# Obligation Type
# ──────────────────────────────────────────────────────────────────

class ObligationType(str, Enum):
    """
    Taxonomy of dated/deadline-bearing clause types.

    Inherits from str so ObligationType.RENEWAL_DATE == "Renewal Date"
    is True — makes JSON serialisation trivial, same pattern as
    src/risk/risk_labels.py's RiskLevel.
    """
    RENEWAL_DATE        = "Renewal Date"
    NOTICE_PERIOD       = "Notice Period"
    TERMINATION_WINDOW  = "Termination Window"
    PAYMENT_DEADLINE    = "Payment Deadline"
    AUTO_RENEWAL        = "Auto-Renewal"
    OTHER_DEADLINE       = "Other Deadline"   # catch-all, mirrors ClauseCategory.GENERAL


# ──────────────────────────────────────────────────────────────────
# Obligation — output of extracting one dated/deadline clause
# ──────────────────────────────────────────────────────────────────

@dataclass
class Obligation:
    """
    Structured extraction result for one obligation/deadline found in a clause.

    This is what extractor.py produces and what the Streamlit dashboard
    displays. Mirrors src/risk/risk_labels.py's RiskLabel shape closely
    so the two features feel consistent in the UI and codebase.

    Attributes
    ----------
    clause_id:
        e.g. "Section 4", "Clause 1(a)" — matches the chunk's clause_id

    source_name:
        Original contract filename — needed because the extractor can
        process multiple contracts in one session

    obligation_type:
        Which category of dated obligation this is (ObligationType enum value)

    description:
        1-2 sentence plain-English summary of the obligation.
        e.g. "The agreement automatically renews for successive 1-year
        terms unless either party gives written notice 60 days before
        the renewal date."

    date_value:
        Absolute date in ISO format (YYYY-MM-DD), if the clause specifies
        one directly (e.g. "This Agreement expires on January 15, 2027").
        None if the clause only specifies a relative period.

    period_value:
        Relative time period, if the clause specifies one instead of an
        absolute date (e.g. "30 days notice", "within 15 days of invoice").
        None if the clause specifies an absolute date instead.

    page_start / page_end:
        Page range of the clause — for UI citation display

    heading:
        Clause heading/title for display

    confidence:
        0.0-1.0 float from the LLM. Extraction-failure fallback rows use
        0.0 so the UI can visually distinguish them from real extractions.
    """

    clause_id:        str
    source_name:       str
    obligation_type:   ObligationType
    description:       str
    date_value:         Optional[str]
    period_value:       Optional[str]
    page_start:         int
    page_end:           int
    heading:            str = ""
    confidence:         Optional[float] = None

    @property
    def is_dated(self) -> bool:
        """True if the clause specifies an absolute calendar date."""
        return bool(self.date_value)

    @property
    def is_extraction_failure(self) -> bool:
        """True for fallback rows emitted when the LLM call failed after retries."""
        return self.confidence == 0.0

    @property
    def citation(self) -> str:
        pages = (
            f"p. {self.page_start}"
            if self.page_start == self.page_end
            else f"pp. {self.page_start}–{self.page_end}"
        )
        return f"{self.clause_id} ({self.source_name}, {pages})"

    @property
    def when_display(self) -> str:
        """Human-friendly display of the timing — date takes priority over period."""
        if self.date_value:
            return self.date_value
        if self.period_value:
            return self.period_value
        return "Not specified"

    def to_dict(self) -> dict:
        """Serialise to a plain dict for JSON export, mirrors RiskLabel.to_dict()."""
        return {
            "clause_id":        self.clause_id,
            "source_name":      self.source_name,
            "obligation_type":  self.obligation_type.value,
            "description":      self.description,
            "date_value":       self.date_value,
            "period_value":     self.period_value,
            "page_start":       self.page_start,
            "page_end":         self.page_end,
            "heading":          self.heading,
            "confidence":       self.confidence,
        }

    def __repr__(self) -> str:
        return (
            f"Obligation({self.obligation_type.value} | "
            f"{self.when_display} | "
            f"{self.clause_id} | "
            f"{self.source_name})"
        )


# ──────────────────────────────────────────────────────────────────
# Obligation Type Definitions
# Injected into the LLM system prompt for consistent extraction
# ──────────────────────────────────────────────────────────────────

OBLIGATION_TYPE_DEFINITIONS: dict[str, str] = {
    "Renewal Date": (
        "The clause specifies a fixed date or term-end on which the agreement "
        "renews, expires, or must be actively renewed by one or both parties."
    ),
    "Notice Period": (
        "The clause requires one party to give the other advance written or "
        "verbal notice — of termination, non-renewal, or a material change — "
        "a specified number of days or months before it takes effect."
    ),
    "Termination Window": (
        "The clause specifies a window of time during which a party may "
        "terminate the agreement (for cause or for convenience), or a cure "
        "period before a breach becomes a termination event."
    ),
    "Payment Deadline": (
        "The clause specifies when a payment, invoice, or fee is due — "
        "a fixed date, a period after invoice (e.g. 'net 30'), or a "
        "recurring billing schedule."
    ),
    "Auto-Renewal": (
        "The clause causes the agreement to renew automatically for a further "
        "term unless a party affirmatively opts out, usually by a deadline "
        "tied to a notice period."
    ),
    "Other Deadline": (
        "Any other clause that creates a time-bound obligation or deadline "
        "not covered by the categories above — e.g. a deliverable due date, "
        "an audit window, or an option exercise period."
    ),
}


# ──────────────────────────────────────────────────────────────────
# UI Display Constants
# Used by 4_obligations.py for consistent colour + icon rendering
# ──────────────────────────────────────────────────────────────────

OBLIGATION_ICONS: dict[str, str] = {
    ObligationType.RENEWAL_DATE:       "🔁",
    ObligationType.NOTICE_PERIOD:      "📣",
    ObligationType.TERMINATION_WINDOW: "🚪",
    ObligationType.PAYMENT_DEADLINE:   "💰",
    ObligationType.AUTO_RENEWAL:       "♻️",
    ObligationType.OTHER_DEADLINE:     "📌",
}

OBLIGATION_COLORS: dict[str, str] = {
    ObligationType.RENEWAL_DATE:       "#4B9EFF",
    ObligationType.NOTICE_PERIOD:      "#FFA500",
    ObligationType.TERMINATION_WINDOW: "#FF4B4B",
    ObligationType.PAYMENT_DEADLINE:   "#21C354",
    ObligationType.AUTO_RENEWAL:       "#A64BFF",
    ObligationType.OTHER_DEADLINE:     "#808080",
}

OBLIGATION_BADGE_CSS: dict[str, str] = {
    k: f"background-color:{v};color:white;padding:2px 8px;border-radius:4px;font-weight:bold;"
    for k, v in OBLIGATION_COLORS.items()
}


# ──────────────────────────────────────────────────────────────────
# Valid type names (for LLM output validation in extractor.py)
# ──────────────────────────────────────────────────────────────────

VALID_OBLIGATION_TYPES: set[str] = {t.value for t in ObligationType}
