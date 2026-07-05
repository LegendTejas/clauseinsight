"""
ClauseInsight — Risk Labels
============================

Single source of truth for all risk classification definitions used
by scanner.py and the Streamlit risk dashboard (3_scanner.py).

WHY THIS IS A SEPARATE FILE
-----------------------------
Risk definitions are policy, not code. Keeping them here means:
  - scanner.py's classification logic never needs to change when
    risk definitions are tuned
  - The Streamlit UI imports the same definitions for display
    (colors, labels, descriptions) — no duplication
  - Internship reviewers can read the risk taxonomy without
    digging through scanner logic

WHAT'S IN HERE
---------------
  1. RiskLevel enum          — LOW / MEDIUM / HIGH / UNKNOWN
  2. ClauseCategory enum     — 20 legal clause types
  3. RiskLabel dataclass     — structured output of one classified clause
  4. CATEGORY_DEFINITIONS    — plain-English definition of each clause type
                               (injected into the LLM prompt so it knows
                               what each category means)
  5. RISK_LEVEL_DEFINITIONS  — what LOW/MEDIUM/HIGH means in legal context
                               (injected into prompt for consistent scoring)
  6. UI display constants    — colors and icons for the Streamlit dashboard
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ──────────────────────────────────────────────────────────────────
# Risk Level
# ──────────────────────────────────────────────────────────────────

class RiskLevel(str, Enum):
    """
    Three-tier risk classification for a contract clause.

    Inherits from str so RiskLevel.HIGH == "HIGH" is True —
    makes JSON serialisation and ChromaDB metadata storage trivial.
    """
    LOW     = "LOW"
    MEDIUM  = "MEDIUM"
    HIGH    = "HIGH"
    UNKNOWN = "UNKNOWN"   # fallback when LLM output cannot be parsed


# ──────────────────────────────────────────────────────────────────
# Clause Category
# ──────────────────────────────────────────────────────────────────

class ClauseCategory(str, Enum):
    """
    Legal clause type taxonomy covering the most common contract sections.

    Kept as an enum (not a plain list) so the scanner can validate
    LLM output against known values and fall back to GENERAL if the
    model hallucinates a category name.
    """
    INDEMNIFICATION         = "Indemnification"
    LIMITATION_OF_LIABILITY = "Limitation of Liability"
    TERMINATION             = "Termination"
    INTELLECTUAL_PROPERTY   = "Intellectual Property"
    CONFIDENTIALITY         = "Confidentiality"
    NON_COMPETE             = "Non-Compete"
    GOVERNING_LAW           = "Governing Law"
    DISPUTE_RESOLUTION      = "Dispute Resolution"
    PAYMENT_TERMS           = "Payment Terms"
    WARRANTY                = "Warranty"
    REPRESENTATIONS         = "Representations and Warranties"
    ASSIGNMENT              = "Assignment"
    FORCE_MAJEURE           = "Force Majeure"
    AMENDMENT               = "Amendment"
    NOTICES                 = "Notices"
    ENTIRE_AGREEMENT        = "Entire Agreement"
    DEFINITIONS             = "Definitions"
    COMPLIANCE              = "Compliance"
    DATA_PRIVACY            = "Data Privacy"
    GENERAL                 = "General"          # catch-all for uncategorised clauses


# ──────────────────────────────────────────────────────────────────
# Risk Label — output of classifying one chunk
# ──────────────────────────────────────────────────────────────────

@dataclass
class RiskLabel:
    """
    Structured risk classification result for one contract clause.

    This is what scanner.py produces and what the Streamlit dashboard
    displays. All fields are populated by the LLM response — the
    scanner validates and coerces the values to match the enums.

    Attributes
    ----------
    clause_id:
        e.g. "Section 4", "Clause 1(a)" — matches the chunk's clause_id

    source_name:
        Original contract filename — needed because the scanner can
        process multiple contracts in one session

    risk_level:
        LOW / MEDIUM / HIGH / UNKNOWN

    category:
        Which type of legal clause this is (ClauseCategory enum value)

    reason:
        1-2 sentence plain-English explanation of WHY this risk level
        was assigned. Written for a non-lawyer to understand.
        e.g. "This clause requires the employee to assign all IP created
        during employment to the company, including work done outside
        office hours."

    recommended_action:
        Concrete next step for the reader.
        e.g. "Negotiate to limit IP assignment to work directly related
        to your role. Ensure side projects are explicitly excluded."

    page_start / page_end:
        Page range of the clause — for UI citation display

    heading:
        Clause heading/title for display

    confidence:
        Optional 0.0-1.0 float — not all LLM responses include this,
        stored when available for future ranking/filtering use
    """

    clause_id:           str
    source_name:         str
    risk_level:          RiskLevel
    category:            ClauseCategory
    reason:              str
    recommended_action:  str
    page_start:          int
    page_end:            int
    heading:             str = ""
    confidence:          Optional[float] = None

    @property
    def is_high_risk(self) -> bool:
        return self.risk_level == RiskLevel.HIGH

    @property
    def is_flagged(self) -> bool:
        """True for MEDIUM or HIGH risk — shown in the dashboard summary."""
        return self.risk_level in (RiskLevel.MEDIUM, RiskLevel.HIGH)

    @property
    def citation(self) -> str:
        pages = (
            f"p. {self.page_start}"
            if self.page_start == self.page_end
            else f"pp. {self.page_start}–{self.page_end}"
        )
        return f"{self.clause_id} ({self.source_name}, {pages})"

    def to_dict(self) -> dict:
        """Serialise to a plain dict for SQLite storage or JSON export."""
        return {
            "clause_id":          self.clause_id,
            "source_name":        self.source_name,
            "risk_level":         self.risk_level.value,
            "category":           self.category.value,
            "reason":             self.reason,
            "recommended_action": self.recommended_action,
            "page_start":         self.page_start,
            "page_end":           self.page_end,
            "heading":            self.heading,
            "confidence":         self.confidence,
        }

    def __repr__(self) -> str:
        return (
            f"RiskLabel({self.risk_level.value} | "
            f"{self.category.value} | "
            f"{self.clause_id} | "
            f"{self.source_name})"
        )


# ──────────────────────────────────────────────────────────────────
# Risk Level Definitions
# Injected into the LLM system prompt for consistent scoring
# ──────────────────────────────────────────────────────────────────

RISK_LEVEL_DEFINITIONS: dict[str, str] = {
    "LOW": (
        "The clause is standard, balanced, and presents no unusual obligations "
        "or risks to either party. Common in well-drafted commercial contracts. "
        "Examples: standard notice requirements, governing law selection, "
        "entire agreement clauses, force majeure with balanced carve-outs."
    ),
    "MEDIUM": (
        "The clause contains terms that are non-standard, one-sided, or could "
        "create meaningful obligations under certain circumstances. Worth reviewing "
        "carefully but not immediately alarming. "
        "Examples: broad indemnification with limited carve-outs, termination "
        "for convenience by only one party, unilateral amendment rights."
    ),
    "HIGH": (
        "The clause poses significant legal or financial risk, is heavily "
        "one-sided, or contains terms that a party should not accept without "
        "negotiation or independent legal advice. "
        "Examples: unlimited liability, broad IP assignment including work "
        "outside employment scope, non-compete clauses with wide geographic "
        "scope and long duration, automatic renewal with no opt-out window, "
        "unilateral modification of payment terms."
    ),
}


# ──────────────────────────────────────────────────────────────────
# Clause Category Definitions
# Injected into the LLM prompt so it classifies consistently
# ──────────────────────────────────────────────────────────────────

CATEGORY_DEFINITIONS: dict[str, str] = {
    "Indemnification": (
        "Clauses where one party agrees to compensate the other for losses, "
        "damages, or legal costs arising from specified events or breaches."
    ),
    "Limitation of Liability": (
        "Clauses that cap the total financial exposure of one or both parties, "
        "or exclude certain categories of damages (indirect, consequential, etc.)."
    ),
    "Termination": (
        "Clauses governing when and how the agreement can be ended, including "
        "termination for cause, termination for convenience, and notice periods."
    ),
    "Intellectual Property": (
        "Clauses covering ownership, assignment, licensing, or restrictions on "
        "use of intellectual property created during or related to the agreement."
    ),
    "Confidentiality": (
        "Clauses governing what information must be kept secret, for how long, "
        "and what the consequences of disclosure are."
    ),
    "Non-Compete": (
        "Clauses restricting a party from engaging in competing activities, "
        "working for competitors, or soliciting clients or employees."
    ),
    "Governing Law": (
        "Clauses specifying which jurisdiction's laws govern the agreement "
        "and where disputes must be resolved."
    ),
    "Dispute Resolution": (
        "Clauses specifying how disputes are resolved — litigation, arbitration, "
        "mediation — and any waiver of jury trial."
    ),
    "Payment Terms": (
        "Clauses covering fees, payment schedules, late payment penalties, "
        "refund policies, and financial obligations."
    ),
    "Warranty": (
        "Clauses where a party makes promises about the quality, fitness, or "
        "characteristics of goods or services provided."
    ),
    "Representations and Warranties": (
        "Statements of fact made by one or both parties at the time of signing, "
        "which if false can give rise to claims for misrepresentation."
    ),
    "Assignment": (
        "Clauses governing whether and how a party can transfer its rights or "
        "obligations under the agreement to a third party."
    ),
    "Force Majeure": (
        "Clauses excusing a party from performance when extraordinary events "
        "beyond their control prevent them from fulfilling obligations."
    ),
    "Amendment": (
        "Clauses specifying how the agreement can be modified, and whether one "
        "party can unilaterally change terms."
    ),
    "Notices": (
        "Clauses specifying how formal communications between parties must be "
        "delivered — email, postal, courier — and when they take effect."
    ),
    "Entire Agreement": (
        "Clauses stating that the written contract supersedes all prior "
        "negotiations, representations, and understandings."
    ),
    "Definitions": (
        "Clauses that define key terms used throughout the agreement. Risk "
        "arises when definitions are unusually broad or narrow."
    ),
    "Compliance": (
        "Clauses requiring adherence to laws, regulations, or industry standards, "
        "including export controls, anti-bribery, and data protection laws."
    ),
    "Data Privacy": (
        "Clauses governing the collection, use, storage, and transfer of "
        "personal data, and compliance with privacy regulations (GDPR, etc.)."
    ),
    "General": (
        "Clauses that do not fit neatly into the above categories — boilerplate, "
        "severability, waiver, counterparts, and miscellaneous provisions."
    ),
}


# ──────────────────────────────────────────────────────────────────
# UI Display Constants
# Used by 3_scanner.py for consistent colour + icon rendering
# ──────────────────────────────────────────────────────────────────

RISK_COLORS: dict[str, str] = {
    RiskLevel.HIGH:    "#FF4B4B",   # Streamlit red
    RiskLevel.MEDIUM:  "#FFA500",   # orange
    RiskLevel.LOW:     "#21C354",   # Streamlit green
    RiskLevel.UNKNOWN: "#808080",   # grey
}

RISK_ICONS: dict[str, str] = {
    RiskLevel.HIGH:    "🔴",
    RiskLevel.MEDIUM:  "🟡",
    RiskLevel.LOW:     "🟢",
    RiskLevel.UNKNOWN: "⚪",
}

RISK_BADGE_CSS: dict[str, str] = {
    RiskLevel.HIGH:    "background-color:#FF4B4B;color:white;padding:2px 8px;border-radius:4px;font-weight:bold;",
    RiskLevel.MEDIUM:  "background-color:#FFA500;color:white;padding:2px 8px;border-radius:4px;font-weight:bold;",
    RiskLevel.LOW:     "background-color:#21C354;color:white;padding:2px 8px;border-radius:4px;font-weight:bold;",
    RiskLevel.UNKNOWN: "background-color:#808080;color:white;padding:2px 8px;border-radius:4px;font-weight:bold;",
}


# ──────────────────────────────────────────────────────────────────
# Valid category names (for LLM output validation in scanner.py)
# ──────────────────────────────────────────────────────────────────

VALID_CATEGORIES: set[str] = {c.value for c in ClauseCategory}
VALID_RISK_LEVELS: set[str] = {r.value for r in RiskLevel if r != RiskLevel.UNKNOWN}
