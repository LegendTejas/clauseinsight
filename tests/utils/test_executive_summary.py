"""
Tests for src/utils/executive_summary.py

No mocking needed anywhere in this file — build_executive_summary() is
pure aggregation over already-constructed dataclasses, with zero I/O
and zero LLM calls. Every test just builds real ScanResult/
ExtractionResult objects and checks the output.
"""

from __future__ import annotations

from src.risk.scanner import ScanResult
from src.risk.risk_labels import RiskLabel, RiskLevel, ClauseCategory
from src.obligations.extractor import ExtractionResult
from src.obligations.obligation_labels import Obligation, ObligationType
from src.utils.executive_summary import build_executive_summary, ExecutiveSummary


# ── Fixtures / builders ─────────────────────────────────────────────

def _make_label(clause_id, level, category=ClauseCategory.GENERAL, reason="r"):
    return RiskLabel(
        clause_id=clause_id, source_name="c.pdf",
        risk_level=level, category=category, reason=reason,
        recommended_action="a", page_start=1, page_end=1,
    )


def _make_scan_result(labels):
    high = sum(1 for l in labels if l.risk_level == RiskLevel.HIGH)
    medium = sum(1 for l in labels if l.risk_level == RiskLevel.MEDIUM)
    low = sum(1 for l in labels if l.risk_level == RiskLevel.LOW)
    unknown = sum(1 for l in labels if l.risk_level == RiskLevel.UNKNOWN)
    return ScanResult(
        source_name="c.pdf", total_clauses=len(labels),
        high_count=high, medium_count=medium, low_count=low, unknown_count=unknown,
        labels=labels, elapsed_seconds=1.0,
    )


def _make_obligation(clause_id, date_value=None, period_value=None, confidence=0.9):
    return Obligation(
        clause_id=clause_id, source_name="c.pdf",
        obligation_type=ObligationType.RENEWAL_DATE,
        description="d", date_value=date_value, period_value=period_value,
        page_start=1, page_end=1, confidence=confidence,
    )


def _make_extraction_result(obligations, failed_count=0):
    return ExtractionResult(
        source_name="c.pdf", total_clauses=10,
        obligations=obligations, failed_count=failed_count, elapsed_seconds=1.0,
    )


# ── Teaser / availability states ────────────────────────────────────

class TestAvailabilityStates:
    def test_both_none_is_empty(self):
        summary = build_executive_summary("c.pdf", None, None)
        assert summary.is_empty is True
        assert summary.has_risk_data is False
        assert summary.has_obligation_data is False

    def test_only_risk_available(self):
        scan = _make_scan_result([_make_label("S1", RiskLevel.LOW)])
        summary = build_executive_summary("c.pdf", scan, None)
        assert summary.has_risk_data is True
        assert summary.has_obligation_data is False
        assert summary.is_fully_available is False

    def test_only_obligations_available(self):
        extraction = _make_extraction_result([_make_obligation("S1", period_value="30 days")])
        summary = build_executive_summary("c.pdf", None, extraction)
        assert summary.has_risk_data is False
        assert summary.has_obligation_data is True

    def test_both_available_is_fully_available(self):
        scan = _make_scan_result([_make_label("S1", RiskLevel.LOW)])
        extraction = _make_extraction_result([])
        summary = build_executive_summary("c.pdf", scan, extraction)
        assert summary.is_fully_available is True
        assert summary.is_empty is False


# ── Risk aggregation ─────────────────────────────────────────────────

class TestRiskAggregation:
    def test_risk_counts_match_scan_result(self):
        labels = [
            _make_label("S1", RiskLevel.HIGH),
            _make_label("S2", RiskLevel.HIGH),
            _make_label("S3", RiskLevel.MEDIUM),
            _make_label("S4", RiskLevel.LOW),
        ]
        summary = build_executive_summary("c.pdf", _make_scan_result(labels), None)
        assert summary.risk_counts == {"HIGH": 2, "MEDIUM": 1, "LOW": 1, "UNKNOWN": 0}

    def test_total_clauses_scanned_matches(self):
        labels = [_make_label(f"S{i}", RiskLevel.LOW) for i in range(7)]
        summary = build_executive_summary("c.pdf", _make_scan_result(labels), None)
        assert summary.total_clauses_scanned == 7

    def test_top_risks_only_includes_high(self):
        labels = [
            _make_label("S1", RiskLevel.HIGH),
            _make_label("S2", RiskLevel.MEDIUM),
            _make_label("S3", RiskLevel.LOW),
        ]
        summary = build_executive_summary("c.pdf", _make_scan_result(labels), None)
        assert len(summary.top_risks) == 1
        assert summary.top_risks[0].clause_id == "S1"

    def test_top_risks_capped_at_three(self):
        labels = [_make_label(f"S{i}", RiskLevel.HIGH) for i in range(6)]
        summary = build_executive_summary("c.pdf", _make_scan_result(labels), None)
        assert len(summary.top_risks) == 3

    def test_top_risks_preserves_document_order(self):
        labels = [
            _make_label("S5", RiskLevel.HIGH),
            _make_label("S2", RiskLevel.HIGH),
            _make_label("S9", RiskLevel.HIGH),
        ]
        summary = build_executive_summary("c.pdf", _make_scan_result(labels), None)
        assert [l.clause_id for l in summary.top_risks] == ["S5", "S2", "S9"]

    def test_no_high_risk_gives_empty_top_risks(self):
        labels = [_make_label("S1", RiskLevel.LOW), _make_label("S2", RiskLevel.MEDIUM)]
        summary = build_executive_summary("c.pdf", _make_scan_result(labels), None)
        assert summary.top_risks == []


# ── Obligation aggregation ──────────────────────────────────────────

class TestObligationAggregation:
    def test_dated_obligations_sorted_chronologically(self):
        obligations = [
            _make_obligation("S1", date_value="2027-06-01"),
            _make_obligation("S2", date_value="2026-01-15"),
            _make_obligation("S3", date_value="2026-12-31"),
        ]
        summary = build_executive_summary("c.pdf", None, _make_extraction_result(obligations))
        dates = [o.date_value for o in summary.upcoming_deadlines]
        assert dates == ["2026-01-15", "2026-12-31", "2027-06-01"]

    def test_dated_obligations_come_before_relative_period_ones(self):
        obligations = [
            _make_obligation("S1", period_value="30 days notice"),
            _make_obligation("S2", date_value="2027-01-01"),
        ]
        summary = build_executive_summary("c.pdf", None, _make_extraction_result(obligations))
        assert summary.upcoming_deadlines[0].clause_id == "S2"
        assert summary.upcoming_deadlines[1].clause_id == "S1"

    def test_capped_at_three(self):
        obligations = [_make_obligation(f"S{i}", period_value="30 days") for i in range(6)]
        summary = build_executive_summary("c.pdf", None, _make_extraction_result(obligations))
        assert len(summary.upcoming_deadlines) == 3

    def test_extraction_failures_excluded(self):
        obligations = [
            _make_obligation("S1", period_value="30 days", confidence=0.9),
            _make_obligation("S2", period_value="60 days", confidence=0.0),  # failure
        ]
        summary = build_executive_summary("c.pdf", None, _make_extraction_result(obligations))
        assert len(summary.upcoming_deadlines) == 1
        assert summary.upcoming_deadlines[0].clause_id == "S1"

    def test_no_obligations_gives_empty_list(self):
        summary = build_executive_summary("c.pdf", None, _make_extraction_result([]))
        assert summary.upcoming_deadlines == []


# ── Verdict logic ────────────────────────────────────────────────────

class TestVerdictLogic:
    def test_any_high_risk_gives_urgent_verdict(self):
        labels = [_make_label("S1", RiskLevel.HIGH), _make_label("S2", RiskLevel.LOW)]
        summary = build_executive_summary("c.pdf", _make_scan_result(labels), None)
        assert summary.verdict_level == "urgent"
        assert "review" in summary.verdict.lower()

    def test_medium_only_gives_caution_verdict(self):
        labels = [_make_label("S1", RiskLevel.MEDIUM), _make_label("S2", RiskLevel.LOW)]
        summary = build_executive_summary("c.pdf", _make_scan_result(labels), None)
        assert summary.verdict_level == "caution"

    def test_low_only_gives_clear_verdict(self):
        labels = [_make_label("S1", RiskLevel.LOW), _make_label("S2", RiskLevel.LOW)]
        summary = build_executive_summary("c.pdf", _make_scan_result(labels), None)
        assert summary.verdict_level == "clear"

    def test_high_takes_priority_over_medium(self):
        labels = [_make_label("S1", RiskLevel.HIGH), _make_label("S2", RiskLevel.MEDIUM)]
        summary = build_executive_summary("c.pdf", _make_scan_result(labels), None)
        assert summary.verdict_level == "urgent"

    def test_no_risk_data_gives_unknown_verdict(self):
        summary = build_executive_summary("c.pdf", None, _make_extraction_result([]))
        assert summary.verdict_level == "unknown"
        assert summary.verdict == ""


# ── Determinism (a core selling point of the no-LLM design) ────────

class TestDeterminism:
    def test_same_input_always_produces_same_output(self):
        labels = [_make_label("S1", RiskLevel.HIGH), _make_label("S2", RiskLevel.MEDIUM)]
        obligations = [_make_obligation("S3", date_value="2027-01-01")]

        results = [
            build_executive_summary(
                "c.pdf", _make_scan_result(labels), _make_extraction_result(obligations)
            )
            for _ in range(5)
        ]
        first = results[0]
        for r in results[1:]:
            assert r.risk_counts == first.risk_counts
            assert r.verdict == first.verdict
            assert [o.clause_id for o in r.upcoming_deadlines] == \
                   [o.clause_id for o in first.upcoming_deadlines]
