"""
Tests for src/risk/risk_labels.py and src/risk/scanner.py

Unit tests: enum validation, RiskLabel logic, parse/prompt helpers.
Integration tests: real Gemini API calls marked with @pytest.mark.integration.
"""

from __future__ import annotations

import json
import pytest

from src.risk.risk_labels import (
    RiskLevel,
    ClauseCategory,
    RiskLabel,
    RISK_LEVEL_DEFINITIONS,
    CATEGORY_DEFINITIONS,
    RISK_COLORS,
    RISK_ICONS,
    RISK_BADGE_CSS,
    VALID_CATEGORIES,
    VALID_RISK_LEVELS,
)


# ── RiskLevel tests ────────────────────────────────────────────────

class TestRiskLevel:
    def test_string_equality(self):
        assert RiskLevel.HIGH == "HIGH"
        assert RiskLevel.MEDIUM == "MEDIUM"
        assert RiskLevel.LOW == "LOW"
        assert RiskLevel.UNKNOWN == "UNKNOWN"

    def test_all_levels_present(self):
        levels = {r.value for r in RiskLevel}
        assert {"HIGH", "MEDIUM", "LOW", "UNKNOWN"} == levels

    def test_unknown_not_in_valid_output(self):
        assert "UNKNOWN" not in VALID_RISK_LEVELS

    def test_valid_risk_levels_complete(self):
        assert VALID_RISK_LEVELS == {"HIGH", "MEDIUM", "LOW"}


# ── ClauseCategory tests ───────────────────────────────────────────

class TestClauseCategory:
    def test_total_count(self):
        assert len(list(ClauseCategory)) == 20

    def test_string_equality(self):
        assert ClauseCategory.INDEMNIFICATION == "Indemnification"
        assert ClauseCategory.GENERAL == "General"

    def test_all_categories_in_definitions(self):
        for cat in ClauseCategory:
            assert cat.value in CATEGORY_DEFINITIONS, (
                f"Category '{cat.value}' missing from CATEGORY_DEFINITIONS"
            )

    def test_all_categories_in_valid_set(self):
        for cat in ClauseCategory:
            assert cat.value in VALID_CATEGORIES

    def test_valid_categories_count_matches_enum(self):
        assert len(VALID_CATEGORIES) == len(list(ClauseCategory))


# ── RiskLabel tests ────────────────────────────────────────────────

class TestRiskLabel:
    def _make(self, risk=RiskLevel.HIGH, cat=ClauseCategory.INDEMNIFICATION,
              ps=8, pe=10):
        return RiskLabel(
            clause_id="Section 9", source_name="agency.pdf",
            risk_level=risk, category=cat,
            reason="Unlimited liability exposure.",
            recommended_action="Negotiate a liability cap.",
            page_start=ps, page_end=pe, heading="Indemnification",
        )

    def test_is_high_risk_true(self):
        assert self._make(risk=RiskLevel.HIGH).is_high_risk is True

    def test_is_high_risk_false_for_medium(self):
        assert self._make(risk=RiskLevel.MEDIUM).is_high_risk is False

    def test_is_flagged_high(self):
        assert self._make(risk=RiskLevel.HIGH).is_flagged is True

    def test_is_flagged_medium(self):
        assert self._make(risk=RiskLevel.MEDIUM).is_flagged is True

    def test_is_flagged_false_low(self):
        assert self._make(risk=RiskLevel.LOW).is_flagged is False

    def test_is_flagged_false_unknown(self):
        assert self._make(risk=RiskLevel.UNKNOWN).is_flagged is False

    def test_citation_multi_page(self):
        label = self._make(ps=8, pe=10)
        assert label.citation == "Section 9 (agency.pdf, pp. 8–10)"

    def test_citation_single_page(self):
        label = self._make(ps=3, pe=3)
        assert label.citation == "Section 9 (agency.pdf, p. 3)"

    def test_to_dict_contains_all_fields(self):
        label = self._make()
        d = label.to_dict()
        required = {"clause_id", "source_name", "risk_level", "category",
                    "reason", "recommended_action", "page_start", "page_end"}
        assert required.issubset(d.keys())

    def test_to_dict_serialises_enum_values(self):
        label = self._make()
        d = label.to_dict()
        assert d["risk_level"] == "HIGH"           # string, not enum
        assert d["category"] == "Indemnification"  # string, not enum

    def test_repr_contains_key_info(self):
        label = self._make()
        r = repr(label)
        assert "HIGH" in r
        assert "Indemnification" in r
        assert "Section 9" in r


# ── Risk definitions coverage ──────────────────────────────────────

class TestDefinitionsCoverage:
    def test_risk_level_definitions_complete(self):
        assert set(RISK_LEVEL_DEFINITIONS.keys()) == {"LOW", "MEDIUM", "HIGH"}

    def test_definitions_are_non_empty(self):
        for k, v in RISK_LEVEL_DEFINITIONS.items():
            assert len(v) > 20, f"Definition for {k} is too short"
        for k, v in CATEGORY_DEFINITIONS.items():
            assert len(v) > 20, f"Definition for {k} is too short"


# ── UI constants tests ─────────────────────────────────────────────

class TestUIConstants:
    def test_colors_for_all_levels(self):
        for level in RiskLevel:
            assert level in RISK_COLORS

    def test_icons_for_all_levels(self):
        for level in RiskLevel:
            assert level in RISK_ICONS

    def test_badge_css_for_all_levels(self):
        for level in RiskLevel:
            assert level in RISK_BADGE_CSS

    def test_high_risk_is_red(self):
        assert "#FF" in RISK_COLORS[RiskLevel.HIGH].upper() or \
               "ff" in RISK_COLORS[RiskLevel.HIGH].lower()

    def test_low_risk_is_green(self):
        color = RISK_COLORS[RiskLevel.LOW].lower()
        assert "21c354" in color or "green" in color or "0" in color


# ── Scanner helper unit tests ──────────────────────────────────────

class TestScannerHelpers:
    """
    Tests for scanner.py helper functions without real API calls.
    We extract and test the pure logic directly.
    """

    def _make_batch(self):
        return [
            {"clause_id": "Section 4", "page_start": 8, "page_end": 10,
             "heading": "Indemnification", "full_text": "Company shall indemnify..."},
            {"clause_id": "Section 9", "page_start": 20, "page_end": 22,
             "heading": "Termination", "full_text": "Either party may terminate..."},
        ]

    def test_system_prompt_contains_risk_definitions(self):
        from src.risk.scanner import _build_system_prompt
        prompt = _build_system_prompt()
        assert "HIGH" in prompt
        assert "MEDIUM" in prompt
        assert "LOW" in prompt
        assert "JSON array" in prompt

    def test_system_prompt_contains_categories(self):
        from src.risk.scanner import _build_system_prompt
        prompt = _build_system_prompt()
        assert "Indemnification" in prompt
        assert "Confidentiality" in prompt
        assert "Termination" in prompt

    def test_user_prompt_contains_all_clauses(self):
        from src.risk.scanner import _build_user_prompt
        batch = self._make_batch()
        prompt = _build_user_prompt(batch)
        assert "Section 4" in prompt
        assert "Section 9" in prompt
        assert "CLAUSE 1" in prompt
        assert "CLAUSE 2" in prompt

    def test_parse_valid_response(self):
        from src.risk.scanner import _parse_llm_response
        batch = self._make_batch()
        response = json.dumps([
            {"clause_id": "Section 4", "risk_level": "HIGH",
             "category": "Indemnification", "reason": "Unlimited.",
             "recommended_action": "Cap it."},
            {"clause_id": "Section 9", "risk_level": "LOW",
             "category": "Termination", "reason": "Standard.",
             "recommended_action": "No action."},
        ])
        labels = _parse_llm_response(response, batch, "agency.pdf")
        assert len(labels) == 2
        assert labels[0].risk_level == RiskLevel.HIGH
        assert labels[0].category == ClauseCategory.INDEMNIFICATION
        assert labels[1].risk_level == RiskLevel.LOW
        assert labels[0].page_start == 8

    def test_parse_strips_markdown_fences(self):
        from src.risk.scanner import _parse_llm_response
        batch = self._make_batch()[:1]
        response = "```json\n" + json.dumps([
            {"clause_id": "Section 4", "risk_level": "LOW",
             "category": "General", "reason": "Ok.", "recommended_action": "None."}
        ]) + "\n```"
        labels = _parse_llm_response(response, batch, "agency.pdf")
        assert len(labels) == 1

    def test_parse_unknown_risk_level_becomes_unknown(self):
        from src.risk.scanner import _parse_llm_response
        batch = self._make_batch()[:1]
        response = json.dumps([
            {"clause_id": "Section 4", "risk_level": "EXTREME",
             "category": "Indemnification", "reason": "...", "recommended_action": "..."}
        ])
        labels = _parse_llm_response(response, batch, "agency.pdf")
        assert labels[0].risk_level == RiskLevel.UNKNOWN

    def test_parse_unknown_category_becomes_general(self):
        from src.risk.scanner import _parse_llm_response
        batch = self._make_batch()[:1]
        response = json.dumps([
            {"clause_id": "Section 4", "risk_level": "HIGH",
             "category": "FakeCategory", "reason": "...", "recommended_action": "..."}
        ])
        labels = _parse_llm_response(response, batch, "agency.pdf")
        assert labels[0].category == ClauseCategory.GENERAL

    def test_parse_wrong_clause_id_uses_positional_fallback(self):
        from src.risk.scanner import _parse_llm_response
        batch = self._make_batch()[:1]
        response = json.dumps([
            {"clause_id": "WRONG_ID", "risk_level": "MEDIUM",
             "category": "Termination", "reason": "...", "recommended_action": "..."}
        ])
        labels = _parse_llm_response(response, batch, "agency.pdf")
        assert labels[0].clause_id == "Section 4"  # positional fallback

    def test_parse_invalid_json_returns_empty(self):
        from src.risk.scanner import _parse_llm_response
        batch = self._make_batch()
        assert _parse_llm_response("not json", batch, "agency.pdf") == []
        assert _parse_llm_response("", batch, "agency.pdf") == []
        assert _parse_llm_response("{}", batch, "agency.pdf") == []

    def test_fallback_labels_produced_for_all_batch_items(self):
        from src.risk.scanner import _fallback_labels
        batch = self._make_batch()
        labels = _fallback_labels(batch, "agency.pdf")
        assert len(labels) == 2
        assert all(l.risk_level == RiskLevel.UNKNOWN for l in labels)
        assert all(l.source_name == "agency.pdf" for l in labels)

    def test_sub_clause_filter_removes_sub_clauses(self):
        import re
        chunks = [
            {"clause_id": "Section 4"},
            {"clause_id": "Section 4(a)"},
            {"clause_id": "Clause 1"},
            {"clause_id": "Clause 1(i)"},
            {"clause_id": "Para 1"},
        ]
        top_level = [
            c for c in chunks
            if not re.search(r'\([a-z]{1,2}\)$|\([ivxlc]+\)$', c["clause_id"])
        ]
        assert len(top_level) == 3
        assert all("(" not in c["clause_id"] for c in top_level)


# ── ScanResult tests ───────────────────────────────────────────────

class TestScanResult:
    def test_flagged_count(self):
        from src.risk.scanner import ScanResult
        sr = ScanResult("c.pdf", 10, 3, 4, 3, 0, [], 15.2)
        assert sr.flagged_count == 7

    def test_success_true_when_no_unknowns(self):
        from src.risk.scanner import ScanResult
        sr = ScanResult("c.pdf", 10, 3, 4, 3, 0, [], 15.2)
        assert sr.success is True

    def test_success_false_when_unknowns_present(self):
        from src.risk.scanner import ScanResult
        sr = ScanResult("c.pdf", 10, 3, 4, 2, 1, [], 15.2)
        assert sr.success is False

    def test_repr_contains_counts(self):
        from src.risk.scanner import ScanResult
        sr = ScanResult("c.pdf", 10, 3, 4, 3, 0, [], 15.2)
        r = str(sr)
        assert "HIGH" in r
        assert "MEDIUM" in r


# ── Integration tests ──────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.slow
class TestScannerIntegration:
    def test_scan_contract_returns_labels(self, require_api_key, ingested_contract):
        """ingested_contract parses+chunks a real sample PDF into
        tmp_sqlite so this doesn't depend on a pre-existing
        data/metadata.db (which only exists locally, not on a fresh CI
        checkout) — and runs once per contract in SCANNER_TEST_CONTRACTS
        so it isn't coupled to any single document."""
        from src.risk.scanner import scan_contract
        conn, source_name = ingested_contract
        result = scan_contract(source_name, conn=conn)
        assert result.total_clauses > 0
        assert all(isinstance(l, RiskLabel) for l in result.labels)

    def test_scan_all_labels_have_valid_risk_level(self, require_api_key, ingested_contract):
        from src.risk.scanner import scan_contract
        conn, source_name = ingested_contract
        result = scan_contract(source_name, conn=conn)
        valid = set(RiskLevel)
        assert all(l.risk_level in valid for l in result.labels)

    def test_scan_all_labels_have_valid_category(self, require_api_key, ingested_contract):
        from src.risk.scanner import scan_contract
        conn, source_name = ingested_contract
        result = scan_contract(source_name, conn=conn)
        valid = set(ClauseCategory)
        assert all(l.category in valid for l in result.labels)

    def test_scan_labels_have_non_empty_reason(self, require_api_key, ingested_contract):
        from src.risk.scanner import scan_contract
        conn, source_name = ingested_contract
        result = scan_contract(source_name, conn=conn)
        assert all(len(l.reason) > 10 for l in result.labels)

    def test_scan_labels_have_non_empty_action(self, require_api_key, ingested_contract):
        from src.risk.scanner import scan_contract
        conn, source_name = ingested_contract
        result = scan_contract(source_name, conn=conn)
        assert all(len(l.recommended_action) > 5 for l in result.labels)

    def test_scan_clauses_targeted(self, require_api_key, tmp_sqlite, populated_sqlite):
        """scan_clauses() should only classify the given clause IDs."""
        from src.risk.scanner import scan_clauses
        labels = scan_clauses(
            clause_ids=["Section 4"],
            source_name="agency.pdf",
            conn=populated_sqlite,
        )
        assert len(labels) == 1
        assert labels[0].clause_id == "Section 4"