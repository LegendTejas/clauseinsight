"""
Tests for src/obligations/obligation_labels.py and src/obligations/extractor.py

Unit tests: enum validation, Obligation logic, parse/prompt helpers.
Integration tests: real OpenAI API calls marked with @pytest.mark.integration.

Mirrors tests/risk/test_scanner.py's structure and coverage shape closely.
"""

from __future__ import annotations

import json
import pytest

from src.obligations.obligation_labels import (
    ObligationType,
    Obligation,
    OBLIGATION_TYPE_DEFINITIONS,
    OBLIGATION_COLORS,
    OBLIGATION_ICONS,
    OBLIGATION_BADGE_CSS,
    VALID_OBLIGATION_TYPES,
)


# ── ObligationType tests ───────────────────────────────────────────

class TestObligationType:
    def test_string_equality(self):
        assert ObligationType.RENEWAL_DATE == "Renewal Date"
        assert ObligationType.NOTICE_PERIOD == "Notice Period"
        assert ObligationType.AUTO_RENEWAL == "Auto-Renewal"

    def test_total_count(self):
        assert len(list(ObligationType)) == 6

    def test_all_types_in_definitions(self):
        for t in ObligationType:
            assert t.value in OBLIGATION_TYPE_DEFINITIONS, (
                f"Type '{t.value}' missing from OBLIGATION_TYPE_DEFINITIONS"
            )

    def test_all_types_in_valid_set(self):
        for t in ObligationType:
            assert t.value in VALID_OBLIGATION_TYPES

    def test_valid_types_count_matches_enum(self):
        assert len(VALID_OBLIGATION_TYPES) == len(list(ObligationType))


# ── Obligation dataclass tests ─────────────────────────────────────

class TestObligation:
    def _make(self, otype=ObligationType.AUTO_RENEWAL, date_value=None,
              period_value="60 days notice", confidence=0.9, ps=8, pe=8):
        return Obligation(
            clause_id="Section 12", source_name="agency.pdf",
            obligation_type=otype,
            description="Auto-renews yearly unless notice given.",
            date_value=date_value, period_value=period_value,
            page_start=ps, page_end=pe, heading="Renewal",
            confidence=confidence,
        )

    def test_is_dated_true_with_date_value(self):
        ob = self._make(date_value="2027-01-15", period_value=None)
        assert ob.is_dated is True

    def test_is_dated_false_without_date_value(self):
        ob = self._make(date_value=None, period_value="60 days notice")
        assert ob.is_dated is False

    def test_is_extraction_failure_true_at_zero_confidence(self):
        ob = self._make(confidence=0.0)
        assert ob.is_extraction_failure is True

    def test_is_extraction_failure_false_otherwise(self):
        ob = self._make(confidence=0.9)
        assert ob.is_extraction_failure is False

    def test_citation_multi_page(self):
        ob = self._make(ps=8, pe=10)
        assert ob.citation == "Section 12 (agency.pdf, pp. 8–10)"

    def test_citation_single_page(self):
        ob = self._make(ps=8, pe=8)
        assert ob.citation == "Section 12 (agency.pdf, p. 8)"

    def test_when_display_prefers_date_over_period(self):
        ob = self._make(date_value="2027-01-15", period_value="60 days notice")
        assert ob.when_display == "2027-01-15"

    def test_when_display_falls_back_to_period(self):
        ob = self._make(date_value=None, period_value="60 days notice")
        assert ob.when_display == "60 days notice"

    def test_when_display_falls_back_to_not_specified(self):
        ob = self._make(date_value=None, period_value=None)
        assert ob.when_display == "Not specified"

    def test_to_dict_contains_all_fields(self):
        ob = self._make()
        d = ob.to_dict()
        required = {"clause_id", "source_name", "obligation_type", "description",
                    "date_value", "period_value", "page_start", "page_end"}
        assert required.issubset(d.keys())

    def test_to_dict_serialises_enum_value(self):
        ob = self._make(otype=ObligationType.PAYMENT_DEADLINE)
        d = ob.to_dict()
        assert d["obligation_type"] == "Payment Deadline"

    def test_repr_contains_key_info(self):
        ob = self._make(otype=ObligationType.AUTO_RENEWAL, period_value="60 days notice")
        r = repr(ob)
        assert "Auto-Renewal" in r
        assert "Section 12" in r


# ── Definitions coverage ───────────────────────────────────────────

class TestDefinitionsCoverage:
    def test_definitions_are_non_empty(self):
        for k, v in OBLIGATION_TYPE_DEFINITIONS.items():
            assert len(v) > 20, f"Definition for {k} is too short"


# ── UI constants tests ─────────────────────────────────────────────

class TestUIConstants:
    def test_colors_for_all_types(self):
        for t in ObligationType:
            assert t in OBLIGATION_COLORS

    def test_icons_for_all_types(self):
        for t in ObligationType:
            assert t in OBLIGATION_ICONS

    def test_badge_css_for_all_types(self):
        for t in ObligationType:
            assert t in OBLIGATION_BADGE_CSS


# ── Extractor helper unit tests ────────────────────────────────────

class TestExtractorHelpers:
    """
    Tests for extractor.py helper functions without real API calls.
    Mirrors TestScannerHelpers in tests/risk/test_scanner.py.
    """

    def _make_batch(self):
        return [
            {"clause_id": "Section 12", "page_start": 8, "page_end": 8,
             "heading": "Renewal", "full_text": "This Agreement renews automatically..."},
            {"clause_id": "Section 20", "page_start": 15, "page_end": 15,
             "heading": "Governing Law", "full_text": "This Agreement is governed by..."},
        ]

    def test_system_prompt_contains_obligation_types(self):
        from src.obligations.extractor import _build_system_prompt
        prompt = _build_system_prompt()
        assert "Renewal Date" in prompt
        assert "Notice Period" in prompt
        assert "Auto-Renewal" in prompt
        assert "JSON array" in prompt

    def test_user_prompt_contains_all_clauses(self):
        from src.obligations.extractor import _build_user_prompt
        batch = self._make_batch()
        prompt = _build_user_prompt(batch)
        assert "Section 12" in prompt
        assert "Section 20" in prompt
        assert "CLAUSE 1" in prompt
        assert "CLAUSE 2" in prompt

    def test_parse_valid_response_with_obligation(self):
        from src.obligations.extractor import _parse_llm_response
        batch = self._make_batch()
        response = json.dumps([
            {"clause_id": "Section 12", "has_obligation": True,
             "obligation_type": "Auto-Renewal",
             "description": "Renews yearly unless notice given.",
             "date_value": None, "period_value": "60 days notice",
             "confidence": 0.9},
            {"clause_id": "Section 20", "has_obligation": False,
             "obligation_type": None, "description": None,
             "date_value": None, "period_value": None, "confidence": 0.85},
        ])
        results = _parse_llm_response(response, batch, "agency.pdf")
        assert len(results) == 2
        assert results[0]["obligation"] is not None
        assert results[0]["obligation"].obligation_type == ObligationType.AUTO_RENEWAL
        assert results[1]["obligation"] is None

    def test_parse_valid_response_with_fixed_date(self):
        from src.obligations.extractor import _parse_llm_response
        batch = self._make_batch()[:1]
        response = json.dumps([
            {"clause_id": "Section 12", "has_obligation": True,
             "obligation_type": "Renewal Date",
             "description": "Expires on a fixed date.",
             "date_value": "2027-01-15", "period_value": None,
             "confidence": 0.95},
        ])
        results = _parse_llm_response(response, batch, "agency.pdf")
        ob = results[0]["obligation"]
        assert ob.date_value == "2027-01-15"
        assert ob.is_dated is True

    def test_parse_strips_markdown_fences(self):
        from src.obligations.extractor import _parse_llm_response
        batch = self._make_batch()[:1]
        response = "```json\n" + json.dumps([
            {"clause_id": "Section 12", "has_obligation": False,
             "obligation_type": None, "description": None,
             "date_value": None, "period_value": None, "confidence": 0.8}
        ]) + "\n```"
        results = _parse_llm_response(response, batch, "agency.pdf")
        assert len(results) == 1

    def test_parse_unknown_obligation_type_becomes_other(self):
        from src.obligations.extractor import _parse_llm_response
        batch = self._make_batch()[:1]
        response = json.dumps([
            {"clause_id": "Section 12", "has_obligation": True,
             "obligation_type": "FakeType",
             "description": "Something.", "date_value": None,
             "period_value": "30 days", "confidence": 0.7}
        ])
        results = _parse_llm_response(response, batch, "agency.pdf")
        assert results[0]["obligation"].obligation_type == ObligationType.OTHER_DEADLINE

    def test_parse_wrong_clause_id_uses_positional_fallback(self):
        from src.obligations.extractor import _parse_llm_response
        batch = self._make_batch()[:1]
        response = json.dumps([
            {"clause_id": "WRONG_ID", "has_obligation": False,
             "obligation_type": None, "description": None,
             "date_value": None, "period_value": None, "confidence": 0.8}
        ])
        results = _parse_llm_response(response, batch, "agency.pdf")
        assert results[0]["clause_id"] == "Section 12"  # positional fallback

    def test_parse_invalid_json_returns_empty(self):
        from src.obligations.extractor import _parse_llm_response
        batch = self._make_batch()
        assert _parse_llm_response("not json", batch, "agency.pdf") == []
        assert _parse_llm_response("", batch, "agency.pdf") == []
        assert _parse_llm_response("{}", batch, "agency.pdf") == []

    def test_fallback_results_produced_for_all_batch_items(self):
        from src.obligations.extractor import _fallback_results
        batch = self._make_batch()
        results = _fallback_results(batch, "agency.pdf")
        assert len(results) == 2
        assert all(r["obligation"].is_extraction_failure for r in results)
        assert all(r["obligation"].source_name == "agency.pdf" for r in results)

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


# ── ExtractionResult tests ─────────────────────────────────────────

class TestExtractionResult:
    def _make_obligations(self):
        return [
            Obligation(
                clause_id="Section 12", source_name="c.pdf",
                obligation_type=ObligationType.RENEWAL_DATE,
                description="Expires on a date.", date_value="2027-01-15",
                period_value=None, page_start=8, page_end=8, confidence=0.9,
            ),
            Obligation(
                clause_id="Section 15", source_name="c.pdf",
                obligation_type=ObligationType.PAYMENT_DEADLINE,
                description="Net 30.", date_value=None,
                period_value="net 30 days", page_start=10, page_end=10, confidence=0.85,
            ),
            Obligation(
                clause_id="Section 20", source_name="c.pdf",
                obligation_type=ObligationType.OTHER_DEADLINE,
                description="Extraction failed.", date_value=None,
                period_value=None, page_start=15, page_end=15, confidence=0.0,
            ),
        ]

    def test_obligations_found_excludes_failures(self):
        from src.obligations.extractor import ExtractionResult
        er = ExtractionResult("c.pdf", 10, self._make_obligations(), 1, 12.0)
        assert er.obligations_found == 2

    def test_dated_count_excludes_failures_and_relative(self):
        from src.obligations.extractor import ExtractionResult
        er = ExtractionResult("c.pdf", 10, self._make_obligations(), 1, 12.0)
        assert er.dated_count == 1

    def test_success_false_when_failures_present(self):
        from src.obligations.extractor import ExtractionResult
        er = ExtractionResult("c.pdf", 10, self._make_obligations(), 1, 12.0)
        assert er.success is False

    def test_success_true_when_no_failures(self):
        from src.obligations.extractor import ExtractionResult
        er = ExtractionResult("c.pdf", 10, self._make_obligations()[:2], 0, 12.0)
        assert er.success is True

    def test_repr_contains_counts(self):
        from src.obligations.extractor import ExtractionResult
        er = ExtractionResult("c.pdf", 10, self._make_obligations(), 1, 12.0)
        r = str(er)
        assert "2 obligations found" in r
        assert "1 clauses failed" in r


# ── Integration tests ──────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.slow
class TestExtractorIntegration:
    def test_extract_obligations_returns_result(self, require_api_key, ingested_contract):
        """ingested_contract parses+chunks a real sample PDF into tmp_sqlite
        so this doesn't depend on a pre-existing data/metadata.db, and runs
        once per contract in EXTRACTOR_TEST_CONTRACTS so it isn't coupled
        to any single document."""
        from src.obligations.extractor import extract_obligations
        conn, source_name = ingested_contract
        result = extract_obligations(source_name, conn=conn)
        assert result.total_clauses > 0
        assert all(isinstance(o, Obligation) for o in result.obligations)

    def test_extract_all_obligations_have_valid_type(self, require_api_key, ingested_contract):
        from src.obligations.extractor import extract_obligations
        conn, source_name = ingested_contract
        result = extract_obligations(source_name, conn=conn)
        valid = set(ObligationType)
        assert all(o.obligation_type in valid for o in result.obligations)

    def test_extract_obligations_have_non_empty_description(self, require_api_key, ingested_contract):
        from src.obligations.extractor import extract_obligations
        conn, source_name = ingested_contract
        result = extract_obligations(source_name, conn=conn)
        real = [o for o in result.obligations if not o.is_extraction_failure]
        assert all(len(o.description) > 5 for o in real)

    def test_extract_dated_obligations_have_iso_looking_dates(self, require_api_key, ingested_contract):
        from src.obligations.extractor import extract_obligations
        conn, source_name = ingested_contract
        result = extract_obligations(source_name, conn=conn)
        dated = [o for o in result.obligations if o.is_dated]
        for o in dated:
            assert o.date_value is not None
            # Loose check: YYYY-MM-DD shape, not a strict calendar validation
            parts = o.date_value.split("-")
            assert len(parts) == 3, f"date_value {o.date_value!r} not in YYYY-MM-DD shape"
            assert len(parts[0]) == 4
