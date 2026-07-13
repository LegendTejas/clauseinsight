"""
Tests for src/risk/web_grounding.py

Unit tests: query building, result parsing/dedup, per-clause retry/fallback —
all against mocked ddgs.DDGS calls, so no real network traffic runs in CI.
Integration tests: real DuckDuckGo searches via ddgs, marked with
@pytest.mark.integration. These are free (no billing, no API key) but
still marked integration since they depend on network access and an
external service's availability/rate limits.
"""

from __future__ import annotations

import pytest

from ddgs.exceptions import DDGSException, RatelimitException

from src.risk.risk_labels import RiskLabel, RiskLevel, ClauseCategory
from src.risk.web_grounding import (
    WebSource,
    GroundingResult,
    _build_grounding_query,
    _extract_sources_from_results,
)


# ── WebSource / GroundingResult tests ──────────────────────────────

class TestWebSource:
    def test_to_dict(self):
        src = WebSource(title="Example", url="https://example.com", snippet="A snippet.")
        assert src.to_dict() == {
            "title": "Example", "url": "https://example.com", "snippet": "A snippet."
        }

    def test_snippet_defaults_empty(self):
        src = WebSource(title="Example", url="https://example.com")
        assert src.snippet == ""


class TestGroundingResult:
    def _make(self, sources=None, error=None):
        return GroundingResult(
            clause_id="Section 9", source_name="agency.pdf",
            query_used="test query", sources=sources or [], error=error,
        )

    def test_success_true_without_error(self):
        assert self._make().success is True

    def test_success_false_with_error(self):
        assert self._make(error="failed").success is False

    def test_has_sources_true(self):
        gr = self._make(sources=[WebSource("A", "https://a.com")])
        assert gr.has_sources is True

    def test_has_sources_false_when_empty(self):
        assert self._make(sources=[]).has_sources is False

    def test_has_sources_false_when_errored_even_with_sources(self):
        gr = self._make(sources=[WebSource("A", "https://a.com")], error="failed")
        assert gr.has_sources is False


# ── Query construction tests ───────────────────────────────────────

class TestBuildGroundingQuery:
    def _make_label(self, reason=None):
        return RiskLabel(
            clause_id="Section 9", source_name="agency.pdf",
            risk_level=RiskLevel.HIGH, category=ClauseCategory.INDEMNIFICATION,
            reason=reason or "Unlimited indemnification with no liability cap.",
            recommended_action="Negotiate a cap.",
            page_start=8, page_end=10, heading="Indemnification",
        )

    def test_query_contains_category(self):
        query = _build_grounding_query(self._make_label())
        assert "Indemnification" in query

    def test_query_contains_reason_keywords(self):
        query = _build_grounding_query(self._make_label())
        assert "indemnification" in query.lower()

    def test_query_truncates_long_reason(self):
        long_reason = " ".join(f"word{i}" for i in range(50))
        label = self._make_label(reason=long_reason)
        query = _build_grounding_query(label)
        # Only first 12 words of reason should appear, not all 50.
        assert "word49" not in query
        assert "word0" in query

    def test_query_is_keyword_style_not_a_question(self):
        query = _build_grounding_query(self._make_label())
        assert not query.strip().endswith("?")


# ── Result parsing tests ───────────────────────────────────────────

class TestExtractSourcesFromResults:
    def test_extracts_single_result(self):
        raw = [{"title": "Source A", "href": "https://a.com", "body": "Some snippet."}]
        sources = _extract_sources_from_results(raw)
        assert len(sources) == 1
        assert sources[0].url == "https://a.com"
        assert sources[0].title == "Source A"
        assert sources[0].snippet == "Some snippet."

    def test_extracts_multiple_results(self):
        raw = [
            {"title": "Source A", "href": "https://a.com", "body": "A"},
            {"title": "Source B", "href": "https://b.com", "body": "B"},
        ]
        sources = _extract_sources_from_results(raw)
        assert len(sources) == 2

    def test_deduplicates_by_url(self):
        raw = [
            {"title": "Source A", "href": "https://a.com", "body": "A"},
            {"title": "Source A Again", "href": "https://a.com", "body": "A2"},
        ]
        sources = _extract_sources_from_results(raw)
        assert len(sources) == 1

    def test_caps_at_max_sources(self):
        from src.risk import web_grounding
        raw = [
            {"title": f"Source {i}", "href": f"https://site{i}.com", "body": ""}
            for i in range(10)
        ]
        sources = _extract_sources_from_results(raw)
        assert len(sources) <= web_grounding.MAX_SOURCES_PER_CLAUSE

    def test_skips_results_missing_url(self):
        raw = [{"title": "No URL", "body": "..."}]
        sources = _extract_sources_from_results(raw)
        assert len(sources) == 0

    def test_falls_back_to_url_when_title_missing(self):
        raw = [{"href": "https://a.com", "body": ""}]
        sources = _extract_sources_from_results(raw)
        assert sources[0].title == "https://a.com"

    def test_empty_results_returns_empty(self):
        assert _extract_sources_from_results([]) == []

    def test_ignores_non_dict_entries(self):
        raw = ["not a dict", {"title": "A", "href": "https://a.com", "body": ""}]
        sources = _extract_sources_from_results(raw)
        assert len(sources) == 1

    def test_missing_body_defaults_to_empty_snippet(self):
        raw = [{"title": "A", "href": "https://a.com"}]
        sources = _extract_sources_from_results(raw)
        assert sources[0].snippet == ""


# ── ground_clause retry/fallback tests (mocked ddgs) ────────────────

class TestGroundClause:
    def _make_label(self):
        return RiskLabel(
            clause_id="Section 9", source_name="agency.pdf",
            risk_level=RiskLevel.HIGH, category=ClauseCategory.INDEMNIFICATION,
            reason="Unlimited indemnification.",
            recommended_action="Negotiate a cap.",
            page_start=8, page_end=10,
        )

    def test_returns_grounding_result_on_success(self, monkeypatch):
        from src.risk import web_grounding

        def fake_call(query):
            return [{"title": "Source A", "href": "https://a.com", "body": "..."}]

        monkeypatch.setattr(web_grounding, "_call_web_search", fake_call)
        result = web_grounding.ground_clause(self._make_label())
        assert result.success is True
        assert len(result.sources) == 1

    def test_returns_error_result_after_all_retries_fail(self, monkeypatch):
        from src.risk import web_grounding

        def failing_call(query):
            raise DDGSException("network error")

        monkeypatch.setattr(web_grounding, "_call_web_search", failing_call)
        monkeypatch.setattr(web_grounding, "RETRY_DELAY", 0)
        result = web_grounding.ground_clause(self._make_label())
        assert result.success is False
        assert result.sources == []
        assert result.error is not None

    def test_retries_before_succeeding(self, monkeypatch):
        from src.risk import web_grounding

        calls = {"count": 0}

        def flaky_call(query):
            calls["count"] += 1
            if calls["count"] < 2:
                raise RatelimitException("rate limited")
            return []

        monkeypatch.setattr(web_grounding, "_call_web_search", flaky_call)
        monkeypatch.setattr(web_grounding, "RETRY_DELAY", 0)
        result = web_grounding.ground_clause(self._make_label())
        assert result.success is True
        assert calls["count"] == 2

    def test_handles_non_ddgs_exceptions_too(self, monkeypatch):
        from src.risk import web_grounding

        def failing_call(query):
            raise ConnectionError("no network")

        monkeypatch.setattr(web_grounding, "_call_web_search", failing_call)
        monkeypatch.setattr(web_grounding, "RETRY_DELAY", 0)
        result = web_grounding.ground_clause(self._make_label())
        assert result.success is False


# ── ground_high_risk_clauses tests (mocked ddgs) ────────────────────

class TestGroundHighRiskClauses:
    def _make_labels(self):
        high = RiskLabel(
            clause_id="Section 9", source_name="agency.pdf",
            risk_level=RiskLevel.HIGH, category=ClauseCategory.INDEMNIFICATION,
            reason="Unlimited indemnification.", recommended_action="Cap it.",
            page_start=8, page_end=10,
        )
        medium = RiskLabel(
            clause_id="Section 4", source_name="agency.pdf",
            risk_level=RiskLevel.MEDIUM, category=ClauseCategory.TERMINATION,
            reason="Somewhat one-sided.", recommended_action="Review.",
            page_start=2, page_end=3,
        )
        return [high, medium]

    def test_filters_to_high_risk_only(self, monkeypatch):
        from src.risk import web_grounding

        monkeypatch.setattr(
            web_grounding, "ground_clause",
            lambda label: GroundingResult(
                clause_id=label.clause_id, source_name=label.source_name,
                query_used="q", sources=[],
            ),
        )
        monkeypatch.setattr(web_grounding, "INTER_GROUND_SLEEP", 0)

        results = web_grounding.ground_high_risk_clauses(self._make_labels())
        assert len(results) == 1
        assert "agency.pdf::Section 9" in results

    def test_empty_when_no_high_risk_labels(self, monkeypatch):
        from src.risk import web_grounding
        medium_only = [l for l in self._make_labels() if l.risk_level != RiskLevel.HIGH]
        results = web_grounding.ground_high_risk_clauses(medium_only)
        assert results == {}

    def test_caps_at_max_clauses(self, monkeypatch):
        from src.risk import web_grounding

        many_high = [
            RiskLabel(
                clause_id=f"Section {i}", source_name="agency.pdf",
                risk_level=RiskLevel.HIGH, category=ClauseCategory.GENERAL,
                reason="r", recommended_action="a", page_start=i, page_end=i,
            )
            for i in range(5)
        ]

        monkeypatch.setattr(
            web_grounding, "ground_clause",
            lambda label: GroundingResult(
                clause_id=label.clause_id, source_name=label.source_name,
                query_used="q", sources=[],
            ),
        )
        monkeypatch.setattr(web_grounding, "INTER_GROUND_SLEEP", 0)

        results = web_grounding.ground_high_risk_clauses(many_high, max_clauses=2)
        assert len(results) == 2


# ── Integration tests ──────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.slow
class TestWebGroundingIntegration:
    def test_ground_clause_real_search(self):
        """Real ddgs call — free, but depends on network + DuckDuckGo
        availability, so kept in the integration tier rather than run
        by default in CI."""
        from src.risk.web_grounding import ground_clause

        label = RiskLabel(
            clause_id="Section 9", source_name="agency.pdf",
            risk_level=RiskLevel.HIGH, category=ClauseCategory.INDEMNIFICATION,
            reason="Unlimited indemnification with no liability cap.",
            recommended_action="Negotiate a cap.",
            page_start=8, page_end=10,
        )
        result = ground_clause(label)
        assert isinstance(result, GroundingResult)
        # We don't assert has_sources — real search results vary — only
        # that the call completes and returns a well-formed result.
        assert result.clause_id == "Section 9"
