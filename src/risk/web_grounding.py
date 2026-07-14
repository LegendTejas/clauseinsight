"""
ClauseInsight — Web Search Grounding
=======================================

Finds supporting web sources (case law summaries, "is X clause standard"
articles, regulatory guidance) for HIGH and MEDIUM risk clauses, using DuckDuckGo
search via the `ddgs` package.

WHY DDGS INSTEAD OF THE OPENAI WEB_SEARCH TOOL
--------------------------------------------------
The OpenAI Responses API's web_search tool works (see git history for
the earlier version of this module) but is billed per call ($10/1k
calls + ~8,000 input tokens per search — see OpenAI's pricing page).
For a student project where every extra billed feature eats into a
limited API budget, `ddgs` gets the same "find supporting sources"
outcome at zero cost: no API key, no signup, no billing. It queries
DuckDuckGo's public search results directly.

Trade-off worth knowing: `ddgs` isn't an official, contractually
supported API — it's a maintained open-source client that scrapes
DuckDuckGo's result pages, so there's no formal SLA and it can
occasionally get rate-limited under heavy use. For the volume this
project needs (a handful of searches per demo/review session, capped
per run — see MAX_CLAUSES_PER_RUN), that's a non-issue in practice,
and it's a common, well-understood trade-off for hobby/student projects.

WHY ONLY FLAGGED CLAUSES
----------------------------
Even though search itself is free now, keeping this scoped to HIGH and MEDIUM risk
clauses still makes sense: LOW clauses are rarely the ones a
reviewer needs external validation for, and unscoped high-volume queries
are exactly the pattern most likely to trigger DuckDuckGo rate limiting.

PIPELINE POSITION
------------------
    list[RiskLabel] (FLAGGED only) → web_grounding.py → dict[str, GroundingResult]
                                                       → 3_scanner.py (display)

This is a separate, optional pass on top of scanner.py's output — it
does not change scan_contract()'s behaviour. The UI calls this only
when the user explicitly asks for it (a button click).

ERROR HANDLING
---------------
A failed search for one clause shouldn't block the others — each
clause's search is independent, so failures are caught per-clause and
reported as a GroundingResult with an `error` message and an empty
source list, rather than retried aggressively.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from ddgs import DDGS
from ddgs.exceptions import DDGSException

from pathlib import Path as _Path
import sys as _sys
_root_dir = str(_Path(__file__).resolve().parent.parent.parent)
if _root_dir not in _sys.path:
    _sys.path.insert(0, _root_dir)

from src.risk.risk_labels import RiskLabel, RiskLevel

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────

# DuckDuckGo region for search results — "us-en" is a sensible default
# for legal/commercial contract queries in English.
DDGS_REGION = "us-en"

# Max sources kept per clause, after de-duplicating by URL.
MAX_SOURCES_PER_CLAUSE = 3

# ddgs returns more than we display so a mediocre top result doesn't
# starve better ones further down — trimmed to MAX_SOURCES_PER_CLAUSE
# after fetching.
RAW_RESULTS_PER_QUERY = 6

# Sleep between clauses — polite pacing to avoid tripping DuckDuckGo's
# informal rate limiting under repeated runs.
INTER_GROUND_SLEEP = 1.5  # seconds

# Retries are minimal — a failed search shouldn't block other clauses.
MAX_GROUND_RETRIES = 2
RETRY_DELAY = 3.0  # seconds

# Hard cap on how many HIGH risk clauses get grounded in one run,
# even if more are selected — keeps runs quick and avoids hammering
# DuckDuckGo with a long burst of queries.
MAX_CLAUSES_PER_RUN = 10


# ──────────────────────────────────────────────────────────────────
# Result types
# ──────────────────────────────────────────────────────────────────

@dataclass
class WebSource:
    """One supporting web source found for a clause."""
    title:   str
    url:     str
    snippet: str = ""

    def to_dict(self) -> dict:
        return {"title": self.title, "url": self.url, "snippet": self.snippet}


@dataclass
class GroundingResult:
    """
    Web search grounding outcome for one HIGH risk clause.

    Attributes
    ----------
    clause_id / source_name:
        Identify which clause this grounds — same identifiers as RiskLabel,
        so the UI can look this up by (clause_id, source_name) key.

    query_used:
        The search query actually sent — shown in the UI for transparency,
        since "why did it find these sources" matters for a legal tool.

    sources:
        De-duplicated list of WebSource, capped at MAX_SOURCES_PER_CLAUSE.
        Each carries its own snippet — unlike the OpenAI web_search tool,
        ddgs doesn't synthesize a combined answer, so there's no separate
        summary field; the snippets themselves are the evidence.

    error:
        None on success. Set to a short message if the search call failed
        after retries — the UI shows this instead of a source list.
    """
    clause_id:   str
    source_name: str
    query_used:  str
    sources:     list[WebSource] = field(default_factory=list)
    error:       Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None

    @property
    def has_sources(self) -> bool:
        return self.success and len(self.sources) > 0


# ──────────────────────────────────────────────────────────────────
# Query construction
# ──────────────────────────────────────────────────────────────────

def _build_grounding_query(label: RiskLabel) -> str:
    """
    Build a web search query from a flagged clause's classification.

    Uses the category + a trimmed slice of the reason (not the full
    clause text, and not a full natural-language question) — DuckDuckGo
    is a keyword search engine, not an LLM, so a short keyword-style
    query built from category + risk terms returns more relevant pages
    than a conversational question would.
    """
    reason_keywords = " ".join(label.reason.split()[:12])
    return f"{label.category.value} clause {reason_keywords} commercial contract"


# ──────────────────────────────────────────────────────────────────
# ddgs search call
# ──────────────────────────────────────────────────────────────────

def _call_web_search(query: str) -> list[dict]:
    """
    Run a DuckDuckGo text search via ddgs and return raw result dicts.
    Raises on failure — caller handles retries.

    ddgs.DDGS().text() result dicts have keys: "title", "href", "body".
    """
    with DDGS() as ddgs:
        return ddgs.text(
            query,
            region=DDGS_REGION,
            safesearch="moderate",
            max_results=RAW_RESULTS_PER_QUERY,
        )


def _extract_sources_from_results(raw_results: list[dict]) -> list[WebSource]:
    """
    Convert raw ddgs result dicts into de-duplicated WebSource objects,
    capped at MAX_SOURCES_PER_CLAUSE.

    Defensive on purpose: ddgs is an unofficial client scraping public
    search pages, so result dicts occasionally have a missing/odd field —
    a malformed result should be skipped, not crash the whole run.
    """
    sources: list[WebSource] = []
    seen_urls: set[str] = set()

    for result in raw_results:
        if not isinstance(result, dict):
            continue
        url = result.get("href")
        if not url or url in seen_urls:
            continue
        title = result.get("title") or url
        snippet = result.get("body") or ""
        seen_urls.add(url)
        sources.append(WebSource(title=title, url=url, snippet=snippet))
        if len(sources) >= MAX_SOURCES_PER_CLAUSE:
            break

    return sources


# ──────────────────────────────────────────────────────────────────
# Per-clause grounding with retries
# ──────────────────────────────────────────────────────────────────

def ground_clause(label: RiskLabel) -> GroundingResult:
    """
    Find supporting web sources for one HIGH risk clause.

    Retries a small, fixed number of times on transient failures
    (DuckDuckGo rate limiting is the most common one — see
    RatelimitException below). On total failure, returns a
    GroundingResult with `error` set rather than raising — callers
    processing many clauses should not have one failed search abort
    the rest.
    """
    query = _build_grounding_query(label)
    delay = RETRY_DELAY

    for attempt in range(1, MAX_GROUND_RETRIES + 1):
        try:
            raw_results = _call_web_search(query)
            sources = _extract_sources_from_results(raw_results)
            return GroundingResult(
                clause_id=label.clause_id,
                source_name=label.source_name,
                query_used=query,
                sources=sources,
            )
        except DDGSException as exc:
            logger.warning(
                "Grounding attempt %d/%d failed for %r: %s",
                attempt, MAX_GROUND_RETRIES, label.clause_id, exc,
            )
            if attempt < MAX_GROUND_RETRIES:
                time.sleep(delay)
                delay *= 2
        except Exception as exc:
            # Non-ddgs errors (e.g. network issues) — same retry treatment.
            logger.warning(
                "Grounding attempt %d/%d failed for %r: %s",
                attempt, MAX_GROUND_RETRIES, label.clause_id, exc,
            )
            if attempt < MAX_GROUND_RETRIES:
                time.sleep(delay)
                delay *= 2

    logger.error(
        "Grounding failed after %d attempts for %r — returning empty result.",
        MAX_GROUND_RETRIES, label.clause_id,
    )
    return GroundingResult(
        clause_id=label.clause_id,
        source_name=label.source_name,
        query_used=query,
        sources=[],
        error="Web search failed after retries. Try again or search manually.",
    )


# ──────────────────────────────────────────────────────────────────
# Main entry point — grounds a batch of flagged clauses
# ──────────────────────────────────────────────────────────────────

def ground_flagged_clauses(
    labels: list[RiskLabel],
    max_clauses: int = MAX_CLAUSES_PER_RUN,
) -> dict[str, GroundingResult]:
    """
    Find supporting web sources for HIGH and MEDIUM risk clauses in a scan result.

    Args:
        labels:      RiskLabel list — typically result.labels from
                     scan_contract(). Non-flagged labels are filtered out
                     internally, so callers can pass the full list.
        max_clauses: Hard cap on clauses grounded in one call, to bound
                     cost. Extra flagged clauses beyond this are skipped
                     (not errored) — logged so it's visible why.

    Returns:
        Dict keyed by f"{source_name}::{clause_id}" (matches your
        chunk store's composite ID convention) mapping to GroundingResult.
    """
    flagged_risk = [l for l in labels if l.is_flagged]

    if not flagged_risk:
        return {}

    if len(flagged_risk) > max_clauses:
        logger.info(
            "%d flagged clauses found, grounding only the first %d "
            "(keeps the run quick) — increase max_clauses to ground more.",
            len(flagged_risk), max_clauses,
        )
        flagged_risk = flagged_risk[:max_clauses]

    results: dict[str, GroundingResult] = {}

    for i, label in enumerate(flagged_risk):
        key = f"{label.source_name}::{label.clause_id}"
        results[key] = ground_clause(label)

        if i < len(flagged_risk) - 1:
            time.sleep(INTER_GROUND_SLEEP)

    return results


def ground_high_risk_clauses(
    labels: list[RiskLabel],
    max_clauses: int = MAX_CLAUSES_PER_RUN,
) -> dict[str, GroundingResult]:
    """
    Backward-compatible wrapper for callers/tests that expect HIGH-only grounding.
    """
    high_risk_labels = [label for label in labels if label.risk_level == RiskLevel.HIGH]
    return ground_flagged_clauses(high_risk_labels, max_clauses=max_clauses)


# ──────────────────────────────────────────────────────────────────
# Smoke test:
#   python src/risk/web_grounding.py
#   (no API key needed — ddgs is free)
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import logging
    from src.risk.risk_labels import RiskLevel, ClauseCategory

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    sample_label = RiskLabel(
        clause_id="Section 9",
        source_name="agency.pdf",
        risk_level=RiskLevel.HIGH,
        category=ClauseCategory.INDEMNIFICATION,
        reason="This clause requires unlimited indemnification with no cap on liability.",
        recommended_action="Negotiate a liability cap tied to contract value.",
        page_start=8, page_end=10, heading="Indemnification",
    )

    print(f"\nGrounding: {sample_label.clause_id!r}")
    result = ground_clause(sample_label)

    print(f"\n{'='*50}")
    print(f"Query: {result.query_used}")
    print(f"Success: {result.success}")
    if result.error:
        print(f"Error: {result.error}")
    print(f"\nSources found ({len(result.sources)}):")
    for src in result.sources:
        print(f"  - {src.title}\n    {src.url}")
        if src.snippet:
            print(f"    {src.snippet[:150]}...")
