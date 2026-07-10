"""
ClauseInsight — Risk Scanner Page
===================================

Runs the risk scanner on the active contract and displays results
as an interactive dashboard — summary metrics, filterable table,
and expandable clause details.
"""

import streamlit as st
from dotenv import load_dotenv

import sys
from pathlib import Path
root_dir = str(Path(__file__).resolve().parent.parent.parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

load_dotenv()

from src.utils.logger import get_logger
from src.utils.store import (
    get_chroma_collection,
    get_sqlite_connection,
    list_ingested_contracts,
    DEFAULT_CHROMA_DIR,
    DEFAULT_SQLITE_PATH,
)
from src.risk.scanner import scan_contract
from src.risk.risk_labels import (
    RISK_COLORS, RISK_ICONS, RISK_BADGE_CSS,
    RiskLevel, ClauseCategory,
)

logger = get_logger(__name__)

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Risk Scanner · ClauseInsight",
    page_icon="🔴",
    layout="wide",
)

st.title("🔴 Risk Scanner")
st.markdown(
    "Automatically classifies every clause as **LOW**, **MEDIUM**, or **HIGH** risk "
    "with a plain-English reason and recommended action."
)

# ── Shared store connections ───────────────────────────────────────
@st.cache_resource
def get_stores():
    return (
        get_chroma_collection(DEFAULT_CHROMA_DIR),
        get_sqlite_connection(DEFAULT_SQLITE_PATH),
    )

collection, conn = get_stores()
contracts = list_ingested_contracts(conn)

if not contracts:
    st.warning("No contracts ingested yet. Go to **Upload Contract** first.")
    st.stop()

# ── Sidebar: contract selection + scan settings ────────────────────
with st.sidebar:
    st.markdown("### Contract to Scan")
    contract_names = [c["source_name"] for c in contracts]

    default_idx = 0
    if "active_contract" in st.session_state:
        try:
            default_idx = contract_names.index(st.session_state["active_contract"])
        except ValueError:
            default_idx = 0

    selected_contract = st.selectbox("Select contract", contract_names, index=default_idx)

    st.markdown("### Scan Settings")
    skip_sub = st.toggle(
        "Top-level clauses only",
        value=True,
        help="Skips sub-clauses like Section 4(a). Faster and avoids redundant classifications.",
    )
    batch_size = st.select_slider(
        "Clauses per API call",
        options=[3, 5, 7, 10],
        value=5,
        help="Larger batches are faster but slightly less accurate.",
    )

    if "active_contract" in st.session_state:
        st.success(f"**Active:** {st.session_state['active_contract']}")

# ── Scan trigger ───────────────────────────────────────────────────
# Cache scan results per contract so re-renders don't re-run the scan
if "scan_results" not in st.session_state:
    st.session_state["scan_results"] = {}

col_btn, col_info = st.columns([1, 4])
with col_btn:
    run_scan = st.button("🔍 Run Risk Scan", type="primary")
with col_info:
    if selected_contract in st.session_state["scan_results"]:
        prev = st.session_state["scan_results"][selected_contract]
        st.caption(
            f"Last scan: {prev.total_clauses} clauses · "
            f"{prev.high_count} HIGH · {prev.medium_count} MEDIUM · "
            f"{prev.low_count} LOW · {prev.elapsed_seconds:.1f}s"
        )

if run_scan:
    with st.spinner(f"Scanning '{selected_contract}'... this may take a minute."):
        try:
            result = scan_contract(
                source_name=selected_contract,
                conn=conn,
                batch_size=batch_size,
                skip_sub_clauses=skip_sub,
            )
            st.session_state["scan_results"][selected_contract] = result
            st.session_state["active_contract"] = selected_contract
        except EnvironmentError as exc:
            st.error(str(exc))
            st.stop()
        except Exception as exc:
            st.error(f"Scan failed: {exc}")
            logger.exception("Scan failed for %s", selected_contract)
            st.stop()

# ── Display results ────────────────────────────────────────────────
if selected_contract not in st.session_state["scan_results"]:
    st.info("Click **Run Risk Scan** to analyse this contract.")
    st.stop()

result = st.session_state["scan_results"][selected_contract]

if not result.labels:
    st.warning("No clauses found. Make sure the contract has been ingested.")
    st.stop()

st.divider()

# ── Summary metrics ────────────────────────────────────────────────
st.subheader("Summary")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total Clauses", result.total_clauses)
m2.metric("🔴 HIGH", result.high_count)
m3.metric("🟡 MEDIUM", result.medium_count)
m4.metric("🟢 LOW", result.low_count)
m5.metric("⚪ Unknown", result.unknown_count)

if result.high_count > 0:
    st.error(
        f"⚠️ **{result.high_count} HIGH risk clause(s) found.** "
        "Review these before signing."
    )
elif result.medium_count > 0:
    st.warning(
        f"**{result.medium_count} MEDIUM risk clause(s) found.** "
        "Worth reviewing carefully."
    )
else:
    st.success("✅ No HIGH or MEDIUM risk clauses detected.")

st.divider()

# ── Filter controls ────────────────────────────────────────────────
st.subheader("Clause Results")

filter_col1, filter_col2 = st.columns(2)
with filter_col1:
    risk_filter = st.multiselect(
        "Filter by risk level",
        options=["HIGH", "MEDIUM", "LOW", "UNKNOWN"],
        default=["HIGH", "MEDIUM"],
    )
with filter_col2:
    category_filter = st.multiselect(
        "Filter by category",
        options=sorted(set(l.category.value for l in result.labels)),
        default=[],
    )

# Apply filters
filtered = result.labels
if risk_filter:
    filtered = [l for l in filtered if l.risk_level.value in risk_filter]
if category_filter:
    filtered = [l for l in filtered if l.category.value in category_filter]

# Sort: HIGH first, then MEDIUM, then LOW
risk_order = {RiskLevel.HIGH: 0, RiskLevel.MEDIUM: 1, RiskLevel.LOW: 2, RiskLevel.UNKNOWN: 3}
filtered = sorted(filtered, key=lambda l: (risk_order.get(l.risk_level, 3), l.page_start))

st.caption(f"Showing {len(filtered)} of {result.total_clauses} clauses")

# ── Clause cards ───────────────────────────────────────────────────
for label in filtered:
    icon = RISK_ICONS.get(label.risk_level, "⚪")
    color = RISK_COLORS.get(label.risk_level, "#808080")

    with st.expander(
        f"{icon} {label.clause_id} — {label.category.value} · {label.source_name}",
        expanded=label.is_high_risk,   # auto-expand HIGH risk clauses
    ):
        # Risk badge
        badge_css = RISK_BADGE_CSS.get(label.risk_level, "")
        st.markdown(
            f'<span style="{badge_css}">{label.risk_level.value}</span>',
            unsafe_allow_html=True,
        )

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("**📍 Location**")
            st.markdown(label.citation)

            st.markdown("**📋 Category**")
            st.markdown(label.category.value)

        with col_right:
            st.markdown("**⚠️ Why this risk level?**")
            st.markdown(label.reason)

            st.markdown("**✅ Recommended Action**")
            st.markdown(label.recommended_action)

# ── Export ────────────────────────────────────────────────────────
st.divider()
st.subheader("Export")

if st.button("📥 Download results as JSON"):
    import json
    export_data = {
        "contract": selected_contract,
        "summary": {
            "total": result.total_clauses,
            "high": result.high_count,
            "medium": result.medium_count,
            "low": result.low_count,
            "unknown": result.unknown_count,
        },
        "clauses": [l.to_dict() for l in result.labels],
    }
    st.download_button(
        label="💾 Save JSON",
        data=json.dumps(export_data, indent=2),
        file_name=f"{selected_contract}_risk_report.json",
        mime="application/json",
    )
