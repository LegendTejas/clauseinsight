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
from src.risk.web_grounding import ground_flagged_clauses, MAX_CLAUSES_PER_RUN
from src.ui.theme import (
    apply_theme, gradient_header, top_bar, sidebar_brand, risk_badge_html
)

logger = get_logger(__name__)

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Risk Scanner · ClauseInsight",
    page_icon="🛡️",
    layout="wide",
)

# ── Apply Theme ────────────────────────────────────────────────────
apply_theme()
top_bar()
sidebar_brand()

# ── Page Header ────────────────────────────────────────────────────
gradient_header(
    title="Risk Scanner",
    subtitle="Automatically classifies every clause as LOW, MEDIUM, or HIGH risk with a plain-English reason and recommended action.",
    emoji="🛡️",
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
    st.markdown("""
<div style="
    font-family:'Inter',sans-serif;
    font-weight:700;
    font-size:0.9rem;
    color:#94A3B8;
    letter-spacing:0.05em;
    text-transform:uppercase;
    margin-bottom:0.5rem;
">Contract to Scan</div>
""", unsafe_allow_html=True)

    contract_names = [c["source_name"] for c in contracts]

    default_idx = 0
    if "active_contract" in st.session_state:
        try:
            default_idx = contract_names.index(st.session_state["active_contract"])
        except ValueError:
            default_idx = 0

    selected_contract = st.selectbox("Select contract", contract_names, index=default_idx)

    st.markdown("""
<div style="
    font-family:'Inter',sans-serif;
    font-weight:700;
    font-size:0.9rem;
    color:#94A3B8;
    letter-spacing:0.05em;
    text-transform:uppercase;
    margin:1rem 0 0.5rem 0;
">Scan Settings</div>
""", unsafe_allow_html=True)

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

col_btn, col_info = st.columns([1.5, 3.5])
with col_btn:
    run_scan = st.button("🔍 Run Risk Scan", type="primary")
with col_info:
    if selected_contract in st.session_state["scan_results"]:
        prev = st.session_state["scan_results"][selected_contract]
        st.markdown(f"""
<div style="
    display:flex;
    align-items:center;
    gap:0.6rem;
    flex-wrap:wrap;
    padding:0.3rem 0;
    animation:fadeIn 0.5s ease-out;
">
    <span style="
        font-family:'Inter',sans-serif;
        color:#64748B;
        font-size:0.82rem;
    ">Last scan:</span>
    <span style="background:rgba(255,255,255,0.06);color:#F1F5F9;padding:2px 8px;border-radius:6px;font-family:'Inter',sans-serif;font-size:0.8rem;font-weight:600;">
        {prev.total_clauses} clauses
    </span>
    <span style="background:rgba(255,75,75,0.1);color:#FF4B4B;padding:2px 8px;border-radius:6px;font-family:'Inter',sans-serif;font-size:0.8rem;font-weight:600;">
        {prev.high_count} HIGH
    </span>
    <span style="background:rgba(245,158,11,0.1);color:#F59E0B;padding:2px 8px;border-radius:6px;font-family:'Inter',sans-serif;font-size:0.8rem;font-weight:600;">
        {prev.medium_count} MEDIUM
    </span>
    <span style="background:rgba(16,185,129,0.1);color:#10B981;padding:2px 8px;border-radius:6px;font-family:'Inter',sans-serif;font-size:0.8rem;font-weight:600;">
        {prev.low_count} LOW
    </span>
    <span style="font-family:'Inter',sans-serif;color:#475569;font-size:0.78rem;">
        {prev.elapsed_seconds:.1f}s
    </span>
</div>
""", unsafe_allow_html=True)

import concurrent.futures
import threading
from src.utils.store import BACKGROUND_TASKS

if "bg_executor" not in st.session_state:
    st.session_state["bg_executor"] = concurrent.futures.ThreadPoolExecutor(max_workers=4)

job_key = f"scan_job_{selected_contract}"
prog_key = f"scan_prog_{selected_contract}"
res_key = f"scan_res_{selected_contract}"

if run_scan:
    BACKGROUND_TASKS[job_key] = "running"
    BACKGROUND_TASKS[prog_key] = (0, 1)

    def background_scan():
        try:
            def update_progress(current, total):
                BACKGROUND_TASKS[prog_key] = (current, total)
                
            from src.utils.store import get_sqlite_connection
            thread_conn = get_sqlite_connection()
            result = scan_contract(
                source_name=selected_contract,
                conn=thread_conn,
                batch_size=batch_size,
                skip_sub_clauses=skip_sub,
                progress_callback=update_progress,
                is_cancelled=lambda: BACKGROUND_TASKS.get(job_key) == "cancelled"
            )
            
            BACKGROUND_TASKS[res_key] = result
            BACKGROUND_TASKS[job_key] = "done"
        except Exception as exc:
            logger.exception("Scan failed for %s", selected_contract)
            BACKGROUND_TASKS[job_key] = f"Error: {exc}"

    st.session_state["bg_executor"].submit(background_scan)
    st.rerun()

status = BACKGROUND_TASKS.get(job_key)

if status == "running":
    @st.fragment(run_every="1s")
    def poll_scan_progress():
        if BACKGROUND_TASKS.get(job_key) != "running":
            st.rerun()
            return
            
        current, total = BACKGROUND_TASKS.get(prog_key, (0, 1))
        pct = int((current / total) * 100) if total > 0 else 0
        
        if pct < 100:
            icon_html = """<style>
.loader {
  border: 2px solid rgba(255,255,255,0.1);
  border-top: 2px solid #3B82F6;
  border-radius: 50%;
  width: 14px;
  height: 14px;
  animation: spin 1s linear infinite;
  display: inline-block;
}
@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
</style>
<div class="loader"></div>"""
        else:
            icon_html = "✅"
            
        st.markdown(f"""
<div style="display:flex; align-items:center; gap:8px; margin-bottom: 4px;">
    <span style="font-size: 14px;">Scanning '{selected_contract}' in background... {pct}% ({current}/{total} batches)</span>
    {icon_html}
</div>
""", unsafe_allow_html=True)
        st.progress(pct)
        
        if st.button("🛑 Cancel Scan", use_container_width=True):
            BACKGROUND_TASKS[job_key] = "cancelled"
            st.rerun()
        
    poll_scan_progress()
    st.info("The scan is running securely in the background. You can safely navigate to other pages and it will continue!")
    st.stop()
elif status == "done":
    if "scan_results" not in st.session_state:
        st.session_state["scan_results"] = {}
    st.session_state["scan_results"][selected_contract] = BACKGROUND_TASKS.pop(res_key, None)
    st.session_state["active_contract"] = selected_contract
    del BACKGROUND_TASKS[job_key]
    st.rerun()
elif status == "cancelled":
    st.warning("Scan was cancelled by the user.")
    if st.button("Dismiss"):
        del BACKGROUND_TASKS[job_key]
        st.rerun()
    st.stop()
elif str(status).startswith("Error:"):
    st.error(status)
    if st.button("Dismiss"):
        del BACKGROUND_TASKS[job_key]
        st.rerun()
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
st.markdown("""
<div style="animation:fadeInUp 0.5s ease-out;">
    <h2 style="
        font-family:'Inter',sans-serif;
        font-weight:700;
        font-size:1.3rem;
        color:#F1F5F9;
        margin:0 0 0.8rem 0;
    ">📊 Summary</h2>
</div>
""", unsafe_allow_html=True)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total Clauses", result.total_clauses)
m2.metric("🔴 HIGH", result.high_count)
m3.metric("🟡 MEDIUM", result.medium_count)
m4.metric("🟢 LOW", result.low_count)
m5.metric("⚪ Unknown", result.unknown_count)

# Risk summary banner
if result.high_count > 0:
    st.markdown(f"""
<div style="
    background:rgba(255,75,75,0.08);
    border:1px solid rgba(255,75,75,0.25);
    border-left:4px solid #FF4B4B;
    border-radius:12px;
    padding:1rem 1.2rem;
    margin:0.8rem 0;
    animation:fadeInUp 0.6s ease-out;
">
    <p style="
        font-family:'Inter',sans-serif;
        font-weight:600;
        color:#FF4B4B;
        font-size:1rem;
        margin:0;
    ">⚠️ {result.high_count} HIGH risk clause(s) found — review these before signing.</p>
</div>
""", unsafe_allow_html=True)
elif result.medium_count > 0:
    st.markdown(f"""
<div style="
    background:rgba(245,158,11,0.08);
    border:1px solid rgba(245,158,11,0.25);
    border-left:4px solid #F59E0B;
    border-radius:12px;
    padding:1rem 1.2rem;
    margin:0.8rem 0;
    animation:fadeInUp 0.6s ease-out;
">
    <p style="
        font-family:'Inter',sans-serif;
        font-weight:600;
        color:#F59E0B;
        font-size:1rem;
        margin:0;
    ">{result.medium_count} MEDIUM risk clause(s) found — worth reviewing carefully.</p>
</div>
""", unsafe_allow_html=True)
else:
    st.markdown("""
<div style="
    background:rgba(16,185,129,0.08);
    border:1px solid rgba(16,185,129,0.25);
    border-left:4px solid #10B981;
    border-radius:12px;
    padding:1rem 1.2rem;
    margin:0.8rem 0;
    animation:fadeInUp 0.6s ease-out;
">
    <p style="
        font-family:'Inter',sans-serif;
        font-weight:600;
        color:#10B981;
        font-size:1rem;
        margin:0;
    ">✅ No HIGH or MEDIUM risk clauses detected.</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Filter controls ────────────────────────────────────────────────
st.markdown("""
<div style="animation:fadeInUp 0.5s ease-out;">
    <h2 style="
        font-family:'Inter',sans-serif;
        font-weight:700;
        font-size:1.3rem;
        color:#F1F5F9;
        margin:0 0 0.8rem 0;
    ">📋 Clause Results</h2>
</div>
""", unsafe_allow_html=True)

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

# ── Web search grounding for FLAGGED risk clauses ─────────────────────
# Free (DuckDuckGo via ddgs, no API key) — still opt-in and scoped to
# HIGH and MEDIUM only, mainly to keep runs quick and avoid rate limiting under
# heavy use. See src/risk/web_grounding.py
if "grounding_results" not in st.session_state:
    st.session_state["grounding_results"] = {}

flagged_in_view = [l for l in filtered if l.is_flagged]

if flagged_in_view:
    ground_col1, ground_col2 = st.columns([1, 4])
    with ground_col1:
        run_grounding = st.button("🌐 Find Supporting Sources")
    with ground_col2:
        capped = len(flagged_in_view) > MAX_CLAUSES_PER_RUN
        st.caption(
            f"Free web search (DuckDuckGo) for {min(len(flagged_in_view), MAX_CLAUSES_PER_RUN)} "
            f"HIGH/MEDIUM risk clause(s) currently shown"
            + (f" (capped from {len(flagged_in_view)} to keep the run quick)" if capped else "")
            + "."
        )

    if run_grounding:
        with st.spinner("Searching the web for supporting sources..."):
            try:
                new_results = ground_flagged_clauses(flagged_in_view)
                st.session_state["grounding_results"].update(new_results)
            except Exception as exc:
                st.error(f"Web search grounding failed: {exc}")
                logger.exception("Grounding failed for %s", selected_contract)

# ── Clause cards ───────────────────────────────────────────────────
for idx, label in enumerate(filtered):
    icon = RISK_ICONS.get(label.risk_level, "⚪")
    color = RISK_COLORS.get(label.risk_level, "#808080")

    with st.expander(
        f"{icon} {label.clause_id} — {label.category.value} · {label.source_name}",
        expanded=label.is_high_risk,   # auto-expand HIGH risk clauses
    ):
        # Styled risk badge
        st.markdown(
            risk_badge_html(label.risk_level.value),
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

        # Supporting web sources — only shown for flagged (HIGH/MEDIUM) risk clauses that
        # have been grounded (via the "Find Supporting Sources" button above)
        if label.is_flagged:
            grounding_key = f"{label.source_name}::{label.clause_id}"
            grounding = st.session_state["grounding_results"].get(grounding_key)

            if grounding is not None:
                st.markdown("---")
                st.markdown("**🌐 Supporting Sources**")
                if grounding.error:
                    st.caption(f"⚠️ {grounding.error}")
                elif grounding.has_sources:
                    st.caption(f"Search: _{grounding.query_used}_")
                    for src in grounding.sources:
                        st.markdown(f"- [{src.title}]({src.url})")
                        if src.snippet:
                            st.caption(src.snippet)
                else:
                    st.caption("No relevant sources found for this clause.")

# ── Export ────────────────────────────────────────────────────────
st.divider()

st.markdown("""
<div style="animation:fadeInUp 0.5s ease-out;">
    <h2 style="
        font-family:'Inter',sans-serif;
        font-weight:700;
        font-size:1.3rem;
        color:#F1F5F9;
        margin:0 0 0.8rem 0;
    ">📥 Export</h2>
</div>
""", unsafe_allow_html=True)

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




