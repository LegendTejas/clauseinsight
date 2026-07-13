"""
ClauseInsight — Obligations & Deadlines Page
===============================================

Runs the obligation extractor on the active contract and displays
results as a sortable timeline — renewal dates, notice periods,
termination windows, payment deadlines, and auto-renewal terms —
so a reviewer can see every date-bound action at a glance without
reading the whole contract.
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
from src.obligations.extractor import extract_obligations
from src.obligations.obligation_labels import (
    OBLIGATION_COLORS, OBLIGATION_ICONS, OBLIGATION_BADGE_CSS,
    ObligationType,
)
from src.ui.theme import (
    apply_theme, gradient_header, obligation_badge_html,
    sidebar_brand, top_bar,
)

logger = get_logger(__name__)

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Obligations & Deadlines · ClauseInsight",
    page_icon="📅",
    layout="wide",
)

# ── Apply Theme ────────────────────────────────────────────────────
apply_theme()
top_bar()
sidebar_brand()

# ── Page Header ────────────────────────────────────────────────────
gradient_header(
    title="Obligations & Deadlines",
    subtitle=(
        "Scans every clause and pulls out renewal dates, notice periods, "
        "termination windows, payment deadlines, and auto-renewal terms "
        "into one sortable list."
    ),
    emoji="📅",
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

# ── Sidebar: contract selection + extraction settings ──────────────
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
">Contract to Analyse</div>
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
">Extraction Settings</div>
""", unsafe_allow_html=True)

    skip_sub = st.toggle(
        "Top-level clauses only",
        value=True,
        help="Skips sub-clauses like Section 4(a). Faster and avoids redundant extractions.",
    )
    batch_size = st.select_slider(
        "Clauses per API call",
        options=[3, 5, 7, 10],
        value=5,
        help="Larger batches are faster but slightly less accurate.",
    )

    if "active_contract" in st.session_state:
        st.success(f"**Active:** {st.session_state['active_contract']}")

# ── Extraction trigger ──────────────────────────────────────────────
# Cache results per contract so re-renders don't re-run extraction
if "obligation_results" not in st.session_state:
    st.session_state["obligation_results"] = {}

col_btn, col_info = st.columns([1, 4])
with col_btn:
    run_extraction = st.button("📅 Extract Obligations", type="primary")
with col_info:
    if selected_contract in st.session_state["obligation_results"]:
        prev = st.session_state["obligation_results"][selected_contract]
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
    ">Last run:</span>
    <span style="background:rgba(255,255,255,0.06);color:#F1F5F9;padding:2px 8px;border-radius:6px;font-family:'Inter',sans-serif;font-size:0.8rem;font-weight:600;">
        {prev.obligations_found} obligations
    </span>
    <span style="background:rgba(59,130,246,0.1);color:#3B82F6;padding:2px 8px;border-radius:6px;font-family:'Inter',sans-serif;font-size:0.8rem;font-weight:600;">
        {prev.dated_count} with dates
    </span>
    <span style="font-family:'Inter',sans-serif;color:#475569;font-size:0.78rem;">
        {prev.elapsed_seconds:.1f}s
    </span>
</div>
""", unsafe_allow_html=True)

import concurrent.futures
import threading
from streamlit.runtime.scriptrunner import add_script_run_ctx

if "bg_executor" not in st.session_state:
    st.session_state["bg_executor"] = concurrent.futures.ThreadPoolExecutor(max_workers=4)

job_key = f"ext_job_{selected_contract}"
prog_key = f"ext_prog_{selected_contract}"

if run_extraction:
    st.session_state[job_key] = "running"
    st.session_state[prog_key] = (0, 1)

    def background_extract():
        try:
            add_script_run_ctx(threading.current_thread())
        except Exception:
            pass

        try:
            def update_progress(current, total):
                st.session_state[prog_key] = (current, total)
                
            from src.utils.store import get_sqlite_connection
            thread_conn = get_sqlite_connection()
            result = extract_obligations(
                source_name=selected_contract,
                conn=thread_conn,
                batch_size=batch_size,
                skip_sub_clauses=skip_sub,
                progress_callback=update_progress,
            )
            st.session_state["obligation_results"][selected_contract] = result
            st.session_state["active_contract"] = selected_contract
            st.session_state[job_key] = "done"
        except Exception as exc:
            logger.exception("Obligation extraction failed for %s", selected_contract)
            st.session_state[job_key] = f"Error: {exc}"

    st.session_state["bg_executor"].submit(background_extract)
    st.rerun()

if st.session_state.get(job_key) == "running":
    @st.fragment(run_every="1s")
    def poll_ext_progress():
        if st.session_state.get(job_key) != "running":
            st.rerun()
            return
            
        current, total = st.session_state.get(prog_key, (0, 1))
        pct = int((current / total) * 100) if total > 0 else 0
        st.progress(pct, text=f"Scanning '{selected_contract}' for obligations in background... {pct}% ({current}/{total} batches)")
        
    poll_ext_progress()
    st.info("The extraction is running securely in the background. You can safely navigate to other pages and it will continue!")
    st.stop()
elif str(st.session_state.get(job_key)).startswith("Error:"):
    st.error(st.session_state[job_key])
    if st.button("Dismiss"):
        del st.session_state[job_key]
        st.rerun()
    st.stop()

# ── Display results ────────────────────────────────────────────────
if selected_contract not in st.session_state["obligation_results"]:
    st.info("Click **Extract Obligations** to analyse this contract.")
    st.stop()

result = st.session_state["obligation_results"][selected_contract]

if result.total_clauses == 0:
    st.warning("No clauses found. Make sure the contract has been ingested.")
    st.stop()

real_obligations = [o for o in result.obligations if not o.is_extraction_failure]
failed_obligations = [o for o in result.obligations if o.is_extraction_failure]

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

m1, m2, m3, m4 = st.columns(4)
m1.metric("Clauses Scanned", result.total_clauses)
m2.metric("📅 Obligations Found", result.obligations_found)
m3.metric("🗓️ With Fixed Dates", result.dated_count)
m4.metric("⚠️ Failed Extractions", result.failed_count)

# Summary banners
if not real_obligations:
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
    <p style="font-family:'Inter',sans-serif;font-weight:600;color:#10B981;font-size:1rem;margin:0;">
        ✅ No dated obligations or deadlines detected in this contract.
    </p>
</div>
""", unsafe_allow_html=True)
elif result.dated_count > 0:
    st.markdown(f"""
<div style="
    background:rgba(59,130,246,0.08);
    border:1px solid rgba(59,130,246,0.25);
    border-left:4px solid #3B82F6;
    border-radius:12px;
    padding:1rem 1.2rem;
    margin:0.8rem 0;
    animation:fadeInUp 0.6s ease-out;
">
    <p style="font-family:'Inter',sans-serif;font-weight:600;color:#3B82F6;font-size:1rem;margin:0;">
        📅 {result.dated_count} obligation(s) have a specific calendar date — put these on your calendar.
    </p>
</div>
""", unsafe_allow_html=True)

if failed_obligations:
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
    <p style="font-family:'Inter',sans-serif;font-weight:600;color:#F59E0B;font-size:1rem;margin:0;">
        ⚠️ {len(failed_obligations)} clause(s) failed extraction after retries. Review these manually.
    </p>
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
    ">📋 Obligations Timeline</h2>
</div>
""", unsafe_allow_html=True)

if real_obligations:
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        type_filter = st.multiselect(
            "Filter by type",
            options=sorted(set(o.obligation_type.value for o in real_obligations)),
            default=[],
        )
    with filter_col2:
        dated_only = st.toggle("Show only obligations with a fixed date", value=False)

    filtered = real_obligations
    if type_filter:
        filtered = [o for o in filtered if o.obligation_type.value in type_filter]
    if dated_only:
        filtered = [o for o in filtered if o.is_dated]

    # Sort: dated obligations first (chronological), then relative-period ones
    filtered = sorted(
        filtered,
        key=lambda o: (o.date_value is None, o.date_value or "", o.page_start),
    )

    st.caption(f"Showing {len(filtered)} of {result.obligations_found} obligations")

    # ── Obligation cards ────────────────────────────────────────────
    for idx, ob in enumerate(filtered):
        icon = OBLIGATION_ICONS.get(ob.obligation_type, "📌")
        ob_color = OBLIGATION_COLORS.get(ob.obligation_type, "#808080")

        with st.expander(
            f"{icon} {ob.when_display} — {ob.obligation_type.value} · {ob.clause_id}",
            expanded=ob.is_dated,   # auto-expand obligations with a fixed date
        ):
            # Styled obligation badge
            st.markdown(
                obligation_badge_html(ob.obligation_type.value, ob_color),
                unsafe_allow_html=True,
            )

            col_left, col_right = st.columns(2)

            with col_left:
                st.markdown("**📍 Location**")
                st.markdown(ob.citation)

                st.markdown("**🗓️ When**")
                # Styled date/period display
                when_val = ob.when_display
                if ob.is_dated:
                    st.markdown(f"""
<span style="
    background:rgba(59,130,246,0.1);
    color:#3B82F6;
    padding:4px 10px;
    border-radius:8px;
    font-family:'Inter',sans-serif;
    font-weight:600;
    font-size:0.9rem;
">📅 {when_val}</span>
""", unsafe_allow_html=True)
                else:
                    st.markdown(when_val)

            with col_right:
                st.markdown("**📝 What it means**")
                st.markdown(ob.description)

                if ob.confidence is not None:
                    # Confidence bar
                    conf_pct = ob.confidence * 100
                    conf_color = "#10B981" if ob.confidence >= 0.7 else "#F59E0B" if ob.confidence >= 0.4 else "#EF4444"
                    st.markdown(f"""
<div style="margin-top:0.5rem;">
    <span style="
        font-family:'Inter',sans-serif;
        color:#64748B;
        font-size:0.78rem;
    ">Extraction confidence</span>
    <div style="
        background:rgba(255,255,255,0.06);
        border-radius:6px;
        height:6px;
        margin-top:4px;
        overflow:hidden;
    ">
        <div style="
            background:{conf_color};
            width:{conf_pct}%;
            height:100%;
            border-radius:6px;
            transition:width 0.5s ease;
        "></div>
    </div>
    <span style="
        font-family:'Inter',sans-serif;
        color:{conf_color};
        font-size:0.75rem;
        font-weight:600;
    ">{ob.confidence:.0%}</span>
</div>
""", unsafe_allow_html=True)
else:
    st.info("No obligations to display. Try running extraction without 'Top-level clauses only'.")

# ── Failed extractions ──────────────────────────────────────────────
if failed_obligations:
    st.divider()
    with st.expander(f"⚠️ Failed Extractions ({len(failed_obligations)})"):
        st.markdown(
            "These clauses could not be reliably analysed after retries. "
            "Review them manually — they may or may not contain obligations."
        )
        for ob in failed_obligations:
            st.markdown(f"- **{ob.clause_id}** ({ob.citation})")

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
            "clauses_scanned": result.total_clauses,
            "obligations_found": result.obligations_found,
            "dated": result.dated_count,
            "failed": result.failed_count,
        },
        "obligations": [o.to_dict() for o in real_obligations],
    }
    st.download_button(
        label="💾 Save JSON",
        data=json.dumps(export_data, indent=2),
        file_name=f"{selected_contract}_obligations.json",
        mime="application/json",
    )

# ── Footer ─────────────────────────────────────────────────────────

