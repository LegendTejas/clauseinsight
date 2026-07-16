"""
ClauseInsight — Upload Contract Page
======================================

Handles PDF upload, full ingestion pipeline, and shows progress
to the user in real time.

Pipeline triggered here:
    PDF bytes → parser.py → chunker.py → embedder.py → ChromaDB + SQLite

Uses st.session_state to pass the ingested source_name to the Q&A
and scanner pages without re-processing.
"""

import os
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
    delete_contract,
    DEFAULT_CHROMA_DIR,
    DEFAULT_SQLITE_PATH,
)
from src.pipeline.parser import parse_pdf, PasswordProtectedError, CorruptedPDFError
from src.pipeline.chunker import chunk_document
from src.pipeline.embedder import embed_and_store
from src.ui.theme import (
    apply_theme, gradient_header, sidebar_brand, glass_card, feature_card, top_bar,
)

logger = get_logger(__name__)

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Upload · ClauseInsight",
    page_icon="📄",
    layout="wide",
)

# ── Apply Theme ────────────────────────────────────────────────────
apply_theme()
top_bar()
sidebar_brand()

# ── Page Header ────────────────────────────────────────────────────
gradient_header(
    title="Upload Contract",
    subtitle="Upload a legal contract PDF to begin. Supported: NDA, MSA, employment agreements, SaaS terms.",
    emoji="📄",
)

MAX_MB = int(os.environ.get("MAX_UPLOAD_MB", "50"))

# ── Shared store connections (cached for session) ──────────────────
@st.cache_resource
def get_stores():
    collection = get_chroma_collection(DEFAULT_CHROMA_DIR)
    conn = get_sqlite_connection(DEFAULT_SQLITE_PATH)
    return collection, conn

collection, conn = get_stores()

# ── Already ingested contracts ─────────────────────────────────────
st.markdown("""
<div style="animation:fadeInUp 0.5s ease-out;">
    <h2 style="
        font-family:'Inter',sans-serif;
        font-weight:700;
        font-size:1.3rem;
        color:#F1F5F9;
        margin:0 0 0.8rem 0;
    ">📋 Ingested Contracts</h2>
</div>
""", unsafe_allow_html=True)

contracts = list_ingested_contracts(conn)

if not contracts:
    st.info("No contracts ingested yet. Upload one below to get started.")
else:
    theme_mode = st.session_state.get("theme_mode", "dark")
    is_light = theme_mode == "light"
    for idx, contract in enumerate(contracts):
        is_selected = (
            st.session_state.get("active_contract") == contract["source_name"]
        )
        if is_selected:
            border_color = "rgba(139,92,246,0.6)"
            glow = "box-shadow:0 0 15px rgba(139,92,246,0.1);"
        else:
            border_color = "#000000" if is_light else "rgba(255,255,255,0.08)"
            glow = ""
        
        text_color = "#000000" if is_light else "#F1F5F9"

        # Contract card with glassmorphism
        st.markdown(f"""
<div style="
    background:rgba(255,255,255,0.03);
    border:1px solid {border_color};
    border-radius:12px;
    padding:0.9rem 1.2rem;
    margin-bottom:0.5rem;
    backdrop-filter:blur(10px);
    animation:fadeInUp 0.5s ease-out {idx * 0.08}s both;
    transition:all 0.3s cubic-bezier(0.4,0,0.2,1); {glow}
    display:flex;
    align-items:center;
    gap:1rem;
"
onmouseover="this.style.borderColor='rgba(255,255,255,0.15)';this.style.boxShadow='0 4px 16px rgba(0,0,0,0.3)';"
onmouseout="this.style.borderColor='{border_color}';this.style.boxShadow='{"0 0 15px rgba(139,92,246,0.1)" if is_selected else "none"}';"
>
    <div style="flex:1;">
        <span style="
            font-family:'Inter',sans-serif;
            font-weight:600;
            color:{text_color};
            font-size:0.95rem;
        ">{'✅ ' if is_selected else '📄 '}{contract['source_name']}</span>
    </div>
    <div style="
        font-family:'Inter',sans-serif;
        color:#64748B;
        font-size:0.78rem;
        font-weight:500;
    ">
        <span style="
            background:rgba(139,92,246,0.1);
            color:#8B5CF6;
            padding:2px 8px;
            border-radius:6px;
            font-weight:600;
            margin-right:0.5rem;
        ">{contract['chunk_count']} chunks</span>
        {contract['last_ingested'][:16]}
    </div>
</div>
""", unsafe_allow_html=True)

        # Streamlit buttons need to be outside the HTML card
        col_select, col_spacer = st.columns([1, 6])
        with col_select:
            if st.button("Select", key=f"select_{contract['source_name']}",
                         type="primary" if not is_selected else "secondary"):
                st.session_state["active_contract"] = contract["source_name"]
                st.success(f"Selected: {contract['source_name']}")
                st.rerun()

    # Delete contract option
    with st.expander("🗑️ Remove a contract"):
        to_delete = st.selectbox(
            "Select contract to remove",
            options=[c["source_name"] for c in contracts],
            key="delete_select",
        )
        if st.button("Remove", type="secondary"):
            n = delete_contract(to_delete, collection, conn)
            if st.session_state.get("active_contract") == to_delete:
                del st.session_state["active_contract"]
            st.success(f"Removed {n} chunks for '{to_delete}'.")
            st.rerun()

st.divider()

# ── Upload section ─────────────────────────────────────────────────
st.markdown("""
<div style="animation:fadeInUp 0.6s ease-out;">
    <h2 style="
        font-family:'Inter',sans-serif;
        font-weight:700;
        font-size:1.3rem;
        color:#F1F5F9;
        margin:0 0 0.8rem 0;
    ">⬆️ Upload New Contract</h2>
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader(
    f"Choose a PDF (max {MAX_MB} MB)",
    type=["pdf"],
    help="The contract will be parsed, chunked, and embedded automatically.",
)

if uploaded is not None:
    # Size check
    size_mb = len(uploaded.getvalue()) / (1024 * 1024)
    if size_mb > MAX_MB:
        st.error(f"File too large ({size_mb:.1f} MB). Maximum is {MAX_MB} MB.")
        st.stop()

    # File info card
    st.markdown(f"""
<div style="
    background:rgba(139,92,246,0.05);
    border:1px solid rgba(139,92,246,0.2);
    border-radius:12px;
    padding:0.8rem 1.2rem;
    margin:0.5rem 0;
    animation:fadeInUp 0.4s ease-out;
    display:flex;
    align-items:center;
    gap:0.8rem;
">
    <span style="font-size:1.5rem;">📎</span>
    <div>
        <span style="font-family:'Inter',sans-serif;font-weight:600;color:#F1F5F9;">
            {uploaded.name}
        </span>
        <span style="font-family:'Inter',sans-serif;color:#64748B;font-size:0.85rem;margin-left:0.5rem;">
            {size_mb:.2f} MB
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

    # Check if already ingested
    existing = [c["source_name"] for c in contracts]
    if uploaded.name in existing:
        st.warning(
            f"'{uploaded.name}' is already ingested. "
            "Uploading again will skip already-embedded chunks (idempotent)."
        )

    if st.button("🚀 Ingest Contract", type="primary"):
        pdf_bytes = uploaded.getvalue()

        # ── Step 1: Parse ──────────────────────────────────────────
        with st.status("Processing contract...", expanded=True) as status:
            st.write("📖 Parsing PDF...")
            try:
                parsed = parse_pdf(pdf_bytes, source_name=uploaded.name)
            except PasswordProtectedError:
                st.error(
                    "This PDF is password protected. "
                    "Please upload an unlocked version."
                )
                st.stop()
            except CorruptedPDFError:
                st.error(
                    "This file could not be read as a valid PDF. "
                    "It may be corrupted."
                )
                st.stop()
            except Exception as exc:
                st.error(f"Unexpected error during parsing: {exc}")
                logger.exception("Parse failed for %s", uploaded.name)
                st.stop()

            if parsed.likely_scanned:
                st.warning(
                    "⚠️ This PDF appears to be a scanned image with no text layer. "
                    "Extraction may be incomplete. OCR is not currently supported."
                )

            st.write(
                f"✅ Parsed: {parsed.total_pages} pages, "
                f"{parsed.total_word_count:,} words"
            )

            # ── Step 2: Chunk ──────────────────────────────────────
            st.write("✂️ Detecting format and chunking clauses...")
            try:
                chunks = chunk_document(parsed, source=pdf_bytes)
            except Exception as exc:
                st.error(f"Chunking failed: {exc}")
                logger.exception("Chunking failed for %s", uploaded.name)
                st.stop()

            st.write(f"✅ Chunked: {len(chunks)} clauses identified")

            # ── Step 3: Embed ──────────────────────────────────────
            st.write("🧠 Embedding clauses and storing...")
            try:
                result = embed_and_store(
                    chunks,
                    source_name=uploaded.name,
                    collection=collection,
                    conn=conn,
                )
            except EnvironmentError as exc:
                st.error(str(exc))
                st.stop()
            except Exception as exc:
                st.error(f"Embedding failed: {exc}")
                logger.exception("Embedding failed for %s", uploaded.name)
                st.stop()

            if result.failed_count > 0:
                st.warning(
                    f"⚠️ {result.failed_count} chunks failed to embed. "
                    "Try re-ingesting — the successful chunks are already stored."
                )

            st.write(
                f"✅ Stored: {result.embedded_count} embedded, "
                f"{result.skipped_count} skipped (already existed), "
                f"{result.elapsed_seconds:.1f}s"
            )

            status.update(label="✅ Contract ready!", state="complete")

        # Set as active contract and prompt navigation
        st.session_state["active_contract"] = uploaded.name

        # Success card
        st.markdown(f"""
<div style="
    background:rgba(16,185,129,0.06);
    border:1px solid rgba(16,185,129,0.25);
    border-radius:12px;
    padding:1.2rem;
    margin-top:1rem;
    animation:fadeInUp 0.5s ease-out;
">
    <p style="
        font-family:'Inter',sans-serif;
        font-weight:600;
        color:#10B981;
        font-size:1.05rem;
        margin:0 0 0.4rem 0;
    ">✅ '{uploaded.name}' is ready!</p>
    <p style="
        font-family:'Inter',sans-serif;
        color:#94A3B8;
        font-size:0.9rem;
        margin:0;
    ">Use the sidebar to ask questions or run the risk scanner.</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar: active contract indicator ────────────────────────────
if "active_contract" in st.session_state:
    st.sidebar.success(
        f"**Active contract:**\n{st.session_state['active_contract']}"
    )
else:
    st.sidebar.info("No contract selected.")



