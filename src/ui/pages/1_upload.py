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

logger = get_logger(__name__)

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Upload · ClauseInsight",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Upload Contract")
st.markdown("Upload a legal contract PDF to begin. Supported formats: NDA, MSA, employment agreements, SaaS terms.")

MAX_MB = int(os.environ.get("MAX_UPLOAD_MB", "50"))

# ── Shared store connections (cached for session) ──────────────────
@st.cache_resource
def get_stores():
    collection = get_chroma_collection(DEFAULT_CHROMA_DIR)
    conn = get_sqlite_connection(DEFAULT_SQLITE_PATH)
    return collection, conn

collection, conn = get_stores()

# ── Already ingested contracts ─────────────────────────────────────
st.subheader("Ingested Contracts")

contracts = list_ingested_contracts(conn)

if not contracts:
    st.info("No contracts ingested yet. Upload one below to get started.")
else:
    for contract in contracts:
        col1, col2, col3, col4 = st.columns([4, 1, 2, 1])
        with col1:
            # Highlight selected contract
            is_selected = (
                st.session_state.get("active_contract") == contract["source_name"]
            )
            label = f"{'✅ ' if is_selected else ''}{contract['source_name']}"
            st.markdown(f"**{label}**")
        with col2:
            st.markdown(f"{contract['chunk_count']} chunks")
        with col3:
            st.markdown(f"Ingested: {contract['last_ingested'][:16]}")
        with col4:
            if st.button("Select", key=f"select_{contract['source_name']}"):
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
st.subheader("Upload New Contract")

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

    st.markdown(f"**File:** {uploaded.name} · {size_mb:.2f} MB")

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
        st.success(
            f"**'{uploaded.name}' is ready.** "
            "Use the sidebar to ask questions or run the risk scanner."
        )

# ── Sidebar: active contract indicator ────────────────────────────
if "active_contract" in st.session_state:
    st.sidebar.success(
        f"**Active contract:**\n{st.session_state['active_contract']}"
    )
else:
    st.sidebar.info("No contract selected.")
