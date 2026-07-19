"""
ClauseInsight — Streamlit Entry Point
======================================

This file does exactly three things:
  1. Loads environment variables from .env
  2. Configures logging for the whole app
  3. Sets Streamlit page config (title, icon, layout)

Everything else — upload, Q&A, risk scanner — lives in pages/.
Streamlit automatically discovers and renders pages/ as a multipage app.

Run with:
    streamlit run src/ui/app.py
"""

import os
from dotenv import load_dotenv

import sys
from pathlib import Path
root_dir = str(Path(__file__).resolve().parent.parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Load .env before anything else so all modules see the env vars
load_dotenv()

import streamlit as st

# ── Streamlit Community Cloud secrets bridge ────────────────────────
# Locally, OPENAI_API_KEY comes from .env via load_dotenv() above.
# On Community Cloud there is no .env file — secrets are set via the
# dashboard and only exposed through st.secrets, not os.environ.
# Every other module in this app (scanner.py, embedder.py, 2_qa.py,
# extractor.py) reads the key via os.environ.get("OPENAI_API_KEY"),
# so bridging it here once, before those modules are imported/called,
# makes the rest of the codebase work unchanged in both environments.
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

from src.utils.logger import setup_logging
import threading
from src.pipeline.embedder import sync_sqlite_to_chroma
from src.ui.theme import (
    apply_theme, gradient_header, feature_card, footer, sidebar_brand, top_bar, sidebar_footer
)

# Configure logging once — all modules inherit this
setup_logging()

# ── Background Sync ────────────────────────────────────────────────
@st.cache_resource
def init_background_sync():
    thread = threading.Thread(target=sync_sqlite_to_chroma, daemon=True)
    thread.start()
    return thread

init_background_sync()

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title=os.environ.get("APP_TITLE", "ClauseInsight"),
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/LegendTejas/ClauseInsight",
        "Report a bug": "https://github.com/LegendTejas/ClauseInsight/issues",
        "About": (
            "**ClauseInsight** — Upload a contract. "
            "Understand every clause. Know every risk.\n\n"
            "Built by Tejas T. P. · Foundations of Applied ML Internship 2026"
        ),
    },
)

# ── Apply Theme ────────────────────────────────────────────────────
apply_theme()
top_bar()
sidebar_brand()

# ── Hero Header ────────────────────────────────────────────────────
gradient_header(
    title="ClauseInsight",
    subtitle="Upload a contract. Understand every clause. Know every risk.",
    emoji="⚖️",
)

st.divider()

# ── Feature Cards ──────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        feature_card(
            icon="📄",
            title="Upload & Parse",
            description=(
                "Upload any legal contract PDF — NDA, MSA, employment agreement, "
                "SaaS terms. ClauseInsight parses and indexes every clause automatically "
                "in seconds."
            ),
            delay="0.1s",
            gradient="135deg, #3B82F6, #06B6D4",
        ),
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        feature_card(
            icon="💬",
            title="Ask Questions",
            description=(
                "Ask plain-English questions and get answers cited to the exact "
                "clause and page number. No legal jargon required — just ask "
                "what you need to know."
            ),
            delay="0.2s",
            gradient="135deg, #8B5CF6, #EC4899",
        ),
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        feature_card(
            icon="🛡️",
            title="Risk Scanner",
            description=(
                "Every clause is automatically classified as LOW, MEDIUM, or HIGH risk "
                "with a plain-English reason and recommended action. Never miss a red flag."
            ),
            delay="0.3s",
            gradient="135deg, #F59E0B, #EF4444",
        ),
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

# ── Second row of features ────────────────────────────────────────
col4, col5 = st.columns(2)

with col4:
    st.markdown(
        feature_card(
            icon="📅",
            title="Obligations & Deadlines",
            description=(
                "Automatically extracts renewal dates, notice periods, termination windows, "
                "and payment deadlines into one sortable list — so you know what to act on and when."
            ),
            delay="0.4s",
            gradient="135deg, #14B8A6, #3B82F6",
        ),
        unsafe_allow_html=True,
    )

with col5:
    st.markdown(
        feature_card(
            icon="🔍",
            title="Compare Contracts",
            description=(
                "Upload two versions of a contract and ask what changed. "
                "Multi-document reasoning surfaces the differences clause by clause."
            ),
            delay="0.5s",
            gradient="135deg, #6366F1, #8B5CF6",
        ),
        unsafe_allow_html=True,
    )

st.divider()

# ── Call to Action ─────────────────────────────────────────────────
st.markdown("""
<div style="
    text-align:center;
    padding:2rem 0;
    animation:fadeInUp 0.8s ease-out 0.6s both;
">
    <p style="
        font-family:'Inter',sans-serif;
        color:#94A3B8;
        font-size:1.1rem;
        font-weight:500;
        margin:0;
    ">
        👈 <strong style="color:#F1F5F9;">Get started</strong> by selecting
        <strong style="
            background:linear-gradient(135deg, #4F46E5, #06B6D4);
            -webkit-background-clip:text;
            -webkit-text-fill-color:transparent;
            background-clip:text;
        ">Upload Contract</strong> from the sidebar
    </p>
    <div style="
        margin-top:0.8rem;
        font-size:1.5rem;
        animation:float 2s ease-in-out infinite;
        display:inline-block;
    ">←</div>
</div>
""", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────
footer()
sidebar_footer()
