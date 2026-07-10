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
from src.utils.logger import setup_logging

# Configure logging once — all modules inherit this
setup_logging()

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

# ── Landing page content ───────────────────────────────────────────
st.title("⚖️ ClauseInsight")
st.markdown(
    "**Upload a contract. Understand every clause. Know every risk.**"
)

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📄 Upload")
    st.markdown(
        "Upload any legal contract PDF — NDA, MSA, employment agreement, "
        "SaaS terms. ClauseInsight parses and indexes every clause automatically."
    )

with col2:
    st.markdown("### 💬 Ask Questions")
    st.markdown(
        "Ask plain-English questions and get answers cited to the exact "
        "clause and page number. No legal jargon required."
    )

with col3:
    st.markdown("### 🔴 Risk Scanner")
    st.markdown(
        "Every clause is automatically classified as LOW, MEDIUM, or HIGH risk "
        "with a plain-English reason and recommended action."
    )

st.divider()
st.markdown(
    "👈 **Get started** by selecting **Upload Contract** from the sidebar."
)
