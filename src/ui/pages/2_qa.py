"""
ClauseInsight — Q&A Page
==========================

Let the user ask plain-English questions about the active contract
(or all contracts) and get answers grounded in retrieved clause text,
with citations and optional inline risk classification.
"""

import streamlit as st
from dotenv import load_dotenv

import sys
from pathlib import Path
root_dir = str(Path(__file__).resolve().parent.parent.parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

load_dotenv()

import os
from google import genai
from google.genai import types as genai_types

from src.utils.logger import get_logger
from src.utils.store import (
    get_chroma_collection,
    get_sqlite_connection,
    list_ingested_contracts,
    DEFAULT_CHROMA_DIR,
    DEFAULT_SQLITE_PATH,
)
from src.retrieval.retriever import retrieve, retrieve_for_contract
from src.retrieval.context_builder import build_qa_context
from src.risk.risk_labels import RISK_COLORS, RISK_ICONS, RiskLevel

logger = get_logger(__name__)

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Q&A · ClauseInsight",
    page_icon="💬",
    layout="wide",
)

st.title("💬 Ask the Contract")
st.markdown("Ask a plain-English question and get an answer cited to the exact clause and page.")

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
    st.warning("No contracts ingested yet. Go to **Upload Contract** to get started.")
    st.stop()

# ── Sidebar: contract selection + settings ─────────────────────────
with st.sidebar:
    st.markdown("### Contract")
    contract_options = ["🌐 All contracts"] + [c["source_name"] for c in contracts]

    # Default to active_contract from session if set
    default_idx = 0
    if "active_contract" in st.session_state:
        try:
            default_idx = contract_options.index(st.session_state["active_contract"])
        except ValueError:
            default_idx = 0

    selected = st.selectbox("Search in", contract_options, index=default_idx)
    source_filter = None if selected == "🌐 All contracts" else selected

    st.markdown("### Settings")
    top_k = st.slider("Clauses to retrieve", min_value=1, max_value=10, value=5)
    show_risk = st.toggle("Show risk level for retrieved clauses", value=True)
    show_context = st.toggle("Show retrieved clause text", value=False)

    if "active_contract" in st.session_state:
        st.success(f"**Active:** {st.session_state['active_contract']}")

# ── LLM answer generation ──────────────────────────────────────────
def generate_answer(context_text: str, query: str) -> str:
    """Call Gemini to generate a grounded answer from context."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return "❌ GOOGLE_API_KEY not set. Check your .env file."

    client = genai.Client(api_key=api_key)

    system_prompt = """You are a legal contract analyst. Answer the user's question
based ONLY on the contract clauses provided. Be precise and cite the specific clause.

Rules:
- Only use information from the provided clauses
- Always mention the clause ID and page number when referencing specific text
- If the answer is not in the provided clauses, say so clearly
- Write for a non-lawyer — avoid jargon where possible
- Keep answers concise (3-5 sentences unless complexity requires more)
- Never fabricate clauses or page numbers"""

    user_prompt = f"{context_text}\n\nQUESTION: {query}\n\nAnswer based only on the clauses above:"

    try:
        response = client.models.generate_content(
            model=os.environ.get("LLM_MODEL", "gemini-2.0-flash"),
            contents=user_prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=float(os.environ.get("LLM_TEMPERATURE", "0.2")),
                max_output_tokens=int(os.environ.get("LLM_MAX_TOKENS", "1024")),
            ),
        )
        return response.text
    except Exception as exc:
        logger.error("LLM call failed: %s", exc)
        return f"❌ Error generating answer: {exc}"

# ── Chat history ───────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Display past messages
for msg in st.session_state["chat_history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("citations"):
            with st.expander("📎 Sources"):
                for cit in msg["citations"]:
                    st.markdown(f"- {cit}")

# ── Query input ────────────────────────────────────────────────────
query = st.chat_input("Ask a question about the contract...")

if query:
    # Show user message
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state["chat_history"].append({"role": "user", "content": query})

    with st.chat_message("assistant"):
        with st.spinner("Searching clauses..."):

            # ── Retrieve ───────────────────────────────────────────
            try:
                if source_filter:
                    chunks = retrieve_for_contract(
                        query, source_filter,
                        top_k=top_k, collection=collection,
                    )
                else:
                    chunks = retrieve(
                        query, top_k=top_k, collection=collection,
                    )
            except Exception as exc:
                st.error(f"Retrieval failed: {exc}")
                logger.exception("Retrieval error for query: %s", query)
                st.stop()

            if not chunks:
                answer = (
                    "I couldn't find any relevant clauses for that question. "
                    "Try rephrasing or check that the contract has been ingested."
                )
                st.markdown(answer)
                st.session_state["chat_history"].append(
                    {"role": "assistant", "content": answer, "citations": []}
                )
                st.stop()

            # ── Build context ──────────────────────────────────────
            ctx = build_qa_context(chunks, query=query)

            if ctx.was_truncated:
                st.caption(
                    "⚠️ Some clauses were truncated to fit the context window."
                )

            # ── Generate answer ────────────────────────────────────
            with st.spinner("Generating answer..."):
                answer = generate_answer(ctx.context_text, query)

        st.markdown(answer)

        # ── Citations ──────────────────────────────────────────────
        with st.expander(f"📎 Sources ({ctx.chunk_count} clauses)"):
            for chunk in ctx.chunks:
                risk_badge = ""
                if show_risk:
                    # Quick inline risk from similarity score as proxy
                    # (full scan is in 3_scanner.py — this is lightweight)
                    score = chunk.similarity_score
                    if score >= 0.8:
                        risk_badge = " · 🟢 High relevance"
                    elif score >= 0.6:
                        risk_badge = " · 🟡 Medium relevance"
                    else:
                        risk_badge = " · 🔴 Low relevance"

                st.markdown(f"**{chunk.citation}**{risk_badge}")

                if show_context:
                    st.markdown(
                        f"> {chunk.full_text[:400]}{'...' if len(chunk.full_text) > 400 else ''}"
                    )

        # ── Multi-contract source indicator ───────────────────────
        if ctx.is_multi_contract:
            st.caption(
                f"Results drawn from: {', '.join(sorted(ctx.source_names))}"
            )

        # Store in history
        st.session_state["chat_history"].append({
            "role": "assistant",
            "content": answer,
            "citations": ctx.citations,
        })

# ── Clear chat button ─────────────────────────────────────────────
if st.session_state["chat_history"]:
    if st.button("🗑️ Clear chat history"):
        st.session_state["chat_history"] = []
        st.rerun()
