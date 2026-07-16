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
import openai

from src.utils.logger import get_logger
from src.utils.store import (
    get_chroma_collection,
    get_sqlite_connection,
    list_ingested_contracts,
    save_chat_session,
    list_chat_sessions,
    load_chat_session,
    delete_chat_session,
    DEFAULT_CHROMA_DIR,
    DEFAULT_SQLITE_PATH,
)
from src.retrieval.retriever import retrieve, retrieve_for_contract
from src.retrieval.context_builder import build_qa_context
from src.risk.risk_labels import RISK_COLORS, RISK_ICONS, RiskLevel
from src.ui.theme import (
    apply_theme, gradient_header, top_bar, sidebar_brand, risk_badge_html
)

logger = get_logger(__name__)

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Q&A · ClauseInsight",
    page_icon="💬",
    layout="wide",
)

# ── Apply Theme ────────────────────────────────────────────────────
apply_theme()
top_bar()
sidebar_brand()

# ── Page Header ────────────────────────────────────────────────────
gradient_header(
    title="Ask the Contract",
    subtitle="Ask a plain-English question and get an answer cited to the exact clause and page.",
    emoji="💬",
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
    st.warning("No contracts ingested yet. Go to **Upload Contract** to get started.")
    st.stop()

# ── Sidebar: contract selection + settings ─────────────────────────
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
">Contract</div>
""", unsafe_allow_html=True)

    contract_names = [c["source_name"] for c in contracts]
    default_sel = []
    if "active_contract" in st.session_state and st.session_state["active_contract"] in contract_names:
        default_sel = [st.session_state["active_contract"]]

    selected = st.multiselect(
        "Search in (leave empty for All)", 
        contract_names, 
        default=default_sel,
        placeholder="🌐 All contracts"
    )
    source_filter = selected if selected else None

    st.markdown("""
<div style="
    font-family:'Inter',sans-serif;
    font-weight:700;
    font-size:0.9rem;
    color:#94A3B8;
    letter-spacing:0.05em;
    text-transform:uppercase;
    margin:1rem 0 0.5rem 0;
">Settings</div>
""", unsafe_allow_html=True)

    top_k = st.slider("Clauses to retrieve", min_value=1, max_value=10, value=5)
    show_risk = st.toggle("Show relevance indicators", value=True)
    show_context = st.toggle("Show retrieved clause text", value=False)

    st.markdown("---")
    st.markdown("""
<div style="
    font-family:'Inter',sans-serif;
    font-weight:700;
    font-size:0.9rem;
    color:#94A3B8;
    letter-spacing:0.05em;
    text-transform:uppercase;
    margin-bottom:0.5rem;
">🕒 Past Conversations</div>
""", unsafe_allow_html=True)

    if st.button("➕ New Chat", use_container_width=True):
        st.session_state["chat_history"] = []
        if "chat_session_id" in st.session_state:
            del st.session_state["chat_session_id"]
        st.rerun()

    past_sessions = list_chat_sessions(conn)
    if not past_sessions:
        st.caption("No past conversations yet.")
    else:
        for s in past_sessions:
            colA, colB = st.columns([8, 2])
            with colA:
                if st.button(s["title"] or "Untitled Chat", key=f"load_{s['id']}", use_container_width=True):
                    st.session_state["chat_history"] = load_chat_session(conn, s["id"])
                    st.session_state["chat_session_id"] = s["id"]
                    st.rerun()
            with colB:
                if st.button("🗑️", key=f"del_{s['id']}", help="Delete chat"):
                    # Save to session state for undo
                    msgs = load_chat_session(conn, s["id"])
                    st.session_state["recently_deleted_chat"] = {
                        "id": s["id"],
                        "title": s["title"],
                        "messages": msgs
                    }
                    delete_chat_session(conn, s["id"])
                    if st.session_state.get("chat_session_id") == s["id"]:
                        st.session_state["chat_history"] = []
                        if "chat_session_id" in st.session_state:
                            del st.session_state["chat_session_id"]
                    st.rerun()

    # Undo Button & Ctrl+Z Listener
    if "recently_deleted_chat" in st.session_state:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("↩️ Undo Delete (Ctrl+Z)", use_container_width=True, key="undo_delete_btn"):
            chat = st.session_state.pop("recently_deleted_chat")
            save_chat_session(conn, chat["id"], chat["title"], chat["messages"])
            st.toast("Chat restored!")
            st.rerun()

        import streamlit.components.v1 as components
        components.html(
            """
            <script>
            const doc = window.parent.document;
            function handleKeyDown(e) {
                if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
                    const buttons = Array.from(doc.querySelectorAll('button'));
                    const undoBtn = buttons.find(b => b.textContent.includes('Undo Delete'));
                    if (undoBtn) {
                        e.preventDefault();
                        undoBtn.click();
                        doc.removeEventListener('keydown', handleKeyDown);
                    }
                }
            }
            doc.removeEventListener('keydown', doc.handleKeyDownRef);
            doc.handleKeyDownRef = handleKeyDown;
            doc.addEventListener('keydown', handleKeyDown);
            </script>
            """,
            height=0,
            width=0,
        )

    if "active_contract" in st.session_state:
        st.success(f"**Active:** {st.session_state['active_contract']}")

# ── LLM answer generation ──────────────────────────────────────────
def generate_answer(context_text: str, query: str) -> str:
    """Call OpenAI to generate a grounded answer from context."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return "❌ OPENAI_API_KEY not set. Check your .env file."

    client = openai.OpenAI(api_key=api_key)

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
        response = client.chat.completions.create(
            model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=float(os.environ.get("LLM_TEMPERATURE", "0.2")),
            max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "1024")),
        )
        if not response.choices or not response.choices[0].message.content:
            return "❌ LLM returned an empty response."
        return response.choices[0].message.content
    except Exception as exc:
        logger.error("LLM call failed: %s", exc)
        return f"❌ Error generating answer: {exc}"

# ── Chat history ───────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Display past messages
for idx, msg in enumerate(st.session_state["chat_history"]):
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            # For user messages, place the copy button on the right corner
            col_text, col_btn = st.columns([15, 1])
            with col_text:
                st.markdown(msg["content"])
            with col_btn:
                if st.button("\u200b", icon=":material/content_copy:", key=f"copy_{idx}", type="tertiary", help="Copy question"):
                    st.toast("Question ready to copy! (Select and Ctrl+C)")
        else:
            # For assistant, text is rendered first, then sources, then action buttons at the bottom
            st.markdown(msg["content"])
            
            if msg.get("citations"):
                with st.expander("📎 Sources"):
                    for cit in msg["citations"]:
                        st.markdown(f"- {cit}")
            
            # Assistant action buttons row
            cols = st.columns([1, 1, 14])
            with cols[0]:
                if st.button("\u200b", icon=":material/content_copy:", key=f"copy_{idx}", type="tertiary", help="Copy response"):
                    st.toast("Response ready to copy! (Select and Ctrl+C)")
            with cols[1]:
                if st.button("\u200b", icon=":material/refresh:", key=f"regen_{idx}", type="tertiary", help="Regenerate response"):
                    # Get the preceding user message
                    last_user_msg = st.session_state["chat_history"][idx-1]["content"]
                    # Keep history up to (but not including) the user message
                    st.session_state["chat_history"] = st.session_state["chat_history"][:idx-1]
                    st.session_state["regenerate_query"] = last_user_msg
                    st.rerun()

# ── Suggested Questions ────────────────────────────────────────────
suggested_query = None
if True:
    st.markdown("<div style='margin-top: 2rem; margin-bottom: 1rem; color: var(--ci-text-muted); font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;'>Suggested Questions</div>", unsafe_allow_html=True)
    if source_filter is None and len(contracts) > 1:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⚖️ Compare termination clauses", use_container_width=True):
                suggested_query = "Compare the termination clauses between all ingested contracts and list the differences."
            if st.button("💰 Compare payment terms", use_container_width=True):
                suggested_query = "What are the differences in payment terms across these contracts?"
        with col2:
            if st.button("🔍 Analyze common liabilities", use_container_width=True):
                suggested_query = "What are the common liabilities and indemnification terms across these documents?"
            if st.button("⚖️ Which governing law?", use_container_width=True):
                suggested_query = "Compare the governing law and jurisdiction of these contracts."
    else:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📝 Summarize payment terms", use_container_width=True):
                suggested_query = "Summarize the payment terms in this contract."
            if st.button("⚖️ What is the governing law?", use_container_width=True):
                suggested_query = "What is the governing law and jurisdiction of this contract?"
        with col2:
            if st.button("⏳ List termination conditions", use_container_width=True):
                suggested_query = "What are the termination conditions in this contract?"
            if st.button("🚩 Highlight key risks", use_container_width=True):
                suggested_query = "Highlight any key risks, strict obligations, or red flags in this contract."

# ── Query input ────────────────────────────────────────────────────
user_input = st.chat_input("Ask a question about the contract...")
regen_query = st.session_state.pop("regenerate_query", None)
query = suggested_query or regen_query or user_input

if query:
    # Show user message
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state["chat_history"].append({"role": "user", "content": query})

    with st.chat_message("assistant"):
        with st.spinner("Searching clauses..."):

            # ── Retrieve ───────────────────────────────────────────
            try:
                chunks = []
                if source_filter and isinstance(source_filter, list):
                    # Fetch top_k chunks per selected document to ensure balanced context for comparison
                    for doc in source_filter:
                        doc_chunks = retrieve_for_contract(
                            query, doc,
                            top_k=top_k, collection=collection,
                        )
                        chunks.extend(doc_chunks)
                    
                    # Sort combined chunks by similarity score (descending)
                    chunks.sort(key=lambda x: x.similarity_score, reverse=True)
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
                    score = chunk.similarity_score
                    if score >= 0.8:
                        badge_color = "#10B981"
                        badge_text = "High relevance"
                        badge_icon = "🟢"
                    elif score >= 0.6:
                        badge_color = "#F59E0B"
                        badge_text = "Medium relevance"
                        badge_icon = "🟡"
                    else:
                        badge_color = "#EF4444"
                        badge_text = "Low relevance"
                        badge_icon = "🔴"

                    risk_badge = (
                        f' · <span style="'
                        f'color:{badge_color};'
                        f'font-family:Inter,sans-serif;'
                        f'font-weight:600;'
                        f'font-size:0.82rem;'
                        f'">{badge_icon} {badge_text}</span>'
                    )

                st.markdown(
                    f"**{chunk.citation}**{risk_badge}",
                    unsafe_allow_html=True,
                )

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

        # ── 4. Save Chat Session ───────────────────────────────
        title = "New Chat"
        if len(st.session_state["chat_history"]) >= 2:
            first_q = st.session_state["chat_history"][0]["content"]
            title = (first_q[:30] + '...') if len(first_q) > 30 else first_q
            
        new_id = save_chat_session(
            conn, 
            st.session_state.get("chat_session_id", ""), 
            title, 
            st.session_state["chat_history"]
        )
        st.session_state["chat_session_id"] = new_id

# ── Clear chat button ─────────────────────────────────────────────
if st.session_state["chat_history"]:
    if st.button("🧹 Clear screen (keep history)"):
        st.session_state["chat_history"] = []
        if "chat_session_id" in st.session_state:
            del st.session_state["chat_session_id"]
        st.rerun()




