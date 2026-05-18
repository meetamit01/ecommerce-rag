"""
app.py — E-Commerce RAG Assistant (Streamlit UI)
================================================
Streamlit-based web interface for the RAG chatbot.
Provides a polished chat UI with:
  - Conversation history display
  - Source document citations
  - Expandable "Retrieved Documents" panel
  - Feedback mechanism (thumbs up/down)
  - Session controls (clear history, view stats)

Usage:
    streamlit run app.py
"""

import os
import sys
import logging
from typing import List, Dict, Any, Optional

import streamlit as st
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))
from chatbot import RAGChatbot, load_environment, REFUSAL_PHRASE

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ShopAssist AI — E-Commerce RAG",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Import Fonts */
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

    /* ── Global Theme ── */
    :root {
        --primary: #1a1a2e;
        --accent: #e94560;
        --accent-soft: #ff6b6b;
        --surface: #16213e;
        --surface-2: #0f3460;
        --text-primary: #eaeaea;
        --text-secondary: #a0aec0;
        --success: #48bb78;
        --warning: #ed8936;
        --border: rgba(255,255,255,0.08);
        --gradient: linear-gradient(135deg, #e94560 0%, #0f3460 100%);
    }

    /* ── Base ── */
    html, body, .stApp {
        background-color: var(--primary);
        color: var(--text-primary);
        font-family: 'DM Sans', sans-serif;
    }

    /* ── Hide Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }

    /* ── Custom scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: var(--primary); }
    ::-webkit-scrollbar-thumb { background: var(--surface-2); border-radius: 3px; }

    /* ── Header ── */
    .shop-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #0f3460 50%, #1a1a2e 100%);
        border-bottom: 1px solid var(--border);
        padding: 1.5rem 2rem;
        margin: -1rem -1rem 1.5rem -1rem;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    .shop-header h1 {
        font-family: 'DM Serif Display', serif;
        font-size: 1.8rem;
        background: linear-gradient(90deg, #e94560, #ff6b6b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .shop-header p {
        color: var(--text-secondary);
        margin: 0;
        font-size: 0.85rem;
    }

    /* ── Chat messages ── */
    .user-bubble {
        background: linear-gradient(135deg, #e94560 0%, #c13547 100%);
        color: white;
        padding: 1rem 1.2rem;
        border-radius: 18px 18px 4px 18px;
        margin: 0.5rem 0;
        max-width: 80%;
        margin-left: auto;
        font-size: 0.95rem;
        line-height: 1.5;
        box-shadow: 0 4px 15px rgba(233, 69, 96, 0.3);
    }
    .bot-bubble {
        background: var(--surface);
        border: 1px solid var(--border);
        color: var(--text-primary);
        padding: 1rem 1.2rem;
        border-radius: 18px 18px 18px 4px;
        margin: 0.5rem 0;
        max-width: 85%;
        font-size: 0.95rem;
        line-height: 1.6;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    .bot-bubble.refusal {
        border-left: 3px solid var(--warning);
        background: rgba(237, 137, 54, 0.1);
    }

    /* ── Source citation badge ── */
    .source-badge {
        display: inline-block;
        background: rgba(233, 69, 96, 0.15);
        border: 1px solid rgba(233, 69, 96, 0.4);
        color: #ff6b6b;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 500;
        margin: 0.2rem 0.2rem 0 0;
    }

    /* ── Input area ── */
    .stTextInput > div > div > input {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
        font-family: 'DM Sans', sans-serif !important;
        padding: 0.8rem 1rem !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 2px rgba(233, 69, 96, 0.2) !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: var(--gradient) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(233, 69, 96, 0.4) !important;
    }

    /* ── Sidebar ── */
    .css-1d391kg, [data-testid="stSidebar"] {
        background-color: var(--surface) !important;
        border-right: 1px solid var(--border) !important;
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        background: var(--surface-2) !important;
        border-radius: 8px !important;
        color: var(--text-secondary) !important;
        font-size: 0.85rem !important;
    }

    /* ── Metrics ── */
    [data-testid="metric-container"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 0.5rem;
    }

    /* ── Suggestion pills ── */
    .suggestion-pill {
        display: inline-block;
        background: var(--surface-2);
        border: 1px solid var(--border);
        color: var(--text-secondary);
        padding: 0.4rem 0.9rem;
        border-radius: 20px;
        font-size: 0.82rem;
        margin: 0.25rem;
        cursor: pointer;
        transition: all 0.2s;
    }

    /* ── Feedback ── */
    .feedback-row {
        display: flex;
        gap: 0.5rem;
        margin-top: 0.5rem;
        align-items: center;
    }
    .feedback-label {
        color: var(--text-secondary);
        font-size: 0.78rem;
    }

    /* ── Status dot ── */
    .status-dot {
        width: 8px;
        height: 8px;
        background: var(--success);
        border-radius: 50%;
        display: inline-block;
        margin-right: 6px;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }

    /* ── Divider ── */
    hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)


# ── Session State Initialization ──────────────────────────────────────────────
def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []  # chat display history
    if "bot" not in st.session_state:
        st.session_state.bot = None
    if "error" not in st.session_state:
        st.session_state.error = None
    if "stats" not in st.session_state:
        st.session_state.stats = {"queries": 0, "answered": 0, "refused": 0}
    if "feedback" not in st.session_state:
        st.session_state.feedback = {}  # {msg_index: "up"/"down"}
    if "top_k" not in st.session_state:
        st.session_state.top_k = 5
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None


# ── Bot Initialization ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_bot(top_k: int) -> Optional[RAGChatbot]:
    """Initialize (and cache) the RAG chatbot."""
    try:
        env = load_environment()
        bot = RAGChatbot(env=env, top_k=top_k, max_history=10)
        return bot
    except SystemExit:
        return None
    except Exception as e:
        logger.error(f"Bot initialization failed: {e}")
        return None


# ── Suggestion Queries ─────────────────────────────────────────────────────────
SUGGESTIONS = [
    "What is Amazon's return policy?",
    "Can I return electronics at Best Buy?",
    "How long does standard shipping take?",
    "What items are non-returnable at Walmart?",
    "Does eBay cover stolen items?",
    "How do I reset my wireless earbuds?",
    "What is the warranty for the smartwatch?",
    "Is there a restocking fee for returns?",
]


# ── Helpers ────────────────────────────────────────────────────────────────────
def format_source_name(filename: str) -> str:
    """Convert filename to human-readable source label."""
    name_map = {
        "amazon_return_policy.txt": "Amazon Return Policy",
        "ebay_buyer_protection_policy.txt": "eBay Buyer Protection",
        "walmart_return_policy.txt": "Walmart Return Policy",
        "bestbuy_return_policy.txt": "Best Buy Return Policy",
        "ecommerce_shipping_policy.txt": "Shipping Policy",
        "electronics_product_manual.txt": "Electronics Manual",
    }
    return name_map.get(filename, filename.replace("_", " ").replace(".txt", "").title())


def render_message(msg: Dict, index: int):
    """Render a single chat message with metadata."""
    role = msg["role"]
    content = msg["content"]

    if role == "user":
        st.markdown(
            f'<div class="user-bubble">👤 {content}</div>',
            unsafe_allow_html=True,
        )
    else:
        is_refusal = REFUSAL_PHRASE in content
        bubble_class = "bot-bubble refusal" if is_refusal else "bot-bubble"

        # Render answer
        st.markdown(
            f'<div class="{bubble_class}">🤖 {content}</div>',
            unsafe_allow_html=True,
        )

        # Source badges
        if msg.get("sources"):
            badges = " ".join(
                f'<span class="source-badge">📄 {format_source_name(s)}</span>'
                for s in msg["sources"]
            )
            st.markdown(badges, unsafe_allow_html=True)

        # Retrieved docs expander
        if msg.get("retrieved_docs"):
            with st.expander(
                f"🔍 View {len(msg['retrieved_docs'])} retrieved chunk(s)", expanded=False
            ):
                for i, doc in enumerate(msg["retrieved_docs"], 1):
                    src = format_source_name(doc.metadata.get("source", "unknown"))
                    st.markdown(f"**Chunk {i} — {src}**")
                    st.text(doc.page_content[:400] + ("..." if len(doc.page_content) > 400 else ""))
                    if i < len(msg["retrieved_docs"]):
                        st.divider()

        # Feedback row
        col1, col2, col3 = st.columns([1, 1, 8])
        fb = st.session_state.feedback.get(index)
        with col1:
            if st.button("👍", key=f"up_{index}", help="Helpful"):
                st.session_state.feedback[index] = "up"
        with col2:
            if st.button("👎", key=f"down_{index}", help="Not helpful"):
                st.session_state.feedback[index] = "down"
        if fb:
            with col3:
                label = "✅ Thanks!" if fb == "up" else "🔧 Noted, will improve!"
                st.caption(label)


# ── Sidebar ────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown(
            '<span class="status-dot"></span><strong style="color:#eaeaea">System Online</strong>',
            unsafe_allow_html=True,
        )
        st.divider()

        # ── Stats ────────────────────────────────────────────────────────────
        st.markdown("### 📊 Session Stats")
        s = st.session_state.stats
        c1, c2 = st.columns(2)
        c1.metric("Queries", s["queries"])
        c2.metric("Answered", s["answered"])
        c1.metric("Refused", s["refused"])
        satisfaction = (
            f"{round(s['answered'] / s['queries'] * 100)}%"
            if s["queries"] > 0 else "—"
        )
        c2.metric("Hit Rate", satisfaction)

        st.divider()

        # ── Config ───────────────────────────────────────────────────────────
        st.markdown("### ⚙️ Settings")
        top_k = st.slider(
            "Retrieved chunks (top-k)",
            min_value=2,
            max_value=10,
            value=st.session_state.top_k,
            help="More chunks = more context but slower response",
        )
        if top_k != st.session_state.top_k:
            st.session_state.top_k = top_k
            st.session_state.bot = None  # force re-init
            st.rerun()

        st.divider()

        # ── Documents ────────────────────────────────────────────────────────
        st.markdown("### 📚 Knowledge Base")
        docs = {
            "🛍️ Amazon Return Policy": "amazon_return_policy.txt",
            "🛒 eBay Buyer Protection": "ebay_buyer_protection_policy.txt",
            "🏪 Walmart Returns": "walmart_return_policy.txt",
            "💻 Best Buy Returns": "bestbuy_return_policy.txt",
            "🚚 Shipping Policy": "ecommerce_shipping_policy.txt",
            "📱 Electronics Manual": "electronics_product_manual.txt",
        }
        for label in docs:
            st.markdown(
                f'<div style="color:#a0aec0; font-size:0.82rem; '
                f'padding:0.2rem 0;">✓ {label}</div>',
                unsafe_allow_html=True,
            )

        st.divider()

        # ── Controls ─────────────────────────────────────────────────────────
        st.markdown("### 🎛️ Controls")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.stats = {"queries": 0, "answered": 0, "refused": 0}
            st.session_state.feedback = {}
            if st.session_state.bot:
                st.session_state.bot.clear_history()
            st.rerun()

        # ── Feedback export ──────────────────────────────────────────────────
        if st.session_state.feedback:
            st.markdown("### 💬 Feedback Log")
            up = sum(1 for v in st.session_state.feedback.values() if v == "up")
            down = sum(1 for v in st.session_state.feedback.values() if v == "down")
            st.caption(f"👍 {up} helpful  |  👎 {down} not helpful")


# ── Main App ────────────────────────────────────────────────────────────────────
def main():
    init_session()
    render_sidebar()

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="shop-header">
        <div>
            <h1>🛒 ShopAssist AI</h1>
            <p>E-Commerce RAG Assistant — Powered by GPT-4o-mini + Pinecone</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Bot initialization ─────────────────────────────────────────────────────
    if st.session_state.bot is None:
        with st.spinner("🔌 Connecting to knowledge base..."):
            bot = get_bot(st.session_state.top_k)
            if bot is None:
                st.error(
                    "❌ **Setup Required**\n\n"
                    "Could not connect to the knowledge base.\n\n"
                    "**Steps:**\n"
                    "1. Copy `.env.example` to `.env`\n"
                    "2. Add your `OPENAI_API_KEY` and `PINECONE_API_KEY`\n"
                    "3. Run `python ingest.py` first\n"
                    "4. Restart this app"
                )
                st.stop()
            st.session_state.bot = bot

    bot = st.session_state.bot

    # ── Suggestions (shown when chat is empty) ─────────────────────────────────
    if not st.session_state.messages:
        st.markdown(
            '<p style="color:#a0aec0; font-size:0.9rem; margin-bottom:0.5rem;">'
            '💡 Try asking:</p>',
            unsafe_allow_html=True,
        )
        cols = st.columns(2)
        for i, suggestion in enumerate(SUGGESTIONS[:6]):
            with cols[i % 2]:
                if st.button(suggestion, key=f"sug_{i}", use_container_width=True):
                    st.session_state.pending_query = suggestion
                    st.rerun()

    # ── Chat history ──────────────────────────────────────────────────────────
    chat_container = st.container()
    with chat_container:
        for i, msg in enumerate(st.session_state.messages):
            render_message(msg, i)

    st.divider()

    # ── Input area ────────────────────────────────────────────────────────────
    col1, col2 = st.columns([8, 1])
    with col1:
        user_input = st.text_input(
            "Message",
            key="user_input",
            placeholder="Ask about return policies, shipping, product manuals...",
            label_visibility="collapsed",
        )
    with col2:
        send_clicked = st.button("Send →", use_container_width=True)

    # ── Process query ─────────────────────────────────────────────────────────
    query = None
    if send_clicked and user_input:
        query = user_input
    elif st.session_state.pending_query:
        query = st.session_state.pending_query
        st.session_state.pending_query = None

    if query:
        # Add user message to display
        st.session_state.messages.append({"role": "user", "content": query})
        st.session_state.stats["queries"] += 1

        # RAG pipeline
        with st.spinner("🔍 Searching knowledge base..."):
            result = bot.chat(query)

        answer = result["answer"]
        sources = result.get("sources", [])
        docs = result.get("retrieved_docs", [])

        # Track stats
        if REFUSAL_PHRASE in answer:
            st.session_state.stats["refused"] += 1
        else:
            st.session_state.stats["answered"] += 1

        # Add bot response to display
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "retrieved_docs": docs,
        })

        st.rerun()


if __name__ == "__main__":
    main()
