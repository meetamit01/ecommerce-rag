"""
chatbot.py — E-Commerce RAG Conversational Assistant (Terminal Mode)
=====================================================================
A terminal-based RAG chatbot that:
  - Retrieves top-k relevant chunks from Pinecone per user query
  - Injects retrieved context + conversation history into the LLM prompt
  - Answers strictly from retrieved content (refuses to hallucinate)
  - Supports multi-turn follow-up questions via conversation history
  - Cites source documents for each answer

Usage:
    python chatbot.py
    python chatbot.py --top-k 5 --max-history 10
"""

import os
import sys
import logging
import argparse
from typing import List, Dict, Any, Tuple, Optional

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document, HumanMessage, AIMessage, SystemMessage
from langchain.schema.messages import BaseMessage

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,  # quieter in terminal mode
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ── Configuration ─────────────────────────────────────────────────────────────
DEFAULT_TOP_K = 5
DEFAULT_MAX_HISTORY = 10  # number of past turns to keep
DEFAULT_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

REFUSAL_PHRASE = "I don't have enough information in the provided documents."

SYSTEM_PROMPT = """You are a knowledgeable e-commerce assistant that helps customers with:
- Product information and manuals
- Return and exchange policies
- Shipping policies and timelines
- Warranty and protection plan details

STRICT RULES you MUST follow:
1. Answer ONLY based on the retrieved context provided below. Do NOT use any external knowledge.
2. If the retrieved context does not contain enough information to answer the question, respond EXACTLY with:
   "I don't have enough information in the provided documents."
3. Always cite the source document(s) at the end of your answer using the format:
   📄 Source(s): [document name(s)]
4. Be concise, accurate, and helpful.
5. For follow-up questions, use the conversation history to understand context, but still answer only from the retrieved documents.
6. Do NOT make up policies, prices, or product specifications not present in the documents."""


# ── Environment Loading ───────────────────────────────────────────────────────
def load_environment() -> Dict[str, str]:
    """Load and validate environment variables."""
    load_dotenv()

    config = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
        "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY", ""),
        "PINECONE_INDEX_NAME": os.getenv("PINECONE_INDEX_NAME", "ecommerce-rag"),
        "OPENAI_CHAT_MODEL": os.getenv("OPENAI_CHAT_MODEL", DEFAULT_CHAT_MODEL),
        "OPENAI_EMBEDDING_MODEL": os.getenv(
            "OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
        ),
    }

    missing = [k for k, v in config.items() if not v and k != "OPENAI_CHAT_MODEL"]
    if missing:
        print(f"\n❌ Missing environment variables: {', '.join(missing)}")
        print("   Copy .env.example to .env and fill in your API keys.\n")
        sys.exit(1)

    return config


# ── Vector Store Retriever ────────────────────────────────────────────────────
def build_retriever(env: Dict[str, str], top_k: int):
    """Initialize the Pinecone-backed retriever."""
    embeddings = OpenAIEmbeddings(
        model=env["OPENAI_EMBEDDING_MODEL"],
        openai_api_key=env["OPENAI_API_KEY"],
    )

    vector_store = PineconeVectorStore(
        index_name=env["PINECONE_INDEX_NAME"],
        embedding=embeddings,
        pinecone_api_key=env["PINECONE_API_KEY"],
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )

    return retriever


# ── Context Formatting ────────────────────────────────────────────────────────
def format_context(docs: List[Document]) -> Tuple[str, List[str]]:
    """
    Format retrieved documents into a structured context block for the LLM.
    Returns the formatted context string and a list of unique source names.
    """
    if not docs:
        return "", []

    context_parts = []
    sources = []

    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        content = doc.page_content.strip()

        context_parts.append(
            f"[Document {i} | Source: {source}]\n{content}"
        )

        if source not in sources:
            sources.append(source)

    return "\n\n---\n\n".join(context_parts), sources


# ── RAG Chain ─────────────────────────────────────────────────────────────────
class RAGChatbot:
    """
    RAG-powered conversational assistant with conversation history support.
    """

    def __init__(self, env: Dict[str, str], top_k: int, max_history: int):
        self.env = env
        self.top_k = top_k
        self.max_history = max_history
        self.conversation_history: List[Dict[str, str]] = []

        print("🔌 Connecting to Pinecone vector store...")
        self.retriever = build_retriever(env, top_k)

        print(f"🤖 Loading LLM: {env['OPENAI_CHAT_MODEL']}")
        self.llm = ChatOpenAI(
            model=env["OPENAI_CHAT_MODEL"],
            temperature=0.0,  # deterministic — no hallucination
            openai_api_key=env["OPENAI_API_KEY"],
        )

    def _build_messages(
        self, query: str, context: str
    ) -> List[BaseMessage]:
        """
        Construct the full message list for the LLM:
        [SystemMessage] + [ConversationHistory] + [CurrentUserMessage with context]
        """
        messages: List[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]

        # Add previous conversation turns (trimmed to max_history)
        recent_history = self.conversation_history[-self.max_history:]
        for turn in recent_history:
            if turn["role"] == "user":
                messages.append(HumanMessage(content=turn["content"]))
            else:
                messages.append(AIMessage(content=turn["content"]))

        # Current user query with injected context
        user_message = (
            f"RETRIEVED CONTEXT FROM DOCUMENTS:\n"
            f"{'=' * 50}\n"
            f"{context}\n"
            f"{'=' * 50}\n\n"
            f"USER QUESTION: {query}"
        )
        messages.append(HumanMessage(content=user_message))

        return messages

    def chat(self, query: str) -> Dict[str, Any]:
        """
        Process a user query through the full RAG pipeline:
        1. Retrieve relevant chunks
        2. Format context
        3. Call LLM with context + history
        4. Update conversation history
        5. Return answer with sources
        """
        # Step 1: Retrieve
        docs = self.retriever.invoke(query)

        if not docs:
            answer = REFUSAL_PHRASE
            return {
                "answer": answer,
                "sources": [],
                "num_docs_retrieved": 0,
            }

        # Step 2: Format context
        context, sources = format_context(docs)

        # Step 3: Build messages and call LLM
        messages = self._build_messages(query, context)
        response = self.llm.invoke(messages)
        answer = response.content.strip()

        # Step 4: Update history (store only the user query, not the injected context)
        self.conversation_history.append({"role": "user", "content": query})
        self.conversation_history.append({"role": "assistant", "content": answer})

        return {
            "answer": answer,
            "sources": sources,
            "num_docs_retrieved": len(docs),
            "retrieved_docs": docs,
        }

    def clear_history(self) -> None:
        """Reset conversation history for a fresh session."""
        self.conversation_history = []
        print("\n🗑️  Conversation history cleared.\n")


# ── Terminal UI ───────────────────────────────────────────────────────────────
BANNER = """
╔══════════════════════════════════════════════════════════════╗
║        🛒  E-Commerce RAG Assistant (Terminal Mode)          ║
║     Ask about return policies, shipping, product manuals     ║
╠══════════════════════════════════════════════════════════════╣
║  Commands:  'quit' or 'exit' → Exit                          ║
║             'clear' → Clear conversation history             ║
║             'history' → Show conversation history            ║
║             'docs' → Show retrieved documents (last query)   ║
╚══════════════════════════════════════════════════════════════╝
"""


def print_separator(char: str = "─", width: int = 65) -> None:
    print(char * width)


def run_terminal_chat(bot: RAGChatbot) -> None:
    """Run the interactive terminal chat loop."""
    print(BANNER)
    last_result: Optional[Dict[str, Any]] = None

    while True:
        try:
            user_input = input("\n👤 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Goodbye!\n")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        # ── Special Commands ──────────────────────────────────────────────────
        if cmd in ("quit", "exit"):
            print("\n👋 Goodbye!\n")
            break

        elif cmd == "clear":
            bot.clear_history()
            last_result = None
            continue

        elif cmd == "history":
            if not bot.conversation_history:
                print("\n📋 No conversation history yet.\n")
            else:
                print("\n📋 Conversation History:")
                print_separator()
                for i, turn in enumerate(bot.conversation_history):
                    role = "👤 You" if turn["role"] == "user" else "🤖 Bot"
                    print(f"{role}: {turn['content'][:200]}...")
                    if i < len(bot.conversation_history) - 1:
                        print()
                print_separator()
            continue

        elif cmd == "docs":
            if not last_result or not last_result.get("retrieved_docs"):
                print("\n📄 No retrieved documents from last query.\n")
            else:
                print("\n📄 Last Retrieved Documents:")
                print_separator()
                for i, doc in enumerate(last_result["retrieved_docs"], 1):
                    src = doc.metadata.get("source", "unknown")
                    preview = doc.page_content[:200].replace("\n", " ")
                    print(f"[{i}] {src}\n    {preview}...")
                    print()
                print_separator()
            continue

        # ── RAG Query ─────────────────────────────────────────────────────────
        print("\n🔍 Retrieving relevant information...")

        result = bot.chat(user_input)
        last_result = result

        print_separator()
        print(f"\n🤖 Assistant:\n\n{result['answer']}\n")

        if result.get("sources"):
            print(f"📊 Retrieved {result['num_docs_retrieved']} chunk(s) "
                  f"from {len(result['sources'])} source(s)")

        print_separator()


# ── CLI Args ──────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="E-Commerce RAG Chatbot — Terminal Mode"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of chunks to retrieve per query (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--max-history",
        type=int,
        default=DEFAULT_MAX_HISTORY,
        help=f"Max conversation turns to retain (default: {DEFAULT_MAX_HISTORY})",
    )
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    env = load_environment()

    print(f"\n⚙️  Config: top_k={args.top_k}, max_history={args.max_history}")
    print(f"   Model: {env['OPENAI_CHAT_MODEL']} | Index: {env['PINECONE_INDEX_NAME']}")

    bot = RAGChatbot(env=env, top_k=args.top_k, max_history=args.max_history)
    print("✅ Ready!\n")

    run_terminal_chat(bot)


if __name__ == "__main__":
    main()
