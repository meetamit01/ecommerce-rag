"""
ingest.py — E-Commerce RAG Document Ingestion Pipeline
=======================================================
Loads documents from the /docs folder, splits them into semantic chunks,
generates OpenAI embeddings, and stores them in a Pinecone vector store.

Usage:
    python ingest.py
    python ingest.py --docs-dir ./docs --chunk-size 800 --chunk-overlap 100

Sources ingested:
    - amazon_return_policy.txt       (Amazon.com Return Policy)
    - ebay_buyer_protection_policy.txt (eBay Buyer Protection & Returns)
    - walmart_return_policy.txt      (Walmart Return & Exchange Policy)
    - bestbuy_return_policy.txt      (Best Buy Return & Exchange Policy)
    - ecommerce_shipping_policy.txt  (Standard E-Commerce Shipping Policy)
    - electronics_product_manual.txt (Consumer Electronics Product Manual)
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document
from pinecone import Pinecone, ServerlessSpec

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Configuration ─────────────────────────────────────────────────────────────
DEFAULT_DOCS_DIR = "./docs"
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
PINECONE_DIMENSION = 1536  # dimension for text-embedding-3-small


# ── Helper: Load environment variables ───────────────────────────────────────
def load_environment() -> Dict[str, str]:
    """Load and validate required environment variables."""
    load_dotenv()

    required_vars = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY"),
        "PINECONE_INDEX_NAME": os.getenv("PINECONE_INDEX_NAME", "ecommerce-rag"),
    }

    missing = [k for k, v in required_vars.items() if not v]
    if missing:
        logger.error(f"Missing required environment variables: {', '.join(missing)}")
        logger.error("Copy .env.example to .env and fill in your API keys.")
        sys.exit(1)

    return {
        **required_vars,
        "OPENAI_EMBEDDING_MODEL": os.getenv(
            "OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
        ),
    }


# ── Step 1: Document Loading ──────────────────────────────────────────────────
def load_documents(docs_dir: str) -> List[Document]:
    """
    Load all .txt and .pdf documents from the specified directory.
    Attaches source metadata (filename, file path) to each document.
    """
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        logger.error(f"Documents directory not found: {docs_dir}")
        sys.exit(1)

    logger.info(f"Loading documents from: {docs_path.resolve()}")

    loader = DirectoryLoader(
        path=str(docs_path),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )

    documents = loader.load()

    if not documents:
        logger.error("No documents found in the docs directory.")
        sys.exit(1)

    # Enrich metadata with cleaner source names
    for doc in documents:
        raw_path = doc.metadata.get("source", "unknown")
        doc.metadata["source"] = Path(raw_path).name
        doc.metadata["file_path"] = raw_path

    logger.info(f"Loaded {len(documents)} document(s):")
    for doc in documents:
        word_count = len(doc.page_content.split())
        logger.info(f"  • {doc.metadata['source']} ({word_count:,} words)")

    return documents


# ── Step 2: Text Splitting ─────────────────────────────────────────────────────
def split_documents(
    documents: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Document]:
    """
    Split documents into semantic chunks using RecursiveCharacterTextSplitter.
    The splitter respects paragraph and sentence boundaries for better coherence.
    """
    logger.info(
        f"Splitting documents → chunk_size={chunk_size}, overlap={chunk_overlap}"
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        length_function=len,
        add_start_index=True,  # track character offset within original doc
    )

    chunks = splitter.split_documents(documents)

    logger.info(f"Created {len(chunks)} chunks across {len(documents)} documents")

    # Per-document chunk summary
    source_counts: Dict[str, int] = {}
    for chunk in chunks:
        src = chunk.metadata.get("source", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1

    for src, count in source_counts.items():
        logger.info(f"  • {src}: {count} chunks")

    return chunks


# ── Step 3: Pinecone Index Setup ──────────────────────────────────────────────
def setup_pinecone_index(api_key: str, index_name: str) -> None:
    """
    Create the Pinecone index if it does not already exist.
    Uses serverless spec on AWS us-east-1 (free tier compatible).
    """
    pc = Pinecone(api_key=api_key)
    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if index_name in existing_indexes:
        logger.info(f"Pinecone index '{index_name}' already exists — reusing it.")
    else:
        logger.info(f"Creating Pinecone index '{index_name}' (dim={PINECONE_DIMENSION})...")
        pc.create_index(
            name=index_name,
            dimension=PINECONE_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        # Wait for index to become ready
        logger.info("Waiting for index to become ready...")
        for _ in range(30):
            status = pc.describe_index(index_name).status
            if status.get("ready"):
                break
            time.sleep(2)
        logger.info(f"Index '{index_name}' is ready.")


# ── Step 4: Embedding & Indexing ──────────────────────────────────────────────
def embed_and_index(
    chunks: List[Document],
    env: Dict[str, str],
    batch_size: int = 100,
) -> PineconeVectorStore:
    """
    Generate embeddings using OpenAI and upsert vectors into Pinecone.
    Documents are processed in batches to respect API rate limits.
    """
    logger.info(
        f"Generating embeddings with model: {env['OPENAI_EMBEDDING_MODEL']}"
    )

    embeddings = OpenAIEmbeddings(
        model=env["OPENAI_EMBEDDING_MODEL"],
        openai_api_key=env["OPENAI_API_KEY"],
    )

    # Ensure Pinecone index exists
    setup_pinecone_index(env["PINECONE_API_KEY"], env["PINECONE_INDEX_NAME"])

    logger.info(
        f"Upserting {len(chunks)} chunks to Pinecone index "
        f"'{env['PINECONE_INDEX_NAME']}' in batches of {batch_size}..."
    )

    vector_store = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=env["PINECONE_INDEX_NAME"],
        pinecone_api_key=env["PINECONE_API_KEY"],
    )

    logger.info("All chunks embedded and indexed successfully.")
    return vector_store


# ── Step 5: Verification ──────────────────────────────────────────────────────
def verify_index(env: Dict[str, str]) -> None:
    """Run a quick sanity-check query against the populated index."""
    logger.info("Running verification query: 'return policy for electronics'")

    embeddings = OpenAIEmbeddings(
        model=env["OPENAI_EMBEDDING_MODEL"],
        openai_api_key=env["OPENAI_API_KEY"],
    )

    vector_store = PineconeVectorStore(
        index_name=env["PINECONE_INDEX_NAME"],
        embedding=embeddings,
        pinecone_api_key=env["PINECONE_API_KEY"],
    )

    results = vector_store.similarity_search(
        "What is the return policy for electronics?", k=3
    )

    logger.info(f"Verification returned {len(results)} result(s):")
    for i, doc in enumerate(results, 1):
        preview = doc.page_content[:120].replace("\n", " ")
        source = doc.metadata.get("source", "unknown")
        logger.info(f"  [{i}] [{source}] {preview}...")


# ── CLI Argument Parsing ──────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest e-commerce documents into Pinecone vector store"
    )
    parser.add_argument(
        "--docs-dir",
        default=DEFAULT_DOCS_DIR,
        help=f"Path to the documents directory (default: {DEFAULT_DOCS_DIR})",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Characters per chunk (default: {DEFAULT_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help=f"Overlap characters between chunks (default: {DEFAULT_CHUNK_OVERLAP})",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip the post-ingestion verification query",
    )
    return parser.parse_args()


# ── Main Entrypoint ───────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    logger.info("=" * 60)
    logger.info("E-Commerce RAG — Document Ingestion Pipeline")
    logger.info("=" * 60)

    # Load env
    env = load_environment()

    # Pipeline
    documents = load_documents(args.docs_dir)
    chunks = split_documents(documents, args.chunk_size, args.chunk_overlap)
    embed_and_index(chunks, env)

    if not args.skip_verify:
        verify_index(env)

    logger.info("=" * 60)
    logger.info("Ingestion complete! Run the chatbot with:")
    logger.info("  streamlit run app.py   (Streamlit UI)")
    logger.info("  python chatbot.py      (Terminal mode)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
