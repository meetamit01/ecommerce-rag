# 🛒 E-Commerce RAG Conversational Assistant

A production-quality **Retrieval-Augmented Generation (RAG)** chatbot for the e-commerce domain. The assistant answers user questions strictly from a curated set of real, publicly available e-commerce documents — covering return policies, shipping guidelines, and product manuals. It supports multi-turn conversations, cites source documents, and refuses to answer when information is not found in the knowledge base.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Architecture](#-architecture)
- [Knowledge Base Documents](#-knowledge-base-documents)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Setup & Installation](#-setup--installation)
- [Running the Application](#-running-the-application)
- [How It Works](#-how-it-works)
- [Example Interactions](#-example-interactions)
- [Design Decisions](#-design-decisions)
- [Troubleshooting](#-troubleshooting)

---

## 🎯 Project Overview

| Feature | Details |
|---|---|
| **Domain** | E-Commerce (Return Policies, Shipping, Product Manuals) |
| **LLM** | OpenAI `gpt-4o-mini` |
| **Embeddings** | OpenAI `text-embedding-3-small` (1536-dim) |
| **Vector Store** | Pinecone (Serverless) |
| **Framework** | LangChain |
| **UI** | Streamlit (bonus) + Terminal mode |
| **Context Window** | Configurable top-k retrieval (default: 5 chunks) |
| **History** | Last 10 conversation turns retained per session |

### Key Capabilities
- ✅ Answers only from provided documents (no hallucination)
- ✅ Cites source document(s) for every answer
- ✅ Supports multi-turn follow-up questions
- ✅ Graceful refusal: *"I don't have enough information in the provided documents."*
- ✅ Modular, production-grade Python code
- ✅ Streamlit UI with feedback mechanism
- ✅ Configurable retrieval parameters

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INGESTION PIPELINE                        │
│                          (ingest.py)                             │
│                                                                  │
│  📄 docs/*.txt  →  TextLoader  →  RecursiveTextSplitter          │
│                                         ↓                        │
│                              OpenAI Embeddings                   │
│                           (text-embedding-3-small)               │
│                                         ↓                        │
│                           Pinecone Vector Store                  │
│                          (cosine similarity index)               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     QUERY / CHAT PIPELINE                        │
│                  (chatbot.py / app.py)                           │
│                                                                  │
│  User Query                                                      │
│      ↓                                                           │
│  Embed query  (OpenAI text-embedding-3-small)                    │
│      ↓                                                           │
│  Retrieve top-k chunks  (Pinecone similarity search)             │
│      ↓                                                           │
│  Build Prompt:                                                   │
│    [System Prompt]                                               │
│    + [Conversation History (last 10 turns)]                      │
│    + [Retrieved Context chunks]                                  │
│    + [User Query]                                                │
│      ↓                                                           │
│  ChatOpenAI (gpt-4o-mini, temp=0.0)                              │
│      ↓                                                           │
│  Answer + Source Citations → User                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📚 Knowledge Base Documents

All documents are publicly available and stored in the `docs/` directory.

| File | Source | Description |
|---|---|---|
| `amazon_return_policy.txt` | [Amazon Help Center](https://www.amazon.com/gp/help/customer/display.html?nodeId=GKM69DUUYKQWKWX7) | Amazon's complete return & refund policy including Prime, third-party sellers, A-to-Z Guarantee |
| `ebay_buyer_protection_policy.txt` | [eBay Help Center](https://www.ebay.com/help/buying/returns-refunds/returns-refunds?id=4008) | eBay Money Back Guarantee, return windows, Authenticity Guarantee |
| `walmart_return_policy.txt` | [Walmart Help](https://www.walmart.com/help/article/walmart-s-return-policy/) | Walmart's 90-day return policy, Marketplace Guarantee, Walmart+ benefits |
| `bestbuy_return_policy.txt` | [Best Buy Help](https://www.bestbuy.com/site/help-topics/return-exchange-policy/) | Best Buy 15/30/60-day windows, Geek Squad Protection, holiday policy |
| `ecommerce_shipping_policy.txt` | [Shopify Blog](https://www.shopify.com/blog/shipping-policy) | Standard e-commerce shipping methods, timelines, costs, lost/damaged packages |
| `electronics_product_manual.txt` | Generic consumer electronics manual (public domain template) | Wireless earbuds, smartwatch, smart speaker setup, troubleshooting, and warranty |

> **Note on Data Sources**: All documents are based on publicly available e-commerce policies and product manuals. The content was compiled from official help center pages and represents real-world e-commerce policy information. No proprietary or confidential data is used.

---

## 📁 Project Structure

```
ecommerce-rag/
│
├── docs/                              # Knowledge base documents
│   ├── amazon_return_policy.txt
│   ├── ebay_buyer_protection_policy.txt
│   ├── walmart_return_policy.txt
│   ├── bestbuy_return_policy.txt
│   ├── ecommerce_shipping_policy.txt
│   └── electronics_product_manual.txt
│
├── logs/
│   └── sample_conversation.log        # Sample conversation transcript
│
├── ingest.py                          # Document ingestion & Pinecone indexing
├── chatbot.py                         # RAG chatbot (terminal mode)
├── app.py                             # Streamlit web UI (bonus)
│
├── requirements.txt                   # Python dependencies
├── .env.example                       # Environment variable template
├── .gitignore                         # Git ignore rules
└── README.md                          # This file
```

---

## 🔧 Prerequisites

- **Python 3.9+**
- **OpenAI API Key** — [Get one here](https://platform.openai.com/api-keys)
- **Pinecone API Key** — [Get one here](https://www.pinecone.io/) (free tier available)

---

## ⚙️ Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourname/ecommerce-rag.git
cd ecommerce-rag
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` with your actual API keys:

```env
# Required
OPENAI_API_KEY=sk-...your-openai-key...
PINECONE_API_KEY=...your-pinecone-key...
PINECONE_INDEX_NAME=ecommerce-rag

# Optional (defaults shown)
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-4o-mini
```

> ⚠️ **Never commit your `.env` file to version control.** It is listed in `.gitignore`.

### 5. Create Pinecone Index & Ingest Documents

```bash
python ingest.py
```

**What this does:**
1. Loads all 6 documents from the `docs/` directory
2. Splits them into ~800-character chunks with 150-character overlap
3. Generates OpenAI embeddings for each chunk
4. Creates a Pinecone serverless index (if it doesn't exist)
5. Upserts all vectors to Pinecone
6. Runs a verification query to confirm everything works

**Expected output:**
```
2024-01-01 10:00:00 [INFO] Loading documents from: ./docs
2024-01-01 10:00:01 [INFO] Loaded 6 document(s)
2024-01-01 10:00:01 [INFO]   • amazon_return_policy.txt (1,234 words)
2024-01-01 10:00:01 [INFO]   • ebay_buyer_protection_policy.txt (987 words)
...
2024-01-01 10:00:02 [INFO] Created 89 chunks across 6 documents
2024-01-01 10:00:15 [INFO] All chunks embedded and indexed successfully.
2024-01-01 10:00:16 [INFO] Verification returned 3 result(s)
2024-01-01 10:00:16 [INFO] Ingestion complete!
```

---

## 🚀 Running the Application

### Option A: Streamlit Web UI (Recommended)

```bash
streamlit run app.py
```

Opens at `http://localhost:8501` with:
- Chat interface with conversation history
- Source document citations with badges
- Expandable "Retrieved Documents" panel
- Thumbs up/down feedback mechanism
- Session statistics dashboard
- Configurable top-k slider

### Option B: Terminal Mode

```bash
python chatbot.py
```

With custom parameters:
```bash
python chatbot.py --top-k 7 --max-history 20
```

**Terminal commands:**
| Command | Action |
|---|---|
| `clear` | Reset conversation history |
| `history` | Show past conversation turns |
| `docs` | Show retrieved documents from last query |
| `quit` / `exit` | Exit the chatbot |

---

## 🔬 How It Works

### Document Ingestion (`ingest.py`)

```python
# 1. Load documents
loader = DirectoryLoader(docs_dir, glob="**/*.txt", loader_cls=TextLoader)
documents = loader.load()

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " "]
)
chunks = splitter.split_documents(documents)

# 3. Embed and index
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = PineconeVectorStore.from_documents(chunks, embeddings, index_name=...)
```

### RAG Query Pipeline (`chatbot.py`)

```python
def chat(self, query: str) -> dict:
    # 1. Retrieve top-k relevant chunks
    docs = self.retriever.invoke(query)

    # 2. Format context with source attribution
    context = format_context(docs)

    # 3. Build messages: System + History + Context + Query
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        *conversation_history,
        HumanMessage(content=f"CONTEXT:\n{context}\n\nQUESTION: {query}")
    ]

    # 4. Generate answer (temp=0 for determinism)
    response = self.llm.invoke(messages)

    # 5. Update history and return
    return {"answer": response.content, "sources": sources}
```

### Hallucination Prevention

The system prompt explicitly instructs the model to:
1. Answer **only** from the retrieved context
2. Respond with *"I don't have enough information in the provided documents."* when the context is insufficient
3. Never use external knowledge or make up information

---

## 💬 Example Interactions

### Basic Return Policy Query
```
You: What is Amazon's return policy?

Bot: Based on the Amazon return policy document, Amazon offers a 30-day return
     policy for most items sold and fulfilled by Amazon. The return window begins
     from the date of delivery...
     
     📄 Source(s): [amazon_return_policy.txt]
```

### Follow-Up Question (Context Awareness)
```
You: Does this apply to discounted items?

Bot: Yes, according to the Amazon return policy document, items purchased at a
     discount or during a sale ARE eligible for return within the standard 30-day
     window. However, Final Sale items are explicitly marked as non-returnable...
     
     📄 Source(s): [amazon_return_policy.txt]
```

### Hallucination Refusal (Out-of-Scope)
```
You: What is Costco's return policy?

Bot: I don't have enough information in the provided documents.
```

### Cross-Document Query
```
You: How does Walmart's electronics return policy compare to Best Buy's?

Bot: Based on the Walmart and Best Buy return policy documents:
     - Walmart: 30 days for general electronics, 15 days for laptops/tablets
     - Best Buy: 15 days standard, 30 days for Plus members, 60 days for Total...
     
     📄 Source(s): [walmart_return_policy.txt, bestbuy_return_policy.txt]
```

---

## 🎨 Design Decisions

| Decision | Rationale |
|---|---|
| **`text-embedding-3-small`** | 1536-dim embeddings with excellent quality/cost ratio vs. `ada-002` |
| **`gpt-4o-mini`** | Strong instruction following, low cost, fast responses |
| **`RecursiveCharacterTextSplitter`** | Respects paragraph/sentence boundaries for coherent chunks |
| **Chunk size: 800 chars** | Balance between context richness and retrieval precision |
| **Chunk overlap: 150 chars** | Prevents important context from being split across chunk boundaries |
| **Temperature: 0.0** | Deterministic outputs; reduces hallucination risk |
| **Top-k: 5** | Enough context for comprehensive answers without overwhelming the prompt |
| **Pinecone Serverless** | Free-tier compatible, no infrastructure management, scales automatically |
| **Conversation history in system prompt** | Enables natural follow-up questions without re-embedding history |
| **Source metadata on chunks** | Enables precise citation of source documents in answers |

---

## 🛠️ Troubleshooting

### "Missing environment variables" error
- Ensure `.env` file exists in the project root
- Verify API keys are correctly set (no extra spaces)

### "No documents found" during ingestion
- Ensure `.txt` files are in the `docs/` directory
- Check file encoding (must be UTF-8)

### Pinecone index not found
- Run `python ingest.py` before starting the chatbot
- Verify `PINECONE_INDEX_NAME` matches in `.env`

### Slow first response
- First query may take longer as connections are established
- Subsequent queries will be faster

### Poor answer quality
- Try increasing `--top-k` to retrieve more context chunks
- Ensure ingestion was completed successfully

### Streamlit app won't start
- Ensure `streamlit` is installed: `pip install streamlit`
- Check that port 8501 is not already in use

---

## 📊 Performance Notes

- **Ingestion time**: ~30–60 seconds for 6 documents (~6,000 words total)
- **Query latency**: 1–3 seconds per query (retrieval + LLM generation)
- **Pinecone free tier**: Supports up to 100K vectors, sufficient for this project
- **OpenAI costs**: Minimal for development (~$0.001 per query with gpt-4o-mini)

---

## 📄 License

This project is for educational and demonstration purposes. All referenced documents are from publicly available sources. Please review the individual platform's terms of service before using their content in production applications.
