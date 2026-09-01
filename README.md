# ⚡ SignalRAG: Production-Ready Local RAG System

SignalRAG is a containerized Retrieval-Augmented Generation (RAG) system built with FastAPI, PostgreSQL (`pgvector`), Streamlit, and Google Gemini.

## 🚀 Key Features
* **Real-Time Token Streaming**: Streams answers token-by-token using NDJSON and FastAPI `StreamingResponse`.
* **Metadata Pre-Filtering**: Restricts vector search to specific uploaded document filenames.
* **Conversational Query Disambiguation**: Rewrites multi-turn follow-up questions into standalone search queries using Gemini.
* **Sub-Millisecond Vector Search**: Leverages HNSW indexing (`vector_cosine_ops`) in `pgvector`.
* **Page-Tracked Ingestion**: Parses PDFs page-by-page preserving accurate source attribution.

## 🛠️ Tech Stack
* **Backend**: FastAPI, Python 3.11, PyPDF, Tenacity
* **Database**: PostgreSQL 16 + `pgvector` extension
* **LLM & Embeddings**: Google Gemini API (`gemini-3.6-flash`, `gemini-embedding-001`)
* **Frontend**: Streamlit
* **Orchestration**: Docker & Docker Compose

## ⚡ Quick Start

1. **Clone Repository**:
   ```bash
   git clone [https://github.com/vinothat1/signal-rag.git](https://github.com/vinothat1/signal-rag.git)
   cd signal-rag
