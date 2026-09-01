import io
import os
import time
import json
from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pypdf import PdfReader
import psycopg2
from psycopg2.extras import RealDictCursor
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

# Initialize FastAPI App
app = FastAPI(title="SignalRAG Backend API")

# Fetch Environment Variables
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@db:5432/signalrag")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize Gemini Client
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# Pydantic Schemas
class ChatMessage(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3
    chat_history: Optional[List[ChatMessage]] = []
    filename: Optional[str] = None  # Phase 7: Metadata pre-filter

class IngestRequest(BaseModel):
    text: str

def get_db_connection():
    """Creates a fresh connection to PostgreSQL."""
    return psycopg2.connect(DATABASE_URL)

@app.on_event("startup")
def startup_db_setup():
    """Automated database schema migration and HNSW vector indexing on startup."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. Enable pgvector extension
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    
    # 2. Create document_chunks table with metadata columns
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS document_chunks (
            id SERIAL PRIMARY KEY,
            content TEXT NOT NULL,
            embedding vector(768),
            filename TEXT,
            page_number INT
        );
    """)
    
    cursor.execute("ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS filename TEXT;")
    cursor.execute("ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS page_number INT;")
    
    # Phase 8: HNSW Index for ultra-fast vector similarity search
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS document_chunks_embedding_hnsw_idx 
        ON document_chunks 
        USING hnsw (embedding vector_cosine_ops);
    """)
    
    conn.commit()
    cursor.close()
    conn.close()

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=2, max=15),
    reraise=True
)
def get_embedding_with_retry(text: str):
    """Generates embeddings with automatic retries on 429 rate limit errors."""
    return client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=types.EmbedContentConfig(output_dimensionality=768)
    )

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Slices text into overlapping character chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def rewrite_query_for_retrieval(question: str, history: List[ChatMessage]) -> str:
    """Phase 7: Query Disambiguation - converts follow-up questions into standalone vector search queries."""
    if not history or not client:
        return question

    formatted_history = "\n".join([
        f"{'User' if m.role == 'user' else 'Assistant'}: {m.content}" 
        for m in history[-4:]
    ])
    
    prompt = f"""Given the conversation history and a follow-up question, rewrite the follow-up question to be a self-contained search query. Do not answer the question; only rewrite it. If it is already standalone, return it unchanged.

Conversation History:
{formatted_history}

Follow-up Question: {question}

Standalone Search Query:"""

    try:
        res = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )
        rewritten = res.text.strip()
        return rewritten if rewritten else question
    except Exception:
        return question

@app.get("/health")
def health_check():
    return {"status": "ok", "database_connected": True}

@app.get("/filenames")
def get_filenames():
    """Phase 7: Returns distinct uploaded document filenames for UI filtering."""
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute("SELECT DISTINCT filename FROM document_chunks WHERE filename IS NOT NULL;")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return [r["filename"] for r in rows if r["filename"]]

@app.post("/ingest-pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    contents = await file.read()
    pdf_file = io.BytesIO(contents)
    reader = PdfReader(pdf_file)

    conn = get_db_connection()
    cursor = conn.cursor()
    ingested_count = 0

    try:
        for page_idx, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text()
            if not page_text or not page_text.strip():
                continue

            chunks = chunk_text(page_text, chunk_size=500, overlap=50)

            for chunk in chunks:
                chunk_str = chunk.strip()
                if not chunk_str:
                    continue

                emb_response = get_embedding_with_retry(chunk_str)
                embedding = emb_response.embeddings[0].values

                cursor.execute(
                    """
                    INSERT INTO document_chunks (content, embedding, filename, page_number) 
                    VALUES (%s, %s, %s, %s);
                    """,
                    (chunk_str, str(embedding), file.filename, page_idx)
                )
                ingested_count += 1
                time.sleep(0.1)

        conn.commit()
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"PDF Ingestion Failed: {str(e)}")
    finally:
        cursor.close()
        conn.close()

    return {
        "filename": file.filename,
        "chunks_ingested": ingested_count,
        "message": f"Successfully ingested {file.filename} into {ingested_count} page-tracked chunks!"
    }

@app.post("/ingest")
def ingest_text(payload: IngestRequest):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    
    emb_response = get_embedding_with_retry(payload.text)
    embedding = emb_response.embeddings[0].values

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO document_chunks (content, embedding, filename) VALUES (%s, %s, %s);",
        (payload.text, str(embedding), "Plain Text")
    )
    conn.commit()
    cursor.close()
    conn.close()

    return {"message": "Text ingested successfully!"}

@app.post("/ask")
def ask_rag(payload: QueryRequest):
    """Performs disambiguated vector search, metadata filtering, latency tracking, and streams Gemini responses."""
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    start_time = time.time()

    # Phase 7: Disambiguate follow-up question
    standalone_query = rewrite_query_for_retrieval(payload.question, payload.chat_history or [])

    # Vector embedding generation
    emb_response = get_embedding_with_retry(standalone_query)
    query_vector = emb_response.embeddings[0].values
    
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    # Phase 7: Metadata filtering query
    if payload.filename and payload.filename != "All Documents":
        search_query = """
            SELECT content, filename, page_number, (embedding <=> %s::vector) AS distance
            FROM document_chunks
            WHERE filename = %s
            ORDER BY distance ASC
            LIMIT %s;
        """
        cursor.execute(search_query, (str(query_vector), payload.filename, payload.top_k))
    else:
        search_query = """
            SELECT content, filename, page_number, (embedding <=> %s::vector) AS distance
            FROM document_chunks
            ORDER BY distance ASC
            LIMIT %s;
        """
        cursor.execute(search_query, (str(query_vector), payload.top_k))

    results = cursor.fetchall()
    cursor.close()
    conn.close()

    # Phase 8: Latency metrics computation
    retrieval_ms = round((time.time() - start_time) * 1000, 2)

    retrieved_sources = [
        {
            "content": row["content"],
            "filename": row.get("filename") or "Plain Text",
            "page_number": row.get("page_number"),
            "distance": round(float(row.get("distance", 0)), 4) if row.get("distance") is not None else None
        }
        for row in results
    ]

    context_block = "\n---\n".join([
        f"[{s['filename']} - Page {s['page_number'] if s['page_number'] else 'N/A'}]: {s['content']}" 
        for s in retrieved_sources
    ]) if retrieved_sources else "No context found."

    history_block = ""
    if payload.chat_history:
        formatted_turns = [
            f"{'User' if msg.role == 'user' else 'Assistant'}: {msg.content}" 
            for msg in payload.chat_history
        ]
        history_block = "\n".join(formatted_turns)

    prompt = f"""You are a helpful AI assistant with access to retrieved document context.
Answer the user's latest question using the context and previous conversation history.
If the context doesn't contain enough information to answer, state that clearly.

Context:
{context_block}

Conversation History:
{history_block if history_block else "No prior conversation history."}

User Question:
{payload.question}

Answer:"""

    def stream_generator():
        # Phase 6 & 8: Yield sources, query metrics, and rewritten query first
        metrics = {
            "retrieval_ms": retrieval_ms,
            "standalone_query": standalone_query,
            "query_rewritten": standalone_query != payload.question
        }
        yield json.dumps({"type": "sources", "data": retrieved_sources, "metrics": metrics}) + "\n"
        
        response_stream = client.models.generate_content_stream(
            model="gemini-3.6-flash",
            contents=prompt
        )
        for chunk in response_stream:
            if chunk.text:
                yield json.dumps({"type": "token", "text": chunk.text}) + "\n"

    return StreamingResponse(stream_generator(), media_type="application/x-ndjson")

@app.delete("/reset")
def reset_database():
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("TRUNCATE TABLE document_chunks RESTART IDENTITY;")
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to reset database: {str(e)}")
    finally:
        cursor.close()
        conn.close()

    return {"message": "Knowledge base wiped successfully!"}