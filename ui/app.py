import os
import json
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")

st.set_page_config(page_title="SignalRAG Assistant", page_icon="⚡", layout="wide")
st.title("⚡ SignalRAG Assistant")
st.caption("Minimalist RAG powered by FastAPI, PostgreSQL (pgvector), and Google Gemini")

if "messages" not in st.session_state:
    st.session_state.messages = []

def get_available_filenames():
    try:
        res = requests.get(f"{BACKEND_URL}/filenames", timeout=5)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return []

tab1, tab2 = st.tabs(["💬 Ask Questions", "📥 Ingest Documents"])

# Tab 1: Ask Questions (Multi-turn Chat)
with tab1:
    st.header("Query the Knowledge Base")

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        top_k = st.slider("Context chunks to retrieve (top_k):", min_value=1, max_value=5, value=3)
    with col2:
        available_files = ["All Documents"] + get_available_filenames()
        selected_file = st.selectbox("Filter by Document (Metadata):", options=available_files)
    with col3:
        st.write(" ")
        st.write(" ")
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    st.divider()

    # Render previous conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            
            # Display query rewriting and latency metrics
            if "metrics" in msg and msg["metrics"]:
                m = msg["metrics"]
                if m.get("query_rewritten"):
                    st.caption(f"🔍 **Rewritten Query:** *{m.get('standalone_query')}*")
                if m.get("retrieval_ms"):
                    st.caption(f"⚡ **Retrieval Latency:** {m.get('retrieval_ms')} ms")

            if "sources" in msg and msg["sources"]:
                with st.expander("Retrieved Context Chunks"):
                    for i, source in enumerate(msg["sources"], 1):
                        filename = source.get("filename", "Unknown")
                        page = source.get("page_number")
                        page_label = f"Page {page}" if page else "N/A"
                        dist = source.get("distance")
                        dist_label = f" | Cosine Distance: {dist}" if dist is not None else ""
                        content = source.get("content", "")
                        st.markdown(f"**Source {i}:** `{filename}` ({page_label}){dist_label}")
                        st.info(content)

    # Chat input box
    if prompt := st.chat_input("Ask a question or follow-up..."):
        st.chat_message("user").write(prompt)

        chat_history_payload = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ]

        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/ask",
                    json={
                        "question": prompt,
                        "top_k": top_k,
                        "filename": selected_file,
                        "chat_history": chat_history_payload
                    },
                    stream=True,
                    timeout=30
                )

                if response.status_code == 200:
                    # Container dictionary to avoid scope binding issues
                    sources_holder = {"data": [], "metrics": {}}

                    def token_generator():
                        for line in response.iter_lines():
                            if line:
                                payload = json.loads(line.decode("utf-8"))
                                if payload.get("type") == "sources":
                                    sources_holder["data"] = payload.get("data", [])
                                    sources_holder["metrics"] = payload.get("metrics", {})
                                elif payload.get("type") == "token":
                                    yield payload.get("text", "")

                    full_response = st.write_stream(token_generator())

                    retrieved_sources = sources_holder["data"]
                    retrieved_metrics = sources_holder["metrics"]

                    # Render query rewriting and latency indicators
                    if retrieved_metrics.get("query_rewritten"):
                        st.caption(f"🔍 **Rewritten Query:** *{retrieved_metrics.get('standalone_query')}*")
                    if retrieved_metrics.get("retrieval_ms"):
                        st.caption(f"⚡ **Retrieval Latency:** {retrieved_metrics.get('retrieval_ms')} ms")

                    if retrieved_sources:
                        with st.expander("Retrieved Context Chunks"):
                            for i, source in enumerate(retrieved_sources, 1):
                                filename = source.get("filename", "Unknown")
                                page = source.get("page_number")
                                page_label = f"Page {page}" if page else "N/A"
                                dist = source.get("distance")
                                dist_label = f" | Cosine Distance: {dist}" if dist is not None else ""
                                content = source.get("content", "")
                                st.markdown(f"**Source {i}:** `{filename}` ({page_label}){dist_label}")
                                st.info(content)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "sources": retrieved_sources,
                        "metrics": retrieved_metrics
                    })
                else:
                    st.error(f"Backend Error ({response.status_code}): {response.text}")
            except Exception as e:
                st.error(f"Could not connect to backend at {BACKEND_URL}: {str(e)}")

# Tab 2: Ingest Documents
with tab2:
    st.header("Add New Knowledge")

    ingest_type = st.radio("Choose Input Method:", ["📄 PDF Document", "📝 Plain Text"], horizontal=True)

    if ingest_type == "📄 PDF Document":
        uploaded_file = st.file_uploader("Upload a PDF file:", type=["pdf"])

        if st.button("Upload & Embed PDF", type="primary"):
            if uploaded_file is None:
                st.warning("Please select a PDF file first.")
            else:
                with st.spinner("Extracting text, generating embeddings, and storing in pgvector..."):
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                        response = requests.post(f"{BACKEND_URL}/ingest-pdf", files=files, timeout=300)

                        if response.status_code == 200:
                            data = response.json()
                            st.success(f"Successfully processed **{data.get('filename')}** into {data.get('chunks_ingested')} page-tracked chunks!")
                            st.rerun()
                        else:
                            st.error(f"Failed to process PDF ({response.status_code}): {response.text}")
                    except Exception as e:
                        st.error(f"Error connecting to backend: {str(e)}")

    else:
        raw_text = st.text_area("Paste text or content to embed:", height=200, placeholder="Paste reference text here...")

        if st.button("Ingest Knowledge", type="primary"):
            if not raw_text.strip():
                st.warning("Please provide text content to ingest.")
            else:
                with st.spinner("Generating embeddings and saving to pgvector..."):
                    try:
                        response = requests.post(
                            f"{BACKEND_URL}/ingest",
                            json={"text": raw_text},
                            timeout=30
                        )
                        if response.status_code == 200:
                            st.success("Successfully embedded and stored text in PostgreSQL!")
                            st.rerun()
                        else:
                            st.error(f"Ingestion failed ({response.status_code}): {response.text}")
                    except Exception as e:
                        st.error(f"Could not connect to backend at {BACKEND_URL}: {str(e)}")

    st.divider()
    st.subheader("🧹 Database Management")
    st.caption("Wipe all stored vectors and start fresh with new documents.")

    if st.button("Clear Knowledge Base", type="secondary"):
        with st.spinner("Wiping stored vector chunks..."):
            try:
                response = requests.delete(f"{BACKEND_URL}/reset", timeout=10)
                if response.status_code == 200:
                    st.success("Knowledge base cleared successfully!")
                    st.rerun()
                else:
                    st.error(f"Failed to clear database ({response.status_code}): {response.text}")
            except Exception as e:
                st.error(f"Could not connect to backend at {BACKEND_URL}: {str(e)}")