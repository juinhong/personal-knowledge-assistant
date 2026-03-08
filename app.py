import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader

from src.ingest import chunk_documents, build_vectorstore
from src.retriever import load_vectorstore
from src.rag import RAGPipeline

load_dotenv()

# --- Page config ---
st.set_page_config(
    page_title="Personal Knowledge Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Personal Knowledge Assistant")


# --- Helper ---
def render_sources(sources: str):
    """Render sources in a clean expandable block."""
    if not sources:
        return

    with st.expander("📄 View sources", expanded=False):
        source_lines = [s for s in sources.strip().split("\n") if s.strip()]

        for line in source_lines:
            if "→" in line:
                parts = line.split("]", 1)
                if len(parts) == 2:
                    filename = parts[0].split("[")[-1]
                    preview = parts[1].strip().rstrip("...")
                    st.markdown(f"**📄 {filename}**")
                    st.caption(preview)
                    st.divider()
                else:
                    st.text(line)


# --- Sidebar ---
with st.sidebar:
    st.header("📂 Knowledge Base")

    # Show current chunk count
    try:
        vs = load_vectorstore()
        st.success(f"{vs._collection.count()} chunks loaded")
    except Exception:
        st.warning("No knowledge base found")

    st.divider()

    # File uploader
    st.subheader("Add Documents")
    uploaded_files = st.file_uploader(
        "Upload .txt or .pdf files",
        type=["txt", "pdf"],
        accept_multiple_files=True
    )

    if uploaded_files and st.button("📥 Ingest Documents"):
        with st.spinner("Ingesting..."):
            all_docs = []

            for uploaded_file in uploaded_files:
                suffix = ".pdf" if uploaded_file.name.endswith(".pdf") else ".txt"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                if suffix == ".pdf":
                    loader = PyPDFLoader(tmp_path)
                else:
                    loader = TextLoader(tmp_path)

                docs = loader.load()

                for doc in docs:
                    doc.metadata["source"] = uploaded_file.name

                all_docs.extend(docs)
                os.unlink(tmp_path)

            chunks = chunk_documents(all_docs)
            build_vectorstore(chunks)

            if "rag" in st.session_state:
                del st.session_state["rag"]

            st.success(f"✅ Ingested {len(all_docs)} doc(s), {len(chunks)} chunks")
            st.rerun()

    st.divider()

    # Reset conversation
    if st.button("🔄 Reset Conversation"):
        st.session_state.messages = []
        if "rag" in st.session_state:
            st.session_state.rag.reset()
        st.rerun()

# --- Init RAG pipeline ---
if "rag" not in st.session_state:
    with st.spinner("Loading knowledge base..."):
        try:
            vectorstore = load_vectorstore()
            st.session_state.rag = RAGPipeline(vectorstore)
        except Exception:
            st.info("👈 Upload documents in the sidebar to get started.")
            st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Render conversation history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        render_sources(msg.get("sources", ""))

# --- Chat input ---
if prompt := st.chat_input("Ask anything from your knowledge base..."):

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Get and show assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = st.session_state.rag.ask(prompt, verbose=False)

        st.write(result["answer"])

        if result["sources"]:
            render_sources(result["sources"])
        else:
            st.caption("⚠️ No relevant sources found in knowledge base")

    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"]
    })
