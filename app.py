import streamlit as st
from dotenv import load_dotenv
from src.retriever import load_vectorstore
from src.rag import RAGPipeline
from src.ingest import chunk_documents, build_vectorstore
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import tempfile
import os

load_dotenv()

st.set_page_config(
    page_title="Personal Knowledge Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Personal Knowledge Assistant")

# --- Sidebar ---
with st.sidebar:
    st.header("📂 Knowledge Base")

    # Show current chunk count
    try:
        vs = load_vectorstore()
        st.success(f"{vs._collection.count()} chunks loaded")
    except:
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
                # Save to temp file so loaders can read it
                suffix = ".pdf" if uploaded_file.name.endswith(".pdf") else ".txt"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                # Load using appropriate loader
                if suffix == ".pdf":
                    loader = PyPDFLoader(tmp_path)
                else:
                    loader = TextLoader(tmp_path)

                docs = loader.load()

                # Fix metadata to show original filename
                for doc in docs:
                    doc.metadata["source"] = uploaded_file.name

                all_docs.extend(docs)
                os.unlink(tmp_path)  # clean up temp file

            # Chunk and store
            chunks = chunk_documents(all_docs)
            build_vectorstore(chunks)

            # Reset RAG pipeline so it picks up new docs
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

# --- Main chat ---
if "rag" not in st.session_state:
    with st.spinner("Loading knowledge base..."):
        try:
            vectorstore = load_vectorstore()
            st.session_state.rag = RAGPipeline(vectorstore)
        except:
            st.info("👈 Upload documents in the sidebar to get started.")
            st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 Sources"):
                st.write(msg["sources"])

# Chat input
if prompt := st.chat_input("Ask anything from your knowledge base..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = st.session_state.rag.ask(prompt, verbose=False)

        st.write(result["answer"])

        if result["sources"]:
            with st.expander("📄 Sources"):
                st.write(result["sources"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"]
    })
