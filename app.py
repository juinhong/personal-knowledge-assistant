import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader

from src.ingest import chunk_documents, build_vectorstore
from src.retriever import load_vectorstore
from src.rag import RAGPipeline

# Works both locally (.env) and on Streamlit Cloud (secrets)
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

load_dotenv()

# --- Page config ---
st.set_page_config(
    page_title="Personal Knowledge Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .stApp { background-color: #f8f9fb; }
    .stChatInput input {
        border: 1.5px solid #d0d5dd;
        border-radius: 12px;
        background-color: #ffffff;
        color: #1a1a2e;
        font-size: 0.95rem;
    }
    [data-testid="stChatMessageContent"] {
        border-radius: 12px;
        padding: 4px;
        color: #1a1a2e;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e8eaf0;
    }
    .streamlit-expanderHeader {
        background-color: #f0f2f6;
        border-radius: 8px;
        font-size: 0.85rem;
        color: #444;
    }
    hr { border-color: #e8eaf0; }
    .stAlert { border-radius: 8px; }
    h1 { color: #1a1a2e; font-weight: 700; }
    h2, h3 { color: #2d3250; }
    p, li { color: #2d3250; }
    .stCaption { color: #6b7280; }
    [data-testid="stMetricValue"] { color: #2d3250; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Personal Knowledge Assistant")


# --- Helper functions ---
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


def get_vectorstore_count():
    try:
        vs = load_vectorstore()
        return vs._collection.count()
    except Exception:
        return None


# --- Sidebar ---
with st.sidebar:
    st.markdown("## 🤖 Knowledge Assistant")
    st.caption("Powered by GPT-4o-mini + ChromaDB")
    st.divider()

    # Knowledge base status
    st.markdown("### 📂 Knowledge Base")
    chunk_count = get_vectorstore_count()
    if chunk_count:
        st.success(f"✅ {chunk_count} chunks indexed")
    else:
        st.warning("⚠️ No knowledge base found")

    st.divider()

    # File uploader
    st.markdown("### ➕ Add Documents")
    uploaded_files = st.file_uploader(
        "Upload .txt or .pdf files",
        type=["txt", "pdf"],
        accept_multiple_files=True,
        help="Documents will be chunked, embedded, and added to your knowledge base."
    )

    if uploaded_files and st.button("📥 Ingest Documents", use_container_width=True):
        with st.spinner("Processing documents..."):
            all_docs = []
            progress = st.progress(0)

            for i, uploaded_file in enumerate(uploaded_files):
                suffix = ".pdf" if uploaded_file.name.endswith(".pdf") else ".txt"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                loader = PyPDFLoader(tmp_path) if suffix == ".pdf" else TextLoader(tmp_path)
                docs = loader.load()

                for doc in docs:
                    doc.metadata["source"] = uploaded_file.name

                all_docs.extend(docs)
                os.unlink(tmp_path)
                progress.progress((i + 1) / len(uploaded_files))

            chunks = chunk_documents(all_docs)
            build_vectorstore(chunks)

            if "rag" in st.session_state:
                del st.session_state["rag"]

            st.success(f"✅ {len(all_docs)} doc(s) → {len(chunks)} chunks")
            st.rerun()

    st.divider()

    # Settings
    st.markdown("### ⚙️ Settings")
    show_sources = st.toggle("Show sources", value=True)
    show_reformulated = st.toggle("Show reformulated queries", value=False)

    st.divider()

    # Conversation controls
    st.markdown("### 💬 Conversation")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Reset", use_container_width=True, help="Clear conversation history"):
            st.session_state.messages = []
            if "rag" in st.session_state:
                st.session_state.rag.reset()
            st.rerun()
    with col2:
        msg_count = len(st.session_state.get("messages", []))
        st.metric("Messages", msg_count)

    st.divider()
    st.caption("Built with LangChain · OpenAI · Streamlit")

# --- Init RAG pipeline ---
if "rag" not in st.session_state:
    with st.spinner("Loading knowledge base..."):
        try:
            vectorstore = load_vectorstore()
            st.session_state.rag = RAGPipeline(vectorstore)
        except Exception:
            st.markdown("""
            ### 👋 Welcome!
            Upload your documents in the sidebar to get started.
            Your personal knowledge assistant will answer questions
            based on your own files.
            """)
            st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Welcome message on first load
if len(st.session_state.messages) == 0:
    with st.chat_message("assistant"):
        st.write(
            f"👋 Hi! I've loaded your knowledge base with **{get_vectorstore_count()} chunks**. Ask me anything about your documents!")

# --- Render conversation history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if show_sources:
            render_sources(msg.get("sources", ""))

# --- Chat input ---
if prompt := st.chat_input("Ask anything from your knowledge base..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = st.session_state.rag.ask(prompt, verbose=show_reformulated)

        st.write(result["answer"])

        if show_sources:
            if result["sources"]:
                render_sources(result["sources"])
            else:
                st.caption("⚠️ No relevant sources found in knowledge base")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"]
    })
