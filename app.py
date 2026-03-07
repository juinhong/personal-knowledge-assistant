import streamlit as st
from dotenv import load_dotenv
from src.retriever import load_vectorstore
from src.rag import RAGPipeline

load_dotenv()

# Page config
st.set_page_config(
    page_title="Personal Knowledge Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Personal Knowledge Assistant")

# Initialize session state — runs once, persists across reruns
if "rag" not in st.session_state:
    with st.spinner("Loading knowledge base..."):
        vectorstore = load_vectorstore()
        st.session_state.rag = RAGPipeline(vectorstore)

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

    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Get RAG response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = st.session_state.rag.ask(prompt, verbose=False)

        st.write(result["answer"])

        if result["sources"]:
            with st.expander("📄 Sources"):
                st.write(result["sources"])

    # Save assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"]
    })
