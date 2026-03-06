"""
retriever.py — Load ChromaDB and retrieve relevant chunks for a query.
"""

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

CHROMA_DIR = ".chroma"
EMBEDDING_MODEL = "text-embedding-3-small"
SCORE_THRESHOLD = 1.5  # lower = more similar in ChromaDB distance metric
TOP_K = 5


def load_vectorstore():
    """Load existing ChromaDB vectorstore from disk."""
    embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedding_model
    )


def retrieve(vectorstore, query: str, k: int = TOP_K):
    """
    Search vectorstore for relevant chunks.
    Returns list of (Document, score) tuples that pass the score threshold.
    """
    results = vectorstore.similarity_search_with_score(query, k=k)
    relevant = [(doc, score) for doc, score in results if score < SCORE_THRESHOLD]
    return relevant


def format_sources(docs_and_scores: list) -> str:
    """Format retrieved docs as a readable source list."""
    seen = set()
    lines = []
    for doc, score in docs_and_scores:
        source = doc.metadata.get("source", "unknown")
        preview = doc.page_content[:80].replace("\n", " ")
        key = f"{source}:{preview}"
        if key not in seen:
            seen.add(key)
            lines.append(f"  → [{source}] {preview}...")
    return "\n".join(lines)
