"""
ingest.py — Load, chunk, embed, and store documents into ChromaDB.
Usage: python -m src.ingest
"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

DOCS_DIR = "docs"
CHROMA_DIR = ".chroma"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "text-embedding-3-small"


def load_documents(docs_dir: str):
    """Load all .txt and .pdf files from the docs directory."""
    documents = []

    # Load .txt files
    txt_loader = DirectoryLoader(
        docs_dir,
        glob="**/*.txt",
        loader_cls=TextLoader,
        silent_errors=True
    )
    documents.extend(txt_loader.load())

    # Load .pdf files individually (DirectoryLoader + PyPDFLoader can be finicky)
    for filename in os.listdir(docs_dir):
        if filename.endswith(".pdf"):
            path = os.path.join(docs_dir, filename)
            loader = PyPDFLoader(path)
            documents.extend(loader.load())

    return documents


def chunk_documents(documents):
    """Split documents into chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


def build_vectorstore(chunks):
    """Embed chunks and store in ChromaDB."""
    embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_DIR
    )
    return vectorstore


def ingest():
    print(f"📂 Loading documents from '{DOCS_DIR}'...")
    documents = load_documents(DOCS_DIR)

    if not documents:
        print("❌ No documents found. Add .txt or .pdf files to the docs/ folder.")
        return

    print(f"✅ Loaded {len(documents)} document(s):")
    for doc in documents:
        print(f"   → {doc.metadata.get('source', 'unknown')}")

    print(f"\n✂️  Chunking documents...")
    chunks = chunk_documents(documents)
    print(f"✅ Created {len(chunks)} chunks")

    print(f"\n🔢 Embedding and storing in ChromaDB...")
    vectorstore = build_vectorstore(chunks)
    print(f"✅ Stored {vectorstore._collection.count()} chunks in ChromaDB")
    print(f"\n🎉 Ingestion complete! Run 'python main.py' to start chatting.")


if __name__ == "__main__":
    ingest()
