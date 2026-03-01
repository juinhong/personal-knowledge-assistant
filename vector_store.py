from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os

load_dotenv()

# Load ALL files from docs/ folder at once
loader = DirectoryLoader(
    "docs/",
    glob="**/*.pdf",  # change to **/*.pdf for PDFs
    loader_cls=PyPDFLoader
)

documents = loader.load()
print(f"Loaded {len(documents)} document(s)")
for doc in documents:
    print(f"  → {doc.metadata['source']}")

# Chunk
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
chunks = splitter.split_documents(documents)
print(f"\nTotal chunks: {len(chunks)}")

# Embed and store
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=".chroma"
)

print(f"\n✅ Stored {vectorstore._collection.count()} chunks in ChromaDB")
