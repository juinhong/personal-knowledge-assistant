from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import os

load_dotenv()

# --- Change this to your filename ---
FILE_PATH = "docs/untitled.txt"  # or .txt

# Load based on file type
if FILE_PATH.endswith(".pdf"):
    loader = PyPDFLoader(FILE_PATH)
elif FILE_PATH.endswith(".txt"):
    loader = TextLoader(FILE_PATH)

documents = loader.load()

print(f"Loaded {len(documents)} page(s)\n")

for i, doc in enumerate(documents):
    print(f"--- Page {i+1} ---")
    print(f"Content preview: {doc.page_content[:200]}")
    print(f"Metadata: {doc.metadata}\n")