from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load
loader = TextLoader("docs/notes.txt")
documents = loader.load()

# Chunk
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # max characters per chunk
    chunk_overlap=50,      # overlap between chunks
)

chunks = splitter.split_documents(documents)

print(f"Original: {len(documents)} document(s)")
print(f"After chunking: {len(chunks)} chunks\n")

for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} ---")
    print(chunk.page_content)
    print()