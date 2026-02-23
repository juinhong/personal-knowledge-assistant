from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
client = OpenAI()

# Step 1 — Load and chunk
loader = TextLoader("docs/notes.txt")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
chunks = splitter.split_documents(documents)

print(f"Total chunks to embed: {len(chunks)}\n")

# Step 2 — Embed each chunk
def embed_texts(texts):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [item.embedding for item in response.data]

# Extract just the text from each chunk
texts = [chunk.page_content for chunk in chunks]

# Embed all at once (more efficient than one by one)
embeddings = embed_texts(texts)

print(f"Embedded {len(embeddings)} chunks")
print(f"Each embedding has {len(embeddings[0])} dimensions\n")

# Step 3 — Preview
for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
    print(f"--- Chunk {i+1} ---")
    print(f"Text: {chunk.page_content[:100]}...")
    print(f"Embedding preview: {embedding[:5]}...")
    print()