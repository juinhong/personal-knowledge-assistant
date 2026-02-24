from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

# Load the existing vectorstore from disk (no re-embedding needed)
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma(
    persist_directory=".chroma",
    embedding_function=embedding_model
)

print(f"Loaded vectorstore with {vectorstore._collection.count()} chunks\n")

# Query it
queries = [
    "what are the three container types?",
    "when should I use Roaring Bitmaps over hash sets?",
    "how does memory efficiency work?",
]

for query in queries:
    print(f"🔍 Query: {query}")
    results = vectorstore.similarity_search(query, k=2)  # top 2 chunks 
    
    for i, doc in enumerate(results):
        print(f"\n  Result {i+1}:")
        print(f"  {doc.page_content[:300]}")
    print("\n" + "—"*60 + "\n")