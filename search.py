from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma(
    persist_directory=".chroma",
    embedding_function=embedding_model
)

queries = [
    "what are the three container types?",
    "what is the capital of France?",  # not in your docs
]

for query in queries:
    print(f"🔍 Query: {query}")
    
    # similarity_search_with_score returns (doc, score) tuples
    results = vectorstore.similarity_search_with_score(query, k=3)
    
    for doc, score in results:
        print(f"\n  Score: {score:.4f}")
        print(f"  Content: {doc.page_content[:200]}")
    print("\n" + "—"*60 + "\n")