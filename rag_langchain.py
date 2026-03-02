from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

# Load vectorstore
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    persist_directory=".chroma",
    embedding_function=embedding_model
)

# Build retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# Custom prompt
prompt_template = """You are a personal knowledge assistant.
Your job is to answer questions based strictly on the provided context.

Rules:
- Answer ONLY using the context below. Never use outside knowledge.
- If the answer is not in the context, say "I don't have that information in my knowledge base."
- Be concise — get to the point in 1-3 sentences unless a longer answer is clearly needed.
- If the question asks for a list, respond with a clean numbered or bulleted list.
- If the question is ambiguous or unclear, say "Could you clarify what you mean by [unclear part]?"
- If you're unsure, say so — don't guess.

Context:
{context}

Question: {question}

Answer:"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# return_source_documents=True is the key change
chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)


def ask(query):
    print(f"\n🔍 Query: {query}")

    # Manually check scores instead of using threshold
    docs_and_scores = vectorstore.similarity_search_with_score(query, k=5)

    # Filter — remember lower score = more similar for ChromaDB distance
    relevant_docs = [doc for doc, score in docs_and_scores if score < 1.5]

    if not relevant_docs:
        print("🤖 I don't have any relevant information about that in my knowledge base.")
        print("—" * 60)
        return

    result = chain.invoke({"query": query})
    print(f"🤖 {result['result']}")

    print("\n📄 Sources used:")
    seen = set()
    for doc in result["source_documents"]:
        source = doc.metadata.get("source", "unknown")
        preview = doc.page_content[:80].replace("\n", " ")
        key = f"{source}:{preview}"
        if key not in seen:
            seen.add(key)
            print(f"  → [{source}] {preview}...")

    print("—" * 60)


questions = [
    "what are the three container types in Roaring Bitmaps?",  # normal
    "what is the capital of France?",  # no relevant docs
    "tell me more",  # ambiguous
    "summarize everything",  # vague
    "what did they say about performance?",  # vague reference
]

for q in questions:
    ask(q)
