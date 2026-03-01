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

questions = [
    "what are the three container types in Roaring Bitmaps?",
    "what are the advantages AND disadvantages of Roaring Bitmaps?",
    "what is the capital of France?",
]

for q in questions:
    print(f"\n🔍 Query: {q}")
    result = chain.invoke({"query": q})

    print(f"🤖 {result['result']}")

    if result.get("source_documents"):
        print("\n📄 Sources used:")
        seen = set()
        for doc in result["source_documents"]:
            preview = doc.page_content[:100].replace("\n", " ")
            if preview not in seen:
                seen.add(preview)
                print(f"  → {preview}...")

    print("—" * 60)
