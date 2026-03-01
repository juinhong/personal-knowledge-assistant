from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()
client = OpenAI()

# Load vectorstore from disk
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    persist_directory=".chroma",
    embedding_function=embedding_model
)


def retrieve(query, k=5):
    results = vectorstore.similarity_search_with_score(query, k=k)
    # Filter out low relevance chunks (score > 1.5 = too far)
    relevant = [(doc, score) for doc, score in results if score < 1.5]
    return relevant


def augment(query, retrieved_chunks):
    context = "\n\n".join([doc.page_content for doc, _ in retrieved_chunks])

    system_prompt = f"""You are a helpful assistant. 
Answer the user's question using ONLY the context below.
If the answer is not in the context, say "I don't have that information."
Be concise and precise.

Context:
{context}"""

    return system_prompt


def generate(query, system_prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
    )
    return response.choices[0].message.content


def rag(query):
    print(f"\n🔍 Query: {query}")

    # Step 1 — Retrieve
    chunks = retrieve(query)

    if not chunks:
        return "I don't have that information in my knowledge base."

    print(f"📄 Retrieved {len(chunks)} relevant chunks")

    # Step 2 — Augment
    system_prompt = augment(query, chunks)

    # Step 3 — Generate
    answer = generate(query, system_prompt)

    return answer


# def rag_debug(query):
#     chunks = retrieve(query)
#     print(f"\n🔍 Query: {query}")
#     print(f"📄 Retrieved {len(chunks)} chunks:\n")
#     for doc, score in chunks:
#         print(f"Score: {score:.4f}")
#         print(f"Content: {doc.page_content}")
#         print("---")
#
#
# rag_debug("what are the three container types in Roaring Bitmaps?")

# Test it
questions = [
    "what are the three container types in Roaring Bitmaps?",
    "when should I use hash sets instead of Roaring Bitmaps?",
    "what is the capital of France?",
]

for q in questions:
    answer = rag(q)
    print(f"🤖 {answer}\n")
    print("—" * 60)
