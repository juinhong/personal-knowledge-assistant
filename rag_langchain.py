from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

# Load vectorstore
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    persist_directory=".chroma",
    embedding_function=embedding_model
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Manual memory — same pattern as Session 2.4
chat_history = []


def summarize_history(chat_history):
    if not chat_history:
        return []

    history_text = "\n".join([
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in chat_history
    ])

    summary_prompt = f"""Summarize this conversation concisely in 3-5 sentences.
Capture the key topics discussed and important answers given.
Return ONLY the summary, nothing else.

Conversation:
{history_text}"""

    response = llm.invoke([{"role": "user", "content": summary_prompt}])
    summary = response.content.strip()

    # Replace full history with single summary message
    return [{"role": "system", "content": f"Previous conversation summary: {summary}"}]


def reformulate_query(query, chat_history):
    # No history yet — use query as-is
    if not chat_history:
        return query

    # Ask GPT to rewrite the question with full context
    history_text = "\n".join([
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in chat_history[-4:]  # last 2 exchanges only
    ])

    reformulation_prompt = f"""Given this conversation history:
{history_text}

Rewrite this follow-up question to be fully self-contained and specific,
so it can be understood without the conversation history.
Return ONLY the rewritten question, nothing else.

Follow-up question: {query}"""

    response = llm.invoke([{"role": "user", "content": reformulation_prompt}])
    reformulated = response.content.strip()
    print(f"🔄 Reformulated: {reformulated}")
    return reformulated


def ask(query):
    global chat_history

    print(f"\n👤 {query}")

    # Summarize if history gets long
    if len(chat_history) >= 6:
        print("📝 Summarizing conversation history...")
        chat_history = summarize_history(chat_history)
        print(f"✅ Compressed to 1 summary message\n")

    # Reformulate query using chat history before retrieval
    search_query = reformulate_query(query, chat_history)

    # Use reformulated query for retrieval
    docs_and_scores = vectorstore.similarity_search_with_score(search_query, k=5)
    relevant = [doc for doc, score in docs_and_scores if score < 1.5]

    if not relevant:
        print("🤖 I don't have any relevant information about that in my knowledge base.")
        return

    context = "\n\n".join([doc.page_content for doc in relevant])

    system_prompt = f"""You are a personal knowledge assistant.
Answer the user's question using ONLY the context below.
If the answer is not in the context, say "I don't have that information."
Be concise and precise.

Context:
{context}"""

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": query})

    response = llm.invoke(messages)
    answer = response.content

    chat_history.append({"role": "user", "content": query})
    chat_history.append({"role": "assistant", "content": answer})

    print(f"🤖 {answer}")

    print("\n📄 Sources:")
    seen = set()
    for doc in relevant:
        preview = doc.page_content[:80].replace("\n", " ")
        if preview not in seen:
            seen.add(preview)
            print(f"  → {preview}...")

# Test conversation
print("=" * 60)
ask("what are the three container types in Roaring Bitmaps?")
ask("which one is best for sparse data?")
ask("and what about dense data?")
ask("summarize what we just discussed")