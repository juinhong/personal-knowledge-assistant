"""
rag.py — Full RAG pipeline with memory and query reformulation.
"""

from langchain_openai import ChatOpenAI
from src.retriever import retrieve, format_sources
import tiktoken

LLM_MODEL = "gpt-4o-mini"
SYSTEM_PROMPT_TEMPLATE = """You are a personal knowledge assistant.
Answer the user's question using ONLY the context below.
If the answer is not in the context, say "I don't have that information in my knowledge base."
Be concise and precise. Use bullet points or numbered lists when the question asks for multiple items.
If the question is ambiguous, ask for clarification.

Context:
{context}"""


class RAGPipeline:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
        self.chat_history = []

    # Add this at the top of RAGPipeline class
    def _count_tokens(self, messages: list) -> int:
        """Count total tokens in a messages list."""
        encoder = tiktoken.encoding_for_model("gpt-4o-mini")
        total = 0
        for message in messages:
            # Every message has ~4 token overhead
            total += 4
            total += len(encoder.encode(message["content"]))
        return total

    def _reformulate_query(self, query: str) -> str:
        """Rewrite follow-up questions to be self-contained using chat history."""
        if not self.chat_history:
            return query

        history_text = "\n".join([
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in self.chat_history[-4:]  # last 2 exchanges
        ])

        prompt = f"""Given this conversation history:
{history_text}

Rewrite this follow-up question to be fully self-contained and specific,
so it can be understood without the conversation history.
Return ONLY the rewritten question, nothing else.

Follow-up question: {query}"""

        response = self.llm.invoke([{"role": "user", "content": prompt}])
        return response.content.strip()

    def _summarize_history(self) -> list:
        """Compress chat history into a single summary message."""
        history_text = "\n".join([
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in self.chat_history
        ])

        prompt = f"""Summarize this conversation concisely in 3-5 sentences.
Capture the key topics discussed and important answers given.
Return ONLY the summary, nothing else.

Conversation:
{history_text}"""

        response = self.llm.invoke([{"role": "user", "content": prompt}])
        summary = response.content.strip()
        return [{"role": "system", "content": f"Previous conversation summary: {summary}"}]

    def ask(self, query: str, verbose: bool = True) -> dict:
        """
        Ask a question and get an answer from the RAG pipeline.
        Returns dict with 'answer' and 'sources'.
        """
        # Summarize history if it's getting long (> 6 messages = 3 exchanges)
        if len(self.chat_history) >= 6:
            if verbose:
                print("📝 Summarizing conversation history...")
            self.chat_history = self._summarize_history()

        # Reformulate query using history
        search_query = self._reformulate_query(query)
        if verbose and search_query != query:
            print(f"🔄 Reformulated: {search_query}")

        # Retrieve relevant chunks
        docs_and_scores = retrieve(self.vectorstore, search_query)

        if not docs_and_scores:
            return {
                "answer": "I don't have any relevant information about that in my knowledge base.",
                "sources": ""
            }

        # Build context from chunks
        context = "\n\n".join([doc.page_content for doc, _ in docs_and_scores])

        # Build messages with memory
        messages = [{"role": "system", "content": SYSTEM_PROMPT_TEMPLATE.format(context=context)}]
        messages.extend(self.chat_history)
        messages.append({"role": "user", "content": query})

        # Count tokens before sending
        token_count = self._count_tokens(messages)
        token_limit = 128000  # gpt-4o-mini context window
        print(f"📊 Tokens in prompt: {token_count:,} / {token_limit:,} ({token_count / token_limit * 100:.1f}%)")

        if token_count > token_limit * 0.8:  # warn at 80%
            print("⚠️  Approaching token limit — consider summarizing history")

        # Generate answer
        response = self.llm.invoke(messages)
        answer = response.content

        # Update memory
        self.chat_history.append({"role": "user", "content": query})
        self.chat_history.append({"role": "assistant", "content": answer})

        return {
            "answer": answer,
            "sources": format_sources(docs_and_scores)
        }

    def reset(self):
        """Clear conversation history."""
        self.chat_history = []
        print("🔄 Conversation history cleared.")
