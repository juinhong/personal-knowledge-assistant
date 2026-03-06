"""
main.py — CLI entrypoint for the Personal Knowledge Assistant.

Usage:
    python main.py          # start chatting
    python main.py --ingest # re-ingest documents first, then chat
"""

import sys
from dotenv import load_dotenv
from src.retriever import load_vectorstore
from src.rag import RAGPipeline

load_dotenv()

BANNER = """
╔══════════════════════════════════════════╗
║     🤖 Personal Knowledge Assistant      ║
║                                          ║
║  Commands:                               ║
║    'quit'  — exit                        ║
║    'reset' — clear conversation history  ║
║    'help'  — show this message           ║
╚══════════════════════════════════════════╝
"""


def main():
    # Optional: re-ingest before chatting
    if "--ingest" in sys.argv:
        from src.ingest import ingest
        ingest()
        print()

    print(BANNER)

    # Load vectorstore
    try:
        vectorstore = load_vectorstore()
        chunk_count = vectorstore._collection.count()
        print(f"✅ Loaded knowledge base ({chunk_count} chunks)\n")
    except Exception as e:
        print("❌ No knowledge base found. Run: python main.py --ingest")
        sys.exit(1)

    # Start RAG pipeline
    rag = RAGPipeline(vectorstore)

    # Chat loop
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye! 👋")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("Bye! 👋")
            break

        if user_input.lower() == "reset":
            rag.reset()
            continue

        if user_input.lower() == "help":
            print(BANNER)
            continue

        result = rag.ask(user_input)

        print(f"\n🤖 {result['answer']}")

        if result["sources"]:
            print(f"\n📄 Sources:\n{result['sources']}")

        print()


# if __name__ == "__main__":
#     main()

# Add this temporary test at the bottom of main.py
if __name__ == "__main__" and "--test-tokens" in sys.argv:
    from src.retriever import load_vectorstore
    from src.rag import RAGPipeline

    vectorstore = load_vectorstore()
    rag = RAGPipeline(vectorstore)

    # Simulate a long conversation
    questions = [
        "what are the three container types in Roaring Bitmaps?",
        "explain array containers in detail",
        "what about bitmap containers?",
        "and run containers?",
        "how do AND operations work?",
        "what about OR operations?",
        "summarize all set operations",
        "what are the best practices?",
    ]

    for q in questions:
        print(f"\n{'=' * 50}")
        result = rag.ask(q)
        print(f"🤖 {result['answer'][:100]}...")
        print(f"💬 History length: {len(rag.chat_history)} messages")
