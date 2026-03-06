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


if __name__ == "__main__":
    main()
