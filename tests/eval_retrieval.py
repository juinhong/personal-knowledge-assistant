"""
eval_retrieval.py — Evaluate whether the right chunks are being retrieved
for each question in the eval set.

Usage: python -m tests.eval_retrieval
"""

import json
from dotenv import load_dotenv
from src.retriever import load_vectorstore, retrieve

load_dotenv()

EVAL_PATH = "tests/eval_set.json"
PASS_THRESHOLD = 0.8  # 80% keyword match = retrieval pass


def keywords_found(chunks, expected: str) -> tuple[bool, list[str], list[str]]:
    """
    Check if key terms from the expected answer appear in retrieved chunks.
    Returns (passed, found_keywords, missing_keywords)
    """
    # Extract meaningful keywords from expected answer (ignore short words)
    stopwords = {"a", "an", "the", "is", "are", "for", "of", "in", "to",
                 "and", "or", "it", "its", "i", "you", "that", "this",
                 "with", "at", "by", "from", "only", "not", "need"}

    words = expected.lower().replace(",", "").replace(".", "").split()
    keywords = [w for w in words if w not in stopwords and len(w) > 3]
    keywords = list(set(keywords))  # dedupe

    # Combine all retrieved chunk text
    all_text = " ".join([doc.page_content.lower() for doc in chunks])

    found = [kw for kw in keywords if kw in all_text]
    missing = [kw for kw in keywords if kw not in all_text]

    score = len(found) / len(keywords) if keywords else 0
    passed = score >= PASS_THRESHOLD

    return passed, found, missing


def evaluate_retrieval():
    print("🔍 Loading vectorstore...")
    vectorstore = load_vectorstore()

    print(f"📋 Loading eval set from {EVAL_PATH}...\n")
    with open(EVAL_PATH) as f:
        eval_set = json.load(f)

    results = []
    passed = 0
    skipped = 0

    print("=" * 60)

    for item in eval_set:
        qid = item["id"]
        question = item["question"]
        expected = item["expected"]
        qtype = item["type"]

        # Skip out-of-scope questions — retrieval should return nothing
        if qtype == "out_of_scope":
            docs_and_scores = retrieve(vectorstore, question)
            retrieved_count = len(docs_and_scores)
            status = "✅ PASS" if retrieved_count == 0 else f"⚠️  WARN ({retrieved_count} chunks retrieved, expected 0)"
            print(f"[{qid:02d}] {qtype.upper()}: {question}")
            print(f"      {status}\n")
            skipped += 1
            if retrieved_count == 0:
                passed += 1
            continue

        # Retrieve chunks
        docs_and_scores = retrieve(vectorstore, question)
        docs = [doc for doc, _ in docs_and_scores]

        if not docs:
            print(f"[{qid:02d}] ❌ FAIL — No chunks retrieved")
            print(f"      Question: {question}\n")
            results.append({"id": qid, "passed": False, "reason": "no chunks retrieved"})
            continue

        # Check keyword coverage
        passed_check, found, missing = keywords_found(docs, expected)

        status = "✅ PASS" if passed_check else "❌ FAIL"
        print(f"[{qid:02d}] {status} | {qtype.upper()} | {question}")
        print(f"      Chunks retrieved: {len(docs)}")
        print(f"      Keywords found:   {found}")

        if missing:
            print(f"      Keywords missing: {missing}")

        print()

        results.append({
            "id": qid,
            "passed": passed_check,
            "found": found,
            "missing": missing
        })

        if passed_check:
            passed += 1

    # Summary
    total = len(eval_set)
    in_scope = total - skipped
    retrieval_pass_rate = passed / total * 100

    print("=" * 60)
    print(f"📊 RETRIEVAL EVAL RESULTS")
    print(f"   Total questions:     {total}")
    print(f"   In-scope:            {in_scope}")
    print(f"   Out-of-scope:        {skipped}")
    print(f"   Passed:              {passed}/{total}")
    print(f"   Pass rate:           {retrieval_pass_rate:.1f}%")

    if retrieval_pass_rate >= 80:
        print(f"\n🎉 Retrieval quality is GOOD (>= 80%)")
    elif retrieval_pass_rate >= 60:
        print(f"\n⚠️  Retrieval quality is FAIR — consider tuning chunk size or k")
    else:
        print(f"\n❌ Retrieval quality is POOR — review chunking and document quality")

    return results


if __name__ == "__main__":
    evaluate_retrieval()
