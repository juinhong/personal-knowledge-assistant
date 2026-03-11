"""
eval_answers.py — Evaluate answer quality using LLM-as-judge.

Usage: python -m tests.eval_answers
"""

import json
from dotenv import load_dotenv
from openai import OpenAI
from src.retriever import load_vectorstore
from src.rag import RAGPipeline

load_dotenv()

EVAL_PATH = "tests/eval_set.json"
client = OpenAI()


def judge_answer(question: str, expected: str, actual: str, qtype: str) -> dict:
    """Use GPT to judge whether the actual answer matches the expected answer."""

    prompt = f"""You are an expert evaluator for a RAG (Retrieval-Augmented Generation) system.

Your job is to judge whether the actual answer correctly addresses the question
based on the expected answer as ground truth.

Question: {question}
Expected answer: {expected}
Actual answer: {actual}

Evaluate the actual answer on these criteria:
1. Correctness — does it contain the key facts from the expected answer?
2. Completeness — does it cover all important points?
3. Hallucination — does it add false information not in the expected answer?

Respond ONLY with a JSON object in this exact format:
{{
  "score": <0-3>,
  "correctness": "<pass|fail>",
  "completeness": "<pass|partial|fail>",
  "hallucination": "<none|minor|major>",
  "reason": "<one sentence explanation>"
}}

Scoring guide:
3 = correct, complete, no hallucination
2 = mostly correct, minor gaps, no hallucination
1 = partially correct or incomplete
0 = wrong, missing key facts, or hallucinated"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if present
    clean = raw.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        return {
            "score": 0,
            "correctness": "fail",
            "completeness": "fail",
            "hallucination": "unknown",
            "reason": f"Could not parse judge response: {raw}"
        }


def evaluate_answers():
    print("🔍 Loading vectorstore and RAG pipeline...")
    vectorstore = load_vectorstore()
    rag = RAGPipeline(vectorstore)

    print(f"📋 Loading eval set from {EVAL_PATH}...\n")
    with open(EVAL_PATH) as f:
        eval_set = json.load(f)

    total_score = 0
    max_score = 0
    results = []

    print("=" * 60)

    for item in eval_set:
        qid = item["id"]
        question = item["question"]
        expected = item["expected"]
        qtype = item["type"]

        # Get actual answer from RAG
        rag.reset()  # fresh history per question
        result = rag.ask(question, verbose=False)
        actual = result["answer"]

        # Judge it
        judgment = judge_answer(question, expected, actual, qtype)
        score = judgment.get("score", 0)
        total_score += score
        max_score += 3

        # Display
        emoji = "✅" if score == 3 else "🟡" if score == 2 else "🟠" if score == 1 else "❌"
        print(f"[{qid:02d}] {emoji} Score: {score}/3 | {qtype.upper()}")
        print(f"      Q: {question}")
        print(f"      Expected:  {expected[:100]}...")
        print(f"      Actual:    {actual[:100]}...")
        print(f"      Correct: {judgment['correctness']} | "
              f"Complete: {judgment['completeness']} | "
              f"Hallucination: {judgment['hallucination']}")
        print(f"      Reason: {judgment['reason']}")
        print()

        results.append({
            "id": qid,
            "question": question,
            "expected": expected,
            "actual": actual,
            "judgment": judgment
        })

    # Summary
    pct = total_score / max_score * 100
    print("=" * 60)
    print(f"📊 ANSWER QUALITY RESULTS")
    print(f"   Total score:   {total_score}/{max_score}")
    print(f"   Quality score: {pct:.1f}%")

    # Breakdown by type
    by_type = {}
    for r in results:
        qtype = next(i["type"] for i in eval_set if i["id"] == r["id"])
        by_type.setdefault(qtype, []).append(r["judgment"]["score"])

    print(f"\n   By question type:")
    for qtype, scores in by_type.items():
        avg = sum(scores) / len(scores)
        print(f"   {qtype:15} avg: {avg:.1f}/3")

    if pct >= 80:
        print(f"\n🎉 Answer quality is GOOD (>= 80%)")
    elif pct >= 60:
        print(f"\n⚠️  Answer quality is FAIR — tune your prompt or chunking")
    else:
        print(f"\n❌ Answer quality is POOR — review retrieval and prompt")

    # Save results
    with open("tests/eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Full results saved to tests/eval_results.json")


if __name__ == "__main__":
    evaluate_answers()
