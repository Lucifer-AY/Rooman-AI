"""
===============================================================================
CLI RUNNER FOR RESEARCH AGENT (WITH CITATIONS & TAVILY FALLBACK)
===============================================================================
Usage:
  uv run python main.py               # Interactive CLI mode
  uv run python main.py --query "..." # Single query mode
  uv run python main.py --eval        # Run the evaluation benchmark suite
===============================================================================
"""

import sys
import json
import argparse
from pathlib import Path
from agent import ResearchAgent

# UTF-8 terminal support
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def run_evaluation_suite(agent: ResearchAgent):
    """Executes the test questions in questions.json and prints a scorecard."""
    questions_file = Path(__file__).parent / "questions.json"
    if not questions_file.exists():
        print(f"Error: {questions_file} not found.")
        return

    with open(questions_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print("\n=======================================================")
    print(f"  RUNNING RESEARCH AGENT EVALUATION SUITE ({len(questions)} TESTS)")
    print("=======================================================\n")

    passed = 0
    for idx, item in enumerate(questions, 1):
        q_id = item["id"]
        q_type = item["type"]
        question = item["question"]

        print(f"[{idx}/{len(questions)}] Testing {q_id} ({q_type}):")
        print(f"      Q: \"{question}\"")
        
        res = agent.query(question)
        source_type = res.get("source_type", "none")
        citations = res.get("citations", [])
        
        # Verify correctness based on question type
        if q_type == "Local Folder Data":
            is_correct = (source_type == "local" and len(citations) > 0)
        elif q_type == "Tavily Web Search Fallback":
            is_correct = (source_type == "tavily" and len(citations) > 0)
        else:
            # Unanswerable
            is_correct = (source_type == "none" or "insufficient" in res["answer"].lower())

        if is_correct:
            passed += 1
            status_badge = "[PASS]"
        else:
            status_badge = "[FAIL]"

        print(f"      -> Result: {status_badge} | Route: {source_type.upper()} | Citations: {len(citations)}")
        print(f"      -> Answer Preview: {res['answer'][:120]}...\n")

    accuracy = (passed / len(questions)) * 100
    print("-------------------------------------------------------")
    print(f"Benchmark Result: {passed}/{len(questions)} ({accuracy:.1f}%) Passed")
    print("=======================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Research Agent (with Citations & Tavily Fallback)")
    parser.add_argument("-q", "--query", type=str, help="Ask a single question")
    parser.add_argument("-e", "--eval", action="store_true", help="Run the test evaluation benchmark suite")
    args = parser.parse_args()

    print("========================================================")
    print("   RESEARCH AGENT WITH CITATIONS (LangGraph + Tavily)   ")
    print("========================================================")
    print("Local documents loaded from 'data/' folder.")
    print("Tavily Web Search active for queries outside local data.\n")

    agent = ResearchAgent()

    if args.eval:
        run_evaluation_suite(agent)
    elif args.query:
        print(f"Question: {args.query}\n")
        res = agent.query(args.query)
        print("--- Answer ---")
        print(res["answer"])
        print(f"\nRetrieval Source: {res['source_type'].upper()}")
        if res["citations"]:
            print(f"Citations: {', '.join(res['citations'])}")
        print()
    else:
        print("Interactive Mode. Type your questions below (or 'exit' to quit):\n")
        while True:
            try:
                user_q = input("Question > ").strip()
                if not user_q:
                    continue
                if user_q.lower() in ["exit", "quit", "q"]:
                    print("Goodbye!")
                    break

                res = agent.query(user_q)
                print("\n" + "="*50)
                print(res["answer"])
                print("\n" + "-"*50)
                print(f"Source Route: {res['source_type'].upper()}")
                if res["citations"]:
                    print(f"Citations: {', '.join(res['citations'])}")
                print("="*50 + "\n")
            except (KeyboardInterrupt, EOFError):
                break


if __name__ == "__main__":
    main()
