#!/usr/bin/env python3
"""
Interactive Query Interface for GraphRAG

Usage:
    python scripts/query.py --mode graphrag
    python scripts/query.py --mode naiverag
    python scripts/query.py --mode both
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the GraphRAG system")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["graphrag", "naiverag", "both"],
        default="graphrag",
        help="Query mode (default: graphrag)",
    )

    args = parser.parse_args()

    retriever_graph = None
    retriever_naive = None

    if args.mode in ("graphrag", "both"):
        from graphrag.retrieval import GraphRetriever

        retriever_graph = GraphRetriever()
        print("✓ GraphRAG retriever loaded")

    if args.mode in ("naiverag", "both"):
        from naiverag.retrieval import NaiveRAGRetriever

        retriever_naive = NaiveRAGRetriever()
        print("✓ NaiveRAG retriever loaded")

    print(f"\nGraphRAG Query Interface  [{args.mode} mode]")
    print("=" * 60)
    print("Type your question, or 'quit' to exit.\n")

    while True:
        try:
            question = input("Question: ").strip()
            if not question or question.lower() in ("quit", "exit", "q"):
                break

            if retriever_graph:
                answer, _ = retriever_graph.answer_question(question)
                print(f"\n  GraphRAG → {answer}")

            if retriever_naive:
                result = retriever_naive.answer(question)
                print(f"\n  NaiveRAG → {result['answer']}")

            print()

        except KeyboardInterrupt:
            break

    print("\nGoodbye!")


if __name__ == "__main__":
    main()
