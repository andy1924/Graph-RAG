"""
Smoke test for the NaiveRAG pipeline.

Usage:
    python test_naiverag.py [--ingest] [--question "your question"]

Flags:
    --ingest       Run ingestion on the Attention PDF (default: skip if already done)
    --question     Custom question string (default: Transformer architecture question)
"""

import argparse
import logging
import os
import sys

# Ensure src/ is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from naiverag import ingest_pdf, NaiveRAGRetriever, config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("test_naiverag")


PDF_PATH = os.path.join(
    os.path.dirname(__file__),
    "data", "multiModalPDF", "Attention_Is_All_You_Need_RP.pdf"
)

DEFAULT_QUESTION = "What are the main characteristics of the Transformer architecture?"


def run_ingestion():
    print("\n" + "=" * 60)
    print("STEP 1: Ingesting PDF into ChromaDB")
    print("=" * 60)

    if not os.path.exists(PDF_PATH):
        print(f"[ERROR] PDF not found at: {PDF_PATH}")
        sys.exit(1)

    num_chunks = ingest_pdf(PDF_PATH, config)
    print(f"\n[+] Ingestion complete — {num_chunks} chunks stored in ChromaDB.")
    print(f"    Collection : '{config.collection_name}'")
    print(f"    Persist dir: '{config.chroma_dir}'")
    return num_chunks


def run_retrieval(question: str):
    print("\n" + "=" * 60)
    print("STEP 2: Retrieving answer from ChromaDB")
    print("=" * 60)
    print(f"\nQuestion: {question}\n")

    retriever = NaiveRAGRetriever(config)

    # --- Context-only lookup ---
    context, sources = retriever.get_context(question, k=config.top_k)
    print("[+] Top retrieved chunks:")
    for i, src in enumerate(sources, 1):
        print(f"  {i}. {src}")

    # --- Full QA ---
    answer, meta = retriever.answer_question(question)
    print(f"\n[+] Answer ({meta['response_time']:.3f}s):\n")
    print(answer)
    print(f"\n[+] Chunks used: {meta['num_chunks_retrieved']}")
    return answer, meta


def main():
    parser = argparse.ArgumentParser(description="NaiveRAG smoke test")
    parser.add_argument("--ingest", action="store_true",
                        help="Run ingestion before retrieval")
    parser.add_argument("--question", type=str, default=DEFAULT_QUESTION,
                        help="Question to answer")
    args = parser.parse_args()

    if args.ingest:
        run_ingestion()

    run_retrieval(args.question)


if __name__ == "__main__":
    main()
