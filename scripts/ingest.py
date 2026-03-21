#!/usr/bin/env python3
"""
Data Ingestion Script for GraphRAG

Ingests documents into Neo4j (GraphRAG) and ChromaDB (NaiveRAG).

Usage:
    python scripts/ingest.py --corpus attention_paper
    python scripts/ingest.py --corpus tesla --target naiverag
    python scripts/ingest.py --all
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv()


# Corpus → source file mapping
CORPUS_FILES = {
    "attention_paper": ("data/multiModalPDF/Attention_Is_All_You_Need_RP.pdf", "pdf"),
    "tesla": ("data/raw/Tesla.txt", "text"),
    "google": ("data/raw/Google.txt", "text"),
    "spacex": ("data/raw/SpaceX.txt", "text"),
}


def ingest_graphrag(corpus_id: str) -> None:
    """Ingest a corpus into Neo4j for GraphRAG."""
    filepath, _ = CORPUS_FILES[corpus_id]
    filepath = str(Path(__file__).resolve().parent.parent / filepath)

    if not os.path.exists(filepath):
        print(f"  ✗ Source file not found: {filepath}")
        return

    from graphrag.ingestion.multimodal_ingestion import MultimodalIngestion

    ingestion = MultimodalIngestion()
    print(f"  → Ingesting into Neo4j...")
    ingestion.ingest(filepath)
    enrich_result = ingestion.ingestor.enrich_quantitative_properties()
    if enrich_result.get("success"):
        print(
            "  → Quantitative enrichment: "
            f"complexity={enrich_result.get('complexity_updates', 0)}, "
            f"BLEU={enrich_result.get('bleu_property_updates', 0)}"
        )
    else:
        print(f"  ! Quantitative enrichment failed: {enrich_result.get('error')}")
    print(f"  ✓ Neo4j ingestion complete")


def ingest_naiverag(corpus_id: str) -> None:
    """Ingest a corpus into ChromaDB for NaiveRAG."""
    filepath, ftype = CORPUS_FILES[corpus_id]
    filepath = str(Path(__file__).resolve().parent.parent / filepath)

    if not os.path.exists(filepath):
        print(f"  ✗ Source file not found: {filepath}")
        return

    if ftype == "pdf":
        from naiverag.ingestion import ingest_pdf

        print(f"  → Ingesting PDF into ChromaDB...")
        count = ingest_pdf(filepath)
    else:
        from naiverag.ingestion import ingest_text_file

        print(f"  → Ingesting text into ChromaDB...")
        count = ingest_text_file(filepath)

    print(f"  ✓ Stored {count} chunks in ChromaDB")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest data into GraphRAG / NaiveRAG"
    )
    parser.add_argument(
        "--corpus",
        type=str,
        choices=list(CORPUS_FILES.keys()),
        help="Corpus to ingest",
    )
    parser.add_argument("--all", action="store_true", help="Ingest all corpora")
    parser.add_argument(
        "--target",
        type=str,
        choices=["graphrag", "naiverag", "both"],
        default="both",
        help="Target system (default: both)",
    )

    args = parser.parse_args()

    if args.all:
        corpora = list(CORPUS_FILES.keys())
    elif args.corpus:
        corpora = [args.corpus]
    else:
        parser.print_help()
        return

    for corpus_id in corpora:
        print(f"\n{'='*60}")
        print(f"  Corpus: {corpus_id}")
        print(f"{'='*60}")

        if args.target in ("graphrag", "both"):
            ingest_graphrag(corpus_id)
        if args.target in ("naiverag", "both"):
            ingest_naiverag(corpus_id)

    print("\n✓ All ingestion complete.")


if __name__ == "__main__":
    main()
