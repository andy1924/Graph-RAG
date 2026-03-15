"""
Data Ingestion Pipeline for Tesla and Google Text Corpora

This script implements the complete data ingestion pipeline for the two new
text-based corpora (data/raw/Tesla.txt, data/raw/Google.txt):
  1. Load raw .txt documents
  2. Split into manageable chunks (the files are 200-300 KB each)
  3. Transform chunks into structured graph representation via LLM
  4. Serialize graphs to JSON for inspection
  5. Ingest structured data into Neo4j knowledge graph

The resulting knowledge graph nodes/relationships are ADDED to the same
Neo4j database that already contains the Attention-paper graph.

Usage:
    python ingest_corpora.py                    # ingest both Tesla & Google
    python ingest_corpora.py --corpus tesla      # ingest only Tesla
    python ingest_corpora.py --corpus google      # ingest only Google

Output:
    - data/preprocessed/tesla_graph_data.json
    - data/preprocessed/google_graph_data.json
    - Neo4j database: populated with new graph entities and relationships

Author: GraphRAG Research Team
Date: 2026
"""

import argparse
import json
import os
import sys

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Ensure src is on the path so graphrag package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from graphrag.ingestion import GraphDocumentGenerator, Neo4jGraphIngestor
from graphrag.config import Config

cfg = Config()

# ------------------------------------------------------------------ #
# Corpus definitions
# ------------------------------------------------------------------ #
CORPORA = {
    "tesla": {
        "txt_path": "data/raw/Tesla.txt",
        "json_path": "data/preprocessed/tesla_graph_data.json",
        "label": "Tesla",
    },
    "google": {
        "txt_path": "data/raw/Google.txt",
        "json_path": "data/preprocessed/google_graph_data.json",
        "label": "Google",
    },
}

# ------------------------------------------------------------------ #
# Chunking parameters
# ------------------------------------------------------------------ #
CHUNK_SIZE = 4000        # characters per chunk
CHUNK_OVERLAP = 400      # overlap between chunks


def ingest_corpus(corpus_key: str) -> bool:
    """
    Run the full ingestion pipeline for a single corpus.

    Returns True on success, False on failure.
    """
    meta = CORPORA[corpus_key]
    txt_path = meta["txt_path"]
    json_path = meta["json_path"]
    label = meta["label"]

    print(f"\n{'=' * 60}")
    print(f"  Ingesting corpus: {label}  ({txt_path})")
    print(f"{'=' * 60}")

    # --- Step 1: Load -------------------------------------------------
    if not os.path.exists(txt_path):
        print(f"❌ File not found: {txt_path}")
        return False

    print("Loading text file...")
    loader = TextLoader(txt_path, encoding="utf-8")
    raw_docs = loader.load()
    print(f"✓ Loaded {len(raw_docs)} document(s)  "
          f"({sum(len(d.page_content) for d in raw_docs):,} characters)")

    # --- Step 2: Split into chunks ------------------------------------
    print(f"Splitting into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    documents = splitter.split_documents(raw_docs)

    # Add corpus metadata to each chunk
    for i, doc in enumerate(documents):
        doc.metadata["corpus"] = corpus_key
        doc.metadata["chunk_index"] = i

    print(f"✓ Created {len(documents)} chunks")

    # --- Step 3: Transform to graph -----------------------------------
    print("Transforming chunks to graph (this may take several minutes)...")
    generator = GraphDocumentGenerator(cfg.model.llm_model)
    graph_documents = generator.transform_to_graph(documents)
    print(f"✓ Generated {len(graph_documents)} graph documents")

    # --- Step 4: Serialize --------------------------------------------
    print("Serializing graph...")
    serialized = generator.serialize_graph_documents(graph_documents)
    print(f"✓ Serialized {len(serialized)} items")

    # --- Step 5: Save JSON --------------------------------------------
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serialized, f, indent=2)
    print(f"✓ Saved to {json_path}")

    total_nodes = sum(len(item["nodes"]) for item in serialized)
    total_rels = sum(len(item["relationships"]) for item in serialized)
    print(f"  Total nodes: {total_nodes}, Total relationships: {total_rels}")

    if total_nodes == 0:
        print("\n⚠️  WARNING: No nodes were extracted! Possible causes:")
        print("  1. OpenAI API key is invalid or missing")
        print("  2. LLM extraction failed for the text content")
        print("\nSkipping Neo4j ingestion for this corpus.")
        return False

    # --- Step 6: Ingest into Neo4j ------------------------------------
    print("\nIngesting into Neo4j...")
    try:
        ingestor = Neo4jGraphIngestor()
        result = ingestor.ingest_graph_data(json_path)
        if result.get("success"):
            print(f"✅ Ingested {label} into Neo4j successfully!")
            print(f"   Documents: {result.get('num_documents', 'N/A')}")
            print(f"   Total nodes: {result.get('total_nodes', 'N/A')}")
            print(f"   Total relationships: {result.get('total_relationships', 'N/A')}")
            return True
        else:
            print(f"❌ Ingestion failed: {result.get('error')}")
            return False
    except Exception as e:
        print(f"❌ Error during ingestion: {e}")
        return False


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser(
        description="Ingest Tesla and/or Google text corpora into Neo4j."
    )
    parser.add_argument(
        "--corpus",
        choices=["tesla", "google", "all"],
        default="all",
        help="Which corpus to ingest (default: all)",
    )
    args = parser.parse_args()

    targets = list(CORPORA.keys()) if args.corpus == "all" else [args.corpus]

    results = {}
    for key in targets:
        results[key] = ingest_corpus(key)

    # ---- Summary ----
    print(f"\n{'=' * 60}")
    print("  INGESTION SUMMARY")
    print(f"{'=' * 60}")
    for key, ok in results.items():
        status = "✅ Success" if ok else "❌ Failed"
        print(f"  {CORPORA[key]['label']:10s}  {status}")
    print()


if __name__ == "__main__":
    main()
