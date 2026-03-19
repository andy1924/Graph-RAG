"""Ingest Tesla, Google, SpaceX text files into ChromaDB for NaiveRAG."""
import sys
import os
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

logging.basicConfig(level=logging.INFO)

from naiverag.ingestion import ingest_text_file

raw_dir = os.path.join(os.path.dirname(__file__), "data", "raw")

files_to_ingest = [
    os.path.join(raw_dir, "Tesla.txt"),
    os.path.join(raw_dir, "Google.txt"),
    os.path.join(raw_dir, "SpaceX.txt"),
]

for filepath in files_to_ingest:
    if os.path.exists(filepath):
        print(f"\n{'='*60}")
        print(f"Ingesting: {filepath}")
        print(f"{'='*60}")
        count = ingest_text_file(filepath)
        print(f"  -> Stored {count} chunks")
    else:
        print(f"SKIP: {filepath} not found")

print("\nDone! All corpora ingested into ChromaDB.")
