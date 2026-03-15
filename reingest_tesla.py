"""
Re-ingest already generated graph JSON into Neo4j.
"""

import sys
import os

# Ensure src is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from graphrag.ingestion import Neo4jGraphIngestor

def main():
    json_path = "data/preprocessed/tesla_graph_data.json"
    if not os.path.exists(json_path):
        print(f"❌ File not found: {json_path}")
        sys.exit(1)

    print(f"Ingesting {json_path} into Neo4j...")
    try:
        ingestor = Neo4jGraphIngestor()
        result = ingestor.ingest_graph_data(json_path)
        if result.get("success"):
            print(f"✅ Ingested Tesla into Neo4j successfully!")
            print(f"   Documents: {result.get('num_documents', 'N/A')}")
            print(f"   Total nodes: {result.get('total_nodes', 'N/A')}")
            print(f"   Total relationships: {result.get('total_relationships', 'N/A')}")
        else:
            print(f"❌ Ingestion failed: {result.get('error')}")
    except Exception as e:
        print(f"❌ Error during ingestion: {e}")

if __name__ == "__main__":
    main()
