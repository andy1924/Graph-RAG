"""
Loads pre-built graph JSON files directly into Neo4j.
Skips LLM graph generation entirely — zero API cost.
Run this once with Neo4j Desktop open and running.
"""
import sys, os, json
sys.path.insert(0, 'src')
from dotenv import load_dotenv
load_dotenv()

from graphrag.ingestion import Neo4jGraphIngestor

FILES = {
    "Tesla":  "data/preprocessed/tesla_graph_data.json",
    "Google": "data/preprocessed/google_graph_data.json",
    "SpaceX": "data/preprocessed/spacex_graph_data.json",
}

ingestor = Neo4jGraphIngestor()

for label, path in FILES.items():
    print(f"\nIngesting {label} from {path}...")
    result = ingestor.ingest_graph_data(path)
    if result.get("success"):
        print(f"  ✅ {label}: {result['total_nodes']} nodes, "
              f"{result['total_relationships']} rels")
    else:
        print(f"  ❌ {label} FAILED: {result.get('error')}")

# Verify final node count
from neo4j import GraphDatabase
from graphrag.config import Config
cfg = Config()
driver = GraphDatabase.driver(
    cfg.neo4j.uri,
    auth=(cfg.neo4j.username, cfg.neo4j.password)
)
with driver.session(database=cfg.neo4j.database) as s:
    total = s.run("MATCH (n) RETURN count(n) as c").single()["c"]
    print(f"\nTotal nodes now in Neo4j: {total}")
    print("READY ✅" if total > 3000 else
          "⚠️  Low count — check Neo4j connection and re-run")
driver.close()