import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from graphrag.retrieval import get_graph_context
from graphrag.config import config
from neo4j import GraphDatabase
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Use config values from environment/config
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
driver = GraphDatabase.driver(
    config.neo4j.uri,
    auth=(config.neo4j.username, config.neo4j.password)
)

question = "What was Tesla's total revenue in 2024?"
context, sources, nodes, relations = get_graph_context(
    question, client, driver, config.neo4j.database
)

print("=" * 80)
print("QUESTION:", question)
print("=" * 80)
print("\nRETRIEVED NODES:", nodes[:10] if nodes else [])
print("\nSAMPLE RELATIONS:")
if relations:
    for r in relations[:5]:
        print(f"  {r}")
else:
    print("  No relations found.")

print("\nFINAL CONTEXT:")
print(context[:1000] if context else "No context retrieved.")

driver.close()
