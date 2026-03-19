"""Quick smoke test for the apostrophe fix."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from graphrag.retrieval import get_graph_context
from openai import OpenAI
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)
database = os.getenv("NEO4J_DATABASE")

# Test with apostrophe-containing questions
test_questions = [
    "What are Tesla's main product lines?",
    "What was Google's revenue growth?",
]

for q in test_questions:
    print(f"\nQ: {q}")
    context, sources, nodes, relations = get_graph_context(q, client, driver, database)
    if "Error" in context:
        print(f"  *** ERROR: {context[:200]}")
    else:
        print(f"  OK: {len(context)} chars, {len(nodes)} nodes, {len(relations)} relations")
        print(f"  Context preview: {context[:200]}")

driver.close()
print("\nDone!")
