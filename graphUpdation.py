import os
import json
from langchain_neo4j import Neo4jGraph
from langchain_community.graphs.graph_document import Node, Relationship, GraphDocument
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# 1. Initialize Neo4j Connection
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD")
)

# 2. Load the preprocessed data
input_file = "data/preprocessed/graph_data.json"

if not os.path.exists(input_file):
    print(f"File {input_file} not found. Please run the preprocessing script first.")
else:
    print("Process Initiated....")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 3. Reconstruct GraphDocument objects
    reconstructed_docs = []
    for item in data:
        # Recreate Nodes
        nodes = [Node(id=n['id'], type=n['type'], properties=n['properties']) for n in item['nodes']]

        # Helper to find node object by ID for relationships
        node_map = {node.id: node for node in nodes}

        # Recreate Relationships
        relationships = []
        for r in item['relationships']:
            # Use .get() to avoid KeyError. If the node is missing, create a generic one.
            source_node = node_map.get(r['source'], Node(id=r['source'], type="Entity"))
            target_node = node_map.get(r['target'], Node(id=r['target'], type="Entity"))

            relationships.append(
                Relationship(
                    source=source_node,
                    target=target_node,
                    type=r['type'],
                    properties=r.get('properties', {})
                )
            )
        # Recreate the Source Document
        source_doc = Document(page_content="", metadata=item['source'])

        reconstructed_docs.append(
            GraphDocument(nodes=nodes, relationships=relationships, source=source_doc)
        )

    # 4. Update Neo4j
    # include_source=True will link nodes to the original document metadata
    # baseEntityLabel=True adds an '__Entity__' label for faster indexing
    graph.add_graph_documents(
        reconstructed_docs,
        baseEntityLabel=True,
        include_source=True
    )

    print(f"Successfully updated Neo4j with {len(reconstructed_docs)} documents.")