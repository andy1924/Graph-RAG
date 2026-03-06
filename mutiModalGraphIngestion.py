import os
import json
from langchain_neo4j import Neo4jGraph
from langchain_community.graphs.graph_document import Node, Relationship, GraphDocument
from langchain_core.documents import Document
from dotenv import load_dotenv
load_dotenv()
def connect_to_graph():
    """Initializes and returns the Neo4j connection."""
    return Neo4jGraph(
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
        database = "8c3134ef"
    )
def reconstruct_multimodal_graph(data):
    """
    Transforms the multimodal JSON payload into LangChain GraphDocuments.
    Ensures sequential logic via PRECEDES relationships.
    """
    reconstructed_docs = []
    prev_node = None  # Used for topological linking
    for item in data:
        # 1. Create the Multimodal Node (Chunk, Table, or Image)
        # Using item['content'] which contains our LLM summary
        current_node = Node(
            id=item['id'],
            type=item.get('type', 'Chunk'),
            properties={
                "content": item['content'],
                "page_number": item.get('metadata', {}).get('page_number', 'unknown')
            }
        )
        nodes = [current_node]
        relationships = []
        # 2. Add Topological/Sequential relationship (Addressing Structural Gap)
        if prev_node:
            relationships.append(
                Relationship(
                    source=prev_node,
                    target=current_node,
                    type="PRECEDES"
                )
            )
        # 3. Create Source Document object
        source_doc = Document(
            page_content=item['content'],
            metadata=item.get('metadata', {})
        )
        # 4. Wrap into GraphDocument
        reconstructed_docs.append(
            GraphDocument(nodes=nodes, relationships=relationships, source=source_doc)
        )
        # Update pointer for next iteration
        prev_node = current_node
    return reconstructed_docs
def update_neo4j_store(graph, documents):
    """Commits the GraphDocuments to the Neo4j instance."""
    # baseEntityLabel=True ensures the '__Entity__' label is applied
    # include_source=True links elements to the root Document node
    graph.add_graph_documents(
        documents,
        baseEntityLabel=True,
        include_source=True
    )
def main():
    # File Path
    input_file = "data/preprocessed/graph_data.json"
    # 1. Validation
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Please run your preprocessing script.")
        return
    # 2. Init Connection
    print("Connecting to Neo4j...")
    try:
        graph = connect_to_graph()
    except Exception as e:
        print(f"Connection Failed: {e}")
        return
    # 3. Load Data
    print("Loading Preprocessed JSON...")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 4. Transform and Update
    print(f"Reconstructing graph for {len(data)} elements...")
    try:
        reconstructed_docs = reconstruct_multimodal_graph(data)
        print("Pushing to Neo4j (Updating Schema)...")
        update_neo4j_store(graph, reconstructed_docs)
        print(f"--- SUCCESS: Neo4j updated with {len(reconstructed_docs)} multimodal nodes ---")
    except Exception as e:
        print(f"Update failed: {e}")
if __name__ == "__main__":
    main()