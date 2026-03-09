"""
Data Ingestion Pipeline for Attention Is All You Need Paper

This script implements the complete data ingestion pipeline:
1. Load PDF document (Attention Is All You Need paper)
2. Transform PDF content into structured graph representation
3. Serialize graphs to JSON for inspection
4. Ingest structured data into Neo4j knowledge graph

The resulting knowledge graph contains:
- Entities: Organizations, People, Publications, Concepts, Techniques
- Relationships: Connects entities showing dependencies and relationships
- Properties: Each entity has an ID and type label

Usage:
    python ingest_attention.py
    
Output:
    - data/preprocessed/graph_data.json: Serialized graph structure
    - Neo4j database: Populated with graph entities and relationships

Author: GraphRAG Research Team
Date: 2026
"""

from typing import List, Dict, Any
from graphrag.ingestion import GraphDocumentGenerator, Neo4jGraphIngestor
from graphrag.config import Config
import json
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

cfg = Config()

# Step 1: Load PDF using PyPDFLoader
print("Loading PDF documents...")
pdf_path = 'data/multiModalPDF/Attention_Is_All_You_Need_RP.pdf'
loader = PyPDFLoader(pdf_path)
documents = loader.load()
print(f"✓ Loaded {len(documents)} pages from PDF")

# Step 2: Generate graphs
print("Transforming to graph...")
generator = GraphDocumentGenerator(cfg.model.llm_model)
graph_documents = generator.transform_to_graph(documents)
print(f"✓ Generated {len(graph_documents)} graph documents")

# Step 3: Serialize graphs
print("Serializing graph...")
serialized = generator.serialize_graph_documents(graph_documents)
print(f"✓ Serialized {len(serialized)} items")

# Step 4: Save to file
output_path = 'data/preprocessed/graph_data.json'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(serialized, f, indent=2)
print(f"✓ Saved to {output_path}")

# Compute stats
total_nodes = sum(len(item['nodes']) for item in serialized)
total_rels = sum(len(item['relationships']) for item in serialized)
print(f"  Total nodes: {total_nodes}, Total relationships: {total_rels}")

if total_nodes == 0:
    print("\n⚠️ WARNING: No nodes were extracted! This might mean:")
    print("  1. OpenAI API key is invalid or missing")
    print("  2. Documents are empty or not being processed correctly")
    print("  3. LLM extraction failed for the PDF content")
    print("\nSkipping Neo4j ingestion.")
else:
    # Step 5: Ingest into Neo4j
    print("\nIngesting into Neo4j...")
    try:
        ingestor = Neo4jGraphIngestor()
        result = ingestor.ingest_graph_data(output_path)
        if result.get('success'):
            print(f"✅ Ingested into Neo4j successfully!")
            print(f"   Documents: {result.get('num_documents', 'N/A')}")
            print(f"   Total nodes: {result.get('total_nodes', 'N/A')}")
            print(f"   Total relationships: {result.get('total_relationships', 'N/A')}")
        else:
            print(f"❌ Ingestion failed: {result.get('error')}")
    except Exception as e:
        print(f"❌ Error during ingestion: {str(e)}")
