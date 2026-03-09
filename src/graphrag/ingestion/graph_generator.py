"""
Graph document generation from raw text sources using LLM-based extraction.
Converts unstructured documents into structured knowledge graphs.
"""

import os
import json
import logging
from typing import List, Dict, Any
from pathlib import Path

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer

from ..config import config
from ..utils import get_logger

logger = get_logger(__name__)


class GraphDocumentGenerator:
    """
    Generates graph documents from raw text files using LLM-based extraction.
    
    Research Paper Optimization:
    - Uses LLMGraphTransformer for structured extraction
    - Preserves relationship types for multimodal linking
    - Stores metadata for evaluation traceability
    """
    
    def __init__(self, llm_model: str = None):
        """
        Initialize the graph document generator.
        
        Args:
            llm_model: LLM model to use (default: from config)
        """
        self.llm_model = llm_model or config.model.llm_model
        self.llm = ChatOpenAI(model=self.llm_model)
        self.transformer = LLMGraphTransformer(llm=self.llm)
        self.logger = get_logger(self.__class__.__name__)
    
    def load_documents(self, input_path: str, glob_pattern: str = "*.txt") -> List:
        """
        Load documents from a directory.
        
        Args:
            input_path: Path to directory containing documents
            glob_pattern: File pattern to match
        
        Returns:
            List of loaded documents
        """
        loader = DirectoryLoader(
            path=input_path,
            glob=glob_pattern,
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )
        
        documents = loader.load()
        self.logger.info(f"Loaded {len(documents)} documents from {input_path}")
        return documents
    
    def transform_to_graph(self, documents: List) -> List:
        """
        Transform documents into graph documents.
        
        Args:
            documents: List of documents to transform
        
        Returns:
            List of GraphDocument objects
        """
        self.logger.info(f"Transforming {len(documents)} documents into graphs...")
        graph_documents = self.transformer.convert_to_graph_documents(documents)
        self.logger.info(f"Generated {len(graph_documents)} graph documents")
        return graph_documents
    
    def serialize_graph_documents(self, graph_documents: List) -> List[Dict[str, Any]]:
        """
        Serialize graph documents to JSON-compatible format.
        
        Args:
            graph_documents: List of GraphDocument objects
        
        Returns:
            List of serialized graph documents
        """
        serializable_data = []
        
        for doc in graph_documents:
            item = {
                "nodes": [
                    {
                        "id": n.id,
                        "type": n.type,
                        "properties": n.properties
                    }
                    for n in doc.nodes
                ],
                "relationships": [
                    {
                        "source": r.source.id,
                        "target": r.target.id,
                        "type": r.type,
                        "properties": r.properties
                    }
                    for r in doc.relationships
                ],
                "source": doc.source.metadata
            }
            serializable_data.append(item)
        
        return serializable_data
    
    def save_graph_data(
        self,
        serializable_data: List[Dict[str, Any]],
        output_path: str
    ) -> str:
        """
        Save serialized graph data to JSON file.
        
        Args:
            serializable_data: Serialized graph documents
            output_path: Path to save the output file
        
        Returns:
            Path to saved file
        """
        output_file = os.path.join(output_path, "graph_data.json")
        os.makedirs(output_path, exist_ok=True)
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(serializable_data, f, indent=4)
        
        self.logger.info(f"Saved {len(serializable_data)} graph documents to {output_file}")
        return output_file
    
    def generate_graphs(
        self,
        input_path: str,
        output_path: str = None
    ) -> Dict[str, Any]:
        """
        Complete pipeline: load, transform, serialize, and save graph documents.
        
        Args:
            input_path: Path to raw documents
            output_path: Path to save processed graphs (default: from config)
        
        Returns:
            Dictionary with pipeline results and metadata
        """
        output_path = output_path or config.ingestion.preprocessed_dir
        
        # Load documents
        documents = self.load_documents(input_path)
        if not documents:
            self.logger.warning(f"No documents found at {input_path}")
            return {"success": False, "error": "No documents loaded"}
        
        # Transform to graphs
        graph_documents = self.transform_to_graph(documents)
        
        # Serialize
        serializable_data = self.serialize_graph_documents(graph_documents)
        
        # Save
        output_file = self.save_graph_data(serializable_data, output_path)
        
        # Compute statistics
        total_nodes = sum(len(item["nodes"]) for item in serializable_data)
        total_relationships = sum(len(item["relationships"]) for item in serializable_data)
        
        return {
            "success": True,
            "output_file": output_file,
            "num_documents": len(documents),
            "num_graph_documents": len(graph_documents),
            "total_nodes": total_nodes,
            "total_relationships": total_relationships,
            "avg_nodes_per_doc": total_nodes / len(graph_documents) if graph_documents else 0,
            "avg_relationships_per_doc": total_relationships / len(graph_documents) if graph_documents else 0,
        }


def main():
    """CLI entry point for graph document generation."""
    import sys
    
    input_path = sys.argv[1] if len(sys.argv) > 1 else config.ingestion.raw_data_dir
    output_path = sys.argv[2] if len(sys.argv) > 2 else config.ingestion.preprocessed_dir
    
    generator = GraphDocumentGenerator()
    results = generator.generate_graphs(input_path, output_path)
    
    if results["success"]:
        print(f"\n✓ Successfully generated graphs!")
        print(f"  Documents processed: {results['num_documents']}")
        print(f"  Graph documents created: {results['num_graph_documents']}")
        print(f"  Total nodes: {results['total_nodes']}")
        print(f"  Total relationships: {results['total_relationships']}")
        print(f"  Output: {results['output_file']}")
    else:
        print(f"✗ Error: {results.get('error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
