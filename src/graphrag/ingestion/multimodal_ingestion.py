"""
Multimodal graph ingestion from PDF documents.
Extracts text, tables, and images with vision-based summarization.
"""

import os
import json
import base64
from typing import List, Dict, Any
from dataclasses import dataclass

from unstructured.partition.pdf import partition_pdf
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from ..config import config
from ..utils import get_logger

logger = get_logger(__name__)


@dataclass
class MultimodalElement:
    """Represents a multimodal element (text, table, or image)."""
    id: str
    type: str  # "text", "table", "image"
    content: str
    metadata: Dict[str, Any]


class MultimodalDocumentProcessor:
    """
    Processes multimodal PDF documents with vision-based understanding.
    
    Research Focus:
    - Extracts multimodal elements (text, tables, images)
    - Uses vision-capable LLM for image/table summarization
    - Maintains structural relationships for graph construction
    """
    
    def __init__(self, vision_model: str = None):
        """
        Initialize multimodal document processor.
        
        Args:
            vision_model: Vision-capable model (default: from config)
        """
        self.vision_model = vision_model or config.model.vision_model
        self.llm = ChatOpenAI(model=self.vision_model)
        self.logger = get_logger(self.__class__.__name__)
    
    def partition_pdf(self, pdf_path: str) -> List[Any]:
        """
        Partition PDF into structured elements.
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            List of partitioned elements
        """
        self.logger.info(f"Partitioning PDF: {pdf_path}")
        
        elements = partition_pdf(
            filename=pdf_path,
            extract_images_in_pdf=config.ingestion.extract_images,
            infer_table_structure=config.ingestion.infer_table_structure,
            chunking_strategy=config.ingestion.chunking_strategy,
            extract_image_block_output_dir=config.ingestion.extracted_images_dir
        )
        
        self.logger.info(f"Extracted {len(elements)} elements from {pdf_path}")
        return elements
    
    def encode_image(self, image_path: str) -> str:
        """
        Encode image to base64 for vision API.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Base64 encoded image string
        """
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def summarize_image(self, image_path: str) -> str:
        """
        Summarize image content using vision model.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Image summary
        """
        try:
            encoded_image = self.encode_image(image_path)
            
            message = self.llm.invoke([
                HumanMessage(content=[
                    {
                        "type": "text",
                        "text": "Describe this image/chart in detail, focusing on key relationships, entities, and information. This will be used to create knowledge graph nodes."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}"
                        }
                    }
                ])
            ])
            
            return message.content
        
        except Exception as e:
            self.logger.error(f"Error summarizing image {image_path}: {str(e)}")
            return f"[Failed to process image: {str(e)}]"
    
    def summarize_table(self, table_content: str) -> str:
        """
        Summarize table content for graph representation.
        
        Args:
            table_content: HTML or text representation of table
        
        Returns:
            Table summary for graph nodes
        """
        try:
            response = self.llm.invoke(
                f"Summarize this table data for knowledge graph representation. "
                f"Focus on identifying rows, columns, and key relationships:\n\n{table_content}"
            )
            return response.content
        
        except Exception as e:
            self.logger.error(f"Error summarizing table: {str(e)}")
            return f"[Failed to process table: {str(e)}]"
    
    def process_document(self, pdf_path: str) -> List[MultimodalElement]:
        """
        Process PDF document and extract multimodal elements.
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            List of MultimodalElement objects
        """
        elements = self.partition_pdf(pdf_path)
        multimodal_elements = []
        
        element_id = 0
        for idx, element in enumerate(elements):
            element_id += 1
            element_type = type(element).__name__
            
            try:
                if "Image" in element_type:
                    # Process image
                    image_path = element.metadata.image_path if hasattr(element, 'metadata') else None
                    if image_path and os.path.exists(image_path):
                        summary = self.summarize_image(image_path)
                        mm_element = MultimodalElement(
                            id=f"img_{element_id}",
                            type="image",
                            content=summary,
                            metadata={
                                "element_index": idx,
                                "page_number": getattr(element.metadata, 'page_number', None),
                                "image_path": image_path
                            }
                        )
                        multimodal_elements.append(mm_element)
                
                elif "Table" in element_type:
                    # Process table
                    table_content = str(element)
                    summary = self.summarize_table(table_content)
                    mm_element = MultimodalElement(
                        id=f"tbl_{element_id}",
                        type="table",
                        content=summary,
                        metadata={
                            "element_index": idx,
                            "page_number": getattr(element.metadata, 'page_number', None),
                            "raw_table": table_content
                        }
                    )
                    multimodal_elements.append(mm_element)
                
                else:
                    # Process text
                    text_content = str(element)
                    mm_element = MultimodalElement(
                        id=f"txt_{element_id}",
                        type="text",
                        content=text_content,
                        metadata={
                            "element_index": idx,
                            "page_number": getattr(element.metadata, 'page_number', None)
                        }
                    )
                    multimodal_elements.append(mm_element)
            
            except Exception as e:
                self.logger.warning(f"Error processing element {idx}: {str(e)}")
                continue
        
        self.logger.info(f"Extracted {len(multimodal_elements)} multimodal elements")
        return multimodal_elements
    
    def to_dict(self, elements: List[MultimodalElement]) -> List[Dict[str, Any]]:
        """Convert MultimodalElement objects to dictionaries."""
        return [
            {
                "id": el.id,
                "type": el.type,
                "content": el.content,
                "metadata": el.metadata
            }
            for el in elements
        ]


class Neo4jGraphIngestor:
    """
    Ingests processed graph data into Neo4j database.
    Handles both preprocessed graphs and multimodal elements.
    """
    
    def __init__(self):
        """Initialize Neo4j ingestor."""
        from langchain_neo4j import Neo4jGraph
        
        self.graph = Neo4jGraph(
            url=config.neo4j.uri,
            username=config.neo4j.username,
            password=config.neo4j.password,
            database=config.neo4j.database
        )
        self.logger = get_logger(self.__class__.__name__)
    
    def ingest_graph_data(self, json_file: str) -> Dict[str, Any]:
        """
        Ingest preprocessed graph data into Neo4j.
        
        Args:
            json_file: Path to preprocessed graph JSON file
        
        Returns:
            Ingestion results
        """
        from langchain_community.graphs.graph_document import Node, Relationship, GraphDocument
        from langchain_core.documents import Document
        
        if not os.path.exists(json_file):
            self.logger.error(f"File not found: {json_file}")
            return {"success": False, "error": "File not found"}
        
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self.logger.info(f"Loading {len(data)} graph documents...")
        
        reconstructed_docs = []
        for item in data:
            nodes = [
                Node(id=n['id'], type=n['type'], properties=n['properties'])
                for n in item['nodes']
            ]
            
            node_map = {node.id: node for node in nodes}
            relationships = []
            
            for r in item['relationships']:
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
            
            source_doc = Document(page_content="", metadata=item['source'])
            reconstructed_docs.append(
                GraphDocument(nodes=nodes, relationships=relationships, source=source_doc)
            )
        
        try:
            self.graph.add_graph_documents(
                reconstructed_docs,
                baseEntityLabel=True,
                include_source=True
            )
            
            self.logger.info(f"Successfully ingested {len(reconstructed_docs)} documents into Neo4j")
            return {
                "success": True,
                "num_documents": len(reconstructed_docs),
                "total_nodes": sum(len(doc.nodes) for doc in reconstructed_docs),
                "total_relationships": sum(len(doc.relationships) for doc in reconstructed_docs)
            }
        
        except Exception as e:
            self.logger.error(f"Neo4j ingestion failed: {str(e)}")
            return {"success": False, "error": str(e)}


def main():
    """CLI entry point for multimodal ingestion."""
    import sys
    
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else None
    
    if not pdf_path:
        print("Usage: python -m graphrag.ingestion.multimodal_ingestion <pdf_path>")
        sys.exit(1)
    
    processor = MultimodalDocumentProcessor()
    elements = processor.process_document(pdf_path)
    
    print(f"\n✓ Extracted {len(elements)} multimodal elements from {pdf_path}")
    for el in elements[:5]:
        print(f"  - {el.id}: {el.type}")


if __name__ == "__main__":
    main()
