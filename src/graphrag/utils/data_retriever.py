"""
Data retriever module for loading relevant graph nodes based on questions.
Retrieves actual node IDs from the preprocessed graph data.
"""

import json
import os
from typing import List, Dict, Any
from pathlib import Path


class RelevantItemsRetriever:
    """Retrieves relevant graph nodes for evaluation questions."""
    
    def __init__(self, graph_data_path: str = None):
        """
        Initialize the retriever with graph data.
        
        Args:
            graph_data_path: Path to the preprocessed graph_data.json file
        """
        if graph_data_path is None:
            # Default path relative to workspace
            graph_data_path = os.path.join(
                os.path.dirname(__file__),
                '..', '..', '..',
                'data', 'preprocessed', 'graph_data.json'
            )
        
        self.graph_data_path = graph_data_path
        self.all_nodes = []
        self.node_by_id = {}
        self._load_graph_data()
    
    def _load_graph_data(self):
        """Load and parse the graph data from JSON file."""
        try:
            with open(self.graph_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Flatten nodes from all pages
            if isinstance(data, list):
                for page in data:
                    if isinstance(page, dict) and "nodes" in page:
                        self.all_nodes.extend(page["nodes"])
            
            # Create ID lookup
            for node in self.all_nodes:
                self.node_by_id[node.get("id")] = node
            
            print(f"[+] Loaded {len(self.all_nodes)} nodes from graph data")
        
        except FileNotFoundError:
            print(f"[!] Warning: Graph data not found at {self.graph_data_path}")
            self.all_nodes = []
            self.node_by_id = {}
    
    def get_relevant_items_for_question(
        self,
        question: str,
        keywords: List[str] = None,
        num_items: int = 5
    ) -> List[str]:
        """
        Get relevant node IDs for a given question.
        
        Args:
            question: The evaluation question
            keywords: Keywords to search for in node IDs (semantic search)
            num_items: Maximum number of items to return
        
        Returns:
            List of relevant node IDs
        """
        if not keywords:
            # Extract potential keywords from question
            keywords = self._extract_keywords(question)
        
        relevant_nodes = []
        
        # Search for nodes matching keywords
        for node in self.all_nodes:
            node_id = node.get("id", "").lower()
            node_type = node.get("type", "").lower()
            
            # Score based on keyword matches
            score = 0
            for keyword in keywords:
                if keyword.lower() in node_id:
                    score += 2  # Higher weight for ID match
                if keyword.lower() in node_type:
                    score += 1
            
            if score > 0:
                relevant_nodes.append({
                    "id": node.get("id"),
                    "type": node.get("type"),
                    "score": score
                })
        
        # Sort by score and return top N
        relevant_nodes.sort(key=lambda x: x["score"], reverse=True)
        result = [node["id"] for node in relevant_nodes[:num_items]]
        
        # If not enough results, pad with additional nodes
        if len(result) < num_items:
            used_ids = set(result)
            for node in self.all_nodes:
                if len(result) >= num_items:
                    break
                if node.get("id") not in used_ids:
                    result.append(node.get("id"))
        
        return result[:num_items]
    
    def _extract_keywords(self, question: str) -> List[str]:
        """
        Extract keywords from a question.
        Simple keyword extraction by splitting on spaces and filtering.
        """
        # Remove common words
        stop_words = {"what", "how", "is", "the", "of", "in", "to", "and", "or", "a", "an"}
        
        words = question.lower().split()
        keywords = [w.strip("?.,!") for w in words if w.strip("?.,!") not in stop_words and len(w) > 2]
        
        return keywords


def get_relevant_items_mapping(
    questions: List[str],
    question_keywords: Dict[str, List[str]] = None
) -> List[List[str]]:
    """
    Get relevant items for a list of questions.
    
    Args:
        questions: List of evaluation questions
        question_keywords: Optional dict mapping question to keywords
                          If not provided, auto-extract keywords
    
    Returns:
        List of relevant item lists (one per question)
    """
    retriever = RelevantItemsRetriever()
    
    if not question_keywords:
        question_keywords = {}
    
    relevant_items_list = []
    
    for i, question in enumerate(questions):
        keywords = question_keywords.get(question, None)
        items = retriever.get_relevant_items_for_question(question, keywords=keywords)
        relevant_items_list.append(items)
        print(f"Q{i+1}: {question[:50]}...")
        print(f"  -> Retrieved {len(items)} relevant items: {items}")
    
    return relevant_items_list


# Optional mapping for manual control of question-to-keywords
QUESTION_KEYWORDS_MAPPING = {
    # Add manual mappings if needed
    # "What are the main characteristics...": ["Transformer", "architecture", "attention"]
}
