"""
Data retriever module for loading relevant graph nodes based on questions.
Retrieves actual node IDs from the preprocessed graph data.
"""

import json
import os
from typing import List, Dict, Any
from pathlib import Path


# ------------------------------------------------------------------ #
# Manually annotated ground-truth relevant nodes per benchmark question
# ------------------------------------------------------------------ #
# Each list contains the node IDs from graph_data.json that are genuinely
# relevant to answering the corresponding question.  These were curated
# by hand to replace noisy keyword matching.

GROUND_TRUTH_RELEVANT_ITEMS: Dict[str, List[str]] = {
    "What are the main characteristics of the Transformer architecture?": [
        "Transformer",
        "Attention Mechanisms",
        "Self-Attention",
        "Self-Attention Mechanism",
        "Encoder",
        "Decoder",
        "Encoder-Decoder Structure",
        "Multi-Head Attention",
        "Attention Is All You Need",
    ],

    "How does Multi-Head Attention relate to Scaled Dot-Product Attention?": [
        "Multi-Head Attention",
        "Scaled Dot-Product Attention",
        "Attention Head",
        "Attention Function",
        "Multi-Head Self-Attention Mechanism",
        "Softmax",
        "Queries",
        "Keys",
        "Values",
    ],

    "What is the performance significance of the Transformer model on the WMT 2014 English-to-German translation task?": [
        "Transformer",
        "Transformer (Big)",
        "Transformer (Base Model)",
        "Wmt 2014 English-To-German Translation Task",
        "Wmt 2014 English-German Dataset",
        "Attention Is All You Need",
    ],

    "Compare the computational complexity per layer of self-attention layers and recurrent layers.": [
        "Self-Attention",
        "Self-Attention Mechanism",
        "Recurrent",
        "Recurrent Layers",
        "Recurrent Neural Networks",
        "Recurrent Language Models",
    ],

    "What is the impact of masking in the decoder's self-attention sub-layer?": [
        "Masking",
        "Decoder",
        "Self-Attention",
        "Self-Attention Mechanism",
        "Encoder-Decoder Attention",
        "Multi-Head Self-Attention Mechanism",
    ],
}


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
            
            # Deduplicated ID lookup — keep node with highest type specificity
            for node in self.all_nodes:
                nid = node.get("id")
                if nid not in self.node_by_id:
                    self.node_by_id[nid] = node
            
            print(f"[+] Loaded {len(self.node_by_id)} unique nodes from graph data")
        
        except FileNotFoundError:
            print(f"[!] Warning: Graph data not found at {self.graph_data_path}")
            self.all_nodes = []
            self.node_by_id = {}
    
    def get_relevant_items_for_question(
        self,
        question: str,
        keywords: List[str] = None,
        num_items: int = 10
    ) -> List[str]:
        """
        Get relevant node IDs for a given question.
        
        First checks GROUND_TRUTH_RELEVANT_ITEMS for a manual annotation.
        Falls back to keyword-based scoring if no annotation exists.
        
        Args:
            question: The evaluation question
            keywords: Keywords to search for in node IDs (semantic search)
            num_items: Maximum number of items to return
        
        Returns:
            List of relevant node IDs (deduplicated)
        """
        # --- Priority 1: manual ground-truth ---
        if question in GROUND_TRUTH_RELEVANT_ITEMS:
            gt_items = GROUND_TRUTH_RELEVANT_ITEMS[question]
            # Validate that the IDs actually exist in the graph
            valid = [nid for nid in gt_items if nid in self.node_by_id]
            if valid:
                return valid
        
        # --- Priority 2: keyword-based search (fallback) ---
        if not keywords:
            keywords = self._extract_keywords(question)
        
        scored: Dict[str, int] = {}  # node_id → best score (dedup)
        
        for node in self.all_nodes:
            node_id = node.get("id", "")
            node_id_lower = node_id.lower()
            node_type = node.get("type", "").lower()
            
            score = 0
            for keyword in keywords:
                kw = keyword.lower()
                if kw in node_id_lower:
                    score += 2
                if kw in node_type:
                    score += 1
            
            if score > 0:
                # Keep highest score per node ID (deduplication)
                if node_id not in scored or score > scored[node_id]:
                    scored[node_id] = score
        
        # Sort by score descending, return top N
        ranked = sorted(scored.items(), key=lambda x: x[1], reverse=True)
        return [nid for nid, _ in ranked[:num_items]]
    
    def _extract_keywords(self, question: str) -> List[str]:
        """
        Extract keywords from a question.
        Simple keyword extraction by splitting on spaces and filtering.
        """
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


# Optional mapping for manual control of question-to-keywords (fallback)
QUESTION_KEYWORDS_MAPPING = {
    "What are the main characteristics of the Transformer architecture?": ["Transformer", "architecture", "attention", "self-attention"],
    "How does Multi-Head Attention relate to Scaled Dot-Product Attention?": ["Multi-Head Attention", "Scaled Dot-Product", "attention", "projection"],
    "What is the performance significance of the Transformer model on the WMT 2014 English-to-German translation task?": ["Transformer", "WMT 2014", "translation", "BLEU", "performance"],
    "Compare the computational complexity per layer of self-attention layers and recurrent layers.": ["self-attention", "recurrent", "complexity", "computational", "O(n)"],
    "What is the impact of masking in the decoder's self-attention sub-layer?": ["masking", "decoder", "self-attention", "positions", "autoregressive"]
}
