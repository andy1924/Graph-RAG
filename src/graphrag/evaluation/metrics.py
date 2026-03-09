"""
Comprehensive evaluation metrics for GraphRAG system.
Measures hallucination reduction, retrieval quality, and answer generation effectiveness.
"""

import json
import os
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
from scipy.spatial.distance import cosine

from ..config import config
from ..utils import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for evaluation results."""
    experiment_id: str
    timestamp: str
    
    # Retrieval metrics
    retrieval_precision: float
    retrieval_recall: float
    retrieval_f1: float
    
    # Answer quality metrics
    rouge_score: float
    semantic_similarity: float
    
    # Hallucination detection
    hallucination_rate: float
    grounded_ratio: float
    
    # Coverage and efficiency
    context_coverage: float
    avg_response_time: float
    
    # Multimodal metrics
    text_modality_usage: float
    table_modality_usage: float
    image_modality_usage: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self, filepath: str):
        """Save to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)


class RetrievalMetrics:
    """Metrics for graph retrieval quality."""
    
    @staticmethod
    def precision(
        retrieved_items: List[str],
        relevant_items: List[str]
    ) -> float:
        """
        Calculate precision: fraction of retrieved items that are relevant.
        
        Args:
            retrieved_items: Items returned by retrieval
            relevant_items: Ground truth relevant items
        
        Returns:
            Precision score [0, 1]
        """
        if not retrieved_items:
            return 0.0
        
        relevant_set = set(relevant_items)
        retrieved_set = set(retrieved_items)
        
        correct = len(retrieved_set & relevant_set)
        return correct / len(retrieved_set)
    
    @staticmethod
    def recall(
        retrieved_items: List[str],
        relevant_items: List[str]
    ) -> float:
        """
        Calculate recall: fraction of relevant items that were retrieved.
        
        Args:
            retrieved_items: Items returned by retrieval
            relevant_items: Ground truth relevant items
        
        Returns:
            Recall score [0, 1]
        """
        if not relevant_items:
            return 1.0
        
        relevant_set = set(relevant_items)
        retrieved_set = set(retrieved_items)
        
        correct = len(relevant_set & retrieved_set)
        return correct / len(relevant_set)
    
    @staticmethod
    def f1_score(precision: float, recall: float) -> float:
        """
        Calculate F1 score: harmonic mean of precision and recall.
        
        Args:
            precision: Precision value
            recall: Recall value
        
        Returns:
            F1 score [0, 1]
        """
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)


class AnswerQualityMetrics:
    """Metrics for answer quality evaluation."""
    
    @staticmethod
    def rouge_score(
        generated: str,
        reference: str,
        use_stemming: bool = True
    ) -> Dict[str, float]:
        """
        Calculate ROUGE score (Recall-Oriented Understudy for Gisting Evaluation).
        Measures n-gram overlap between generated and reference texts.
        
        Args:
            generated: Generated answer text
            reference: Reference/ground truth answer
            use_stemming: Whether to use stemming
        
        Returns:
            Dictionary with ROUGE-1, ROUGE-2, ROUGE-L scores
        """
        try:
            from rouge_score import rouge_scorer
        except ImportError:
            logger.warning("rouge_score not installed. Skipping ROUGE calculation.")
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        
        scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemming=use_stemming
        )
        
        scores = scorer.score(reference, generated)
        
        return {
            "rouge1": scores['rouge1'].fmeasure,
            "rouge2": scores['rouge2'].fmeasure,
            "rougeL": scores['rougeL'].fmeasure,
        }
    
    @staticmethod
    def bert_score(
        generated: str,
        reference: str,
        model_type: str = "roberta-large"
    ) -> Dict[str, float]:
        """
        Calculate BERTScore: semantic similarity using contextual embeddings.
        
        Args:
            generated: Generated answer
            reference: Reference answer
            model_type: BERT model type
        
        Returns:
            Dictionary with precision, recall, F1 scores
        """
        try:
            from bert_score import score as bert_score_func
        except ImportError:
            logger.warning("bert_score not installed. Skipping BERTScore calculation.")
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        try:
            P, R, F1 = bert_score_func(
                [generated],
                [reference],
                model_type=model_type,
                lang="en",
                verbose=False
            )
            
            return {
                "precision": P.item(),
                "recall": R.item(),
                "f1": F1.item(),
            }
        except Exception as e:
            logger.warning(f"BERTScore calculation failed: {str(e)}")
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    @staticmethod
    def semantic_similarity(
        generated: str,
        reference: str,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ) -> float:
        """
        Calculate semantic similarity using sentence embeddings.
        
        Args:
            generated: Generated answer
            reference: Reference answer
            model_name: SentenceTransformer model name
        
        Returns:
            Similarity score [0, 1]
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.warning("sentence-transformers not installed. Using simple similarity.")
            # Fallback: token overlap
            gen_tokens = set(generated.split())
            ref_tokens = set(reference.split())
            if not gen_tokens or not ref_tokens:
                return 0.0
            overlap = len(gen_tokens & ref_tokens)
            return overlap / (len(gen_tokens | ref_tokens) + 1e-6)
        
        try:
            model = SentenceTransformer(model_name)
            
            gen_embedding = model.encode(generated, convert_to_tensor=True)
            ref_embedding = model.encode(reference, convert_to_tensor=True)
            
            # Cosine similarity
            similarity = 1 - cosine(
                gen_embedding.cpu().numpy(),
                ref_embedding.cpu().numpy()
            )
            
            return float(np.clip(similarity, 0, 1))
        
        except Exception as e:
            logger.warning(f"Semantic similarity calculation failed: {str(e)}")
            return 0.0


class HallucinationDetector:
    """Detects and measures hallucinations in LLM outputs."""
    
    @staticmethod
    def detect_unsupported_claims(
        answer: str,
        context: str,
        threshold: float = 0.7
    ) -> Tuple[float, List[str]]:
        """
        Detect claims in answer not supported by context.
        Uses semantic similarity to identify potential hallucinations.
        
        Args:
            answer: Generated answer text
            context: Retrieved context
            threshold: Similarity threshold for grounding
        
        Returns:
            Tuple of (hallucination_rate, ungrounded_claims)
        """
        try:
            from sentence_transformers import SentenceTransformer, util
        except ImportError:
            logger.warning("sentence-transformers not installed.")
            return 0.0, []
        
        try:
            model = SentenceTransformer("all-MiniLM-L6-v2")
            
            # Split answer into sentences
            import nltk
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')
            
            from nltk.tokenize import sent_tokenize
            
            answer_sentences = sent_tokenize(answer)
            context_embedding = model.encode(context, convert_to_tensor=True)
            
            ungrounded_claims = []
            
            for sentence in answer_sentences:
                if len(sentence.split()) < 5:  # Skip short sentences
                    continue
                
                sentence_embedding = model.encode(sentence, convert_to_tensor=True)
                similarity = util.pytorch_cos_sim(
                    sentence_embedding,
                    context_embedding
                ).item()
                
                if similarity < threshold:
                    ungrounded_claims.append(sentence)
            
            hallucination_rate = len(ungrounded_claims) / len(answer_sentences) if answer_sentences else 0
            
            return float(hallucination_rate), ungrounded_claims
        
        except Exception as e:
            logger.warning(f"Hallucination detection failed: {str(e)}")
            return 0.0, []
    
    @staticmethod
    def fact_consistency_check(
        answer: str,
        context: str
    ) -> float:
        """
        Check consistency between answer and context using NER and fact matching.
        
        Args:
            answer: Generated answer
            context: Retrieved context
        
        Returns:
            Consistency score [0, 1]
        """
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            logger.warning(f"Spacy model not available: {str(e)}")
            return 0.5
        
        try:
            # Extract entities from both texts
            answer_doc = nlp(answer)
            context_doc = nlp(context)
            
            answer_entities = {(ent.text, ent.label_) for ent in answer_doc.ents}
            context_entities = {(ent.text, ent.label_) for ent in context_doc.ents}
            
            # Check entity overlap
            if not answer_entities:
                return 1.0
            
            matching_entities = answer_entities & context_entities
            consistency = len(matching_entities) / len(answer_entities)
            
            return float(consistency)
        
        except Exception as e:
            logger.warning(f"Fact consistency check failed: {str(e)}")
            return 0.5


class MultimodalMetrics:
    """Metrics specific to multimodal evaluation."""
    
    @staticmethod
    def modality_coverage(
        context: Dict[str, List[str]]
    ) -> Dict[str, float]:
        """
        Calculate coverage of each modality type in retrieved context.
        
        Args:
            context: Context organized by modality type
        
        Returns:
            Dictionary mapping modality to coverage score
        """
        total_items = sum(len(items) for items in context.values())
        
        coverage = {}
        for modality, items in context.items():
            if total_items > 0:
                coverage[modality] = len(items) / total_items
            else:
                coverage[modality] = 0.0
        
        return coverage
    
    @staticmethod
    def multimodal_relevance(
        question: str,
        context_by_modality: Dict[str, str]
    ) -> Dict[str, float]:
        """
        Calculate relevance of each modality to the question.
        
        Args:
            question: User question
            context_by_modality: Context grouped by modality
        
        Returns:
            Dictionary mapping modality to relevance score
        """
        try:
            from sentence_transformers import SentenceTransformer, util
            model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            logger.warning(f"Could not initialize SentenceTransformer: {str(e)}")
            return {modality: 0.5 for modality in context_by_modality}
        
        question_embedding = model.encode(question, convert_to_tensor=True)
        relevance_scores = {}
        
        for modality, text in context_by_modality.items():
            if text.strip():
                text_embedding = model.encode(text, convert_to_tensor=True)
                similarity = util.pytorch_cos_sim(
                    question_embedding,
                    text_embedding
                ).item()
                relevance_scores[modality] = float(similarity)
            else:
                relevance_scores[modality] = 0.0
        
        return relevance_scores


class EvaluationPipeline:
    """Complete evaluation pipeline for GraphRAG system."""
    
    def __init__(self, experiment_id: str = None):
        """
        Initialize evaluation pipeline.
        
        Args:
            experiment_id: Unique identifier for experiment
        """
        self.experiment_id = experiment_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger = get_logger(self.__class__.__name__)
        
        self.retrieval_metrics = RetrievalMetrics()
        self.answer_metrics = AnswerQualityMetrics()
        self.hallucination_detector = HallucinationDetector()
        self.multimodal_metrics = MultimodalMetrics()
    
    def evaluate(
        self,
        question: str,
        generated_answer: str,
        reference_answer: str,
        retrieved_context: str,
        retrieved_items: List[str] = None,
        relevant_items: List[str] = None,
        multimodal_context: Dict[str, str] = None
    ) -> EvaluationMetrics:
        """
        Run comprehensive evaluation.
        
        Args:
            question: Original question
            generated_answer: Answer generated by system
            reference_answer: Ground truth answer
            retrieved_context: Retrieved context from graph
            retrieved_items: List of retrieved item IDs
            relevant_items: List of relevant item IDs
            multimodal_context: Context by modality type
        
        Returns:
            EvaluationMetrics object with all scores
        """
        self.logger.info(f"Starting evaluation for experiment {self.experiment_id}")
        
        # Retrieval metrics
        if retrieved_items and relevant_items:
            precision = self.retrieval_metrics.precision(retrieved_items, relevant_items)
            recall = self.retrieval_metrics.recall(retrieved_items, relevant_items)
            f1 = self.retrieval_metrics.f1_score(precision, recall)
        else:
            precision = recall = f1 = 0.0
        
        # Answer quality metrics
        rouge = self.answer_metrics.rouge_score(generated_answer, reference_answer)
        semantic_sim = self.answer_metrics.semantic_similarity(generated_answer, reference_answer)
        
        # Hallucination detection
        hallucination_rate, ungrounded = self.hallucination_detector.detect_unsupported_claims(
            generated_answer,
            retrieved_context,
            threshold=config.evaluation.hallucination_threshold
        )
        grounded_ratio = 1.0 - hallucination_rate
        
        # Multimodal metrics
        text_usage = table_usage = image_usage = 0.0
        if multimodal_context:
            coverage = self.multimodal_metrics.modality_coverage(multimodal_context)
            text_usage = coverage.get("text", 0.0)
            table_usage = coverage.get("table", 0.0)
            image_usage = coverage.get("image", 0.0)
        
        # Context coverage
        context_coverage = len(retrieved_context) / 10000 if retrieved_context else 0  # Normalized
        context_coverage = min(context_coverage, 1.0)
        
        metrics = EvaluationMetrics(
            experiment_id=self.experiment_id,
            timestamp=datetime.now().isoformat(),
            retrieval_precision=precision,
            retrieval_recall=recall,
            retrieval_f1=f1,
            rouge_score=rouge.get("rouge1", 0.0),
            semantic_similarity=semantic_sim,
            hallucination_rate=hallucination_rate,
            grounded_ratio=grounded_ratio,
            context_coverage=context_coverage,
            avg_response_time=0.0,  # Should be measured externally
            text_modality_usage=text_usage,
            table_modality_usage=table_usage,
            image_modality_usage=image_usage,
        )
        
        self.logger.info(f"Evaluation complete. F1: {f1:.3f}, Hallucination Rate: {hallucination_rate:.3f}")
        
        return metrics
    
    def save_results(
        self,
        metrics: EvaluationMetrics,
        results_dir: str = None
    ) -> str:
        """
        Save evaluation results to file.
        
        Args:
            metrics: Evaluation metrics
            results_dir: Directory to save results
        
        Returns:
            Path to saved results file
        """
        results_dir = results_dir or config.evaluation.results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        filename = f"evaluation_{self.experiment_id}.json"
        filepath = os.path.join(results_dir, filename)
        
        metrics.to_json(filepath)
        self.logger.info(f"Results saved to {filepath}")
        
        return filepath


def main():
    """CLI entry point for evaluation."""
    print("Evaluation metrics module. Use EvaluationPipeline class directly.")


if __name__ == "__main__":
    main()
