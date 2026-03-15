"""
Comprehensive evaluation experiment for GraphRAG system.
Compares performance across different modality combinations and retrieval strategies.
"""

import os
import sys
import time
import json
from typing import Dict, List, Tuple, Any
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from graphrag.config import config
from graphrag.retrieval import GraphRetriever, SemanticGraphRetriever, MultimodalGraphRetriever, get_graph_context, ask_llm_with_context
from graphrag.evaluation import EvaluationPipeline
from graphrag.utils import ExperimentLogger
from graphrag.utils.data_retriever import get_relevant_items_mapping, QUESTION_KEYWORDS_MAPPING

logger = ExperimentLogger("comprehensive_eval")


class BenchmarkDataset:
    """Container for benchmark questions and answers."""
    
    def __init__(self):
        self.questions = [
            "What are the main characteristics of the Transformer architecture?",
            "How does Multi-Head Attention relate to Scaled Dot-Product Attention?",
            "What is the performance significance of the Transformer model on the WMT 2014 English-to-German translation task?",
            "Compare the computational complexity per layer of self-attention layers and recurrent layers.",
            "What is the impact of masking in the decoder's self-attention sub-layer?"
        ]
        
        self.references = [
            "The Transformer is a network architecture based solely on attention mechanisms, dispensing with recurrence and convolutions entirely, and using stacked self-attention and point-wise, fully connected layers. [cite: 17, 78]",
            "Multi-Head Attention connects to Scaled Dot-Product Attention by linearly projecting queries, keys, and values h times, and performing the scaled dot-product attention function in parallel on each projected version. [cite: 126, 127]",
            "The Transformer model achieved a new state-of-the-art BLEU score of 28.4 on the WMT 2014 English-to-German translation task, improving over existing best results by over 2 BLEU. [cite: 19]",
            "Self-attention layers have a complexity of O(n^2 * d) per layer, while recurrent layers have a complexity of O(n * d^2), making self-attention faster when sequence length n is smaller than representation dimensionality d. [cite: 163, 187, 188, 189]",
            "Masking impacts the decoder by preventing positions from attending to subsequent positions, ensuring that predictions for position i can depend only on the known outputs at positions less than i, preserving the auto-regressive property. [cite: 88, 89]"
        ]
        
        # Retrieve relevant items from actual graph data
        print("\n" + "=" * 60)
        print("Loading relevant items from graph data...")
        print("=" * 60)
        self.relevant_items = get_relevant_items_mapping(
            self.questions,
            question_keywords=QUESTION_KEYWORDS_MAPPING
        )
        print("=" * 60 + "\n")
    
    def __len__(self):
        return len(self.questions)


def run_baseline_experiment(
    dataset: BenchmarkDataset
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Run baseline GraphRetriever experiment.
    
    Args:
        dataset: Benchmark dataset
    
    Returns:
        Tuple of (metrics_summary, per_question_results)
    """
    logger.get_logger().info("=" * 60)
    logger.get_logger().info("Running BASELINE EXPERIMENT: Standard Graph Retrieval")
    logger.get_logger().info("=" * 60)
    
    retriever = GraphRetriever()
    evaluator = EvaluationPipeline("baseline_exp")
    
    per_question_results = []
    all_metrics = []
    
    for i, (question, reference) in enumerate(zip(dataset.questions, dataset.references)):
        logger.get_logger().info(f"\nQuestion {i+1}/{len(dataset)}: {question[:50]}...")
        
        try:
            # Time the real Neo4j retrieval + LLM answer generation directly,
            # bypassing the retriever wrapper so the timer captures actual latency.
            start = time.perf_counter()
            retrieved_context, sources, retrieved_nodes, _relations = get_graph_context(
                question, retriever.client, retriever.driver, retriever.database
            )
            answer = ask_llm_with_context(question, retrieved_context, retriever.client)
            elapsed = time.perf_counter() - start

            assert retrieved_context != answer, "ERROR: retrieved_context must be different from generated answer (circular evaluation detected)"
            
            # Evaluate with actual retrieved context
            metrics = evaluator.evaluate(
                question=question,
                generated_answer=answer,
                reference_answer=reference,
                retrieved_context=retrieved_context,
                retrieved_items=retrieved_nodes,
                relevant_items=dataset.relevant_items[i],
                response_time=elapsed
            )
            
            all_metrics.append(metrics)
            
            per_question_results.append({
                "question": question,
                "answer": answer,
                "retrieved_context": retrieved_context,
                "metrics": metrics.to_dict(),
                "response_time": elapsed
            })
            
            logger.get_logger().info(f"[+] F1: {metrics.retrieval_f1:.3f}, "
                                    f"Hallucination: {metrics.hallucination_rate:.3f}, "
                                    f"Response Time: {elapsed:.4f}s")
        
        except Exception as e:
            logger.get_logger().error(f"[!] Error processing question {i+1}: {str(e)}")
    
    # Aggregate metrics
    n = len(all_metrics)
    metrics_summary = {
        "experiment": "baseline",
        "num_questions": len(dataset),
        "avg_f1": sum(m.retrieval_f1 for m in all_metrics) / n if n else 0.0,
        "avg_hallucination_rate": sum(m.hallucination_rate for m in all_metrics) / n if n else 0.0,
        "avg_semantic_similarity": sum(m.semantic_similarity for m in all_metrics) / n if n else 0.0,
        "avg_response_time": sum(m.avg_response_time for m in all_metrics) / n if n else 0.0,
    }
    
    return metrics_summary, per_question_results


def run_multimodal_experiment(
    dataset: BenchmarkDataset,
    modality_combinations: List[List[str]]
) -> Dict[str, Dict[str, Any]]:
    """
    Run multimodal retrieval ablation study.
    
    Args:
        dataset: Benchmark dataset
        modality_combinations: List of modality combinations to test
    
    Returns:
        Dictionary with results for each combination
    """
    logger.get_logger().info("\n" + "=" * 60)
    logger.get_logger().info("Running MULTIMODAL ABLATION STUDY")
    logger.get_logger().info("=" * 60)
    
    retriever = MultimodalGraphRetriever()
    results_by_combo = {}
    
    for combo in modality_combinations:
        combo_name = "+".join(combo)
        logger.get_logger().info(f"\nTesting modality combination: {combo_name}")
        
        evaluator = EvaluationPipeline(f"multimodal_{combo_name}")
        metrics_list = []
        
        for i, (question, reference) in enumerate(zip(dataset.questions, dataset.references)):
            try:
                start = time.perf_counter()
                answer, metadata = retriever.answer_with_multimodal_context(
                    question,
                    include_modalities=combo
                )
                elapsed = time.perf_counter() - start
                
                # Extract retrieved context from metadata
                retrieved_context = metadata.get("context", "")
                assert retrieved_context != answer, "ERROR: retrieved_context must be different from generated answer (circular evaluation detected)"
                
                metrics = evaluator.evaluate(
                    question=question,
                    generated_answer=answer,
                    reference_answer=reference,
                    retrieved_context=retrieved_context,
                    retrieved_items=metadata.get("retrieved_nodes", []),
                    relevant_items=dataset.relevant_items[i],
                    multimodal_context={"text": metadata.get("text_content", ""), 
                                       "table": metadata.get("table_content", ""), 
                                       "image": metadata.get("image_content", "")},
                    response_time=elapsed
                )
                
                metrics_list.append(metrics)
            
            except AssertionError as e:
                logger.get_logger().error(f"[!] Circular evaluation error: {str(e)}")
            except Exception as e:
                logger.get_logger().warning(f"Error processing question: {str(e)}")
        
        # Aggregate
        if metrics_list:
            results_by_combo[combo_name] = {
                "combination": combo,
                "num_questions": len(metrics_list),
                "avg_f1": sum(m.retrieval_f1 for m in metrics_list) / len(metrics_list),
                "avg_hallucination": sum(m.hallucination_rate for m in metrics_list) / len(metrics_list),
                "avg_semantic_sim": sum(m.semantic_similarity for m in metrics_list) / len(metrics_list),
                "avg_response_time": sum(m.avg_response_time for m in metrics_list) / len(metrics_list),
                "text_usage": sum(m.text_modality_usage for m in metrics_list) / len(metrics_list),
                "table_usage": sum(m.table_modality_usage for m in metrics_list) / len(metrics_list),
                "image_usage": sum(m.image_modality_usage for m in metrics_list) / len(metrics_list),
            }
            
            logger.get_logger().info(f"  [+] Avg F1: {results_by_combo[combo_name]['avg_f1']:.3f}")
            logger.get_logger().info(f"  [+] Avg Hallucination: {results_by_combo[combo_name]['avg_hallucination']:.3f}")
    
    return results_by_combo


def save_results(
    baseline_results: Tuple,
    multimodal_results: Dict,
    output_dir: str = "results"
) -> str:
    """
    Save comprehensive evaluation results.
    
    Args:
        baseline_results: Baseline experiment results
        multimodal_results: Multimodal ablation results
        output_dir: Output directory for results
    
    Returns:
        Path to results file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "baseline": baseline_results[0],
        "baseline_details": baseline_results[1],
        "multimodal_ablation": multimodal_results
    }
    
    filepath = os.path.join(output_dir, "comprehensive_evaluation.json")
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.get_logger().info(f"\n[+] Results saved to {filepath}")
    return filepath


def main():
    """Run comprehensive evaluation experiments."""
    logger.get_logger().info("GraphRAG Comprehensive Evaluation")
    logger.get_logger().info(f"Experiment ID: {logger.experiment_name}")
    
    # Load benchmark dataset
    dataset = BenchmarkDataset()
    logger.get_logger().info(f"Loaded benchmark dataset with {len(dataset)} questions")
    
    # Run baseline
    baseline_results = run_baseline_experiment(dataset)
    
    # Run multimodal ablation
    modality_combos = [
        ["text"],
        ["text", "table"],
        ["text", "table", "image"],
        ["table"],
        ["image"],
    ]
    
    multimodal_results = run_multimodal_experiment(dataset, modality_combos)
    
    # Save results
    results_file = save_results(baseline_results, multimodal_results)
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"\nBaseline Results:")
    print(f"  F1 Score: {baseline_results[0]['avg_f1']:.3f}")
    print(f"  Hallucination Rate: {baseline_results[0]['avg_hallucination_rate']:.3f}")
    print(f"  Semantic Similarity: {baseline_results[0]['avg_semantic_similarity']:.3f}")
    print(f"  Avg Response Time: {baseline_results[0]['avg_response_time']:.3f}s")
    
    print(f"\nBest Multimodal Configuration:")
    best_config = max(multimodal_results.items(), 
                      key=lambda x: x[1]['avg_f1'] - x[1]['avg_hallucination'])
    print(f"  Modalities: {best_config[0]}")
    print(f"  F1 Score: {best_config[1]['avg_f1']:.3f}")
    print(f"  Hallucination Rate: {best_config[1]['avg_hallucination']:.3f}")
    
    logger.get_logger().info("\n[+] Evaluation complete!")


if __name__ == "__main__":
    main()
