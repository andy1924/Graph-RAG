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
from graphrag.retrieval import GraphRetriever, SemanticGraphRetriever, MultimodalGraphRetriever
from graphrag.evaluation import EvaluationPipeline
from graphrag.utils import ExperimentLogger

logger = ExperimentLogger("comprehensive_eval")


class BenchmarkDataset:
    """Container for benchmark questions and answers."""
    
    def __init__(self):
        self.questions = [
            "What are the main characteristics of entity A?",
            "How does entity B relate to entity C?",
            "What is the historical significance of X?",
            "Compare the features of Y and Z.",
            "What is the impact of process P on entity E?"
        ]
        
        self.references = [
            "Entity A is characterized by properties: P1, P2, P3, found in documents D1.",
            "Entity B connects to C through relationship RELATED_TO with properties P1.",
            "X has historical significance due to event E1 in time period T1.",
            "Y has features F1, F2 while Z has features F1, F3, F4.",
            "Process P impacts entity E in ways: I1, I2, as documented in source S1."
        ]
        
        # Simulated retrieved items
        self.relevant_items = [
            ["item1", "item2", "item3", "item4", "item5"],
            ["item2", "item3", "item6"],
            ["item7", "item8", "item9", "item10"],
            ["item4", "item11", "item12"],
            ["item6", "item7", "item13", "item14", "item15"]
        ]
    
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
            # Simulate retrieval (in real scenario, would use actual retriever)
            start_time = time.time()
            answer, metadata = retriever.answer_question(question)
            response_time = time.time() - start_time
            
            # Evaluate
            metrics = evaluator.evaluate(
                question=question,
                generated_answer=answer,
                reference_answer=reference,
                retrieved_context=answer,  # Simplified
                retrieved_items=dataset.relevant_items[i][:3],
                relevant_items=dataset.relevant_items[i]
            )
            
            metrics.avg_response_time = response_time
            all_metrics.append(metrics)
            
            per_question_results.append({
                "question": question,
                "answer": answer,
                "metrics": metrics.to_dict(),
                "response_time": response_time
            })
            
            logger.get_logger().info(f"✓ F1: {metrics.retrieval_f1:.3f}, "
                                    f"Hallucination: {metrics.hallucination_rate:.3f}")
        
        except Exception as e:
            logger.get_logger().error(f"✗ Error processing question {i+1}: {str(e)}")
    
    # Aggregate metrics
    metrics_summary = {
        "experiment": "baseline",
        "num_questions": len(dataset),
        "avg_f1": sum(m.retrieval_f1 for m in all_metrics) / len(all_metrics),
        "avg_hallucination_rate": sum(m.hallucination_rate for m in all_metrics) / len(all_metrics),
        "avg_semantic_similarity": sum(m.semantic_similarity for m in all_metrics) / len(all_metrics),
        "avg_response_time": sum(m.avg_response_time for m in all_metrics) / len(all_metrics),
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
                start_time = time.time()
                answer, metadata = retriever.answer_with_multimodal_context(
                    question,
                    include_modalities=combo
                )
                response_time = time.time() - start_time
                
                metrics = evaluator.evaluate(
                    question=question,
                    generated_answer=answer,
                    reference_answer=reference,
                    retrieved_context=answer,
                    retrieved_items=dataset.relevant_items[i][:3],
                    relevant_items=dataset.relevant_items[i],
                    multimodal_context={"text": "sample", "table": "sample" if "table" in combo else "", 
                                       "image": "sample" if "image" in combo else ""}
                )
                
                metrics.avg_response_time = response_time
                metrics_list.append(metrics)
            
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
            
            logger.get_logger().info(f"  ✓ Avg F1: {results_by_combo[combo_name]['avg_f1']:.3f}")
            logger.get_logger().info(f"  ✓ Avg Hallucination: {results_by_combo[combo_name]['avg_hallucination']:.3f}")
    
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
    
    logger.get_logger().info(f"\n✓ Results saved to {filepath}")
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
    
    logger.get_logger().info("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()
