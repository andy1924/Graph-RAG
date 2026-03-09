"""
Multimodal Ablation Study
Investigates the impact of different modality combinations on system performance.
"""

import os
import sys
import json
import time
from typing import Dict, List, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from graphrag.retrieval import MultimodalGraphRetriever
from graphrag.evaluation import EvaluationPipeline, MultimodalMetrics
from graphrag.utils import ExperimentLogger

logger = ExperimentLogger("multimodal_ablation")


class MultimodalAblationStudy:
    """Systematic study of multimodal retrieval effectiveness."""
    
    def __init__(self):
        """Initialize ablation study."""
        self.retriever = MultimodalGraphRetriever()
        self.modality_metrics = MultimodalMetrics()
        self.logger = logger.get_logger()
    
    def test_modality_combination(
        self,
        questions: List[str],
        references: List[str],
        modalities: List[str],
        combo_name: str
    ) -> Dict[str, Any]:
        """
        Test a specific modality combination.
        
        Args:
            questions: List of test questions
            references: List of reference answers
            modalities: Modality types to use
            combo_name: Name for this combination
        
        Returns:
            Results dictionary
        """
        self.logger.info(f"\nTesting combination: {combo_name} ({', '.join(modalities)})")
        
        evaluator = EvaluationPipeline(f"ablation_{combo_name}")
        metrics_list = []
        all_answers = []
        
        for i, (question, reference) in enumerate(zip(questions, references)):
            try:
                start_time = time.time()
                
                # Get answer with specific modalities
                answer, metadata = self.retriever.answer_with_multimodal_context(
                    question,
                    include_modalities=modalities
                )
                
                response_time = time.time() - start_time
                
                # Evaluate
                metrics = evaluator.evaluate(
                    question=question,
                    generated_answer=answer,
                    reference_answer=reference,
                    retrieved_context=answer,
                    retrieved_items=[],  # Simplified for this study
                    relevant_items=[],
                    multimodal_context={
                        mod: f"sample_{mod}_content" if mod in modalities else ""
                        for mod in ["text", "table", "image"]
                    }
                )
                
                metrics.avg_response_time = response_time
                metrics_list.append(metrics)
                all_answers.append({
                    "question": question,
                    "answer": answer,
                    "response_time": response_time
                })
                
                self.logger.debug(f"  Q{i+1}: F1={metrics.retrieval_f1:.3f}, "
                                f"Hall={metrics.hallucination_rate:.3f}")
            
            except Exception as e:
                self.logger.warning(f"Error processing question {i+1}: {str(e)}")
        
        # Aggregate results
        if not metrics_list:
            return None
        
        return {
            "combination": combo_name,
            "modalities": modalities,
            "num_questions": len(metrics_list),
            "metrics": {
                "avg_f1": sum(m.retrieval_f1 for m in metrics_list) / len(metrics_list),
                "avg_recall": sum(m.retrieval_recall for m in metrics_list) / len(metrics_list),
                "avg_precision": sum(m.retrieval_precision for m in metrics_list) / len(metrics_list),
                "avg_rouge": sum(m.rouge_score for m in metrics_list) / len(metrics_list),
                "avg_semantic_similarity": sum(m.semantic_similarity for m in metrics_list) / len(metrics_list),
                "avg_hallucination_rate": sum(m.hallucination_rate for m in metrics_list) / len(metrics_list),
                "avg_grounded_ratio": sum(m.grounded_ratio for m in metrics_list) / len(metrics_list),
                "avg_response_time": sum(m.avg_response_time for m in metrics_list) / len(metrics_list),
            },
            "modality_usage": {
                "text": sum(m.text_modality_usage for m in metrics_list) / len(metrics_list),
                "table": sum(m.table_modality_usage for m in metrics_list) / len(metrics_list),
                "image": sum(m.image_modality_usage for m in metrics_list) / len(metrics_list),
            },
            "answers": all_answers
        }
    
    def run_full_ablation(
        self,
        questions: List[str],
        references: List[str]
    ) -> Dict[str, Any]:
        """
        Run comprehensive ablation study.
        
        Args:
            questions: Test questions
            references: Reference answers
        
        Returns:
            Complete ablation study results
        """
        self.logger.info("=" * 70)
        self.logger.info("MULTIMODAL ABLATION STUDY")
        self.logger.info("=" * 70)
        
        # Define modality combinations to test
        combinations = [
            (["text"], "text_only"),
            (["table"], "table_only"),
            (["image"], "image_only"),
            (["text", "table"], "text_table"),
            (["text", "image"], "text_image"),
            (["table", "image"], "table_image"),
            (["text", "table", "image"], "all_modalities"),
        ]
        
        results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_questions": len(questions),
            "combinations": {}
        }
        
        for modalities, combo_name in combinations:
            combo_results = self.test_modality_combination(
                questions, references, modalities, combo_name
            )
            
            if combo_results:
                results["combinations"][combo_name] = combo_results
                
                self.logger.info(f"  ✓ {combo_name}:")
                self.logger.info(f"     F1={combo_results['metrics']['avg_f1']:.3f}, "
                               f"Hall={combo_results['metrics']['avg_hallucination_rate']:.3f}, "
                               f"Time={combo_results['metrics']['avg_response_time']:.3f}s")
        
        return results


def analyze_ablation_results(results: Dict[str, Any]) -> None:
    """Analyze and print ablation study results."""
    print("\n" + "=" * 70)
    print("ABLATION STUDY ANALYSIS")
    print("=" * 70)
    
    if not results.get("combinations"):
        print("No results to analyze.")
        return
    
    # Find best performers
    combos = results["combinations"]
    
    print("\nPerformance by Metric:")
    print("-" * 70)
    
    # Best F1
    best_f1 = max(combos.items(), key=lambda x: x[1]["metrics"]["avg_f1"])
    print(f"Best F1 Score: {best_f1[0]}")
    print(f"  F1: {best_f1[1]['metrics']['avg_f1']:.4f}")
    print(f"  Modalities: {', '.join(best_f1[1]['modalities'])}")
    
    # Best hallucination control
    best_grounding = max(combos.items(), key=lambda x: x[1]["metrics"]["avg_grounded_ratio"])
    print(f"\nBest Grounding (Lowest Hallucination): {best_grounding[0]}")
    print(f"  Grounded Ratio: {best_grounding[1]['metrics']['avg_grounded_ratio']:.4f}")
    print(f"  Hallucination Rate: {best_grounding[1]['metrics']['avg_hallucination_rate']:.4f}")
    print(f"  Modalities: {', '.join(best_grounding[1]['modalities'])}")
    
    # Fastest
    fastest = min(combos.items(), key=lambda x: x[1]["metrics"]["avg_response_time"])
    print(f"\nFastest Response: {fastest[0]}")
    print(f"  Response Time: {fastest[1]['metrics']['avg_response_time']:.4f}s")
    print(f"  Modalities: {', '.join(fastest[1]['modalities'])}")
    
    # Modality impact analysis
    print("\n" + "-" * 70)
    print("Modality Impact Analysis:")
    print("-" * 70)
    
    text_only = combos.get("text_only", {}).get("metrics", {}).get("avg_f1", 0)
    
    for combo_name, combo_data in combos.items():
        if "text" in combo_data["modalities"]:
            improvement = (combo_data["metrics"]["avg_f1"] - text_only) * 100
            additional_mods = set(combo_data["modalities"]) - {"text"}
            
            if additional_mods and text_only > 0:
                print(f"\nAdding {', '.join(additional_mods)} to text:")
                print(f"  F1 improvement: {improvement:+.2f}%")
                print(f"  Response time: {combo_data['metrics']['avg_response_time']:.3f}s")
                print(f"  Modality usage - Text: {combo_data['modality_usage']['text']:.2%}, "
                      f"Table: {combo_data['modality_usage']['table']:.2%}, "
                      f"Image: {combo_data['modality_usage']['image']:.2%}")
    
    # Recommendations
    print("\n" + "-" * 70)
    print("Recommendations:")
    print("-" * 70)
    
    quality_speed_tradeoff_idx = min(
        ((combo_name, combo_data) for combo_name, combo_data in combos.items()),
        key=lambda x: abs(x[1]["metrics"]["avg_f1"] - best_f1[1]["metrics"]["avg_f1"] * 0.95)
                     + x[1]["metrics"]["avg_response_time"] / 10
    )
    
    print(f"\n1. Best Overall Quality: {best_f1[0]}")
    print(f"   Use when answer accuracy is critical")
    
    print(f"\n2. Best for Production (Quality-Speed Tradeoff): {quality_speed_tradeoff_idx[0]}")
    print(f"   Use for real-time applications")
    
    print(f"\n3. Best for Hallucination Control: {best_grounding[0]}")
    print(f"   Use when factual accuracy and grounding matter most")


def main():
    """Run multimodal ablation study."""
    # Sample questions and references
    questions = [
        "What are the main characteristics of the system?",
        "How do different components interact?",
        "What is the performance of the system?",
        "What are the key benefits?",
        "How can the system be improved?"
    ]
    
    references = [
        "Main characteristics include efficiency, multimodal support, and hallucination reduction.",
        "Components interact through a centralized knowledge graph with Neo4j backend.",
        "System achieves F1 score of 0.90 with hallucination rate below 15%.",
        "Key benefits include reduced hallucinations, multimodal context, and semantic ranking.",
        "Improvements include better indexing, caching, and relation filtering."
    ]
    
    # Run study
    study = MultimodalAblationStudy()
    results = study.run_full_ablation(questions, references)
    
    # Save results
    os.makedirs("results", exist_ok=True)
    output_file = "results/multimodal_ablation.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.get_logger().info(f"\n✓ Results saved to {output_file}")
    
    # Analyze
    analyze_ablation_results(results)


if __name__ == "__main__":
    main()
