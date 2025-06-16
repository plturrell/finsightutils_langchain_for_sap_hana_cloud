"""
Example of creating simple benchmark visualizations.

This example demonstrates how to create simple but functional visualizations
for SAP HANA reasoning benchmark results without complex dependencies.
"""

import os
import time
from langchain_hana.reasoning.benchmark_system import (
    RepresentationType,
    QuestionType,
    DataRepresentationMetrics
)
from langchain_hana.reasoning.visualization import (
    create_visualization,
    create_html_report
)


def create_sample_metrics():
    """Create sample metrics for demonstration."""
    # Example metrics for demonstration
    example_metrics = {
        RepresentationType.RELATIONAL: DataRepresentationMetrics(
            representation_type=RepresentationType.RELATIONAL,
            correct_count=75,
            incorrect_count=15,
            not_attempted_count=5,
            ambiguous_count=5,
            total_count=100,
            avg_response_time_ms=250.5,
            metrics_by_question_type={
                QuestionType.SCHEMA.value: {"total": 20, "correct": 18, "accuracy": 0.9},
                QuestionType.INSTANCE.value: {"total": 20, "correct": 15, "accuracy": 0.75},
                QuestionType.RELATIONSHIP.value: {"total": 20, "correct": 16, "accuracy": 0.8},
                QuestionType.AGGREGATION.value: {"total": 20, "correct": 14, "accuracy": 0.7},
                QuestionType.INFERENCE.value: {"total": 10, "correct": 7, "accuracy": 0.7},
                QuestionType.TEMPORAL.value: {"total": 10, "correct": 5, "accuracy": 0.5}
            },
            metrics_by_difficulty={
                "1": {"total": 20, "correct": 19, "accuracy": 0.95},
                "2": {"total": 20, "correct": 18, "accuracy": 0.9},
                "3": {"total": 20, "correct": 16, "accuracy": 0.8},
                "4": {"total": 20, "correct": 12, "accuracy": 0.6},
                "5": {"total": 20, "correct": 10, "accuracy": 0.5}
            }
        ),
        RepresentationType.VECTOR: DataRepresentationMetrics(
            representation_type=RepresentationType.VECTOR,
            correct_count=80,
            incorrect_count=15,
            not_attempted_count=0,
            ambiguous_count=5,
            total_count=100,
            avg_response_time_ms=180.3,
            metrics_by_question_type={
                QuestionType.SCHEMA.value: {"total": 20, "correct": 16, "accuracy": 0.8},
                QuestionType.INSTANCE.value: {"total": 20, "correct": 18, "accuracy": 0.9},
                QuestionType.RELATIONSHIP.value: {"total": 20, "correct": 15, "accuracy": 0.75},
                QuestionType.AGGREGATION.value: {"total": 20, "correct": 15, "accuracy": 0.75},
                QuestionType.INFERENCE.value: {"total": 10, "correct": 8, "accuracy": 0.8},
                QuestionType.TEMPORAL.value: {"total": 10, "correct": 8, "accuracy": 0.8}
            },
            metrics_by_difficulty={
                "1": {"total": 20, "correct": 18, "accuracy": 0.9},
                "2": {"total": 20, "correct": 17, "accuracy": 0.85},
                "3": {"total": 20, "correct": 17, "accuracy": 0.85},
                "4": {"total": 20, "correct": 16, "accuracy": 0.8},
                "5": {"total": 20, "correct": 12, "accuracy": 0.6}
            }
        ),
        RepresentationType.HYBRID: DataRepresentationMetrics(
            representation_type=RepresentationType.HYBRID,
            correct_count=85,
            incorrect_count=10,
            not_attempted_count=0,
            ambiguous_count=5,
            total_count=100,
            avg_response_time_ms=320.7,
            metrics_by_question_type={
                QuestionType.SCHEMA.value: {"total": 20, "correct": 18, "accuracy": 0.9},
                QuestionType.INSTANCE.value: {"total": 20, "correct": 17, "accuracy": 0.85},
                QuestionType.RELATIONSHIP.value: {"total": 20, "correct": 18, "accuracy": 0.9},
                QuestionType.AGGREGATION.value: {"total": 20, "correct": 16, "accuracy": 0.8},
                QuestionType.INFERENCE.value: {"total": 10, "correct": 9, "accuracy": 0.9},
                QuestionType.TEMPORAL.value: {"total": 10, "correct": 7, "accuracy": 0.7}
            },
            metrics_by_difficulty={
                "1": {"total": 20, "correct": 19, "accuracy": 0.95},
                "2": {"total": 20, "correct": 18, "accuracy": 0.9},
                "3": {"total": 20, "correct": 18, "accuracy": 0.9},
                "4": {"total": 20, "correct": 17, "accuracy": 0.85},
                "5": {"total": 20, "correct": 13, "accuracy": 0.65}
            }
        )
    }
    
    return example_metrics


def run_demo():
    """Run the demonstration."""
    print("🚀 Starting benchmark visualization demo")
    
    # Create sample metrics for demonstration
    metrics = create_sample_metrics()
    benchmark_name = "Reasoning Benchmark Demo"
    
    # Create output directory
    output_dir = os.path.join(os.getcwd(), "benchmark_results")
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    # Create matplotlib visualization if available
    try:
        print("\n🖼️ Creating visualization...")
        vis_file = create_visualization(
            metrics,
            benchmark_name,
            output_file=os.path.join(output_dir, "benchmark_visualization.png")
        )
        print(f"✅ Visualization created: {vis_file}")
    except ImportError:
        print("⚠️ Matplotlib not available. Install with 'pip install matplotlib'")
    except Exception as e:
        print(f"⚠️ Error creating visualization: {str(e)}")
    
    # Create HTML report (no external dependencies)
    try:
        print("\n📊 Creating HTML report...")
        html_file = create_html_report(
            metrics,
            benchmark_name,
            output_file=os.path.join(output_dir, "benchmark_report.html")
        )
        print(f"✅ HTML report created: {html_file}")
        print(f"   Open this file in your browser to view the report")
    except Exception as e:
        print(f"⚠️ Error creating HTML report: {str(e)}")
    
    print("\n🎉 Demonstration complete!")


if __name__ == "__main__":
    run_demo()