"""
Example of creating an interactive benchmark dashboard.

This example demonstrates how to create beautiful, animated visualizations
for SAP HANA reasoning benchmark results.
"""

import os
import time
from langchain_openai import ChatOpenAI
from langchain_hana.reasoning.benchmark_system import (
    RepresentationType,
    QuestionType,
    DataRepresentationMetrics
)
from langchain_hana.reasoning.simplified_interface import SimpleBenchmark, BenchmarkConfig
from langchain_hana.reasoning.visualization import (
    create_static_visualization,
    create_interactive_visualization,
    create_animated_dashboard
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
    output_dir = os.path.join(os.getcwd(), "benchmark_visualizations")
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    # Create static visualization (requires matplotlib)
    try:
        print("\n🖼️ Creating static visualization...")
        static_file = create_static_visualization(
            metrics,
            benchmark_name,
            output_file=os.path.join(output_dir, "static_benchmark.png"),
            show_animation=True
        )
        print(f"✅ Static visualization created: {static_file}")
    except ImportError:
        print("⚠️ Matplotlib not available. Install with 'pip install matplotlib'")
    
    # Create interactive visualization (requires plotly)
    try:
        print("\n📊 Creating interactive visualization...")
        interactive_file = create_interactive_visualization(
            metrics,
            benchmark_name,
            output_file=os.path.join(output_dir, "interactive_benchmark.html"),
            open_browser=False  # Change to True to open in browser
        )
        print(f"✅ Interactive visualization created: {interactive_file}")
    except ImportError:
        print("⚠️ Plotly not available. Install with 'pip install plotly'")
    
    # Create animated dashboard
    try:
        print("\n✨ Creating animated dashboard...")
        dashboard_file = create_animated_dashboard(
            metrics,
            benchmark_name,
            output_directory=output_dir,
            open_browser=False  # Change to True to open in browser
        )
        print(f"✅ Animated dashboard created: {dashboard_file}")
    except Exception as e:
        print(f"⚠️ Error creating animated dashboard: {str(e)}")
    
    print("\n🎉 Demonstration complete!")
    print("To view interactive visualizations, open the HTML files in your browser.")


def run_with_openai_api():
    """Run a live benchmark with OpenAI API."""
    # Check for OpenAI API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("⚠️ OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key or use run_demo() instead.")
        return
    
    print("🚀 Running live benchmark with OpenAI API")
    
    # Initialize language model
    llm = ChatOpenAI(api_key=api_key, temperature=0)
    
    # Create configuration
    config = BenchmarkConfig(
        name="Live Demo Benchmark",
        question_counts={
            "schema": 2,   # Reduced number for demo
            "instance": 2,
            "relationship": 2
        },
        representation_types=["relational", "vector", "hybrid"],
        save_results=True,
        output_file="benchmark_results_live.json"
    )
    
    # Create and run benchmark
    benchmark = SimpleBenchmark(llm=llm, config=config)
    results = benchmark.run()
    
    # Create visualizations
    metrics = results["metrics"]
    
    # Create animated dashboard
    create_animated_dashboard(
        metrics,
        config.name,
        open_browser=True
    )
    
    print("🎉 Live benchmark complete!")


if __name__ == "__main__":
    # Use demo by default (doesn't require API key)
    run_demo()
    
    # Uncomment to run with OpenAI API
    # run_with_openai_api()