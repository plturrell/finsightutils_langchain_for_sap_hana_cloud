"""
Visualization tools for benchmark results.

This module provides visualization tools for benchmark results with a focus
on simplicity and functionality.
"""

import os
import time
import json
import logging
from typing import Dict, List, Any, Optional, Union

from langchain_hana.reasoning.benchmark_system import (
    RepresentationType,
    QuestionType,
    DataRepresentationMetrics
)

logger = logging.getLogger(__name__)

# Check for matplotlib availability
try:
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib not available. Install with 'pip install matplotlib'")


def create_visualization(
    metrics: Dict[RepresentationType, DataRepresentationMetrics],
    benchmark_name: str,
    output_file: Optional[str] = None
) -> str:
    """
    Create a simple visualization of benchmark results.
    
    Args:
        metrics: Dictionary of metrics by representation type
        benchmark_name: Name of the benchmark
        output_file: Output file path (default: benchmark_viz_{timestamp}.png)
        
    Returns:
        Path to saved visualization file
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib is required for visualizations")
    
    # Determine output file
    if output_file is None:
        output_file = f"benchmark_viz_{time.strftime('%Y%m%d_%H%M%S')}.png"
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=100)
    fig.suptitle(f"Benchmark Results: {benchmark_name}", fontweight='bold', fontsize=16)
    
    # Define colors for representation types
    colors = {
        "relational": "#007AFF",  # Blue
        "vector": "#FF9500",      # Orange
        "ontology": "#4CD964",    # Green
        "hybrid": "#AF52DE"       # Purple
    }
    
    # Representation types and their metrics
    representation_types = [rt.value for rt in metrics.keys()]
    representation_colors = [colors.get(rt, "#999999") for rt in representation_types]
    
    # 1. Accuracy by representation type
    ax1 = axes[0, 0]
    accuracies = [metrics[rt].accuracy for rt in metrics.keys()]
    
    bars = ax1.bar(
        representation_types, 
        accuracies, 
        color=representation_colors,
        width=0.6,
        edgecolor='white',
        linewidth=1
    )
    
    # Add data labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.01,
            f"{height:.2%}",
            ha='center', 
            va='bottom',
            fontweight='bold'
        )
    
    ax1.set_title("Accuracy by Representation Method")
    ax1.set_ylim(0, 1.1)
    ax1.set_ylabel("Accuracy")
    ax1.set_xlabel("Representation Method")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    
    # 2. Performance by question type
    ax2 = axes[0, 1]
    
    # Extract and prepare data
    question_types = set()
    data_by_rep_type = {}
    
    for rt, rt_metrics in metrics.items():
        data_by_rep_type[rt.value] = {}
        for qt, qt_metrics in rt_metrics.metrics_by_question_type.items():
            question_types.add(qt)
            data_by_rep_type[rt.value][qt] = qt_metrics.get("accuracy", 0)
    
    question_types = sorted(question_types)
    
    # Create grouped bars
    x = np.arange(len(question_types))
    width = 0.8 / len(data_by_rep_type)
    
    i = 0
    for rt, qt_data in data_by_rep_type.items():
        values = [qt_data.get(qt, 0) for qt in question_types]
        offset = i * width - (len(data_by_rep_type) - 1) * width / 2
        ax2.bar(x + offset, values, width, label=rt.capitalize(), color=colors.get(rt, "#999999"))
        i += 1
    
    ax2.set_title("Performance by Question Type")
    ax2.set_xticks(x)
    ax2.set_xticklabels([qt.capitalize() for qt in question_types], rotation=45, ha='right')
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax2.legend(title="Representation Method")
    
    # 3. Performance by difficulty
    ax3 = axes[1, 0]
    
    # Extract and prepare data
    difficulty_levels = set()
    data_by_rep_type_diff = {}
    
    for rt, rt_metrics in metrics.items():
        data_by_rep_type_diff[rt.value] = {}
        for diff, diff_metrics in rt_metrics.metrics_by_difficulty.items():
            difficulty_levels.add(int(diff))
            data_by_rep_type_diff[rt.value][int(diff)] = diff_metrics.get("accuracy", 0)
    
    difficulty_levels = sorted(difficulty_levels)
    
    # Create grouped bars
    x = np.arange(len(difficulty_levels))
    width = 0.8 / len(data_by_rep_type_diff)
    
    i = 0
    for rt, diff_data in data_by_rep_type_diff.items():
        values = [diff_data.get(diff, 0) for diff in difficulty_levels]
        offset = i * width - (len(data_by_rep_type_diff) - 1) * width / 2
        ax3.bar(x + offset, values, width, label=rt.capitalize(), color=colors.get(rt, "#999999"))
        i += 1
    
    ax3.set_title("Performance by Difficulty Level")
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"Level {diff}" for diff in difficulty_levels])
    ax3.set_ylabel("Accuracy")
    ax3.set_ylim(0, 1)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax3.legend(title="Representation Method")
    
    # 4. Performance metrics comparison
    ax4 = axes[1, 1]
    
    metrics_to_show = ["Accuracy", "F-Score", "Attempted"]
    metrics_values = {
        "Accuracy": [metrics[rt].accuracy for rt in metrics.keys()],
        "F-Score": [metrics[rt].f_score for rt in metrics.keys()],
        "Attempted": [metrics[rt].attempted_accuracy for rt in metrics.keys()]
    }
    
    x = np.arange(len(metrics_to_show))
    width = 0.8 / len(representation_types)
    
    i = 0
    for j, rt in enumerate(metrics.keys()):
        values = [
            metrics[rt].accuracy,
            metrics[rt].f_score,
            metrics[rt].attempted_accuracy
        ]
        offset = i * width - (len(representation_types) - 1) * width / 2
        ax4.bar(x + offset, values, width, label=rt.value.capitalize(), color=colors.get(rt.value, "#999999"))
        i += 1
    
    ax4.set_title("Performance Metrics Comparison")
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics_to_show)
    ax4.set_ylabel("Value")
    ax4.set_ylim(0, 1)
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax4.legend(title="Representation Method")
    
    # Add a watermark
    fig.text(0.99, 0.01, "SAP HANA Reasoning Benchmark", ha='right', va='bottom', 
             color='gray', fontsize=8, alpha=0.5)
    
    # Improve layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the figure
    plt.savefig(output_file, dpi=100, bbox_inches='tight')
    logger.info(f"Visualization saved to {output_file}")
    
    # Return the output file path
    return output_file


def create_html_report(
    metrics: Dict[RepresentationType, DataRepresentationMetrics],
    benchmark_name: str,
    output_file: Optional[str] = None
) -> str:
    """
    Create a simple HTML report for benchmark results.
    
    Args:
        metrics: Dictionary of metrics by representation type
        benchmark_name: Name of the benchmark
        output_file: Output file path (default: benchmark_report_{timestamp}.html)
        
    Returns:
        Path to saved HTML file
    """
    # Determine output file
    if output_file is None:
        output_file = f"benchmark_report_{time.strftime('%Y%m%d_%H%M%S')}.html"
    
    # Simple HTML template
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Benchmark Results: {benchmark_name}</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f7;
                color: #1d1d1f;
            }
            .container {
                max-width: 1000px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 {
                text-align: center;
                margin-bottom: 30px;
            }
            .summary {
                margin-bottom: 30px;
                padding: 20px;
                background-color: #f5f5f7;
                border-radius: 10px;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 30px;
            }
            th, td {
                padding: 12px 15px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }
            th {
                background-color: #f8f8f8;
                font-weight: 600;
            }
            .footer {
                text-align: center;
                margin-top: 30px;
                color: #86868b;
                font-size: 12px;
            }
            .metric-good {
                color: #34c759;
                font-weight: bold;
            }
            .metric-average {
                color: #ff9500;
                font-weight: bold;
            }
            .metric-poor {
                color: #ff3b30;
                font-weight: bold;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Benchmark Results: {benchmark_name}</h1>
            
            <div class="summary">
                <h2>Summary</h2>
                <p>Benchmark compared {representation_count} representation methods: {representation_list}.</p>
                <p>Best performance: <strong>{best_representation}</strong> with {best_accuracy}% accuracy.</p>
                <p>Total questions: {total_questions}</p>
            </div>
            
            <h2>Accuracy by Representation Method</h2>
            <table>
                <tr>
                    <th>Representation</th>
                    <th>Accuracy</th>
                    <th>F-Score</th>
                    <th>Correct</th>
                    <th>Incorrect</th>
                    <th>Not Attempted</th>
                    <th>Ambiguous</th>
                </tr>
                {accuracy_rows}
            </table>
            
            <h2>Performance by Question Type</h2>
            <table>
                <tr>
                    <th>Question Type</th>
                    {representation_headers}
                </tr>
                {question_type_rows}
            </table>
            
            <h2>Performance by Difficulty Level</h2>
            <table>
                <tr>
                    <th>Difficulty</th>
                    {representation_headers}
                </tr>
                {difficulty_rows}
            </table>
            
            <div class="footer">
                <p>© 2023 SAP HANA Reasoning Benchmark System</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Prepare data for template
    representation_types = [rt.value.capitalize() for rt in metrics.keys()]
    representation_list = ", ".join(representation_types)
    representation_count = len(representation_types)
    
    # Find best representation
    best_representation = ""
    best_accuracy = 0
    total_questions = 0
    
    for rt, rt_metrics in metrics.items():
        if rt_metrics.accuracy > best_accuracy:
            best_accuracy = rt_metrics.accuracy
            best_representation = rt.value.capitalize()
        if total_questions == 0:
            total_questions = rt_metrics.total_count
    
    # Format best accuracy as percentage
    best_accuracy_pct = f"{best_accuracy:.1%}"
    
    # Generate table rows
    accuracy_rows = ""
    for rt, rt_metrics in metrics.items():
        accuracy_class = "metric-good" if rt_metrics.accuracy >= 0.8 else "metric-average" if rt_metrics.accuracy >= 0.6 else "metric-poor"
        accuracy_rows += f"""
        <tr>
            <td>{rt.value.capitalize()}</td>
            <td class="{accuracy_class}">{rt_metrics.accuracy:.1%}</td>
            <td>{rt_metrics.f_score:.1%}</td>
            <td>{rt_metrics.correct_count}</td>
            <td>{rt_metrics.incorrect_count}</td>
            <td>{rt_metrics.not_attempted_count}</td>
            <td>{rt_metrics.ambiguous_count}</td>
        </tr>
        """
    
    # Generate representation headers
    representation_headers = ""
    for rt in metrics.keys():
        representation_headers += f"<th>{rt.value.capitalize()}</th>"
    
    # Gather all question types
    question_types = set()
    for rt, rt_metrics in metrics.items():
        for qt in rt_metrics.metrics_by_question_type.keys():
            question_types.add(qt)
    question_types = sorted(question_types)
    
    # Generate question type rows
    question_type_rows = ""
    for qt in question_types:
        question_type_rows += f"<tr><td>{qt.capitalize()}</td>"
        for rt, rt_metrics in metrics.items():
            if qt in rt_metrics.metrics_by_question_type:
                accuracy = rt_metrics.metrics_by_question_type[qt].get("accuracy", 0)
                accuracy_class = "metric-good" if accuracy >= 0.8 else "metric-average" if accuracy >= 0.6 else "metric-poor"
                question_type_rows += f'<td class="{accuracy_class}">{accuracy:.1%}</td>'
            else:
                question_type_rows += "<td>N/A</td>"
        question_type_rows += "</tr>"
    
    # Gather all difficulty levels
    difficulty_levels = set()
    for rt, rt_metrics in metrics.items():
        for diff in rt_metrics.metrics_by_difficulty.keys():
            difficulty_levels.add(int(diff))
    difficulty_levels = sorted(difficulty_levels)
    
    # Generate difficulty rows
    difficulty_rows = ""
    for diff in difficulty_levels:
        difficulty_rows += f"<tr><td>Level {diff}</td>"
        for rt, rt_metrics in metrics.items():
            if str(diff) in rt_metrics.metrics_by_difficulty:
                accuracy = rt_metrics.metrics_by_difficulty[str(diff)].get("accuracy", 0)
                accuracy_class = "metric-good" if accuracy >= 0.8 else "metric-average" if accuracy >= 0.6 else "metric-poor"
                difficulty_rows += f'<td class="{accuracy_class}">{accuracy:.1%}</td>'
            else:
                difficulty_rows += "<td>N/A</td>"
        difficulty_rows += "</tr>"
    
    # Fill the template
    html_content = html_template.format(
        benchmark_name=benchmark_name,
        representation_count=representation_count,
        representation_list=representation_list,
        best_representation=best_representation,
        best_accuracy=best_accuracy_pct,
        total_questions=total_questions,
        accuracy_rows=accuracy_rows,
        representation_headers=representation_headers,
        question_type_rows=question_type_rows,
        difficulty_rows=difficulty_rows
    )
    
    # Write the HTML file
    with open(output_file, "w") as f:
        f.write(html_content)
    
    logger.info(f"HTML report saved to {output_file}")
    
    return output_file


if __name__ == "__main__":
    # Example metrics for testing
    from langchain_hana.reasoning.benchmark_system import (
        RepresentationType,
        QuestionType,
        DataRepresentationMetrics
    )
    
    # Create example metrics
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
    
    # Create visualizations if matplotlib is available
    try:
        if MATPLOTLIB_AVAILABLE:
            vis_file = create_visualization(example_metrics, "Example Benchmark")
            print(f"Created visualization: {vis_file}")
        else:
            print("Matplotlib not available. Skipping visualization.")
    except Exception as e:
        print(f"Error creating visualization: {str(e)}")
    
    # Create HTML report (always available)
    try:
        html_file = create_html_report(example_metrics, "Example Benchmark")
        print(f"Created HTML report: {html_file}")
    except Exception as e:
        print(f"Error creating HTML report: {str(e)}")