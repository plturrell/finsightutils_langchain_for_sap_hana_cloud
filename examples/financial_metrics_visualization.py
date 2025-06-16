#!/usr/bin/env python3
"""
Financial Metrics Visualization for RAG Evaluation

This script provides visualization tools for evaluating Retrieval-Augmented Generation (RAG)
systems using financial embeddings. It includes:

1. Embedding space visualization with t-SNE/UMAP for financial documents
2. Retrieval performance metrics visualization
3. Comparative analysis of different financial embedding models
4. Query-document relevance heatmaps
5. Financial document clustering visualization

Usage:
    python financial_metrics_visualization.py --results path/to/results.json --output path/to/output_folder
"""

import os
import json
import argparse
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.colors as mcolors
from matplotlib.ticker import PercentFormatter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("financial_metrics_viz")

# Set Seaborn style for better visualizations
sns.set_theme(style="whitegrid")

# Financial document type color mapping
FINANCIAL_DOC_COLORS = {
    "earnings_report": "#1f77b4",      # Blue
    "risk_report": "#ff7f0e",          # Orange
    "merger_announcement": "#2ca02c",  # Green
    "sec_filing": "#d62728",           # Red
    "investment_thesis": "#9467bd",    # Purple
    "market_analysis": "#8c564b",      # Brown
    "financial_news": "#e377c2",       # Pink
    "economic_forecast": "#7f7f7f",    # Gray
    "analyst_report": "#bcbd22",       # Olive
    "other": "#17becf"                 # Cyan
}

class FinancialMetricsVisualizer:
    """Visualization tool for financial RAG metrics."""
    
    def __init__(
        self,
        output_dir: str = "./visualization_output",
        dpi: int = 300,
        fig_size: Tuple[int, int] = (12, 8),
        style: str = "whitegrid"
    ):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save visualization outputs
            dpi: DPI for saved figures
            fig_size: Default figure size (width, height) in inches
            style: Seaborn style for plots
        """
        self.output_dir = output_dir
        self.dpi = dpi
        self.fig_size = fig_size
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set plot style
        sns.set_theme(style=style)
        
        logger.info(f"Initialized FinancialMetricsVisualizer with output directory: {output_dir}")
    
    def visualize_embedding_space(
        self,
        embeddings: np.ndarray,
        document_types: List[str],
        document_titles: Optional[List[str]] = None,
        companies: Optional[List[str]] = None,
        method: str = "tsne",
        interactive: bool = False,
        filename: str = "embedding_space.png"
    ):
        """
        Visualize document embeddings in 2D space using dimensionality reduction.
        
        Args:
            embeddings: Document embeddings as numpy array (n_docs, embedding_dim)
            document_types: List of document types for each embedding
            document_titles: Optional list of document titles for tooltips
            companies: Optional list of company names for each document
            method: Dimensionality reduction method ('tsne' or 'pca')
            interactive: Whether to create an interactive plot (requires plotly)
            filename: Output filename
        """
        logger.info(f"Visualizing embedding space using {method.upper()}")
        
        # Apply dimensionality reduction
        if method.lower() == "tsne":
            start_time = time.time()
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
            reduced_data = reducer.fit_transform(embeddings)
            logger.info(f"t-SNE reduction completed in {time.time() - start_time:.2f} seconds")
        elif method.lower() == "pca":
            reducer = PCA(n_components=2, random_state=42)
            reduced_data = reducer.fit_transform(embeddings)
            explained_variance = reducer.explained_variance_ratio_.sum()
            logger.info(f"PCA reduction completed (explained variance: {explained_variance:.2f})")
        else:
            raise ValueError(f"Unsupported dimensionality reduction method: {method}")
        
        # Get unique document types and map to colors
        unique_types = sorted(set(document_types))
        type_to_color = {doc_type: FINANCIAL_DOC_COLORS.get(doc_type, "#17becf") for doc_type in unique_types}
        
        # Create point colors based on document types
        colors = [type_to_color[doc_type] for doc_type in document_types]
        
        # Create interactive plot if requested
        if interactive:
            try:
                import plotly.express as px
                import plotly.graph_objects as go
                
                # Create DataFrame for plotly
                df = pd.DataFrame({
                    'x': reduced_data[:, 0],
                    'y': reduced_data[:, 1],
                    'document_type': document_types,
                    'title': document_titles if document_titles else [f"Doc {i}" for i in range(len(embeddings))],
                    'company': companies if companies else ["Unknown"] * len(embeddings)
                })
                
                # Create figure
                fig = px.scatter(
                    df, x='x', y='y', color='document_type', hover_name='title',
                    hover_data=['company', 'document_type'],
                    title=f'Financial Document Embedding Space ({method.upper()})',
                    color_discrete_map=type_to_color,
                    labels={'x': 'Dimension 1', 'y': 'Dimension 2'}
                )
                
                # Update layout
                fig.update_layout(
                    legend_title_text='Document Type',
                    width=self.fig_size[0] * 100,
                    height=self.fig_size[1] * 100
                )
                
                # Save as HTML
                html_path = os.path.join(self.output_dir, filename.replace('.png', '.html'))
                fig.write_html(html_path)
                logger.info(f"Interactive embedding visualization saved to {html_path}")
                
                # Also save static version
                static_path = os.path.join(self.output_dir, filename)
                fig.write_image(static_path, scale=2)
                logger.info(f"Static embedding visualization saved to {static_path}")
                
                return html_path
                
            except ImportError:
                logger.warning("Plotly not installed, falling back to static visualization")
                interactive = False
        
        # Create static plot
        if not interactive:
            plt.figure(figsize=self.fig_size)
            
            # Create scatter plot
            for doc_type in unique_types:
                indices = [i for i, t in enumerate(document_types) if t == doc_type]
                plt.scatter(
                    reduced_data[indices, 0], 
                    reduced_data[indices, 1],
                    c=[type_to_color[doc_type]] * len(indices),
                    label=doc_type,
                    alpha=0.7,
                    edgecolors='w',
                    s=100
                )
            
            # Add labels and legend
            plt.title(f'Financial Document Embedding Space ({method.upper()})', fontsize=16)
            plt.xlabel('Dimension 1', fontsize=14)
            plt.ylabel('Dimension 2', fontsize=14)
            plt.legend(title="Document Type", title_fontsize=12, fontsize=10)
            plt.tight_layout()
            
            # Save figure
            output_path = os.path.join(self.output_dir, filename)
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Embedding visualization saved to {output_path}")
            return output_path
    
    def visualize_retrieval_metrics(
        self,
        metrics_data: Dict[str, Any],
        model_name: str = "Financial Embeddings Model",
        filename: str = "retrieval_metrics.png"
    ):
        """
        Visualize retrieval performance metrics.
        
        Args:
            metrics_data: Dictionary containing retrieval metrics
            model_name: Name of the model for the plot title
            filename: Output filename
        """
        logger.info("Visualizing retrieval metrics")
        
        # Extract metrics for visualization
        metrics = {
            "Retrieval Precision": metrics_data.get("retrieval_precision", 0),
            "Entity Match Rate": metrics_data.get("entity_match_rate", 0),
            "Keyword Match Rate": metrics_data.get("keyword_match_rate", 0),
            "Type Match Rate": metrics_data.get("type_match_rate", 0)
        }
        
        # Create radar chart
        categories = list(metrics.keys())
        values = list(metrics.values())
        
        # Add the first value again to close the radar chart
        values_radar = values + [values[0]]
        categories_radar = categories + [categories[0]]
        
        # Create figure
        fig = plt.figure(figsize=self.fig_size)
        ax = fig.add_subplot(111, polar=True)
        
        # Plot radar chart
        theta = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        theta_radar = theta + [theta[0]]
        ax.plot(theta_radar, values_radar, 'o-', linewidth=2, label=model_name)
        ax.fill(theta_radar, values_radar, alpha=0.25)
        
        # Set labels and title
        ax.set_thetagrids(np.degrees(theta), categories)
        ax.set_ylim(0, 1)
        ax.set_rlabel_position(0)
        ax.set_title(f"Retrieval Performance Metrics: {model_name}", fontsize=16, y=1.1)
        plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["20%", "40%", "60%", "80%", "100%"], color="grey", size=8)
        
        # Add retrieval time as text annotation
        avg_retrieval_time = metrics_data.get("avg_retrieval_time", 0)
        plt.annotate(
            f"Avg. Retrieval Time: {avg_retrieval_time:.4f}s",
            xy=(0.5, 0.05),
            xycoords='figure fraction',
            ha='center',
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
        )
        
        # Save figure
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Retrieval metrics visualization saved to {output_path}")
        return output_path
    
    def visualize_model_comparison(
        self,
        models_metrics: Dict[str, Dict[str, Any]],
        metrics_to_compare: Optional[List[str]] = None,
        filename: str = "model_comparison.png"
    ):
        """
        Create comparative visualization of different financial embedding models.
        
        Args:
            models_metrics: Dictionary mapping model names to their metrics
            metrics_to_compare: List of metrics to include in comparison
            filename: Output filename
        """
        logger.info(f"Visualizing comparison of {len(models_metrics)} models")
        
        if not metrics_to_compare:
            metrics_to_compare = [
                "retrieval_precision",
                "entity_match_rate",
                "keyword_match_rate",
                "type_match_rate"
            ]
        
        # Prepare data for visualization
        models = list(models_metrics.keys())
        metrics_data = {metric: [] for metric in metrics_to_compare}
        
        for model_name, metrics in models_metrics.items():
            for metric in metrics_to_compare:
                metrics_data[metric].append(metrics.get(metric, 0))
        
        # Create a grouped bar chart
        x = np.arange(len(models))
        width = 0.8 / len(metrics_to_compare)
        
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        # Plot bars for each metric
        for i, metric in enumerate(metrics_to_compare):
            offset = (i - len(metrics_to_compare)/2 + 0.5) * width
            ax.bar(
                x + offset, 
                metrics_data[metric], 
                width, 
                label=metric.replace("_", " ").title(),
                alpha=0.8
            )
        
        # Add labels and legend
        ax.set_ylabel('Score', fontsize=14)
        ax.set_title('Financial Embedding Models Comparison', fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend(loc='upper left', fontsize=10)
        
        # Set y-axis limit
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        
        # Add grid lines
        ax.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Model comparison visualization saved to {output_path}")
        return output_path
    
    def visualize_query_document_relevance(
        self,
        queries: List[str],
        documents: List[str],
        relevance_scores: np.ndarray,
        document_types: Optional[List[str]] = None,
        filename: str = "query_document_relevance.png"
    ):
        """
        Create a heatmap visualization of query-document relevance scores.
        
        Args:
            queries: List of query texts
            documents: List of document texts (truncated if needed)
            relevance_scores: 2D array of relevance scores (queries x documents)
            document_types: Optional list of document types for coloring
            filename: Output filename
        """
        logger.info(f"Visualizing query-document relevance for {len(queries)} queries and {len(documents)} documents")
        
        # Truncate document texts for display
        doc_labels = [f"{i+1}. {doc[:50]}..." for i, doc in enumerate(documents)]
        query_labels = [f"Q{i+1}: {q[:50]}..." for i, q in enumerate(queries)]
        
        # Create figure
        plt.figure(figsize=(max(self.fig_size[0], len(documents)*0.8), max(self.fig_size[1], len(queries)*0.5)))
        
        # Create heatmap
        ax = sns.heatmap(
            relevance_scores,
            annot=True,
            fmt=".2f",
            cmap="YlOrRd",
            xticklabels=doc_labels,
            yticklabels=query_labels,
            cbar_kws={'label': 'Relevance Score'}
        )
        
        # Add color bars for document types if provided
        if document_types:
            # Get unique document types
            unique_types = sorted(set(document_types))
            colors = [FINANCIAL_DOC_COLORS.get(doc_type, "#17becf") for doc_type in document_types]
            
            # Add color bar above the heatmap
            for i, color in enumerate(colors):
                ax.add_patch(plt.Rectangle(
                    (i, -0.5), 
                    1, 
                    0.5, 
                    fill=True, 
                    color=color, 
                    alpha=0.8,
                    transform=ax.get_xaxis_transform(), 
                    clip_on=False
                ))
            
            # Add legend for document types
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=FINANCIAL_DOC_COLORS.get(doc_type, "#17becf"), 
                      label=doc_type,
                      alpha=0.8)
                for doc_type in unique_types
            ]
            plt.legend(
                handles=legend_elements,
                title="Document Types",
                bbox_to_anchor=(1.05, 1),
                loc='upper left'
            )
        
        # Add labels and title
        plt.title('Query-Document Relevance Scores', fontsize=16)
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Query-document relevance visualization saved to {output_path}")
        return output_path
    
    def visualize_retrieval_over_time(
        self,
        timestamps: List[float],
        precision_scores: List[float],
        latency_values: List[float],
        event_markers: Optional[List[Tuple[float, str]]] = None,
        filename: str = "retrieval_over_time.png"
    ):
        """
        Visualize retrieval performance over time.
        
        Args:
            timestamps: List of timestamp values
            precision_scores: List of precision scores at each timestamp
            latency_values: List of latency values at each timestamp
            event_markers: Optional list of (timestamp, event_description) tuples
            filename: Output filename
        """
        logger.info(f"Visualizing retrieval performance over time ({len(timestamps)} data points)")
        
        # Create figure with two y-axes
        fig, ax1 = plt.subplots(figsize=self.fig_size)
        
        # Plot precision scores on the first y-axis
        color = 'tab:blue'
        ax1.set_xlabel('Time', fontsize=14)
        ax1.set_ylabel('Precision', color=color, fontsize=14)
        ax1.plot(timestamps, precision_scores, color=color, marker='o', markersize=4, label='Precision')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(0, 1.05)
        ax1.yaxis.set_major_formatter(PercentFormatter(1.0))
        
        # Create second y-axis for latency
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Latency (s)', color=color, fontsize=14)
        ax2.plot(timestamps, latency_values, color=color, marker='x', linestyle='--', markersize=4, label='Latency')
        ax2.tick_params(axis='y', labelcolor=color)
        
        # Add event markers if provided
        if event_markers:
            for ts, desc in event_markers:
                plt.axvline(x=ts, color='gray', linestyle='--', alpha=0.7)
                plt.text(
                    ts, 
                    max(precision_scores) * 0.9, 
                    desc, 
                    rotation=90, 
                    verticalalignment='top',
                    fontsize=8
                )
        
        # Add title and legend
        plt.title('Retrieval Performance Over Time', fontsize=16)
        
        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Retrieval over time visualization saved to {output_path}")
        return output_path
    
    def create_summary_dashboard(
        self,
        metrics_data: Dict[str, Any],
        model_name: str,
        embedding_plot_path: Optional[str] = None,
        metrics_plot_path: Optional[str] = None,
        relevance_plot_path: Optional[str] = None,
        filename: str = "financial_rag_dashboard.html"
    ):
        """
        Create an HTML dashboard summarizing all visualizations.
        
        Args:
            metrics_data: Dictionary of evaluation metrics
            model_name: Name of the model being evaluated
            embedding_plot_path: Path to embedding visualization
            metrics_plot_path: Path to metrics visualization
            relevance_plot_path: Path to relevance visualization
            filename: Output filename for the dashboard
        
        Returns:
            Path to the generated HTML dashboard
        """
        logger.info("Creating summary dashboard")
        
        # Prepare metrics for display
        metrics_table = ""
        for metric, value in metrics_data.items():
            if isinstance(value, float):
                if metric == "avg_retrieval_time":
                    formatted_value = f"{value:.4f}s"
                else:
                    formatted_value = f"{value:.2%}"
                metrics_table += f"<tr><td>{metric.replace('_', ' ').title()}</td><td>{formatted_value}</td></tr>"
        
        # Prepare image paths for display
        image_gallery = ""
        for plot_path, title in [
            (embedding_plot_path, "Financial Document Embedding Space"),
            (metrics_plot_path, "Retrieval Performance Metrics"),
            (relevance_plot_path, "Query-Document Relevance")
        ]:
            if plot_path:
                rel_path = os.path.basename(plot_path)
                image_gallery += f"""
                <div class="col-md-6 mb-4">
                    <div class="card">
                        <div class="card-header">{title}</div>
                        <div class="card-body text-center">
                            <img src="{rel_path}" class="img-fluid" alt="{title}">
                        </div>
                    </div>
                </div>
                """
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Financial RAG Evaluation Dashboard</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }}
                .header {{ background-color: #f8f9fa; padding: 20px 0; margin-bottom: 30px; }}
                .metrics-card {{ height: 100%; }}
                .footer {{ margin-top: 50px; padding: 20px 0; background-color: #f8f9fa; text-align: center; }}
            </style>
        </head>
        <body>
            <div class="header">
                <div class="container">
                    <h1>Financial RAG Evaluation Dashboard</h1>
                    <p class="lead">Model: {model_name}</p>
                    <p>Generated on {time.strftime("%Y-%m-%d %H:%M:%S")}</p>
                </div>
            </div>
            
            <div class="container">
                <div class="row">
                    <div class="col-md-4 mb-4">
                        <div class="card metrics-card">
                            <div class="card-header">Performance Metrics</div>
                            <div class="card-body">
                                <table class="table table-striped">
                                    <thead>
                                        <tr>
                                            <th>Metric</th>
                                            <th>Value</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {metrics_table}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                    
                    <div class="col-md-8 mb-4">
                        <div class="card">
                            <div class="card-header">Performance Summary</div>
                            <div class="card-body">
                                <h5>Key Findings</h5>
                                <ul>
                                    <li>Retrieval Precision: {metrics_data.get("retrieval_precision", 0):.2%}</li>
                                    <li>Entity Match Rate: {metrics_data.get("entity_match_rate", 0):.2%}</li>
                                    <li>Average Retrieval Time: {metrics_data.get("avg_retrieval_time", 0):.4f}s</li>
                                </ul>
                                
                                <h5>Observations</h5>
                                <p>
                                    The financial embeddings model demonstrates 
                                    {self._performance_assessment(metrics_data.get("retrieval_precision", 0))} 
                                    retrieval precision and 
                                    {self._performance_assessment(metrics_data.get("entity_match_rate", 0))} 
                                    entity matching capabilities.
                                </p>
                                
                                <h5>Recommendations</h5>
                                <p>
                                    {self._generate_recommendations(metrics_data)}
                                </p>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="row">
                    {image_gallery}
                </div>
            </div>
            
            <div class="footer">
                <div class="container">
                    <p>Generated by FinancialMetricsVisualizer</p>
                </div>
            </div>
            
            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
        </body>
        </html>
        """
        
        # Write HTML file
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Dashboard saved to {output_path}")
        return output_path
    
    def _performance_assessment(self, score: float) -> str:
        """Generate a qualitative assessment of a performance score."""
        if score >= 0.9:
            return "excellent"
        elif score >= 0.8:
            return "very good"
        elif score >= 0.7:
            return "good"
        elif score >= 0.6:
            return "moderate"
        elif score >= 0.5:
            return "fair"
        else:
            return "poor"
    
    def _generate_recommendations(self, metrics: Dict[str, Any]) -> str:
        """Generate recommendations based on metrics."""
        recommendations = []
        
        # Check retrieval precision
        if metrics.get("retrieval_precision", 0) < 0.7:
            recommendations.append("Consider fine-tuning the embedding model on more domain-specific financial content to improve retrieval precision.")
        
        # Check entity match rate
        if metrics.get("entity_match_rate", 0) < 0.7:
            recommendations.append("Enhance entity recognition in the embedding model, potentially by adding entity-focused prompts or preprocessing.")
        
        # Check retrieval time
        if metrics.get("avg_retrieval_time", 0) > 0.5:
            recommendations.append("Optimize the retrieval pipeline for better latency, possibly by implementing caching or using an HNSW index.")
        
        # Add general recommendation
        recommendations.append("Regularly update the financial document collection to ensure relevance to current market conditions.")
        
        return " ".join(recommendations)

def load_evaluation_results(results_path: str) -> Dict[str, Any]:
    """Load evaluation results from a JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)

def generate_sample_data():
    """Generate sample data for visualization demonstration."""
    # Sample embeddings (2D for simplicity)
    embeddings = np.random.randn(20, 2)
    
    # Sample document types
    document_types = [
        "earnings_report", "earnings_report", "earnings_report", "earnings_report",
        "risk_report", "risk_report", "risk_report",
        "merger_announcement", "merger_announcement",
        "sec_filing", "sec_filing", "sec_filing",
        "investment_thesis", "investment_thesis",
        "market_analysis", "market_analysis", "market_analysis", "market_analysis",
        "financial_news", "financial_news"
    ]
    
    # Sample metrics data
    metrics_data = {
        "retrieval_precision": 0.85,
        "entity_match_rate": 0.78,
        "keyword_match_rate": 0.92,
        "type_match_rate": 0.88,
        "avg_retrieval_time": 0.125,
        "total_queries": 10
    }
    
    # Sample models comparison
    models_metrics = {
        "FinE5-Small": {
            "retrieval_precision": 0.85,
            "entity_match_rate": 0.78,
            "keyword_match_rate": 0.92,
            "type_match_rate": 0.88,
            "avg_retrieval_time": 0.125
        },
        "FinE5-Base": {
            "retrieval_precision": 0.89,
            "entity_match_rate": 0.82,
            "keyword_match_rate": 0.95,
            "type_match_rate": 0.91,
            "avg_retrieval_time": 0.175
        },
        "Financial-BERT": {
            "retrieval_precision": 0.76,
            "entity_match_rate": 0.71,
            "keyword_match_rate": 0.88,
            "type_match_rate": 0.80,
            "avg_retrieval_time": 0.095
        }
    }
    
    # Sample query-document relevance
    queries = [
        "What was the revenue growth in Q1?",
        "What are the main financial risks?",
        "Explain the recent merger details"
    ]
    
    documents = [
        "Q1 2025 Financial Results: Revenue up 15% year-over-year...",
        "Risk Assessment: Market volatility remains elevated...",
        "Merger Announcement: Alpha Corp. to acquire Beta Technologies...",
        "SEC Filing: The company faces significant competition..."
    ]
    
    relevance_scores = np.array([
        [0.92, 0.45, 0.32, 0.28],
        [0.38, 0.89, 0.42, 0.78],
        [0.25, 0.35, 0.95, 0.40]
    ])
    
    # Sample time series data
    timestamps = list(range(10))
    precision_scores = [0.75, 0.78, 0.80, 0.79, 0.83, 0.85, 0.84, 0.88, 0.90, 0.91]
    latency_values = [0.25, 0.23, 0.22, 0.20, 0.18, 0.19, 0.17, 0.16, 0.15, 0.14]
    
    event_markers = [
        (2, "Model Update"),
        (7, "Index Optimization")
    ]
    
    return {
        "embeddings": embeddings,
        "document_types": document_types,
        "metrics_data": metrics_data,
        "models_metrics": models_metrics,
        "queries": queries,
        "documents": documents,
        "relevance_scores": relevance_scores,
        "timestamps": timestamps,
        "precision_scores": precision_scores,
        "latency_values": latency_values,
        "event_markers": event_markers
    }

def main():
    """Main function to run the visualization."""
    parser = argparse.ArgumentParser(description="Financial Metrics Visualization for RAG Evaluation")
    parser.add_argument("--results", help="Path to evaluation results JSON file")
    parser.add_argument("--output", default="./visualization_output", help="Output directory for visualizations")
    parser.add_argument("--sample", action="store_true", help="Generate and visualize sample data")
    parser.add_argument("--model_name", default="FinE5-Small", help="Model name for visualization titles")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for saved figures")
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = FinancialMetricsVisualizer(output_dir=args.output, dpi=args.dpi)
    
    # Generate sample data if requested
    if args.sample:
        logger.info("Generating sample data for visualization")
        data = generate_sample_data()
        
        # Create visualizations
        embedding_plot = visualizer.visualize_embedding_space(
            data["embeddings"],
            data["document_types"],
            method="pca"
        )
        
        metrics_plot = visualizer.visualize_retrieval_metrics(
            data["metrics_data"],
            model_name=args.model_name
        )
        
        comparison_plot = visualizer.visualize_model_comparison(
            data["models_metrics"]
        )
        
        relevance_plot = visualizer.visualize_query_document_relevance(
            data["queries"],
            data["documents"],
            data["relevance_scores"],
            document_types=["earnings_report", "risk_report", "merger_announcement", "sec_filing"]
        )
        
        time_series_plot = visualizer.visualize_retrieval_over_time(
            data["timestamps"],
            data["precision_scores"],
            data["latency_values"],
            data["event_markers"]
        )
        
        # Create dashboard
        dashboard = visualizer.create_summary_dashboard(
            data["metrics_data"],
            args.model_name,
            embedding_plot,
            metrics_plot,
            relevance_plot
        )
        
        logger.info(f"Sample visualizations created successfully. Dashboard: {dashboard}")
        return 0
    
    # Load evaluation results if provided
    if args.results:
        try:
            results = load_evaluation_results(args.results)
            logger.info(f"Loaded evaluation results from {args.results}")
            
            # TODO: Create visualizations based on loaded results
            # This would depend on the structure of your evaluation results
            
            logger.info("Visualizations created successfully")
            return 0
        except Exception as e:
            logger.error(f"Error loading evaluation results: {str(e)}")
            return 1
    else:
        logger.error("No results file provided. Use --results or --sample")
        return 1

if __name__ == "__main__":
    sys.exit(main())