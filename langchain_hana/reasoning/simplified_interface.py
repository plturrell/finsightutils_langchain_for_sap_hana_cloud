"""
Simplified interface for the benchmark system.

This module provides a streamlined interface to the benchmark system with
sensible defaults and minimal configuration requirements, designed for 
ease of use while maintaining full functionality.
"""

import os
import time
import uuid
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass

from langchain_core.language_models import BaseLanguageModel

from langchain_hana.reasoning.benchmark_system import (
    BenchmarkQuestion,
    BenchmarkAnswer,
    DataRepresentationMetrics, 
    QuestionType,
    RepresentationType,
    QuestionGenerator,
    HanaStorageManager,
    FactualityGrade
)
from langchain_hana.reasoning.factuality import FactualityEvaluator
from langchain_hana.reasoning.benchmark_system_part2 import (
    BenchmarkRunner,
    RelationalRepresentationHandler,
    VectorRepresentationHandler,
    OntologyRepresentationHandler,
    HybridRepresentationHandler
)

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """
    Simple configuration for benchmark runs.
    
    This class provides a simplified configuration interface with
    sensible defaults for common benchmark scenarios.
    """
    # Basic identification
    name: str = f"Benchmark-{time.strftime('%Y%m%d-%H%M')}"
    description: Optional[str] = None
    
    # Database settings - optional if using a connection directly
    host: Optional[str] = None
    port: Optional[int] = None
    user: Optional[str] = None
    password: Optional[str] = None
    schema: str = "SALES"
    
    # Question generation settings
    question_counts: Dict[str, int] = None  # If None, uses balanced defaults
    
    # Representation methods to benchmark
    representation_types: List[str] = None  # If None, benchmarks all types
    
    # Storage settings
    storage_schema: str = "BENCHMARK"
    
    # Output settings
    save_results: bool = True
    output_file: Optional[str] = None
    
    def __post_init__(self):
        """Initialize with sensible defaults."""
        # Set default question counts if not provided
        if self.question_counts is None:
            self.question_counts = {
                "schema": 5,
                "instance": 5,
                "relationship": 5,
                "aggregation": 5,
                "inference": 3,
                "temporal": 3
            }
        
        # Set default representation types if not provided
        if self.representation_types is None:
            self.representation_types = [
                "relational",
                "vector",
                "hybrid"
            ]
        
        # Set default output file if saving results
        if self.save_results and self.output_file is None:
            self.output_file = f"benchmark_results_{time.strftime('%Y%m%d_%H%M%S')}.json"


class SimpleBenchmark:
    """
    Simplified interface for the SAP HANA reasoning benchmark system.
    
    This class provides a streamlined interface with:
    - One-line setup
    - Sensible defaults
    - Minimal required configuration
    - Elegant error handling
    - Seamless integration with existing workflows
    """
    
    def __init__(
        self,
        llm: BaseLanguageModel,
        connection=None,  # Can be None if using connection params
        config: Optional[BenchmarkConfig] = None
    ):
        """
        Initialize the benchmark system with minimal configuration.
        
        Args:
            llm: Language model for question generation and evaluation
            connection: Optional database connection (if not provided, will create from config)
            config: Optional configuration (uses defaults if not provided)
        """
        self.llm = llm
        self.connection = connection
        self.config = config or BenchmarkConfig()
        
        # Initialize core components
        self._initialize_components()
        
        logger.info(f"Initialized SimpleBenchmark with name: {self.config.name}")
    
    def _initialize_components(self):
        """Initialize benchmark system components with sensible defaults."""
        # Create connection if needed
        if self.connection is None and all([self.config.host, self.config.port, self.config.user, self.config.password]):
            try:
                from hdbcli import dbapi
                self.connection = dbapi.connect(
                    address=self.config.host,
                    port=self.config.port,
                    user=self.config.user,
                    password=self.config.password
                )
                logger.info(f"Connected to SAP HANA at {self.config.host}:{self.config.port}")
            except ImportError:
                raise ImportError("hdbcli package is required when not providing a connection")
            except Exception as e:
                raise ConnectionError(f"Failed to connect to SAP HANA: {str(e)}")
        
        # Initialize question generator
        self.question_generator = QuestionGenerator(
            generation_model=self.llm,
            model_id=getattr(self.llm, "model_name", "benchmark-model")
        )
        
        # Initialize storage manager if connection is available
        if self.connection:
            self.storage_manager = HanaStorageManager(
                connection=self.connection,
                schema_name=self.config.storage_schema
            )
        else:
            self.storage_manager = None
            logger.warning("No connection provided. Results will not be stored persistently.")
        
        # Initialize evaluator
        self.evaluator = FactualityEvaluator(
            grading_model=self.llm,
            model_id=getattr(self.llm, "model_name", "evaluator-model")
        )
    
    def _prepare_handlers(self, schema_info: str, data_info: str, vector_store=None):
        """Prepare representation handlers based on configuration."""
        # Ensure we have a connection
        if not self.connection:
            raise ValueError("Database connection is required for benchmark handlers")
        
        # Initialize handlers based on requested representation types
        handlers = {}
        
        if "relational" in self.config.representation_types:
            handlers[RepresentationType.RELATIONAL] = RelationalRepresentationHandler(
                connection=self.connection,
                schema_name=self.config.schema,
                llm=self.llm
            )
        
        if "vector" in self.config.representation_types:
            # If vector_store not provided, create a simple in-memory one
            if vector_store is None:
                logger.warning("No vector store provided, using a simple in-memory one")
                from langchain.vectorstores import FAISS
                from langchain_openai import OpenAIEmbeddings
                
                try:
                    embeddings = OpenAIEmbeddings()
                    vector_store = FAISS.from_texts(
                        ["This is a placeholder vector store for testing"],
                        embeddings
                    )
                except Exception as e:
                    logger.error(f"Failed to create test vector store: {str(e)}")
                    raise
            
            # Create vector representation handler
            handlers[RepresentationType.VECTOR] = VectorRepresentationHandler(
                connection=self.connection,
                schema_name=self.config.schema,
                vector_store=vector_store,
                embeddings=getattr(vector_store, "embedding_function", None),
                llm=self.llm
            )
        
        if "ontology" in self.config.representation_types:
            handlers[RepresentationType.ONTOLOGY] = OntologyRepresentationHandler(
                connection=self.connection,
                schema_name=self.config.schema,
                sparql_endpoint="http://example.com/sparql",  # Mock endpoint
                llm=self.llm
            )
        
        if "hybrid" in self.config.representation_types and len(handlers) >= 2:
            # Create hybrid handler with available handlers
            component_handlers = []
            if RepresentationType.RELATIONAL in handlers:
                component_handlers.append(handlers[RepresentationType.RELATIONAL])
            if RepresentationType.VECTOR in handlers:
                component_handlers.append(handlers[RepresentationType.VECTOR])
            
            handlers[RepresentationType.HYBRID] = HybridRepresentationHandler(
                connection=self.connection,
                schema_name=self.config.schema,
                handlers=component_handlers,
                llm=self.llm
            )
        
        return handlers
    
    def _extract_schema_info(self):
        """Extract schema information from the database."""
        if not self.connection:
            raise ValueError("Database connection is required to extract schema info")
        
        try:
            cursor = self.connection.cursor()
            
            # Get tables
            cursor.execute(f"""
            SELECT TABLE_NAME, COMMENTS
            FROM SYS.TABLES
            WHERE SCHEMA_NAME = '{self.config.schema}'
            ORDER BY TABLE_NAME
            """)
            
            tables = cursor.fetchall()
            
            schema_info = f"Schema: {self.config.schema}\n\n"
            schema_info += "Tables:\n"
            
            for table_name, comments in tables:
                schema_info += f"- {table_name}"
                if comments:
                    schema_info += f" ({comments})"
                schema_info += "\n"
                
                # Get columns for this table
                cursor.execute(f"""
                SELECT COLUMN_NAME, DATA_TYPE_NAME, LENGTH, SCALE, IS_NULLABLE, COMMENTS, POSITION
                FROM SYS.TABLE_COLUMNS
                WHERE SCHEMA_NAME = '{self.config.schema}' AND TABLE_NAME = '{table_name}'
                ORDER BY POSITION
                """)
                
                columns = cursor.fetchall()
                
                schema_info += "  Columns:\n"
                for col_name, data_type, length, scale, is_nullable, col_comments, position in columns:
                    nullable = "NULL" if is_nullable == "TRUE" else "NOT NULL"
                    schema_info += f"  - {col_name}: {data_type}"
                    
                    if data_type in ["VARCHAR", "NVARCHAR", "CHAR", "NCHAR"]:
                        schema_info += f"({length})"
                    elif data_type in ["DECIMAL"]:
                        schema_info += f"({length},{scale})"
                    
                    schema_info += f" {nullable}"
                    
                    if col_comments:
                        schema_info += f" ({col_comments})"
                    
                    schema_info += "\n"
                
                # Get primary key
                cursor.execute(f"""
                SELECT COLUMN_NAME
                FROM SYS.CONSTRAINTS AS C
                JOIN SYS.CONSTRAINT_COLUMNS AS CC ON C.SCHEMA_NAME = CC.SCHEMA_NAME AND C.TABLE_NAME = CC.TABLE_NAME AND C.CONSTRAINT_NAME = CC.CONSTRAINT_NAME
                WHERE C.SCHEMA_NAME = '{self.config.schema}' AND C.TABLE_NAME = '{table_name}' AND C.IS_PRIMARY_KEY = 'TRUE'
                ORDER BY CC.POSITION
                """)
                
                pk_columns = cursor.fetchall()
                
                if pk_columns:
                    pk_cols = [col[0] for col in pk_columns]
                    schema_info += f"  Primary Key: {', '.join(pk_cols)}\n"
                
                # Get foreign keys
                cursor.execute(f"""
                SELECT C.CONSTRAINT_NAME, CC.COLUMN_NAME, C.REFERENCED_SCHEMA_NAME, C.REFERENCED_TABLE_NAME, C.REFERENCED_COLUMN_NAME
                FROM SYS.CONSTRAINTS AS C
                JOIN SYS.CONSTRAINT_COLUMNS AS CC ON C.SCHEMA_NAME = CC.SCHEMA_NAME AND C.TABLE_NAME = CC.TABLE_NAME AND C.CONSTRAINT_NAME = CC.CONSTRAINT_NAME
                WHERE C.SCHEMA_NAME = '{self.config.schema}' AND C.TABLE_NAME = '{table_name}' AND C.IS_FOREIGN_KEY = 'TRUE'
                ORDER BY C.CONSTRAINT_NAME, CC.POSITION
                """)
                
                fk_constraints = cursor.fetchall()
                
                if fk_constraints:
                    schema_info += "  Foreign Keys:\n"
                    current_constraint = None
                    fk_columns = []
                    
                    for constraint_name, column_name, ref_schema, ref_table, ref_column in fk_constraints:
                        if constraint_name != current_constraint:
                            if current_constraint:
                                schema_info += f"    - {', '.join(fk_columns)} -> {ref_schema}.{ref_table}.{ref_column}\n"
                                fk_columns = []
                            current_constraint = constraint_name
                        
                        fk_columns.append(column_name)
                    
                    if fk_columns:
                        schema_info += f"    - {', '.join(fk_columns)} -> {ref_schema}.{ref_table}.{ref_column}\n"
                
                schema_info += "\n"
            
            cursor.close()
            
            return schema_info
        
        except Exception as e:
            logger.error(f"Error extracting schema info: {str(e)}")
            raise
    
    def _extract_data_info(self, max_rows_per_table=5):
        """Extract data information from the database."""
        if not self.connection:
            raise ValueError("Database connection is required to extract data info")
        
        try:
            cursor = self.connection.cursor()
            
            # Get tables
            cursor.execute(f"""
            SELECT TABLE_NAME
            FROM SYS.TABLES
            WHERE SCHEMA_NAME = '{self.config.schema}'
            ORDER BY TABLE_NAME
            """)
            
            tables = cursor.fetchall()
            
            data_info = f"Schema: {self.config.schema}\n\n"
            data_info += "Table Data:\n\n"
            
            for table_name, in tables:
                data_info += f"Table: {table_name}\n"
                
                # Get table statistics
                try:
                    cursor.execute(f"""
                    SELECT COUNT(*) FROM "{self.config.schema}"."{table_name}"
                    """)
                    count = cursor.fetchone()[0]
                    data_info += f"- Row count: {count}\n"
                except Exception as e:
                    logger.warning(f"Error getting row count for table {table_name}: {str(e)}")
                    data_info += f"- Row count: Error\n"
                
                # Get column statistics
                cursor.execute(f"""
                SELECT COLUMN_NAME, DATA_TYPE_NAME
                FROM SYS.TABLE_COLUMNS
                WHERE SCHEMA_NAME = '{self.config.schema}' AND TABLE_NAME = '{table_name}'
                ORDER BY POSITION
                """)
                
                columns = cursor.fetchall()
                
                # Sample rows
                try:
                    cursor.execute(f"""
                    SELECT * FROM "{self.config.schema}"."{table_name}"
                    LIMIT {max_rows_per_table}
                    """)
                    
                    rows = cursor.fetchall()
                    
                    if rows:
                        data_info += f"- Sample rows ({min(len(rows), max_rows_per_table)}):\n"
                        
                        # Get column names
                        col_names = [col[0] for col in cursor.description]
                        
                        for row in rows:
                            row_str = ", ".join(f"{col}={val}" for col, val in zip(col_names, row))
                            data_info += f"  - {row_str}\n"
                
                except Exception as e:
                    logger.warning(f"Error getting sample rows for table {table_name}: {str(e)}")
                    data_info += f"- Sample rows: Error\n"
                
                data_info += "\n"
            
            cursor.close()
            
            return data_info
        
        except Exception as e:
            logger.error(f"Error extracting data info: {str(e)}")
            raise
    
    def run(self, vector_store=None, existing_questions=None):
        """
        Run the benchmark with one line of code.
        
        Args:
            vector_store: Optional vector store for vector-based benchmarks
            existing_questions: Optional list of existing questions to use
            
        Returns:
            Benchmark results
        """
        try:
            # Extract schema and data information
            logger.info("Extracting schema and data information...")
            schema_info = self._extract_schema_info()
            data_info = self._extract_data_info()
            
            # Generate questions if not provided
            if existing_questions is None:
                logger.info("Generating benchmark questions...")
                questions = self.question_generator.create_benchmark_questions(
                    schema_info=schema_info,
                    data_info=data_info,
                    num_schema_questions=self.config.question_counts.get("schema", 5),
                    num_instance_questions=self.config.question_counts.get("instance", 5),
                    num_relationship_questions=self.config.question_counts.get("relationship", 5),
                    num_aggregation_questions=self.config.question_counts.get("aggregation", 5),
                    num_inference_questions=self.config.question_counts.get("inference", 3),
                    num_temporal_questions=self.config.question_counts.get("temporal", 3)
                )
            else:
                questions = existing_questions
                logger.info(f"Using {len(questions)} existing questions")
            
            # Store questions if storage is available
            if self.storage_manager:
                for question in questions:
                    try:
                        self.storage_manager.create_question(question)
                    except ValueError as e:
                        if "already exists" in str(e):
                            # Skip already existing questions
                            continue
                        raise
            
            # Prepare representation handlers
            logger.info("Initializing representation handlers...")
            handlers = self._prepare_handlers(schema_info, data_info, vector_store)
            
            # Initialize benchmark runner
            logger.info("Initializing benchmark runner...")
            benchmark_runner = BenchmarkRunner(
                storage_manager=self.storage_manager,
                evaluator=self.evaluator,
                handlers=handlers,
                benchmark_name=self.config.name
            )
            
            # Run benchmark
            logger.info("Running benchmark...")
            representation_types = [RepresentationType(rt) for rt in self.config.representation_types if rt in [rt.value for rt in RepresentationType]]
            metrics = benchmark_runner.run_benchmark(
                questions=questions,
                representation_types=representation_types
            )
            
            # Generate recommendations
            logger.info("Generating recommendations...")
            recommendations = benchmark_runner.get_recommendations(metrics)
            
            # Save results if requested
            if self.config.save_results and self.config.output_file:
                import json
                
                results = {
                    "benchmark_id": benchmark_runner.benchmark_id,
                    "benchmark_name": benchmark_runner.benchmark_name,
                    "timestamp": time.time(),
                    "metrics": {rt.value: {
                        "total_count": rt_metrics.total_count,
                        "correct_count": rt_metrics.correct_count,
                        "incorrect_count": rt_metrics.incorrect_count,
                        "not_attempted_count": rt_metrics.not_attempted_count,
                        "ambiguous_count": rt_metrics.ambiguous_count,
                        "accuracy": rt_metrics.accuracy,
                        "f_score": rt_metrics.f_score,
                        "avg_response_time_ms": rt_metrics.avg_response_time_ms,
                        "metrics_by_question_type": rt_metrics.metrics_by_question_type,
                        "metrics_by_difficulty": rt_metrics.metrics_by_difficulty,
                    } for rt, rt_metrics in metrics.items()},
                    "recommendations": recommendations,
                }
                
                with open(self.config.output_file, "w") as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Results saved to {self.config.output_file}")
            
            logger.info("Benchmark completed successfully")
            return {"metrics": metrics, "recommendations": recommendations, "questions": questions}
        
        except Exception as e:
            logger.error(f"Error running benchmark: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def visualize_results(self, metrics=None, filename=None):
        """
        Create beautiful visualizations of benchmark results.
        
        Args:
            metrics: Optional metrics from a previous run
            filename: Optional filename to save visualization
            
        Returns:
            Path to saved visualization file
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from matplotlib.colors import LinearSegmentedColormap
        except ImportError:
            logger.error("Visualization requires matplotlib, numpy, and pandas")
            raise ImportError("Please install matplotlib, numpy, and pandas for visualization")
        
        # Load metrics from file if not provided
        if metrics is None and self.config.output_file:
            import json
            try:
                with open(self.config.output_file, "r") as f:
                    data = json.load(f)
                    metrics = {RepresentationType(k): v for k, v in data["metrics"].items()}
            except Exception as e:
                logger.error(f"Error loading metrics from file: {str(e)}")
                raise
        
        if not metrics:
            raise ValueError("No metrics provided for visualization")
        
        # Create beautiful colors inspired by Apple's design language
        colors = {
            "relational": "#007AFF",  # iOS blue
            "vector": "#FF9500",      # iOS orange
            "ontology": "#4CD964",    # iOS green
            "hybrid": "#AF52DE"       # iOS purple
        }
        
        # Create figure with Apple-inspired aesthetics
        plt.figure(figsize=(12, 8), dpi=300)
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Set a clean, minimalist aesthetic
        plt.rcParams.update({
            'font.family': 'SF Pro Display, -apple-system, BlinkMacSystemFont, Helvetica Neue, Arial, sans-serif',
            'font.size': 12,
            'axes.titlesize': 18,
            'axes.labelsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.titlesize': 20
        })
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=300, constrained_layout=True)
        fig.suptitle(f"Benchmark Results: {self.config.name}", fontweight='bold', fontsize=20)
        
        # Accuracy by representation type
        ax1 = axes[0, 0]
        representation_types = [rt.value for rt in metrics.keys()]
        accuracies = [metrics[rt].accuracy for rt in metrics.keys()]
        
        bars = ax1.bar(
            representation_types, 
            accuracies, 
            color=[colors.get(rt, "#999999") for rt in representation_types],
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
        
        ax1.set_title("Accuracy by Representation Method", fontweight='bold')
        ax1.set_ylim(0, 1.15)
        ax1.set_ylabel("Accuracy")
        ax1.set_xlabel("Representation Method")
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        
        # Performance by question type
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
        
        ax2.set_title("Performance by Question Type", fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([qt.capitalize() for qt in question_types], rotation=45, ha='right')
        ax2.set_ylabel("Accuracy")
        ax2.set_ylim(0, 1)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax2.legend(title="Representation Method")
        
        # Performance by difficulty
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
        
        ax3.set_title("Performance by Difficulty Level", fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels([f"Level {diff}" for diff in difficulty_levels])
        ax3.set_ylabel("Accuracy")
        ax3.set_ylim(0, 1)
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax3.legend(title="Representation Method")
        
        # Spider chart for different metrics
        ax4 = axes[1, 1]
        
        # Define metrics to show
        metrics_to_show = ["accuracy", "f_score", "attempted_accuracy"]
        metric_labels = ["Accuracy", "F-Score", "Attempted Accuracy"]
        
        # Create the spider chart
        angles = np.linspace(0, 2*np.pi, len(metrics_to_show), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        ax4.set_theta_offset(np.pi / 2)
        ax4.set_theta_direction(-1)
        ax4.set_rlabel_position(0)
        
        # Set the labels
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(metric_labels)
        
        # Set y-axis limits
        ax4.set_ylim(0, 1)
        ax4.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax4.set_yticklabels(["20%", "40%", "60%", "80%", "100%"])
        
        # Plot each representation type
        for rt, rt_metrics in metrics.items():
            values = [
                rt_metrics.accuracy,
                rt_metrics.f_score,
                rt_metrics.attempted_accuracy
            ]
            values += values[:1]  # Close the loop
            
            ax4.plot(angles, values, linewidth=2, label=rt.value.capitalize(), color=colors.get(rt.value, "#999999"))
            ax4.fill(angles, values, alpha=0.1, color=colors.get(rt.value, "#999999"))
        
        ax4.set_title("Performance Metrics Comparison", fontweight='bold')
        ax4.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # Add a subtle watermark
        fig.text(0.99, 0.01, "SAP HANA Reasoning Benchmark", ha='right', va='bottom', 
                 color='gray', fontsize=8, alpha=0.5)
        
        # Improve layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save the figure if filename is provided
        if filename is None:
            filename = f"benchmark_visualization_{time.strftime('%Y%m%d_%H%M%S')}.png"
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logger.info(f"Visualization saved to {filename}")
        
        return filename


# Example usage:
if __name__ == "__main__":
    from langchain_openai import ChatOpenAI
    
    # Initialize the language model
    llm = ChatOpenAI(temperature=0)
    
    # Create a simple configuration
    config = BenchmarkConfig(
        name="Example Benchmark",
        schema="SALES",
        question_counts={
            "schema": 3,
            "instance": 3,
            "relationship": 2
        },
        representation_types=["relational", "vector", "hybrid"]
    )
    
    # Create and run the benchmark with one line
    benchmark = SimpleBenchmark(llm=llm, config=config)
    results = benchmark.run()
    
    # Create beautiful visualizations
    benchmark.visualize_results(metrics=results["metrics"])