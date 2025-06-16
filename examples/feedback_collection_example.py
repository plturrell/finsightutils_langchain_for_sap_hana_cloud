#!/usr/bin/env python
"""
Example of using the feedback collection framework for the SAP HANA Vector Knowledge System.

This example demonstrates how to collect user feedback on various aspects of the system,
store it in different backends, and use it to improve system performance.
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Any, Optional, Tuple

from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

# Add parent directory to path to import local modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_hana.reasoning.feedback import (
    UserFeedbackCollector, 
    InMemoryFeedbackStorage,
    HanaFeedbackStorage,
    FeedbackVisualization,
)
from langchain_hana.reasoning.data_lineage import LineageTracker
from langchain_hana.reasoning.transparent_pipeline import (
    TransparentEmbeddingPipeline,
    TextPreprocessingStage,
    ModelEmbeddingStage,
    VectorPostprocessingStage,
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleEmbeddings(Embeddings):
    """
    Simple embeddings implementation for demonstration purposes.
    """
    
    def __init__(self, vector_size: int = 128):
        """Initialize simple embeddings."""
        self.vector_size = vector_size
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate simple embeddings for documents."""
        import numpy as np
        return [list(np.random.rand(self.vector_size)) for _ in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """Generate simple embedding for a query."""
        import numpy as np
        return list(np.random.rand(self.vector_size))


def setup_pipeline() -> TransparentEmbeddingPipeline:
    """Set up a transparent embedding pipeline for demonstration."""
    
    # Create embedding model
    embeddings = SimpleEmbeddings(vector_size=128)
    
    # Create pipeline stages
    preprocessor = TextPreprocessingStage(
        name="basic_preprocessing",
        description="Basic text preprocessing for demonstration",
        lowercase=True,
        remove_punctuation=True,
    )
    
    postprocessor = VectorPostprocessingStage(
        name="basic_postprocessing",
        description="Basic vector postprocessing for demonstration",
        normalize_vectors=True,
    )
    
    # Create pipeline
    pipeline = TransparentEmbeddingPipeline(
        embedding_model=embeddings,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        track_fingerprints=True,
    )
    
    return pipeline


def run_example(args: argparse.Namespace) -> None:
    """
    Run the feedback collection example.
    
    Args:
        args: Command line arguments
    """
    # Set up pipeline
    pipeline = setup_pipeline()
    
    # Set up storage backend
    if args.use_hana and args.connection_path:
        try:
            import json
            from hdbcli import dbapi
            
            # Load connection details
            with open(args.connection_path) as f:
                conn_details = json.load(f)
            
            # Connect to HANA
            connection = dbapi.connect(
                address=conn_details.get("host"),
                port=conn_details.get("port"),
                user=conn_details.get("user"),
                password=conn_details.get("password"),
            )
            
            # Create HANA storage backend
            storage = HanaFeedbackStorage(
                connection=connection,
                schema_name=args.schema,
                table_name="FEEDBACK_ITEMS",
            )
            
            # Create lineage tracker
            lineage_tracker = LineageTracker(
                connection=connection,
                schema_name=args.schema,
                table_name="DATA_LINEAGE",
            )
            
            logger.info("Using SAP HANA for feedback storage")
        except Exception as e:
            logger.warning(f"Could not connect to HANA: {e}. Using in-memory storage instead.")
            storage = InMemoryFeedbackStorage()
            lineage_tracker = None
    else:
        # Use in-memory storage
        storage = InMemoryFeedbackStorage()
        lineage_tracker = None
        logger.info("Using in-memory feedback storage")
    
    # Create feedback collector
    collector = UserFeedbackCollector(
        storage_backend=storage,
        lineage_tracker=lineage_tracker,
        pipeline=pipeline,
    )
    
    # Generate some example embeddings
    texts = [
        "This is a sample document about artificial intelligence.",
        "SAP HANA is a high-performance in-memory database.",
        "Vector databases are optimized for similarity search.",
        "Machine learning models can be used for various tasks.",
        "Knowledge graphs represent information as connected entities.",
    ]
    
    logger.info(f"Generating embeddings for {len(texts)} documents")
    vectors, inspection_data = pipeline.embed_documents(texts, return_inspection_data=True)
    
    # Generate some IDs for the embeddings
    embedding_ids = [f"emb_{i}" for i in range(len(texts))]
    
    # Add some embedding feedback
    logger.info("Adding embedding feedback examples")
    
    # Good feedback
    collector.add_embedding_feedback(
        rating=5,
        text=texts[0],
        embedding_id=embedding_ids[0],
        suggestions=[],
        user_id="user_1",
        metadata={"context": "testing"},
    )
    
    # Average feedback with suggestions
    collector.add_embedding_feedback(
        rating=3,
        text=texts[1],
        embedding_id=embedding_ids[1],
        suggestions=["Improve technical term handling"],
        user_id="user_2",
        metadata={"context": "testing"},
    )
    
    # Poor feedback with suggestions
    collector.add_embedding_feedback(
        rating=1,
        text=texts[2],
        embedding_id=embedding_ids[2],
        suggestions=[
            "Improve handling of domain-specific terminology",
            "Better context awareness",
        ],
        user_id="user_3",
        metadata={"context": "testing"},
    )
    
    # Add retrieval feedback
    logger.info("Adding retrieval feedback examples")
    
    # Simulate a search
    query = "What is vector search?"
    query_id = "query_1"
    
    # Generate dummy results
    result_ids = [f"result_{i}" for i in range(5)]
    
    collector.add_retrieval_feedback(
        query=query,
        relevant_results=[result_ids[0], result_ids[2]],
        irrelevant_results=[result_ids[1], result_ids[4]],
        query_id=query_id,
        missing_results=[f"missing_{i}" for i in range(2)],
        user_id="user_1",
        metadata={"search_type": "semantic"},
    )
    
    # Add reasoning feedback
    logger.info("Adding reasoning feedback examples")
    
    reasoning_id = "reasoning_1"
    collector.add_reasoning_feedback(
        rating=4,
        reasoning_id=reasoning_id,
        corrections=[
            {"step": 2, "correction": "Consider alternative hypotheses"},
        ],
        explanation="Good reasoning overall but missed some alternatives",
        user_id="user_2",
        metadata={"reasoning_type": "causal"},
    )
    
    # Add transformation feedback
    logger.info("Adding transformation feedback examples")
    
    transformation_id = "transform_1"
    collector.add_transformation_feedback(
        rating=2,
        stage="preprocessing",
        transformation_id=transformation_id,
        suggestions=["Improve handling of technical terms"],
        example_inputs=["SAP HANA Cloud", "S/4HANA"],
        example_expected_outputs=["sap hana cloud", "s/4hana"],
        user_id="user_3",
        metadata={"domain": "technical"},
    )
    
    # Get feedback statistics
    stats = collector.get_feedback_stats()
    logger.info(f"Total feedback items: {stats.get('total_feedback')}")
    logger.info(f"Feedback by type: {stats.get('by_type')}")
    
    # Get recommendations
    recommendations = collector.get_improvement_recommendations()
    logger.info(f"Improvement recommendations: {json.dumps(recommendations, indent=2)}")
    
    # Generate visualization summary
    summary = FeedbackVisualization.generate_feedback_summary(collector)
    logger.info(f"Visualization summary: {json.dumps(summary, indent=2)}")
    
    # Save summary to file if specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved visualization summary to {args.output}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Feedback collection example")
    
    parser.add_argument(
        "--use-hana",
        action="store_true",
        help="Use SAP HANA for feedback storage",
    )
    
    parser.add_argument(
        "--connection-path",
        type=str,
        help="Path to SAP HANA connection details JSON file",
    )
    
    parser.add_argument(
        "--schema",
        type=str,
        default="FEEDBACK_SCHEMA",
        help="Database schema for feedback storage",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save visualization summary JSON",
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_example(args)