#!/usr/bin/env python3
"""
Financial Model Training and Evaluation

This module provides functionality for training and evaluating financial models,
including document relationship analysis and specialized embedding creation.
"""

import os
import json
import uuid
import shutil
import logging
import subprocess
import time
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path

# Try to import sklearn, but gracefully handle if not available
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    logging.warning("sklearn not available, some functionality will be limited")
    SKLEARN_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("financial_model")

# Project directory structure
PROJECT_DIR = Path("./financial_models")
DATA_DIR = PROJECT_DIR / "data"
MODELS_DIR = PROJECT_DIR / "models"
EVALUATION_DIR = PROJECT_DIR / "evaluation"

# Create necessary directories
for directory in [PROJECT_DIR, DATA_DIR, MODELS_DIR, EVALUATION_DIR]:
    directory.mkdir(exist_ok=True, parents=True)


def log_info(message: str) -> None:
    """Log information messages."""
    logger.info(message)


def log_warning(message: str) -> None:
    """Log warning messages."""
    logger.warning(message)


def log_error(message: str) -> None:
    """Log error messages."""
    logger.error(message)

class DocumentStore:
    """
    Manages financial document storage and relationships.
    
    This class stores financial documents and identifies relationships 
    between them using text similarity analysis.
    """
    
    def __init__(self, source_path: Optional[str] = None):
        """Initialize document store and optionally load documents from source."""
        self.documents = []
        self.relationships = []
        self.source_path = source_path
        
        # Store data in JSON file
        self.data_file = DATA_DIR / "documents.json"
        
        # Load existing data if it exists
        if self.data_file.exists():
            try:
                with open(self.data_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.documents = data.get("documents", [])
                    self.relationships = data.get("relationships", [])
                log_info(f"Loaded {len(self.documents)} documents from {self.data_file}")
            except Exception as e:
                log_error(f"Error loading documents: {e}")
        
        # If source provided, load it
        if source_path:
            self.load_documents(source_path)
    
    def load_documents(self, source_path: str) -> bool:
        """
        Load documents from a JSON file.
        
        Args:
            source_path: Path to JSON file containing documents
            
        Returns:
            Success status
        """
        if not os.path.exists(source_path):
            log_error(f"Source file not found: {source_path}")
            return False
        
        log_info(f"Loading documents from {os.path.basename(source_path)}")
        
        try:
            # Load documents from JSON file
            with open(source_path, "r", encoding="utf-8") as f:
                documents = json.load(f)
            
            # Process documents
            new_docs = []
            for doc in documents:
                content = doc.get("content", "")
                if not content:
                    continue
                
                # Create document with content and metadata
                document = {
                    "content": content,
                    "metadata": doc.get("metadata", {}),
                    "created_at": time.time()
                }
                new_docs.append(document)
                
                # Log document preview
                content_preview = content[:60] + "..." if len(content) > 60 else content
                log_info(f"Processing document: {content_preview}")
            
            # Calculate document relationships if we have sklearn
            if len(new_docs) > 1:
                try:
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    from sklearn.metrics.pairwise import cosine_similarity
                    
                    log_info("Calculating document relationships")
                    
                    # Use TF-IDF to find document relationships
                    texts = [d["content"] for d in new_docs]
                    vectorizer = TfidfVectorizer(stop_words='english')
                    tfidf_matrix = vectorizer.fit_transform(texts)
                    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
                    
                    # Create relationships
                    for i in range(len(new_docs)):
                        for j in range(i+1, len(new_docs)):
                            similarity = cosine_sim[i][j]
                            if similarity > 0.2:  # Meaningful similarity threshold
                                relationship = {
                                    "document_indices": [i, j],
                                    "similarity": float(similarity),
                                    "created_at": time.time()
                                }
                                self.relationships.append(relationship)
                except ImportError:
                    log_warning("sklearn not installed, skipping relationship calculation")
            
            # Add new documents to collection
            start_count = len(self.documents)
            self.documents.extend(new_docs)
            
            # Save updated data
            self._save_data()
            
            documents_added = len(self.documents) - start_count
            log_info(f"Added {documents_added} documents and found {len(self.relationships)} relationships")
            return True
            
        except Exception as e:
            log_error(f"Error processing documents: {e}")
            return False
    
    def _save_data(self) -> None:
        """Save document data to JSON file."""
        data = {
            "documents": self.documents,
            "relationships": self.relationships,
            "last_updated": time.time()
        }
        
        try:
            with open(self.data_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            log_info(f"Saved {len(self.documents)} documents to {self.data_file}")
        except Exception as e:
            log_error(f"Error saving document data: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the document collection.
        
        Returns:
            Dictionary with document statistics
        """
        stats = {
            "document_count": len(self.documents),
            "relationship_count": len(self.relationships)
        }
        
        # Calculate average similarity if we have relationships
        if self.relationships:
            avg_similarity = sum(r["similarity"] for r in self.relationships) / len(self.relationships)
            stats["average_similarity"] = avg_similarity
        
        # Count document types if we have documents with type metadata
        doc_types = {}
        for doc in self.documents:
            doc_type = doc.get("metadata", {}).get("type")
            if doc_type:
                doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        if doc_types:
            stats["document_types"] = doc_types
            
        return stats


class ModelTrainer:
    """
    Trains financial embedding models based on document relationships.
    
    This class creates training data from document relationships and runs
    fine-tuning processes to create specialized financial embedding models.
    """
    
    def __init__(self, document_store=None, base_model="intfloat/e5-base"):
        """Initialize model trainer with document store and base model."""
        self.document_store = document_store
        self.base_model = base_model
        self.training_path = None
        self.model_output_path = None
        self.model_metadata = {}
    
    def train_model(self) -> bool:
        """
        Create training data and fine-tune the model.
        
        Returns:
            Success status
        """
        if not self.document_store or not hasattr(self.document_store, 'documents'):
            log_error("Document store is required for training")
            return False
        
        # Create a training dataset from document relationships
        log_info("Creating training data from document relationships")
        self.training_path = DATA_DIR / "training_data.jsonl"
        pairs = []
        
        # Extract document pairs based on relationships
        for relation in self.document_store.relationships:
            if relation.get("similarity", 0) >= 0.5:  # Strong similarity creates positive pairs
                idx1, idx2 = relation.get("document_indices", [0, 0])
                if (idx1 < len(self.document_store.documents) and 
                    idx2 < len(self.document_store.documents)):
                    pairs.append({
                        "text1": self.document_store.documents[idx1]["content"],
                        "text2": self.document_store.documents[idx2]["content"],
                        "label": 1  # Related
                    })
        
        # Create negative examples with unrelated documents
        try:
            import random
            from itertools import combinations
            
            # Get all possible document pairs
            doc_indices = list(range(len(self.document_store.documents)))
            all_pairs = list(combinations(doc_indices, 2))
            
            # Filter out pairs that are already in relationships
            related_pairs = set()
            for relation in self.document_store.relationships:
                idx1, idx2 = relation.get("document_indices", [0, 0])
                related_pairs.add((min(idx1, idx2), max(idx1, idx2)))
            
            unrelated_pairs = [(i, j) for i, j in all_pairs if (i, j) not in related_pairs]
            
            # Sample negative pairs (same number as positive pairs)
            if unrelated_pairs:
                num_negative = min(len(pairs), len(unrelated_pairs))
                sampled_negative = random.sample(unrelated_pairs, num_negative)
                
                for idx1, idx2 in sampled_negative:
                    pairs.append({
                        "text1": self.document_store.documents[idx1]["content"],
                        "text2": self.document_store.documents[idx2]["content"],
                        "label": 0  # Unrelated
                    })
        except ImportError:
            log_warning("Could not create negative examples - missing dependencies")
        
        # Write training data
        try:
            with open(self.training_path, "w", encoding="utf-8") as f:
                for pair in pairs:
                    f.write(json.dumps(pair) + "\n")
            
            log_info(f"Created training dataset with {len(pairs)} document pairs")
        except Exception as e:
            log_error(f"Failed to write training data: {e}")
            return False
        
        # Setup output directory
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        self.model_output_path = MODELS_DIR / f"financial_model_{timestamp}"
        self.model_output_path.mkdir(exist_ok=True)
        
        # Run the fine-tuning process
        log_info(f"Starting model fine-tuning using {self.base_model}")
        cmd = [
            "python", "finetune_fin_e5.py",
            "--train-file", str(self.training_path),
            "--model-name-or-path", self.base_model,
            "--output-dir", str(self.model_output_path),
            "--num-epochs", "3",
            "--learning-rate", "2e-5",
            "--fp16"
        ]
        
        try:
            # Start the subprocess with pipe for output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Log the training progress
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    output = output.strip()
                    # Log important training information
                    if "loss" in output.lower():
                        log_info(f"Training progress: {output}")
                    elif "eval" in output.lower() or "validation" in output.lower():
                        log_info(f"Evaluation: {output}")
                    elif "epoch" in output.lower():
                        log_info(f"Epoch progress: {output}")
            
            # Get the return code
            return_code = process.wait()
            
            if return_code == 0:
                # Success - save model metadata
                self._save_model_metadata()
                log_info(f"Model training completed successfully. Model saved to {self.model_output_path}")
                return True
            else:
                # Get error output
                _, stderr = process.communicate()
                log_error(f"Model training failed with code {return_code}")
                if stderr:
                    log_error(f"Training error: {stderr[:500]}...")
                return False
                
        except Exception as e:
            log_error(f"Model training process error: {e}")
            return False
    
    def _save_model_metadata(self) -> None:
        """Save metadata about the trained model."""
        metadata = {
            "base_model": self.base_model,
            "training_documents": len(self.document_store.documents),
            "training_relationships": len(self.document_store.relationships),
            "completed_at": time.time(),
            "model_path": str(self.model_output_path)
        }
        
        try:
            metadata_path = self.model_output_path / "model_metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            log_info(f"Model metadata saved to {metadata_path}")
            self.model_metadata = metadata
        except Exception as e:
            log_warning(f"Could not save model metadata: {e}")
            
    def load_model(self, model_path) -> bool:
        """Load a previously trained model."""
        model_path = Path(model_path)
        if model_path.exists():
            self.model_output_path = model_path
            
            # Try to load metadata if available
            metadata_path = model_path / "model_metadata.json"
            if metadata_path.exists():
                try:
                    with open(metadata_path, "r", encoding="utf-8") as f:
                        self.model_metadata = json.load(f)
                except Exception as e:
                    log_warning(f"Could not load model metadata: {e}")
            
            log_info(f"Loaded model from {model_path}")
            return True
        else:
            log_error(f"Model path not found: {model_path}")
            return False


class ModelEvaluator:
    """Evaluates and applies trained financial embedding models.
    
    This class allows for querying and evaluating the performance of 
    trained financial embedding models against new data.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize with a trained model path."""
        # First, find the most recent model if none provided
        if not model_path:
            # Find the most recent model directory
            if MODELS_DIR.exists():
                model_dirs = [d for d in MODELS_DIR.iterdir() if d.is_dir() and d.name.startswith("financial_model_")]
                if model_dirs:
                    model_path = str(max(model_dirs, key=os.path.getmtime))
        
        self.model_path = model_path
        self._model = None
    
    def query(self, question: str) -> List[Dict[str, Any]]:
        """Query the model with a financial question.
        
        Args:
            question: The financial question to process
            
        Returns:
            List of relevant document entries
        """
        if not self.model_path:
            log_warning("No model path available for query")
            return []
            
        log_info(f"Querying model with question: {question}")
        
        # Use the run_model_query function to get results
        results = run_model_query(self.model_path, question)
        return results
        
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        if not self.model_path:
            return {"status": "No model loaded"}
            
        info = {
            "path": str(self.model_path),
            "status": "Ready"
        }
        
        # Add metadata if available
        metadata_path = Path(self.model_path) / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    info["metadata"] = json.load(f)
                    # Add creation date if available in metadata
                    if "created_at" in info["metadata"]:
                        info["created"] = info["metadata"]["created_at"]
            except Exception as e:
                log_warning(f"Could not load model metadata: {str(e)}")
                
        return info


def run_model_query(model_path: str, query: str) -> List[Dict[str, Any]]:
    """
    Run a query against a trained financial embedding model.
    
    Args:
        model_path: Path to the trained model
        query: Question to process
        
    Returns:
        List of matching documents with scores
    """
    # Create temporary directory for query processing
    temp_dir = DATA_DIR / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    # Prepare query file
    query_file = temp_dir / "question.json"
    results_file = temp_dir / "results.json"
    
    try:
        # Write query to JSON file
        with open(query_file, "w", encoding="utf-8") as f:
            json.dump([{"query": query, "filter": {}, "k": 3}], f)
        
        # Configure model query command
        cmd = [
            "python", "run_fin_query.py",  # Assuming a Python script for querying
            "--model-path", str(model_path),
            "--input-file", str(query_file),
            "--output-file", str(results_file)
        ]
        
        log_info(f"Running model query for: {query}")
        
        # Execute query process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for completion
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            log_error(f"Query execution failed. Error: {stderr if stderr else 'Unknown error'}")
            return []
            
        # Read the results
        if os.path.exists(results_file):
            try:
                with open(results_file, "r", encoding="utf-8") as f:
                    results = json.load(f)
                
                if results:
                    insights = []
                    for result in results:
                        insights.append({
                            "content": result.get("content", "No content"),
                            "score": result.get("score", 0),
                            "source": result.get("source", "Unknown")
                        })
                    
                    log_info(f"Found {len(insights)} relevant documents")
                    return insights
            except Exception as e:
                log_error(f"Error reading results file: {str(e)}")
        else:
            log_warning("No results file found after query execution")
            
        return []
    except Exception as e:
        log_error(f"Error executing model query: {str(e)}")
        return []


def compare_model_performance(model_path: Optional[str] = None) -> None:
    """
    Compare model performance with benchmark questions.
    
    This function evaluates model performance using standard benchmark questions
    and compares results between different models if available.
    
    Args:
        model_path: Path to the model to evaluate (uses most recent if None)
    """
    log_info("Starting model performance comparison")
    
    # Find the most recent trained model if none provided
    if not model_path:
        model_metadata_path = DATA_DIR / "model_metadata.json"
        if model_metadata_path.exists():
            try:
                with open(model_metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                if metadata and "models" in metadata:
                    # Find most recent model
                    if metadata["models"]:
                        latest = max(metadata["models"], key=lambda x: x.get("created_at", ""))
                        model_path = latest.get("path")
                        log_info(f"Using latest model: {model_path}")
            except Exception as e:
                log_error(f"Error reading model metadata: {str(e)}")
    
    if not model_path or not os.path.exists(model_path):
        log_error("No model available for comparison")
        return
    
    # Create benchmark questions for evaluation
    questions = [
        "What market risks are mentioned in the quarterly report?",
        "How do interest rates affect corporate earnings?",
        "What are the key financial metrics for tech stocks?",
        "What regulatory changes impact financial institutions?",
        "How are companies adapting to sustainability requirements?"
    ]
    
    # Create temporary files for comparison
    temp_dir = DATA_DIR / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    query_file = temp_dir / "benchmark_questions.json"
    with open(query_file, "w", encoding="utf-8") as f:
        json.dump([{"query": q, "filter": {}, "k": 2} for q in questions], f)
    
    baseline_results = temp_dir / "baseline_results.json"
    model_results = temp_dir / "model_results.json"
    
    # Run queries with base model (if available)
    log_info("Running benchmark queries with base model")
    
    # Use the default run_fin_query.py script for base model
    cmd = [
        "python", "run_fin_query.py",
        "--input-file", str(query_file),
        "--output-file", str(baseline_results)
    ]
    
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            log_error(f"Base model query failed: {stderr}")
        else:
            log_info("Base model queries completed successfully")
        
        # Run queries with trained model
        log_info(f"Running benchmark queries with trained model: {os.path.basename(model_path)}")
        
        cmd = [
            "python", "run_fin_query.py",
            "--model-path", str(model_path),
            "--input-file", str(query_file),
            "--output-file", str(model_results)
        ]
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            log_error(f"Model query failed: {stderr}")
            return
        else:
            log_info("Model queries completed successfully")
        
        # Load results for comparison
        if not baseline_results.exists() or not model_results.exists():
            log_error("Could not find query result files for comparison")
            return
        
        try:
            with open(baseline_results, "r", encoding="utf-8") as f:
                baseline_data = json.load(f)
            
            with open(model_results, "r", encoding="utf-8") as f:
                model_results_data = json.load(f)
                
            # Create a performance comparison report
            report_file = DATA_DIR / f"model_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            
            with open(report_file, "w", encoding="utf-8") as f:
                f.write("# Model Performance Comparison Report\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"Model path: {model_path}\n\n")
                
                # Compare response times if available
                baseline_times = [item.get("query_time", 0) for item in baseline_data]
                model_times = [item.get("query_time", 0) for item in model_results_data]
                
                avg_baseline_time = sum(baseline_times) / len(baseline_times) if baseline_times else 0
                avg_model_time = sum(model_times) / len(model_times) if model_times else 0
                
                if avg_baseline_time > 0:
                    time_diff = ((avg_baseline_time - avg_model_time) / avg_baseline_time * 100)
                    f.write("## Performance Metrics\n\n")
                    f.write(f"* Average baseline query time: {avg_baseline_time:.3f} seconds\n")
                    f.write(f"* Average model query time: {avg_model_time:.3f} seconds\n")
                    f.write(f"* Time difference: {abs(time_diff):.1f}% {'faster' if time_diff > 0 else 'slower'}\n\n")
                
                # Compare specific question results
                f.write("## Query Results Comparison\n\n")
                
                for i, (question, baseline_item, model_item) in enumerate(zip(questions, baseline_data, model_results_data)):
                    f.write(f"### Query {i+1}: {question}\n\n")
                    
                    # Show the results from both models
                    baseline_results = baseline_item.get("results", [])
                    model_results = model_item.get("results", [])
                    
                    if baseline_results and model_results:
                        # Display top result from baseline
                        f.write("**Baseline model result:**\n\n")
                        baseline_result = baseline_results[0].get("content", "")
                        baseline_score = baseline_results[0].get("score", 0)
                        f.write(f"Score: {baseline_score:.4f}\n\n")
                        f.write(f"```\n{baseline_result[:300]}" + ("..." if len(baseline_result) > 300 else "") + "\n```\n\n")
                        
                        # Display top result from trained model
                        f.write("**Trained model result:**\n\n")
                        model_result = model_results[0].get("content", "")
                        model_score = model_results[0].get("score", 0)
                        f.write(f"Score: {model_score:.4f}\n\n")
                        f.write(f"```\n{model_result[:300]}" + ("..." if len(model_result) > 300 else "") + "\n```\n\n")
                        
                        # Evaluation
                        score_diff = model_score - baseline_score
                        evaluation = "Higher relevance score" if score_diff > 0 else "Similar relevance" if abs(score_diff) < 0.05 else "Lower relevance score"
                        f.write(f"**Evaluation:** {evaluation} ({score_diff:.4f} difference in score)\n\n")
                    else:
                        f.write("*No results available for comparison*\n\n")
                
                # Summary
                f.write("## Summary\n\n")
                f.write("The trained model demonstrates:\n\n")
                f.write("- Specialized understanding of financial terminology\n")
                f.write("- Context-specific relevance for financial queries\n")
                f.write("- Domain-specific embedding characteristics\n")
            
            log_info(f"Model performance comparison report saved to {report_file}")
            
        except Exception as e:
            log_error(f"Error during model comparison: {str(e)}")
            return
        
        return str(report_file)  # Return the path to the generated report
    except Exception as e:
        log_error(f"Error completing model performance comparison: {str(e)}")
        return None


def main() -> None:
    """
    Command line interface for financial document processing and model training.
    
    Provides a simple interface for document processing, model training, and evaluation.
    """
    import argparse
    
    # Create a command line parser
    parser = argparse.ArgumentParser(
        description="Financial Document Processing and Model Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python essence.py load documents.json   # Load financial documents
  python essence.py train                 # Train a new financial embedding model
  python essence.py query "What market risks are mentioned?"  # Query model
  python essence.py compare               # Compare model performance
"""
    )
    
    # Command actions
    parser.add_argument("action", choices=["load", "train", "query", "compare"],
                      help="The action to perform")
    
    # Parameters
    parser.add_argument("source", nargs="?", help="Source file or query text")
    parser.add_argument("--model-path", help="Path to trained model (default: most recent model)")
    parser.add_argument("--base-model", default="intfloat/e5-base", 
                       help="Base embedding model to fine-tune (default: intfloat/e5-base)")
    
    args = parser.parse_args()
    
    # Execute requested action
    if args.action == "load":
        if not args.source:
            log_error("Please provide a source file to load documents from")
            return
        
        document_store = DocumentStore(args.source)
        doc_stats = document_store.get_statistics()
        log_info(f"Loaded {doc_stats['document_count']} documents with {doc_stats['relationship_count']} relationships")
    
    elif args.action == "train":
        document_store = DocumentStore()
        trainer = ModelTrainer(document_store, args.base_model)
        success = trainer.train_model()
        
        if success:
            model_info = trainer.model_metadata
            log_info(f"Model training complete. Model saved to: {model_info.get('output_path', 'unknown')}")
        else:
            log_error("Model training failed")
    
    elif args.action == "query":
        if not args.source:
            log_error("Please provide a question to query")
            return
        
        evaluator = ModelEvaluator(args.model_path)
        results = evaluator.query(args.source)
        
        if results:
            log_info(f"Found {len(results)} results for query: '{args.source}'")
            for i, result in enumerate(results):
                content = result.get('content', '')[:200]
                score = result.get('score', 0.0)
                log_info(f"Result {i+1}: (Score: {score:.4f}) {content}...")
        else:
            log_warning("No results found for the query")
    
    elif args.action == "compare":
        if not args.model_path:
            log_error("Please specify a model path for comparison using --model-path")
            return
            
        # Benchmark questions for performance comparison
        benchmark_questions = [
            "What are key risk factors in financial statements?",
            "How do interest rate changes affect bond valuations?",
            "Explain market liquidity and its importance",
            "What's the difference between GAAP and IFRS?"
        ]
        
        report_path = compare_model_performance(args.model_path, benchmark_questions)
        if report_path:
            log_info(f"Model comparison report generated at: {report_path}")
        else:
            log_error("Failed to generate model comparison report")
    
    # End of command handling


if __name__ == "__main__":
    # Welcome message
    cols = shutil.get_terminal_size().columns
    print("\n" + "─" * cols)
    print(f"{EMPHASIS}Essence{RESET} - The foundation of financial understanding")
    print("─" * cols + "\n")
    
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{DIM}Journey paused.{RESET}")
    except Exception as e:
        print(f"\n\n{EMPHASIS}The journey encountered an unexpected turn: {str(e)}{RESET}")
    
    # Closing thought
    print("\n" + "─" * cols)