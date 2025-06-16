#!/usr/bin/env python
"""
Performance benchmarking script for comparing embedding generation 
between CPU and GPU implementations.

This script measures:
1. Initialization time for different embedding implementations
2. Throughput (tokens/sec) for individual query embedding
3. Batch processing performance with different batch sizes
4. Memory usage during embedding operations
"""

import os
import sys
import time
import gc
import logging
import argparse
import numpy as np
import psutil
from pathlib import Path
from contextlib import contextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("benchmark")

# Add project root to path for imports
project_root = str(Path(__file__).parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Test data
SAMPLE_TEXTS = [
    "This is a short sentence to embed.",
    "Here's another example of text that needs to be converted to embeddings.",
    "The SAP HANA Cloud integration with LangChain provides powerful capabilities.",
    "Embedding models convert text to numerical vectors for semantic operations.",
    "Large language models leverage these embeddings for various NLP tasks.",
    "The quality and performance of embeddings directly impact downstream applications.",
    "GPU acceleration can significantly improve embedding generation throughput.",
    "These vectors enable semantic search and retrieval within document collections.",
    "The dimensionality of embeddings affects both performance and accuracy.",
    "Optimizing embedding workflows is critical for production applications."
]

LONG_TEXT = """
SAP HANA Cloud integration with LangChain provides a powerful foundation for building AI-enabled 
applications that combine the robust database capabilities of SAP HANA with the flexible AI 
orchestration abilities of LangChain. This integration enables developers to create applications 
that can store, retrieve, and process data efficiently while leveraging large language models 
and other AI components. By using embeddings to convert text into numerical vectors, these 
applications can perform semantic search, similarity comparisons, and other operations that 
understand the meaning behind text, not just keywords. The optimization of these embedding 
workflows, particularly through GPU acceleration when available, ensures that applications 
can scale efficiently and provide responsive user experiences even with large datasets or 
complex queries. Additionally, the ability to gracefully fall back to CPU-compatible embedding 
models ensures system reliability across different deployment environments.
"""

# Number of repetitions for reliable measurements
NUM_REPEATS = 5
BATCH_SIZES = [1, 4, 8, 16, 32, 64]

@contextmanager
def timer(operation_name):
    """Context manager for timing operations."""
    start_time = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        logger.info(f"{operation_name}: {elapsed:.4f} seconds")
        return elapsed

def get_memory_usage():
    """Get current memory usage of the process."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def load_embedding_model(model_type, model_name="all-MiniLM-L6-v2"):
    """
    Load the specified embedding model.
    
    Args:
        model_type: Type of model to load ('cpu', 'gpu', 'tensorrt')
        model_name: Name of the model to use
    
    Returns:
        The loaded embedding model
    """
    logger.info(f"Loading {model_type} embedding model: {model_name}")
    initial_memory = get_memory_usage()
    
    with timer(f"Loading {model_type} model"):
        if model_type == 'cpu':
            try:
                from langchain_hana.embeddings import HanaInternalEmbeddings
                embeddings = HanaInternalEmbeddings(
                    model_name=model_name
                )
            except ImportError:
                logger.warning("HanaInternalEmbeddings not available, using mock")
                class MockEmbeddings:
                    def __init__(self, model_name=None):
                        self.model_name = model_name
                        self.dimension = 384
                        time.sleep(1)  # Simulate loading time
                    
                    def embed_query(self, text):
                        time.sleep(0.01)  # Simulate embedding time
                        return np.random.rand(384).tolist()
                    
                    def embed_documents(self, texts):
                        time.sleep(0.01 * len(texts))  # Simulate batch embedding time
                        return [np.random.rand(384).tolist() for _ in texts]
                
                embeddings = MockEmbeddings(model_name=model_name)
        
        elif model_type == 'gpu':
            try:
                # Try to import GPU accelerated embeddings
                from api.gpu import gpu_utils
                if not gpu_utils.is_gpu_available():
                    logger.warning("No GPU available, falling back to CPU")
                    return load_embedding_model('cpu', model_name)
                
                try:
                    from gpu_embeddings import GPUAcceleratedEmbeddings
                    embeddings = GPUAcceleratedEmbeddings(
                        model_name=model_name,
                        device="cuda",
                        batch_size=32
                    )
                except ImportError:
                    logger.warning("GPUAcceleratedEmbeddings not available, falling back to CPU")
                    return load_embedding_model('cpu', model_name)
            except ImportError:
                logger.warning("GPU utilities not available, falling back to CPU")
                return load_embedding_model('cpu', model_name)
        
        elif model_type == 'tensorrt':
            try:
                # Try to import TensorRT optimized embeddings
                from api.gpu import gpu_utils
                if not gpu_utils.is_gpu_available():
                    logger.warning("No GPU available, falling back to CPU")
                    return load_embedding_model('cpu', model_name)
                
                try:
                    from gpu_embeddings import TensorRTEmbeddings
                    embeddings = TensorRTEmbeddings(
                        model_name=model_name,
                        device="cuda",
                        precision="fp16"
                    )
                except ImportError:
                    logger.warning("TensorRTEmbeddings not available, falling back to GPU")
                    return load_embedding_model('gpu', model_name)
            except ImportError:
                logger.warning("GPU utilities not available, falling back to CPU")
                return load_embedding_model('cpu', model_name)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    # Measure memory footprint
    final_memory = get_memory_usage()
    memory_increase = final_memory - initial_memory
    logger.info(f"{model_type} model memory footprint: {memory_increase:.2f} MB")
    
    return embeddings

def benchmark_single_query(embeddings, text=SAMPLE_TEXTS[0], num_repeats=NUM_REPEATS):
    """Benchmark embedding a single query."""
    logger.info(f"Benchmarking single query embedding")
    times = []
    
    # Clear cache before benchmarking
    gc.collect()
    
    for i in range(num_repeats):
        start_time = time.time()
        result = embeddings.embed_query(text)
        elapsed = time.time() - start_time
        times.append(elapsed)
        
    avg_time = sum(times) / len(times)
    logger.info(f"Single query average time: {avg_time:.4f} seconds")
    
    # Basic validation of result
    if hasattr(result, '__len__'):
        logger.info(f"Embedding dimension: {len(result)}")
    
    return avg_time

def benchmark_batch_processing(embeddings, batch_size=32, num_repeats=NUM_REPEATS):
    """Benchmark batch processing with the specified batch size."""
    # Take a subset of sample texts to match batch size
    batch_texts = (SAMPLE_TEXTS * (batch_size // len(SAMPLE_TEXTS) + 1))[:batch_size]
    
    logger.info(f"Benchmarking batch processing with size {batch_size}")
    times = []
    
    # Clear cache before benchmarking
    gc.collect()
    
    for i in range(num_repeats):
        start_time = time.time()
        if hasattr(embeddings, 'embed_documents'):
            results = embeddings.embed_documents(batch_texts)
        else:
            # Fall back to sequential embedding if batch not supported
            results = [embeddings.embed_query(text) for text in batch_texts]
        elapsed = time.time() - start_time
        times.append(elapsed)
    
    avg_time = sum(times) / len(times)
    logger.info(f"Batch size {batch_size} average time: {avg_time:.4f} seconds")
    logger.info(f"Throughput: {batch_size / avg_time:.2f} texts/second")
    
    return avg_time, batch_size / avg_time

def benchmark_long_text(embeddings, num_repeats=NUM_REPEATS):
    """Benchmark embedding a long text passage."""
    logger.info(f"Benchmarking long text embedding")
    times = []
    
    # Clear cache before benchmarking
    gc.collect()
    
    for i in range(num_repeats):
        start_time = time.time()
        result = embeddings.embed_query(LONG_TEXT)
        elapsed = time.time() - start_time
        times.append(elapsed)
    
    avg_time = sum(times) / len(times)
    logger.info(f"Long text average time: {avg_time:.4f} seconds")
    
    # Calculate approximate tokens (rough estimate)
    approx_tokens = len(LONG_TEXT.split())
    logger.info(f"Approximate token count: {approx_tokens}")
    logger.info(f"Tokens per second: {approx_tokens / avg_time:.2f}")
    
    return avg_time, approx_tokens / avg_time

def run_benchmarks(model_types, model_name="all-MiniLM-L6-v2"):
    """Run benchmarks for all specified model types."""
    results = {}
    
    for model_type in model_types:
        logger.info(f"\n{'='*50}\nBenchmarking {model_type} embeddings\n{'='*50}")
        
        try:
            # Load the embedding model
            embeddings = load_embedding_model(model_type, model_name)
            
            # Run benchmarks
            results[model_type] = {
                "single_query_time": benchmark_single_query(embeddings),
                "long_text": benchmark_long_text(embeddings),
                "batch_processing": {}
            }
            
            # Batch processing benchmarks
            for batch_size in BATCH_SIZES:
                results[model_type]["batch_processing"][batch_size] = benchmark_batch_processing(
                    embeddings, batch_size
                )
            
            logger.info(f"Completed benchmarks for {model_type} embeddings")
            
        except Exception as e:
            logger.error(f"Error benchmarking {model_type} embeddings: {str(e)}")
    
    return results

def print_comparison(results):
    """Print a comparison of benchmark results."""
    logger.info("\n\n" + "="*80)
    logger.info("EMBEDDING PERFORMANCE COMPARISON")
    logger.info("="*80)
    
    # Single query comparison
    logger.info("\nSINGLE QUERY PERFORMANCE")
    logger.info("-" * 40)
    for model_type, data in results.items():
        if "single_query_time" in data:
            logger.info(f"{model_type:10}: {data['single_query_time']:.6f} seconds")
    
    # Long text comparison
    logger.info("\nLONG TEXT PERFORMANCE (tokens/sec)")
    logger.info("-" * 40)
    for model_type, data in results.items():
        if "long_text" in data:
            _, throughput = data["long_text"]
            logger.info(f"{model_type:10}: {throughput:.2f} tokens/sec")
    
    # Batch processing comparison
    logger.info("\nBATCH PROCESSING PERFORMANCE (texts/sec)")
    logger.info("-" * 60)
    logger.info(f"{'Batch Size':<10} " + " ".join(f"{model:>12}" for model in results.keys()))
    logger.info("-" * 60)
    
    for batch_size in BATCH_SIZES:
        line = f"{batch_size:<10} "
        for model_type in results.keys():
            if ("batch_processing" in results[model_type] and 
                batch_size in results[model_type]["batch_processing"]):
                _, throughput = results[model_type]["batch_processing"][batch_size]
                line += f"{throughput:>12.2f}"
            else:
                line += f"{'N/A':>12}"
        logger.info(line)

def main():
    parser = argparse.ArgumentParser(description="Benchmark embedding models performance")
    parser.add_argument(
        "--models", 
        nargs="+", 
        choices=["cpu", "gpu", "tensorrt"], 
        default=["cpu", "gpu", "tensorrt"],
        help="Model types to benchmark"
    )
    parser.add_argument(
        "--model-name", 
        default="all-MiniLM-L6-v2",
        help="Name of the embedding model to use"
    )
    args = parser.parse_args()
    
    logger.info("Starting embedding performance benchmarks")
    logger.info(f"Models to benchmark: {args.models}")
    logger.info(f"Model name: {args.model_name}")
    
    # Force CPU mode for CPU benchmarks
    os.environ["FORCE_CPU"] = "1"
    cpu_results = run_benchmarks(["cpu"], args.model_name)
    
    # Remove CPU force for GPU benchmarks
    os.environ.pop("FORCE_CPU", None)
    
    # Only benchmark GPU models if we're not forcing CPU-only mode
    gpu_model_types = [m for m in args.models if m != "cpu"]
    if gpu_model_types:
        gpu_results = run_benchmarks(gpu_model_types, args.model_name)
        # Combine results
        results = {**cpu_results, **gpu_results}
    else:
        results = cpu_results
    
    # Print comparison
    print_comparison(results)
    
    logger.info("\nBenchmark completed")

if __name__ == "__main__":
    main()
