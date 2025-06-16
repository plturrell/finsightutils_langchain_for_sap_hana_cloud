"""
GPU Data Layer Acceleration Example

This example demonstrates how to use the GPU Data Layer Accelerator for SAP HANA Cloud
to perform high-performance vector operations directly on the GPU with minimal data transfer.

Key features demonstrated:
1. GPU-accelerated similarity search with and without filtering
2. Hybrid search approach (database filtering + GPU scoring)
3. Full GPU search approach (all operations on GPU)
4. Maximum Marginal Relevance (MMR) search with GPU acceleration
5. GPU memory management and caching strategies
6. Performance comparison between CPU, hybrid, and GPU approaches

Requirements:
- SAP HANA Cloud database connection
- NVIDIA GPU
- PyTorch with CUDA support
- FAISS-GPU (optional, for index building)
"""

import argparse
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
import json
import numpy as np
from pprint import pformat

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Try to import required modules
try:
    from hdbcli import dbapi
    HAS_HDBCLI = True
except ImportError:
    logger.warning("hdbcli module not found. Using mock implementation for demonstration.")
    HAS_HDBCLI = False

try:
    import torch
    HAS_TORCH = torch.cuda.is_available()
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch with CUDA support not available. Some features will be disabled.")

# Try to import the data layer accelerator
try:
    from langchain_hana.gpu.data_layer_accelerator import (
        get_vector_engine,
        HanaGPUVectorEngine,
        MemoryManager
    )
    from langchain_hana.utils import DistanceStrategy
    HAS_ACCELERATOR = True
except ImportError:
    logger.warning("Data layer accelerator not found. Using mock implementation for demonstration.")
    HAS_ACCELERATOR = False
    
    # Define mock classes for demonstration
    class DistanceStrategy:
        COSINE = "cosine"
        EUCLIDEAN_DISTANCE = "euclidean"
    
    class MemoryManager:
        def __init__(self, gpu_id=0, cache_size_gb=4.0, precision="float32"):
            self.gpu_id = gpu_id
            self.cache_size_gb = cache_size_gb
            self.precision = precision
            self.device = torch.device(f"cuda:{gpu_id}") if HAS_TORCH and torch.cuda.is_available() else None
            logger.info(f"Initialized mock MemoryManager (GPU: {self.device})")
        
        def get_vector_tensor(self, vectors, cache_key=None):
            return torch.tensor(vectors, device=self.device) if self.device else np.array(vectors)
        
        def compute_similarity(self, query_vector, document_vectors, distance_strategy, batch_size=1024):
            time.sleep(0.01)  # Simulate computation time
            return np.random.random(len(document_vectors)).tolist()
        
        def release(self):
            logger.info("Released GPU memory")
    
    class HanaGPUVectorEngine:
        def __init__(self, connection, table_name, content_column, metadata_column, vector_column, **kwargs):
            self.connection = connection
            self.table_name = table_name
            self.content_column = content_column
            self.metadata_column = metadata_column
            self.vector_column = vector_column
            self.distance_strategy = kwargs.get("distance_strategy", DistanceStrategy.COSINE)
            self.gpu_ids = kwargs.get("gpu_ids", [0])
            self.batch_size = kwargs.get("batch_size", 1024)
            self.enable_prefetch = kwargs.get("enable_prefetch", True)
            
            # Initialize memory managers
            self.gpu_available = HAS_TORCH and torch.cuda.is_available()
            self.memory_managers = [MemoryManager(gpu_id=gpu_id) for gpu_id in self.gpu_ids] if self.gpu_available else []
            self.primary_gpu = self.memory_managers[0] if self.memory_managers else None
            
            logger.info(f"Initialized mock HanaGPUVectorEngine (GPU available: {self.gpu_available})")
        
        def similarity_search(self, query_vector, k=4, filter=None, fetch_all_vectors=False):
            time.sleep(0.1)  # Simulate search time
            return [(f"Content {i}", json.dumps({"id": i}), 0.9 - (i * 0.1)) for i in range(k)]
        
        def mmr_search(self, query_vector, k=4, fetch_k=20, lambda_mult=0.5, filter=None):
            from langchain_core.documents import Document
            time.sleep(0.15)  # Simulate search time
            return [Document(page_content=f"Content {i}", metadata={"id": i}) for i in range(k)]
        
        def build_index(self, index_type="hnsw", m=16, ef_construction=200, ef_search=100):
            time.sleep(0.5)  # Simulate index building
            logger.info(f"Built mock {index_type} index")
        
        def release(self):
            for manager in self.memory_managers:
                manager.release()
            logger.info("Released all GPU resources")
    
    def get_vector_engine(connection, table_name, content_column, metadata_column, vector_column, **kwargs):
        return HanaGPUVectorEngine(
            connection=connection,
            table_name=table_name,
            content_column=content_column,
            metadata_column=metadata_column,
            vector_column=vector_column,
            **kwargs
        )


# Mock connection function for demonstration
def get_hana_connection(host, port, user, password):
    """Get a connection to SAP HANA Cloud."""
    if not HAS_HDBCLI:
        logger.warning("Using mock connection since hdbcli is not available")
        return MockConnection()
    
    try:
        conn = dbapi.connect(
            address=host,
            port=port,
            user=user,
            password=password
        )
        return conn
    except Exception as e:
        logger.error(f"Error connecting to SAP HANA Cloud: {e}")
        return None


class MockConnection:
    """Mock connection class for demonstration."""
    
    def cursor(self):
        return MockCursor()
    
    def close(self):
        pass


class MockCursor:
    """Mock cursor class for demonstration."""
    
    def execute(self, sql, params=None):
        logger.debug(f"Executing SQL: {sql}")
        return True
    
    def fetchall(self):
        # Generate some mock vector data
        if "SELECT" in getattr(self, "last_sql", ""):
            # Generate 100 mock vectors of dimension 384
            return [(self._create_mock_vector_binary(384),) for _ in range(100)]
        return []
    
    def fetchone(self):
        return ("Mock content", json.dumps({"id": 1}), self._create_mock_vector_binary(384))
    
    def close(self):
        pass
    
    def _create_mock_vector_binary(self, dim=384):
        """Create a mock vector in binary format."""
        import struct
        vector = np.random.random(dim).astype(np.float32)
        # First 4 bytes are dimension as uint32, then dim * 4 bytes of float32 values
        return struct.pack(f"<I{dim}f", dim, *vector)


def generate_sample_query_vector(dim=384):
    """Generate a sample query vector."""
    return np.random.random(dim).astype(np.float32).tolist()


def basic_similarity_search_example(vector_engine, query_vector, k=5):
    """Demonstrate basic similarity search."""
    logger.info(f"Running basic similarity search (k={k})")
    
    # Time the search
    start_time = time.time()
    results = vector_engine.similarity_search(
        query_vector=query_vector,
        k=k
    )
    elapsed_time = time.time() - start_time
    
    # Display results
    logger.info(f"Search completed in {elapsed_time:.4f} seconds")
    logger.info(f"Found {len(results)} results:")
    
    for i, (content, metadata_json, score) in enumerate(results):
        metadata = json.loads(metadata_json) if isinstance(metadata_json, str) else metadata_json
        logger.info(f"  {i+1}. Score: {score:.4f}, Content: {content[:50]}..., Metadata: {metadata}")
    
    return results, elapsed_time


def filtered_search_example(vector_engine, query_vector, filter_dict, k=5):
    """Demonstrate similarity search with metadata filtering."""
    logger.info(f"Running filtered similarity search (k={k}, filter={filter_dict})")
    
    # Time the search
    start_time = time.time()
    results = vector_engine.similarity_search(
        query_vector=query_vector,
        k=k,
        filter=filter_dict
    )
    elapsed_time = time.time() - start_time
    
    # Display results
    logger.info(f"Search completed in {elapsed_time:.4f} seconds")
    logger.info(f"Found {len(results)} results:")
    
    for i, (content, metadata_json, score) in enumerate(results):
        metadata = json.loads(metadata_json) if isinstance(metadata_json, str) else metadata_json
        logger.info(f"  {i+1}. Score: {score:.4f}, Content: {content[:50]}..., Metadata: {metadata}")
    
    return results, elapsed_time


def hybrid_vs_full_gpu_example(vector_engine, query_vector, k=5):
    """Compare hybrid search (database filtering + GPU scoring) vs. full GPU search."""
    logger.info("Comparing hybrid search vs. full GPU search")
    
    # Run hybrid search (default)
    logger.info("Running hybrid search (database filtering + GPU scoring)...")
    hybrid_start = time.time()
    hybrid_results = vector_engine.similarity_search(
        query_vector=query_vector,
        k=k,
        fetch_all_vectors=False  # This is the default
    )
    hybrid_time = time.time() - hybrid_start
    
    # Run full GPU search
    logger.info("Running full GPU search (all vectors loaded to GPU)...")
    full_gpu_start = time.time()
    full_gpu_results = vector_engine.similarity_search(
        query_vector=query_vector,
        k=k,
        fetch_all_vectors=True  # This enables full GPU search
    )
    full_gpu_time = time.time() - full_gpu_start
    
    # Display results
    logger.info(f"Hybrid search: {hybrid_time:.4f} seconds, {len(hybrid_results)} results")
    logger.info(f"Full GPU search: {full_gpu_time:.4f} seconds, {len(full_gpu_results)} results")
    
    if hybrid_time > 0 and full_gpu_time > 0:
        speedup = hybrid_time / full_gpu_time
        logger.info(f"Full GPU search is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than hybrid search")
    
    # Compare result quality
    if hybrid_results and full_gpu_results:
        common_results = 0
        for hr in hybrid_results:
            for gr in full_gpu_results:
                if hr[0] == gr[0]:  # Compare content
                    common_results += 1
                    break
        
        similarity = common_results / min(len(hybrid_results), len(full_gpu_results))
        logger.info(f"Result similarity between approaches: {similarity:.2f} ({common_results} common results)")
    
    return {
        "hybrid": {
            "time": hybrid_time,
            "results": hybrid_results
        },
        "full_gpu": {
            "time": full_gpu_time,
            "results": full_gpu_results
        }
    }


def mmr_search_example(vector_engine, query_vector, k=5, fetch_k=10, lambda_mult=0.5):
    """Demonstrate Maximum Marginal Relevance search with GPU acceleration."""
    logger.info(f"Running GPU-accelerated MMR search (k={k}, fetch_k={fetch_k}, lambda_mult={lambda_mult})")
    
    # Time the search
    start_time = time.time()
    results = vector_engine.mmr_search(
        query_vector=query_vector,
        k=k,
        fetch_k=fetch_k,
        lambda_mult=lambda_mult
    )
    elapsed_time = time.time() - start_time
    
    # Display results
    logger.info(f"MMR search completed in {elapsed_time:.4f} seconds")
    logger.info(f"Found {len(results)} results:")
    
    for i, doc in enumerate(results):
        logger.info(f"  {i+1}. Content: {doc.page_content[:50]}..., Metadata: {doc.metadata}")
    
    return results, elapsed_time


def index_building_example(vector_engine):
    """Demonstrate GPU-accelerated index building."""
    logger.info("Building HNSW index with GPU acceleration")
    
    # Configure index parameters
    index_type = "hnsw"  # Options: "hnsw", "flat"
    m = 16               # Number of connections per layer
    ef_construction = 200  # Size of the dynamic list for constructing the graph
    ef_search = 100      # Size of the dynamic list for searching the graph
    
    # Time the index building
    start_time = time.time()
    vector_engine.build_index(
        index_type=index_type,
        m=m,
        ef_construction=ef_construction,
        ef_search=ef_search
    )
    elapsed_time = time.time() - start_time
    
    logger.info(f"Index building completed in {elapsed_time:.4f} seconds")
    logger.info(f"Index type: {index_type}, m={m}, ef_construction={ef_construction}, ef_search={ef_search}")
    
    return elapsed_time


def memory_management_example(vector_engine):
    """Demonstrate GPU memory management features."""
    if not hasattr(vector_engine, "primary_gpu") or not vector_engine.primary_gpu:
        logger.warning("GPU memory management requires GPU support. Skipping example.")
        return
    
    logger.info("Demonstrating GPU memory management")
    
    # Generate multiple query vectors
    query_vectors = [generate_sample_query_vector() for _ in range(10)]
    
    # Run similarity searches to populate cache
    for i, query_vector in enumerate(query_vectors):
        logger.info(f"Running search {i+1}/10 to populate cache...")
        results = vector_engine.similarity_search(
            query_vector=query_vector,
            k=5
        )
        logger.info(f"  Search returned {len(results)} results")
    
    # Get memory usage from primary GPU
    if hasattr(vector_engine.primary_gpu, "allocated_memory"):
        logger.info(f"GPU memory usage:")
        logger.info(f"  Allocated memory: {vector_engine.primary_gpu.allocated_memory / (1024*1024):.2f} MB")
        logger.info(f"  Maximum memory: {vector_engine.primary_gpu.max_memory / (1024*1024):.2f} MB")
        logger.info(f"  Cached items: {len(vector_engine.primary_gpu.cached_data)}")
    
    # Force cache cleanup if implemented
    if hasattr(vector_engine.primary_gpu, "_clean_cache"):
        logger.info("Forcing cache cleanup...")
        vector_engine.primary_gpu._clean_cache()
        
        # Check memory usage after cleanup
        if hasattr(vector_engine.primary_gpu, "allocated_memory"):
            logger.info(f"GPU memory usage after cleanup:")
            logger.info(f"  Allocated memory: {vector_engine.primary_gpu.allocated_memory / (1024*1024):.2f} MB")
            logger.info(f"  Cached items: {len(vector_engine.primary_gpu.cached_data)}")
    
    # Release all GPU resources
    logger.info("Releasing all GPU resources...")
    vector_engine.release()
    logger.info("GPU resources released.")


def performance_comparison_example(connection, table_name, content_column, metadata_column, vector_column):
    """Compare performance with different configurations."""
    logger.info("Running performance comparison with different configurations")
    
    # Generate test data
    query_vector = generate_sample_query_vector()
    k = 10
    
    # Define configurations to test
    configs = [
        {
            "name": "CPU Only",
            "use_gpu": False,
            "batch_size": 1024,
            "distance_strategy": DistanceStrategy.COSINE
        },
        {
            "name": "GPU (Hybrid Search)",
            "use_gpu": True,
            "batch_size": 1024,
            "distance_strategy": DistanceStrategy.COSINE,
            "fetch_all_vectors": False
        },
        {
            "name": "GPU (Full Search)",
            "use_gpu": True,
            "batch_size": 1024,
            "distance_strategy": DistanceStrategy.COSINE,
            "fetch_all_vectors": True
        },
        {
            "name": "GPU (Large Batch)",
            "use_gpu": True,
            "batch_size": 4096,
            "distance_strategy": DistanceStrategy.COSINE,
            "fetch_all_vectors": True
        },
        {
            "name": "GPU (Euclidean)",
            "use_gpu": True,
            "batch_size": 1024,
            "distance_strategy": DistanceStrategy.EUCLIDEAN_DISTANCE,
            "fetch_all_vectors": True
        }
    ]
    
    results = []
    
    for config in configs:
        logger.info(f"Testing configuration: {config['name']}")
        
        # Create vector engine with this configuration
        vector_engine = get_vector_engine(
            connection=connection,
            table_name=table_name,
            content_column=content_column,
            metadata_column=metadata_column,
            vector_column=vector_column,
            distance_strategy=config["distance_strategy"],
            batch_size=config["batch_size"],
            gpu_ids=None if config["use_gpu"] else []
        )
        
        # Run similarity search
        start_time = time.time()
        search_results = vector_engine.similarity_search(
            query_vector=query_vector,
            k=k,
            fetch_all_vectors=config.get("fetch_all_vectors", False)
        )
        elapsed_time = time.time() - start_time
        
        # Store results
        results.append({
            "config": config,
            "time": elapsed_time,
            "num_results": len(search_results)
        })
        
        logger.info(f"  Time: {elapsed_time:.4f} seconds, Results: {len(search_results)}")
        
        # Clean up
        vector_engine.release()
    
    # Display summary
    logger.info("\nPerformance Comparison Summary:")
    logger.info("-" * 80)
    logger.info(f"{'Configuration':<20} {'Time (s)':<12} {'Results':<10} {'Relative Speed':<15}")
    logger.info("-" * 80)
    
    # Find baseline (CPU time)
    baseline_time = next((r["time"] for r in results if r["config"]["name"] == "CPU Only"), 1.0)
    
    for result in results:
        config_name = result["config"]["name"]
        elapsed_time = result["time"]
        num_results = result["num_results"]
        relative_speed = baseline_time / elapsed_time if elapsed_time > 0 else 0
        
        logger.info(f"{config_name:<20} {elapsed_time:<12.4f} {num_results:<10} {relative_speed:<15.2f}x")
    
    logger.info("-" * 80)
    
    return results


def main():
    """Main function to run the example."""
    parser = argparse.ArgumentParser(description="GPU Data Layer Acceleration Example")
    parser.add_argument("--host", type=str, default="localhost", help="SAP HANA host")
    parser.add_argument("--port", type=int, default=39015, help="SAP HANA port")
    parser.add_argument("--user", type=str, default="SYSTEM", help="SAP HANA user")
    parser.add_argument("--password", type=str, default="", help="SAP HANA password")
    parser.add_argument("--table", type=str, default="VECTOR_TABLE", help="Vector table name")
    parser.add_argument("--content-column", type=str, default="CONTENT", help="Content column name")
    parser.add_argument("--metadata-column", type=str, default="METADATA", help="Metadata column name")
    parser.add_argument("--vector-column", type=str, default="VECTOR", help="Vector column name")
    parser.add_argument("--example", type=str, default="all", 
                        choices=["basic", "filtered", "hybrid", "mmr", "index", "memory", "performance", "all"],
                        help="Example to run (default: all)")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--cache-size", type=float, default=4.0, help="GPU cache size in GB")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size for processing")
    args = parser.parse_args()
    
    # Check GPU availability
    gpu_info = "Not available"
    if HAS_TORCH and torch.cuda.is_available():
        device = torch.cuda.current_device()
        gpu_info = f"{torch.cuda.get_device_name(device)} (Compute Capability: {torch.cuda.get_device_capability(device)})"
    
    logger.info("=" * 80)
    logger.info("GPU Data Layer Acceleration Example")
    logger.info("=" * 80)
    logger.info(f"GPU: {gpu_info}")
    logger.info(f"Database: {args.host}:{args.port}")
    logger.info(f"Table: {args.table}")
    logger.info(f"Columns: {args.content_column}, {args.metadata_column}, {args.vector_column}")
    logger.info(f"GPU Cache: {args.cache_size}GB, Batch Size: {args.batch_size}")
    logger.info("=" * 80)
    
    try:
        # Connect to SAP HANA
        logger.info(f"Connecting to SAP HANA Cloud at {args.host}:{args.port}...")
        connection = get_hana_connection(args.host, args.port, args.user, args.password)
        
        if not connection:
            logger.error("Failed to connect to SAP HANA Cloud. Exiting.")
            return
        
        # Create vector engine
        logger.info("Initializing GPU vector engine...")
        vector_engine = get_vector_engine(
            connection=connection,
            table_name=args.table,
            content_column=args.content_column,
            metadata_column=args.metadata_column,
            vector_column=args.vector_column,
            distance_strategy=DistanceStrategy.COSINE,
            gpu_ids=[args.gpu_id],
            cache_size_gb=args.cache_size,
            batch_size=args.batch_size
        )
        
        # Generate sample query vector
        query_vector = generate_sample_query_vector()
        
        # Run the selected example
        try:
            if args.example in ["basic", "all"]:
                logger.info("\n\n>> Running Basic Similarity Search Example <<\n")
                basic_similarity_search_example(vector_engine, query_vector)
            
            if args.example in ["filtered", "all"]:
                logger.info("\n\n>> Running Filtered Search Example <<\n")
                filter_dict = {"category": "technology", "importance": "high"}
                filtered_search_example(vector_engine, query_vector, filter_dict)
            
            if args.example in ["hybrid", "all"]:
                logger.info("\n\n>> Running Hybrid vs. Full GPU Search Example <<\n")
                hybrid_vs_full_gpu_example(vector_engine, query_vector)
            
            if args.example in ["mmr", "all"]:
                logger.info("\n\n>> Running MMR Search Example <<\n")
                mmr_search_example(vector_engine, query_vector)
            
            if args.example in ["index", "all"]:
                logger.info("\n\n>> Running Index Building Example <<\n")
                index_building_example(vector_engine)
            
            if args.example in ["memory", "all"]:
                logger.info("\n\n>> Running Memory Management Example <<\n")
                memory_management_example(vector_engine)
            
            if args.example in ["performance", "all"]:
                logger.info("\n\n>> Running Performance Comparison Example <<\n")
                performance_comparison_example(
                    connection, args.table, 
                    args.content_column, args.metadata_column, args.vector_column
                )
        
        finally:
            # Clean up resources
            logger.info("\nCleaning up resources...")
            vector_engine.release()
            connection.close()
        
        logger.info("\nExample completed successfully!")
    
    except Exception as e:
        logger.error(f"Error running example: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()