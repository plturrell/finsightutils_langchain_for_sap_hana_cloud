"""
Benchmarking utilities for SAP HANA Cloud LangChain integration.

This module provides utilities for benchmarking the performance of various components,
including embedding generation, similarity search, and vector operations.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import numpy as np
from hdbcli import dbapi
from langchain_core.embeddings import Embeddings

from langchain_hana.config.deployment import DeploymentConfig, get_config
from langchain_hana.gpu.utils import detect_gpu_capabilities
from langchain_hana.monitoring.health import get_system_health
from langchain_hana.vectorstores import HanaDB

logger = logging.getLogger(__name__)


class BenchmarkType(str, Enum):
    """Types of benchmarks that can be performed."""
    EMBEDDING_GENERATION = "embedding_generation"
    SIMILARITY_SEARCH = "similarity_search"
    MMR_SEARCH = "mmr_search"
    DATABASE_OPERATIONS = "database_operations"
    VECTOR_OPERATIONS = "vector_operations"
    END_TO_END = "end_to_end"


@dataclass
class BenchmarkResult:
    """Result of a benchmark operation."""
    name: str
    type: BenchmarkType
    start_time: str
    end_time: str
    duration_ms: float
    operations_per_second: float
    samples: int
    success: bool
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class BenchmarkSuite:
    """Suite of benchmark results."""
    name: str
    start_time: str
    end_time: str
    results: List[BenchmarkResult]
    system_info: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "results": [result.to_dict() for result in self.results],
            "system_info": self.system_info,
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    def save(self, file_path: str) -> None:
        """Save benchmark results to a file."""
        with open(file_path, "w") as f:
            f.write(self.to_json())


def _get_system_info() -> Dict[str, Any]:
    """Get system information for benchmarking."""
    # Get health information
    system_health = get_system_health()
    health_info = system_health.to_dict()
    
    # Get GPU information
    gpu_info = detect_gpu_capabilities()
    
    # Get configuration
    config = get_config()
    
    # Combine information
    system_info = {
        "health": health_info,
        "gpu": gpu_info,
        "config": {
            "environment": config.environment.value,
            "backend_platform": config.backend_platform.value,
            "frontend_platform": config.frontend_platform.value,
            "gpu_enabled": config.gpu_config.enabled,
            "use_internal_embedding": config.embedding_config.use_internal_embedding,
            "embedding_model": config.embedding_config.model_name,
        }
    }
    
    return system_info


def run_timed_operation(
    operation: Callable[[], Any],
    name: str,
    type: BenchmarkType,
    samples: int = 1,
    details: Optional[Dict[str, Any]] = None,
) -> BenchmarkResult:
    """
    Run a timed operation and return benchmark results.
    
    Args:
        operation: The operation to benchmark
        name: Name of the benchmark
        type: Type of benchmark
        samples: Number of samples processed by the operation
        details: Additional details to include in the result
        
    Returns:
        BenchmarkResult: Benchmark results
    """
    start_time = datetime.utcnow().isoformat()
    start_timestamp = time.time()
    
    try:
        # Run the operation
        result = operation()
        
        # Calculate duration
        end_timestamp = time.time()
        duration_seconds = end_timestamp - start_timestamp
        duration_ms = duration_seconds * 1000
        operations_per_second = samples / duration_seconds if duration_seconds > 0 else 0
        
        # Create result
        benchmark_result = BenchmarkResult(
            name=name,
            type=type,
            start_time=start_time,
            end_time=datetime.utcnow().isoformat(),
            duration_ms=duration_ms,
            operations_per_second=operations_per_second,
            samples=samples,
            success=True,
            details=details or {},
        )
        
        # Add operation result to details if it's simple enough
        if isinstance(result, (dict, list, str, int, float, bool)) or result is None:
            benchmark_result.details["result"] = result
        else:
            benchmark_result.details["result_type"] = str(type(result))
        
        return benchmark_result
    
    except Exception as e:
        logger.error(f"Benchmark '{name}' failed: {str(e)}")
        
        # Calculate duration
        end_timestamp = time.time()
        duration_seconds = end_timestamp - start_timestamp
        duration_ms = duration_seconds * 1000
        
        # Create error result
        return BenchmarkResult(
            name=name,
            type=type,
            start_time=start_time,
            end_time=datetime.utcnow().isoformat(),
            duration_ms=duration_ms,
            operations_per_second=0,
            samples=samples,
            success=False,
            error=str(e),
            details=details or {},
        )


def benchmark_embedding_generation(
    embedding_model: Embeddings,
    texts: List[str],
    batch_size: Optional[int] = None,
) -> BenchmarkResult:
    """
    Benchmark embedding generation.
    
    Args:
        embedding_model: Embedding model to use
        texts: Texts to generate embeddings for
        batch_size: Batch size to use (if None, all texts are processed at once)
        
    Returns:
        BenchmarkResult: Benchmark results
    """
    # Determine details
    embedding_type = embedding_model.__class__.__name__
    
    # Create details
    details = {
        "embedding_type": embedding_type,
        "texts_count": len(texts),
        "avg_text_length": sum(len(text) for text in texts) / len(texts),
        "batch_size": batch_size or len(texts),
    }
    
    # Define operation based on batch size
    if batch_size is None or batch_size >= len(texts):
        # Process all texts at once
        operation = lambda: embedding_model.embed_documents(texts)
    else:
        # Process in batches
        def batched_operation():
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                embeddings = embedding_model.embed_documents(batch)
                all_embeddings.extend(embeddings)
            return all_embeddings
        
        operation = batched_operation
    
    # Run benchmark
    return run_timed_operation(
        operation=operation,
        name=f"Embedding Generation ({embedding_type})",
        type=BenchmarkType.EMBEDDING_GENERATION,
        samples=len(texts),
        details=details,
    )


def benchmark_similarity_search(
    vectorstore: HanaDB,
    queries: List[str],
    k: int = 4,
    filter: Optional[Dict[str, Any]] = None,
) -> BenchmarkResult:
    """
    Benchmark similarity search.
    
    Args:
        vectorstore: Vector store to search
        queries: Queries to search for
        k: Number of results to return for each query
        filter: Filter to apply to the search
        
    Returns:
        BenchmarkResult: Benchmark results
    """
    # Determine details
    use_internal_embeddings = getattr(vectorstore, "use_internal_embeddings", False)
    embedding_type = vectorstore.embedding.__class__.__name__
    
    # Create details
    details = {
        "use_internal_embeddings": use_internal_embeddings,
        "embedding_type": embedding_type,
        "queries_count": len(queries),
        "avg_query_length": sum(len(query) for query in queries) / len(queries),
        "k": k,
        "filter": filter,
    }
    
    # Define operation
    def search_operation():
        results = []
        for query in queries:
            result = vectorstore.similarity_search(query, k=k, filter=filter)
            results.append(len(result))
        return {"avg_results_count": sum(results) / len(results) if results else 0}
    
    # Run benchmark
    return run_timed_operation(
        operation=search_operation,
        name="Similarity Search",
        type=BenchmarkType.SIMILARITY_SEARCH,
        samples=len(queries),
        details=details,
    )


def benchmark_mmr_search(
    vectorstore: HanaDB,
    queries: List[str],
    k: int = 4,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
    filter: Optional[Dict[str, Any]] = None,
) -> BenchmarkResult:
    """
    Benchmark maximal marginal relevance search.
    
    Args:
        vectorstore: Vector store to search
        queries: Queries to search for
        k: Number of results to return for each query
        fetch_k: Number of results to fetch before reranking
        lambda_mult: Lambda multiplier for diversity vs. similarity
        filter: Filter to apply to the search
        
    Returns:
        BenchmarkResult: Benchmark results
    """
    # Determine details
    use_internal_embeddings = getattr(vectorstore, "use_internal_embeddings", False)
    embedding_type = vectorstore.embedding.__class__.__name__
    
    # Create details
    details = {
        "use_internal_embeddings": use_internal_embeddings,
        "embedding_type": embedding_type,
        "queries_count": len(queries),
        "avg_query_length": sum(len(query) for query in queries) / len(queries),
        "k": k,
        "fetch_k": fetch_k,
        "lambda_mult": lambda_mult,
        "filter": filter,
    }
    
    # Define operation
    def mmr_operation():
        results = []
        for query in queries:
            result = vectorstore.max_marginal_relevance_search(
                query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, filter=filter
            )
            results.append(len(result))
        return {"avg_results_count": sum(results) / len(results) if results else 0}
    
    # Run benchmark
    return run_timed_operation(
        operation=mmr_operation,
        name="MMR Search",
        type=BenchmarkType.MMR_SEARCH,
        samples=len(queries),
        details=details,
    )


def benchmark_database_operations(
    connection: dbapi.Connection,
    table_name: str,
    num_samples: int = 10,
    batch_size: int = 10,
) -> BenchmarkResult:
    """
    Benchmark database operations.
    
    Args:
        connection: Database connection
        table_name: Table to benchmark operations on
        num_samples: Number of operations to perform
        batch_size: Batch size for insertion operations
        
    Returns:
        BenchmarkResult: Benchmark results
    """
    # Create details
    details = {
        "table_name": table_name,
        "num_samples": num_samples,
        "batch_size": batch_size,
        "operations": [],
    }
    
    # Define operation
    def db_operations():
        # Create a temporary test table
        temp_table = f"{table_name}_benchmark_{int(time.time())}"
        cursor = connection.cursor()
        
        try:
            # Create table
            create_start = time.time()
            cursor.execute(f'CREATE TABLE "{temp_table}" (id INTEGER, value NCLOB)')
            create_time = time.time() - create_start
            details["operations"].append({
                "name": "create_table",
                "duration_ms": create_time * 1000,
            })
            
            # Insert data
            insert_start = time.time()
            for i in range(0, num_samples, batch_size):
                batch_end = min(i + batch_size, num_samples)
                values = [(j, f"Value {j}") for j in range(i, batch_end)]
                cursor.executemany(
                    f'INSERT INTO "{temp_table}" (id, value) VALUES (?, ?)',
                    values
                )
            insert_time = time.time() - insert_start
            details["operations"].append({
                "name": "insert_data",
                "duration_ms": insert_time * 1000,
                "records": num_samples,
            })
            
            # Query data
            query_start = time.time()
            cursor.execute(f'SELECT * FROM "{temp_table}"')
            rows = cursor.fetchall()
            query_time = time.time() - query_start
            details["operations"].append({
                "name": "query_data",
                "duration_ms": query_time * 1000,
                "records": len(rows),
            })
            
            # Drop table
            drop_start = time.time()
            cursor.execute(f'DROP TABLE "{temp_table}"')
            drop_time = time.time() - drop_start
            details["operations"].append({
                "name": "drop_table",
                "duration_ms": drop_time * 1000,
            })
            
            return {
                "create_time_ms": create_time * 1000,
                "insert_time_ms": insert_time * 1000,
                "query_time_ms": query_time * 1000,
                "drop_time_ms": drop_time * 1000,
                "total_time_ms": (create_time + insert_time + query_time + drop_time) * 1000,
            }
        
        finally:
            # Ensure temporary table is dropped
            try:
                cursor.execute(f'DROP TABLE "{temp_table}"')
            except:
                pass
            cursor.close()
    
    # Run benchmark
    return run_timed_operation(
        operation=db_operations,
        name="Database Operations",
        type=BenchmarkType.DATABASE_OPERATIONS,
        samples=num_samples * 2 + 2,  # Create, inserts, query, drop
        details=details,
    )


def benchmark_vector_operations(
    dimension: int = 768,
    num_vectors: int = 1000,
    use_gpu: bool = False,
) -> BenchmarkResult:
    """
    Benchmark vector operations.
    
    Args:
        dimension: Dimension of vectors
        num_vectors: Number of vectors to generate
        use_gpu: Whether to use GPU for operations
        
    Returns:
        BenchmarkResult: Benchmark results
    """
    # Create details
    details = {
        "dimension": dimension,
        "num_vectors": num_vectors,
        "use_gpu": use_gpu,
        "operations": [],
    }
    
    # Define operation
    def vector_operations():
        # Check GPU availability if requested
        if use_gpu:
            try:
                import torch
                has_gpu = torch.cuda.is_available()
                if not has_gpu:
                    details["use_gpu"] = False
                    details["gpu_warning"] = "GPU requested but not available, falling back to CPU"
            except ImportError:
                details["use_gpu"] = False
                details["gpu_warning"] = "torch not installed, falling back to CPU"
        
        # Generate random vectors
        gen_start = time.time()
        if use_gpu and details["use_gpu"] is not False:
            import torch
            vectors = torch.randn((num_vectors, dimension), device="cuda")
            query = torch.randn((dimension,), device="cuda")
        else:
            vectors = np.random.randn(num_vectors, dimension).astype(np.float32)
            query = np.random.randn(dimension).astype(np.float32)
        gen_time = time.time() - gen_start
        details["operations"].append({
            "name": "generate_vectors",
            "duration_ms": gen_time * 1000,
        })
        
        # Calculate dot products
        dot_start = time.time()
        if use_gpu and details["use_gpu"] is not False:
            import torch
            dot_products = torch.matmul(vectors, query)
        else:
            dot_products = np.dot(vectors, query)
        dot_time = time.time() - dot_start
        details["operations"].append({
            "name": "dot_products",
            "duration_ms": dot_time * 1000,
        })
        
        # Calculate norms
        norm_start = time.time()
        if use_gpu and details["use_gpu"] is not False:
            import torch
            norms = torch.norm(vectors, dim=1)
        else:
            norms = np.linalg.norm(vectors, axis=1)
        norm_time = time.time() - norm_start
        details["operations"].append({
            "name": "calculate_norms",
            "duration_ms": norm_time * 1000,
        })
        
        # Calculate cosine similarities
        cos_start = time.time()
        if use_gpu and details["use_gpu"] is not False:
            import torch
            query_norm = torch.norm(query)
            cosine_similarities = dot_products / (norms * query_norm)
        else:
            query_norm = np.linalg.norm(query)
            cosine_similarities = dot_products / (norms * query_norm)
        cos_time = time.time() - cos_start
        details["operations"].append({
            "name": "cosine_similarities",
            "duration_ms": cos_time * 1000,
        })
        
        # Find top-k
        k = min(10, num_vectors)
        topk_start = time.time()
        if use_gpu and details["use_gpu"] is not False:
            import torch
            _, indices = torch.topk(cosine_similarities, k)
            indices = indices.cpu().numpy()
        else:
            indices = np.argsort(-cosine_similarities)[:k]
        topk_time = time.time() - topk_start
        details["operations"].append({
            "name": "find_topk",
            "duration_ms": topk_time * 1000,
        })
        
        return {
            "generate_time_ms": gen_time * 1000,
            "dot_product_time_ms": dot_time * 1000,
            "norm_time_ms": norm_time * 1000,
            "cosine_time_ms": cos_time * 1000,
            "topk_time_ms": topk_time * 1000,
            "total_time_ms": (gen_time + dot_time + norm_time + cos_time + topk_time) * 1000,
        }
    
    # Run benchmark
    return run_timed_operation(
        operation=vector_operations,
        name="Vector Operations",
        type=BenchmarkType.VECTOR_OPERATIONS,
        samples=num_vectors,
        details=details,
    )


def run_benchmark_suite(
    connection: dbapi.Connection,
    vectorstore: HanaDB,
    name: str = "LangChain HANA Benchmark",
    embedding_sample_count: int = 100,
    query_sample_count: int = 10,
) -> BenchmarkSuite:
    """
    Run a comprehensive benchmark suite.
    
    Args:
        connection: Database connection
        vectorstore: Vector store to benchmark
        name: Name of the benchmark suite
        embedding_sample_count: Number of embedding samples to generate
        query_sample_count: Number of query samples to use
        
    Returns:
        BenchmarkSuite: Benchmark results
    """
    # Record start time
    start_time = datetime.utcnow().isoformat()
    
    # Generate sample texts
    sample_texts = [
        f"Sample document {i} for benchmarking with some additional text to increase length and variability."
        for i in range(embedding_sample_count)
    ]
    
    sample_queries = [
        f"Query {i} for testing similarity search performance"
        for i in range(query_sample_count)
    ]
    
    # Create benchmark results list
    results = []
    
    # Run embedding generation benchmark
    embedding_result = benchmark_embedding_generation(
        embedding_model=vectorstore.embedding,
        texts=sample_texts,
    )
    results.append(embedding_result)
    
    # Run batched embedding generation benchmark
    batched_embedding_result = benchmark_embedding_generation(
        embedding_model=vectorstore.embedding,
        texts=sample_texts,
        batch_size=10,
    )
    results.append(batched_embedding_result)
    
    # Run similarity search benchmark
    similarity_result = benchmark_similarity_search(
        vectorstore=vectorstore,
        queries=sample_queries,
    )
    results.append(similarity_result)
    
    # Run MMR search benchmark
    mmr_result = benchmark_mmr_search(
        vectorstore=vectorstore,
        queries=sample_queries,
    )
    results.append(mmr_result)
    
    # Run database operations benchmark
    db_result = benchmark_database_operations(
        connection=connection,
        table_name=vectorstore.table_name,
    )
    results.append(db_result)
    
    # Run vector operations benchmark
    vector_result = benchmark_vector_operations(
        use_gpu=getattr(vectorstore, "use_gpu", False),
    )
    results.append(vector_result)
    
    # Record end time
    end_time = datetime.utcnow().isoformat()
    
    # Get system information
    system_info = _get_system_info()
    
    # Create benchmark suite
    return BenchmarkSuite(
        name=name,
        start_time=start_time,
        end_time=end_time,
        results=results,
        system_info=system_info,
    )