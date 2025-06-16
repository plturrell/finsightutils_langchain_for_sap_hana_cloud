"""
Integration tests for Arrow Flight integration.

This module contains integration tests for the Arrow Flight integration with SAP HANA,
testing the complete workflow from client to server to database and back.
"""

import os
import pytest
import numpy as np
import time
from typing import Dict, List, Any, Optional

# Set environment variables for testing
os.environ["FLIGHT_AUTO_START"] = "true"
os.environ["FLIGHT_HOST"] = "localhost"
os.environ["FLIGHT_PORT"] = "8815"

try:
    import pyarrow as pa
    import pyarrow.flight as flight
    HAS_ARROW_FLIGHT = True
except ImportError:
    HAS_ARROW_FLIGHT = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from hdbcli import dbapi
    HAS_HDBCLI = True
except ImportError:
    HAS_HDBCLI = False

# Skip tests if dependencies are not available
pytestmark = [
    pytest.mark.skipif(not HAS_ARROW_FLIGHT, reason="Arrow Flight not available"),
    pytest.mark.skipif(not HAS_HDBCLI, reason="SAP HANA client not available"),
]

# Connection parameters (can be overridden with environment variables)
HANA_HOST = os.environ.get("HANA_HOST", "localhost")
HANA_PORT = int(os.environ.get("HANA_PORT", "39017"))
HANA_USER = os.environ.get("HANA_USER", "SYSTEM")
HANA_PASSWORD = os.environ.get("HANA_PASSWORD", "manager")
FLIGHT_HOST = os.environ.get("FLIGHT_HOST", "localhost")
FLIGHT_PORT = int(os.environ.get("FLIGHT_PORT", "8815"))


@pytest.fixture(scope="module")
def db_connection():
    """Create a database connection for testing."""
    try:
        connection = dbapi.connect(
            address=HANA_HOST,
            port=HANA_PORT,
            user=HANA_USER,
            password=HANA_PASSWORD
        )
        yield connection
    finally:
        connection.close()


@pytest.fixture(scope="module")
def flight_client():
    """Create an Arrow Flight client for testing."""
    from langchain_hana.gpu.arrow_flight_client import ArrowFlightClient
    
    client = ArrowFlightClient(
        host=FLIGHT_HOST,
        port=FLIGHT_PORT,
        use_tls=False,
        username=HANA_USER,
        password=HANA_PASSWORD
    )
    
    yield client
    
    client.close()


@pytest.fixture(scope="module")
def test_vectors():
    """Create test vectors for testing."""
    # Create 100 test vectors
    dim = 4
    num_vectors = 100
    
    # Generate random vectors
    vectors = np.random.randn(num_vectors, dim).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)  # Normalize
    
    # Generate texts and metadata
    texts = [f"Test document {i}" for i in range(num_vectors)]
    metadatas = [{"index": i, "source": "test"} for i in range(num_vectors)]
    
    return vectors.tolist(), texts, metadatas


@pytest.fixture(scope="module")
def test_table(db_connection):
    """Create a test table for testing."""
    table_name = "ARROW_FLIGHT_TEST"
    cursor = db_connection.cursor()
    
    # Drop table if it exists
    try:
        cursor.execute(f"DROP TABLE {table_name}")
        db_connection.commit()
    except:
        pass
    
    # Create table
    cursor.execute(f"""
    CREATE TABLE {table_name} (
        ID VARCHAR(100) PRIMARY KEY,
        DOCUMENT NVARCHAR(5000),
        METADATA NCLOB,
        VECTOR REAL_VECTOR
    )
    """)
    db_connection.commit()
    
    yield table_name
    
    # Clean up
    try:
        cursor.execute(f"DROP TABLE {table_name}")
        db_connection.commit()
    except:
        pass


def test_arrow_flight_client_operations(flight_client, test_vectors, test_table):
    """Test Arrow Flight client operations."""
    vectors, texts, metadatas = test_vectors
    
    # Upload vectors
    ids = flight_client.upload_vectors(
        table_name=test_table,
        vectors=vectors[:10],  # Use first 10 vectors
        texts=texts[:10],
        metadata=metadatas[:10],
        batch_size=5
    )
    
    assert len(ids) == 10
    
    # Perform similarity search
    results = flight_client.similarity_search(
        table_name=test_table,
        query_vector=vectors[0],  # Use first vector as query
        k=5,
        include_metadata=True,
        include_vectors=True,
        distance_strategy="cosine"
    )
    
    assert len(results) == 5
    assert "id" in results[0]
    assert "score" in results[0]
    assert "metadata" in results[0]
    assert "vector" in results[0]


def test_vectorstore_integration(test_vectors, test_table):
    """Test Arrow Flight vectorstore integration with LangChain."""
    from langchain_hana.gpu import HanaArrowFlightVectorStore
    from langchain.embeddings import HuggingFaceEmbeddings
    
    vectors, texts, metadatas = test_vectors
    
    # Initialize embedding model (with mock embedding function)
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    
    # Override embed_documents and embed_query to use test vectors
    def mock_embed_documents(texts):
        # Map texts to vectors using index
        indices = [int(text.split()[-1]) for text in texts]
        return [vectors[i] for i in indices]
    
    def mock_embed_query(text):
        # For testing, use first vector as query embedding
        return vectors[0]
    
    embeddings.embed_documents = mock_embed_documents
    embeddings.embed_query = mock_embed_query
    
    # Initialize vectorstore
    vectorstore = HanaArrowFlightVectorStore(
        embedding=embeddings,
        host=FLIGHT_HOST,
        port=FLIGHT_PORT,
        table_name=test_table,
        username=HANA_USER,
        password=HANA_PASSWORD,
        pre_delete_collection=True  # Start with a fresh table
    )
    
    # Add documents
    vectorstore.add_texts(
        texts=texts[10:20],  # Use next 10 vectors
        metadatas=metadatas[10:20]
    )
    
    # Perform similarity search
    results = vectorstore.similarity_search(
        "test query",  # Will use vectors[0] as embedding
        k=5
    )
    
    assert len(results) == 5
    assert hasattr(results[0], "page_content")
    assert hasattr(results[0], "metadata")
    
    # Perform similarity search with score
    results_with_score = vectorstore.similarity_search_with_score(
        "test query",
        k=5
    )
    
    assert len(results_with_score) == 5
    assert hasattr(results_with_score[0][0], "page_content")
    assert isinstance(results_with_score[0][1], float)
    
    # Perform MMR search
    mmr_results = vectorstore.max_marginal_relevance_search(
        "test query",
        k=5,
        fetch_k=10,
        lambda_mult=0.5
    )
    
    assert len(mmr_results) == 5
    assert hasattr(mmr_results[0], "page_content")


@pytest.mark.skipif(not HAS_TORCH or not torch.cuda.is_available(),
                    reason="PyTorch with CUDA not available")
def test_gpu_memory_manager():
    """Test GPU-aware Arrow memory manager."""
    from langchain_hana.gpu.arrow_gpu_memory_manager import ArrowGpuMemoryManager
    
    # Initialize memory manager
    memory_manager = ArrowGpuMemoryManager(
        device_id=0,
        max_memory_fraction=0.5,
        batch_size=100
    )
    
    # Test optimal batch size calculation
    batch_size = memory_manager.get_optimal_batch_size()
    assert batch_size > 0
    
    # Test batch similarity search
    dim = 4
    num_query_vectors = 10
    num_stored_vectors = 100
    
    # Generate random vectors
    query_vectors = np.random.randn(num_query_vectors, dim).astype(np.float32)
    query_vectors = query_vectors / np.linalg.norm(query_vectors, axis=1, keepdims=True)
    
    stored_vectors = np.random.randn(num_stored_vectors, dim).astype(np.float32)
    stored_vectors = stored_vectors / np.linalg.norm(stored_vectors, axis=1, keepdims=True)
    
    # Convert to Arrow arrays
    query_array = memory_manager.vectors_to_fixed_size_list_array(query_vectors)
    stored_array = memory_manager.vectors_to_fixed_size_list_array(stored_vectors)
    
    # Perform batch similarity search
    distances, indices = memory_manager.batch_similarity_search(
        query_vectors=query_array,
        stored_vectors=stored_array,
        k=5,
        metric="cosine"
    )
    
    assert isinstance(distances, torch.Tensor)
    assert isinstance(indices, torch.Tensor)
    assert distances.shape == (num_query_vectors, 5)
    assert indices.shape == (num_query_vectors, 5)
    
    # Test MMR rerank
    query_vector = query_vectors[0]
    
    mmr_indices = memory_manager.mmr_rerank(
        query_vector=query_vector,
        vectors=stored_vectors,
        indices=list(range(20)),  # Consider first 20 vectors
        k=5,
        lambda_mult=0.5
    )
    
    assert len(mmr_indices) == 5
    assert all(0 <= idx < 20 for idx in mmr_indices)
    
    # Clean up
    memory_manager.cleanup()


@pytest.mark.skipif(not HAS_TORCH or not torch.cuda.is_available() or 
                    torch.cuda.device_count() < 2,
                    reason="At least 2 GPUs required")
def test_multi_gpu_manager(test_vectors, test_table):
    """Test Arrow Flight multi-GPU manager."""
    from langchain_hana.gpu.arrow_flight_multi_gpu import ArrowFlightMultiGPUManager
    from langchain_hana.gpu.arrow_flight_client import ArrowFlightClient
    
    vectors, texts, metadatas = test_vectors
    
    # Create flight clients for each GPU
    flight_clients = [
        ArrowFlightClient(
            host=FLIGHT_HOST,
            port=FLIGHT_PORT,
            use_tls=False,
            username=HANA_USER,
            password=HANA_PASSWORD
        )
        for _ in range(2)  # For 2 GPUs
    ]
    
    # Initialize multi-GPU manager
    mgpu_manager = ArrowFlightMultiGPUManager(
        flight_clients=flight_clients,
        gpu_ids=[0, 1],  # Use GPUs 0 and 1
        batch_size=10,
        distribution_strategy="round_robin"
    )
    
    # Test upload_vectors_multi_gpu
    ids = mgpu_manager.upload_vectors_multi_gpu(
        table_name=test_table,
        vectors=vectors[20:40],  # Use next 20 vectors
        texts=texts[20:40],
        metadata=metadatas[20:40]
    )
    
    assert len(ids) == 20
    
    # Test similarity_search_multi_gpu
    results = mgpu_manager.similarity_search_multi_gpu(
        table_name=test_table,
        query_vectors=vectors[:5],  # Use first 5 vectors as queries
        k=3,
        include_metadata=True
    )
    
    assert len(results) == 5
    assert len(results[0]) == 3  # 3 results per query
    
    # Test batch_similarity_search_multi_gpu
    distances, indices = mgpu_manager.batch_similarity_search_multi_gpu(
        query_vectors=vectors[:5],
        stored_vectors=vectors[20:40],
        k=3,
        metric="cosine"
    )
    
    assert isinstance(distances, np.ndarray)
    assert isinstance(indices, np.ndarray)
    assert distances.shape == (5, 3)
    assert indices.shape == (5, 3)
    
    # Clean up
    mgpu_manager.close()


def test_benchmark_utilities(test_vectors, test_table):
    """Test benchmark utilities."""
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../benchmarks")))
    from arrow_flight_benchmark import (
        benchmark_serialization_performance,
        load_or_generate_test_data
    )
    
    vectors, _, _ = test_vectors
    
    # Test serialization benchmark
    results = benchmark_serialization_performance(
        vectors=vectors[:10],
        num_iterations=2
    )
    
    assert "methods" in results
    assert "binary" in results["methods"]
    assert "arrow_batch" in results["methods"]
    
    # Test data generation
    data_vectors, data_texts, data_metadatas = load_or_generate_test_data(
        num_documents=10,
        embedding_dim=4,
        file_path=None
    )
    
    assert len(data_vectors) == 10
    assert len(data_texts) == 10
    assert len(data_metadatas) == 10