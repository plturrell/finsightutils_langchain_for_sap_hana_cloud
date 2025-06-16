"""
Unit tests for Arrow Flight components.

This module contains unit tests for the Arrow Flight integration components,
including the client, server, and memory manager.
"""

import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

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

# Skip tests if dependencies are not available
pytestmark = [
    pytest.mark.skipif(not HAS_ARROW_FLIGHT, reason="Arrow Flight not available"),
]


# Test vector serialization with Arrow format
def test_vector_serialization_arrow_format():
    """Test vector serialization with Arrow format."""
    from langchain_hana.gpu.vector_serialization import (
        vectors_to_arrow_batch,
        arrow_batch_to_vectors,
        serialize_arrow_batch,
        deserialize_arrow_batch
    )
    
    # Create test vectors
    vectors = [
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.9, 1.0, 1.1, 1.2]
    ]
    
    # Convert to Arrow batch
    batch = vectors_to_arrow_batch(vectors)
    assert isinstance(batch, pa.RecordBatch)
    assert batch.num_rows == 3
    assert "vector" in batch.schema.names
    
    # Serialize and deserialize batch
    serialized = serialize_arrow_batch(batch)
    assert isinstance(serialized, bytes)
    
    deserialized = deserialize_arrow_batch(serialized)
    assert isinstance(deserialized, pa.RecordBatch)
    assert deserialized.num_rows == 3
    assert "vector" in deserialized.schema.names
    
    # Convert back to vectors
    result_vectors = arrow_batch_to_vectors(deserialized)
    assert len(result_vectors) == 3
    
    # Check values are preserved (with small float precision error allowance)
    for i in range(len(vectors)):
        for j in range(len(vectors[i])):
            assert abs(vectors[i][j] - result_vectors[i][j]) < 1e-5


# Test Arrow Flight client (with mocked Flight server)
@patch("pyarrow.flight.FlightClient")
def test_arrow_flight_client(mock_flight_client):
    """Test Arrow Flight client with mocked Flight server."""
    from langchain_hana.gpu.arrow_flight_client import ArrowFlightClient
    
    # Configure mock
    mock_instance = MagicMock()
    mock_flight_client.return_value = mock_instance
    
    # Batch mock setup
    mock_batch = MagicMock()
    mock_batch.num_rows = 3
    
    # Mock do_put method
    mock_writer = MagicMock()
    mock_instance.do_put.return_value = (mock_writer, None)
    
    # Initialize client
    client = ArrowFlightClient(
        host="localhost",
        port=8815,
        use_tls=False,
        username="user",
        password="password"
    )
    
    # Test upload_vectors
    vectors = [
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.9, 1.0, 1.1, 1.2]
    ]
    texts = ["doc1", "doc2", "doc3"]
    metadata = [{"source": "test1"}, {"source": "test2"}, {"source": "test3"}]
    
    client.upload_vectors(
        table_name="test_table",
        vectors=vectors,
        texts=texts,
        metadata=metadata,
        batch_size=100
    )
    
    # Verify do_put was called
    mock_instance.do_put.assert_called_once()
    
    # Test similarity_search
    mock_info = MagicMock()
    mock_endpoint = MagicMock()
    mock_ticket = MagicMock()
    mock_endpoint.ticket = mock_ticket
    mock_info.endpoints = [mock_endpoint]
    mock_instance.get_flight_info.return_value = mock_info
    
    # Mock reader for do_get
    mock_reader = MagicMock()
    mock_batch_result = MagicMock()
    
    # Setup column mocks
    mock_id_col = MagicMock()
    mock_id_col.to_pylist.return_value = ["id1", "id2"]
    
    mock_doc_col = MagicMock()
    mock_doc_col.to_pylist.return_value = ["doc1", "doc2"]
    
    mock_meta_col = MagicMock()
    mock_meta_col.to_pylist.return_value = ['{"source": "test1"}', '{"source": "test2"}']
    
    mock_score_col = MagicMock()
    mock_score_col.to_pylist.return_value = [0.9, 0.8]
    
    # Setup batch schema names
    mock_batch_result.schema.names = ["id", "document", "metadata", "score"]
    
    # Setup column access
    def column_side_effect(name):
        if name == "id":
            return mock_id_col
        elif name == "document":
            return mock_doc_col
        elif name == "metadata":
            return mock_meta_col
        elif name == "score":
            return mock_score_col
        return MagicMock()
    
    mock_batch_result.column.side_effect = column_side_effect
    
    # Setup read_chunk to return the batch once then None
    read_chunk_returns = [(mock_batch_result, None), (None, None)]
    mock_reader.read_chunk.side_effect = read_chunk_returns
    
    mock_instance.do_get.return_value = mock_reader
    
    # Perform similarity search
    results = client.similarity_search(
        table_name="test_table",
        query_vector=[0.1, 0.2, 0.3, 0.4],
        k=2,
        include_metadata=True
    )
    
    # Verify get_flight_info and do_get were called
    mock_instance.get_flight_info.assert_called_once()
    mock_instance.do_get.assert_called_once()
    
    # Verify results
    assert len(results) == 2
    assert results[0]["id"] == "id1"
    assert results[0]["score"] == 0.9
    assert results[1]["id"] == "id2"
    assert results[1]["score"] == 0.8


# Test GPU-aware Arrow memory manager (if torch and CUDA available)
@pytest.mark.skipif(not HAS_TORCH or not torch.cuda.is_available(),
                   reason="PyTorch with CUDA not available")
def test_arrow_gpu_memory_manager():
    """Test GPU-aware Arrow memory manager with CUDA."""
    from langchain_hana.gpu.arrow_gpu_memory_manager import ArrowGpuMemoryManager
    
    # Initialize memory manager
    memory_manager = ArrowGpuMemoryManager(
        device_id=0,
        max_memory_fraction=0.5,
        batch_size=100
    )
    
    # Test get_optimal_batch_size
    batch_size = memory_manager.get_optimal_batch_size()
    assert batch_size > 0
    
    # Test vectors_to_fixed_size_list_array
    vectors = np.random.randn(10, 4).astype(np.float32)
    array = memory_manager.vectors_to_fixed_size_list_array(vectors)
    assert isinstance(array, pa.FixedSizeListArray)
    assert len(array) == 10
    
    # Test fixed_size_list_array_to_torch
    tensor = memory_manager.fixed_size_list_array_to_torch(array)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (10, 4)
    assert tensor.device.type == "cuda"
    
    # Test batch_similarity_search
    query_vectors = np.random.randn(5, 4).astype(np.float32)
    stored_vectors = np.random.randn(20, 4).astype(np.float32)
    
    # Convert to Arrow arrays
    query_array = memory_manager.vectors_to_fixed_size_list_array(query_vectors)
    stored_array = memory_manager.vectors_to_fixed_size_list_array(stored_vectors)
    
    # Perform batch similarity search
    distances, indices = memory_manager.batch_similarity_search(
        query_vectors=query_array,
        stored_vectors=stored_array,
        k=3,
        metric="cosine"
    )
    
    # Verify results
    assert isinstance(distances, torch.Tensor)
    assert isinstance(indices, torch.Tensor)
    assert distances.shape == (5, 3)
    assert indices.shape == (5, 3)
    
    # Test cleanup
    memory_manager.cleanup()


# Test Arrow Flight multi-GPU manager (with mocked components)
@patch("langchain_hana.gpu.arrow_flight_multi_gpu.ArrowFlightClient")
def test_arrow_flight_multi_gpu_manager(mock_client_class):
    """Test Arrow Flight multi-GPU manager with mocked components."""
    from langchain_hana.gpu.arrow_flight_multi_gpu import ArrowFlightMultiGPUManager
    
    # Configure mocks
    mock_client1 = MagicMock()
    mock_client2 = MagicMock()
    
    # Setup upload_vectors method mock
    mock_client1.upload_vectors.return_value = ["id1", "id2"]
    mock_client2.upload_vectors.return_value = ["id3", "id4"]
    
    # Setup similarity_search method mock
    mock_client1.similarity_search.return_value = [{"id": "id1", "score": 0.9}]
    mock_client2.similarity_search.return_value = [{"id": "id2", "score": 0.8}]
    
    # Create multi-GPU manager with mocked clients
    mgpu_manager = ArrowFlightMultiGPUManager(
        flight_clients=[mock_client1, mock_client2],
        gpu_ids=[0, 1],
        batch_size=100,
        distribution_strategy="round_robin"
    )
    
    # Test distribute_batch
    vectors = [
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.9, 1.0, 1.1, 1.2],
        [1.3, 1.4, 1.5, 1.6]
    ]
    texts = ["doc1", "doc2", "doc3", "doc4"]
    metadata = [
        {"source": "test1"},
        {"source": "test2"},
        {"source": "test3"},
        {"source": "test4"}
    ]
    
    batches = mgpu_manager.distribute_batch(vectors, texts, metadata)
    assert len(batches) == 2
    assert batches[0]["gpu_id"] == 0
    assert batches[1]["gpu_id"] == 1
    assert len(batches[0]["vectors"]) == 2
    assert len(batches[1]["vectors"]) == 2
    
    # Test upload_vectors_multi_gpu
    ids = mgpu_manager.upload_vectors_multi_gpu(
        table_name="test_table",
        vectors=vectors,
        texts=texts,
        metadata=metadata
    )
    
    assert len(ids) == 4
    mock_client1.upload_vectors.assert_called_once()
    mock_client2.upload_vectors.assert_called_once()
    
    # Test similarity_search_multi_gpu
    query_vectors = [
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8]
    ]
    
    results = mgpu_manager.similarity_search_multi_gpu(
        table_name="test_table",
        query_vectors=query_vectors,
        k=1
    )
    
    assert len(results) == 2
    mock_client1.similarity_search.assert_called_once()
    mock_client2.similarity_search.assert_called_once()
    
    # Test close
    mgpu_manager.close()
    mock_client1.close.assert_called_once()
    mock_client2.close.assert_called_once()