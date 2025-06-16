"""
Unit tests for the langchain-hana integration.

This module contains comprehensive unit tests for the SAP HANA Cloud integration with LangChain,
focusing on the VectorStore implementation. These tests use mocking to avoid actual database
connections while testing:

1. Connection management
2. Vector serialization/deserialization 
3. Basic vectorstore operations (add, search, delete)
4. Advanced search (MMR, filtering)
5. Distance strategies
"""

import unittest
import json
from unittest.mock import MagicMock, patch, call
from typing import List, Dict, Any, Tuple

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from langchain_hana_integration.vectorstore import SAP_HANA_VectorStore
from langchain_hana_integration.connection import get_connection, create_connection_pool
from langchain_hana_integration.utils.distance import DistanceStrategy, compute_similarity, normalize_vector
from langchain_hana_integration.utils.serialization import (
    serialize_vector, 
    deserialize_vector, 
    serialize_metadata,
    deserialize_metadata
)


# Mock data for tests
MOCK_TEXTS = [
    "This is the first document about finance",
    "This is the second document about technology",
    "This is the third document about healthcare",
    "This is the fourth document about science",
    "This is the fifth document about history"
]

MOCK_METADATAS = [
    {"source": "finance_docs", "id": "1", "category": "finance", "priority": 1},
    {"source": "tech_docs", "id": "2", "category": "technology", "priority": 2},
    {"source": "health_docs", "id": "3", "category": "healthcare", "priority": 1},
    {"source": "science_docs", "id": "4", "category": "science", "priority": 3},
    {"source": "history_docs", "id": "5", "category": "history", "priority": 2}
]

MOCK_EMBEDDINGS = [
    [0.1, 0.2, 0.3, 0.4, 0.5],
    [0.2, 0.3, 0.4, 0.5, 0.6],
    [0.3, 0.4, 0.5, 0.6, 0.7],
    [0.4, 0.5, 0.6, 0.7, 0.8],
    [0.5, 0.6, 0.7, 0.8, 0.9]
]


class MockEmbeddings(Embeddings):
    """Mock embedding class for testing."""
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return mock embeddings for documents."""
        # For simplicity, return pre-defined embeddings based on index
        # or the last embedding if out of range
        result = []
        for i in range(len(texts)):
            if i < len(MOCK_EMBEDDINGS):
                result.append(MOCK_EMBEDDINGS[i])
            else:
                result.append(MOCK_EMBEDDINGS[-1])
        return result
        
    def embed_query(self, text: str) -> List[float]:
        """Return mock embedding for query."""
        # For testing, we'll make the query embedding match the second document
        return MOCK_EMBEDDINGS[1]


class TestSerializationUtils(unittest.TestCase):
    """Test vector and metadata serialization/deserialization utilities."""
    
    def test_serialize_deserialize_real_vector(self):
        """Test serializing and deserializing real vectors."""
        vector = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        # Serialize
        binary = serialize_vector(vector, vector_type="REAL_VECTOR")
        
        # Deserialize
        result = deserialize_vector(binary, vector_type="REAL_VECTOR")
        
        # Check results
        self.assertEqual(len(result), len(vector))
        for i in range(len(vector)):
            self.assertAlmostEqual(result[i], vector[i], places=5)
    
    def test_serialize_deserialize_half_vector(self):
        """Test serializing and deserializing half vectors."""
        vector = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        try:
            # Serialize
            binary = serialize_vector(vector, vector_type="HALF_VECTOR")
            
            # Deserialize
            result = deserialize_vector(binary, vector_type="HALF_VECTOR")
            
            # Check results (less precision for half vectors)
            self.assertEqual(len(result), len(vector))
            for i in range(len(vector)):
                self.assertAlmostEqual(result[i], vector[i], places=2)
        except ImportError:
            self.skipTest("NumPy not available for HALF_VECTOR tests")
    
    def test_serialize_empty_vector(self):
        """Test serializing an empty vector raises an error."""
        with self.assertRaises(ValueError):
            serialize_vector([], vector_type="REAL_VECTOR")
    
    def test_serialize_invalid_vector_type(self):
        """Test serializing with invalid vector type raises an error."""
        with self.assertRaises(ValueError):
            serialize_vector([0.1, 0.2, 0.3], vector_type="INVALID_TYPE")
    
    def test_deserialize_invalid_data(self):
        """Test deserializing invalid data raises an error."""
        with self.assertRaises(ValueError):
            deserialize_vector(b'', vector_type="REAL_VECTOR")
        
        with self.assertRaises(ValueError):
            deserialize_vector(b'123', vector_type="REAL_VECTOR")
    
    def test_serialize_deserialize_metadata(self):
        """Test serializing and deserializing metadata."""
        metadata = {
            "source": "test_source",
            "id": "123",
            "categories": ["finance", "technology"],
            "priority": 2,
            "is_active": True
        }
        
        # Serialize
        json_str = serialize_metadata(metadata)
        
        # Deserialize
        result = deserialize_metadata(json_str)
        
        # Check results
        self.assertEqual(result["source"], metadata["source"])
        self.assertEqual(result["id"], metadata["id"])
        self.assertEqual(result["categories"], metadata["categories"])
        self.assertEqual(result["priority"], metadata["priority"])
        self.assertEqual(result["is_active"], metadata["is_active"])
    
    def test_serialize_metadata_with_invalid_keys(self):
        """Test serializing metadata with invalid keys raises an error."""
        with self.assertRaises(ValueError):
            serialize_metadata({"invalid key": "value"})
        
        with self.assertRaises(ValueError):
            serialize_metadata({"invalid-key": "value"})
    
    def test_deserialize_invalid_metadata(self):
        """Test deserializing invalid metadata raises an error."""
        with self.assertRaises(ValueError):
            deserialize_metadata("{invalid json")


class TestDistanceUtils(unittest.TestCase):
    """Test distance computation utilities."""
    
    def test_normalize_vector(self):
        """Test vector normalization."""
        vector = [1.0, 2.0, 3.0]
        normalized = normalize_vector(vector)
        
        # Check that the normalized vector has unit length
        norm = np.linalg.norm(normalized)
        self.assertAlmostEqual(norm, 1.0, places=5)
    
    def test_normalize_zero_vector(self):
        """Test normalizing a zero vector returns the same vector."""
        vector = [0.0, 0.0, 0.0]
        normalized = normalize_vector(vector)
        self.assertEqual(normalized, vector)
    
    def test_compute_cosine_similarity(self):
        """Test computing cosine similarity."""
        v1 = [1.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0]
        v3 = [1.0, 1.0, 0.0]
        
        # Perpendicular vectors have similarity 0
        similarity = compute_similarity(v1, v2, DistanceStrategy.COSINE)
        self.assertAlmostEqual(similarity, 0.0, places=5)
        
        # Same vector has similarity 1
        similarity = compute_similarity(v1, v1, DistanceStrategy.COSINE)
        self.assertAlmostEqual(similarity, 1.0, places=5)
        
        # 45 degree angle has similarity 0.7071
        similarity = compute_similarity(v1, v3, DistanceStrategy.COSINE)
        self.assertAlmostEqual(similarity, 0.7071, places=4)
    
    def test_compute_euclidean_distance(self):
        """Test computing Euclidean distance."""
        v1 = [1.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0]
        
        # Distance between unit vectors at right angles is sqrt(2)
        distance = compute_similarity(v1, v2, DistanceStrategy.EUCLIDEAN_DISTANCE)
        self.assertAlmostEqual(distance, np.sqrt(2), places=5)
        
        # Distance to self is 0
        distance = compute_similarity(v1, v1, DistanceStrategy.EUCLIDEAN_DISTANCE)
        self.assertAlmostEqual(distance, 0.0, places=5)
    
    def test_compute_dot_product(self):
        """Test computing dot product."""
        v1 = [1.0, 2.0, 3.0]
        v2 = [4.0, 5.0, 6.0]
        
        # Dot product of these vectors is 1*4 + 2*5 + 3*6 = 32
        dot_product = compute_similarity(v1, v2, DistanceStrategy.DOT_PRODUCT)
        self.assertAlmostEqual(dot_product, 32.0, places=5)
    
    def test_compute_l1_distance(self):
        """Test computing L1 distance (Manhattan distance)."""
        v1 = [1.0, 2.0, 3.0]
        v2 = [4.0, 5.0, 6.0]
        
        # L1 distance is |1-4| + |2-5| + |3-6| = 3 + 3 + 3 = 9
        distance = compute_similarity(v1, v2, DistanceStrategy.L1_DISTANCE)
        self.assertAlmostEqual(distance, 9.0, places=5)
    
    def test_compute_similarity_different_dimensions(self):
        """Test computing similarity between vectors of different dimensions raises an error."""
        v1 = [1.0, 2.0, 3.0]
        v2 = [1.0, 2.0]
        
        with self.assertRaises(ValueError):
            compute_similarity(v1, v2, DistanceStrategy.COSINE)
    
    def test_compute_similarity_invalid_strategy(self):
        """Test computing similarity with an invalid strategy raises an error."""
        v1 = [1.0, 2.0, 3.0]
        v2 = [4.0, 5.0, 6.0]
        
        with self.assertRaises(ValueError):
            # Using string instead of enum
            compute_similarity(v1, v2, "invalid_strategy")


@patch('langchain_hana_integration.connection.dbapi.connect')
class TestConnectionManagement(unittest.TestCase):
    """Test connection management utilities."""
    
    def test_create_connection_pool(self, mock_connect):
        """Test creating a connection pool."""
        # Set up the mock
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        
        # Create pool
        create_connection_pool(
            connection_params={
                "host": "test-host",
                "port": 123,
                "user": "test-user",
                "password": "test-password"
            },
            pool_name="test_pool"
        )
        
        # Get connection from pool
        conn = get_connection("test_pool")
        
        # Check results
        mock_connect.assert_called_once()
        self.assertEqual(conn, mock_conn)
    
    def test_get_connection_with_invalid_pool(self, mock_connect):
        """Test getting a connection from an invalid pool raises an error."""
        with self.assertRaises(ValueError):
            get_connection("non_existent_pool")


@patch('langchain_hana_integration.vectorstore.get_connection')
class TestSAP_HANA_VectorStore(unittest.TestCase):
    """Test the SAP HANA Vector Store implementation."""
    
    def setUp(self):
        """Set up common test fixtures."""
        self.embeddings = MockEmbeddings()
        self.mock_cursor = MagicMock()
        self.mock_connection = MagicMock()
        self.mock_connection.cursor.return_value = self.mock_cursor
        
        # Set up cursor behavior for table check
        self.mock_cursor.fetchone.return_value = [0]  # Table doesn't exist
    
    def test_initialization(self, mock_get_connection):
        """Test initializing the vector store."""
        # Set up the mock
        mock_get_connection.return_value.__enter__.return_value = self.mock_connection
        
        # Create vector store
        store = SAP_HANA_VectorStore(
            embedding=self.embeddings,
            connection_params={
                "host": "test-host",
                "port": 123,
                "user": "test-user",
                "password": "test-password"
            },
            table_name="test_table",
            create_table=True
        )
        
        # Check that the connection was used
        mock_get_connection.assert_called()
        self.mock_connection.cursor.assert_called()
        
        # Check that create table was attempted
        self.mock_cursor.execute.assert_called()
        create_call = self.mock_cursor.execute.call_args_list[0]
        self.assertIn("SELECT COUNT(*)", create_call[0][0])  # First check if table exists
        
        # Since mock returns 0 (table doesn't exist), it should create the table
        create_table_call = self.mock_cursor.execute.call_args_list[1]
        self.assertIn("CREATE TABLE", create_table_call[0][0])
    
    def test_add_texts(self, mock_get_connection):
        """Test adding texts to the vector store."""
        # Set up the mock
        mock_get_connection.return_value.__enter__.return_value = self.mock_connection
        
        # Create vector store with mock table already existing
        store = SAP_HANA_VectorStore(
            embedding=self.embeddings,
            connection_params={"host": "test-host"},
            table_name="test_table",
            create_table=False
        )
        
        # Add texts
        store.add_texts(MOCK_TEXTS[:2], MOCK_METADATAS[:2])
        
        # Check that executemany was called for batch insert
        self.mock_cursor.executemany.assert_called_once()
        
        # Check that the correct number of parameters were passed
        insert_call = self.mock_cursor.executemany.call_args
        self.assertIn("INSERT INTO", insert_call[0][0])
        self.assertEqual(len(insert_call[0][1]), 2)  # Two documents
        
        # Verify commit was called
        self.mock_connection.commit.assert_called_once()
    
    def test_similarity_search(self, mock_get_connection):
        """Test similarity search functionality."""
        # Set up the mock
        mock_get_connection.return_value.__enter__.return_value = self.mock_connection
        
        # Set up cursor to return search results
        self.mock_cursor.fetchall.return_value = [
            (
                MOCK_TEXTS[1],  # content
                json.dumps(MOCK_METADATAS[1]),  # metadata
                serialize_vector(MOCK_EMBEDDINGS[1], "REAL_VECTOR"),  # vector
                0.95  # similarity score
            ),
            (
                MOCK_TEXTS[0],  # content
                json.dumps(MOCK_METADATAS[0]),  # metadata
                serialize_vector(MOCK_EMBEDDINGS[0], "REAL_VECTOR"),  # vector
                0.85  # similarity score
            )
        ]
        
        # Create vector store
        store = SAP_HANA_VectorStore(
            embedding=self.embeddings,
            connection_params={"host": "test-host"},
            table_name="test_table",
            create_table=False
        )
        
        # Perform search
        results = store.similarity_search("test query", k=2)
        
        # Check that execute was called for the search
        search_calls = [call for call in self.mock_cursor.execute.call_args_list 
                        if "SELECT" in call[0][0] and "similarity_score" in call[0][0]]
        self.assertTrue(len(search_calls) > 0)
        
        # Check results
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].page_content, MOCK_TEXTS[1])
        self.assertEqual(results[0].metadata["source"], MOCK_METADATAS[1]["source"])
        self.assertEqual(results[0].metadata["category"], MOCK_METADATAS[1]["category"])
        self.assertEqual(results[1].page_content, MOCK_TEXTS[0])
    
    def test_similarity_search_with_score(self, mock_get_connection):
        """Test similarity search with scores."""
        # Set up the mock
        mock_get_connection.return_value.__enter__.return_value = self.mock_connection
        
        # Set up cursor to return search results
        self.mock_cursor.fetchall.return_value = [
            (
                MOCK_TEXTS[1],  # content
                json.dumps(MOCK_METADATAS[1]),  # metadata
                serialize_vector(MOCK_EMBEDDINGS[1], "REAL_VECTOR"),  # vector
                0.95  # similarity score
            ),
            (
                MOCK_TEXTS[0],  # content
                json.dumps(MOCK_METADATAS[0]),  # metadata
                serialize_vector(MOCK_EMBEDDINGS[0], "REAL_VECTOR"),  # vector
                0.85  # similarity score
            )
        ]
        
        # Create vector store
        store = SAP_HANA_VectorStore(
            embedding=self.embeddings,
            connection_params={"host": "test-host"},
            table_name="test_table",
            create_table=False
        )
        
        # Perform search with scores
        results = store.similarity_search_with_score("test query", k=2)
        
        # Check results
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0].page_content, MOCK_TEXTS[1])
        self.assertAlmostEqual(results[0][1], 0.95)
        self.assertEqual(results[1][0].page_content, MOCK_TEXTS[0])
        self.assertAlmostEqual(results[1][1], 0.85)
    
    def test_similarity_search_with_filter(self, mock_get_connection):
        """Test similarity search with filtering."""
        # Set up the mock
        mock_get_connection.return_value.__enter__.return_value = self.mock_connection
        
        # Set up cursor to return filtered search results
        self.mock_cursor.fetchall.return_value = [
            (
                MOCK_TEXTS[0],  # content
                json.dumps(MOCK_METADATAS[0]),  # metadata
                serialize_vector(MOCK_EMBEDDINGS[0], "REAL_VECTOR"),  # vector
                0.85  # similarity score
            )
        ]
        
        # Create vector store
        store = SAP_HANA_VectorStore(
            embedding=self.embeddings,
            connection_params={"host": "test-host"},
            table_name="test_table",
            create_table=False
        )
        
        # Perform search with filter
        filter_dict = {"category": "finance"}
        results = store.similarity_search("test query", k=2, filter=filter_dict)
        
        # Check that execute was called with a filter
        search_calls = [call for call in self.mock_cursor.execute.call_args_list 
                        if "SELECT" in call[0][0] and "WHERE" in call[0][0]]
        self.assertTrue(len(search_calls) > 0)
        
        # Check results
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].page_content, MOCK_TEXTS[0])
        self.assertEqual(results[0].metadata["category"], "finance")
    
    def test_advanced_filter_operators(self, mock_get_connection):
        """Test advanced filtering with comparison operators."""
        # Set up the mock
        mock_get_connection.return_value.__enter__.return_value = self.mock_connection
        
        # Set up cursor to return filtered search results
        self.mock_cursor.fetchall.return_value = [
            (
                MOCK_TEXTS[3],  # content
                json.dumps(MOCK_METADATAS[3]),  # metadata
                serialize_vector(MOCK_EMBEDDINGS[3], "REAL_VECTOR"),  # vector
                0.75  # similarity score
            )
        ]
        
        # Create vector store
        store = SAP_HANA_VectorStore(
            embedding=self.embeddings,
            connection_params={"host": "test-host"},
            table_name="test_table",
            create_table=False
        )
        
        # Perform search with advanced filter
        filter_dict = {"priority": {"$gt": 2}}
        results = store.similarity_search("test query", k=2, filter=filter_dict)
        
        # Check that execute was called with a filter containing >
        search_calls = [call for call in self.mock_cursor.execute.call_args_list 
                        if "SELECT" in call[0][0] and "WHERE" in call[0][0]]
        self.assertTrue(len(search_calls) > 0)
        
        # Check results
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].page_content, MOCK_TEXTS[3])
        self.assertEqual(results[0].metadata["priority"], 3)
    
    def test_delete_with_filter(self, mock_get_connection):
        """Test deleting documents with a filter."""
        # Set up the mock
        mock_get_connection.return_value.__enter__.return_value = self.mock_connection
        
        # Create vector store
        store = SAP_HANA_VectorStore(
            embedding=self.embeddings,
            connection_params={"host": "test-host"},
            table_name="test_table",
            create_table=False
        )
        
        # Delete with filter
        store.delete(filter={"category": "finance"})
        
        # Check that execute was called for the delete
        delete_calls = [call for call in self.mock_cursor.execute.call_args_list 
                        if "DELETE FROM" in call[0][0]]
        self.assertTrue(len(delete_calls) > 0)
        
        # Verify commit was called
        self.mock_connection.commit.assert_called()
    
    def test_delete_without_filter(self, mock_get_connection):
        """Test deleting documents without a filter raises an error."""
        # Set up the mock
        mock_get_connection.return_value.__enter__.return_value = self.mock_connection
        
        # Create vector store
        store = SAP_HANA_VectorStore(
            embedding=self.embeddings,
            connection_params={"host": "test-host"},
            table_name="test_table",
            create_table=False
        )
        
        # Delete without filter should raise error
        with self.assertRaises(ValueError):
            store.delete()
    
    def test_update_texts(self, mock_get_connection):
        """Test updating documents."""
        # Set up the mock
        mock_get_connection.return_value.__enter__.return_value = self.mock_connection
        
        # Create vector store
        store = SAP_HANA_VectorStore(
            embedding=self.embeddings,
            connection_params={"host": "test-host"},
            table_name="test_table",
            create_table=False
        )
        
        # Update texts
        store.update_texts(
            texts=["Updated document about finance"],
            filter={"category": "finance"},
            metadatas=[{"source": "finance_docs", "id": "1", "category": "finance", "priority": 2}]
        )
        
        # Check that execute was called for the update
        update_calls = [call for call in self.mock_cursor.execute.call_args_list 
                        if "UPDATE" in call[0][0]]
        self.assertTrue(len(update_calls) > 0)
        
        # Verify commit was called
        self.mock_connection.commit.assert_called()
    
    def test_create_hnsw_index(self, mock_get_connection):
        """Test creating an HNSW index."""
        # Set up the mock
        mock_get_connection.return_value.__enter__.return_value = self.mock_connection
        
        # Set up cursor to return no existing index
        self.mock_cursor.fetchone.side_effect = [[0], [0]]  # No table, no index
        
        # Create vector store
        store = SAP_HANA_VectorStore(
            embedding=self.embeddings,
            connection_params={"host": "test-host"},
            table_name="test_table",
            create_table=True
        )
        
        # Create index
        store.create_hnsw_index(
            m=64,
            ef_construction=128,
            ef_search=64,
            index_name="test_index"
        )
        
        # Check that execute was called for creating the index
        index_calls = [call for call in self.mock_cursor.execute.call_args_list 
                      if "CREATE HNSW VECTOR INDEX" in call[0][0]]
        self.assertTrue(len(index_calls) > 0)
        
        # Check that index parameters were included
        index_call = index_calls[0]
        self.assertIn("BUILD CONFIGURATION", index_call[0][0])
        self.assertIn("SEARCH CONFIGURATION", index_call[0][0])
        
        # Verify commit was called
        self.mock_connection.commit.assert_called()
    
    def test_max_marginal_relevance_search(self, mock_get_connection):
        """Test MMR search functionality."""
        # Set up the mock
        mock_get_connection.return_value.__enter__.return_value = self.mock_connection
        
        # Set up cursor to return initial search results
        self.mock_cursor.fetchall.side_effect = [
            [
                # Initial search results
                (
                    MOCK_TEXTS[1],
                    json.dumps(MOCK_METADATAS[1]),
                    serialize_vector(MOCK_EMBEDDINGS[1], "REAL_VECTOR"),
                    0.95
                ),
                (
                    MOCK_TEXTS[0],
                    json.dumps(MOCK_METADATAS[0]),
                    serialize_vector(MOCK_EMBEDDINGS[0], "REAL_VECTOR"),
                    0.85
                ),
                (
                    MOCK_TEXTS[2],
                    json.dumps(MOCK_METADATAS[2]),
                    serialize_vector(MOCK_EMBEDDINGS[2], "REAL_VECTOR"),
                    0.75
                )
            ],
            # For vector lookups
            [(serialize_vector(MOCK_EMBEDDINGS[1], "REAL_VECTOR"),)],
            [(serialize_vector(MOCK_EMBEDDINGS[0], "REAL_VECTOR"),)],
            [(serialize_vector(MOCK_EMBEDDINGS[2], "REAL_VECTOR"),)]
        ]
        
        # Create vector store
        store = SAP_HANA_VectorStore(
            embedding=self.embeddings,
            connection_params={"host": "test-host"},
            table_name="test_table",
            create_table=False
        )
        
        # Patch the _calculate_mmr method to return predictable indices
        with patch.object(store, '_calculate_mmr', return_value=[0, 2]):
            # Perform MMR search
            results = store.max_marginal_relevance_search(
                "test query",
                k=2,
                fetch_k=3,
                lambda_mult=0.5
            )
            
            # Check results
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0].page_content, MOCK_TEXTS[1])
            self.assertEqual(results[1].page_content, MOCK_TEXTS[2])


if __name__ == '__main__':
    unittest.main()