"""
Tests for the SAP HANA Cloud Vector Store for LangChain.

This module contains unit tests for the LangChain integration with SAP HANA Cloud,
focusing on the VectorStore implementation, connection management, and utilities.
"""

import json
import unittest
import os
import pickle
import time
from unittest.mock import MagicMock, patch, call
import numpy as np
import tempfile

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from langchain_hana.vectorstore import HanaVectorStore
from langchain_hana.connection import create_connection, test_connection, get_connection
from langchain_hana.utils import (
    DistanceStrategy, 
    serialize_vector, 
    deserialize_vector,
    create_vector_table,
    create_hnsw_index,
    convert_distance_strategy_to_sql
)
from langchain_hana.embeddings import HanaInternalEmbeddings, HanaEmbeddingsCache


# Mock data for tests
MOCK_TEXTS = [
    "This is the first document",
    "This is the second document",
    "This is the third document with more content"
]

MOCK_METADATAS = [
    {"source": "test1", "id": "1"},
    {"source": "test2", "id": "2"},
    {"source": "test3", "id": "3"}
]

MOCK_EMBEDDINGS = [
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6],
    [0.7, 0.8, 0.9]
]


class MockEmbeddings(Embeddings):
    """Mock embedding class for testing."""
    
    def embed_documents(self, texts):
        """Return mock embeddings for documents."""
        return MOCK_EMBEDDINGS[:len(texts)]
        
    def embed_query(self, text):
        """Return mock embedding for query."""
        return MOCK_EMBEDDINGS[1]  # Same as second document


class TestConnection(unittest.TestCase):
    """Test the connection utilities."""
    
    @patch('langchain_hana.connection.dbapi.connect')
    def test_create_connection(self, mock_connect):
        """Test creating a connection."""
        # Set up the mock
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        
        # Call the function
        conn = create_connection(
            host="test-host",
            port=123,
            user="test-user",
            password="test-password"
        )
        
        # Check the results
        mock_connect.assert_called_once()
        self.assertEqual(conn, mock_conn)
    
    @patch('langchain_hana.connection.dbapi.connect')
    def test_get_connection_with_dict(self, mock_connect):
        """Test get_connection with dictionary."""
        # Set up the mock
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        
        # Call the function
        conn = get_connection({
            "host": "test-host",
            "port": 123,
            "user": "test-user",
            "password": "test-password"
        })
        
        # Check the results
        mock_connect.assert_called_once()
        self.assertEqual(conn, mock_conn)
    
    @patch('langchain_hana.connection.dbapi.connect')
    def test_connection_from_env_vars(self, mock_connect):
        """Test creating a connection from environment variables."""
        # Set up the mock
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        
        # Mock environment variables
        with patch.dict('os.environ', {
            'HANA_HOST': 'env-host',
            'HANA_PORT': '456',
            'HANA_USER': 'env-user',
            'HANA_PASSWORD': 'env-password'
        }):
            # Call the function
            conn = create_connection()
            
            # Check the results
            mock_connect.assert_called_once()
            call_args = mock_connect.call_args[1]
            self.assertEqual(call_args['address'], 'env-host')
            self.assertEqual(call_args['port'], 456)
            self.assertEqual(call_args['user'], 'env-user')
            self.assertEqual(call_args['password'], 'env-password')
            self.assertEqual(conn, mock_conn)


class TestUtils(unittest.TestCase):
    """Test the utility functions."""
    
    def test_serialize_deserialize_vector(self):
        """Test serializing and deserializing vectors."""
        # Create a test vector
        vector = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        # Serialize
        binary = serialize_vector(vector, vector_type="REAL_VECTOR")
        
        # Deserialize
        result = deserialize_vector(binary, vector_type="REAL_VECTOR")
        
        # Check the results
        self.assertEqual(len(result), len(vector))
        for i in range(len(vector)):
            self.assertAlmostEqual(result[i], vector[i], places=5)
    
    def test_half_vector_serialize_deserialize(self):
        """Test serializing and deserializing half precision vectors."""
        # Create a test vector
        vector = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        # Serialize as HALF_VECTOR
        binary = serialize_vector(vector, vector_type="HALF_VECTOR")
        
        # Deserialize
        result = deserialize_vector(binary, vector_type="HALF_VECTOR")
        
        # Check the results
        self.assertEqual(len(result), len(vector))
        for i in range(len(vector)):
            self.assertAlmostEqual(result[i], vector[i], places=2)  # Lower precision
    
    def test_distance_strategy(self):
        """Test distance strategy enum."""
        # Test conversion to SQL function and sort order
        distance_func, sort_order = convert_distance_strategy_to_sql(DistanceStrategy.COSINE)
        self.assertEqual(distance_func, "COSINE_SIMILARITY")
        self.assertEqual(sort_order, "DESC")
        
        distance_func, sort_order = convert_distance_strategy_to_sql(DistanceStrategy.EUCLIDEAN_DISTANCE)
        self.assertEqual(distance_func, "L2DISTANCE")
        self.assertEqual(sort_order, "ASC")
    
    @patch('langchain_hana.utils.logger')
    def test_create_vector_table(self, mock_logger):
        """Test creating a vector table."""
        # Create mock cursor
        cursor = MagicMock()
        
        # Call function
        create_vector_table(
            cursor=cursor,
            table_name='"TEST"."VECTOR_TABLE"',
            content_column="CONTENT",
            metadata_column="METADATA",
            vector_column="VECTOR",
            vector_column_type="REAL_VECTOR",
            vector_column_length=384,
            if_not_exists=True
        )
        
        # Check SQL execution
        cursor.execute.assert_called_once()
        sql = cursor.execute.call_args[0][0]
        self.assertIn('CREATE TABLE IF NOT EXISTS "TEST"."VECTOR_TABLE"', sql)
        self.assertIn('"CONTENT" NCLOB', sql)
        self.assertIn('"METADATA" NCLOB', sql)
        self.assertIn('"VECTOR" REAL_VECTOR(384)', sql)
    
    @patch('langchain_hana.utils.logger')
    def test_create_hnsw_index(self, mock_logger):
        """Test creating an HNSW index."""
        # Create mock cursor
        cursor = MagicMock()
        
        # Call function
        create_hnsw_index(
            cursor=cursor,
            table_name='"TEST"."VECTOR_TABLE"',
            vector_column="VECTOR",
            distance_function="COSINE_SIMILARITY",
            m=16,
            ef_construction=128,
            ef_search=64
        )
        
        # Check SQL execution
        cursor.execute.assert_called_once()
        sql = cursor.execute.call_args[0][0]
        self.assertIn('CREATE HNSW VECTOR INDEX', sql)
        self.assertIn('"TEST"."VECTOR_TABLE"', sql)
        self.assertIn('SIMILARITY FUNCTION COSINE_SIMILARITY', sql)
        self.assertIn('"M":16', sql)
        self.assertIn('"efConstruction":128', sql)
        self.assertIn('"efSearch":64', sql)


@patch('langchain_hana.vectorstore.HanaVectorStore._create_table')
@patch('langchain_hana.vectorstore.test_connection')
class TestVectorStore(unittest.TestCase):
    """Test the vector store class."""
    
    def setUp(self):
        """Set up the test."""
        # Create a mock connection
        self.mock_conn = MagicMock()
        
        # Create a mock cursor
        self.mock_cursor = MagicMock()
        self.mock_conn.cursor.return_value = self.mock_cursor
        
        # Set up cursor behavior for table check
        self.mock_cursor.has_result_set.return_value = True
        self.mock_cursor.fetchone.return_value = [1]  # Table exists
        
        # Create embeddings
        self.embeddings = MockEmbeddings()
    
    def test_init(self, mock_test_conn, mock_create_table):
        """Test initializing the vector store."""
        # Set up the mocks
        mock_test_conn.return_value = (True, {"current_schema": "TEST"})
        
        # Create the vector store
        store = HanaVectorStore(
            connection=self.mock_conn,
            embedding=self.embeddings,
            table_name="test_table",
            create_table=True
        )
        
        # Check the results
        mock_test_conn.assert_called_once()
        mock_create_table.assert_called_once()
        self.assertEqual(store.table_name, "test_table")
        self.assertEqual(store.embedding, self.embeddings)
        self.assertFalse(store.use_internal_embeddings)
    
    def test_add_texts(self, mock_test_conn, mock_create_table):
        """Test adding texts."""
        # Set up the mocks
        mock_test_conn.return_value = (True, {"current_schema": "TEST"})
        
        # Create the vector store
        store = HanaVectorStore(
            connection=self.mock_conn,
            embedding=self.embeddings,
            table_name="test_table"
        )
        
        # Set up cursor behavior
        self.mock_cursor.executemany = MagicMock()
        
        # Add texts
        ids = store.add_texts(MOCK_TEXTS, MOCK_METADATAS)
        
        # Check the results
        self.mock_cursor.executemany.assert_called_once()
        self.assertEqual(len(ids), 3)
        
        # Check that commit was called
        self.mock_conn.commit.assert_called_once()
    
    def test_similarity_search(self, mock_test_conn, mock_create_table):
        """Test similarity search."""
        # Set up the mocks
        mock_test_conn.return_value = (True, {"current_schema": "TEST"})
        
        # Create the vector store
        store = HanaVectorStore(
            connection=self.mock_conn,
            embedding=self.embeddings,
            table_name="test_table"
        )
        
        # Set up cursor behavior for search
        self.mock_cursor.execute = MagicMock()
        self.mock_cursor.fetchall.return_value = [
            ("This is the second document", json.dumps({"source": "test2", "id": "2"}), b'\x03\x00\x00\x00\xcd\xcc\xcc\x3d\x00\x00\x00\x3f\x33\x33\x33\x3f', 0.95),
            ("This is the first document", json.dumps({"source": "test1", "id": "1"}), b'\x03\x00\x00\x00\xcd\xcc\xcc\x3d\x00\x00\x00\x3f\x33\x33\x33\x3f', 0.85),
        ]
        
        # Perform search
        results = store.similarity_search("test query", k=2)
        
        # Check the results
        self.mock_cursor.execute.assert_called_once()
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].page_content, "This is the second document")
        self.assertEqual(results[0].metadata["source"], "test2")
        self.assertEqual(results[0].metadata["id"], "2")
    
    def test_similarity_search_with_filter(self, mock_test_conn, mock_create_table):
        """Test similarity search with filter."""
        # Set up the mocks
        mock_test_conn.return_value = (True, {"current_schema": "TEST"})
        
        # Create the vector store
        store = HanaVectorStore(
            connection=self.mock_conn,
            embedding=self.embeddings,
            table_name="test_table"
        )
        
        # Set up cursor behavior for search
        self.mock_cursor.execute = MagicMock()
        self.mock_cursor.fetchall.return_value = [
            ("This is the second document", json.dumps({"source": "test2", "id": "2"}), b'\x03\x00\x00\x00\xcd\xcc\xcc\x3d\x00\x00\x00\x3f\x33\x33\x33\x3f', 0.95),
        ]
        
        # Perform search with filter
        filter_dict = {"source": "test2"}
        results = store.similarity_search("test query", k=2, filter=filter_dict)
        
        # Check the results
        self.mock_cursor.execute.assert_called_once()
        call_args = self.mock_cursor.execute.call_args[0]
        self.assertIn("WHERE", call_args[0])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].page_content, "This is the second document")
    
    def test_similarity_search_with_score(self, mock_test_conn, mock_create_table):
        """Test similarity search with score."""
        # Set up the mocks
        mock_test_conn.return_value = (True, {"current_schema": "TEST"})
        
        # Create the vector store
        store = HanaVectorStore(
            connection=self.mock_conn,
            embedding=self.embeddings,
            table_name="test_table"
        )
        
        # Set up cursor behavior for search
        self.mock_cursor.execute = MagicMock()
        self.mock_cursor.fetchall.return_value = [
            ("This is the second document", json.dumps({"source": "test2", "id": "2"}), b'\x03\x00\x00\x00\xcd\xcc\xcc\x3d\x00\x00\x00\x3f\x33\x33\x33\x3f', 0.95),
            ("This is the first document", json.dumps({"source": "test1", "id": "1"}), b'\x03\x00\x00\x00\xcd\xcc\xcc\x3d\x00\x00\x00\x3f\x33\x33\x33\x3f', 0.85),
        ]
        
        # Perform search with score
        results = store.similarity_search_with_score("test query", k=2)
        
        # Check the results
        self.mock_cursor.execute.assert_called_once()
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0].page_content, "This is the second document")
        self.assertEqual(results[0][1], 0.95)
    
    def test_delete(self, mock_test_conn, mock_create_table):
        """Test deleting documents."""
        # Set up the mocks
        mock_test_conn.return_value = (True, {"current_schema": "TEST"})
        
        # Create the vector store
        store = HanaVectorStore(
            connection=self.mock_conn,
            embedding=self.embeddings,
            table_name="test_table"
        )
        
        # Set up cursor behavior
        self.mock_cursor.execute = MagicMock()
        
        # Delete with filter
        result = store.delete(filter={"source": "test1"})
        
        # Check the results
        self.mock_cursor.execute.assert_called_once()
        call_args = self.mock_cursor.execute.call_args[0]
        self.assertIn("DELETE FROM", call_args[0])
        self.assertTrue(result)
        
        # Check that commit was called
        self.mock_conn.commit.assert_called_once()
    
    @patch('langchain_hana.vectorstore.maximal_marginal_relevance')
    def test_mmr_search(self, mock_mmr, mock_test_conn, mock_create_table):
        """Test MMR search."""
        # Set up the mocks
        mock_test_conn.return_value = (True, {"current_schema": "TEST"})
        mock_mmr.return_value = [1, 0]  # Return indices 1 and 0
        
        # Create the vector store
        store = HanaVectorStore(
            connection=self.mock_conn,
            embedding=self.embeddings,
            table_name="test_table"
        )
        
        # Mock the _similarity_search_with_embeddings method
        mock_results = [
            (Document(page_content="This is the second document", metadata={"source": "test2", "id": "2"}), 0.95, [0.4, 0.5, 0.6]),
            (Document(page_content="This is the first document", metadata={"source": "test1", "id": "1"}), 0.85, [0.1, 0.2, 0.3]),
        ]
        store._similarity_search_with_embeddings = MagicMock(return_value=mock_results)
        
        # Perform MMR search
        results = store.max_marginal_relevance_search(
            "test query",
            k=2,
            fetch_k=3,
            lambda_mult=0.5
        )
        
        # Check the results
        store._similarity_search_with_embeddings.assert_called_once()
        mock_mmr.assert_called_once()
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].page_content, "This is the first document")
        self.assertEqual(results[1].page_content, "This is the second document")


class TestInternalEmbeddings(unittest.TestCase):
    """Test the HanaInternalEmbeddings class."""
    
    def test_init(self):
        """Test initializing the HanaInternalEmbeddings."""
        embeddings = HanaInternalEmbeddings(model_id="SAP_NEB.20240301")
        self.assertEqual(embeddings.model_id, "SAP_NEB.20240301")
    
    def test_get_model_id(self):
        """Test the get_model_id method."""
        embeddings = HanaInternalEmbeddings(model_id="SAP_NEB.20240301")
        self.assertEqual(embeddings.get_model_id(), "SAP_NEB.20240301")
    
    def test_embed_documents_not_implemented(self):
        """Test that embed_documents raises NotImplementedError."""
        embeddings = HanaInternalEmbeddings(model_id="SAP_NEB.20240301")
        with self.assertRaises(NotImplementedError):
            embeddings.embed_documents(["Test document"])
    
    def test_embed_query_not_implemented(self):
        """Test that embed_query raises NotImplementedError."""
        embeddings = HanaInternalEmbeddings(model_id="SAP_NEB.20240301")
        with self.assertRaises(NotImplementedError):
            embeddings.embed_query("Test query")


class TestEmbeddingsCache(unittest.TestCase):
    """Test the HanaEmbeddingsCache class."""
    
    def setUp(self):
        """Set up the test."""
        self.base_embeddings = MockEmbeddings()
    
    def test_init(self):
        """Test initializing the cache."""
        cache = HanaEmbeddingsCache(
            base_embeddings=self.base_embeddings,
            ttl_seconds=3600,
            max_size=100
        )
        self.assertEqual(cache.base_embeddings, self.base_embeddings)
        self.assertEqual(cache.ttl_seconds, 3600)
        self.assertEqual(cache.max_size, 100)
        self.assertEqual(len(cache.query_cache), 0)
        self.assertEqual(len(cache.document_cache), 0)
    
    def test_embed_query_cache_miss(self):
        """Test embed_query with cache miss."""
        cache = HanaEmbeddingsCache(
            base_embeddings=self.base_embeddings,
            ttl_seconds=3600
        )
        
        # First call should be a cache miss
        result = cache.embed_query("test query")
        
        self.assertEqual(result, MOCK_EMBEDDINGS[1])
        self.assertEqual(cache.query_hits, 0)
        self.assertEqual(cache.query_misses, 1)
        self.assertEqual(len(cache.query_cache), 1)
    
    def test_embed_query_cache_hit(self):
        """Test embed_query with cache hit."""
        cache = HanaEmbeddingsCache(
            base_embeddings=self.base_embeddings,
            ttl_seconds=3600
        )
        
        # First call (cache miss)
        cache.embed_query("test query")
        
        # Second call (cache hit)
        result = cache.embed_query("test query")
        
        self.assertEqual(result, MOCK_EMBEDDINGS[1])
        self.assertEqual(cache.query_hits, 1)
        self.assertEqual(cache.query_misses, 1)
    
    def test_embed_documents_cache_miss(self):
        """Test embed_documents with cache miss."""
        cache = HanaEmbeddingsCache(
            base_embeddings=self.base_embeddings,
            ttl_seconds=3600
        )
        
        # First call should be a cache miss
        results = cache.embed_documents(["doc1", "doc2"])
        
        self.assertEqual(results, MOCK_EMBEDDINGS[:2])
        self.assertEqual(cache.document_hits, 0)
        self.assertEqual(cache.document_misses, 2)
        self.assertEqual(len(cache.document_cache), 2)
    
    def test_embed_documents_cache_hit(self):
        """Test embed_documents with cache hit."""
        cache = HanaEmbeddingsCache(
            base_embeddings=self.base_embeddings,
            ttl_seconds=3600
        )
        
        # First call (cache miss)
        cache.embed_documents(["doc1", "doc2"])
        
        # Second call (cache hit)
        results = cache.embed_documents(["doc1", "doc2"])
        
        self.assertEqual(results, MOCK_EMBEDDINGS[:2])
        self.assertEqual(cache.document_hits, 2)
        self.assertEqual(cache.document_misses, 2)
    
    def test_mixed_cache_hit_miss(self):
        """Test embed_documents with both hits and misses."""
        cache = HanaEmbeddingsCache(
            base_embeddings=self.base_embeddings,
            ttl_seconds=3600
        )
        
        # First call (all cache misses)
        cache.embed_documents(["doc1", "doc2"])
        
        # Second call (one hit, one miss)
        results = cache.embed_documents(["doc1", "doc3"])
        
        self.assertEqual(len(results), 2)
        self.assertEqual(cache.document_hits, 1)
        self.assertEqual(cache.document_misses, 3)
    
    def test_ttl_expiration(self):
        """Test that cache entries expire after TTL."""
        cache = HanaEmbeddingsCache(
            base_embeddings=self.base_embeddings,
            ttl_seconds=0.1  # Very short TTL for testing
        )
        
        # First call (cache miss)
        cache.embed_query("test query")
        
        # Wait for TTL to expire
        time.sleep(0.2)
        
        # Second call (should be a cache miss again due to TTL)
        cache.embed_query("test query")
        
        self.assertEqual(cache.query_hits, 0)
        self.assertEqual(cache.query_misses, 2)
    
    def test_max_size_enforcement(self):
        """Test that max_size is enforced."""
        cache = HanaEmbeddingsCache(
            base_embeddings=self.base_embeddings,
            max_size=2
        )
        
        # Add 3 items (exceeding max_size)
        cache.embed_query("query1")
        cache.embed_query("query2")
        cache.embed_query("query3")
        
        # Should only have 2 items in the cache
        self.assertEqual(len(cache.query_cache), 2)
        self.assertIn("query2", cache.query_cache)
        self.assertIn("query3", cache.query_cache)
        self.assertNotIn("query1", cache.query_cache)
    
    def test_clear_cache(self):
        """Test clearing the cache."""
        cache = HanaEmbeddingsCache(
            base_embeddings=self.base_embeddings
        )
        
        # Add items to cache
        cache.embed_query("query1")
        cache.embed_documents(["doc1", "doc2"])
        
        # Clear cache
        cache.clear_cache()
        
        self.assertEqual(len(cache.query_cache), 0)
        self.assertEqual(len(cache.document_cache), 0)
    
    def test_persist_and_load(self):
        """Test persisting and loading the cache."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            try:
                # Create cache with persistence
                cache = HanaEmbeddingsCache(
                    base_embeddings=self.base_embeddings,
                    persist_path=tmp.name
                )
                
                # Add items to cache
                cache.embed_query("query1")
                cache.embed_documents(["doc1", "doc2"])
                
                # Force persistence
                cache._persist_cache()
                
                # Create new cache that loads from disk
                new_cache = HanaEmbeddingsCache(
                    base_embeddings=self.base_embeddings,
                    persist_path=tmp.name,
                    load_on_init=True
                )
                
                self.assertEqual(len(new_cache.query_cache), 1)
                self.assertEqual(len(new_cache.document_cache), 2)
                self.assertIn("query1", new_cache.query_cache)
                self.assertIn("doc1", new_cache.document_cache)
                self.assertIn("doc2", new_cache.document_cache)
            finally:
                os.unlink(tmp.name)


class TestVectorStoreWithInternalEmbeddings(unittest.TestCase):
    """Test the VectorStore with HanaInternalEmbeddings."""
    
    @patch('langchain_hana.vectorstore.HanaVectorStore._create_table')
    @patch('langchain_hana.vectorstore.test_connection')
    def test_init_with_internal_embeddings(self, mock_test_conn, mock_create_table):
        """Test initializing the vector store with internal embeddings."""
        # Set up the mocks
        mock_test_conn.return_value = (True, {"current_schema": "TEST"})
        
        # Create a mock connection
        mock_conn = MagicMock()
        
        # Create internal embeddings
        embeddings = HanaInternalEmbeddings(model_id="SAP_NEB.20240301")
        
        # Create the vector store
        store = HanaVectorStore(
            connection=mock_conn,
            embedding=embeddings,
            table_name="test_table",
            create_table=True
        )
        
        # Check the results
        mock_test_conn.assert_called_once()
        mock_create_table.assert_called_once()
        self.assertEqual(store.table_name, "test_table")
        self.assertEqual(store.embedding, embeddings)
        self.assertTrue(store.use_internal_embeddings)
        self.assertEqual(store.internal_embedding_model_id, "SAP_NEB.20240301")
    
    @patch('langchain_hana.vectorstore.HanaVectorStore._create_table')
    @patch('langchain_hana.vectorstore.test_connection')
    def test_add_texts_with_internal_embeddings(self, mock_test_conn, mock_create_table):
        """Test adding texts with internal embeddings."""
        # Set up the mocks
        mock_test_conn.return_value = (True, {"current_schema": "TEST"})
        
        # Create a mock connection
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        
        # Create internal embeddings
        embeddings = HanaInternalEmbeddings(model_id="SAP_NEB.20240301")
        
        # Create the vector store
        store = HanaVectorStore(
            connection=mock_conn,
            embedding=embeddings,
            table_name="test_table"
        )
        
        # Set up cursor behavior
        mock_cursor.execute = MagicMock()
        
        # Add texts
        ids = store.add_texts(MOCK_TEXTS[:2], MOCK_METADATAS[:2])
        
        # Check the results
        self.assertEqual(len(ids), 2)
        self.assertEqual(mock_cursor.execute.call_count, 2)  # Once per document
        
        # Verify the SQL contains VECTOR_EMBEDDING function
        for call_args in mock_cursor.execute.call_args_list:
            sql = call_args[0][0]
            self.assertIn("VECTOR_EMBEDDING", sql)
            self.assertIn(":model_id", sql)
            
            # Check parameters
            params = call_args[0][1]
            self.assertEqual(params["model_id"], "SAP_NEB.20240301")
        
        # Check that commit was called
        mock_conn.commit.assert_called_once()
    
    @patch('langchain_hana.vectorstore.HanaVectorStore._create_table')
    @patch('langchain_hana.vectorstore.test_connection')
    def test_similarity_search_with_internal_embeddings(self, mock_test_conn, mock_create_table):
        """Test similarity search with internal embeddings."""
        # Set up the mocks
        mock_test_conn.return_value = (True, {"current_schema": "TEST"})
        
        # Create a mock connection
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        
        # Create internal embeddings
        embeddings = HanaInternalEmbeddings(model_id="SAP_NEB.20240301")
        
        # Create the vector store
        store = HanaVectorStore(
            connection=mock_conn,
            embedding=embeddings,
            table_name="test_table"
        )
        
        # Set up cursor behavior for search
        mock_cursor.execute = MagicMock()
        mock_cursor.fetchall.return_value = [
            ("This is the first document", json.dumps({"source": "test1", "id": "1"}), 0.95),
            ("This is the second document", json.dumps({"source": "test2", "id": "2"}), 0.85),
        ]
        
        # Perform search
        results = store.similarity_search("test query", k=2)
        
        # Check the results
        mock_cursor.execute.assert_called_once()
        sql = mock_cursor.execute.call_args[0][0]
        self.assertIn("VECTOR_EMBEDDING(?, 'QUERY', ?)", sql)
        
        params = mock_cursor.execute.call_args[0][1]
        self.assertEqual(params[0], "test query")
        self.assertEqual(params[1], "SAP_NEB.20240301")
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].page_content, "This is the first document")


class TestComplexFilters(unittest.TestCase):
    """Test complex filters in the VectorStore."""
    
    @patch('langchain_hana.vectorstore.HanaVectorStore._create_table')
    @patch('langchain_hana.vectorstore.test_connection')
    def setUp(self, mock_test_conn, mock_create_table):
        """Set up the test."""
        # Set up the mocks
        mock_test_conn.return_value = (True, {"current_schema": "TEST"})
        
        # Create a mock connection
        self.mock_conn = MagicMock()
        self.mock_cursor = MagicMock()
        self.mock_conn.cursor.return_value = self.mock_cursor
        
        # Create embeddings
        self.embeddings = MockEmbeddings()
        
        # Create the vector store
        self.store = HanaVectorStore(
            connection=self.mock_conn,
            embedding=self.embeddings,
            table_name="test_table"
        )
    
    def test_simple_filter(self):
        """Test a simple filter."""
        # Set up cursor behavior
        self.mock_cursor.execute = MagicMock()
        
        # Define filter
        filter_dict = {"source": "test1"}
        
        # Build filter clause
        where_clause, params = self.store._build_filter_clause(filter_dict)
        
        # Check results
        self.assertIn('JSON_VALUE("METADATA", \'$."source"\') = ?', where_clause)
        self.assertEqual(params[0], '"test1"')
    
    def test_numeric_comparison_filter(self):
        """Test numeric comparison filters."""
        # Set up cursor behavior
        self.mock_cursor.execute = MagicMock()
        
        # Define filter
        filter_dict = {"year": {"$gt": 2020}}
        
        # Build filter clause
        where_clause, params = self.store._build_filter_clause(filter_dict)
        
        # Check results
        self.assertIn('CAST(JSON_VALUE("METADATA", \'$."year"\') AS FLOAT) > ?', where_clause)
        self.assertEqual(params[0], 2020.0)
    
    def test_contains_filter(self):
        """Test contains filter."""
        # Set up cursor behavior
        self.mock_cursor.execute = MagicMock()
        
        # Define filter
        filter_dict = {"tags": {"$contains": "cloud"}}
        
        # Build filter clause
        where_clause, params = self.store._build_filter_clause(filter_dict)
        
        # Check results
        self.assertIn('CONTAINS(JSON_VALUE("METADATA", \'$."tags"\'), ?)', where_clause)
        self.assertEqual(params[0], "cloud")
    
    def test_in_filter(self):
        """Test in filter."""
        # Set up cursor behavior
        self.mock_cursor.execute = MagicMock()
        
        # Define filter
        filter_dict = {"category": {"$in": ["database", "cloud", "api"]}}
        
        # Build filter clause
        where_clause, params = self.store._build_filter_clause(filter_dict)
        
        # Check results
        self.assertIn('JSON_VALUE("METADATA", \'$."category"\') IN (?, ?, ?)', where_clause)
        self.assertEqual(params, ['"database"', '"cloud"', '"api"'])
    
    def test_complex_and_filter(self):
        """Test complex AND filter."""
        # Set up cursor behavior
        self.mock_cursor.execute = MagicMock()
        
        # Define filter
        filter_dict = {
            "category": "database",
            "year": {"$gte": 2020},
            "source": "test1"
        }
        
        # Build filter clause
        where_clause, params = self.store._build_filter_clause(filter_dict)
        
        # Check results
        self.assertEqual(len(params), 3)
        self.assertEqual(where_clause.count("AND"), 2)
    
    def test_similarity_search_with_complex_filter(self):
        """Test similarity search with a complex filter."""
        # Set up cursor behavior for search
        self.mock_cursor.execute = MagicMock()
        self.mock_cursor.fetchall.return_value = [
            ("This is the first document", json.dumps({"source": "test1", "id": "1", "year": 2022}), b'\x03\x00\x00\x00\xcd\xcc\xcc\x3d\x00\x00\x00\x3f\x33\x33\x33\x3f', 0.95),
        ]
        
        # Define complex filter
        filter_dict = {
            "source": "test1",
            "year": {"$gte": 2020}
        }
        
        # Perform search with filter
        results = self.store.similarity_search("test query", k=2, filter=filter_dict)
        
        # Check the results
        self.mock_cursor.execute.assert_called_once()
        call_args = self.mock_cursor.execute.call_args[0]
        self.assertIn("WHERE", call_args[0])
        self.assertIn("AND", call_args[0])
        self.assertEqual(len(results), 1)


if __name__ == '__main__':
    unittest.main()