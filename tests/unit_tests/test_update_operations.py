"""
Unit tests for update operations in the SAP HANA Cloud Vector Engine.

These tests verify that the update and upsert operations work correctly 
for the HanaDB vectorstore class.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import json
import struct
import numpy as np

from hdbcli import dbapi
from langchain_core.documents import Document
from langchain_hana import HanaDB
from langchain_hana.embeddings import HanaInternalEmbeddings
from langchain_hana.utils import DistanceStrategy


class TestHanaDBUpdateOperations(unittest.TestCase):
    """Test case for HanaDB update operations."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock connection and cursor
        self.mock_cursor = Mock()
        self.mock_connection = Mock()
        self.mock_connection.cursor.return_value = self.mock_cursor
        
        # Create mock embedding function
        self.mock_embedding = Mock()
        self.mock_embedding.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]
        self.mock_embedding.embed_documents.return_value = [[0.1, 0.2, 0.3, 0.4]]
        
        # Create HanaDB instance with mocks
        self.hana_db = HanaDB(
            connection=self.mock_connection,
            embedding=self.mock_embedding,
            distance_strategy=DistanceStrategy.COSINE,
            table_name="TEST_EMBEDDINGS",
            content_column="CONTENT",
            metadata_column="METADATA",
            vector_column="VECTOR"
        )
        
        # Mock _sanitize_metadata_keys to return input unchanged
        self.hana_db._sanitize_metadata_keys = lambda x: x
        
        # Mock _serialize_binary_format to return a dummy binary
        self.hana_db._serialize_binary_format = lambda x: b'binary_data'
        
    def test_update_texts_with_external_embedding(self):
        """Test updating texts with external embeddings."""
        # Setup
        texts = ["Updated document content"]
        metadatas = [{"source": "test", "author": "unit_test"}]
        filter_dict = {"author": "unit_test"}
        
        # Set result for where clause generation
        self.hana_db._update_texts_using_external_embedding = MagicMock(return_value=True)
        
        # Execute
        result = self.hana_db.update_texts(
            texts=texts, 
            filter=filter_dict,
            metadatas=metadatas
        )
        
        # Assert
        self.assertTrue(result)
        self.hana_db._update_texts_using_external_embedding.assert_called_once()
        
    def test_update_texts_with_internal_embedding(self):
        """Test updating texts with internal embeddings."""
        # Setup
        texts = ["Updated document content"]
        metadatas = [{"source": "test", "author": "unit_test"}]
        filter_dict = {"author": "unit_test"}
        
        # Configure HanaDB to use internal embeddings
        self.hana_db.use_internal_embeddings = True
        self.hana_db.internal_embedding_model_id = "test_model"
        self.hana_db._update_texts_using_internal_embedding = MagicMock(return_value=True)
        
        # Execute
        result = self.hana_db.update_texts(
            texts=texts, 
            filter=filter_dict,
            metadatas=metadatas
        )
        
        # Assert
        self.assertTrue(result)
        self.hana_db._update_texts_using_internal_embedding.assert_called_once()
        
    def test_update_texts_without_updating_embeddings(self):
        """Test updating texts without regenerating embeddings."""
        # Setup
        texts = ["Updated document content"]
        metadatas = [{"source": "test", "author": "unit_test"}]
        filter_dict = {"author": "unit_test"}
        
        self.hana_db._update_texts_without_embeddings = MagicMock(return_value=True)
        
        # Execute
        result = self.hana_db.update_texts(
            texts=texts, 
            filter=filter_dict,
            metadatas=metadatas,
            update_embeddings=False
        )
        
        # Assert
        self.assertTrue(result)
        self.hana_db._update_texts_without_embeddings.assert_called_once()
        
    def test_upsert_texts_insert_new(self):
        """Test upserting texts when documents don't exist (insert)."""
        # Setup
        texts = ["New document content"]
        metadatas = [{"source": "test", "author": "unit_test"}]
        filter_dict = {"author": "unit_test"}
        
        # Mock cursor to return 0 count (no existing documents)
        self.mock_cursor.fetchone.return_value = [0]
        
        # Mock add_texts and update_texts
        self.hana_db.add_texts = MagicMock(return_value=[])
        self.hana_db.update_texts = MagicMock(return_value=True)
        
        # Execute
        result = self.hana_db.upsert_texts(
            texts=texts, 
            metadatas=metadatas,
            filter=filter_dict
        )
        
        # Assert
        self.assertEqual(result, [])
        self.hana_db.add_texts.assert_called_once()
        self.hana_db.update_texts.assert_not_called()
        
    def test_upsert_texts_update_existing(self):
        """Test upserting texts when documents exist (update)."""
        # Setup
        texts = ["Updated document content"]
        metadatas = [{"source": "test", "author": "unit_test"}]
        filter_dict = {"author": "unit_test"}
        
        # Mock cursor to return 1 count (existing documents)
        self.mock_cursor.fetchone.return_value = [1]
        
        # Mock add_texts and update_texts
        self.hana_db.add_texts = MagicMock(return_value=[])
        self.hana_db.update_texts = MagicMock(return_value=True)
        
        # Execute
        result = self.hana_db.upsert_texts(
            texts=texts, 
            metadatas=metadatas,
            filter=filter_dict
        )
        
        # Assert
        self.assertEqual(result, [])
        self.hana_db.update_texts.assert_called_once()
        self.hana_db.add_texts.assert_not_called()
        
    def test_upsert_texts_without_filter(self):
        """Test upserting texts without filter (should default to add)."""
        # Setup
        texts = ["New document content"]
        metadatas = [{"source": "test", "author": "unit_test"}]
        
        # Mock add_texts
        self.hana_db.add_texts = MagicMock(return_value=[])
        
        # Execute
        result = self.hana_db.upsert_texts(
            texts=texts, 
            metadatas=metadatas
        )
        
        # Assert
        self.assertEqual(result, [])
        self.hana_db.add_texts.assert_called_once()


if __name__ == '__main__':
    unittest.main()