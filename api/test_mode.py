"""
Simple mock implementations for testing the API without a real HANA connection.
This module provides basic mock classes that simulate HANA functionality for testing purposes.
"""

import os
import json
import logging
import random
import struct
import sys
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from unittest.mock import MagicMock

# Mock GPU utilities
sys.modules['gpu_utils'] = MagicMock()
sys.modules['tensorrt_utils'] = MagicMock()

# Set environment variables for test mode
os.environ["HANA_MOCK_ENABLED"] = "true"
os.environ["GPU_MOCK_ENABLED"] = "true"

logger = logging.getLogger(__name__)

class MockConnection:
    """Mock HANA database connection for testing."""
    
    def __init__(self):
        """Initialize mock connection."""
        self.schema = "TEST_SCHEMA"
        self.is_connected = True
        self.cursors = []
        self.version = "2024.2 (QRC 1/2024)"
        self.available_datatypes = [
            "REAL_VECTOR", "VARCHAR", "NVARCHAR", "INTEGER", 
            "BIGINT", "FLOAT", "DOUBLE", "BLOB", "CLOB", "NCLOB"
        ]
        self.mock_tables = {
            "EMBEDDINGS": {
                "columns": [
                    {"name": "VEC_TEXT", "type": "NCLOB"},
                    {"name": "VEC_META", "type": "NCLOB"},
                    {"name": "VEC_VECTOR", "type": "REAL_VECTOR"}
                ],
                "data": []
            }
        }
    
    def cursor(self):
        """Create and return a mock cursor."""
        cursor = MockCursor(self)
        self.cursors.append(cursor)
        return cursor
    
    def commit(self):
        """Mock commit operation."""
        logger.debug("Mock connection: commit")
    
    def rollback(self):
        """Mock rollback operation."""
        logger.debug("Mock connection: rollback")
    
    def close(self):
        """Mock close operation."""
        logger.debug("Mock connection: close")
        self.is_connected = False


class MockCursor:
    """Mock HANA database cursor for testing."""
    
    def __init__(self, connection):
        """Initialize mock cursor with reference to mock connection."""
        self.connection = connection
        self._result_set = None
        self._has_result = False
        self._row_count = 0
        self.description = None
    
    def execute(self, query, *args, **kwargs):
        """Mock query execution."""
        logger.debug(f"Mock cursor executing: {query}")
        
        # Extract operation type from query
        query_upper = query.upper()
        
        if "CREATE TABLE" in query_upper:
            self._handle_create_table(query)
        elif "INSERT INTO" in query_upper:
            self._handle_insert(query, args, kwargs)
        elif "SELECT" in query_upper:
            self._handle_select(query, args, kwargs)
        elif "DELETE FROM" in query_upper:
            self._handle_delete(query)
        elif "CLOUD_VERSION FROM SYS.M_DATABASE" in query_upper:
            self._result_set = [(self.connection.version,)]
            self._has_result = True
        elif "TYPE_NAME FROM SYS.DATA_TYPES" in query_upper:
            self._result_set = [(dtype,) for dtype in self.connection.available_datatypes]
            self._has_result = True
        elif "SYS.TABLES" in query_upper:
            self._handle_table_check(query, args)
        elif "SYS.TABLE_COLUMNS" in query_upper:
            self._handle_column_check(query, args)
        elif "VECTOR_EMBEDDING" in query_upper:
            self._handle_vector_embedding(query, args, kwargs)
        else:
            # Default behavior for unhandled queries
            self._result_set = None
            self._has_result = False
    
    def executemany(self, query, params):
        """Mock batch query execution."""
        logger.debug(f"Mock cursor executemany: {query} with {len(params)} params")
        for param in params:
            if isinstance(param, dict):
                self.execute(query, **param)
            else:
                self.execute(query, *param)
    
    def fetchone(self):
        """Fetch one result from the result set."""
        if self._has_result and self._result_set and len(self._result_set) > 0:
            return self._result_set[0]
        return None
    
    def fetchall(self):
        """Fetch all results from the result set."""
        if self._has_result and self._result_set:
            return self._result_set
        return []
    
    def has_result_set(self):
        """Check if the cursor has a result set."""
        return self._has_result
    
    def close(self):
        """Close the cursor."""
        logger.debug("Mock cursor: close")
    
    def _handle_create_table(self, query):
        """Handle CREATE TABLE statements."""
        # Extract table name
        match = query.upper().find('CREATE TABLE')
        if match >= 0:
            # Simple parsing for demo purposes
            parts = query[match:].split()
            if len(parts) >= 3:
                table_name = parts[2].strip('"').strip("'")
                if table_name not in self.connection.mock_tables:
                    self.connection.mock_tables[table_name] = {
                        "columns": [],
                        "data": []
                    }
                logger.debug(f"Created mock table: {table_name}")
    
    def _handle_insert(self, query, args, kwargs):
        """Handle INSERT statements."""
        # For simplicity, just log that an insert happened
        self._row_count = 1
        self._has_result = False
        self._result_set = None
    
    def _handle_select(self, query, args, kwargs):
        """Handle SELECT statements."""
        if "SYS.DUMMY" in query.upper():
            # Handle dummy queries
            if "VECTOR_EMBEDDING" in query.upper():
                # Mock embedding vector generation
                vector = self._generate_mock_embedding(384)  # Default size
                self._result_set = [(vector,)]
                self._has_result = True
            else:
                # Default dummy result
                self._result_set = [(1,)]
                self._has_result = True
        elif "COSINE_SIMILARITY" in query.upper() or "L2DISTANCE" in query.upper():
            # Mock vector search results
            self._result_set = [
                ("Document 1", json.dumps({"source": "test1.pdf", "page": 1}), self._generate_mock_embedding(384), 0.95),
                ("Document 2", json.dumps({"source": "test2.pdf", "page": 5}), self._generate_mock_embedding(384), 0.85),
                ("Document 3", json.dumps({"source": "test3.pdf", "page": 2}), self._generate_mock_embedding(384), 0.75),
                ("Document 4", json.dumps({"source": "test4.pdf", "page": 7}), self._generate_mock_embedding(384), 0.65)
            ]
            self._has_result = True
        else:
            # Default behavior
            self._result_set = []
            self._has_result = True
    
    def _handle_delete(self, query):
        """Handle DELETE statements."""
        self._row_count = 0  # Pretend no rows were deleted
        self._has_result = False
        self._result_set = None
    
    def _handle_table_check(self, query, args):
        """Handle table existence checks."""
        if len(args) > 0 and args[0] in self.connection.mock_tables:
            self._result_set = [(1,)]
        else:
            self._result_set = [(0,)]
        self._has_result = True
    
    def _handle_column_check(self, query, args):
        """Handle column checks."""
        if len(args) < 2:
            self._result_set = []
            self._has_result = True
            return
            
        table_name, column_name = args
        
        if (table_name in self.connection.mock_tables and 
            any(col["name"] == column_name for col in self.connection.mock_tables[table_name]["columns"])):
            # Find the column
            for col in self.connection.mock_tables[table_name]["columns"]:
                if col["name"] == column_name:
                    col_type = col["type"]
                    col_length = 384 if col_type == "REAL_VECTOR" else 0
                    self._result_set = [(col_type, col_length)]
                    break
        else:
            self._result_set = []
        
        self._has_result = True
    
    def _handle_vector_embedding(self, query, args, kwargs):
        """Handle VECTOR_EMBEDDING function calls."""
        # Generate a mock embedding vector
        vector = self._generate_mock_embedding(384)  # Default size
        self._result_set = [(vector,)]
        self._has_result = True
    
    def _generate_mock_embedding(self, dim=384):
        """Generate a mock embedding vector with the specified dimension."""
        # Create random vector
        vector = [random.uniform(-1, 1) for _ in range(dim)]
        
        # Normalize (simplified version)
        magnitude = sum(x*x for x in vector) ** 0.5
        if magnitude > 0:
            vector = [x/magnitude for x in vector]
        
        # Convert to binary format (similar to HANA REAL_VECTOR)
        return struct.pack(f"<I{dim}f", dim, *vector)


def get_mock_connection(*args, **kwargs):
    """Get a mock HANA connection for testing."""
    return MockConnection()


def patch_hana_connection():
    """
    Patch the HANA connection with mock objects for testing.
    
    This function should be called at application startup when TEST_MODE=true.
    """
    import hdbcli.dbapi
    
    # Save original connect function for later restoration
    original_connect = hdbcli.dbapi.connect
    
    # Replace with mock implementation
    hdbcli.dbapi.connect = get_mock_connection
    
    logger.info("Patched HANA connection with mock implementation for testing")
    return original_connect


def is_test_mode_enabled():
    """Check if test mode is enabled via environment variable."""
    return os.environ.get("TEST_MODE", "").lower() in ("true", "1", "yes", "y")


# Initialize test mode if enabled
if is_test_mode_enabled():
    try:
        patch_hana_connection()
        logger.info("Test mode enabled: Using mock HANA implementation")
    except Exception as e:
        logger.error(f"Failed to initialize test mode: {str(e)}")