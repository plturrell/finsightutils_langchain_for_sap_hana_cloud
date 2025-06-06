"""
End-to-end tests for basic API functionality.

These tests verify the core functionality of the API, including:
- Health check endpoints
- Embedding generation
- Text storage and retrieval
- Query functionality
"""

import os
import time
import unittest
from typing import Dict, List, Any

from .base import BaseEndToEndTest, logger


class BasicFunctionalityTest(BaseEndToEndTest):
    """Test basic API functionality end-to-end."""
    
    def test_health_check(self):
        """Test the health check endpoint."""
        response, _ = self.api_request("GET", "/health/ping")
        self.assert_response_contains_keys(response, ["status", "timestamp", "version"])
        self.assertEqual(response["status"], "ok")
    
    def test_api_info(self):
        """Test the API info endpoint."""
        response, _ = self.api_request("GET", "/")
        self.assert_response_contains_keys(response, ["message", "version"])
        self.assertIn("SAP HANA Cloud Vector Store API", response["message"])
    
    def test_openapi_schema(self):
        """Test the OpenAPI schema endpoint."""
        response, _ = self.api_request("GET", "/openapi.json")
        self.assert_response_contains_keys(response, ["openapi", "info", "paths"])
        self.assertIn("title", response["info"])
        self.assertIn("version", response["info"])
        self.assertIn("/query", response["paths"])
    
    def test_embedding_generation(self):
        """Test the embedding generation endpoint."""
        test_texts = ["This is a test sentence.", "Another test sentence for embedding."]
        
        response, _ = self.api_request("POST", "/embeddings", {
            "texts": test_texts,
            "model": "all-MiniLM-L6-v2"  # Specify the model explicitly
        })
        
        self.assert_response_contains_keys(response, ["embeddings", "count", "dimensions"])
        self.assertEqual(response["count"], len(test_texts))
        self.assertTrue(response["dimensions"] > 0)
        
        # Validate each embedding
        for embedding in response["embeddings"]:
            self.assert_embedding_dimensions(embedding)
    
    def test_add_and_query_texts(self):
        """Test adding texts and querying them."""
        # Create unique test data
        timestamp = int(time.time())
        test_texts = [
            f"This is a test document about SAP HANA Cloud ({timestamp}).",
            f"Vector stores enable semantic search capabilities ({timestamp}).",
            f"GPU acceleration improves embedding generation performance ({timestamp}).",
        ]
        test_metadatas = [
            {"source": "e2e-test", "category": "database", "timestamp": timestamp},
            {"source": "e2e-test", "category": "vector-search", "timestamp": timestamp},
            {"source": "e2e-test", "category": "performance", "timestamp": timestamp},
        ]
        
        # First, add the texts
        add_response, _ = self.api_request("POST", "/texts", {
            "texts": test_texts,
            "metadatas": test_metadatas
        })
        
        self.assert_response_contains_keys(add_response, ["success", "message"])
        self.assertTrue(add_response["success"])
        
        # Now query for one of the texts
        query_text = "GPU acceleration performance"
        query_response, _ = self.api_request("POST", "/query", {
            "query": query_text,
            "k": 2,
            "filter": {"source": "e2e-test", "timestamp": timestamp}
        })
        
        self.assert_response_contains_keys(query_response, ["results"])
        self.assertGreaterEqual(len(query_response["results"]), 1)
        
        # The most relevant result should be the one about GPU acceleration
        self.assertIn("GPU acceleration", query_response["results"][0]["text"])
        
        # Check that metadata is returned
        self.assert_response_contains_keys(query_response["results"][0], ["text", "metadata"])
        self.assertEqual(query_response["results"][0]["metadata"]["category"], "performance")
        
        # Clean up test data
        delete_response, _ = self.api_request("POST", "/delete", {
            "filter": {"source": "e2e-test", "timestamp": timestamp}
        })
        
        self.assert_response_contains_keys(delete_response, ["success", "message"])
        self.assertTrue(delete_response["success"])
    
    def test_mmr_search(self):
        """Test MMR search functionality for diverse results."""
        # Create unique test data with deliberate similarity
        timestamp = int(time.time())
        test_texts = [
            f"SAP HANA Cloud is a powerful in-memory database platform ({timestamp}).",
            f"SAP HANA Cloud offers high-performance analytics capabilities ({timestamp}).",
            f"SAP HANA Cloud provides real-time insights for businesses ({timestamp}).",
            f"Vector embeddings enable semantic search in applications ({timestamp}).",
            f"Embedding vectors capture the meaning of text ({timestamp}).",
        ]
        test_metadatas = [
            {"source": "e2e-test", "category": "database", "subcategory": "overview", "timestamp": timestamp},
            {"source": "e2e-test", "category": "database", "subcategory": "analytics", "timestamp": timestamp},
            {"source": "e2e-test", "category": "database", "subcategory": "benefits", "timestamp": timestamp},
            {"source": "e2e-test", "category": "vector-search", "subcategory": "applications", "timestamp": timestamp},
            {"source": "e2e-test", "category": "vector-search", "subcategory": "theory", "timestamp": timestamp},
        ]
        
        # Add the texts
        add_response, _ = self.api_request("POST", "/texts", {
            "texts": test_texts,
            "metadatas": test_metadatas
        })
        
        self.assertTrue(add_response["success"])
        
        # Regular query should return similar results (all about SAP HANA)
        regular_query_response, _ = self.api_request("POST", "/query", {
            "query": "SAP HANA Cloud capabilities",
            "k": 3,
            "filter": {"source": "e2e-test", "timestamp": timestamp}
        })
        
        # All results should be about HANA
        for result in regular_query_response["results"]:
            self.assertIn("SAP HANA Cloud", result["text"])
        
        # MMR query should return more diverse results
        mmr_query_response, _ = self.api_request("POST", "/query/mmr", {
            "query": "SAP HANA Cloud capabilities",
            "k": 3,
            "fetch_k": 5,
            "lambda_mult": 0.3,  # Lower lambda for more diversity
            "filter": {"source": "e2e-test", "timestamp": timestamp}
        })
        
        # Check that we have at least one result about something other than HANA
        non_hana_results = [r for r in mmr_query_response["results"] if "vector" in r["text"].lower()]
        self.assertGreaterEqual(len(non_hana_results), 1, "MMR search should return diverse results")
        
        # Clean up test data
        delete_response, _ = self.api_request("POST", "/delete", {
            "filter": {"source": "e2e-test", "timestamp": timestamp}
        })
        
        self.assertTrue(delete_response["success"])
    
    def test_gpu_info_endpoint(self):
        """Test the GPU information endpoint."""
        response, _ = self.api_request("GET", "/gpu/info")
        self.assert_response_contains_keys(response, ["gpu_available", "torch_available"])


if __name__ == "__main__":
    unittest.main()