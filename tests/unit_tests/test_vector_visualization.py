"""
Tests for vector visualization components and data preprocessing.

This module contains tests for the vector data preprocessing and 
visualization rendering functionality.
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import json
from typing import List, Dict, Any, Optional

# Mock implementation for testing (simulating the actual frontend component's logic)
class VectorVisualizationHelper:
    """Helper class that simulates the vector visualization component's data processing logic."""
    
    @staticmethod
    def reduce_dimensions(vectors: List[List[float]], method: str = 'tsne', perplexity: int = 30) -> List[List[float]]:
        """Simulate dimension reduction for testing purposes."""
        if not vectors or len(vectors) == 0:
            return []
            
        # For testing, just return simulated 2D coordinates
        # In real implementation, this would use UMAP, t-SNE, or PCA
        num_vectors = len(vectors)
        
        # Create deterministic 2D points based on the vectors
        # (in reality, dimension reduction is non-deterministic)
        points = []
        for i, vector in enumerate(vectors):
            # Use the first values of the vector to influence the point position
            # Scale to be between -10 and 10
            x = (sum(vector[:len(vector)//2]) % 20) - 10
            y = (sum(vector[len(vector)//2:]) % 20) - 10
            points.append([float(x), float(y)])
            
        return points
    
    @staticmethod
    def calculate_similarities(query_vector: List[float], document_vectors: List[List[float]]) -> List[float]:
        """Calculate cosine similarities between query vector and document vectors."""
        if not query_vector or not document_vectors:
            return []
            
        # Simplified cosine similarity calculation for testing
        similarities = []
        for doc_vector in document_vectors:
            # Normalize vectors
            query_norm = np.sqrt(sum([x**2 for x in query_vector]))
            doc_norm = np.sqrt(sum([x**2 for x in doc_vector]))
            
            if query_norm == 0 or doc_norm == 0:
                similarities.append(0.0)
                continue
                
            # Calculate dot product
            dot_product = sum([a * b for a, b in zip(query_vector, doc_vector)])
            
            # Calculate cosine similarity
            similarity = dot_product / (query_norm * doc_norm)
            similarities.append(float(similarity))
            
        return similarities
    
    @staticmethod
    def prepare_visualization_data(
        query_vector: Optional[List[float]],
        document_vectors: List[List[float]],
        document_metadata: List[Dict[str, Any]],
        document_contents: List[str],
        reduction_method: str = 'tsne'
    ) -> Dict[str, Any]:
        """Prepare data for visualization component."""
        if not document_vectors:
            return {
                "points": [],
                "metadata": [],
                "contents": [],
                "similarities": []
            }
            
        # Add query vector if provided
        all_vectors = document_vectors.copy()
        if query_vector:
            all_vectors = [query_vector] + all_vectors
            
        # Reduce dimensions to 2D
        points_2d = VectorVisualizationHelper.reduce_dimensions(all_vectors, method=reduction_method)
        
        # Calculate similarities if query vector is provided
        similarities = []
        if query_vector:
            similarities = VectorVisualizationHelper.calculate_similarities(query_vector, document_vectors)
            # Remove the query point from points_2d
            query_point = points_2d[0]
            points_2d = points_2d[1:]
            
        return {
            "points": points_2d,
            "metadata": document_metadata,
            "contents": document_contents,
            "similarities": similarities,
            "query_point": query_point if query_vector else None
        }


class TestVectorVisualization(unittest.TestCase):
    """Tests for vector visualization data processing."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample embedding vectors (768-dimensional)
        self.query_vector = [0.1] * 384  # 384-dimensional query vector
        self.document_vectors = [
            [0.2] * 384,  # Similar to query
            [0.05] * 384,  # Very similar to query
            [-0.1] * 384,  # Opposite of query
            [0.5] * 384,   # Different magnitude
        ]
        self.document_metadata = [
            {"title": "Doc 1", "source": "source1.txt"},
            {"title": "Doc 2", "source": "source2.txt"},
            {"title": "Doc 3", "source": "source3.txt"},
            {"title": "Doc 4", "source": "source4.txt"},
        ]
        self.document_contents = [
            "This is document 1",
            "This is document 2",
            "This is document 3",
            "This is document 4",
        ]
    
    def test_dimension_reduction(self):
        """Test dimension reduction from high-dimensional to 2D."""
        # Reduce dimensions
        points_2d = VectorVisualizationHelper.reduce_dimensions(self.document_vectors)
        
        # Check result
        self.assertEqual(len(points_2d), len(self.document_vectors))
        for point in points_2d:
            self.assertEqual(len(point), 2)  # Should be 2D
            self.assertIsInstance(point[0], float)
            self.assertIsInstance(point[1], float)
    
    def test_similarity_calculation(self):
        """Test similarity calculation between vectors."""
        # Calculate similarities
        similarities = VectorVisualizationHelper.calculate_similarities(
            self.query_vector, self.document_vectors
        )
        
        # Check result
        self.assertEqual(len(similarities), len(self.document_vectors))
        
        # Verify similarity values make sense
        # Doc 1 (0.2) should be more similar than Doc 3 (-0.1)
        self.assertGreater(similarities[0], similarities[2])
        
        # Doc 2 (0.05) should be very similar to query (0.1)
        self.assertGreater(similarities[1], 0.9)  # Close to 1.0
        
        # Doc 3 (-0.1) should be opposite of query (0.1)
        self.assertLess(similarities[2], 0)  # Negative similarity
    
    def test_data_preparation(self):
        """Test preparation of visualization data."""
        # Prepare visualization data
        vis_data = VectorVisualizationHelper.prepare_visualization_data(
            self.query_vector,
            self.document_vectors,
            self.document_metadata,
            self.document_contents
        )
        
        # Check result structure
        self.assertIn("points", vis_data)
        self.assertIn("metadata", vis_data)
        self.assertIn("contents", vis_data)
        self.assertIn("similarities", vis_data)
        self.assertIn("query_point", vis_data)
        
        # Check data dimensions
        self.assertEqual(len(vis_data["points"]), len(self.document_vectors))
        self.assertEqual(len(vis_data["metadata"]), len(self.document_metadata))
        self.assertEqual(len(vis_data["contents"]), len(self.document_contents))
        self.assertEqual(len(vis_data["similarities"]), len(self.document_vectors))
        
        # Check query point is 2D
        self.assertEqual(len(vis_data["query_point"]), 2)
    
    def test_json_serialization(self):
        """Test that visualization data can be properly serialized to JSON."""
        # Prepare visualization data
        vis_data = VectorVisualizationHelper.prepare_visualization_data(
            self.query_vector,
            self.document_vectors,
            self.document_metadata,
            self.document_contents
        )
        
        # Serialize to JSON
        json_str = json.dumps(vis_data)
        parsed = json.loads(json_str)
        
        # Check structure is preserved
        self.assertIn("points", parsed)
        self.assertIn("metadata", parsed)
        self.assertIn("contents", parsed)
        self.assertIn("similarities", parsed)
        self.assertIn("query_point", parsed)
        
        # Check data dimensions
        self.assertEqual(len(parsed["points"]), len(self.document_vectors))
    
    def test_empty_vectors(self):
        """Test handling of empty vector sets."""
        # Prepare visualization data with empty vectors
        vis_data = VectorVisualizationHelper.prepare_visualization_data(
            None, [], [], []
        )
        
        # Check result structure
        self.assertEqual(vis_data["points"], [])
        self.assertEqual(vis_data["metadata"], [])
        self.assertEqual(vis_data["contents"], [])
        self.assertEqual(vis_data["similarities"], [])
        self.assertNotIn("query_point", vis_data)
    
    def test_without_query_vector(self):
        """Test visualization without a query vector."""
        # Prepare visualization data without query vector
        vis_data = VectorVisualizationHelper.prepare_visualization_data(
            None,
            self.document_vectors,
            self.document_metadata,
            self.document_contents
        )
        
        # Check result structure
        self.assertIn("points", vis_data)
        self.assertIn("metadata", vis_data)
        self.assertIn("contents", vis_data)
        self.assertIn("similarities", vis_data)
        
        # Check data dimensions
        self.assertEqual(len(vis_data["points"]), len(self.document_vectors))
        self.assertEqual(vis_data["similarities"], [])  # No similarities without query
        self.assertIsNone(vis_data.get("query_point"))  # No query point


if __name__ == "__main__":
    unittest.main()