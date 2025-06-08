"""
Mock embeddings module for langchain_hana testing.
"""

import random
from typing import List, Dict, Any, Optional, Union


class HanaInternalEmbeddings:
    """Mock HanaInternalEmbeddings class for testing."""
    
    def __init__(self, connection=None, model_name="all-MiniLM-L6-v2", 
                dimension=384, cache_embeddings=True, **kwargs):
        """Initialize the mock embeddings class."""
        self.connection = connection
        self.model_name = model_name
        self.dimension = dimension
        self.cache_embeddings = cache_embeddings
        self.extra_kwargs = kwargs
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate mock embeddings for a list of documents."""
        # Generate random unit vectors for each text
        embeddings = []
        for _ in texts:
            # Create a random vector
            vector = [random.uniform(-1, 1) for _ in range(self.dimension)]
            
            # Normalize to unit length
            magnitude = sum(x*x for x in vector) ** 0.5
            if magnitude > 0:
                vector = [x/magnitude for x in vector]
                
            embeddings.append(vector)
            
        return embeddings
        
    def embed_query(self, text: str) -> List[float]:
        """Generate a mock embedding for a query."""
        # Create a random vector
        vector = [random.uniform(-1, 1) for _ in range(self.dimension)]
        
        # Normalize to unit length
        magnitude = sum(x*x for x in vector) ** 0.5
        if magnitude > 0:
            vector = [x/magnitude for x in vector]
            
        return vector