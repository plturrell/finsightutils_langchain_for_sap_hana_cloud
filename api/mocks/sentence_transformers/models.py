"""
Mock models module for sentence_transformers testing.
"""

import random
from typing import List, Dict, Any, Union, Optional


class SentenceTransformer:
    """Mock SentenceTransformer class for testing."""
    
    def __init__(self, model_name_or_path: str, **kwargs):
        """Initialize the mock sentence transformer."""
        self.model_name_or_path = model_name_or_path
        self.dimension = 384  # Default dimension for all-MiniLM-L6-v2
        self.kwargs = kwargs
    
    def encode(self, sentences: Union[str, List[str]], 
              batch_size: int = 32, 
              show_progress_bar: bool = False, 
              convert_to_tensor: bool = False,
              normalize_embeddings: bool = True) -> List[List[float]]:
        """Generate mock embeddings for input text."""
        if isinstance(sentences, str):
            sentences = [sentences]
            
        # Generate random unit vectors for each sentence
        embeddings = []
        for _ in sentences:
            # Create a random vector
            vector = [random.uniform(-1, 1) for _ in range(self.dimension)]
            
            # Normalize to unit length
            if normalize_embeddings:
                magnitude = sum(x*x for x in vector) ** 0.5
                if magnitude > 0:
                    vector = [x/magnitude for x in vector]
                    
            embeddings.append(vector)
            
        # For a single sentence, return just the vector
        if len(sentences) == 1 and isinstance(sentences, str):
            return embeddings[0]
            
        return embeddings