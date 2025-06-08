"""
Mock vectorstores module for langchain_hana testing.
"""

import json
import random
import logging
from typing import List, Dict, Any, Optional, Tuple, Union, Callable, Type
from .utils import DistanceStrategy

logger = logging.getLogger(__name__)


class Document:
    """Simple document class to mimic langchain Document."""
    
    def __init__(self, page_content: str, metadata: Dict[str, Any] = None):
        """Initialize a document with content and metadata."""
        self.page_content = page_content
        self.metadata = metadata or {}


class HanaDB:
    """Mock HanaDB vectorstore implementation for testing."""
    
    def __init__(self, connection=None, embedding=None, distance_strategy=DistanceStrategy.COSINE, 
                table_name="EMBEDDINGS", text_column="VEC_TEXT", 
                metadata_column="VEC_META", vector_column="VEC_VECTOR", **kwargs):
        """Initialize the mock HanaDB vectorstore."""
        self.connection = connection
        self.embedding = embedding
        self.distance_strategy = distance_strategy
        self.table_name = table_name
        self.text_column = text_column
        self.metadata_column = metadata_column
        self.vector_column = vector_column
        self.extra_kwargs = kwargs
        
        # Mock data store for testing
        self.mock_documents = [
            Document("This is a sample document about artificial intelligence.", 
                    {"source": "sample1.txt", "category": "AI"}),
            Document("SAP HANA Cloud provides powerful database capabilities.", 
                    {"source": "sample2.txt", "category": "Database"}),
            Document("Vector databases enable efficient similarity search.", 
                    {"source": "sample3.txt", "category": "Vector Search"}),
            Document("Machine learning models can be integrated with databases.", 
                    {"source": "sample4.txt", "category": "ML"}),
            Document("Embeddings represent semantic meaning in vector space.", 
                    {"source": "sample5.txt", "category": "NLP"})
        ]
        
        logger.info(f"Mock HanaDB initialized with distance strategy: {distance_strategy}")
    
    def similarity_search(self, query: str, k: int = 4, filter: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Perform mock similarity search and return documents."""
        logger.info(f"Performing similarity search for '{query}' with k={k}")
        
        # Apply filter if provided
        docs = self._apply_filter(self.mock_documents, filter)
        
        # Limit to k documents
        results = docs[:min(k, len(docs))]
        
        return results
    
    def similarity_search_with_score(self, query: str, k: int = 4, 
                                   filter: Optional[Dict[str, Any]] = None) -> List[Tuple[Document, float]]:
        """Perform mock similarity search and return documents with scores."""
        logger.info(f"Performing similarity search with scores for '{query}' with k={k}")
        
        # Apply filter if provided
        docs = self._apply_filter(self.mock_documents, filter)
        
        # Generate mock similarity scores
        doc_scores = []
        for i, doc in enumerate(docs):
            # Generate decreasing scores from 0.95 to 0.60
            score = max(0.95 - (i * 0.07), 0.1)
            doc_scores.append((doc, score))
        
        # Limit to k documents
        results = doc_scores[:min(k, len(doc_scores))]
        
        return results
    
    def max_marginal_relevance_search(self, query: str, k: int = 4, fetch_k: int = 20,
                                    lambda_mult: float = 0.5, 
                                    filter: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Perform mock MMR search and return diverse documents."""
        logger.info(f"Performing MMR search for '{query}' with k={k}, fetch_k={fetch_k}, lambda_mult={lambda_mult}")
        
        # Apply filter if provided
        docs = self._apply_filter(self.mock_documents, filter)
        
        # Limit to k documents (in a real implementation, this would use MMR)
        # For mock purposes, just return random selection to simulate diversity
        if len(docs) > k:
            # Shuffle to simulate MMR's diversity
            random.shuffle(docs)
        
        results = docs[:min(k, len(docs))]
        return results
    
    def _apply_filter(self, docs: List[Document], filter: Optional[Dict[str, Any]]) -> List[Document]:
        """Apply metadata filter to documents."""
        if not filter:
            return docs
            
        filtered_docs = []
        for doc in docs:
            match = True
            for key, value in filter.items():
                if key not in doc.metadata or doc.metadata[key] != value:
                    match = False
                    break
            if match:
                filtered_docs.append(doc)
                
        return filtered_docs