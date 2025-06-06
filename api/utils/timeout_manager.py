"""
Timeout Manager for SAP HANA Cloud LangChain T4 GPU Integration

This module provides configurable timeout management for API endpoints,
allowing different operations to have appropriate timeout values based
on their expected execution time and resource requirements.
"""

import os
from typing import Dict, Any, Optional

# Default timeout values in seconds
DEFAULT_TIMEOUT = 30
DEFAULT_HEALTH_TIMEOUT = 10
DEFAULT_EMBEDDING_TIMEOUT = 60
DEFAULT_SEARCH_TIMEOUT = 45
DEFAULT_AUTH_TIMEOUT = 15

class TimeoutManager:
    """
    Manages timeouts for different API operations
    """
    def __init__(self):
        # Load timeouts from environment variables with defaults
        self.default_timeout = int(os.getenv("DEFAULT_TIMEOUT", str(DEFAULT_TIMEOUT)))
        self.health_timeout = int(os.getenv("HEALTH_CHECK_TIMEOUT", str(DEFAULT_HEALTH_TIMEOUT)))
        self.embedding_timeout = int(os.getenv("EMBEDDING_TIMEOUT", str(DEFAULT_EMBEDDING_TIMEOUT)))
        self.search_timeout = int(os.getenv("SEARCH_TIMEOUT", str(DEFAULT_SEARCH_TIMEOUT)))
        self.auth_timeout = int(os.getenv("AUTH_TIMEOUT", str(DEFAULT_AUTH_TIMEOUT)))
        
        # Cache of endpoint patterns to timeout values
        self._pattern_cache = {}
        
        # Initialize endpoint patterns and their corresponding timeouts
        self._initialize_patterns()
    
    def _initialize_patterns(self):
        """Initialize the mapping of endpoint patterns to timeout values"""
        self._pattern_cache = {
            # Health check endpoints
            "health": self.health_timeout,
            "proxy-health": self.health_timeout,
            
            # Authentication endpoints
            "auth": self.auth_timeout,
            "token": self.auth_timeout,
            "login": self.auth_timeout,
            
            # Embedding endpoints
            "embeddings": self.embedding_timeout,
            "embed": self.embedding_timeout,
            
            # Search endpoints
            "search": self.search_timeout,
            "mmr_search": self.search_timeout,
            "vectorstore/search": self.search_timeout,
            "similarity": self.search_timeout,
            
            # Default for any other endpoint
            "default": self.default_timeout
        }
    
    def get_timeout(self, path: str) -> int:
        """
        Get the appropriate timeout for a given API path
        
        Args:
            path (str): The API endpoint path
            
        Returns:
            int: The timeout value in seconds
        """
        # Check if path contains any of the known patterns
        for pattern, timeout in self._pattern_cache.items():
            if pattern in path:
                return timeout
        
        # Default timeout if no pattern matches
        return self.default_timeout
    
    def get_all_timeouts(self) -> Dict[str, int]:
        """
        Get all configured timeout values
        
        Returns:
            Dict[str, int]: Dictionary of timeout configurations
        """
        return {
            "default": self.default_timeout,
            "health": self.health_timeout,
            "embedding": self.embedding_timeout,
            "search": self.search_timeout,
            "auth": self.auth_timeout,
            "patterns": self._pattern_cache
        }

# Singleton instance
timeout_manager = TimeoutManager()

def get_timeout(path: str) -> int:
    """
    Get the appropriate timeout for a given API path
    
    Args:
        path (str): The API endpoint path
        
    Returns:
        int: The timeout value in seconds
    """
    return timeout_manager.get_timeout(path)

def get_all_timeouts() -> Dict[str, int]:
    """
    Get all configured timeout values
    
    Returns:
        Dict[str, int]: Dictionary of timeout configurations
    """
    return timeout_manager.get_all_timeouts()