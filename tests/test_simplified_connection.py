#!/usr/bin/env python3
"""
Test simplified connection to SAP HANA Cloud.

This script attempts to connect to SAP HANA Cloud using minimal parameters
to isolate the connection issue.
"""

import os
import sys
import json
import logging
import platform
from typing import Dict, Any, Optional

from hdbcli import dbapi

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_connection_config(config_path: str = "config/connection.json") -> Dict[str, Any]:
    """Load connection configuration from file."""
    if os.path.exists(config_path):
        logger.info(f"Loading connection configuration from {config_path}")
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

def test_simplified_connection(config: Dict[str, Any]) -> bool:
    """
    Test connection to SAP HANA Cloud using minimal parameters.
    
    Args:
        config: Connection configuration dictionary
        
    Returns:
        True if connection successful, False otherwise
    """
    logger.info("Testing simplified connection with minimal parameters...")
    
    # Extract minimal required parameters
    address = config["address"]
    port = config["port"]
    user = config["user"]
    password = config["password"]
    
    logger.info(f"Connecting to {address}:{port} as {user}...")
    
    try:
        # Connect with minimal parameters
        connection = dbapi.connect(
            address=address,
            port=port,
            user=user,
            password=password
        )
        
        # Test with a simple query
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM SYS.DUMMY")
        result = cursor.fetchone()
        cursor.close()
        
        # Close connection
        connection.close()
        
        logger.info("Connection test successful!")
        logger.info(f"Result from SYS.DUMMY: {result}")
        return True
        
    except Exception as e:
        logger.error(f"Simplified connection test failed: {e}")
        return False

def main():
    """Main function."""
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Platform: {platform.platform()}")
    
    # Load connection configuration
    config = load_connection_config()
    
    # Test simplified connection
    return test_simplified_connection(config)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)