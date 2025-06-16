#\!/usr/bin/env python
"""
Test connection to SAP HANA Cloud.

This script tests the connection to SAP HANA Cloud using the provided credentials.
"""

import os
import json
import logging
import argparse
from typing import Dict, Any, Optional

from hdbcli import dbapi

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_connection_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load connection configuration from file or environment variables."""
    # Try to load from file
    if config_path is None:
        possible_paths = [
            "connection.json",
            "config/connection.json",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break
    
    if config_path and os.path.exists(config_path):
        logger.info(f"Loading connection configuration from {config_path}")
        with open(config_path, 'r') as f:
            return json.load(f)
    
    # Try environment variables
    logger.info("Loading connection configuration from environment variables")
    return {
        "address": os.environ.get("HANA_HOST"),
        "port": int(os.environ.get("HANA_PORT", "443")),
        "user": os.environ.get("HANA_USER"),
        "password": os.environ.get("HANA_PASSWORD"),
    }


def test_connection(connection_params: Dict[str, Any]) -> bool:
    """
    Test connection to SAP HANA Cloud.
    
    Args:
        connection_params: Connection parameters
        
    Returns:
        True if connection successful, False otherwise
    """
    logger.info(f"Testing connection to {connection_params['address']}:{connection_params['port']} as {connection_params['user']}")
    
    try:
        # Print all connection parameters for debugging
        logger.info("Connection parameters:")
        for key, value in connection_params.items():
            if key != "password":  # Don't log the password
                logger.info(f"  {key}: {value}")

        # Create connection with all parameters from config
        connection_kwargs = connection_params.copy()
        # Required parameters are passed explicitly, others as kwargs
        address = connection_kwargs.pop("address")
        port = connection_kwargs.pop("port")
        user = connection_kwargs.pop("user")
        password = connection_kwargs.pop("password")
        
        logger.info("Attempting to connect to SAP HANA Cloud...")
        connection = dbapi.connect(
            address=address,
            port=port,
            user=user,
            password=password,
            **connection_kwargs
        )
        
        # Test connection with a simple query
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
        logger.error(f"Connection test failed: {e}")
        return False


def test_schema_setup(connection_params: Dict[str, Any]) -> bool:
    """
    Test schema setup for LangChain integration.
    
    Args:
        connection_params: Connection parameters
        
    Returns:
        True if schema setup successful, False otherwise
    """
    logger.info("Testing schema setup for LangChain integration")
    
    try:
        # Create connection
        connection = dbapi.connect(
            address=connection_params["address"],
            port=connection_params["port"],
            user=connection_params["user"],
            password=connection_params["password"],
            encrypt=connection_params.get("encrypt", True),
            sslValidateCertificate=connection_params.get("sslValidateCertificate", False)
        )
        
        # Create a test table
        cursor = connection.cursor()
        
        # Check if we can create and drop tables
        test_table_name = "LC_CONNECTION_TEST"
        
        try:
            # Try to drop the table if it exists
            cursor.execute(f'DROP TABLE "{test_table_name}"')
            connection.commit()
        except:
            # Ignore errors here
            connection.rollback()
        
        # Create test table
        cursor.execute(f"""
        CREATE TABLE "{test_table_name}" (
            "ID" INTEGER,
            "TEXT" NVARCHAR(100),
            "VEC_VECTOR" REAL_VECTOR
        )
        """)
        connection.commit()
        
        # Insert test data
        cursor.execute(f"""
        INSERT INTO "{test_table_name}" VALUES (
            1, 
            'Test Text', 
            TO_REAL_VECTOR('[0.1, 0.2, 0.3]')
        )
        """)
        connection.commit()
        
        # Query test data
        cursor.execute(f'SELECT * FROM "{test_table_name}"')
        result = cursor.fetchone()
        
        # Drop test table
        cursor.execute(f'DROP TABLE "{test_table_name}"')
        connection.commit()
        
        # Close connection
        cursor.close()
        connection.close()
        
        logger.info("Schema setup test successful!")
        logger.info(f"Test data: {result}")
        return True
        
    except Exception as e:
        logger.error(f"Schema setup test failed: {e}")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test connection to SAP HANA Cloud")
    parser.add_argument("--config", help="Path to connection configuration file")
    args = parser.parse_args()
    
    # Load connection configuration
    connection_params = load_connection_config(args.config)
    
    # Validate required parameters
    required_params = ["address", "port", "user", "password"]
    missing_params = [param for param in required_params if not connection_params.get(param)]
    if missing_params:
        logger.error(f"Missing required connection parameters: {', '.join(missing_params)}")
        logger.error("Please check your connection.json file or environment variables.")
        return False
    
    # Test connection
    connection_success = test_connection(connection_params)
    
    if not connection_success:
        return False
    
    # Test schema setup
    schema_success = test_schema_setup(connection_params)
    
    return connection_success and schema_success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)