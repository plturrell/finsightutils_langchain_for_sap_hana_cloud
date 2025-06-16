#!/usr/bin/env python3
"""
Test connection to SAP HANA Cloud using URL-based connection string.

This script tests the connection to SAP HANA Cloud using a URL-based connection
string instead of individual parameters.
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

def print_system_info():
    """Print system information for diagnostics."""
    logger.info("=== System Information ===")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"hdbcli version: {dbapi.__version__ if hasattr(dbapi, '__version__') else 'unknown'}")

def load_connection_config(config_path: str = "config/connection.json") -> Dict[str, Any]:
    """Load connection configuration from file."""
    if os.path.exists(config_path):
        logger.info(f"Loading connection configuration from {config_path}")
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

def test_connection_with_url(config: Dict[str, Any]) -> bool:
    """
    Test connection to SAP HANA Cloud using URL-based connection string.
    
    Args:
        config: Connection configuration dictionary
        
    Returns:
        True if connection successful, False otherwise
    """
    # Create URL-based connection string
    user = config["user"]
    password = config["password"]
    host = config["address"]
    port = config["port"]
    
    # Build URL with additional parameters
    url = f"hdbcli://{user}:{password}@{host}:{port}"
    
    # Add additional parameters
    params = []
    if "encrypt" in config:
        params.append(f"encrypt={'true' if config['encrypt'] else 'false'}")
    if "sslValidateCertificate" in config:
        params.append(f"sslValidateCertificate={'true' if config['sslValidateCertificate'] else 'false'}")
    if "connectTimeout" in config:
        params.append(f"connectTimeout={config['connectTimeout']}")
    if "communicationTimeout" in config:
        params.append(f"communicationTimeout={config['communicationTimeout']}")
    if "reconnect" in config:
        params.append(f"reconnect={'true' if config['reconnect'] else 'false'}")
    if "pingInterval" in config:
        params.append(f"pingInterval={config['pingInterval']}")
    if "compression" in config:
        params.append(f"compression={'true' if config['compression'] else 'false'}")
    
    # Add parameters to URL
    if params:
        url += "?" + "&".join(params)
    
    # Print URL with password masked
    masked_url = url.replace(password, "********")
    logger.info(f"Connecting using URL: {masked_url}")
    
    try:
        # Try to connect using URL
        connection = dbapi.connect(url)
        
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
        
        # Provide troubleshooting tips
        logger.error("\nAlternative connection approaches to try:")
        logger.error("1. Use direct connection parameters instead of URL-based connection")
        logger.error("2. Try connecting using SAP HANA Studio or another client to verify credentials")
        logger.error("3. Check SAP HANA Cloud instance status in SAP BTP Cockpit")
        logger.error("4. Verify if instance requires specific configuration or certificates")
        
        return False

def main():
    """Main function."""
    print_system_info()
    
    # Load connection configuration
    config = load_connection_config()
    
    # Test connection using URL
    return test_connection_with_url(config)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)