#!/usr/bin/env python3
"""
Enhanced test connection to SAP HANA Cloud with detailed diagnostics.

This script tests the connection to SAP HANA Cloud and provides comprehensive
diagnostics information to help troubleshoot connection issues.
"""

import os
import sys
import json
import socket
import logging
import platform
import argparse
import traceback
import importlib
from typing import Dict, Any, Optional, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_dependencies() -> Tuple[bool, List[str]]:
    """Check if required dependencies are installed."""
    required_packages = ["hdbcli"]
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
        except ImportError:
            missing_packages.append(package)
    
    return len(missing_packages) == 0, missing_packages

def check_network_connectivity(host: str, port: int) -> bool:
    """Check if the host is reachable on the specified port."""
    logger.info(f"Checking network connectivity to {host}:{port}...")
    try:
        # Create a socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)  # 10 second timeout
        
        # Attempt to connect
        result = sock.connect_ex((host, port))
        sock.close()
        
        if result == 0:
            logger.info(f"✓ Network connectivity to {host}:{port} successful")
            return True
        else:
            logger.error(f"✗ Failed to connect to {host}:{port}. Error code: {result}")
            return False
    except Exception as e:
        logger.error(f"✗ Error checking network connectivity: {e}")
        return False

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
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                logger.info(f"Configuration loaded successfully from {config_path}")
                return config_data
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON from {config_path}: {e}")
            raise
    
    # Try environment variables
    logger.info("Loading connection configuration from environment variables")
    return {
        "address": os.environ.get("HANA_HOST"),
        "port": int(os.environ.get("HANA_PORT", "443")),
        "user": os.environ.get("HANA_USER"),
        "password": os.environ.get("HANA_PASSWORD"),
    }

def print_system_info():
    """Print system information for diagnostics."""
    logger.info("=== System Information ===")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Architecture: {platform.architecture()}")
    logger.info(f"Machine: {platform.machine()}")
    logger.info(f"Processor: {platform.processor()}")
    logger.info(f"System: {platform.system()} {platform.release()}")
    
    # Check if running in virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        logger.info(f"Virtual environment: Yes ({sys.prefix})")
    else:
        logger.info("Virtual environment: No")

def test_connection(connection_params: Dict[str, Any]) -> bool:
    """
    Test connection to SAP HANA Cloud with detailed diagnostics.
    
    Args:
        connection_params: Connection parameters
        
    Returns:
        True if connection successful, False otherwise
    """
    # First check if hdbcli is installed
    deps_ok, missing_deps = check_dependencies()
    if not deps_ok:
        logger.error(f"Missing required dependencies: {', '.join(missing_deps)}")
        logger.error("Please install missing dependencies with:")
        logger.error(f"pip install {' '.join(missing_deps)}")
        return False
    
    try:
        from hdbcli import dbapi
    except ImportError:
        logger.error("Failed to import hdbcli.dbapi despite dependency check passing.")
        logger.error("Please reinstall the hdbcli package with: pip install --upgrade hdbcli")
        return False
    
    # Print all connection parameters for debugging
    logger.info("Connection parameters:")
    for key, value in connection_params.items():
        if key != "password":  # Don't log the password
            logger.info(f"  {key}: {value}")
    
    # Check network connectivity
    if not check_network_connectivity(connection_params["address"], connection_params["port"]):
        logger.error("Network connectivity test failed. This could indicate:")
        logger.error("  1. The server is unreachable from your network")
        logger.error("  2. A firewall is blocking the connection")
        logger.error("  3. The hostname or port is incorrect")
        logger.error("  4. The server is down or not accepting connections")
        return False
    
    try:
        # Create connection with all parameters from config
        connection_kwargs = connection_params.copy()
        # Required parameters are passed explicitly, others as kwargs
        address = connection_kwargs.pop("address")
        port = connection_kwargs.pop("port")
        user = connection_kwargs.pop("user")
        password = connection_kwargs.pop("password")
        
        logger.info("Attempting to connect to SAP HANA Cloud...")
        logger.info(f"Using dbapi version: {dbapi.__version__ if hasattr(dbapi, '__version__') else 'unknown'}")
        
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
        logger.error("Error details:")
        logger.error(traceback.format_exc())
        
        # Provide troubleshooting tips based on the error
        if "timeout" in str(e).lower():
            logger.error("\nTroubleshooting timeout errors:")
            logger.error("1. Increase connectTimeout and communicationTimeout values")
            logger.error("2. Check if your network restricts connections to the HANA Cloud endpoint")
            logger.error("3. Verify if VPN or proxy is required to access the SAP HANA Cloud instance")
            logger.error("4. Check if the SAP HANA Cloud instance is running and accessible")
        
        elif "handshake" in str(e).lower() or "ssl" in str(e).lower():
            logger.error("\nTroubleshooting SSL/TLS errors:")
            logger.error("1. Try setting sslValidateCertificate=False")
            logger.error("2. Check if the endpoint requires specific TLS versions")
            logger.error("3. Verify if certificates are properly configured")
        
        elif "authentication" in str(e).lower() or "password" in str(e).lower():
            logger.error("\nTroubleshooting authentication errors:")
            logger.error("1. Verify username and password are correct")
            logger.error("2. Check if the account is locked or password expired")
            logger.error("3. Verify if the user has appropriate permissions")
        
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test connection to SAP HANA Cloud with enhanced diagnostics")
    parser.add_argument("--config", help="Path to connection configuration file")
    args = parser.parse_args()
    
    # Print system information
    print_system_info()
    
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
    
    return connection_success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)