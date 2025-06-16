#!/usr/bin/env python3
"""
Simple script to test SAP HANA Cloud connection.
"""

import os
import sys
import logging
from dotenv import load_dotenv
from hdbcli import dbapi

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def test_hana_connection(
    host: str,
    port: int,
    user: str,
    password: str,
    encrypt: bool = True,
    validate_cert: bool = True
) -> bool:
    """
    Test connection to SAP HANA Cloud.
    
    Args:
        host: HANA host
        port: HANA port
        user: HANA username
        password: HANA password
        encrypt: Whether to use encryption
        validate_cert: Whether to validate certificates
        
    Returns:
        bool: True if connection successful, False otherwise
    """
    logger.info(f"Testing connection to SAP HANA Cloud at {host}:{port}")
    try:
        # Establish connection
        connection = dbapi.connect(
            address=host,
            port=port,
            user=user,
            password=password,
            encrypt=encrypt,
            sslValidateCertificate=validate_cert
        )
        
        # Execute a simple query to verify connection
        cursor = connection.cursor()
        cursor.execute('SELECT VERSION FROM SYS.M_DATABASE')
        version = cursor.fetchone()[0]
        
        # Get schema information
        cursor.execute("SELECT SCHEMA_NAME FROM SYS.SCHEMAS WHERE SCHEMA_OWNER = CURRENT_USER")
        schemas = cursor.fetchall()
        schema_list = [row[0] for row in schemas]
        
        # Get table information from user schemas
        tables = []
        for schema in schema_list:
            cursor.execute(f"SELECT TABLE_NAME FROM SYS.TABLES WHERE SCHEMA_NAME = '{schema}' AND IS_USER_DEFINED_TYPE = 'FALSE'")
            schema_tables = cursor.fetchall()
            for table in schema_tables:
                tables.append(f"{schema}.{table[0]}")
        
        cursor.close()
        connection.close()
        
        logger.info(f"Successfully connected to SAP HANA Cloud. Version: {version}")
        logger.info(f"User schemas: {', '.join(schema_list)}")
        logger.info(f"Found {len(tables)} tables in user schemas")
        if tables:
            logger.info(f"Tables: {', '.join(tables[:10])}" + (f" and {len(tables)-10} more..." if len(tables) > 10 else ""))
        
        return True
    
    except Exception as e:
        logger.error(f"Failed to connect to SAP HANA Cloud: {str(e)}")
        return False

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    # Get HANA credentials from environment variables
    hana_host = os.environ.get('HANA_HOST')
    hana_port = os.environ.get('HANA_PORT')
    hana_user = os.environ.get('HANA_USER')
    hana_password = os.environ.get('HANA_PASSWORD')
    
    # Check if credentials are available
    if not all([hana_host, hana_port, hana_user, hana_password]):
        logger.error("Missing HANA credentials in environment variables")
        logger.error("Please set HANA_HOST, HANA_PORT, HANA_USER, and HANA_PASSWORD")
        sys.exit(1)
    
    # Test the connection
    success = test_hana_connection(
        host=hana_host,
        port=int(hana_port),
        user=hana_user,
        password=hana_password
    )
    
    if success:
        logger.info("HANA connection test passed successfully!")
        sys.exit(0)
    else:
        logger.error("HANA connection test failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()