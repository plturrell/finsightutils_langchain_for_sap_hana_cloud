#!/usr/bin/env python3
"""Simple script to test connection to SAP HANA."""

import sys
import logging

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    # Import the required modules
    from hdbcli import dbapi
    logger.info("Successfully imported hdbcli.dbapi")
    
    # Try to connect using direct dbapi
    conn = dbapi.connect(
        address="d93a8739-44a8-4845-bef3-8ec724dea2ce.hana.prod-us10.hanacloud.ondemand.com",
        port=443,
        user="DBADMIN",
        password="Initial@1",
        encrypt=True,
        sslValidateCertificate=False
    )
    
    logger.info("Successfully connected to SAP HANA Cloud")
    
    # Test connection with a simple query
    cursor = conn.cursor()
    cursor.execute("SELECT VERSION FROM SYS.M_DATABASE")
    version = cursor.fetchone()[0]
    logger.info(f"SAP HANA Version: {version}")
    
    # Get current schema
    cursor.execute("SELECT CURRENT_SCHEMA FROM SYS.DUMMY")
    current_schema = cursor.fetchone()[0]
    logger.info(f"Current schema: {current_schema}")
    
    # Close cursor and connection
    cursor.close()
    conn.close()
    logger.info("Connection closed successfully")
    
except ImportError as e:
    logger.error(f"Import error: {e}")
    sys.exit(1)
except Exception as e:
    logger.error(f"Error connecting to SAP HANA: {e}")
    sys.exit(1)