#!/usr/bin/env python3
"""
Direct test script for SAP HANA Cloud using OAuth authentication.
"""

import os
import sys
import logging
import requests
import argparse
from hdbcli import dbapi

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def get_oauth_token(token_url, client_id, client_secret):
    """Get OAuth token for HANA authentication."""
    logger.info(f"Requesting OAuth token from {token_url}")
    
    try:
        # Prepare the request
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        data = {
            'grant_type': 'client_credentials',
            'client_id': client_id,
            'client_secret': client_secret
        }
        
        # Make the request
        response = requests.post(token_url, headers=headers, data=data)
        
        if response.status_code == 200:
            token_data = response.json()
            access_token = token_data.get('access_token')
            if access_token:
                logger.info("Successfully obtained OAuth token")
                return access_token
            else:
                logger.error("No access token in response")
                logger.error(f"Response: {token_data}")
                return None
        else:
            logger.error(f"Failed to get token: Status code {response.status_code}")
            logger.error(f"Response: {response.text}")
            return None
    
    except Exception as e:
        logger.error(f"Exception getting OAuth token: {str(e)}")
        return None

def test_hana_connection_with_oauth(host, port, token_url, client_id, client_secret):
    """Test connection to SAP HANA Cloud using OAuth token."""
    logger.info(f"Testing connection to SAP HANA Cloud at {host}:{port} using OAuth")
    
    # Get OAuth token
    access_token = get_oauth_token(token_url, client_id, client_secret)
    if not access_token:
        return False
    
    try:
        # Connect to HANA using OAuth token
        connection = dbapi.connect(
            address=host,
            port=int(port),
            user='',  # Not needed with OAuth
            password='',  # Not needed with OAuth
            encrypt=True,
            sslValidateCertificate=True,
            authenticationType=dbapi.AUTHENTICATION_TYPE.TOKEN,
            currentSchema='',
            jwt_token=access_token,
            sslTrustStore=''
        )
        
        # Execute a simple query to verify connection
        cursor = connection.cursor()
        cursor.execute('SELECT VERSION FROM SYS.M_DATABASE')
        version = cursor.fetchone()[0]
        cursor.close()
        connection.close()
        
        logger.info(f"Successfully connected to SAP HANA Cloud. Version: {version}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to connect to SAP HANA Cloud with OAuth: {str(e)}")
        return False

def test_hana_connection_with_basic_auth(host, port, username, password):
    """Test connection to SAP HANA Cloud using basic authentication."""
    logger.info(f"Testing connection to SAP HANA Cloud at {host}:{port} using basic auth")
    
    try:
        # Establish connection with basic auth
        connection = dbapi.connect(
            address=host,
            port=int(port),
            user=username,
            password=password,
            encrypt=True,
            sslValidateCertificate=True
        )
        
        # Execute a simple query to verify connection
        cursor = connection.cursor()
        cursor.execute('SELECT VERSION FROM SYS.M_DATABASE')
        version = cursor.fetchone()[0]
        cursor.close()
        connection.close()
        
        logger.info(f"Successfully connected to SAP HANA Cloud. Version: {version}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to connect to SAP HANA Cloud with basic auth: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Test SAP HANA Cloud connection")
    parser.add_argument('--oauth', action='store_true', help='Use OAuth authentication')
    parser.add_argument('--basic', action='store_true', help='Use basic authentication')
    
    args = parser.parse_args()
    
    if not (args.oauth or args.basic):
        parser.print_help()
        logger.info("No authentication method specified. Trying both OAuth and basic auth...")
        args.oauth = True
        args.basic = True
    
    # Load environment variables
    host = os.environ.get('HANA_HOST', '')
    port = os.environ.get('HANA_PORT', '')
    username = os.environ.get('HANA_USER', '')
    password = os.environ.get('HANA_PASSWORD', '')
    token_url = os.environ.get('DATASPHERE_TOKEN_URL', '')
    client_id = os.environ.get('DATASPHERE_CLIENT_ID', '')
    client_secret = os.environ.get('DATASPHERE_CLIENT_SECRET', '')
    
    # Check required environment variables
    if not host or not port:
        logger.error("Missing HANA_HOST or HANA_PORT environment variables")
        return 1
    
    # Track success
    success = False
    
    # Try OAuth authentication if requested
    if args.oauth:
        if not token_url or not client_id or not client_secret:
            logger.error("Missing OAuth environment variables (DATASPHERE_TOKEN_URL, DATASPHERE_CLIENT_ID, DATASPHERE_CLIENT_SECRET)")
        else:
            oauth_success = test_hana_connection_with_oauth(
                host, port, token_url, client_id, client_secret
            )
            success = success or oauth_success
    
    # Try basic authentication if requested
    if args.basic:
        if not username or not password:
            logger.error("Missing basic auth environment variables (HANA_USER, HANA_PASSWORD)")
        else:
            basic_success = test_hana_connection_with_basic_auth(
                host, port, username, password
            )
            success = success or basic_success
    
    # Report final result
    if success:
        logger.info("Connection test successful!")
        return 0
    else:
        logger.error("All connection tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())