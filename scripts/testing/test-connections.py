#!/usr/bin/env python3
"""
Test script to verify connections to SAP HANA Cloud and SAP DataSphere.
"""

import os
import sys
import time
import logging
import argparse
import requests
from typing import Dict, Optional
from hdbcli import dbapi
from requests.auth import HTTPBasicAuth
from requests_oauthlib import OAuth2Session
from oauthlib.oauth2 import BackendApplicationClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def test_hana_connection(
    host: str,
    port: str,
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
        # Convert port to integer
        port_int = int(port)
        
        # Establish connection
        connection = dbapi.connect(
            address=host,
            port=port_int,
            user=user,
            password=password,
            encrypt=encrypt,
            sslValidateCertificate=validate_cert
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
        logger.error(f"Failed to connect to SAP HANA Cloud: {str(e)}")
        return False


def get_datasphere_oauth_token(
    token_url: str,
    client_id: str,
    client_secret: str
) -> Optional[str]:
    """
    Get OAuth token for SAP DataSphere API.
    
    Args:
        token_url: OAuth token URL
        client_id: Client ID
        client_secret: Client secret
        
    Returns:
        str: OAuth token if successful, None otherwise
    """
    logger.info("Requesting OAuth token for SAP DataSphere")
    try:
        # Create OAuth client
        client = BackendApplicationClient(client_id=client_id)
        oauth = OAuth2Session(client=client)
        
        # Get token
        token = oauth.fetch_token(
            token_url=token_url,
            client_id=client_id,
            client_secret=client_secret
        )
        
        access_token = token.get('access_token')
        if access_token:
            logger.info("Successfully obtained OAuth token")
            return access_token
        else:
            logger.error("No access token in response")
            return None
    
    except Exception as e:
        logger.error(f"Failed to get OAuth token: {str(e)}")
        return None


def test_datasphere_connection(
    api_url: str,
    auth_url: str,
    token_url: str,
    client_id: str,
    client_secret: str
) -> bool:
    """
    Test connection to SAP DataSphere.
    
    Args:
        api_url: DataSphere API URL
        auth_url: OAuth authorization URL
        token_url: OAuth token URL
        client_id: Client ID
        client_secret: Client secret
        
    Returns:
        bool: True if connection successful, False otherwise
    """
    logger.info(f"Testing connection to SAP DataSphere at {api_url}")
    try:
        # Get OAuth token
        token = get_datasphere_oauth_token(token_url, client_id, client_secret)
        if not token:
            return False
        
        # Test API endpoint
        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
        
        # Construct a URL for the spaces endpoint (common API endpoint)
        if not api_url.endswith('/'):
            api_url += '/'
            
        # Try to access the spaces API endpoint (or any available endpoint)
        response = requests.get(f"{api_url}dwc/catalog/spaces", headers=headers)
        
        if response.status_code == 200:
            spaces = response.json()
            spaces_count = len(spaces.get('value', []))
            logger.info(f"Successfully connected to SAP DataSphere API. Found {spaces_count} spaces.")
            return True
        else:
            logger.error(f"Failed to access DataSphere API: Status code {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
    
    except Exception as e:
        logger.error(f"Failed to connect to SAP DataSphere: {str(e)}")
        return False


def load_credentials_from_env() -> Dict[str, Dict[str, str]]:
    """
    Load credentials from environment variables.
    
    Returns:
        Dict: Dictionary containing credentials
    """
    credentials = {
        'hana': {
            'host': os.environ.get('HANA_HOST', ''),
            'port': os.environ.get('HANA_PORT', ''),
            'user': os.environ.get('HANA_USER', ''),
            'password': os.environ.get('HANA_PASSWORD', '')
        },
        'datasphere': {
            'client_id': os.environ.get('DATASPHERE_CLIENT_ID', ''),
            'client_secret': os.environ.get('DATASPHERE_CLIENT_SECRET', ''),
            'auth_url': os.environ.get('DATASPHERE_AUTH_URL', ''),
            'token_url': os.environ.get('DATASPHERE_TOKEN_URL', ''),
            'api_url': os.environ.get('DATASPHERE_API_URL', '')
        }
    }
    return credentials


def main():
    parser = argparse.ArgumentParser(description="Test connections to SAP HANA Cloud and SAP DataSphere")
    parser.add_argument('--test-hana', action='store_true', help='Test connection to SAP HANA Cloud')
    parser.add_argument('--test-datasphere', action='store_true', help='Test connection to SAP DataSphere')
    parser.add_argument('--all', action='store_true', help='Test all connections')
    
    args = parser.parse_args()
    
    if not (args.test_hana or args.test_datasphere or args.all):
        parser.print_help()
        sys.exit(1)
    
    # Load credentials
    credentials = load_credentials_from_env()
    
    # Track overall success
    all_successful = True
    
    # Test HANA connection if requested
    if args.test_hana or args.all:
        hana_creds = credentials['hana']
        if not all(hana_creds.values()):
            logger.error("Missing HANA credentials in environment variables")
            logger.error("Please set HANA_HOST, HANA_PORT, HANA_USER, and HANA_PASSWORD")
            all_successful = False
        else:
            hana_success = test_hana_connection(
                host=hana_creds['host'],
                port=hana_creds['port'],
                user=hana_creds['user'],
                password=hana_creds['password']
            )
            all_successful = all_successful and hana_success
    
    # Test DataSphere connection if requested
    if args.test_datasphere or args.all:
        ds_creds = credentials['datasphere']
        if not all(ds_creds.values()):
            logger.error("Missing DataSphere credentials in environment variables")
            logger.error("Please set DATASPHERE_CLIENT_ID, DATASPHERE_CLIENT_SECRET, DATASPHERE_AUTH_URL, DATASPHERE_TOKEN_URL, and DATASPHERE_API_URL")
            all_successful = False
        else:
            ds_success = test_datasphere_connection(
                api_url=ds_creds['api_url'],
                auth_url=ds_creds['auth_url'],
                token_url=ds_creds['token_url'],
                client_id=ds_creds['client_id'],
                client_secret=ds_creds['client_secret']
            )
            all_successful = all_successful and ds_success
    
    # Print final results
    if all_successful:
        logger.info("All connection tests passed successfully!")
        sys.exit(0)
    else:
        logger.error("One or more connection tests failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()