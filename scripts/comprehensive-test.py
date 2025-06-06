#!/usr/bin/env python3
"""
Comprehensive test script for SAP HANA Cloud and DataSphere.
This script performs real tests with actual queries and operations.
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from hdbcli import dbapi
import requests
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

class HanaConnector:
    """Class to handle SAP HANA Cloud connections and operations."""
    
    def __init__(
        self,
        host: str,
        port: str,
        user: str,
        password: str,
        encrypt: bool = True,
        validate_cert: bool = True
    ):
        """Initialize the connector with connection parameters."""
        self.host = host
        self.port = int(port)
        self.user = user
        self.password = password
        self.encrypt = encrypt
        self.validate_cert = validate_cert
        self.connection = None
    
    def connect(self) -> bool:
        """Establish connection to SAP HANA Cloud."""
        logger.info(f"Connecting to SAP HANA Cloud at {self.host}:{self.port}")
        try:
            self.connection = dbapi.connect(
                address=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                encrypt=self.encrypt,
                sslValidateCertificate=self.validate_cert
            )
            logger.info("Successfully connected to SAP HANA Cloud")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to SAP HANA Cloud: {str(e)}")
            return False
    
    def disconnect(self):
        """Close the connection if open."""
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("Disconnected from SAP HANA Cloud")
    
    def get_version(self) -> Optional[str]:
        """Get SAP HANA version."""
        if not self.connection:
            if not self.connect():
                return None
        
        try:
            cursor = self.connection.cursor()
            cursor.execute('SELECT VERSION FROM SYS.M_DATABASE')
            version = cursor.fetchone()[0]
            cursor.close()
            return version
        except Exception as e:
            logger.error(f"Failed to get version: {str(e)}")
            return None
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get detailed system information."""
        if not self.connection:
            if not self.connect():
                return {}
        
        try:
            info = {}
            
            # Get system overview
            cursor = self.connection.cursor()
            cursor.execute('SELECT * FROM SYS.M_SYSTEM_OVERVIEW')
            rows = cursor.fetchall()
            description = cursor.description
            
            for row in rows:
                key = row[0] if row[0] else 'unknown'
                value = row[1] if len(row) > 1 else None
                info[key] = value
            
            # Get database info
            cursor.execute('SELECT * FROM SYS.M_DATABASE')
            db_info = {}
            rows = cursor.fetchall()
            column_names = [col[0] for col in cursor.description]
            
            if rows:
                db_row = rows[0]
                for i, col_name in enumerate(column_names):
                    if i < len(db_row):
                        db_info[col_name] = db_row[i]
            
            info['database'] = db_info
            
            cursor.close()
            return info
        except Exception as e:
            logger.error(f"Failed to get system info: {str(e)}")
            return {}
    
    def test_schema_access(self, schema: Optional[str] = None) -> List[Dict[str, Any]]:
        """Test access to schemas and list tables."""
        if not self.connection:
            if not self.connect():
                return []
        
        try:
            cursor = self.connection.cursor()
            
            # Get accessible schemas
            if schema:
                cursor.execute('SELECT * FROM SYS.SCHEMAS WHERE SCHEMA_NAME = ?', (schema,))
            else:
                cursor.execute('SELECT * FROM SYS.SCHEMAS')
            
            schemas = []
            rows = cursor.fetchall()
            column_names = [col[0] for col in cursor.description]
            
            for row in rows:
                schema_info = {}
                for i, col_name in enumerate(column_names):
                    if i < len(row):
                        schema_info[col_name] = row[i]
                
                # Get tables in this schema
                schema_name = schema_info.get('SCHEMA_NAME')
                if schema_name:
                    try:
                        cursor.execute('SELECT TABLE_NAME FROM SYS.TABLES WHERE SCHEMA_NAME = ? LIMIT 10', (schema_name,))
                        tables = [row[0] for row in cursor.fetchall()]
                        schema_info['TABLES'] = tables
                    except Exception as e:
                        schema_info['TABLES_ERROR'] = str(e)
                
                schemas.append(schema_info)
            
            cursor.close()
            return schemas
        except Exception as e:
            logger.error(f"Failed to test schema access: {str(e)}")
            return []
    
    def run_test_query(self, query: str) -> Tuple[bool, Any]:
        """Run a test query provided by the user."""
        if not self.connection:
            if not self.connect():
                return False, "Not connected to database"
        
        try:
            cursor = self.connection.cursor()
            start_time = time.time()
            cursor.execute(query)
            
            # Check if query returns results
            if cursor.description:
                column_names = [col[0] for col in cursor.description]
                rows = cursor.fetchall()
                result = []
                
                for row in rows:
                    row_dict = {}
                    for i, col_name in enumerate(column_names):
                        if i < len(row):
                            row_dict[col_name] = row[i]
                    result.append(row_dict)
            else:
                # For non-SELECT queries
                result = f"Query executed successfully. Rows affected: {cursor.rowcount}"
            
            execution_time = time.time() - start_time
            logger.info(f"Query executed in {execution_time:.2f} seconds")
            
            cursor.close()
            return True, result
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            return False, str(e)


class DataSphereConnector:
    """Class to handle SAP DataSphere connections and operations."""
    
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        auth_url: str,
        token_url: str,
        api_url: str
    ):
        """Initialize the connector with connection parameters."""
        self.client_id = client_id
        self.client_secret = client_secret
        self.auth_url = auth_url
        self.token_url = token_url
        self.api_url = api_url
        self.token = None
        self.session = None
    
    def authenticate(self) -> bool:
        """Authenticate and get OAuth token."""
        logger.info("Authenticating with SAP DataSphere")
        try:
            client = BackendApplicationClient(client_id=self.client_id)
            oauth = OAuth2Session(client=client)
            
            token = oauth.fetch_token(
                token_url=self.token_url,
                client_id=self.client_id,
                client_secret=self.client_secret
            )
            
            self.token = token.get('access_token')
            if self.token:
                self.session = requests.Session()
                self.session.headers.update({
                    'Authorization': f'Bearer {self.token}',
                    'Content-Type': 'application/json'
                })
                logger.info("Successfully authenticated with SAP DataSphere")
                return True
            else:
                logger.error("No access token in response")
                return False
        
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            return False
    
    def get_spaces(self) -> Tuple[bool, List[Dict[str, Any]]]:
        """Get list of spaces from DataSphere."""
        if not self.session:
            if not self.authenticate():
                return False, []
        
        try:
            # Ensure API URL ends with slash
            api_url = self.api_url
            if not api_url.endswith('/'):
                api_url += '/'
                
            url = f"{api_url}dwc/catalog/spaces"
            logger.info(f"Fetching spaces from {url}")
            
            response = self.session.get(url)
            if response.status_code == 200:
                data = response.json()
                spaces = data.get('value', [])
                logger.info(f"Found {len(spaces)} spaces")
                return True, spaces
            else:
                logger.error(f"Failed to get spaces: Status code {response.status_code}")
                logger.error(f"Response: {response.text}")
                return False, []
        except Exception as e:
            logger.error(f"Failed to get spaces: {str(e)}")
            return False, []
    
    def get_space_details(self, space_id: str) -> Tuple[bool, Dict[str, Any]]:
        """Get details of a specific space."""
        if not self.session:
            if not self.authenticate():
                return False, {}
        
        try:
            # Ensure API URL ends with slash
            api_url = self.api_url
            if not api_url.endswith('/'):
                api_url += '/'
                
            url = f"{api_url}dwc/catalog/spaces('{space_id}')"
            logger.info(f"Fetching details for space {space_id}")
            
            response = self.session.get(url)
            if response.status_code == 200:
                space = response.json()
                return True, space
            else:
                logger.error(f"Failed to get space details: Status code {response.status_code}")
                logger.error(f"Response: {response.text}")
                return False, {}
        except Exception as e:
            logger.error(f"Failed to get space details: {str(e)}")
            return False, {}
    
    def get_space_assets(self, space_id: str) -> Tuple[bool, List[Dict[str, Any]]]:
        """Get assets in a specific space."""
        if not self.session:
            if not self.authenticate():
                return False, []
        
        try:
            # Ensure API URL ends with slash
            api_url = self.api_url
            if not api_url.endswith('/'):
                api_url += '/'
                
            url = f"{api_url}dwc/catalog/spaces('{space_id}')/assets"
            logger.info(f"Fetching assets for space {space_id}")
            
            response = self.session.get(url)
            if response.status_code == 200:
                data = response.json()
                assets = data.get('value', [])
                logger.info(f"Found {len(assets)} assets")
                return True, assets
            else:
                logger.error(f"Failed to get assets: Status code {response.status_code}")
                logger.error(f"Response: {response.text}")
                return False, []
        except Exception as e:
            logger.error(f"Failed to get assets: {str(e)}")
            return False, []


def load_credentials_from_env() -> Dict[str, Dict[str, str]]:
    """Load credentials from environment variables."""
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


def save_test_results(results: Dict[str, Any], file_path: str):
    """Save test results to a file."""
    try:
        # Add timestamp
        results['timestamp'] = datetime.now().isoformat()
        
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Test results saved to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save test results: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive test for SAP HANA Cloud and DataSphere connections")
    parser.add_argument('--test-hana', action='store_true', help='Test SAP HANA Cloud connection')
    parser.add_argument('--test-datasphere', action='store_true', help='Test SAP DataSphere connection')
    parser.add_argument('--all', action='store_true', help='Test all connections')
    parser.add_argument('--schema', type=str, help='HANA schema to test access')
    parser.add_argument('--space-id', type=str, help='DataSphere space ID to test')
    parser.add_argument('--query', type=str, help='Custom HANA query to execute')
    parser.add_argument('--output', type=str, default='test_results.json', help='Output file for test results')
    
    args = parser.parse_args()
    
    if not (args.test_hana or args.test_datasphere or args.all):
        parser.print_help()
        sys.exit(1)
    
    # Load credentials
    credentials = load_credentials_from_env()
    
    # Track results
    test_results = {
        'hana': {},
        'datasphere': {}
    }
    
    # Test HANA connection if requested
    if args.test_hana or args.all:
        hana_creds = credentials['hana']
        if not all(hana_creds.values()):
            logger.error("Missing HANA credentials in environment variables")
            logger.error("Please set HANA_HOST, HANA_PORT, HANA_USER, and HANA_PASSWORD")
            test_results['hana']['status'] = 'failed'
            test_results['hana']['error'] = 'Missing credentials'
        else:
            # Initialize connector
            hana = HanaConnector(
                host=hana_creds['host'],
                port=hana_creds['port'],
                user=hana_creds['user'],
                password=hana_creds['password']
            )
            
            # Test connection
            if hana.connect():
                test_results['hana']['status'] = 'success'
                test_results['hana']['connected'] = True
                
                # Get version
                version = hana.get_version()
                if version:
                    test_results['hana']['version'] = version
                
                # Get system info
                system_info = hana.get_system_info()
                if system_info:
                    test_results['hana']['system_info'] = system_info
                
                # Test schema access
                if args.schema:
                    schemas = hana.test_schema_access(args.schema)
                else:
                    schemas = hana.test_schema_access()
                
                if schemas:
                    test_results['hana']['schemas'] = schemas
                
                # Run custom query if provided
                if args.query:
                    success, query_result = hana.run_test_query(args.query)
                    test_results['hana']['custom_query'] = {
                        'success': success,
                        'result': query_result
                    }
                
                # Disconnect
                hana.disconnect()
            else:
                test_results['hana']['status'] = 'failed'
                test_results['hana']['connected'] = False
    
    # Test DataSphere connection if requested
    if args.test_datasphere or args.all:
        ds_creds = credentials['datasphere']
        if not all(ds_creds.values()):
            logger.error("Missing DataSphere credentials in environment variables")
            logger.error("Please set DATASPHERE_CLIENT_ID, DATASPHERE_CLIENT_SECRET, DATASPHERE_AUTH_URL, DATASPHERE_TOKEN_URL, and DATASPHERE_API_URL")
            test_results['datasphere']['status'] = 'failed'
            test_results['datasphere']['error'] = 'Missing credentials'
        else:
            # Initialize connector
            datasphere = DataSphereConnector(
                client_id=ds_creds['client_id'],
                client_secret=ds_creds['client_secret'],
                auth_url=ds_creds['auth_url'],
                token_url=ds_creds['token_url'],
                api_url=ds_creds['api_url']
            )
            
            # Test authentication
            if datasphere.authenticate():
                test_results['datasphere']['status'] = 'success'
                test_results['datasphere']['authenticated'] = True
                
                # Get spaces
                success, spaces = datasphere.get_spaces()
                if success and spaces:
                    test_results['datasphere']['spaces'] = spaces
                
                # Get space details and assets if space ID provided
                if args.space_id:
                    # Get space details
                    success, space_details = datasphere.get_space_details(args.space_id)
                    if success and space_details:
                        test_results['datasphere']['space_details'] = space_details
                    
                    # Get space assets
                    success, assets = datasphere.get_space_assets(args.space_id)
                    if success and assets:
                        test_results['datasphere']['space_assets'] = assets
            else:
                test_results['datasphere']['status'] = 'failed'
                test_results['datasphere']['authenticated'] = False
    
    # Save test results
    save_test_results(test_results, args.output)
    
    # Print final results
    overall_success = all(
        test_results.get(service, {}).get('status') == 'success' 
        for service in test_results.keys() 
        if test_results.get(service, {})
    )
    
    if overall_success:
        logger.info("All connection tests passed successfully!")
        sys.exit(0)
    else:
        logger.error("One or more connection tests failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()