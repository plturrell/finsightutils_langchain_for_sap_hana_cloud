#!/usr/bin/env python3
"""
Script to set up the necessary SAP HANA tables for vector embeddings.
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

def connect_to_hana(
    host: str,
    port: int,
    user: str,
    password: str,
    encrypt: bool = True,
    validate_cert: bool = True
):
    """
    Connect to SAP HANA Cloud.
    
    Args:
        host: HANA host
        port: HANA port
        user: HANA username
        password: HANA password
        encrypt: Whether to use encryption
        validate_cert: Whether to validate certificates
        
    Returns:
        Connection object if successful
    """
    logger.info(f"Connecting to SAP HANA Cloud at {host}:{port}")
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
        return connection
    except Exception as e:
        logger.error(f"Failed to connect to SAP HANA Cloud: {str(e)}")
        raise

def create_embeddings_table(
    connection,
    schema_name: str,
    table_name: str,
    embedding_size: int = 384
):
    """
    Create a table for storing embeddings.
    
    Args:
        connection: HANA connection object
        schema_name: Schema name
        table_name: Table name
        embedding_size: Size of embedding vectors
    """
    # Create schema if it doesn't exist
    try:
        cursor = connection.cursor()
        cursor.execute(f"CREATE SCHEMA {schema_name}")
        logger.info(f"Created schema {schema_name}")
    except Exception as e:
        if "exists" in str(e).lower():
            logger.info(f"Schema {schema_name} already exists")
        else:
            logger.error(f"Error creating schema {schema_name}: {str(e)}")
    
    # Create embeddings table with the HANA vector engine
    full_table_name = f"{schema_name}.{table_name}"
    try:
        # Drop table if it exists
        cursor.execute(f"DROP TABLE {full_table_name}")
        logger.info(f"Dropped existing table {full_table_name}")
    except Exception as e:
        if "not found" not in str(e).lower():
            logger.warning(f"Note: {str(e)}")
    
    try:
        # Create table with document ID, content, metadata, and embedding vector
        create_table_sql = f"""
        CREATE TABLE {full_table_name} (
            ID VARCHAR(100) PRIMARY KEY,
            DOCUMENT_CONTENT NCLOB,
            METADATA NCLOB,
            EMBEDDING REAL ARRAY({embedding_size})
        )
        """
        cursor.execute(create_table_sql)
        logger.info(f"Created table {full_table_name}")
        
        # Create vector index for similarity search
        create_index_sql = f"""
        CREATE HNSW INDEX IDX_EMBEDDING_HNSW ON {full_table_name}(EMBEDDING)
        PARAMETERS ('M'='16', 'ef_construction'='64', 'ef'='40', 'distance_measure'='cosine')
        """
        cursor.execute(create_index_sql)
        logger.info(f"Created HNSW vector index on {full_table_name}")
        
        return True
    except Exception as e:
        logger.error(f"Error creating embeddings table: {str(e)}")
        return False
    finally:
        cursor.close()

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    # Get HANA credentials from environment variables
    hana_host = os.environ.get('HANA_HOST')
    hana_port = os.environ.get('HANA_PORT')
    hana_user = os.environ.get('HANA_USER')
    hana_password = os.environ.get('HANA_PASSWORD')
    default_table = os.environ.get('DEFAULT_TABLE_NAME', 'EMBEDDINGS')
    
    # Check if credentials are available
    if not all([hana_host, hana_port, hana_user, hana_password]):
        logger.error("Missing HANA credentials in environment variables")
        logger.error("Please set HANA_HOST, HANA_PORT, HANA_USER, and HANA_PASSWORD")
        sys.exit(1)
    
    # Ask for schema and table names
    schema_name = input(f"Enter schema name [{hana_user}]: ").strip() or hana_user
    table_name = input(f"Enter table name [{default_table}]: ").strip() or default_table
    embedding_size = int(input("Enter embedding size [384]: ").strip() or "384")
    
    try:
        # Connect to HANA
        connection = connect_to_hana(
            host=hana_host,
            port=int(hana_port),
            user=hana_user,
            password=hana_password
        )
        
        # Create embeddings table
        success = create_embeddings_table(
            connection=connection,
            schema_name=schema_name,
            table_name=table_name,
            embedding_size=embedding_size
        )
        
        if success:
            logger.info(f"Successfully created embeddings table {schema_name}.{table_name}")
            
            # Update .env file with the table name
            env_path = '.env'
            if os.path.exists(env_path):
                with open(env_path, 'r') as f:
                    env_lines = f.readlines()
                
                updated = False
                with open(env_path, 'w') as f:
                    for line in env_lines:
                        if line.startswith('DEFAULT_TABLE_NAME='):
                            f.write(f'DEFAULT_TABLE_NAME={table_name}\n')
                            updated = True
                        else:
                            f.write(line)
                    
                    if not updated:
                        f.write(f'DEFAULT_TABLE_NAME={table_name}\n')
                
                logger.info(f"Updated .env file with DEFAULT_TABLE_NAME={table_name}")
            
            connection.close()
            sys.exit(0)
        else:
            logger.error("Failed to create embeddings table")
            connection.close()
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()