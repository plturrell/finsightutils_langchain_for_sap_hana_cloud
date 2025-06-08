#!/usr/bin/env python3
"""
SAP HANA Cloud Table Setup Script

This script connects to SAP HANA Cloud and sets up the necessary table structure
for storing embeddings. It checks if the table exists and creates it if needed.
"""

import os
import sys
import logging
from typing import Optional

try:
    from hdbcli import dbapi
except ImportError:
    print("Error: hdbcli not installed. Install it with: pip install hdbcli")
    print("If you're using an SAP HANA client, make sure the HDBCLI package is in your PYTHONPATH")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("hana-setup")

# Database connection parameters
HOST = os.environ.get("HANA_HOST", "")
PORT = os.environ.get("HANA_PORT", "443")
USER = os.environ.get("HANA_USER", "")
PASSWORD = os.environ.get("HANA_PASSWORD", "")
TABLE_NAME = os.environ.get("DEFAULT_TABLE_NAME", "EMBEDDINGS")


def connect_to_hana() -> Optional[dbapi.Connection]:
    """Connect to SAP HANA Cloud database."""
    try:
        logger.info(f"Connecting to SAP HANA Cloud at {HOST}:{PORT}")
        connection = dbapi.connect(
            address=HOST,
            port=PORT, 
            user=USER, 
            password=PASSWORD,
            encrypt=True, 
            sslValidateCertificate=False
        )
        logger.info("Connected successfully to SAP HANA Cloud")
        return connection
    except Exception as e:
        logger.error(f"Failed to connect to SAP HANA Cloud: {str(e)}")
        return None


def check_table_exists(connection: dbapi.Connection, table_name: str) -> bool:
    """Check if a table exists in the database."""
    try:
        cursor = connection.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM TABLES WHERE TABLE_NAME = '{table_name}'")
        result = cursor.fetchone()
        exists = result[0] > 0 if result else False
        cursor.close()
        return exists
    except Exception as e:
        logger.error(f"Error checking if table exists: {str(e)}")
        return False


def create_embeddings_table(connection: dbapi.Connection, table_name: str) -> bool:
    """Create the embeddings table with the required structure."""
    try:
        cursor = connection.cursor()
        
        # Check if REAL_VECTOR data type is supported
        cursor.execute("SELECT COUNT(*) FROM DATA_TYPES WHERE TYPE_NAME = 'REAL_VECTOR'")
        has_vector_support = cursor.fetchone()[0] > 0
        
        if not has_vector_support:
            logger.error("REAL_VECTOR data type not supported by this HANA instance.")
            logger.error("Please ensure you're using SAP HANA Cloud with vector engine support.")
            return False
        
        # Create table
        create_table_sql = f"""
        CREATE TABLE {table_name} (
            ID INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            VEC_TEXT NCLOB,
            VEC_META NCLOB, 
            VEC_VECTOR REAL_VECTOR(384)
        )
        """
        
        logger.info(f"Creating table {table_name}...")
        cursor.execute(create_table_sql)
        connection.commit()
        logger.info(f"Table {table_name} created successfully")
        cursor.close()
        return True
    except Exception as e:
        logger.error(f"Failed to create table {table_name}: {str(e)}")
        return False


def create_sample_data(connection: dbapi.Connection, table_name: str) -> bool:
    """Create sample data in the embeddings table."""
    try:
        # Only proceed if the user confirms
        response = input("Do you want to create sample data? (y/n): ").strip().lower()
        if response != 'y':
            logger.info("Skipping sample data creation")
            return True
            
        from sentence_transformers import SentenceTransformer
        import json
        import struct
        
        # Load model for embedding generation
        logger.info("Loading embedding model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Sample documents
        sample_docs = [
            {
                "text": "SAP HANA Cloud provides powerful database capabilities.",
                "metadata": {"source": "sample1.txt", "category": "Database"}
            },
            {
                "text": "Vector search enables semantic search in applications.",
                "metadata": {"source": "sample2.txt", "category": "Search"}
            },
            {
                "text": "LangChain is a framework for building LLM applications.",
                "metadata": {"source": "sample3.txt", "category": "LLM"}
            },
            {
                "text": "NVIDIA GPUs accelerate AI workloads significantly.",
                "metadata": {"source": "sample4.txt", "category": "Hardware"}
            }
        ]
        
        # Generate embeddings
        texts = [doc["text"] for doc in sample_docs]
        embeddings = model.encode(texts, normalize_embeddings=True)
        
        # Insert into database
        cursor = connection.cursor()
        
        for i, (doc, embedding) in enumerate(zip(sample_docs, embeddings)):
            # Convert embedding to binary format
            vec_binary = struct.pack(f"<I{len(embedding)}f", len(embedding), *embedding)
            
            # Insert document
            cursor.execute(
                f"INSERT INTO {table_name} (VEC_TEXT, VEC_META, VEC_VECTOR) VALUES (?, ?, ?)",
                (doc["text"], json.dumps(doc["metadata"]), vec_binary)
            )
            
        connection.commit()
        logger.info(f"Added {len(sample_docs)} sample documents to {table_name}")
        cursor.close()
        return True
    except Exception as e:
        logger.error(f"Failed to create sample data: {str(e)}")
        return False


def main():
    """Main function to setup the HANA table."""
    # Check for required environment variables
    missing_vars = []
    for var_name in ["HANA_HOST", "HANA_USER", "HANA_PASSWORD"]:
        if not os.environ.get(var_name):
            missing_vars.append(var_name)
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        logger.error("Please set these variables before running this script.")
        sys.exit(1)
    
    # Connect to HANA
    connection = connect_to_hana()
    if not connection:
        logger.error("Failed to connect to HANA. Please check your credentials.")
        sys.exit(1)
    
    try:
        # Check if table exists
        if check_table_exists(connection, TABLE_NAME):
            logger.info(f"Table {TABLE_NAME} already exists.")
            
            # Ask if user wants to drop and recreate
            response = input(f"Do you want to drop and recreate the {TABLE_NAME} table? (y/n): ").strip().lower()
            if response == 'y':
                logger.info(f"Dropping table {TABLE_NAME}...")
                cursor = connection.cursor()
                cursor.execute(f"DROP TABLE {TABLE_NAME}")
                connection.commit()
                cursor.close()
                logger.info(f"Table {TABLE_NAME} dropped.")
                
                # Create the table
                if not create_embeddings_table(connection, TABLE_NAME):
                    logger.error("Table creation failed.")
                    sys.exit(1)
                    
                # Add sample data
                create_sample_data(connection, TABLE_NAME)
        else:
            logger.info(f"Table {TABLE_NAME} does not exist. Creating...")
            
            # Create the table
            if not create_embeddings_table(connection, TABLE_NAME):
                logger.error("Table creation failed.")
                sys.exit(1)
                
            # Add sample data
            create_sample_data(connection, TABLE_NAME)
        
        logger.info("Setup completed successfully.")
    finally:
        if connection:
            connection.close()
            logger.info("Database connection closed.")


if __name__ == "__main__":
    main()