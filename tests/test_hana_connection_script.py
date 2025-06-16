#!/usr/bin/env python3
"""
Script to test connection to SAP HANA Cloud database and verify data for visualization.
"""

import sys
import json
import logging
from typing import Dict, Any, List

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from langchain_hana.connection import create_connection, test_connection
    from hdbcli import dbapi
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.error("Please ensure that the hdbcli package is installed.")
    sys.exit(1)

def connect_to_hana() -> dbapi.Connection:
    """Connect to SAP HANA Cloud database using provided credentials."""
    try:
        connection = create_connection(
            host="d93a8739-44a8-4845-bef3-8ec724dea2ce.hana.prod-us10.hanacloud.ondemand.com",
            port=443,
            user="DBADMIN",
            password="Initial@1",
            encrypt=True,
            validate_cert=False
        )
        return connection
    except Exception as e:
        logger.error(f"Failed to connect to SAP HANA Cloud: {e}")
        sys.exit(1)

def check_database_info(connection: dbapi.Connection) -> Dict[str, Any]:
    """Check database information."""
    success, info = test_connection(connection)
    
    if not success:
        logger.error(f"Connection test failed: {info.get('error', 'Unknown error')}")
        sys.exit(1)
    
    logger.info(f"Successfully connected to SAP HANA Cloud")
    logger.info(f"Version: {info.get('version', 'Unknown')}")
    logger.info(f"Database: {info.get('database_name', 'Unknown')}")
    logger.info(f"Schema: {info.get('current_schema', 'Unknown')}")
    logger.info(f"Vector support: {'Yes' if info.get('vector_support', False) else 'No'}")
    
    return info

def check_vector_tables(connection: dbapi.Connection) -> List[str]:
    """Check for vector tables in the database."""
    try:
        cursor = connection.cursor()
        
        # Look for tables with 'VECTOR' in the name
        cursor.execute("""
            SELECT TABLE_NAME 
            FROM SYS.TABLES 
            WHERE SCHEMA_NAME = CURRENT_SCHEMA 
            AND TABLE_NAME LIKE '%VECTOR%'
            ORDER BY TABLE_NAME
        """)
        
        tables = [row[0] for row in cursor.fetchall()]
        
        logger.info(f"Found {len(tables)} vector tables:")
        for table in tables:
            logger.info(f"  - {table}")
        
        return tables
    except Exception as e:
        logger.error(f"Error checking vector tables: {e}")
        return []

def check_financial_embeddings(connection: dbapi.Connection, table_name: str = None) -> None:
    """Check for financial embeddings in the database."""
    if not table_name:
        # Find the first table with "FINANCIAL" in the name
        cursor = connection.cursor()
        cursor.execute("""
            SELECT TABLE_NAME 
            FROM SYS.TABLES 
            WHERE SCHEMA_NAME = CURRENT_SCHEMA 
            AND TABLE_NAME LIKE '%FINANCIAL%'
            ORDER BY TABLE_NAME
            LIMIT 1
        """)
        
        result = cursor.fetchone()
        if not result:
            logger.warning("No tables with 'FINANCIAL' in the name found.")
            return
        
        table_name = result[0]
    
    try:
        cursor = connection.cursor()
        
        # Check table structure
        logger.info(f"Checking structure of table {table_name}")
        cursor.execute(f"SELECT COLUMN_NAME, DATA_TYPE_NAME FROM SYS.TABLE_COLUMNS WHERE TABLE_NAME = '{table_name}'")
        columns = cursor.fetchall()
        
        for col_name, col_type in columns:
            logger.info(f"  - {col_name}: {col_type}")
        
        # Check if there are vector columns
        vector_columns = [col_name for col_name, col_type in columns if col_type == 'REAL_VECTOR']
        
        if vector_columns:
            logger.info(f"Found vector columns: {', '.join(vector_columns)}")
            
            # Check row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cursor.fetchone()[0]
            logger.info(f"Table has {row_count} rows")
            
            # Get sample data for visualization
            if row_count > 0:
                # Limit to first 5 rows for sample
                sample_limit = min(5, row_count)
                
                # Get first column as content and vector columns
                id_column = None
                content_column = None
                
                for col_name, col_type in columns:
                    if col_type in ('INTEGER', 'BIGINT') and not id_column:
                        id_column = col_name
                    elif col_type in ('VARCHAR', 'NVARCHAR', 'TEXT') and not content_column:
                        content_column = col_name
                
                if id_column and content_column and vector_columns:
                    # Build query to get sample data
                    query = f"""
                        SELECT {id_column}, {content_column}, {vector_columns[0]}
                        FROM {table_name}
                        LIMIT {sample_limit}
                    """
                    
                    try:
                        cursor.execute(query)
                        sample_data = []
                        
                        for row in cursor.fetchall():
                            # Convert to dict
                            sample_data.append({
                                "id": row[0],
                                "content": row[1],
                                "vector": row[2].tolist() if hasattr(row[2], 'tolist') else row[2]
                            })
                        
                        logger.info(f"Sample data for visualization:")
                        for item in sample_data:
                            vector_preview = str(item["vector"][:3]) + "..." if item["vector"] else "None"
                            logger.info(f"  - ID: {item['id']}, Content: {item['content'][:30]}..., Vector: {vector_preview}")
                    except Exception as e:
                        logger.error(f"Error getting sample data: {e}")
        else:
            logger.warning(f"No vector columns found in table {table_name}")
    
    except Exception as e:
        logger.error(f"Error checking financial embeddings: {e}")

def main():
    """Main function to test connection and check data."""
    logger.info("Testing connection to SAP HANA Cloud...")
    
    try:
        # Connect to database
        connection = connect_to_hana()
        
        # Check database info
        info = check_database_info(connection)
        
        # Check vector tables
        tables = check_vector_tables(connection)
        
        # Check financial embeddings
        if tables:
            # Look for financial tables
            financial_tables = [t for t in tables if 'FINANCIAL' in t or 'FINANCE' in t]
            if financial_tables:
                check_financial_embeddings(connection, financial_tables[0])
            else:
                # Just check the first vector table
                check_financial_embeddings(connection, tables[0])
        
        logger.info("Connection test completed successfully")
    
    except Exception as e:
        logger.error(f"Test failed: {e}")
        sys.exit(1)
    finally:
        # Close connection
        if 'connection' in locals() and connection:
            connection.close()
            logger.info("Connection closed")

if __name__ == "__main__":
    main()