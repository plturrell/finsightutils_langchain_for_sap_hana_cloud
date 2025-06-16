#!/usr/bin/env python3
"""Script to check for vector tables in SAP HANA Cloud."""

import sys
import logging
import json

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    # Import the required modules
    from hdbcli import dbapi
    
    # Connect to SAP HANA
    conn = dbapi.connect(
        address="d93a8739-44a8-4845-bef3-8ec724dea2ce.hana.prod-us10.hanacloud.ondemand.com",
        port=443,
        user="DBADMIN",
        password="Initial@1",
        encrypt=True,
        sslValidateCertificate=False
    )
    
    logger.info("Successfully connected to SAP HANA Cloud")
    
    # Check for vector support
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT COUNT(*) FROM SYS.DATA_TYPES WHERE TYPE_NAME = 'REAL_VECTOR'")
        vector_support = cursor.fetchone()[0] > 0
        logger.info(f"Vector support: {'Yes' if vector_support else 'No'}")
    except Exception as e:
        logger.warning(f"Could not check for vector support: {e}")
        vector_support = False
    
    # Get all tables in the current schema
    cursor.execute("""
        SELECT TABLE_NAME 
        FROM SYS.TABLES 
        WHERE SCHEMA_NAME = CURRENT_SCHEMA 
        ORDER BY TABLE_NAME
    """)
    
    tables = [row[0] for row in cursor.fetchall()]
    logger.info(f"Found {len(tables)} tables in current schema")
    
    # Check for tables with 'VECTOR', 'EMBED', or 'FINANCIAL' in the name
    vector_tables = [t for t in tables if 'VECTOR' in t or 'EMBED' in t or 'FINANCIAL' in t]
    
    if vector_tables:
        logger.info(f"Found {len(vector_tables)} potential vector/embedding tables:")
        for table in vector_tables:
            logger.info(f"  - {table}")
        
        # Check for REAL_VECTOR columns in these tables
        for table in vector_tables:
            cursor.execute(f"""
                SELECT COLUMN_NAME, DATA_TYPE_NAME 
                FROM SYS.TABLE_COLUMNS 
                WHERE TABLE_NAME = '{table}' 
                ORDER BY COLUMN_NAME
            """)
            
            columns = cursor.fetchall()
            vector_columns = [col[0] for col in columns if 'VECTOR' in col[1]]
            
            if vector_columns:
                logger.info(f"Table {table} has vector columns: {', '.join(vector_columns)}")
                
                # Check row count
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                row_count = cursor.fetchone()[0]
                logger.info(f"  - Row count: {row_count}")
                
                # Get a sample row if there's data
                if row_count > 0:
                    # Get all column names
                    all_columns = [col[0] for col in columns]
                    
                    # Find potential content column (text-like column)
                    text_columns = [col[0] for col in columns if 'VARCHAR' in col[1] or 'CHAR' in col[1] or 'TEXT' in col[1]]
                    content_column = text_columns[0] if text_columns else all_columns[0]
                    
                    # Get sample row
                    try:
                        sample_query = f"""
                            SELECT {content_column}, {vector_columns[0]}
                            FROM {table}
                            LIMIT 1
                        """
                        cursor.execute(sample_query)
                        sample = cursor.fetchone()
                        
                        if sample:
                            content_sample = sample[0]
                            if len(content_sample) > 50:
                                content_sample = content_sample[:50] + "..."
                            
                            vector_sample = sample[1]
                            vector_preview = None
                            
                            # Try to convert vector to list for display
                            try:
                                if hasattr(vector_sample, 'tolist'):
                                    vector_list = vector_sample.tolist()
                                    vector_preview = str(vector_list[:3]) + "..." if len(vector_list) > 3 else str(vector_list)
                                else:
                                    vector_preview = "Vector data available but can't display preview"
                            except Exception as e:
                                vector_preview = f"Error getting vector preview: {e}"
                            
                            logger.info(f"  - Sample content: {content_sample}")
                            logger.info(f"  - Sample vector: {vector_preview}")
                            
                            # Get vector dimensions if possible
                            try:
                                if hasattr(vector_sample, 'tolist'):
                                    vector_list = vector_sample.tolist()
                                    logger.info(f"  - Vector dimensions: {len(vector_list)}")
                            except Exception:
                                pass
                    except Exception as e:
                        logger.error(f"Error getting sample from {table}: {e}")
            else:
                logger.info(f"Table {table} has no vector columns")
    else:
        logger.warning("No potential vector/embedding tables found")
    
    # Close connection
    cursor.close()
    conn.close()
    logger.info("Connection closed")

except ImportError as e:
    logger.error(f"Import error: {e}")
    sys.exit(1)
except Exception as e:
    logger.error(f"Error: {e}")
    sys.exit(1)