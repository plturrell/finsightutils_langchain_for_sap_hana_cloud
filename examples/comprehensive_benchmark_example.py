#!/usr/bin/env python
"""
Comprehensive benchmark example for SAP HANA data representation methods.

This example demonstrates a complete workflow for benchmarking different
data representation methods (relational, vector, ontology, hybrid) using
question-answer pairs to measure factual accuracy.
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, List, Any, Optional

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

# Add parent directory to path to import local modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_hana.reasoning.factuality import FactualityEvaluator
from langchain_hana.reasoning.benchmark_system import (
    BenchmarkQuestion,
    RepresentationType,
    QuestionType,
    HanaStorageManager,
    QuestionGenerator,
)
from langchain_hana.reasoning.benchmark_system_part2 import (
    BenchmarkRunner,
    RelationalRepresentationHandler,
    VectorRepresentationHandler,
    OntologyRepresentationHandler,
    HybridRepresentationHandler,
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_schema_info_from_hana(
    host: str,
    port: int,
    user: str,
    password: str,
    schema_name: str,
) -> str:
    """
    Extract schema information from SAP HANA.
    
    Args:
        host: HANA host
        port: HANA port
        user: HANA user
        password: HANA password
        schema_name: Name of the schema to extract
        
    Returns:
        Formatted schema information
    """
    try:
        from hdbcli import dbapi
        
        # Connect to HANA
        connection = dbapi.connect(
            address=host,
            port=port,
            user=user,
            password=password,
        )
        
        cursor = connection.cursor()
        
        # Get tables
        cursor.execute(f"""
        SELECT TABLE_NAME, COMMENTS
        FROM SYS.TABLES
        WHERE SCHEMA_NAME = '{schema_name}'
        ORDER BY TABLE_NAME
        """)
        
        tables = cursor.fetchall()
        
        schema_info = f"Schema: {schema_name}\n\n"
        schema_info += "Tables:\n"
        
        for table_name, comments in tables:
            schema_info += f"- {table_name}"
            if comments:
                schema_info += f" ({comments})"
            schema_info += "\n"
            
            # Get columns for this table
            cursor.execute(f"""
            SELECT COLUMN_NAME, DATA_TYPE_NAME, LENGTH, SCALE, IS_NULLABLE, COMMENTS, POSITION
            FROM SYS.TABLE_COLUMNS
            WHERE SCHEMA_NAME = '{schema_name}' AND TABLE_NAME = '{table_name}'
            ORDER BY POSITION
            """)
            
            columns = cursor.fetchall()
            
            schema_info += "  Columns:\n"
            for col_name, data_type, length, scale, is_nullable, col_comments, position in columns:
                nullable = "NULL" if is_nullable == "TRUE" else "NOT NULL"
                schema_info += f"  - {col_name}: {data_type}"
                
                if data_type in ["VARCHAR", "NVARCHAR", "CHAR", "NCHAR"]:
                    schema_info += f"({length})"
                elif data_type in ["DECIMAL"]:
                    schema_info += f"({length},{scale})"
                
                schema_info += f" {nullable}"
                
                if col_comments:
                    schema_info += f" ({col_comments})"
                
                schema_info += "\n"
            
            # Get primary key
            cursor.execute(f"""
            SELECT COLUMN_NAME
            FROM SYS.CONSTRAINTS AS C
            JOIN SYS.CONSTRAINT_COLUMNS AS CC ON C.SCHEMA_NAME = CC.SCHEMA_NAME AND C.TABLE_NAME = CC.TABLE_NAME AND C.CONSTRAINT_NAME = CC.CONSTRAINT_NAME
            WHERE C.SCHEMA_NAME = '{schema_name}' AND C.TABLE_NAME = '{table_name}' AND C.IS_PRIMARY_KEY = 'TRUE'
            ORDER BY CC.POSITION
            """)
            
            pk_columns = cursor.fetchall()
            
            if pk_columns:
                pk_cols = [col[0] for col in pk_columns]
                schema_info += f"  Primary Key: {', '.join(pk_cols)}\n"
            
            # Get foreign keys
            cursor.execute(f"""
            SELECT C.CONSTRAINT_NAME, CC.COLUMN_NAME, C.REFERENCED_SCHEMA_NAME, C.REFERENCED_TABLE_NAME, C.REFERENCED_COLUMN_NAME
            FROM SYS.CONSTRAINTS AS C
            JOIN SYS.CONSTRAINT_COLUMNS AS CC ON C.SCHEMA_NAME = CC.SCHEMA_NAME AND C.TABLE_NAME = CC.TABLE_NAME AND C.CONSTRAINT_NAME = CC.CONSTRAINT_NAME
            WHERE C.SCHEMA_NAME = '{schema_name}' AND C.TABLE_NAME = '{table_name}' AND C.IS_FOREIGN_KEY = 'TRUE'
            ORDER BY C.CONSTRAINT_NAME, CC.POSITION
            """)
            
            fk_constraints = cursor.fetchall()
            
            if fk_constraints:
                schema_info += "  Foreign Keys:\n"
                current_constraint = None
                fk_columns = []
                
                for constraint_name, column_name, ref_schema, ref_table, ref_column in fk_constraints:
                    if constraint_name != current_constraint:
                        if current_constraint:
                            schema_info += f"    - {', '.join(fk_columns)} -> {ref_schema}.{ref_table}.{ref_column}\n"
                            fk_columns = []
                        current_constraint = constraint_name
                    
                    fk_columns.append(column_name)
                
                if fk_columns:
                    schema_info += f"    - {', '.join(fk_columns)} -> {ref_schema}.{ref_table}.{ref_column}\n"
            
            schema_info += "\n"
        
        # Get views
        cursor.execute(f"""
        SELECT VIEW_NAME, COMMENTS
        FROM SYS.VIEWS
        WHERE SCHEMA_NAME = '{schema_name}'
        ORDER BY VIEW_NAME
        """)
        
        views = cursor.fetchall()
        
        if views:
            schema_info += "Views:\n"
            
            for view_name, comments in views:
                schema_info += f"- {view_name}"
                if comments:
                    schema_info += f" ({comments})"
                schema_info += "\n"
                
                # Get columns for this view
                cursor.execute(f"""
                SELECT COLUMN_NAME, DATA_TYPE_NAME, LENGTH, SCALE, COMMENTS, POSITION
                FROM SYS.VIEW_COLUMNS
                WHERE SCHEMA_NAME = '{schema_name}' AND VIEW_NAME = '{view_name}'
                ORDER BY POSITION
                """)
                
                columns = cursor.fetchall()
                
                schema_info += "  Columns:\n"
                for col_name, data_type, length, scale, col_comments, position in columns:
                    schema_info += f"  - {col_name}: {data_type}"
                    
                    if data_type in ["VARCHAR", "NVARCHAR", "CHAR", "NCHAR"]:
                        schema_info += f"({length})"
                    elif data_type in ["DECIMAL"]:
                        schema_info += f"({length},{scale})"
                    
                    if col_comments:
                        schema_info += f" ({col_comments})"
                    
                    schema_info += "\n"
                
                schema_info += "\n"
        
        cursor.close()
        
        return schema_info
    
    except Exception as e:
        logger.error(f"Error extracting schema info: {str(e)}")
        raise


def extract_data_info_from_hana(
    host: str,
    port: int,
    user: str,
    password: str,
    schema_name: str,
    max_rows_per_table: int = 5,
) -> str:
    """
    Extract data information from SAP HANA.
    
    Args:
        host: HANA host
        port: HANA port
        user: HANA user
        password: HANA password
        schema_name: Name of the schema to extract
        max_rows_per_table: Maximum number of rows to extract per table
        
    Returns:
        Formatted data information
    """
    try:
        from hdbcli import dbapi
        
        # Connect to HANA
        connection = dbapi.connect(
            address=host,
            port=port,
            user=user,
            password=password,
        )
        
        cursor = connection.cursor()
        
        # Get tables
        cursor.execute(f"""
        SELECT TABLE_NAME
        FROM SYS.TABLES
        WHERE SCHEMA_NAME = '{schema_name}'
        ORDER BY TABLE_NAME
        """)
        
        tables = cursor.fetchall()
        
        data_info = f"Schema: {schema_name}\n\n"
        data_info += "Table Data:\n\n"
        
        for table_name, in tables:
            data_info += f"Table: {table_name}\n"
            
            # Get table statistics
            try:
                cursor.execute(f"""
                SELECT COUNT(*) FROM "{schema_name}"."{table_name}"
                """)
                count = cursor.fetchone()[0]
                data_info += f"- Row count: {count}\n"
            except Exception as e:
                logger.warning(f"Error getting row count for table {table_name}: {str(e)}")
                data_info += f"- Row count: Error\n"
            
            # Get column statistics
            cursor.execute(f"""
            SELECT COLUMN_NAME, DATA_TYPE_NAME
            FROM SYS.TABLE_COLUMNS
            WHERE SCHEMA_NAME = '{schema_name}' AND TABLE_NAME = '{table_name}'
            ORDER BY POSITION
            """)
            
            columns = cursor.fetchall()
            
            for col_name, data_type in columns:
                data_info += f"- Column: {col_name} ({data_type})\n"
                
                # Skip CLOB, BLOB, NCLOB, etc.
                if data_type in ["CLOB", "NCLOB", "BLOB", "TEXT"]:
                    data_info += f"  - Statistics: Skipped for {data_type} data type\n"
                    continue
                
                try:
                    # Min, max, avg for numeric columns
                    if data_type in ["INTEGER", "BIGINT", "SMALLINT", "TINYINT", "DECIMAL", "DOUBLE", "REAL"]:
                        cursor.execute(f"""
                        SELECT MIN("{col_name}"), MAX("{col_name}"), AVG("{col_name}")
                        FROM "{schema_name}"."{table_name}"
                        WHERE "{col_name}" IS NOT NULL
                        """)
                        min_val, max_val, avg_val = cursor.fetchone()
                        
                        data_info += f"  - Min: {min_val}\n"
                        data_info += f"  - Max: {max_val}\n"
                        data_info += f"  - Avg: {avg_val}\n"
                    
                    # Min, max for date columns
                    elif data_type in ["DATE", "TIME", "TIMESTAMP"]:
                        cursor.execute(f"""
                        SELECT MIN("{col_name}"), MAX("{col_name}")
                        FROM "{schema_name}"."{table_name}"
                        WHERE "{col_name}" IS NOT NULL
                        """)
                        min_val, max_val = cursor.fetchone()
                        
                        data_info += f"  - Min: {min_val}\n"
                        data_info += f"  - Max: {max_val}\n"
                    
                    # Distinct values count for all columns
                    cursor.execute(f"""
                    SELECT COUNT(DISTINCT "{col_name}")
                    FROM "{schema_name}"."{table_name}"
                    """)
                    distinct_count = cursor.fetchone()[0]
                    
                    data_info += f"  - Distinct values: {distinct_count}\n"
                    
                    # NULL count
                    cursor.execute(f"""
                    SELECT COUNT(*)
                    FROM "{schema_name}"."{table_name}"
                    WHERE "{col_name}" IS NULL
                    """)
                    null_count = cursor.fetchone()[0]
                    
                    data_info += f"  - NULL count: {null_count}\n"
                    
                    # Sample values
                    if data_type not in ["CLOB", "NCLOB", "BLOB", "TEXT"]:
                        cursor.execute(f"""
                        SELECT DISTINCT "{col_name}"
                        FROM "{schema_name}"."{table_name}"
                        WHERE "{col_name}" IS NOT NULL
                        ORDER BY "{col_name}"
                        LIMIT 5
                        """)
                        
                        sample_values = cursor.fetchall()
                        sample_str = ", ".join(str(val[0]) for val in sample_values)
                        
                        data_info += f"  - Sample values: {sample_str}\n"
                
                except Exception as e:
                    logger.warning(f"Error getting statistics for column {col_name}: {str(e)}")
                    data_info += f"  - Statistics: Error\n"
            
            # Sample rows
            try:
                cursor.execute(f"""
                SELECT * FROM "{schema_name}"."{table_name}"
                LIMIT {max_rows_per_table}
                """)
                
                rows = cursor.fetchall()
                
                if rows:
                    data_info += f"- Sample rows ({min(len(rows), max_rows_per_table)}):\n"
                    
                    # Get column names
                    col_names = [col[0] for col in cursor.description]
                    
                    for row in rows:
                        row_str = ", ".join(f"{col}={val}" for col, val in zip(col_names, row))
                        data_info += f"  - {row_str}\n"
            
            except Exception as e:
                logger.warning(f"Error getting sample rows for table {table_name}: {str(e)}")
                data_info += f"- Sample rows: Error\n"
            
            data_info += "\n"
        
        cursor.close()
        
        return data_info
    
    except Exception as e:
        logger.error(f"Error extracting data info: {str(e)}")
        raise


def extract_text_data_for_vectorization(
    connection,
    schema_name: str,
    max_docs: int = 1000,
) -> List[Document]:
    """
    Extract text data for vectorization.
    
    Args:
        connection: Database connection
        schema_name: Schema name
        max_docs: Maximum number of documents to extract
        
    Returns:
        List of documents for vectorization
    """
    documents = []
    
    try:
        cursor = connection.cursor()
        
        # Get tables
        cursor.execute(f"""
        SELECT TABLE_NAME
        FROM SYS.TABLES
        WHERE SCHEMA_NAME = '{schema_name}'
        ORDER BY TABLE_NAME
        """)
        
        tables = cursor.fetchall()
        doc_count = 0
        
        for table_name, in tables:
            if doc_count >= max_docs:
                break
            
            # Get column information
            cursor.execute(f"""
            SELECT COLUMN_NAME, DATA_TYPE_NAME
            FROM SYS.TABLE_COLUMNS
            WHERE SCHEMA_NAME = '{schema_name}' AND TABLE_NAME = '{table_name}'
            AND DATA_TYPE_NAME IN ('VARCHAR', 'NVARCHAR', 'CHAR', 'NCHAR', 'TEXT')
            ORDER BY POSITION
            """)
            
            text_columns = cursor.fetchall()
            
            if not text_columns:
                continue
            
            # Extract text data
            for col_name, data_type in text_columns:
                if doc_count >= max_docs:
                    break
                
                # Get text data
                cursor.execute(f"""
                SELECT "{col_name}", *
                FROM "{schema_name}"."{table_name}"
                WHERE "{col_name}" IS NOT NULL AND LENGTH("{col_name}") > 10
                LIMIT {max_docs - doc_count}
                """)
                
                rows = cursor.fetchall()
                
                if not rows:
                    continue
                
                # Get column names
                col_names = [col[0] for col in cursor.description]
                
                # Create documents
                for row in rows:
                    text = str(row[0])
                    
                    # Create metadata
                    metadata = {
                        "schema": schema_name,
                        "table": table_name,
                        "column": col_name,
                        "source": f"{schema_name}.{table_name}.{col_name}",
                    }
                    
                    # Add other columns as metadata
                    for i, val in enumerate(row[1:], 1):
                        if i < len(col_names):
                            metadata[col_names[i]] = str(val)
                    
                    # Create document
                    document = Document(
                        page_content=text,
                        metadata=metadata,
                    )
                    
                    documents.append(document)
                    doc_count += 1
        
        cursor.close()
        
        return documents
    
    except Exception as e:
        logger.error(f"Error extracting text data: {str(e)}")
        return []


def create_mock_vector_store():
    """
    Create a mock vector store for the example.
    
    Returns:
        FAISS vector store
    """
    # Sample documents
    documents = [
        Document(
            page_content="Customer ID 1 is Aaron Smith with email aaron.smith@example.com and phone +1-202-555-0123",
            metadata={"source": "SALES.CUSTOMERS", "record_id": 1},
        ),
        Document(
            page_content="Customer ID 2 is Alice Johnson with email alice.j@example.com and phone +1-303-555-0187",
            metadata={"source": "SALES.CUSTOMERS", "record_id": 2},
        ),
        Document(
            page_content="Product ID 1 is an Ergonomic Chair priced at $199.99 in the Office category",
            metadata={"source": "SALES.PRODUCTS", "record_id": 1},
        ),
        Document(
            page_content="Product ID 2 is a Gaming Laptop priced at $1299.99 in the Electronics category",
            metadata={"source": "SALES.PRODUCTS", "record_id": 2},
        ),
        Document(
            page_content="Order ID 1 was placed by Customer ID 42 on 2020-01-01 for a total of $249.98",
            metadata={"source": "SALES.ORDERS", "record_id": 1},
        ),
        Document(
            page_content="Order ID 2 was placed by Customer ID 87 on 2020-01-01 for a total of $1349.98",
            metadata={"source": "SALES.ORDERS", "record_id": 2},
        ),
        Document(
            page_content="Customer Aaron Smith (ID 1) has placed 7 orders, mostly for Electronics products",
            metadata={"source": "SALES.CUSTOMER_ANALYSIS", "record_id": 1},
        ),
        Document(
            page_content="December 2022 had the highest number of orders with 250 orders totaling $75,432.50",
            metadata={"source": "SALES.MONTHLY_STATS", "record_id": 12},
        ),
    ]
    
    # Create embeddings
    embeddings = OpenAIEmbeddings()
    
    # Create vector store
    vector_store = FAISS.from_documents(documents, embeddings)
    
    return vector_store


def run_example(args: argparse.Namespace) -> None:
    """
    Run the comprehensive benchmark example.
    
    Args:
        args: Command line arguments
    """
    # Initialize OpenAI models
    llm = ChatOpenAI(
        api_key=args.openai_api_key,
        model="gpt-4",
        temperature=0.2,
    )
    
    sql_generation_llm = ChatOpenAI(
        api_key=args.openai_api_key,
        model="gpt-4",
        temperature=0.0,
    )
    
    evaluation_llm = ChatOpenAI(
        api_key=args.openai_api_key,
        model="gpt-4",
        temperature=0.0,
    )
    
    # Extract schema and data information from HANA
    if args.use_hana:
        try:
            from hdbcli import dbapi
            
            # Connect to HANA
            connection = dbapi.connect(
                address=args.host,
                port=args.port,
                user=args.user,
                password=args.password,
            )
            
            logger.info(f"Extracting schema information from SAP HANA for schema {args.schema}")
            schema_info = extract_schema_info_from_hana(
                host=args.host,
                port=args.port,
                user=args.user,
                password=args.password,
                schema_name=args.schema,
            )
            
            logger.info(f"Extracting data information from SAP HANA for schema {args.schema}")
            data_info = extract_data_info_from_hana(
                host=args.host,
                port=args.port,
                user=args.user,
                password=args.password,
                schema_name=args.schema,
                max_rows_per_table=args.max_rows,
            )
            
            # Extract text data for vectorization
            logger.info("Extracting text data for vectorization")
            documents = extract_text_data_for_vectorization(
                connection=connection,
                schema_name=args.schema,
                max_docs=500,
            )
            
            # Create embeddings
            embeddings = OpenAIEmbeddings(
                api_key=args.openai_api_key,
            )
            
            # Create vector store
            logger.info("Creating vector store")
            if documents:
                vector_store = FAISS.from_documents(documents, embeddings)
            else:
                logger.warning("No documents found for vectorization, using mock vector store")
                vector_store = create_mock_vector_store()
            
            # Initialize storage manager
            logger.info("Initializing storage manager")
            storage_manager = HanaStorageManager(
                connection=connection,
                schema_name=args.storage_schema,
                questions_table="BENCHMARK_QUESTIONS",
                answers_table="BENCHMARK_ANSWERS",
                results_table="BENCHMARK_RESULTS",
            )
            
            # Initialize representation handlers
            logger.info("Initializing representation handlers")
            handlers = {
                RepresentationType.RELATIONAL: RelationalRepresentationHandler(
                    connection=connection,
                    schema_name=args.schema,
                    llm=sql_generation_llm,
                ),
                RepresentationType.VECTOR: VectorRepresentationHandler(
                    connection=connection,
                    schema_name=args.schema,
                    embeddings=embeddings,
                    vector_store=vector_store,
                    llm=llm,
                ),
                RepresentationType.ONTOLOGY: OntologyRepresentationHandler(
                    connection=connection,
                    schema_name=args.schema,
                    sparql_endpoint="http://example.com/sparql",  # Mock endpoint
                    llm=llm,
                ),
            }
            
            # Add hybrid handler
            handlers[RepresentationType.HYBRID] = HybridRepresentationHandler(
                connection=connection,
                schema_name=args.schema,
                handlers=[
                    handlers[RepresentationType.RELATIONAL],
                    handlers[RepresentationType.VECTOR],
                ],
                llm=llm,
            )
        
        except Exception as e:
            logger.error(f"Error connecting to HANA: {str(e)}")
            
            # Use mock data
            logger.info("Using mock schema and data information")
            schema_info = """
            Schema: SALES

            Tables:
            - CUSTOMERS
              Columns:
              - CUSTOMER_ID: INTEGER NOT NULL (Primary key for customer)
              - NAME: NVARCHAR(100) NOT NULL (Customer's full name)
              - EMAIL: VARCHAR(255) NOT NULL (Customer's email address)
              - PHONE: VARCHAR(20) NULL (Customer's phone number)
              - ADDRESS: NVARCHAR(200) NULL (Customer's physical address)
              - SIGNUP_DATE: DATE NOT NULL (Date when customer signed up)
              Primary Key: CUSTOMER_ID

            - PRODUCTS
              Columns:
              - PRODUCT_ID: INTEGER NOT NULL (Primary key for product)
              - NAME: NVARCHAR(100) NOT NULL (Product name)
              - DESCRIPTION: NVARCHAR(1000) NULL (Product description)
              - PRICE: DECIMAL(10,2) NOT NULL (Product price)
              - CATEGORY: VARCHAR(50) NOT NULL (Product category)
              - STOCK: INTEGER NOT NULL (Available stock)
              Primary Key: PRODUCT_ID

            - ORDERS
              Columns:
              - ORDER_ID: INTEGER NOT NULL (Primary key for order)
              - CUSTOMER_ID: INTEGER NOT NULL (Customer who placed the order)
              - ORDER_DATE: TIMESTAMP NOT NULL (Date and time when order was placed)
              - STATUS: VARCHAR(20) NOT NULL (Order status)
              - TOTAL_AMOUNT: DECIMAL(12,2) NOT NULL (Total order amount)
              Primary Key: ORDER_ID
              Foreign Keys:
                - CUSTOMER_ID -> SALES.CUSTOMERS.CUSTOMER_ID

            - ORDER_ITEMS
              Columns:
              - ORDER_ITEM_ID: INTEGER NOT NULL (Primary key for order item)
              - ORDER_ID: INTEGER NOT NULL (Order this item belongs to)
              - PRODUCT_ID: INTEGER NOT NULL (Product ordered)
              - QUANTITY: INTEGER NOT NULL (Quantity ordered)
              - PRICE: DECIMAL(10,2) NOT NULL (Price at time of order)
              Primary Key: ORDER_ITEM_ID
              Foreign Keys:
                - ORDER_ID -> SALES.ORDERS.ORDER_ID
                - PRODUCT_ID -> SALES.PRODUCTS.PRODUCT_ID
            """
            
            data_info = """
            Schema: SALES

            Table Data:

            Table: CUSTOMERS
            - Row count: 1000
            - Column: CUSTOMER_ID (INTEGER)
              - Min: 1
              - Max: 1000
              - Avg: 500.5
              - Distinct values: 1000
              - NULL count: 0
              - Sample values: 1, 2, 3, 4, 5
            - Column: NAME (NVARCHAR)
              - Distinct values: 997
              - NULL count: 0
              - Sample values: Aaron Smith, Alice Johnson, Bob Williams, Carol Davis, David Miller
            - Column: EMAIL (VARCHAR)
              - Distinct values: 1000
              - NULL count: 0
              - Sample values: aaron.smith@example.com, alice.j@example.com, bob.williams@example.com
            - Column: PHONE (VARCHAR)
              - Distinct values: 998
              - NULL count: 23
              - Sample values: +1-202-555-0123, +1-303-555-0187, +1-404-555-0134
            - Column: ADDRESS (NVARCHAR)
              - Distinct values: 980
              - NULL count: 45
              - Sample values: 123 Main St, 456 Oak Ave, 789 Pine Dr
            - Column: SIGNUP_DATE (DATE)
              - Min: 2020-01-01
              - Max: 2023-11-15
              - Distinct values: 842
              - NULL count: 0
              - Sample values: 2020-01-01, 2020-01-02, 2020-01-03
            - Sample rows (5):
              - CUSTOMER_ID=1, NAME=Aaron Smith, EMAIL=aaron.smith@example.com, PHONE=+1-202-555-0123, ADDRESS=123 Main St, SIGNUP_DATE=2020-01-01
              - CUSTOMER_ID=2, NAME=Alice Johnson, EMAIL=alice.j@example.com, PHONE=+1-303-555-0187, ADDRESS=456 Oak Ave, SIGNUP_DATE=2020-01-02
              - CUSTOMER_ID=3, NAME=Bob Williams, EMAIL=bob.williams@example.com, PHONE=+1-404-555-0134, ADDRESS=789 Pine Dr, SIGNUP_DATE=2020-01-03
              - CUSTOMER_ID=4, NAME=Carol Davis, EMAIL=carol.davis@example.com, PHONE=+1-505-555-0112, ADDRESS=321 Elm St, SIGNUP_DATE=2020-01-04
              - CUSTOMER_ID=5, NAME=David Miller, EMAIL=david.m@example.com, PHONE=+1-606-555-0198, ADDRESS=654 Cedar Ave, SIGNUP_DATE=2020-01-05

            Table: PRODUCTS
            - Row count: 500
            - Column: PRODUCT_ID (INTEGER)
              - Min: 1
              - Max: 500
              - Avg: 250.5
              - Distinct values: 500
              - NULL count: 0
              - Sample values: 1, 2, 3, 4, 5
            - Column: NAME (NVARCHAR)
              - Distinct values: 500
              - NULL count: 0
              - Sample values: Ergonomic Chair, Gaming Laptop, Kitchen Mixer, Smart TV, Wireless Headphones
            - Column: DESCRIPTION (NVARCHAR)
              - Distinct values: 498
              - NULL count: 12
              - Sample values: Adjustable office chair, High-performance gaming laptop, Professional kitchen mixer
            - Column: PRICE (DECIMAL)
              - Min: 9.99
              - Max: 1999.99
              - Avg: 299.95
              - Distinct values: 234
              - NULL count: 0
              - Sample values: 9.99, 19.99, 29.99, 49.99, 99.99
            - Column: CATEGORY (VARCHAR)
              - Distinct values: 12
              - NULL count: 0
              - Sample values: Electronics, Furniture, Home, Kitchen, Office
            - Column: STOCK (INTEGER)
              - Min: 0
              - Max: 500
              - Avg: 78.3
              - Distinct values: 113
              - NULL count: 0
              - Sample values: 0, 5, 10, 15, 20
            - Sample rows (5):
              - PRODUCT_ID=1, NAME=Ergonomic Chair, DESCRIPTION=Adjustable office chair with lumbar support, PRICE=199.99, CATEGORY=Office, STOCK=45
              - PRODUCT_ID=2, NAME=Gaming Laptop, DESCRIPTION=High-performance gaming laptop, PRICE=1299.99, CATEGORY=Electronics, STOCK=20
              - PRODUCT_ID=3, NAME=Kitchen Mixer, DESCRIPTION=Professional kitchen mixer, PRICE=249.99, CATEGORY=Kitchen, STOCK=67
              - PRODUCT_ID=4, NAME=Smart TV, DESCRIPTION=55-inch 4K Smart TV, PRICE=599.99, CATEGORY=Electronics, STOCK=38
              - PRODUCT_ID=5, NAME=Wireless Headphones, DESCRIPTION=Noise-cancelling wireless headphones, PRICE=149.99, CATEGORY=Electronics, STOCK=120

            Table: ORDERS
            - Row count: 2500
            - Column: ORDER_ID (INTEGER)
              - Min: 1
              - Max: 2500
              - Avg: 1250.5
              - Distinct values: 2500
              - NULL count: 0
              - Sample values: 1, 2, 3, 4, 5
            - Column: CUSTOMER_ID (INTEGER)
              - Min: 1
              - Max: 1000
              - Avg: 493.2
              - Distinct values: 1000
              - NULL count: 0
              - Sample values: 1, 2, 3, 4, 5
            - Column: ORDER_DATE (TIMESTAMP)
              - Min: 2020-01-01 08:12:34
              - Max: 2023-11-15 22:45:12
              - Distinct values: 2487
              - NULL count: 0
              - Sample values: 2020-01-01 08:12:34, 2020-01-01 09:30:22, 2020-01-01 14:45:10
            - Column: STATUS (VARCHAR)
              - Distinct values: 4
              - NULL count: 0
              - Sample values: Completed, Delivered, Processing, Shipped
            - Column: TOTAL_AMOUNT (DECIMAL)
              - Min: 9.99
              - Max: 5432.87
              - Avg: 412.75
              - Distinct values: 1874
              - NULL count: 0
              - Sample values: 9.99, 19.99, 29.99, 49.99, 99.99
            - Sample rows (5):
              - ORDER_ID=1, CUSTOMER_ID=42, ORDER_DATE=2020-01-01 08:12:34, STATUS=Completed, TOTAL_AMOUNT=249.98
              - ORDER_ID=2, CUSTOMER_ID=87, ORDER_DATE=2020-01-01 09:30:22, STATUS=Completed, TOTAL_AMOUNT=1349.98
              - ORDER_ID=3, CUSTOMER_ID=15, ORDER_DATE=2020-01-01 14:45:10, STATUS=Completed, TOTAL_AMOUNT=199.99
              - ORDER_ID=4, CUSTOMER_ID=103, ORDER_DATE=2020-01-02 10:05:45, STATUS=Completed, TOTAL_AMOUNT=749.98
              - ORDER_ID=5, CUSTOMER_ID=67, ORDER_DATE=2020-01-02 16:22:33, STATUS=Completed, TOTAL_AMOUNT=99.99

            Table: ORDER_ITEMS
            - Row count: 6200
            - Column: ORDER_ITEM_ID (INTEGER)
              - Min: 1
              - Max: 6200
              - Avg: 3100.5
              - Distinct values: 6200
              - NULL count: 0
              - Sample values: 1, 2, 3, 4, 5
            - Column: ORDER_ID (INTEGER)
              - Min: 1
              - Max: 2500
              - Avg: 1247.8
              - Distinct values: 2500
              - NULL count: 0
              - Sample values: 1, 2, 3, 4, 5
            - Column: PRODUCT_ID (INTEGER)
              - Min: 1
              - Max: 500
              - Avg: 251.2
              - Distinct values: 500
              - NULL count: 0
              - Sample values: 1, 2, 3, 4, 5
            - Column: QUANTITY (INTEGER)
              - Min: 1
              - Max: 20
              - Avg: 2.3
              - Distinct values: 20
              - NULL count: 0
              - Sample values: 1, 2, 3, 4, 5
            - Column: PRICE (DECIMAL)
              - Min: 9.99
              - Max: 1999.99
              - Avg: 298.76
              - Distinct values: 234
              - NULL count: 0
              - Sample values: 9.99, 19.99, 29.99, 49.99, 99.99
            - Sample rows (5):
              - ORDER_ITEM_ID=1, ORDER_ID=1, PRODUCT_ID=23, QUANTITY=1, PRICE=249.98
              - ORDER_ITEM_ID=2, ORDER_ID=2, PRODUCT_ID=2, QUANTITY=1, PRICE=1299.99
              - ORDER_ITEM_ID=3, ORDER_ID=2, PRODUCT_ID=17, QUANTITY=1, PRICE=49.99
              - ORDER_ITEM_ID=4, ORDER_ID=3, PRODUCT_ID=1, QUANTITY=1, PRICE=199.99
              - ORDER_ITEM_ID=5, ORDER_ID=4, PRODUCT_ID=4, QUANTITY=1, PRICE=599.99
            """
            
            # Create mock connection and vector store
            import sqlite3
            connection = sqlite3.connect(":memory:")
            
            # Create vector store
            vector_store = create_mock_vector_store()
            
            # Create embeddings
            embeddings = OpenAIEmbeddings(
                api_key=args.openai_api_key,
            )
            
            # Initialize in-memory storage manager
            storage_manager = HanaStorageManager(
                connection=connection,
                schema_name="BENCHMARK",
            )
            
            # Initialize representation handlers (mock versions)
            handlers = {
                RepresentationType.RELATIONAL: RelationalRepresentationHandler(
                    connection=connection,
                    schema_name="SALES",
                    llm=sql_generation_llm,
                ),
                RepresentationType.VECTOR: VectorRepresentationHandler(
                    connection=connection,
                    schema_name="SALES",
                    embeddings=embeddings,
                    vector_store=vector_store,
                    llm=llm,
                ),
                RepresentationType.ONTOLOGY: OntologyRepresentationHandler(
                    connection=connection,
                    schema_name="SALES",
                    sparql_endpoint="http://example.com/sparql",  # Mock endpoint
                    llm=llm,
                ),
            }
            
            # Add hybrid handler
            handlers[RepresentationType.HYBRID] = HybridRepresentationHandler(
                connection=connection,
                schema_name="SALES",
                handlers=[
                    handlers[RepresentationType.RELATIONAL],
                    handlers[RepresentationType.VECTOR],
                ],
                llm=llm,
            )
    else:
        # Use mock data
        logger.info("Using mock schema and data information")
        schema_info = """
        Schema: SALES

        Tables:
        - CUSTOMERS
          Columns:
          - CUSTOMER_ID: INTEGER NOT NULL (Primary key for customer)
          - NAME: NVARCHAR(100) NOT NULL (Customer's full name)
          - EMAIL: VARCHAR(255) NOT NULL (Customer's email address)
          - PHONE: VARCHAR(20) NULL (Customer's phone number)
          - ADDRESS: NVARCHAR(200) NULL (Customer's physical address)
          - SIGNUP_DATE: DATE NOT NULL (Date when customer signed up)
          Primary Key: CUSTOMER_ID

        - PRODUCTS
          Columns:
          - PRODUCT_ID: INTEGER NOT NULL (Primary key for product)
          - NAME: NVARCHAR(100) NOT NULL (Product name)
          - DESCRIPTION: NVARCHAR(1000) NULL (Product description)
          - PRICE: DECIMAL(10,2) NOT NULL (Product price)
          - CATEGORY: VARCHAR(50) NOT NULL (Product category)
          - STOCK: INTEGER NOT NULL (Available stock)
          Primary Key: PRODUCT_ID

        - ORDERS
          Columns:
          - ORDER_ID: INTEGER NOT NULL (Primary key for order)
          - CUSTOMER_ID: INTEGER NOT NULL (Customer who placed the order)
          - ORDER_DATE: TIMESTAMP NOT NULL (Date and time when order was placed)
          - STATUS: VARCHAR(20) NOT NULL (Order status)
          - TOTAL_AMOUNT: DECIMAL(12,2) NOT NULL (Total order amount)
          Primary Key: ORDER_ID
          Foreign Keys:
            - CUSTOMER_ID -> SALES.CUSTOMERS.CUSTOMER_ID

        - ORDER_ITEMS
          Columns:
          - ORDER_ITEM_ID: INTEGER NOT NULL (Primary key for order item)
          - ORDER_ID: INTEGER NOT NULL (Order this item belongs to)
          - PRODUCT_ID: INTEGER NOT NULL (Product ordered)
          - QUANTITY: INTEGER NOT NULL (Quantity ordered)
          - PRICE: DECIMAL(10,2) NOT NULL (Price at time of order)
          Primary Key: ORDER_ITEM_ID
          Foreign Keys:
            - ORDER_ID -> SALES.ORDERS.ORDER_ID
            - PRODUCT_ID -> SALES.PRODUCTS.PRODUCT_ID
        """
        
        data_info = """
        Schema: SALES

        Table Data:

        Table: CUSTOMERS
        - Row count: 1000
        - Column: CUSTOMER_ID (INTEGER)
          - Min: 1
          - Max: 1000
          - Avg: 500.5
          - Distinct values: 1000
          - NULL count: 0
          - Sample values: 1, 2, 3, 4, 5
        - Column: NAME (NVARCHAR)
          - Distinct values: 997
          - NULL count: 0
          - Sample values: Aaron Smith, Alice Johnson, Bob Williams, Carol Davis, David Miller
        - Column: EMAIL (VARCHAR)
          - Distinct values: 1000
          - NULL count: 0
          - Sample values: aaron.smith@example.com, alice.j@example.com, bob.williams@example.com
        - Column: PHONE (VARCHAR)
          - Distinct values: 998
          - NULL count: 23
          - Sample values: +1-202-555-0123, +1-303-555-0187, +1-404-555-0134
        - Column: ADDRESS (NVARCHAR)
          - Distinct values: 980
          - NULL count: 45
          - Sample values: 123 Main St, 456 Oak Ave, 789 Pine Dr
        - Column: SIGNUP_DATE (DATE)
          - Min: 2020-01-01
          - Max: 2023-11-15
          - Distinct values: 842
          - NULL count: 0
          - Sample values: 2020-01-01, 2020-01-02, 2020-01-03
        - Sample rows (5):
          - CUSTOMER_ID=1, NAME=Aaron Smith, EMAIL=aaron.smith@example.com, PHONE=+1-202-555-0123, ADDRESS=123 Main St, SIGNUP_DATE=2020-01-01
          - CUSTOMER_ID=2, NAME=Alice Johnson, EMAIL=alice.j@example.com, PHONE=+1-303-555-0187, ADDRESS=456 Oak Ave, SIGNUP_DATE=2020-01-02
          - CUSTOMER_ID=3, NAME=Bob Williams, EMAIL=bob.williams@example.com, PHONE=+1-404-555-0134, ADDRESS=789 Pine Dr, SIGNUP_DATE=2020-01-03
          - CUSTOMER_ID=4, NAME=Carol Davis, EMAIL=carol.davis@example.com, PHONE=+1-505-555-0112, ADDRESS=321 Elm St, SIGNUP_DATE=2020-01-04
          - CUSTOMER_ID=5, NAME=David Miller, EMAIL=david.m@example.com, PHONE=+1-606-555-0198, ADDRESS=654 Cedar Ave, SIGNUP_DATE=2020-01-05

        Table: PRODUCTS
        - Row count: 500
        - Column: PRODUCT_ID (INTEGER)
          - Min: 1
          - Max: 500
          - Avg: 250.5
          - Distinct values: 500
          - NULL count: 0
          - Sample values: 1, 2, 3, 4, 5
        - Column: NAME (NVARCHAR)
          - Distinct values: 500
          - NULL count: 0
          - Sample values: Ergonomic Chair, Gaming Laptop, Kitchen Mixer, Smart TV, Wireless Headphones
        - Column: DESCRIPTION (NVARCHAR)
          - Distinct values: 498
          - NULL count: 12
          - Sample values: Adjustable office chair, High-performance gaming laptop, Professional kitchen mixer
        - Column: PRICE (DECIMAL)
          - Min: 9.99
          - Max: 1999.99
          - Avg: 299.95
          - Distinct values: 234
          - NULL count: 0
          - Sample values: 9.99, 19.99, 29.99, 49.99, 99.99
        - Column: CATEGORY (VARCHAR)
          - Distinct values: 12
          - NULL count: 0
          - Sample values: Electronics, Furniture, Home, Kitchen, Office
        - Column: STOCK (INTEGER)
          - Min: 0
          - Max: 500
          - Avg: 78.3
          - Distinct values: 113
          - NULL count: 0
          - Sample values: 0, 5, 10, 15, 20
        - Sample rows (5):
          - PRODUCT_ID=1, NAME=Ergonomic Chair, DESCRIPTION=Adjustable office chair with lumbar support, PRICE=199.99, CATEGORY=Office, STOCK=45
          - PRODUCT_ID=2, NAME=Gaming Laptop, DESCRIPTION=High-performance gaming laptop, PRICE=1299.99, CATEGORY=Electronics, STOCK=20
          - PRODUCT_ID=3, NAME=Kitchen Mixer, DESCRIPTION=Professional kitchen mixer, PRICE=249.99, CATEGORY=Kitchen, STOCK=67
          - PRODUCT_ID=4, NAME=Smart TV, DESCRIPTION=55-inch 4K Smart TV, PRICE=599.99, CATEGORY=Electronics, STOCK=38
          - PRODUCT_ID=5, NAME=Wireless Headphones, DESCRIPTION=Noise-cancelling wireless headphones, PRICE=149.99, CATEGORY=Electronics, STOCK=120
        """
        
        # Create mock connection and vector store
        import sqlite3
        connection = sqlite3.connect(":memory:")
        
        # Create vector store
        vector_store = create_mock_vector_store()
        
        # Create embeddings
        embeddings = OpenAIEmbeddings(
            api_key=args.openai_api_key,
        )
        
        # Initialize in-memory storage manager
        storage_manager = HanaStorageManager(
            connection=connection,
            schema_name="BENCHMARK",
        )
        
        # Initialize representation handlers (mock versions)
        handlers = {
            RepresentationType.RELATIONAL: RelationalRepresentationHandler(
                connection=connection,
                schema_name="SALES",
                llm=sql_generation_llm,
            ),
            RepresentationType.VECTOR: VectorRepresentationHandler(
                connection=connection,
                schema_name="SALES",
                embeddings=embeddings,
                vector_store=vector_store,
                llm=llm,
            ),
            RepresentationType.ONTOLOGY: OntologyRepresentationHandler(
                connection=connection,
                schema_name="SALES",
                sparql_endpoint="http://example.com/sparql",  # Mock endpoint
                llm=llm,
            ),
        }
        
        # Add hybrid handler
        handlers[RepresentationType.HYBRID] = HybridRepresentationHandler(
            connection=connection,
            schema_name="SALES",
            handlers=[
                handlers[RepresentationType.RELATIONAL],
                handlers[RepresentationType.VECTOR],
            ],
            llm=llm,
        )
    
    # Initialize question generator
    logger.info("Initializing question generator")
    question_generator = QuestionGenerator(
        generation_model=llm,
        model_id="gpt-4-question-generator",
    )
    
    # Generate benchmark questions
    logger.info("Generating benchmark questions")
    questions = question_generator.create_benchmark_questions(
        schema_info=schema_info,
        data_info=data_info,
        num_schema_questions=args.num_schema_questions,
        num_instance_questions=args.num_instance_questions,
        num_relationship_questions=args.num_relationship_questions,
        num_aggregation_questions=args.num_aggregation_questions,
        num_inference_questions=args.num_inference_questions,
        num_temporal_questions=args.num_temporal_questions,
    )
    
    # Initialize evaluator
    logger.info("Initializing factuality evaluator")
    evaluator = FactualityEvaluator(
        grading_model=evaluation_llm,
        model_id="gpt-4-evaluator",
    )
    
    # Initialize benchmark runner
    logger.info("Initializing benchmark runner")
    benchmark_runner = BenchmarkRunner(
        storage_manager=storage_manager,
        evaluator=evaluator,
        handlers=handlers,
        benchmark_name=args.benchmark_name,
    )
    
    # Run benchmark
    logger.info("Running benchmark")
    metrics = benchmark_runner.run_benchmark(
        questions=questions,
        representation_types=[
            RepresentationType.RELATIONAL,
            RepresentationType.VECTOR,
            RepresentationType.ONTOLOGY,
            RepresentationType.HYBRID,
        ],
    )
    
    # Print metrics
    logger.info("Benchmark results:")
    for rt, rt_metrics in metrics.items():
        logger.info(f"Results for {rt.value} representation:")
        logger.info(f"  Total questions: {rt_metrics.total_count}")
        logger.info(f"  Correct: {rt_metrics.correct_count} ({rt_metrics.accuracy:.2%})")
        logger.info(f"  Incorrect: {rt_metrics.incorrect_count}")
        logger.info(f"  Not attempted: {rt_metrics.not_attempted_count}")
        logger.info(f"  Ambiguous: {rt_metrics.ambiguous_count}")
        logger.info(f"  F-score: {rt_metrics.f_score:.2%}")
        
        logger.info("  By question type:")
        for qt, qt_metrics in rt_metrics.metrics_by_question_type.items():
            if qt_metrics["total"] > 0:
                logger.info(f"    {qt}: {qt_metrics['correct']}/{qt_metrics['total']} ({qt_metrics['accuracy']:.2%})")
        
        logger.info("  By difficulty:")
        for diff, diff_metrics in rt_metrics.metrics_by_difficulty.items():
            if diff_metrics["total"] > 0:
                logger.info(f"    Difficulty {diff}: {diff_metrics['correct']}/{diff_metrics['total']} ({diff_metrics['accuracy']:.2%})")
    
    # Generate recommendations
    logger.info("Generating recommendations")
    recommendations = benchmark_runner.get_recommendations(metrics)
    
    logger.info("Overall recommendations:")
    for rec in recommendations["overall"]:
        logger.info(f"- [{rec['priority']}] {rec['recommendation']}")
        logger.info(f"  Evidence: {rec['evidence']}")
    
    # Save results to file if specified
    if args.output:
        logger.info(f"Saving results to {args.output}")
        
        results = {
            "benchmark_id": benchmark_runner.benchmark_id,
            "benchmark_name": benchmark_runner.benchmark_name,
            "timestamp": time.time(),
            "metrics": {rt.value: {
                "total_count": rt_metrics.total_count,
                "correct_count": rt_metrics.correct_count,
                "incorrect_count": rt_metrics.incorrect_count,
                "not_attempted_count": rt_metrics.not_attempted_count,
                "ambiguous_count": rt_metrics.ambiguous_count,
                "accuracy": rt_metrics.accuracy,
                "f_score": rt_metrics.f_score,
                "avg_response_time_ms": rt_metrics.avg_response_time_ms,
                "metrics_by_question_type": rt_metrics.metrics_by_question_type,
                "metrics_by_difficulty": rt_metrics.metrics_by_difficulty,
            } for rt, rt_metrics in metrics.items()},
            "recommendations": recommendations,
        }
        
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
    
    logger.info("Example completed successfully")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Comprehensive benchmark example for SAP HANA data representation methods")
    
    parser.add_argument(
        "--openai-api-key",
        type=str,
        help="OpenAI API key",
        default=os.environ.get("OPENAI_API_KEY"),
    )
    
    parser.add_argument(
        "--use-hana",
        action="store_true",
        help="Use SAP HANA for schema extraction",
    )
    
    parser.add_argument(
        "--host",
        type=str,
        help="SAP HANA host",
    )
    
    parser.add_argument(
        "--port",
        type=int,
        help="SAP HANA port",
    )
    
    parser.add_argument(
        "--user",
        type=str,
        help="SAP HANA user",
    )
    
    parser.add_argument(
        "--password",
        type=str,
        help="SAP HANA password",
    )
    
    parser.add_argument(
        "--schema",
        type=str,
        default="SALES",
        help="SAP HANA schema name for data",
    )
    
    parser.add_argument(
        "--storage-schema",
        type=str,
        default="BENCHMARK",
        help="SAP HANA schema name for benchmark storage",
    )
    
    parser.add_argument(
        "--max-rows",
        type=int,
        default=5,
        help="Maximum rows per table to extract",
    )
    
    parser.add_argument(
        "--benchmark-name",
        type=str,
        default=f"Comprehensive-Benchmark-{time.strftime('%Y%m%d-%H%M%S')}",
        help="Name of the benchmark",
    )
    
    parser.add_argument(
        "--num-schema-questions",
        type=int,
        default=5,
        help="Number of schema questions to generate",
    )
    
    parser.add_argument(
        "--num-instance-questions",
        type=int,
        default=5,
        help="Number of instance questions to generate",
    )
    
    parser.add_argument(
        "--num-relationship-questions",
        type=int,
        default=5,
        help="Number of relationship questions to generate",
    )
    
    parser.add_argument(
        "--num-aggregation-questions",
        type=int,
        default=5,
        help="Number of aggregation questions to generate",
    )
    
    parser.add_argument(
        "--num-inference-questions",
        type=int,
        default=3,
        help="Number of inference questions to generate",
    )
    
    parser.add_argument(
        "--num-temporal-questions",
        type=int,
        default=3,
        help="Number of temporal questions to generate",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save benchmark results JSON",
    )
    
    args = parser.parse_args()
    
    # Validate OpenAI API key
    if not args.openai_api_key:
        parser.error("OpenAI API key is required (use --openai-api-key or set OPENAI_API_KEY environment variable)")
    
    # Validate HANA connection parameters
    if args.use_hana and not all([args.host, args.port, args.user, args.password]):
        parser.error("HANA connection parameters (host, port, user, password) are required when --use-hana is specified")
    
    return args


if __name__ == "__main__":
    args = parse_args()
    run_example(args)