#!/usr/bin/env python
"""
Example of using the factuality measurement framework for SAP HANA schemas.

This example demonstrates how to generate questions from SAP HANA schemas,
evaluate model answers for factual accuracy, and assess question-answer quality
from both KPMG audit and Jony Ive design perspectives.
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, List, Any, Optional

from langchain_openai import ChatOpenAI

# Add parent directory to path to import local modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_hana.reasoning.factuality import (
    SchemaFactualityBenchmark,
    FactualityEvaluator,
    HanaSchemaQuestionGenerator,
    AuditStyleEvaluator,
    FactualityGrade,
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
        connection.close()
        
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
        connection.close()
        
        return data_info
    
    except Exception as e:
        logger.error(f"Error extracting data info: {str(e)}")
        raise


def run_example(args: argparse.Namespace) -> None:
    """
    Run the factuality benchmark example.
    
    Args:
        args: Command line arguments
    """
    # Initialize OpenAI models with different temperature settings
    generation_model = ChatOpenAI(
        api_key=args.openai_api_key,
        model="gpt-4",
        temperature=0.7,
    )
    
    answer_model = ChatOpenAI(
        api_key=args.openai_api_key,
        model="gpt-4",
        temperature=0.2,
    )
    
    evaluation_model = ChatOpenAI(
        api_key=args.openai_api_key,
        model="gpt-4",
        temperature=0.0,
    )
    
    # Extract schema and data information from HANA
    if args.use_hana:
        try:
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
        except Exception as e:
            logger.error(f"Error connecting to HANA: {str(e)}")
            
            # Use sample data if connection fails
            logger.info("Using sample schema and data information")
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
    else:
        # Use sample data
        logger.info("Using sample schema and data information")
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
    
    # Initialize question generator
    logger.info("Initializing question generator")
    question_generator = HanaSchemaQuestionGenerator(
        generation_model=generation_model,
        model_id="gpt-4-question-generator",
    )
    
    # Create a benchmark
    logger.info("Creating factuality benchmark")
    benchmark = question_generator.create_benchmark_from_schema(
        schema_name=args.schema,
        schema_info=schema_info,
        num_schema_questions=args.num_schema_questions,
        data_info=data_info if args.include_data else None,
        num_data_questions=args.num_data_questions if args.include_data else 0,
        description=f"Factuality benchmark for {args.schema} schema",
        metadata={
            "created_at": time.time(),
            "creator": "factuality_benchmark_example.py",
        },
    )
    
    # Display generated questions
    logger.info(f"Generated {len(benchmark.questions)} questions:")
    for i, (question_id, question) in enumerate(list(benchmark.questions.items())[:5], 1):
        logger.info(f"Question {i}: {question.question_text}")
        logger.info(f"Answer: {question.reference_answer}")
        logger.info(f"Entity: {question.schema_entity}")
        logger.info(f"Type: {question.entity_type}")
        logger.info("")
    
    if len(benchmark.questions) > 5:
        logger.info(f"... and {len(benchmark.questions) - 5} more questions")
    
    # Generate model answers
    logger.info("Generating model answers")
    
    for question_id, question in benchmark.questions.items():
        # Create a simple prompt for the model
        prompt = f"""Answer the following question about a database schema:

Question: {question.question_text}

Provide a short, direct answer without explanations. If you're not sure, say "I don't know."

Answer:"""
        
        # Time the response
        start_time = time.time()
        response = answer_model.invoke(prompt)
        response_time = time.time() - start_time
        
        # Extract the answer
        answer_text = response.content.strip()
        
        # Add the answer to the benchmark
        benchmark.add_model_answer(
            question_id=question_id,
            model_id="gpt-4",
            answer_text=answer_text,
            response_time=response_time,
        )
    
    # Initialize evaluator
    logger.info("Initializing factuality evaluator")
    evaluator = FactualityEvaluator(
        grading_model=evaluation_model,
        model_id="gpt-4-evaluator",
    )
    
    # Grade model answers
    logger.info("Grading model answers")
    evaluator.grade_benchmark(
        benchmark=benchmark,
        model_id="gpt-4",
    )
    
    # Get model performance
    performance = benchmark.get_model_performance(model_id="gpt-4")
    
    logger.info(f"Model performance:")
    logger.info(f"Total questions: {performance['total_questions']}")
    logger.info(f"Correct answers: {performance['correct_answers']} ({performance['overall_accuracy'] * 100:.2f}%)")
    logger.info(f"Incorrect answers: {performance['incorrect_answers']}")
    logger.info(f"Not attempted: {performance['not_attempted']}")
    logger.info(f"Ambiguous: {performance['ambiguous']}")
    logger.info(f"F-score: {performance['f_score'] * 100:.2f}%")
    
    # Get confidence calibration
    calibration = benchmark.get_confidence_calibration(model_id="gpt-4")
    
    # Initialize audit style evaluator
    logger.info("Initializing audit style evaluator")
    audit_evaluator = AuditStyleEvaluator(
        evaluation_model=evaluation_model,
        model_id="gpt-4-audit-evaluator",
    )
    
    # Evaluate a sample of questions for style
    logger.info("Evaluating questions for KPMG audit and Jony Ive design perspectives")
    
    # Take a sample of questions for style evaluation (to reduce API costs)
    sample_questions = list(benchmark.questions.items())[:args.style_sample_size]
    
    kpmg_scores = []
    ive_scores = []
    
    for question_id, question in sample_questions:
        # KPMG audit evaluation
        logger.info(f"Evaluating question '{question.question_text}' with KPMG audit perspective")
        kpmg_eval = audit_evaluator.evaluate_kpmg_style(
            question=question.question_text,
            answer=question.reference_answer,
            entity=question.schema_entity,
            entity_type=question.entity_type,
        )
        
        if kpmg_eval["overall_score"] is not None:
            kpmg_scores.append(kpmg_eval["overall_score"])
            logger.info(f"KPMG overall score: {kpmg_eval['overall_score']}/10")
        
        # Jony Ive design evaluation
        logger.info(f"Evaluating question '{question.question_text}' with Jony Ive design perspective")
        ive_eval = audit_evaluator.evaluate_ive_style(
            question=question.question_text,
            answer=question.reference_answer,
            entity=question.schema_entity,
            entity_type=question.entity_type,
        )
        
        if ive_eval["overall_score"] is not None:
            ive_scores.append(ive_eval["overall_score"])
            logger.info(f"Jony Ive overall score: {ive_eval['overall_score']}/10")
    
    # Calculate average scores
    avg_kpmg_score = sum(kpmg_scores) / len(kpmg_scores) if kpmg_scores else 0
    avg_ive_score = sum(ive_scores) / len(ive_scores) if ive_scores else 0
    
    logger.info(f"Average KPMG audit score: {avg_kpmg_score:.2f}/10")
    logger.info(f"Average Jony Ive design score: {avg_ive_score:.2f}/10")
    
    # Save the benchmark results
    if args.output:
        logger.info(f"Saving benchmark results to {args.output}")
        benchmark.save_to_file(args.output)
    
    logger.info("Example completed successfully")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Factuality benchmark example for SAP HANA schemas")
    
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
        help="SAP HANA schema name",
    )
    
    parser.add_argument(
        "--include-data",
        action="store_true",
        help="Include data-related questions",
    )
    
    parser.add_argument(
        "--max-rows",
        type=int,
        default=5,
        help="Maximum rows per table to extract",
    )
    
    parser.add_argument(
        "--num-schema-questions",
        type=int,
        default=10,
        help="Number of schema questions to generate",
    )
    
    parser.add_argument(
        "--num-data-questions",
        type=int,
        default=10,
        help="Number of data questions to generate",
    )
    
    parser.add_argument(
        "--style-sample-size",
        type=int,
        default=3,
        help="Number of questions to evaluate for style",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save benchmark results",
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