#!/usr/bin/env python3
"""
Setup Analytics Tables Script

This script creates the necessary database tables for the analytics module.
Run this script after setting up the connection to SAP HANA Cloud.
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('setup_analytics_tables')

# Add the parent directory to the path so we can import the necessary modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the database connection
from api.db import get_db_connection

def create_tables():
    """Create the necessary tables for analytics."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Create query logs table
        logger.info("Creating QUERY_LOGS table...")
        query_logs_sql = """
        CREATE TABLE IF NOT EXISTS QUERY_LOGS (
            ID INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            QUERY_TEXT NVARCHAR(2000),
            QUERY_TIMESTAMP TIMESTAMP,
            RESULTS_COUNT INTEGER,
            EXECUTION_TIME INTEGER,
            USER_ID NVARCHAR(100),
            TABLE_NAME NVARCHAR(100)
        )
        """
        cursor.execute(query_logs_sql)
        
        # Create performance benchmarks table
        logger.info("Creating PERFORMANCE_BENCHMARKS table...")
        perf_benchmark_sql = """
        CREATE TABLE IF NOT EXISTS PERFORMANCE_BENCHMARKS (
            ID INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            OPERATION_TYPE NVARCHAR(100),
            OPERATION_TIMESTAMP TIMESTAMP,
            CPU_TIME FLOAT,
            GPU_TIME FLOAT,
            TENSORRT_TIME FLOAT,
            MODEL_NAME NVARCHAR(100),
            BATCH_SIZE INTEGER,
            INPUT_LENGTH INTEGER,
            PRECISION NVARCHAR(10)
        )
        """
        cursor.execute(perf_benchmark_sql)
        
        conn.commit()
        logger.info("Tables created successfully")
        
        return True
    except Exception as e:
        logger.error(f"Error creating tables: {e}")
        return False

def populate_sample_data():
    """Populate the tables with sample data for testing."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if tables are empty before populating
        cursor.execute("SELECT COUNT(*) FROM QUERY_LOGS")
        query_logs_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM PERFORMANCE_BENCHMARKS")
        perf_benchmark_count = cursor.fetchone()[0]
        
        # Only populate if tables are empty
        if query_logs_count == 0:
            logger.info("Populating QUERY_LOGS with sample data...")
            
            # Sample queries
            sample_queries = [
                "What are the sales figures for Q1 2023?",
                "Show me customer satisfaction metrics by region",
                "What is the current market share in Europe?",
                "List top-performing products in Asia-Pacific",
                "Who are our top 10 customers by revenue?",
                "Summarize the quarterly financial report",
                "What were the key factors affecting profit margin?",
                "Show me the supply chain bottlenecks",
                "Compare revenue growth across business units",
                "Analyze customer retention rates by segment"
            ]
            
            # Sample table names
            sample_tables = [
                "SALES_DATA",
                "CUSTOMER_METRICS",
                "MARKET_ANALYSIS",
                "PRODUCT_PERFORMANCE",
                "CUSTOMER_REVENUE",
                "FINANCIAL_REPORTS",
                "PROFIT_ANALYSIS",
                "SUPPLY_CHAIN",
                "REVENUE_GROWTH",
                "CUSTOMER_RETENTION"
            ]
            
            # Generate 50 sample queries over the past 6 months
            now = datetime.now()
            for i in range(50):
                query_text = sample_queries[i % len(sample_queries)]
                query_timestamp = now - timedelta(days=random.randint(0, 180), 
                                                  hours=random.randint(0, 23),
                                                  minutes=random.randint(0, 59))
                results_count = random.randint(1, 20)
                execution_time = random.randint(50, 500)
                user_id = f"user_{random.randint(1, 5)}"
                table_name = sample_tables[i % len(sample_tables)]
                
                insert_query = """
                INSERT INTO QUERY_LOGS 
                (QUERY_TEXT, QUERY_TIMESTAMP, RESULTS_COUNT, EXECUTION_TIME, USER_ID, TABLE_NAME)
                VALUES (?, ?, ?, ?, ?, ?)
                """
                cursor.execute(insert_query, 
                              (query_text, query_timestamp, results_count, 
                               execution_time, user_id, table_name))
            
            conn.commit()
            logger.info(f"Inserted 50 sample queries into QUERY_LOGS")
        else:
            logger.info(f"QUERY_LOGS already contains {query_logs_count} rows, skipping sample data")
        
        # Populate performance benchmarks
        if perf_benchmark_count == 0:
            logger.info("Populating PERFORMANCE_BENCHMARKS with sample data...")
            
            # Sample operation types
            operation_types = [
                "Embedding",
                "Vector Search",
                "Batch Processing",
                "Document Processing",
                "Inference"
            ]
            
            # Sample model names
            model_names = [
                "all-MiniLM-L6-v2",
                "SAP_NEB.20240715",
                "msmarco-distilbert-cos-v5",
                "custom_financial_model"
            ]
            
            # Sample precision types
            precision_types = ["fp32", "fp16", "int8"]
            
            # Generate 50 sample benchmark entries over the past 3 months
            now = datetime.now()
            for i in range(50):
                operation_type = operation_types[i % len(operation_types)]
                operation_timestamp = now - timedelta(days=random.randint(0, 90), 
                                                     hours=random.randint(0, 23),
                                                     minutes=random.randint(0, 59))
                
                # CPU is always slowest
                cpu_time = random.uniform(100, 500)
                
                # GPU is 5-10x faster than CPU
                gpu_time = cpu_time / random.uniform(5, 10)
                
                # TensorRT is 1.5-3x faster than GPU
                tensorrt_time = gpu_time / random.uniform(1.5, 3)
                
                model_name = model_names[random.randint(0, len(model_names) - 1)]
                batch_size = random.choice([1, 8, 16, 32, 64, 128])
                input_length = random.choice([128, 256, 512, 1024])
                precision = precision_types[random.randint(0, len(precision_types) - 1)]
                
                insert_query = """
                INSERT INTO PERFORMANCE_BENCHMARKS 
                (OPERATION_TYPE, OPERATION_TIMESTAMP, CPU_TIME, GPU_TIME, TENSORRT_TIME, 
                MODEL_NAME, BATCH_SIZE, INPUT_LENGTH, PRECISION)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                cursor.execute(insert_query, 
                              (operation_type, operation_timestamp, cpu_time, gpu_time, tensorrt_time,
                               model_name, batch_size, input_length, precision))
            
            conn.commit()
            logger.info(f"Inserted 50 sample benchmark entries into PERFORMANCE_BENCHMARKS")
        else:
            logger.info(f"PERFORMANCE_BENCHMARKS already contains {perf_benchmark_count} rows, skipping sample data")
        
        return True
    except Exception as e:
        logger.error(f"Error populating sample data: {e}")
        return False

def main():
    """Main function to set up analytics tables."""
    logger.info("Starting analytics tables setup...")
    
    if create_tables():
        logger.info("Tables created successfully.")
        
        if populate_sample_data():
            logger.info("Sample data populated successfully.")
        else:
            logger.error("Failed to populate sample data.")
            return 1
    else:
        logger.error("Failed to create tables.")
        return 1
    
    logger.info("Analytics tables setup complete.")
    return 0

if __name__ == "__main__":
    sys.exit(main())