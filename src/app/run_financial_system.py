#!/usr/bin/env python3
"""
Production SAP HANA Cloud Financial System

This script provides a production-ready implementation for running the
financial embedding system with SAP HANA Cloud.

Usage:
  python run_financial_system.py --host <host> --port <port> --user <user> --password <password>
"""

import os
import sys
import time
import json
import logging
import argparse
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from langchain_hana.financial import (
    create_financial_system,
    create_local_model_manager,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("financial_system.log")
    ]
)
logger = logging.getLogger(__name__)


def create_system(args) -> Any:
    """Create financial embedding system."""
    # Get model path if using local model
    model_path = None
    if args.local_model:
        model_manager = create_local_model_manager(
            models_dir=args.models_dir,
            auto_download=args.auto_download,
        )
        model_path = model_manager.get_model_path(args.model_name)
        logger.info(f"Using local model: {model_path}")
    
    # Create financial system
    logger.info(f"Creating financial system with connection to {args.host}:{args.port}")
    
    # Determine model name or quality tier
    model_name = model_path or args.model_name
    quality_tier = args.quality_tier if not model_name else None
    
    # Create the system
    system = create_financial_system(
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        model_name=model_name,
        quality_tier=quality_tier,
        table_name=args.table_name,
        log_file=args.log_file,
        connection_pool_size=args.connection_pool_size,
        enable_semantic_cache=not args.no_cache,
        cache_dir=args.cache_dir,
        redis_url=args.redis_url,
    )
    
    # Check system health
    health = system.health_check()
    logger.info(f"System health: {health['status']}")
    
    if health['status'] != "healthy":
        logger.warning(f"System health issues detected: {health}")
        if args.strict_health_check:
            logger.error("Strict health check enabled. Exiting due to health issues.")
            system.shutdown()
            sys.exit(1)
    
    return system


def add_documents(system, input_file: str) -> List[str]:
    """Add documents from input file."""
    logger.info(f"Adding documents from {input_file}")
    
    if not os.path.exists(input_file):
        logger.error(f"Input file does not exist: {input_file}")
        return []
    
    # Load documents from file
    documents = []
    
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # Convert to Documents
        for item in data:
            doc = Document(
                page_content=item['content'],
                metadata=item.get('metadata', {})
            )
            documents.append(doc)
        
        logger.info(f"Loaded {len(documents)} documents from {input_file}")
        
        # Add to system
        document_ids = system.add_documents(documents)
        logger.info(f"Added {len(documents)} documents to system")
        
        return document_ids
        
    except Exception as e:
        logger.error(f"Error adding documents: {e}")
        return []


def process_queries(system, query_file: str, output_file: Optional[str] = None) -> None:
    """Process queries from file."""
    logger.info(f"Processing queries from {query_file}")
    
    if not os.path.exists(query_file):
        logger.error(f"Query file does not exist: {query_file}")
        return
    
    try:
        # Load queries from file
        with open(query_file, 'r') as f:
            queries = json.load(f)
        
        results = []
        
        # Process each query
        for query_item in queries:
            query = query_item['query']
            filter_dict = query_item.get('filter', {})
            k = query_item.get('k', 4)
            
            logger.info(f"Processing query: {query}")
            
            # Process query
            start_time = time.time()
            query_results = system.similarity_search(
                query=query,
                k=k,
                filter=filter_dict,
            )
            query_time = time.time() - start_time
            
            # Convert results to dictionary
            result_items = []
            for doc in query_results:
                result_items.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                })
            
            # Add to results
            results.append({
                'query': query,
                'filter': filter_dict,
                'k': k,
                'results': result_items,
                'query_time': query_time,
            })
        
        # Save results if output file is specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Results saved to {output_file}")
        
        # Log statistics
        total_queries = len(queries)
        total_time = sum(r['query_time'] for r in results)
        avg_time = total_time / total_queries if total_queries > 0 else 0
        
        logger.info(f"Processed {total_queries} queries in {total_time:.2f}s (avg: {avg_time:.2f}s)")
        
    except Exception as e:
        logger.error(f"Error processing queries: {e}")


def main(args):
    """Main function."""
    try:
        # Create the system
        system = create_system(args)
        
        # Process command
        if args.command == 'add':
            add_documents(system, args.input_file)
        
        elif args.command == 'query':
            process_queries(system, args.input_file, args.output_file)
        
        elif args.command == 'metrics':
            metrics = system.get_metrics()
            print(json.dumps(metrics, indent=2))
        
        elif args.command == 'health':
            health = system.health_check()
            print(json.dumps(health, indent=2))
        
        # Shutdown system
        logger.info("Shutting down system")
        system.shutdown()
        
    except Exception as e:
        logger.error(f"Error in main function: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Production SAP HANA Cloud Financial System")
    
    # Connection parameters
    parser.add_argument("--host", required=True, help="SAP HANA hostname")
    parser.add_argument("--port", type=int, default=443, help="SAP HANA port")
    parser.add_argument("--user", required=True, help="SAP HANA username")
    parser.add_argument("--password", required=True, help="SAP HANA password")
    
    # System parameters
    parser.add_argument("--model-name", default="FinMTEB/Fin-E5", help="Model name or path")
    parser.add_argument("--quality-tier", default="high", choices=["high", "balanced", "efficient"], help="Quality tier")
    parser.add_argument("--table-name", default="FINANCIAL_DOCUMENTS", help="Table name")
    parser.add_argument("--log-file", default="financial_system.log", help="Log file path")
    parser.add_argument("--connection-pool-size", type=int, default=3, help="Connection pool size")
    
    # Cache parameters
    parser.add_argument("--no-cache", action="store_true", help="Disable semantic cache")
    parser.add_argument("--cache-dir", default="./cache", help="Cache directory")
    parser.add_argument("--redis-url", help="Redis URL")
    
    # Local model parameters
    parser.add_argument("--local-model", action="store_true", default=True, help="Use local model")
    parser.add_argument("--models-dir", default="./financial_models", help="Models directory")
    parser.add_argument("--auto-download", action="store_true", default=True, help="Auto-download models")
    
    # Health check parameters
    parser.add_argument("--strict-health-check", action="store_true", help="Exit on health check failure")
    
    # Command parameters
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Add documents command
    add_parser = subparsers.add_parser("add", help="Add documents")
    add_parser.add_argument("--input-file", required=True, help="Input file path (JSON)")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Process queries")
    query_parser.add_argument("--input-file", required=True, help="Input file path (JSON)")
    query_parser.add_argument("--output-file", help="Output file path (JSON)")
    
    # Metrics command
    metrics_parser = subparsers.add_parser("metrics", help="Get system metrics")
    
    # Health command
    health_parser = subparsers.add_parser("health", help="Check system health")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    main(args)