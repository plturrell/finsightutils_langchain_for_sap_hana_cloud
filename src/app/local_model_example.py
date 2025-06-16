#!/usr/bin/env python3
"""
Local Financial Model Example

This script demonstrates how to download, fine-tune, and use local financial 
embedding models with SAP HANA Cloud integration.
"""

import os
import logging
import argparse
from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_hana.financial.local_models import (
    create_local_model_manager,
    create_model_fine_tuner,
)
from langchain_hana.financial.embeddings import FinancialEmbeddings
from langchain_hana.financial.production import create_financial_system

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SAP HANA Cloud connection parameters
HANA_HOST = "d93a8739-44a8-4845-bef3-8ec724dea2ce.hana.prod-us10.hanacloud.ondemand.com"
HANA_PORT = 443
HANA_USER = "DBADMIN"
HANA_PASSWORD = "Initial@1"

# Sample financial texts for fine-tuning
FINANCIAL_TEXTS = [
    # Banking/Finance
    "The bank reported strong quarterly earnings with net interest income rising 15% year-over-year.",
    "Credit card delinquency rates have increased slightly but remain below historical averages.",
    "The investment bank's trading revenue declined 10% due to lower market volatility.",
    "The financial institution maintains a healthy capital adequacy ratio above regulatory requirements.",
    "Commercial loan growth slowed to 3% amid tightening lending standards.",
    
    # Markets/Investing
    "The S&P 500 index reached a new all-time high, led by technology and healthcare sectors.",
    "Bond yields rose sharply following the latest inflation report, pressuring growth stocks.",
    "The central bank signaled it may begin tapering asset purchases in the coming months.",
    "Investors rotated from growth to value stocks as economic recovery gained momentum.",
    "The commodity rally continued with crude oil prices reaching multi-year highs.",
    
    # Risk Assessment
    "Market volatility has increased due to geopolitical tensions and monetary policy uncertainty.",
    "Credit spreads in high-yield bonds have widened, indicating increased default risk perception.",
    "Liquidity in emerging market debt has deteriorated amid concerns about currency stability.",
    "The financial stress index remains below levels that would indicate systemic risk.",
    "Counterparty risk assessments have been updated to reflect changes in credit quality.",
]

# Paired sentences for similarity-based fine-tuning
SIMILAR_PAIRS = [
    # Pair 1
    ("The bank reported strong quarterly earnings with revenue growth exceeding expectations.",
     "Financial results for the quarter were robust, with the bank's income surpassing analyst forecasts."),
    
    # Pair 2
    ("Inflation concerns have prompted central banks to consider accelerating interest rate hikes.",
     "Monetary authorities are contemplating faster rate increases due to persistent inflationary pressures."),
    
    # Pair 3
    ("The asset manager expanded its ESG product lineup with three new sustainable investment funds.",
     "New environmental, social, and governance focused investment vehicles were added to the firm's offerings."),
    
    # Pair 4
    ("Market volatility increased following the release of disappointing economic data.",
     "Financial markets experienced heightened turbulence after weaker-than-expected economic indicators were published."),
    
    # Pair 5
    ("The company's debt-to-equity ratio improved as it used excess cash to reduce outstanding loans.",
     "Leveraging metrics strengthened as the firm allocated surplus liquidity to debt reduction."),
]

def download_model(args):
    """Download a financial embedding model."""
    # Create local model manager
    model_manager = create_local_model_manager(
        models_dir=args.models_dir,
        default_model=args.model_name,
    )
    
    # Download the model
    logger.info(f"Downloading model: {args.model_name}")
    model_path = model_manager.download_model(
        model_name=args.model_name,
        force=args.force,
    )
    
    logger.info(f"Model downloaded to: {model_path}")
    
    # List all downloaded models
    logger.info("All downloaded models:")
    for model in model_manager.list_models():
        logger.info(f"  - {model['name']} ({model['path']})")
    
    return model_manager, model_path

def fine_tune_model(args, model_manager=None):
    """Fine-tune a financial embedding model."""
    # Create local model manager if not provided
    if model_manager is None:
        model_manager = create_local_model_manager(
            models_dir=args.models_dir,
            default_model=args.model_name,
        )
    
    # Create model fine-tuner
    fine_tuner = create_model_fine_tuner(
        model_manager=model_manager,
        output_dir=args.output_dir,
    )
    
    # Prepare training data
    train_texts = []
    train_labels = []
    
    if args.fine_tune_mode == "pairs":
        # Use similar pairs for training
        for text1, text2 in SIMILAR_PAIRS:
            train_texts.append(text1)
            train_texts.append(text2)
    else:
        # Use individual texts
        train_texts = FINANCIAL_TEXTS
    
    # Fine-tune the model
    logger.info(f"Fine-tuning model: {args.model_name}")
    logger.info(f"Training data: {len(train_texts)} texts")
    
    tuned_model_path = fine_tuner.fine_tune(
        base_model=args.model_name,
        train_texts=train_texts,
        train_labels=train_labels if train_labels else None,
        output_model_name=args.output_model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    
    logger.info(f"Fine-tuned model saved to: {tuned_model_path}")
    
    return tuned_model_path

def test_local_model(args, model_path=None):
    """Test a local embedding model."""
    # If model_path is not provided, use model name
    if model_path is None:
        # Create local model manager
        model_manager = create_local_model_manager(
            models_dir=args.models_dir,
            default_model=args.model_name,
        )
        
        # Get model path
        model_path = model_manager.get_model_path(args.model_name)
    
    # Create financial embeddings
    embeddings = FinancialEmbeddings(
        model_name=model_path,
        device="cpu" if args.no_gpu else None,
    )
    
    # Test queries
    test_queries = [
        "What are the bank's latest quarterly earnings?",
        "How has market volatility affected investment performance?",
        "What are the key financial risks in the current economic environment?",
    ]
    
    logger.info(f"Testing model: {model_path}")
    
    # Embed queries
    for query in test_queries:
        logger.info(f"\nQuery: {query}")
        
        # Embed query
        start_time = import_time()
        embedding = embeddings.embed_query(query)
        elapsed_time = import_time() - start_time
        
        logger.info(f"Embedding dimension: {len(embedding)}")
        logger.info(f"Embedding time: {elapsed_time:.3f}s")
        logger.info(f"First 5 values: {embedding[:5]}")
    
    return embeddings

def use_with_hana(args, model_path=None):
    """Use local model with SAP HANA Cloud."""
    # Get parameters from environment variables if available
    host = os.environ.get("HANA_HOST", HANA_HOST)
    port = int(os.environ.get("HANA_PORT", HANA_PORT))
    user = os.environ.get("HANA_USER", HANA_USER)
    password = os.environ.get("HANA_PASSWORD", HANA_PASSWORD)
    
    # If model_path is not provided, use model name
    model_path_or_name = model_path or args.model_name
    
    logger.info(f"Creating financial system with model: {model_path_or_name}")
    logger.info(f"Connecting to SAP HANA: {host}:{port}")
    
    # Create financial system
    system = create_financial_system(
        host=host,
        port=port,
        user=user,
        password=password,
        model_name=model_path_or_name,
        table_name=args.table_name,
        connection_pool_size=1,
    )
    
    # Check system health
    health = system.health_check()
    logger.info(f"System health: {health['status']}")
    
    # Add sample documents
    documents = [
        Document(
            page_content=(
                "The bank reported Q1 earnings with net income of $1.2 billion, "
                "up 15% year-over-year. Return on equity reached 12.5%, while "
                "net interest margin expanded to 3.2%."
            ),
            metadata={"type": "earnings", "entity": "bank"}
        ),
        Document(
            page_content=(
                "Market volatility has increased due to inflation concerns and "
                "central bank policy uncertainty. Credit spreads in high-yield "
                "bonds have widened by 50 basis points."
            ),
            metadata={"type": "market_analysis", "entity": "bonds"}
        ),
    ]
    
    logger.info(f"Adding {len(documents)} documents")
    system.add_documents(documents)
    
    # Test queries
    test_queries = [
        "What are the bank's financial results?",
        "Tell me about market risks and bond spreads",
    ]
    
    for query in test_queries:
        logger.info(f"\nQuery: {query}")
        
        # Search
        results = system.similarity_search(query, k=1)
        
        # Print result
        doc = results[0]
        logger.info(f"Result: {doc.page_content}")
        logger.info(f"Metadata: {doc.metadata}")
    
    # Shutdown system
    system.shutdown()
    
    logger.info("Example completed successfully")

def main(args):
    """Main function."""
    try:
        # Create models directory if it doesn't exist
        if args.models_dir:
            os.makedirs(args.models_dir, exist_ok=True)
        
        # Create output directory if it doesn't exist and needed
        if args.operation in ["fine-tune", "all"] and args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
        
        # Run operations
        model_manager = None
        model_path = None
        
        if args.operation in ["download", "all"]:
            model_manager, model_path = download_model(args)
        
        if args.operation in ["fine-tune", "all"]:
            model_path = fine_tune_model(args, model_manager)
        
        if args.operation in ["test", "all"]:
            test_local_model(args, model_path)
        
        if args.operation in ["use-with-hana", "all"]:
            use_with_hana(args, model_path)
            
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise

def import_time():
    """Import time module and return current time."""
    import time
    return time.time()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local Financial Model Example")
    
    # Operation
    parser.add_argument(
        "--operation", 
        default="all", 
        choices=["download", "fine-tune", "test", "use-with-hana", "all"],
        help="Operation to perform"
    )
    
    # Model parameters
    parser.add_argument(
        "--model-name",
        default="FinMTEB/Fin-E5-small",
        help="Model name or path"
    )
    parser.add_argument(
        "--models-dir",
        default="./financial_models",
        help="Directory to store models"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force download even if already downloaded"
    )
    
    # Fine-tuning parameters
    parser.add_argument(
        "--output-dir",
        default="./fine_tuned_models",
        help="Directory for fine-tuned models"
    )
    parser.add_argument(
        "--output-model-name",
        default=None,
        help="Name for fine-tuned model"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size"
    )
    parser.add_argument(
        "--fine-tune-mode",
        default="texts",
        choices=["texts", "pairs"],
        help="Fine-tuning mode (individual texts or text pairs)"
    )
    
    # Testing parameters
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration"
    )
    
    # SAP HANA parameters
    parser.add_argument(
        "--table-name",
        default="LOCAL_MODEL_TEST",
        help="Table name for vector store"
    )
    
    args = parser.parse_args()
    main(args)