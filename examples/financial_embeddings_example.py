#!/usr/bin/env python3
"""
Financial Embeddings Integration Example with SAP HANA Cloud

This example demonstrates how to use the FinMTEB/Fin-E5 financial domain-specific
embeddings with SAP HANA Cloud for financial document storage and retrieval.

Prerequisites:
- SAP HANA Cloud instance
- PyTorch installed
- sentence-transformers installed
- langchain installed
- hdbcli installed

Example usage:
```
python financial_embeddings_example.py --config_file config/connection.json
```
"""

import os
import json
import argparse
import logging
from typing import List, Dict, Any, Optional

import torch
from hdbcli import dbapi
from langchain_core.documents import Document

# Import our custom financial embeddings
from langchain_hana.financial import (
    FinE5Embeddings,
    FinE5TensorRTEmbeddings,
    create_financial_embeddings,
    FINANCIAL_EMBEDDING_MODELS
)
from langchain_hana.vectorstores import HanaDB
from langchain_hana.utils import DistanceStrategy


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_connection_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load connection configuration from file or environment variables."""
    # Try to load from file
    if config_path is None:
        possible_paths = [
            "connection.json",
            "config/connection.json",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break
    
    if config_path and os.path.exists(config_path):
        logger.info(f"Loading connection configuration from {config_path}")
        with open(config_path, 'r') as f:
            return json.load(f)
    
    # Try environment variables
    logger.info("Loading connection configuration from environment variables")
    return {
        "address": os.environ.get("HANA_HOST"),
        "port": int(os.environ.get("HANA_PORT", "443")),
        "user": os.environ.get("HANA_USER"),
        "password": os.environ.get("HANA_PASSWORD"),
    }


def create_connection(connection_params: Dict[str, Any]) -> dbapi.Connection:
    """Create a connection to SAP HANA Cloud."""
    logger.info(f"Connecting to {connection_params['address']}:{connection_params['port']} as {connection_params['user']}")
    
    connection = dbapi.connect(
        address=connection_params["address"],
        port=connection_params["port"],
        user=connection_params["user"],
        password=connection_params["password"],
        encrypt=connection_params.get("encrypt", True),
        sslValidateCertificate=connection_params.get("sslValidateCertificate", False)
    )
    
    logger.info("Connected to SAP HANA Cloud")
    return connection


def get_sample_financial_documents() -> List[Document]:
    """Get sample financial documents for demonstration."""
    return [
        Document(
            page_content=(
                "Q1 2025 Financial Results: Company XYZ reported revenue of $1.2 billion, "
                "up 15% year-over-year. EBITDA margin improved to 28.5%, while operating "
                "expenses decreased by 3%. The board approved a quarterly dividend of $0.45 per share."
            ),
            metadata={"type": "earnings_report", "quarter": "Q1", "year": 2025, "company": "XYZ"}
        ),
        Document(
            page_content=(
                "Risk Assessment: Market volatility remains elevated due to geopolitical tensions "
                "and inflationary pressures. The credit default swap spreads widened by 25 basis points, "
                "indicating increased default risk in the high-yield sector. We recommend maintaining "
                "a defensive position in fixed income portfolios."
            ),
            metadata={"type": "risk_report", "date": "2025-04-15", "sector": "fixed_income"}
        ),
        Document(
            page_content=(
                "Merger Announcement: Alpha Corp. (NASDAQ: ALPH) announced its intention to acquire "
                "Beta Technologies for $3.5 billion in a cash and stock transaction. The deal values "
                "Beta at 12x forward EBITDA, representing a 40% premium to its current market valuation. "
                "The acquisition is expected to be accretive to earnings by fiscal year 2026."
            ),
            metadata={"type": "merger_announcement", "companies": ["Alpha Corp", "Beta Technologies"], "value": 3.5}
        ),
        Document(
            page_content=(
                "SEC Filing 10-K: Item 1A - Risk Factors. The company faces significant competition "
                "in all markets in which it operates. Failure to attract and retain key personnel could "
                "impair our ability to deliver products and services. Our substantial indebtedness could "
                "adversely affect our financial position and prevent us from fulfilling our obligations under "
                "our credit facilities."
            ),
            metadata={"type": "sec_filing", "filing_type": "10-K", "section": "Risk Factors", "year": 2025}
        ),
        Document(
            page_content=(
                "Investment Thesis: We initiate coverage of Green Energy Solutions (GES) with an Overweight "
                "rating and $75 price target, implying 35% upside potential. GES is well-positioned to benefit "
                "from the global transition to renewable energy, with its innovative battery technology offering "
                "2x the energy density of current market leaders at a 30% lower production cost."
            ),
            metadata={"type": "investment_thesis", "company": "Green Energy Solutions", "analyst": "Morgan Stanley"}
        ),
    ]


def setup_financial_vector_store(
    connection: dbapi.Connection,
    embeddings_model: str = "default",
    table_name: str = "FINANCIAL_DOCUMENTS",
    use_gpu: bool = True,
    use_tensorrt: bool = False,
    add_financial_prefix: bool = True
) -> HanaDB:
    """
    Set up a vector store for financial documents using SAP HANA Cloud.
    
    Args:
        connection: SAP HANA Cloud connection
        embeddings_model: Financial embeddings model type to use
        table_name: Table name for the vector store
        use_gpu: Whether to use GPU acceleration
        use_tensorrt: Whether to use TensorRT for GPU acceleration
        add_financial_prefix: Whether to add financial context prefix
        
    Returns:
        Configured vector store
    """
    # Create financial embeddings
    logger.info(f"Setting up financial embeddings model: {embeddings_model}")
    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        logger.info("Using GPU acceleration")
        if use_tensorrt:
            logger.info("TensorRT acceleration enabled")
    else:
        logger.info("Using CPU for embedding generation")
    
    # Log available models
    logger.info("Available financial models:")
    for model_type, model_name in FINANCIAL_EMBEDDING_MODELS.items():
        logger.info(f"  - {model_type}: {model_name}")
    
    # Create embeddings model
    if use_tensorrt and device == 'cuda':
        # Use TensorRT-accelerated embeddings
        embeddings = FinE5TensorRTEmbeddings(
            model_type=embeddings_model,
            precision="fp16",
            multi_gpu=True,
            add_financial_prefix=add_financial_prefix,
            financial_prefix_type="general",
            enable_caching=True
        )
        logger.info("Created TensorRT-accelerated Fin-E5 embeddings")
    else:
        # Use standard embeddings
        embeddings = create_financial_embeddings(
            model_type=embeddings_model,
            use_gpu=use_gpu,
            use_tensorrt=False,
            add_financial_prefix=add_financial_prefix,
            financial_prefix_type="general",
            enable_caching=True
        )
        logger.info("Created standard Fin-E5 embeddings")
    
    # Log embedding model details
    if isinstance(embeddings, FinE5Embeddings):
        logger.info(f"Embedding model: {embeddings.model_name}")
        logger.info(f"Embedding dimension: {embeddings.get_embedding_dimension()}")
        logger.info(f"Device: {embeddings.device}")
    
    # Create vector store
    logger.info(f"Creating vector store in table: {table_name}")
    vector_store = HanaDB(
        connection=connection,
        embedding=embeddings,
        distance_strategy=DistanceStrategy.COSINE,
        table_name=table_name
    )
    
    return vector_store


def create_hnsw_index(vector_store: HanaDB) -> None:
    """
    Create an HNSW index for fast similarity search.
    
    Args:
        vector_store: Vector store to index
    """
    logger.info("Creating HNSW index for fast similarity search")
    
    try:
        # Create index with optimal parameters for financial text
        vector_store.create_hnsw_index(
            m=16,                  # Number of connections per node
            ef_construction=128,   # Index building quality parameter
            ef_search=64           # Search quality parameter
        )
        logger.info(f"HNSW index created for {vector_store.table_name}")
    except Exception as e:
        logger.error(f"Failed to create HNSW index: {str(e)}")


def run_financial_search_example(vector_store: HanaDB) -> None:
    """
    Run a financial search example using the vector store.
    
    Args:
        vector_store: Vector store to search
    """
    # Define sample financial queries
    financial_queries = [
        "What are the latest earnings results?",
        "What risks are mentioned in SEC filings?",
        "Tell me about recent merger and acquisition activity",
        "What are the investment opportunities in renewable energy?",
        "How is the company's financial position and debt level?"
    ]
    
    # Run similarity search for each query
    for query in financial_queries:
        logger.info(f"\nQuery: {query}")
        
        # Standard similarity search
        results = vector_store.similarity_search(query, k=2)
        
        logger.info("Top matches:")
        for i, doc in enumerate(results):
            logger.info(f"{i+1}. {doc.page_content[:100]}...")
            logger.info(f"   Metadata: {doc.metadata}")
        
        # Try with metadata filtering
        if "risk" in query.lower():
            logger.info("\nFiltered results (SEC filings only):")
            filtered_results = vector_store.similarity_search(
                query,
                k=2,
                filter={"type": "sec_filing"}
            )
            
            for i, doc in enumerate(filtered_results):
                logger.info(f"{i+1}. {doc.page_content[:100]}...")
                logger.info(f"   Metadata: {doc.metadata}")


def run_mmr_search_example(vector_store: HanaDB) -> None:
    """
    Run a Maximal Marginal Relevance (MMR) search example to demonstrate
    diverse results.
    
    Args:
        vector_store: Vector store to search
    """
    logger.info("\n=== Maximal Marginal Relevance (MMR) Search Example ===")
    query = "What financial risks and opportunities are present in the market?"
    
    # Standard similarity search
    logger.info(f"\nStandard similarity search for: {query}")
    standard_results = vector_store.similarity_search(query, k=3)
    
    for i, doc in enumerate(standard_results):
        logger.info(f"{i+1}. {doc.page_content[:100]}...")
        logger.info(f"   Metadata: {doc.metadata}")
    
    # MMR search for more diverse results
    logger.info(f"\nMMR search for more diverse results: {query}")
    mmr_results = vector_store.max_marginal_relevance_search(
        query,
        k=3,
        fetch_k=5,
        lambda_mult=0.5  # Lower lambda value = more diversity
    )
    
    for i, doc in enumerate(mmr_results):
        logger.info(f"{i+1}. {doc.page_content[:100]}...")
        logger.info(f"   Metadata: {doc.metadata}")


def main(args):
    """Main function."""
    # Load connection configuration
    connection_params = load_connection_config(args.config_file)
    
    # Create connection
    connection = create_connection(connection_params)
    
    # Get sample documents
    documents = get_sample_financial_documents()
    logger.info(f"Loaded {len(documents)} sample financial documents")
    
    # Set up vector store with financial embeddings
    vector_store = setup_financial_vector_store(
        connection=connection,
        embeddings_model=args.model_type,
        table_name=args.table_name,
        use_gpu=not args.no_gpu,
        use_tensorrt=args.use_tensorrt,
        add_financial_prefix=not args.no_prefix
    )
    
    # Add documents to vector store
    vector_store.add_documents(documents)
    logger.info(f"Added {len(documents)} documents to vector store")
    
    # Create HNSW index if requested
    if args.create_index:
        create_hnsw_index(vector_store)
    
    # Run search examples
    run_financial_search_example(vector_store)
    
    # Run MMR search example
    run_mmr_search_example(vector_store)
    
    # Close connection
    connection.close()
    logger.info("Example completed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Financial Embeddings Integration Example")
    parser.add_argument("--config_file", help="Path to connection configuration file")
    parser.add_argument("--model_type", default="default", 
                       choices=list(FINANCIAL_EMBEDDING_MODELS.keys()), 
                       help="Financial model type to use")
    parser.add_argument("--table_name", default="FINANCIAL_DOCUMENTS", help="Table name for vector store")
    parser.add_argument("--no_gpu", action="store_true", help="Disable GPU acceleration")
    parser.add_argument("--use_tensorrt", action="store_true", help="Use TensorRT for GPU acceleration")
    parser.add_argument("--no_prefix", action="store_true", help="Disable financial context prefix")
    parser.add_argument("--create_index", action="store_true", help="Create HNSW index for fast similarity search")
    
    args = parser.parse_args()
    main(args)