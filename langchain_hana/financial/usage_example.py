#!/usr/bin/env python3
"""
Production-ready financial embedding system usage example.

This script demonstrates how to use the production-ready financial embedding
system with SAP HANA Cloud for document storage and retrieval.
"""

import os
import argparse
import logging
from typing import List, Dict, Any

from langchain_core.documents import Document

from langchain_hana.financial.production import create_financial_system

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_sample_financial_documents() -> List[Document]:
    """Get sample financial documents."""
    return [
        Document(
            page_content=(
                "Q1 2025 Financial Results: XYZ Corporation reported revenue of $1.2 billion, "
                "a 15% increase year-over-year. EBITDA margin improved to 28.5% from 26.2% "
                "in the prior year period. The company generated $342 million in operating "
                "cash flow and returned $150 million to shareholders through dividends and "
                "share repurchases. Management raised full-year guidance, citing strong demand "
                "for its AI-enabled financial products."
            ),
            metadata={
                "type": "earnings_report",
                "company": "XYZ Corporation",
                "period": "Q1 2025",
                "sector": "Financial Technology",
                "sentiment": "positive"
            }
        ),
        Document(
            page_content=(
                "Market Risk Assessment: Global financial markets remain volatile due to "
                "persistent inflation and geopolitical tensions. The Federal Reserve is "
                "expected to maintain higher interest rates through Q3 2025, potentially "
                "impacting corporate debt refinancing. Credit spreads have widened by 35 basis "
                "points since January, indicating increased risk aversion. We recommend "
                "reducing exposure to high-yield debt and increasing allocation to "
                "quality equities with strong balance sheets."
            ),
            metadata={
                "type": "risk_assessment",
                "date": "2025-04-15",
                "author": "Risk Management Committee",
                "category": "Market Risk",
                "sentiment": "cautious"
            }
        ),
        Document(
            page_content=(
                "Merger Announcement: Alpha Financial Services (NYSE: ALPH) has entered into "
                "a definitive agreement to acquire Beta Payment Systems for $3.8 billion in "
                "cash and stock. The transaction represents a 40% premium to Beta's closing "
                "price and values the company at 12x forward EBITDA. The acquisition is expected "
                "to be immediately accretive to Alpha's earnings and will expand its payment "
                "processing capabilities in emerging markets. Regulatory approval is expected "
                "by Q4 2025."
            ),
            metadata={
                "type": "merger_announcement",
                "companies": ["Alpha Financial Services", "Beta Payment Systems"],
                "transaction_value": 3.8,
                "date": "2025-04-10",
                "sentiment": "neutral"
            }
        ),
        Document(
            page_content=(
                "Regulatory Alert: The Financial Conduct Authority (FCA) has proposed new "
                "regulations requiring financial institutions to disclose climate-related "
                "financial risks in their annual reports. The proposal includes mandatory "
                "reporting of Scope 1, 2, and 3 emissions, climate risk scenario analysis, "
                "and transition plans aligned with the Paris Agreement. Non-compliance may "
                "result in significant penalties. The consultation period ends June 30, 2025, "
                "with implementation expected in fiscal year 2026."
            ),
            metadata={
                "type": "regulatory_alert",
                "regulator": "Financial Conduct Authority",
                "topic": "Climate Risk Disclosure",
                "deadline": "2025-06-30",
                "impact": "high",
                "sentiment": "neutral"
            }
        ),
        Document(
            page_content=(
                "Investment Recommendation: We initiate coverage of Green Energy Finance "
                "(NASDAQ: GREF) with an Overweight rating and a price target of $78, "
                "representing 35% upside potential. GREF is uniquely positioned to benefit "
                "from the transition to sustainable energy with its innovative financing "
                "solutions for renewable projects. The company's proprietary risk assessment "
                "model has delivered a 40% lower default rate than industry averages. We "
                "project 25% annual revenue growth over the next three years, driven by "
                "expanded market share and favorable regulatory tailwinds."
            ),
            metadata={
                "type": "investment_recommendation",
                "company": "Green Energy Finance",
                "ticker": "GREF",
                "rating": "Overweight",
                "price_target": 78,
                "sector": "Financial Services",
                "analyst": "Morgan Stanley",
                "sentiment": "positive"
            }
        ),
    ]


def main(args):
    """Main function."""
    # Create financial embedding system
    logger.info("Creating financial embedding system...")
    system = create_financial_system(
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        encrypt=not args.no_encrypt,
        ssl_validate=args.ssl_validate,
        model_name=args.model_name,
        quality_tier=args.quality_tier,
        table_name=args.table_name,
        log_file=args.log_file,
        connection_pool_size=args.connection_pool_size,
        enable_semantic_cache=not args.no_cache,
        cache_dir=args.cache_dir,
        redis_url=args.redis_url,
    )
    
    # Check system health
    logger.info("Checking system health...")
    health = system.health_check()
    logger.info(f"System health: {health['status']}")
    
    # Get sample documents
    if args.operation in ["add", "all"]:
        logger.info("Adding sample documents...")
        documents = get_sample_financial_documents()
        document_ids = system.add_documents(documents)
        logger.info(f"Added {len(documents)} documents with IDs: {document_ids}")
    
    # Perform similarity search
    if args.operation in ["search", "all"]:
        logger.info("Performing similarity search...")
        
        # Define example queries
        queries = [
            "What are the latest financial results?",
            "Tell me about market risks and economic outlook",
            "Are there any recent mergers or acquisitions?",
            "What new regulations should financial institutions be aware of?",
            "What investment opportunities look promising?",
        ]
        
        # Run queries
        for query in queries:
            logger.info(f"\nQuery: {query}")
            
            # Standard search
            results = system.similarity_search(
                query=query,
                k=2,
                use_cache=not args.no_cache,
            )
            
            # Print results
            logger.info(f"Found {len(results)} results:")
            for i, doc in enumerate(results):
                logger.info(f"  Result {i+1}:")
                logger.info(f"  Content: {doc.page_content[:100]}...")
                logger.info(f"  Metadata: {doc.metadata}")
            
            # Try with filter
            if args.operation == "all":
                filtered_results = system.similarity_search(
                    query=query,
                    k=1,
                    filter={"sentiment": "positive"},
                    use_cache=not args.no_cache,
                )
                
                logger.info(f"\nFiltered results (positive sentiment only):")
                for i, doc in enumerate(filtered_results):
                    logger.info(f"  Result {i+1}:")
                    logger.info(f"  Content: {doc.page_content[:100]}...")
                    logger.info(f"  Metadata: {doc.metadata}")
    
    # Get performance metrics
    if args.operation in ["metrics", "all"]:
        logger.info("\nPerformance metrics:")
        metrics = system.get_metrics()
        
        # Display key metrics
        logger.info(f"  Queries processed: {metrics.get('queries_processed', 0)}")
        logger.info(f"  Documents added: {metrics.get('documents_added', 0)}")
        logger.info(f"  Average query time: {metrics.get('avg_query_time', 0):.3f}s")
        logger.info(f"  Cache hits: {metrics.get('cache_hits', 0)}")
        logger.info(f"  Cache misses: {metrics.get('cache_misses', 0)}")
        logger.info(f"  Errors: {metrics.get('errors', 0)}")
        
        # Calculate cache hit rate
        cache_hits = metrics.get('cache_hits', 0)
        cache_misses = metrics.get('cache_misses', 0)
        total_cache_requests = cache_hits + cache_misses
        
        if total_cache_requests > 0:
            hit_rate = cache_hits / total_cache_requests * 100
            logger.info(f"  Cache hit rate: {hit_rate:.1f}%")
    
    # Shut down system
    logger.info("\nShutting down system...")
    system.shutdown()
    logger.info("Example completed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Financial Embedding System Example")
    
    # Connection parameters
    parser.add_argument("--host", required=True, help="SAP HANA hostname")
    parser.add_argument("--port", type=int, default=443, help="SAP HANA port")
    parser.add_argument("--user", required=True, help="SAP HANA username")
    parser.add_argument("--password", required=True, help="SAP HANA password")
    parser.add_argument("--no-encrypt", action="store_true", help="Disable encryption")
    parser.add_argument("--ssl-validate", action="store_true", help="Validate SSL certificates")
    
    # System parameters
    parser.add_argument("--model-name", help="Custom model name")
    parser.add_argument("--quality-tier", default="balanced", choices=["high", "balanced", "efficient"], help="Quality tier")
    parser.add_argument("--table-name", default="FINANCIAL_DOCUMENTS", help="Table name")
    parser.add_argument("--log-file", help="Log file path")
    parser.add_argument("--connection-pool-size", type=int, default=3, help="Connection pool size")
    
    # Cache parameters
    parser.add_argument("--no-cache", action="store_true", help="Disable semantic cache")
    parser.add_argument("--cache-dir", help="Cache directory")
    parser.add_argument("--redis-url", help="Redis URL")
    
    # Operation
    parser.add_argument("--operation", default="all", choices=["add", "search", "metrics", "all"], help="Operation to perform")
    
    args = parser.parse_args()
    main(args)