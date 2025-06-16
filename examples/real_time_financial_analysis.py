#!/usr/bin/env python3
"""
Real-Time Financial Data Analysis with LangChain and SAP HANA Cloud

This example demonstrates a real-time financial data analysis system using:
1. SAP HANA Cloud as the vector database for financial documents
2. FinE5 financial domain-specific embeddings
3. LangChain for retrieval and analysis
4. Streaming processing of financial updates

The system simulates a real-time financial data environment where:
- Financial documents are continuously added to the vector store
- Market data updates are processed and analyzed in real-time
- Financial insights are generated based on the latest information

Prerequisites:
- SAP HANA Cloud instance
- Python 3.8+
- Required packages: langchain, langchain_hana, langchain-openai, torch, hdbcli
- OpenAI API key or compatible LLM provider

Usage:
    python real_time_financial_analysis.py --config_file config/connection.json
"""

import os
import sys
import json
import time
import logging
import argparse
import datetime
import threading
import queue
from typing import List, Dict, Any, Optional, Tuple

import torch
from hdbcli import dbapi
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

# SAP HANA integration
from langchain_hana.connection import create_connection, test_connection
from langchain_hana.vectorstore import HanaVectorStore
from langchain_hana.financial import create_financial_embeddings
from langchain_hana.utils import DistanceStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("real_time_financial_analysis")

# Queue for new financial documents
document_queue = queue.Queue()

# Queue for market data updates
market_data_queue = queue.Queue()

# Global analysis results for API access
latest_analysis_results = {}

class FinancialDocument:
    """Class representing a financial document with metadata."""
    
    def __init__(self, 
                 content: str, 
                 doc_type: str, 
                 source: str, 
                 timestamp: Optional[datetime.datetime] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a financial document.
        
        Args:
            content: Document text content
            doc_type: Type of financial document (e.g., "earnings_report", "news")
            source: Source of the document
            timestamp: Document timestamp (defaults to current time)
            metadata: Additional metadata fields
        """
        self.content = content
        self.doc_type = doc_type
        self.source = source
        self.timestamp = timestamp or datetime.datetime.now()
        self.metadata = metadata or {}
        
        # Add core fields to metadata
        self.metadata.update({
            "type": doc_type,
            "source": source,
            "timestamp": self.timestamp.isoformat(),
            "document_id": f"{doc_type}-{int(time.time())}-{id(self)}"
        })
    
    def to_langchain_document(self) -> Document:
        """Convert to LangChain Document format."""
        return Document(
            page_content=self.content,
            metadata=self.metadata
        )


class MarketDataUpdate:
    """Class representing a market data update."""
    
    def __init__(self, 
                 ticker: str,
                 price: float,
                 change_pct: float,
                 volume: int,
                 timestamp: Optional[datetime.datetime] = None,
                 additional_data: Optional[Dict[str, Any]] = None):
        """
        Initialize a market data update.
        
        Args:
            ticker: Stock ticker symbol
            price: Current price
            change_pct: Percentage change
            volume: Trading volume
            timestamp: Update timestamp (defaults to current time)
            additional_data: Additional market data fields
        """
        self.ticker = ticker
        self.price = price
        self.change_pct = change_pct
        self.volume = volume
        self.timestamp = timestamp or datetime.datetime.now()
        self.additional_data = additional_data or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "ticker": self.ticker,
            "price": self.price,
            "change_pct": self.change_pct,
            "volume": self.volume,
            "timestamp": self.timestamp.isoformat(),
            **self.additional_data
        }
    
    def to_text(self) -> str:
        """Convert to text representation for analysis."""
        change_direction = "up" if self.change_pct > 0 else "down"
        return (
            f"Market update for {self.ticker}: Price ${self.price:.2f}, "
            f"{change_direction} {abs(self.change_pct):.2f}%, volume {self.volume:,}. "
            f"Timestamp: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
        )


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Real-Time Financial Analysis Example")
    
    # Connection parameters
    parser.add_argument("--config_file", help="Path to connection configuration file")
    parser.add_argument("--host", help="SAP HANA Cloud host")
    parser.add_argument("--port", type=int, default=443, help="SAP HANA Cloud port")
    parser.add_argument("--user", help="SAP HANA Cloud username")
    parser.add_argument("--password", help="SAP HANA Cloud password")
    
    # Application parameters
    parser.add_argument("--table_name", default="FINANCIAL_REALTIME", 
                       help="Table name for vector store")
    parser.add_argument("--model_type", default="default", 
                       choices=["default", "high_quality", "efficient", "tone"],
                       help="Financial embedding model type")
    parser.add_argument("--no_gpu", action="store_true", 
                       help="Disable GPU acceleration")
    parser.add_argument("--run_time", type=int, default=300,
                       help="How long to run the simulation in seconds (default: 300)")
    parser.add_argument("--simulation_speed", type=float, default=1.0,
                       help="Simulation speed multiplier (higher=faster)")
    
    return parser.parse_args()


def load_connection_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load connection configuration from file or environment variables."""
    # Try to load from file
    if config_path and os.path.exists(config_path):
        logger.info(f"Loading connection configuration from {config_path}")
        with open(config_path, 'r') as f:
            return json.load(f)
    
    # Try environment variables
    logger.info("Loading connection configuration from environment variables")
    return {
        "host": os.environ.get("HANA_HOST"),
        "port": int(os.environ.get("HANA_PORT", "443")),
        "user": os.environ.get("HANA_USER"),
        "password": os.environ.get("HANA_PASSWORD"),
    }


def create_hana_connection(args) -> dbapi.Connection:
    """Create connection to SAP HANA Cloud."""
    # Load connection parameters
    config = load_connection_config(args.config_file)
    
    # Use command line arguments if provided
    if args.host and args.user and args.password:
        host = args.host
        port = args.port
        user = args.user
        password = args.password
    else:
        # Use config parameters
        host = config.get("host")
        port = config.get("port", 443)
        user = config.get("user")
        password = config.get("password")
    
    # Validate parameters
    if not all([host, port, user, password]):
        raise ValueError("Missing connection parameters. Provide --config_file or individual parameters.")
    
    # Create connection
    logger.info(f"Connecting to SAP HANA Cloud: {host}:{port}")
    connection = create_connection(
        host=host,
        port=port,
        user=user,
        password=password,
        encrypt=True,
        sslValidateCertificate=False
    )
    
    # Test connection
    is_valid, info = test_connection(connection)
    if not is_valid:
        raise ConnectionError(f"Failed to connect to SAP HANA: {info.get('error')}")
    
    logger.info(f"Connected to SAP HANA Cloud {info.get('version')}")
    return connection


def setup_vector_store(connection: dbapi.Connection, args) -> HanaVectorStore:
    """Set up vector store with financial embeddings."""
    logger.info("Setting up vector store with financial embeddings")
    
    # Create financial embeddings
    embeddings = create_financial_embeddings(
        model_type=args.model_type,
        use_gpu=not args.no_gpu,
        add_financial_prefix=True,
        financial_prefix_type="general",
        enable_caching=True
    )
    
    # Create vector store
    vector_store = HanaVectorStore(
        connection=connection,
        embedding=embeddings,
        table_name=args.table_name,
        create_table=True,  # Create table if it doesn't exist
    )
    
    # Create HNSW index for faster searches
    try:
        logger.info("Creating HNSW index for faster similarity search")
        vector_store.create_hnsw_index()
    except Exception as e:
        logger.warning(f"Failed to create HNSW index: {str(e)}")
    
    return vector_store


def setup_rag_chain(vector_store: HanaVectorStore) -> Tuple[Any, Any]:
    """Set up a RAG chain for financial analysis."""
    logger.info("Setting up RAG chain for financial analysis")
    
    # Check for OpenAI API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found. Set the OPENAI_API_KEY environment variable.")
    
    # Initialize LLM
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.1,  # Low temperature for more factual responses
    )
    
    # Create a retriever from the vector store
    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": 5,  # Retrieve 5 documents
            "filter": {"type": {"$in": ["earnings_report", "news", "market_update", "analyst_report"]}}
        }
    )
    
    # Define formatter for documents
    def format_docs(docs):
        return "\n\n".join([
            f"Document ({doc.metadata.get('type')} - {doc.metadata.get('source')} - {doc.metadata.get('timestamp', 'Unknown date')}):\n{doc.page_content}" 
            for doc in docs
        ])
    
    # Define prompt for financial analysis
    analysis_template = """
    You are a real-time financial analyst processing the latest market information.
    Analyze the following financial information and provide insights.
    
    Today's date: {current_date}
    
    Recent financial documents and market updates:
    {context}
    
    Current market data:
    {market_data}
    
    Question: {question}
    
    Provide a concise analysis addressing the question based on the latest information.
    Include relevant figures, percentages, and trends. Focus on recency of information.
    If there are contradictions between newer and older information, prioritize the newer information.
    
    Analysis:
    """
    
    analysis_prompt = PromptTemplate.from_template(analysis_template)
    
    # Create the RAG chain for analysis
    analysis_chain = (
        {
            "context": retriever | format_docs, 
            "question": RunnablePassthrough(),
            "current_date": lambda _: datetime.datetime.now().strftime("%Y-%m-%d"),
            "market_data": lambda _: format_market_data()
        }
        | analysis_prompt
        | llm
        | StrOutputParser()
    )
    
    return analysis_chain, retriever


def format_market_data() -> str:
    """Format current market data for inclusion in prompts."""
    # Get the latest market data from the global dictionary
    # In a real system, this would query a market data service
    market_data = []
    
    # Convert the latest market data to text
    for ticker, data in latest_analysis_results.get("market_data", {}).items():
        if isinstance(data, dict):
            price = data.get("price", 0)
            change = data.get("change_pct", 0)
            direction = "+" if change >= 0 else ""
            market_data.append(f"{ticker}: ${price:.2f} ({direction}{change:.2f}%)")
    
    if not market_data:
        return "No current market data available."
    
    return "Current prices:\n" + "\n".join(market_data)


def generate_sample_financial_documents() -> List[FinancialDocument]:
    """Generate sample financial documents for the demonstration."""
    return [
        FinancialDocument(
            content=(
                "Quarterly Earnings Report - TechCorp Inc. (TECH)\n\n"
                "TechCorp reported Q2 2025 revenue of $2.35 billion, representing a 14.5% year-over-year increase, "
                "exceeding analyst expectations of $2.2 billion. EBITDA margin improved to 34.5% from 32.1% in the "
                "previous quarter. Operating expenses decreased by 2.8% due to efficiency initiatives. "
                "The company reported earnings per share (EPS) of $1.87, compared to consensus estimates of $1.75. "
                "Cloud services revenue grew by 28% year-over-year, now representing 62% of total revenue. "
                "The board approved a quarterly dividend of $0.38 per share, payable next month."
            ),
            doc_type="earnings_report",
            source="Investor Relations",
            metadata={
                "company": "TechCorp",
                "ticker": "TECH",
                "period": "Q2 2025",
                "sector": "Technology"
            }
        ),
        FinancialDocument(
            content=(
                "Analyst Report: FinCorp (FINC)\n\n"
                "We maintain our Overweight rating on FinCorp with a revised price target of $78 (previously $72), "
                "implying 15% upside from current levels. The company's strategic shift toward digital banking is "
                "showing positive results, with mobile transactions up 45% year-over-year. Net interest margins have "
                "improved to 3.4% despite the challenging rate environment. Loan loss provisions have decreased by "
                "20% sequentially, indicating improved credit quality. We expect EPS growth of 12-15% over the next "
                "two fiscal years, driven by operational efficiencies and market share gains in commercial lending."
            ),
            doc_type="analyst_report",
            source="Global Investment Research",
            metadata={
                "company": "FinCorp",
                "ticker": "FINC",
                "analyst": "Morgan Stanley",
                "rating": "Overweight",
                "sector": "Financial Services"
            }
        ),
        FinancialDocument(
            content=(
                "Healthcare Sector Outlook - Q2 2025\n\n"
                "The healthcare sector shows robust growth prospects for the remainder of 2025, with particular "
                "strength in biopharmaceuticals and health technology. Regulatory approvals are accelerating, with "
                "the FDA approving 15 new molecular entities in Q1 alone. MedTech Inc. (MEDT) appears particularly "
                "well-positioned with its upcoming product launches in remote patient monitoring. Healthcare spending "
                "is projected to increase by 5.8% annually through 2028, driven by aging demographics and technology "
                "adoption. Potential headwinds include drug pricing legislation and supply chain constraints affecting "
                "medical device manufacturers."
            ),
            doc_type="sector_report",
            source="Healthcare Research Team",
            metadata={
                "sector": "Healthcare",
                "companies": ["MedTech Inc", "BioPharma Ltd", "HealthSolutions Corp"],
                "tickers": ["MEDT", "BIOP", "HLTH"]
            }
        ),
        FinancialDocument(
            content=(
                "Breaking: Central Bank Raises Interest Rates by 25 Basis Points\n\n"
                "The Central Bank announced today a 25 basis point increase in the benchmark interest rate, bringing "
                "it to 3.75%. This marks the second rate hike this year and aligns with market expectations. In the "
                "accompanying statement, the bank cited persistent inflation pressures and strong employment data as "
                "key factors in the decision. The committee's dot plot now suggests two more 25bp hikes in 2025, more "
                "hawkish than previous projections. Financial markets reacted with Treasury yields rising 7-10bp across "
                "the curve, while bank stocks gained approximately 2% on the news. The central bank governor indicated "
                "that future decisions will remain data-dependent."
            ),
            doc_type="news",
            source="Financial Times",
            metadata={
                "category": "monetary_policy",
                "importance": "high",
                "region": "United States"
            }
        ),
        FinancialDocument(
            content=(
                "Green Energy Corp (GREN) Secures $2B Government Contract\n\n"
                "Green Energy Corp announced today that it has secured a $2 billion government contract to develop "
                "next-generation solar infrastructure across five states. The contract, spanning five years, represents "
                "the largest deal in the company's history and approximately 40% of its current order backlog. Under the "
                "terms, Green Energy will deploy its advanced photovoltaic systems with integrated battery storage, capable "
                "of operating in extreme weather conditions. Company CEO Sarah Johnson stated this positions Green Energy as "
                "the market leader in utility-scale renewable solutions. The first phase of implementation will begin next "
                "quarter, with revenue recognition expected to start in Q4 2025."
            ),
            doc_type="news",
            source="Business Wire",
            metadata={
                "company": "Green Energy Corp",
                "ticker": "GREN",
                "sector": "Energy",
                "category": "contracts"
            }
        ),
    ]


def generate_market_data_updates() -> List[MarketDataUpdate]:
    """Generate a series of market data updates for simulation."""
    # Base data for several stocks
    base_data = {
        "TECH": {"price": 185.75, "volume": 3500000},
        "FINC": {"price": 67.82, "volume": 2800000},
        "MEDT": {"price": 122.40, "volume": 1950000},
        "BIOP": {"price": 78.25, "volume": 1200000},
        "GREN": {"price": 43.60, "volume": 2100000},
    }
    
    # Generate a series of updates with small variations
    updates = []
    
    # Create a realistic market data series for each ticker
    for ticker, data in base_data.items():
        base_price = data["price"]
        base_volume = data["volume"]
        
        # Create 20 updates per ticker with realistic variations
        for i in range(20):
            # Simulate price movements with some randomness
            import random
            price_change = random.normalvariate(0, 0.5)  # Normal distribution with mean 0, std 0.5
            new_price = base_price * (1 + price_change/100)
            
            # Adjust base price slightly for next iteration (simulating a trend)
            trend_factor = 0.05 * (1 if random.random() > 0.4 else -1)
            base_price = new_price * (1 + trend_factor/100)
            
            # Volume varies from day to day
            volume_factor = random.uniform(0.8, 1.2)
            volume = int(base_volume * volume_factor)
            
            # Create update with a future timestamp (for the simulation)
            timestamp = datetime.datetime.now() + datetime.timedelta(minutes=i*5)
            
            updates.append(MarketDataUpdate(
                ticker=ticker,
                price=new_price,
                change_pct=price_change,
                volume=volume,
                timestamp=timestamp,
                additional_data={
                    "bid": new_price - 0.01,
                    "ask": new_price + 0.01,
                    "day_high": new_price * 1.01,
                    "day_low": new_price * 0.99
                }
            ))
    
    # Sort by timestamp
    updates.sort(key=lambda x: x.timestamp)
    return updates


def document_processor_thread(vector_store: HanaVectorStore, stop_event: threading.Event):
    """Thread to process incoming financial documents and add them to the vector store."""
    logger.info("Starting document processor thread")
    
    processed_count = 0
    
    while not stop_event.is_set():
        try:
            # Try to get a document from the queue with timeout
            try:
                document = document_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            
            # Convert to LangChain document
            langchain_doc = document.to_langchain_document()
            
            # Add to vector store
            vector_store.add_documents([langchain_doc])
            
            processed_count += 1
            logger.info(f"Added document to vector store: {document.doc_type} from {document.source}")
            
            # Mark task as done
            document_queue.task_done()
            
        except Exception as e:
            logger.error(f"Error in document processor: {str(e)}")
    
    logger.info(f"Document processor thread stopped. Processed {processed_count} documents.")


def market_data_processor_thread(stop_event: threading.Event):
    """Thread to process incoming market data updates."""
    logger.info("Starting market data processor thread")
    
    # Initialize market data in the global results
    latest_analysis_results["market_data"] = {}
    processed_count = 0
    
    while not stop_event.is_set():
        try:
            # Try to get a market data update from the queue with timeout
            try:
                update = market_data_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            
            # Process the update
            ticker = update.ticker
            
            # Update global market data
            latest_analysis_results["market_data"][ticker] = update.to_dict()
            
            # For significant price movements, create a document
            if abs(update.change_pct) >= 1.0:
                # Create a market update document for significant changes
                market_doc = FinancialDocument(
                    content=(
                        f"Market Alert: {ticker} moved {update.change_pct:.2f}% to ${update.price:.2f}. "
                        f"Trading volume is {update.volume:,}. "
                        f"This represents a significant price movement compared to recent activity."
                    ),
                    doc_type="market_update",
                    source="Real-time Market Data",
                    timestamp=update.timestamp,
                    metadata={
                        "ticker": ticker,
                        "price": update.price,
                        "change_pct": update.change_pct,
                        "volume": update.volume,
                        "alert_type": "price_movement"
                    }
                )
                
                # Add to document queue for processing
                document_queue.put(market_doc)
            
            processed_count += 1
            logger.debug(f"Processed market data: {ticker} at ${update.price:.2f} ({update.change_pct:.2f}%)")
            
            # Mark task as done
            market_data_queue.task_done()
            
        except Exception as e:
            logger.error(f"Error in market data processor: {str(e)}")
    
    logger.info(f"Market data processor thread stopped. Processed {processed_count} updates.")


def analysis_thread(rag_chain, retriever, stop_event: threading.Event, analysis_interval: int = 30):
    """Thread to periodically run analysis based on current data."""
    logger.info(f"Starting analysis thread (interval: {analysis_interval}s)")
    
    analysis_count = 0
    
    while not stop_event.is_set():
        try:
            # Run analysis every interval seconds
            time.sleep(analysis_interval)
            
            if stop_event.is_set():
                break
            
            # Run different types of analysis
            if analysis_count % 3 == 0:
                question = "Summarize the latest developments for TechCorp (TECH) based on recent information."
                analysis_type = "company_analysis"
            elif analysis_count % 3 == 1:
                question = "What are the most significant market movements in the last hour and their potential causes?"
                analysis_type = "market_movement"
            else:
                question = "Analyze the impact of the latest central bank decision on financial stocks."
                analysis_type = "macro_impact"
            
            # Run the analysis
            logger.info(f"Running analysis: {analysis_type}")
            start_time = time.time()
            result = rag_chain.invoke(question)
            elapsed_time = time.time() - start_time
            
            # Update the global results
            latest_analysis_results[analysis_type] = {
                "question": question,
                "result": result,
                "timestamp": datetime.datetime.now().isoformat(),
                "processed_time": elapsed_time
            }
            
            # Log the result
            logger.info(f"Analysis completed in {elapsed_time:.2f}s: {analysis_type}")
            logger.info(f"Result: {result[:100]}...")  # Log first 100 chars
            
            analysis_count += 1
            
        except Exception as e:
            logger.error(f"Error in analysis thread: {str(e)}")
    
    logger.info(f"Analysis thread stopped. Completed {analysis_count} analyses.")


def simulation_thread(args, stop_event: threading.Event):
    """Thread to simulate incoming financial data for the demonstration."""
    logger.info("Starting simulation thread")
    
    # Generate sample data
    financial_documents = generate_sample_financial_documents()
    market_updates = generate_market_data_updates()
    
    # Calculate timing for simulation
    simulation_speed = args.simulation_speed
    doc_interval = max(10 / simulation_speed, 0.1)  # seconds between documents
    market_interval = max(3 / simulation_speed, 0.05)  # seconds between market updates
    
    # Track what's been sent
    docs_sent = 0
    updates_sent = 0
    
    # Add initial documents to bootstrap the system
    for i, doc in enumerate(financial_documents[:2]):
        document_queue.put(doc)
        docs_sent += 1
        logger.info(f"Added initial document {i+1}/2: {doc.doc_type} from {doc.source}")
        time.sleep(0.5)  # Small delay between initial documents
    
    # Simulation loop
    start_time = time.time()
    last_doc_time = start_time
    last_market_time = start_time
    
    while not stop_event.is_set():
        current_time = time.time()
        elapsed = current_time - start_time
        
        # Check if simulation time is up
        if args.run_time and elapsed > args.run_time:
            logger.info(f"Simulation completed after {elapsed:.1f} seconds")
            break
        
        # Send a document if it's time
        if current_time - last_doc_time > doc_interval and docs_sent < len(financial_documents):
            document_queue.put(financial_documents[docs_sent])
            logger.debug(f"Simulated new document: {financial_documents[docs_sent].doc_type}")
            last_doc_time = current_time
            docs_sent += 1
        
        # Send market updates if it's time
        if current_time - last_market_time > market_interval and updates_sent < len(market_updates):
            market_data_queue.put(market_updates[updates_sent])
            last_market_time = current_time
            updates_sent += 1
        
        # All data sent, break
        if docs_sent >= len(financial_documents) and updates_sent >= len(market_updates):
            logger.info("All simulation data sent")
            break
        
        # Short sleep to prevent CPU spinning
        time.sleep(0.05)
    
    logger.info(f"Simulation thread stopped. Sent {docs_sent} documents and {updates_sent} market updates.")


def display_thread(stop_event: threading.Event, update_interval: int = 5):
    """Thread to periodically display current system status."""
    logger.info(f"Starting display thread (interval: {update_interval}s)")
    
    while not stop_event.is_set():
        try:
            # Display stats every interval seconds
            time.sleep(update_interval)
            
            if stop_event.is_set():
                break
            
            # Display current stats
            print("\n" + "="*80)
            print(f"REAL-TIME FINANCIAL ANALYSIS STATUS - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*80)
            
            # Market data
            market_data = latest_analysis_results.get("market_data", {})
            if market_data:
                print("\nCURRENT MARKET DATA:")
                print("-"*40)
                for ticker, data in sorted(market_data.items()):
                    if isinstance(data, dict):
                        price = data.get("price", 0)
                        change = data.get("change_pct", 0)
                        direction = "+" if change >= 0 else ""
                        print(f"{ticker}: ${price:.2f} ({direction}{change:.2f}%)")
            
            # Latest analyses
            for analysis_type in ["company_analysis", "market_movement", "macro_impact"]:
                analysis = latest_analysis_results.get(analysis_type)
                if analysis:
                    print(f"\nLATEST {analysis_type.upper().replace('_', ' ')}:")
                    print("-"*40)
                    print(f"Q: {analysis.get('question')}")
                    print(f"A: {analysis.get('result')[:300]}...")  # First 300 chars
                    print(f"Time: {analysis.get('timestamp')} (processed in {analysis.get('processed_time', 0):.2f}s)")
            
            # Queue stats
            print("\nSYSTEM STATUS:")
            print("-"*40)
            print(f"Document queue: {document_queue.qsize()} pending")
            print(f"Market data queue: {market_data_queue.qsize()} pending")
            
        except Exception as e:
            logger.error(f"Error in display thread: {str(e)}")
    
    logger.info("Display thread stopped.")


def main():
    """Main function to run the real-time financial analysis system."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Check OpenAI API key
    if not os.environ.get("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY environment variable not set. Set this for LLM functionality.")
    
    try:
        # Create SAP HANA connection
        connection = create_hana_connection(args)
        
        # Set up vector store with financial embeddings
        vector_store = setup_vector_store(connection, args)
        
        # Set up RAG chain for analysis
        rag_chain, retriever = setup_rag_chain(vector_store)
        
        # Create stop event for coordinating threads
        stop_event = threading.Event()
        
        # Start worker threads
        threads = []
        
        # Document processor thread
        doc_processor = threading.Thread(
            target=document_processor_thread,
            args=(vector_store, stop_event),
            name="DocumentProcessor"
        )
        doc_processor.daemon = True
        doc_processor.start()
        threads.append(doc_processor)
        
        # Market data processor thread
        market_processor = threading.Thread(
            target=market_data_processor_thread,
            args=(stop_event,),
            name="MarketDataProcessor"
        )
        market_processor.daemon = True
        market_processor.start()
        threads.append(market_processor)
        
        # Analysis thread
        analyzer = threading.Thread(
            target=analysis_thread,
            args=(rag_chain, retriever, stop_event, 30),  # Run analysis every 30 seconds
            name="Analyzer"
        )
        analyzer.daemon = True
        analyzer.start()
        threads.append(analyzer)
        
        # Display thread
        display = threading.Thread(
            target=display_thread,
            args=(stop_event, 10),  # Update display every 10 seconds
            name="Display"
        )
        display.daemon = True
        display.start()
        threads.append(display)
        
        # Simulation thread
        simulator = threading.Thread(
            target=simulation_thread,
            args=(args, stop_event),
            name="Simulator"
        )
        simulator.daemon = True
        simulator.start()
        threads.append(simulator)
        
        logger.info(f"All threads started. Running for {args.run_time} seconds.")
        
        # Wait for threads to complete or timeout
        try:
            # Wait for simulation to complete
            simulator.join()
            
            # Allow some time for processing queued items
            queue_drain_time = 30  # seconds
            logger.info(f"Simulation completed. Waiting {queue_drain_time}s for queues to drain...")
            
            # Wait for queues to drain with timeout
            drain_start = time.time()
            while (time.time() - drain_start < queue_drain_time and 
                   (not document_queue.empty() or not market_data_queue.empty())):
                time.sleep(1)
            
            # Signal all threads to stop
            logger.info("Signaling threads to stop")
            stop_event.set()
            
            # Wait for all threads to stop with timeout
            for thread in threads:
                thread.join(timeout=5.0)
                if thread.is_alive():
                    logger.warning(f"Thread {thread.name} did not stop gracefully")
            
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received. Stopping threads...")
            stop_event.set()
            
            # Wait for threads to stop
            for thread in threads:
                thread.join(timeout=2.0)
        
        # Final summary
        print("\n" + "="*80)
        print("REAL-TIME FINANCIAL ANALYSIS SUMMARY")
        print("="*80)
        
        # Show the latest analyses
        for analysis_type in ["company_analysis", "market_movement", "macro_impact"]:
            analysis = latest_analysis_results.get(analysis_type)
            if analysis:
                print(f"\n{analysis_type.upper().replace('_', ' ')}:")
                print("-"*40)
                print(f"Q: {analysis.get('question')}")
                print(f"A: {analysis.get('result')}")
                print(f"Time: {analysis.get('timestamp')}")
        
        logger.info("Real-time financial analysis demo completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    finally:
        # Close connection if it exists
        if 'connection' in locals():
            connection.close()
            logger.info("Database connection closed")


if __name__ == "__main__":
    sys.exit(main())