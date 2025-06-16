#!/usr/bin/env python3
"""
Financial RAG Example with SAP HANA Cloud and Fin-E5 Embeddings

This example demonstrates how to build a Retrieval-Augmented Generation (RAG) system
for financial documents using:
1. SAP HANA Cloud as the vector database
2. FinMTEB/Fin-E5 financial domain-specific embeddings
3. LangChain for orchestration
4. Optional GPU acceleration with TensorRT

The example shows how to:
- Connect to SAP HANA Cloud
- Initialize financial domain-specific embeddings
- Create a vector store with financial documents
- Build a RAG chain for financial question answering
- Evaluate retrieval quality with financial metrics

Prerequisites:
- SAP HANA Cloud instance
- Python 3.8+
- Required packages: langchain, langchain_hana, sentence-transformers, torch, openai

Usage:
    python financial_rag_example.py --config path/to/config.json
"""

import os
import sys
import json
import logging
import argparse
import time
import torch
from typing import List, Dict, Any, Optional, Tuple

# LangChain components
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI

# SAP HANA integration
from langchain_hana.connection import create_connection, test_connection
from langchain_hana.vectorstore import HanaVectorStore
from langchain_hana.financial import (
    FinE5Embeddings,
    FinE5TensorRTEmbeddings,
    create_financial_embeddings,
    FINANCIAL_EMBEDDING_MODELS
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("financial_rag_example")

# Sample financial documents for demonstration
FINANCIAL_DOCUMENTS = [
    {
        "content": """
Q1 2025 Financial Results: Company XYZ reported revenue of $1.2 billion, up 15% year-over-year. 
EBITDA margin improved to 28.5%, while operating expenses decreased by 3%. 
The board approved a quarterly dividend of $0.45 per share.
Net income was $320 million, representing an EPS of $1.45, exceeding analyst expectations of $1.32.
Cash flow from operations increased by 22% to $450 million, allowing for a $200 million share repurchase program.
The company has $2.4 billion in cash and cash equivalents on the balance sheet.
""",
        "metadata": {
            "type": "earnings_report",
            "company": "XYZ Corp",
            "period": "Q1 2025",
            "document_id": "ER-2025-Q1-001",
            "source": "Investor Relations"
        }
    },
    {
        "content": """
Risk Assessment Report: Market volatility remains elevated due to geopolitical tensions and inflationary pressures.
The credit default swap spreads widened by 25 basis points, indicating increased default risk in the high-yield sector.
We recommend maintaining a defensive position in fixed income portfolios, with a focus on investment-grade securities.
Interest rate risk remains substantial, with the yield curve inverting further, suggesting potential economic slowdown.
Commodity price volatility has increased, with energy prices showing high sensitivity to supply disruptions.
Regulatory risk has increased in the technology sector, with new data privacy regulations expected to impact margins.
""",
        "metadata": {
            "type": "risk_report",
            "date": "2025-04-15",
            "sector": "multiple",
            "document_id": "RR-2025-04-001",
            "source": "Risk Management Department"
        }
    },
    {
        "content": """
Merger Announcement: Alpha Corp. (NASDAQ: ALPH) announced its intention to acquire Beta Technologies for $3.5 billion in a cash and stock transaction.
The deal values Beta at 12x forward EBITDA, representing a 40% premium to its current market valuation.
The acquisition is expected to be accretive to earnings by fiscal year 2026 and generate annual synergies of approximately $250 million.
The combined entity will have a market share of approximately 35% in the quantum computing hardware segment.
The transaction is subject to regulatory approval and is expected to close in Q3 2025.
Alpha's stock price declined 3% on the announcement, while market analysts have mixed opinions on the valuation.
""",
        "metadata": {
            "type": "merger_announcement",
            "companies": ["Alpha Corp", "Beta Technologies"],
            "date": "2025-05-10",
            "value": 3.5,
            "document_id": "MA-2025-05-001",
            "source": "Press Release"
        }
    },
    {
        "content": """
SEC Filing 10-K: Item 1A - Risk Factors.
The company faces significant competition in all markets in which it operates.
Failure to attract and retain key personnel could impair our ability to deliver products and services.
Our substantial indebtedness could adversely affect our financial position and prevent us from fulfilling our obligations under our credit facilities.
The company has $4.2 billion in long-term debt, with $800 million maturing in the next 18 months.
Our debt-to-EBITDA ratio stands at 2.8x, above our target range of 2.0-2.5x.
Interest expense increased by 15% year-over-year due to higher rates and increased borrowing.
Cybersecurity incidents could result in unauthorized access to our systems and information, causing potential business disruption.
""",
        "metadata": {
            "type": "sec_filing",
            "filing_type": "10-K",
            "section": "Risk Factors",
            "company": "XYZ Corp",
            "year": 2025,
            "document_id": "SEC-10K-2025-001",
            "source": "SEC Edgar Database"
        }
    },
    {
        "content": """
Investment Thesis: We initiate coverage of Green Energy Solutions (GES) with an Overweight rating and $75 price target, implying 35% upside potential.
GES is well-positioned to benefit from the global transition to renewable energy, with its innovative battery technology offering 2x the energy density of current market leaders at a 30% lower production cost.
The company's revenue is projected to grow at a 40% CAGR over the next three years, reaching $2 billion by 2027.
Gross margins are expected to expand from 32% to 38% due to manufacturing scale efficiencies and improved supply chain management.
The company has a robust R&D pipeline, with three major product launches expected in the next 18 months.
Key risks include technology execution, increased competition, and potential regulatory changes in subsidy programs.
Our discounted cash flow analysis yields a valuation range of $68-82 per share, using a 12% WACC and 3% terminal growth rate.
""",
        "metadata": {
            "type": "investment_thesis",
            "company": "Green Energy Solutions",
            "analyst": "Morgan Stanley",
            "rating": "Overweight",
            "target_price": 75,
            "document_id": "IT-2025-GES-001",
            "source": "Equity Research"
        }
    },
    {
        "content": """
Market Analysis Report: Global Financial Markets Q2 2025 Outlook.
Equity markets are showing signs of overvaluation, with the S&P 500 P/E ratio at 22x, above the 10-year average of 18x.
Fixed income presents selective opportunities, particularly in short-duration investment-grade corporate bonds.
Emerging markets are expected to outperform developed markets, with a projected GDP growth differential of 2.5%.
The US dollar is likely to weaken against a basket of currencies due to narrowing interest rate differentials.
Commodities remain supported by supply constraints, with industrial metals expected to outperform.
Sector rotation suggests defensive sectors (healthcare, utilities, consumer staples) may outperform in the near term.
Volatility indices suggest market complacency, which could lead to correction vulnerability if earnings disappoint.
Liquidity conditions are tightening as central banks continue quantitative tightening programs.
""",
        "metadata": {
            "type": "market_analysis",
            "period": "Q2 2025",
            "scope": "Global",
            "author": "Global Strategy Team",
            "document_id": "MA-2025-Q2-001",
            "source": "Investment Strategy Department"
        }
    },
]

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Financial RAG Example with SAP HANA Cloud")
    
    # Connection parameters
    parser.add_argument("--config", help="Path to connection configuration file (JSON)")
    parser.add_argument("--host", help="SAP HANA Cloud host")
    parser.add_argument("--port", type=int, default=443, help="SAP HANA Cloud port (default: 443)")
    parser.add_argument("--user", help="SAP HANA Cloud username")
    parser.add_argument("--password", help="SAP HANA Cloud password")
    
    # Model parameters
    parser.add_argument("--model_type", choices=list(FINANCIAL_EMBEDDING_MODELS.keys()), 
                       default="default", help="Financial embedding model type")
    parser.add_argument("--no_gpu", action="store_true", help="Disable GPU acceleration")
    parser.add_argument("--use_tensorrt", action="store_true", help="Use TensorRT acceleration if available")
    
    # Application parameters
    parser.add_argument("--table_name", default="FINANCIAL_RAG_EXAMPLE", 
                       help="Table name for vector store")
    parser.add_argument("--openai_model", default="gpt-3.5-turbo", 
                       help="OpenAI model for RAG")
    parser.add_argument("--interactive", action="store_true", 
                       help="Run in interactive mode")
    parser.add_argument("--eval", action="store_true", 
                       help="Run evaluation on retrieval quality")
    parser.add_argument("--visualize", action="store_true", 
                       help="Generate visualization of financial embeddings")
    
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from a JSON file."""
    if not config_path or not os.path.exists(config_path):
        return None
    
    with open(config_path, 'r') as f:
        return json.load(f)

def setup_connection(args):
    """Set up connection to SAP HANA Cloud."""
    # Load configuration from file if provided
    config = load_config(args.config) if args.config else None
    
    # Use command line arguments if provided
    if args.host and args.user and args.password:
        connection_params = {
            "host": args.host,
            "port": args.port,
            "user": args.user,
            "password": args.password,
        }
    # Use config file parameters if available
    elif config and "connection" in config:
        connection_params = config["connection"]
    # Try environment variables
    else:
        connection_params = {
            "host": os.environ.get("HANA_HOST"),
            "port": int(os.environ.get("HANA_PORT", "443")),
            "user": os.environ.get("HANA_USER"),
            "password": os.environ.get("HANA_PASSWORD"),
        }
    
    # Check for required parameters
    for param in ["host", "port", "user", "password"]:
        if not connection_params.get(param):
            raise ValueError(f"Missing required connection parameter: {param}")
    
    # Create connection with standard security settings
    connection = create_connection(
        host=connection_params["host"],
        port=connection_params["port"],
        user=connection_params["user"],
        password=connection_params["password"],
        encrypt=True,
        sslValidateCertificate=False
    )
    
    # Test connection
    is_valid, info = test_connection(connection)
    if not is_valid:
        raise ConnectionError(f"Failed to connect to SAP HANA: {info.get('error', 'Unknown error')}")
    
    logger.info(f"Connected to SAP HANA Cloud {info.get('version', 'Unknown version')}")
    logger.info(f"Current schema: {info.get('current_schema', 'Unknown schema')}")
    
    return connection

def setup_embeddings(args):
    """Set up financial domain-specific embeddings."""
    use_gpu = not args.no_gpu and torch.cuda.is_available()
    device = "cuda" if use_gpu else "cpu"
    
    logger.info(f"Setting up financial embeddings: model_type={args.model_type}, device={device}")
    
    # Get the model name for the selected type
    model_name = FINANCIAL_EMBEDDING_MODELS[args.model_type]
    logger.info(f"Using model: {model_name}")
    
    if args.use_tensorrt and device == "cuda":
        # Use TensorRT acceleration if requested and available
        try:
            embeddings = FinE5TensorRTEmbeddings(
                model_type=args.model_type,
                precision="fp16",  # Use FP16 for better performance
                multi_gpu=False,   # Set to True for multi-GPU systems
                add_financial_prefix=True,
                financial_prefix_type="general",
                enable_caching=True
            )
            logger.info("Using TensorRT-accelerated financial embeddings")
            return embeddings
        except Exception as e:
            logger.warning(f"Failed to initialize TensorRT embeddings: {str(e)}")
            logger.info("Falling back to standard embeddings")
    
    # Use standard financial embeddings
    embeddings = FinE5Embeddings(
        model_type=args.model_type,
        device=device,
        use_fp16=device == "cuda",  # Use FP16 on GPU for better performance
        add_financial_prefix=True,
        financial_prefix_type="general",
        enable_caching=True,
        adaptive_batch_size=True
    )
    logger.info(f"Using standard financial embeddings on {device}")
    
    return embeddings

def setup_vector_store(connection, embeddings, args):
    """Set up vector store in SAP HANA Cloud."""
    logger.info(f"Setting up vector store with table name: {args.table_name}")
    
    # Create vector store
    vector_store = HanaVectorStore(
        connection=connection,
        embedding=embeddings,
        table_name=args.table_name,
        create_table=True,  # Create table if it doesn't exist
    )
    
    return vector_store

def prepare_documents(docs_list):
    """Convert document dictionaries to Document objects."""
    return [
        Document(
            page_content=doc["content"],
            metadata=doc["metadata"]
        )
        for doc in docs_list
    ]

def add_documents_to_store(vector_store, documents):
    """Add documents to the vector store with performance tracking."""
    logger.info(f"Adding {len(documents)} documents to vector store")
    
    start_time = time.time()
    
    # Extract texts and metadata
    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]
    
    # Add documents to vector store
    vector_store.add_texts(texts, metadatas=metadatas)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Added documents in {elapsed_time:.2f} seconds ({len(documents)/elapsed_time:.2f} docs/sec)")
    
    # Create HNSW index for faster similarity search
    try:
        logger.info("Creating HNSW index for faster similarity search")
        vector_store.create_hnsw_index()
        logger.info("HNSW index created successfully")
    except Exception as e:
        logger.warning(f"Failed to create HNSW index: {str(e)}")

def setup_rag_chain(vector_store, args):
    """Set up a RAG chain for financial question answering."""
    logger.info(f"Setting up RAG chain with {args.openai_model}")
    
    # Initialize LLM
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found. Set the OPENAI_API_KEY environment variable.")
    
    llm = ChatOpenAI(
        model=args.openai_model,
        temperature=0.1,  # Low temperature for more factual responses
        api_key=api_key
    )
    
    # Create a retriever from the vector store
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}  # Return top 3 results
    )
    
    # Create a financial analyst prompt template
    template = """
    You are a specialized financial analyst assistant with expertise in interpreting financial data, 
    market reports, earnings announcements, and SEC filings. Answer the user's question based on 
    the retrieved financial information.

    Guidelines:
    - Base your answer strictly on the information provided in the context
    - Be precise with numbers, percentages, and financial metrics
    - Maintain professional financial terminology
    - If the information is not in the context, acknowledge the limitation
    - Provide balanced analysis without investment advice
    - For financial metrics, explain their significance briefly

    Context:
    {context}

    Question: {question}

    Financial Analysis:
    """
    
    prompt = PromptTemplate.from_template(template)
    
    # Define the processing function for formatting context
    def format_docs(docs):
        return "\n\n".join([f"Document ({doc.metadata.get('type', 'Unknown')} - {doc.metadata.get('source', 'Unknown')}):\n{doc.page_content}" for doc in docs])
    
    # Create the RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever

def run_rag_query(rag_chain, query):
    """Run a query through the RAG chain and measure performance."""
    logger.info(f"Running query: {query}")
    
    start_time = time.time()
    response = rag_chain.invoke(query)
    elapsed_time = time.time() - start_time
    
    logger.info(f"Query processed in {elapsed_time:.2f} seconds")
    
    return response, elapsed_time

def run_evaluation(vector_store, retriever):
    """Run evaluation on retrieval quality with financial metrics."""
    logger.info("Running evaluation on retrieval quality")
    
    # Sample financial queries for evaluation
    evaluation_queries = [
        {
            "query": "What was XYZ Corp's revenue and growth in the latest quarter?",
            "expected_doc_types": ["earnings_report"],
            "expected_companies": ["XYZ Corp"]
        },
        {
            "query": "What are the main financial risks mentioned in recent reports?",
            "expected_doc_types": ["risk_report", "sec_filing"],
            "expected_keywords": ["risk", "volatility", "default"]
        },
        {
            "query": "Describe the details of the recent merger announcement between Alpha and Beta",
            "expected_doc_types": ["merger_announcement"],
            "expected_companies": ["Alpha Corp", "Beta Technologies"]
        },
        {
            "query": "What is the investment thesis for Green Energy Solutions?",
            "expected_doc_types": ["investment_thesis"],
            "expected_companies": ["Green Energy Solutions"]
        },
        {
            "query": "What is the current market outlook for equities and fixed income?",
            "expected_doc_types": ["market_analysis"],
            "expected_keywords": ["equity", "markets", "fixed income"]
        }
    ]
    
    # Evaluation metrics
    results = {
        "total_queries": len(evaluation_queries),
        "retrieval_precision": 0,
        "entity_match_rate": 0,
        "keyword_match_rate": 0,
        "type_match_rate": 0,
        "avg_retrieval_time": 0,
    }
    
    total_retrieval_time = 0
    correct_doc_types = 0
    correct_entities = 0
    correct_keywords = 0
    
    # Run evaluation queries
    for i, eval_query in enumerate(evaluation_queries):
        query = eval_query["query"]
        logger.info(f"Evaluation query {i+1}/{len(evaluation_queries)}: {query}")
        
        # Measure retrieval time
        start_time = time.time()
        docs = retriever.invoke(query)
        retrieval_time = time.time() - start_time
        total_retrieval_time += retrieval_time
        
        # Check document type match
        retrieved_doc_types = [doc.metadata.get("type") for doc in docs]
        expected_types = eval_query.get("expected_doc_types", [])
        type_matches = sum(1 for doc_type in retrieved_doc_types if doc_type in expected_types)
        type_match_rate = type_matches / max(1, len(retrieved_doc_types))
        correct_doc_types += type_match_rate
        
        # Check entity match if applicable
        if "expected_companies" in eval_query:
            expected_companies = eval_query["expected_companies"]
            retrieved_companies = []
            for doc in docs:
                if "company" in doc.metadata:
                    retrieved_companies.append(doc.metadata["company"])
                if "companies" in doc.metadata:
                    retrieved_companies.extend(doc.metadata["companies"])
            
            entity_matches = sum(1 for company in retrieved_companies if company in expected_companies)
            entity_match_rate = entity_matches / max(1, len(retrieved_companies)) if retrieved_companies else 0
            correct_entities += entity_match_rate
        else:
            correct_entities += 1  # Skip this metric if not applicable
        
        # Check keyword match if applicable
        if "expected_keywords" in eval_query:
            expected_keywords = eval_query["expected_keywords"]
            content_text = " ".join([doc.page_content.lower() for doc in docs])
            keyword_matches = sum(1 for keyword in expected_keywords if keyword.lower() in content_text)
            keyword_match_rate = keyword_matches / len(expected_keywords)
            correct_keywords += keyword_match_rate
        else:
            correct_keywords += 1  # Skip this metric if not applicable
        
        # Log retrieval results
        logger.info(f"Retrieved document types: {retrieved_doc_types}")
        logger.info(f"Type match rate: {type_match_rate:.2f}")
        logger.info(f"Retrieval time: {retrieval_time:.4f}s")
    
    # Calculate final metrics
    results["retrieval_precision"] = correct_doc_types / len(evaluation_queries)
    results["entity_match_rate"] = correct_entities / len(evaluation_queries)
    results["keyword_match_rate"] = correct_keywords / len(evaluation_queries)
    results["type_match_rate"] = correct_doc_types / len(evaluation_queries)
    results["avg_retrieval_time"] = total_retrieval_time / len(evaluation_queries)
    
    # Log evaluation results
    logger.info("Evaluation Results:")
    logger.info(f"Retrieval Precision: {results['retrieval_precision']:.4f}")
    logger.info(f"Entity Match Rate: {results['entity_match_rate']:.4f}")
    logger.info(f"Keyword Match Rate: {results['keyword_match_rate']:.4f}")
    logger.info(f"Document Type Match Rate: {results['type_match_rate']:.4f}")
    logger.info(f"Average Retrieval Time: {results['avg_retrieval_time']:.4f}s")
    
    return results

def run_interactive_mode(rag_chain):
    """Run an interactive Q&A session with the RAG chain."""
    logger.info("Starting interactive Q&A session (press Ctrl+C to exit)")
    print("\n" + "="*60)
    print("Financial RAG System with SAP HANA Cloud")
    print("Ask questions about financial reports, risk assessments, market analysis, etc.")
    print("Type 'exit' to quit")
    print("="*60 + "\n")
    
    try:
        while True:
            query = input("\nQuestion: ")
            if query.lower() in ["exit", "quit", "q"]:
                break
            
            response, elapsed_time = run_rag_query(rag_chain, query)
            print(f"\nFinancial Analysis ({elapsed_time:.2f}s):")
            print("-" * 40)
            print(response)
    except KeyboardInterrupt:
        print("\nExiting...")
    
    print("\nThank you for using the Financial RAG system!")

def main():
    """Main function to run the financial RAG example."""
    # Parse command line arguments
    args = parse_arguments()
    
    try:
        # Set up connection to SAP HANA Cloud
        connection = setup_connection(args)
        
        # Set up financial embeddings
        embeddings = setup_embeddings(args)
        
        # Set up vector store
        vector_store = setup_vector_store(connection, embeddings, args)
        
        # Prepare and add documents
        documents = prepare_documents(FINANCIAL_DOCUMENTS)
        add_documents_to_store(vector_store, documents)
        
        # Set up RAG chain
        rag_chain, retriever = setup_rag_chain(vector_store, args)
        
        # Run sample query
        sample_query = "What financial risks are mentioned in the SEC filing?"
        sample_response, _ = run_rag_query(rag_chain, sample_query)
        
        print("\n" + "="*60)
        print("Sample Financial RAG Query")
        print("="*60)
        print(f"Query: {sample_query}")
        print("\nResponse:")
        print(sample_response)
        
        # Run evaluation if requested
        if args.eval:
            evaluation_results = run_evaluation(vector_store, retriever)
        
        # Run interactive mode if requested
        if args.interactive:
            run_interactive_mode(rag_chain)
        
        # Visualize embeddings if requested
        if args.visualize:
            logger.info("Visualization of financial embeddings is not implemented in this version")
        
        logger.info("Financial RAG example completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    finally:
        # Close connection if it exists
        if 'connection' in locals():
            connection.close()
            logger.info("Connection closed")

if __name__ == "__main__":
    sys.exit(main())