#!/usr/bin/env python
"""
RAG Example using LangChain with SAP HANA Cloud Vector Store

This example demonstrates how to implement a Retrieval-Augmented Generation (RAG) system
using LangChain with SAP HANA Cloud as the vector store. It shows:

1. Connecting to SAP HANA Cloud
2. Creating a vector store with embeddings
3. Adding documents to the vector store
4. Building a retrieval chain
5. Using the chain for question answering

Prerequisites:
- SAP HANA Cloud instance with vector capabilities
- Python 3.8+
- Required packages: langchain, langchain_hana, langchain-openai, hdbcli

Usage:
    python langchain_hana_rag_example.py --host your-hana-host.hanacloud.ondemand.com --port 443 --user your-user --password your-password
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Any

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_hana.vectorstore import HanaVectorStore
from langchain_hana.connection import get_connection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define sample documents
SAMPLE_DOCUMENTS = [
    {
        "content": "SAP HANA Cloud is a cloud-based in-memory database that provides fast data processing and analytics capabilities. It combines OLTP and OLAP workloads on a single platform, enabling real-time analytics on live transactional data.",
        "metadata": {"source": "SAP Documentation", "category": "database", "topic": "cloud"}
    },
    {
        "content": "Vector databases store and query data as high-dimensional vectors, enabling semantic search based on meaning rather than keywords. They're ideal for machine learning and AI applications that require similarity search.",
        "metadata": {"source": "Database Guide", "category": "database", "topic": "vector"}
    },
    {
        "content": "LangChain is a framework for developing applications powered by language models. It provides tools and components for creating context-aware, reasoning applications using LLMs.",
        "metadata": {"source": "LangChain Documentation", "category": "framework", "topic": "ai"}
    },
    {
        "content": "Retrieval-Augmented Generation (RAG) is a technique that enhances large language models by retrieving relevant information from external sources before generating responses. This improves accuracy and reduces hallucinations.",
        "metadata": {"source": "AI Research Paper", "category": "technique", "topic": "ai"}
    },
    {
        "content": "SAP HANA Cloud offers vector capabilities that allow storing and searching high-dimensional vectors efficiently. This makes it suitable for AI applications like semantic search and recommendation systems.",
        "metadata": {"source": "SAP Blog", "category": "database", "topic": "vector"}
    },
]

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="LangChain RAG Example with SAP HANA Cloud")
    
    # Connection parameters
    parser.add_argument("--host", help="SAP HANA Cloud host")
    parser.add_argument("--port", type=int, default=443, help="SAP HANA Cloud port (default: 443)")
    parser.add_argument("--user", help="SAP HANA Cloud username")
    parser.add_argument("--password", help="SAP HANA Cloud password")
    parser.add_argument("--config", help="Path to connection configuration file")
    
    # Optional parameters
    parser.add_argument("--table", default="LANGCHAIN_RAG_EXAMPLE", help="Table name for vector store")
    parser.add_argument("--openai-api-key", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    
    return parser.parse_args()

def setup_vector_store(args):
    """Set up the SAP HANA Cloud Vector Store."""
    logger.info("Setting up vector store...")
    
    # Set up connection to SAP HANA Cloud
    if args.config:
        # Use configuration file if provided
        connection = get_connection(args.config)
    elif args.host and args.user and args.password:
        # Use provided connection parameters
        connection = get_connection({
            "host": args.host,
            "port": args.port,
            "user": args.user,
            "password": args.password,
        })
    else:
        # Try to use environment variables
        connection = get_connection()
    
    # Set up embedding model
    # You can replace this with HuggingFaceEmbeddings or another embedding model
    embeddings = OpenAIEmbeddings()
    
    # Initialize the vector store
    vector_store = HanaVectorStore(
        connection=connection,
        embedding=embeddings,
        table_name=args.table,
        create_table=True,  # Create the table if it doesn't exist
        create_hnsw_index=True,  # Create an HNSW index for faster searches
    )
    
    return vector_store

def add_sample_documents(vector_store):
    """Add sample documents to the vector store."""
    logger.info("Adding sample documents to vector store...")
    
    # Convert to Document objects
    documents = [
        Document(
            page_content=doc["content"],
            metadata=doc["metadata"]
        )
        for doc in SAMPLE_DOCUMENTS
    ]
    
    # Add documents to the vector store
    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]
    vector_store.add_texts(texts, metadatas)
    
    logger.info(f"Added {len(documents)} documents to the vector store")
    
    return documents

def create_rag_chain(vector_store):
    """Create a RAG chain for question answering."""
    logger.info("Creating RAG chain...")
    
    # Initialize the LLM (you can replace this with a different model)
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    
    # Create a retriever from the vector store
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}  # Return top 3 results
    )
    
    # Define the prompt template
    template = """
    You are an assistant that answers questions based on the provided context.
    
    Context:
    {context}
    
    Question: {question}
    
    Answer the question using only the information from the context. If you don't know the answer based on the context, say "I don't have enough information to answer this question."
    
    Answer:
    """
    
    prompt = PromptTemplate.from_template(template)
    
    # Define the processing function for formatting context
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])
    
    # Create the RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

def run_interactive_qa(rag_chain):
    """Run an interactive Q&A session."""
    logger.info("Starting interactive Q&A session (press Ctrl+C to exit)")
    print("\n" + "="*50)
    print("RAG Q&A System with SAP HANA Cloud")
    print("Ask questions or type 'exit' to quit")
    print("="*50 + "\n")
    
    try:
        while True:
            question = input("\nQuestion: ")
            if question.lower() in ["exit", "quit", "q"]:
                break
                
            # Get the answer from the RAG chain
            answer = rag_chain.invoke(question)
            print(f"\nAnswer: {answer}")
            
    except KeyboardInterrupt:
        print("\nExiting...")
    
    print("\nThank you for using the RAG Q&A system!")

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Set OpenAI API key if provided
    if args.openai_api_key:
        os.environ["OPENAI_API_KEY"] = args.openai_api_key
    
    try:
        # Check if OpenAI API key is available
        if not os.environ.get("OPENAI_API_KEY"):
            logger.warning("No OpenAI API key found. Please set OPENAI_API_KEY environment variable or use --openai-api-key")
            return 1
        
        # Set up the vector store
        vector_store = setup_vector_store(args)
        
        # Add sample documents
        documents = add_sample_documents(vector_store)
        
        # Create the RAG chain
        rag_chain = create_rag_chain(vector_store)
        
        # Run the interactive Q&A session
        run_interactive_qa(rag_chain)
        
        return 0
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())