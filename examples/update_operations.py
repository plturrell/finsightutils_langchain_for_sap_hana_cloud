#!/usr/bin/env python3
"""
SAP HANA Cloud LangChain Integration - Update Operations Example

This example demonstrates how to use the update operations in the
SAP HANA Cloud LangChain Integration to manage documents in a vector store.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional

from hdbcli import dbapi
from langchain_core.embeddings import HuggingFaceEmbeddings
from langchain_hana import HanaDB
from langchain_hana.utils import DistanceStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("update_example")

# Database connection parameters
HOST = os.environ.get("HANA_HOST", "localhost")
PORT = os.environ.get("HANA_PORT", "30015")
USER = os.environ.get("HANA_USER", "SYSTEM")
PASSWORD = os.environ.get("HANA_PASSWORD", "")
TABLE_NAME = os.environ.get("DEFAULT_TABLE_NAME", "EMBEDDINGS")


def connect_to_hana() -> dbapi.Connection:
    """Connect to SAP HANA Cloud database."""
    logger.info(f"Connecting to SAP HANA Cloud at {HOST}:{PORT}")
    connection = dbapi.connect(
        address=HOST,
        port=PORT,
        user=USER,
        password=PASSWORD,
        encrypt=True,
        sslValidateCertificate=False
    )
    logger.info("Connected successfully to SAP HANA Cloud")
    return connection


def initialize_vectorstore(connection: dbapi.Connection) -> HanaDB:
    """Initialize the vector store."""
    logger.info("Initializing vector store with embedding model")
    
    # Create embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create vector store instance
    vectorstore = HanaDB(
        connection=connection,
        embedding=embedding_model,
        distance_strategy=DistanceStrategy.COSINE,
        table_name=TABLE_NAME,
        enable_lineage=True,
        enable_audit_logging=True,
        audit_log_to_console=True,
    )
    
    return vectorstore


def add_sample_documents(vectorstore: HanaDB) -> None:
    """Add sample documents to the vector store."""
    logger.info("Adding sample documents to the vector store")
    
    # Sample documents
    documents = [
        {
            "text": "SAP HANA Cloud provides powerful database capabilities.",
            "metadata": {"category": "database", "source": "sample1.txt", "version": "1.0"}
        },
        {
            "text": "Vector databases enable efficient similarity search.",
            "metadata": {"category": "vectors", "source": "sample2.txt", "version": "1.0"}
        },
        {
            "text": "LangChain is a framework for building LLM applications.",
            "metadata": {"category": "llm", "source": "sample3.txt", "version": "1.0"}
        },
        {
            "text": "Update operations complete the CRUD functionality.",
            "metadata": {"category": "api", "source": "sample4.txt", "version": "1.0"}
        }
    ]
    
    # Add documents to the vector store
    texts = [doc["text"] for doc in documents]
    metadatas = [doc["metadata"] for doc in documents]
    
    vectorstore.add_texts(texts=texts, metadatas=metadatas)
    logger.info(f"Added {len(documents)} sample documents")


def demonstrate_update_operations(vectorstore: HanaDB) -> None:
    """Demonstrate various update operations."""
    logger.info("\n=== Demonstrating Update Operations ===")
    
    # 1. Update a document with new content and metadata
    logger.info("\n1. Updating document content and metadata")
    
    # First, find a document to update
    results = vectorstore.similarity_search(
        query="database capabilities",
        k=1,
        filter={"category": "database"}
    )
    
    if results:
        original_doc = results[0]
        logger.info(f"Original document: {original_doc.page_content}")
        logger.info(f"Original metadata: {original_doc.metadata}")
        
        # Update the document
        vectorstore.update_texts(
            texts=["SAP HANA Cloud provides powerful database and vector capabilities."],
            filter={"category": "database"},
            metadatas=[{"category": "database", "source": "sample1.txt", "version": "2.0", "updated": True}],
        )
        
        # Verify the update
        updated_results = vectorstore.similarity_search(
            query="database capabilities",
            k=1,
            filter={"category": "database"}
        )
        
        if updated_results:
            updated_doc = updated_results[0]
            logger.info(f"Updated document: {updated_doc.page_content}")
            logger.info(f"Updated metadata: {updated_doc.metadata}")
    
    # 2. Update metadata only without regenerating embeddings
    logger.info("\n2. Updating metadata only (no embedding regeneration)")
    
    results = vectorstore.similarity_search(
        query="LangChain framework",
        k=1,
        filter={"category": "llm"}
    )
    
    if results:
        original_doc = results[0]
        logger.info(f"Original document: {original_doc.page_content}")
        logger.info(f"Original metadata: {original_doc.metadata}")
        
        # Update metadata only
        vectorstore.update_texts(
            texts=[original_doc.page_content],  # Keep same content
            filter={"category": "llm"},
            metadatas=[{"category": "llm", "source": "sample3.txt", "version": "1.1", "status": "reviewed"}],
            update_embeddings=False  # Don't regenerate embeddings
        )
        
        # Verify the update
        updated_results = vectorstore.similarity_search(
            query="LangChain framework",
            k=1,
            filter={"category": "llm"}
        )
        
        if updated_results:
            updated_doc = updated_results[0]
            logger.info(f"Updated document: {updated_doc.page_content}")
            logger.info(f"Updated metadata: {updated_doc.metadata}")
    
    # 3. Demonstrate upsert operation (update existing)
    logger.info("\n3. Demonstrating upsert (update existing)")
    
    vectorstore.upsert_texts(
        texts=["Vector databases enable extremely efficient similarity search and retrieval."],
        metadatas=[{"category": "vectors", "source": "sample2.txt", "version": "2.0", "updated_by": "upsert"}],
        filter={"category": "vectors"}
    )
    
    # Verify the upsert
    results = vectorstore.similarity_search(
        query="vector similarity",
        k=1,
        filter={"category": "vectors"}
    )
    
    if results:
        doc = results[0]
        logger.info(f"Upserted document (existing): {doc.page_content}")
        logger.info(f"Upserted metadata (existing): {doc.metadata}")
    
    # 4. Demonstrate upsert operation (add new)
    logger.info("\n4. Demonstrating upsert (add new)")
    
    vectorstore.upsert_texts(
        texts=["Generative AI models like GPT-4 have transformed natural language processing."],
        metadatas=[{"category": "gen_ai", "source": "sample5.txt", "version": "1.0"}],
        filter={"category": "gen_ai"}  # This category doesn't exist yet
    )
    
    # Verify the upsert
    results = vectorstore.similarity_search(
        query="generative AI",
        k=1,
        filter={"category": "gen_ai"}
    )
    
    if results:
        doc = results[0]
        logger.info(f"Upserted document (new): {doc.page_content}")
        logger.info(f"Upserted metadata (new): {doc.metadata}")
    
    # 5. Delete operation
    logger.info("\n5. Demonstrating delete operation")
    
    # Count before deletion
    results_before = vectorstore.similarity_search(
        query="CRUD functionality",
        k=10,
        filter={"category": "api"}
    )
    
    logger.info(f"Documents before deletion: {len(results_before)}")
    
    # Delete documents
    vectorstore.delete(filter={"category": "api"})
    
    # Count after deletion
    results_after = vectorstore.similarity_search(
        query="CRUD functionality",
        k=10,
        filter={"category": "api"}
    )
    
    logger.info(f"Documents after deletion: {len(results_after)}")


def main():
    """Main function to run the example."""
    try:
        # Connect to HANA
        connection = connect_to_hana()
        
        # Initialize vector store
        vectorstore = initialize_vectorstore(connection)
        
        # Add sample documents
        add_sample_documents(vectorstore)
        
        # Demonstrate update operations
        demonstrate_update_operations(vectorstore)
        
        logger.info("\nUpdate operations example completed successfully.")
        
    except Exception as e:
        logger.error(f"Error in update operations example: {str(e)}")
    finally:
        if 'connection' in locals():
            connection.close()
            logger.info("Database connection closed.")


if __name__ == "__main__":
    main()