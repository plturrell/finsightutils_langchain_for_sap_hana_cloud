"""
Example: Efficient batch processing with the SAP HANA Cloud LangChain integration.

This example demonstrates how to:
1. Process large document collections efficiently
2. Use dynamic batch sizing based on GPU memory
3. Implement parallel processing with progress tracking
4. Handle errors with automatic retry

Requirements:
- langchain-hana package
- pandas
- tqdm (for progress display)
"""

import os
import time
import pandas as pd
from typing import List, Dict, Any, Optional
from tqdm import tqdm

# Import the SAP HANA Cloud vectorstore
from langchain_hana.vectorstores import HanaVectorStore
from langchain_hana.embeddings import HanaEmbeddings

# For GPU-accelerated embeddings with dynamic batching
from langchain_hana.gpu.batch_processor import EmbeddingBatchProcessor

# Connection details (from environment variables)
HANA_HOST = os.environ.get("HANA_HOST")
HANA_PORT = int(os.environ.get("HANA_PORT", "443"))
HANA_USER = os.environ.get("HANA_USER")
HANA_PASSWORD = os.environ.get("HANA_PASSWORD")

# Vector table details
TABLE_NAME = "VECTOR_STORE"
SCHEMA_NAME = "ML_DATA"


def create_vector_table_if_not_exists():
    """Create the vector store table if it doesn't exist."""
    
    # Connect to HANA
    vectorstore = HanaVectorStore.create_connection(
        host=HANA_HOST,
        port=HANA_PORT,
        user=HANA_USER,
        password=HANA_PASSWORD,
        schema=SCHEMA_NAME,
        table=TABLE_NAME,
    )
    
    # Check if table exists, create if it doesn't
    if not vectorstore.table_exists():
        print(f"Creating vector table {SCHEMA_NAME}.{TABLE_NAME}...")
        vectorstore.create_table()
        print("Table created successfully!")
    else:
        print(f"Table {SCHEMA_NAME}.{TABLE_NAME} already exists.")
    
    return vectorstore


def load_documents_from_csv(csv_path: str) -> List[Dict[str, Any]]:
    """Load documents from a CSV file."""
    
    df = pd.read_csv(csv_path)
    
    # Convert DataFrame to documents
    documents = []
    for _, row in df.iterrows():
        documents.append({
            "page_content": row["content"],
            "metadata": {
                "source": row.get("source", "csv_import"),
                "author": row.get("author", "unknown"),
                "created_at": row.get("created_at", ""),
                "category": row.get("category", ""),
                "id": str(row.get("id", "")),
            }
        })
    
    return documents


def process_documents_in_batches(
    vectorstore: HanaVectorStore,
    documents: List[Dict[str, Any]],
    batch_size: int = 64,
    max_retries: int = 3,
    show_progress: bool = True
):
    """Process documents in batches with error handling and progress tracking."""
    
    # Set up embedding model with TensorRT acceleration
    embeddings = HanaEmbeddings(
        host=HANA_HOST,
        port=HANA_PORT,
        user=HANA_USER,
        password=HANA_PASSWORD,
        use_internal=True,  # Use SAP HANA's internal embedding functionality
    )
    
    # Create batch processor for dynamic batch sizing
    batch_processor = EmbeddingBatchProcessor(
        embedding_fn=embeddings.embed_documents,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim=384,
        initial_batch_size=batch_size,
        min_batch_size=1,
        max_batch_size=128,
        dtype="float32",
        enable_caching=True,
    )
    
    # Set up progress bar if requested
    total_docs = len(documents)
    progress_bar = tqdm(total=total_docs) if show_progress else None
    
    # Process in batches
    start_time = time.time()
    processed_count = 0
    error_count = 0
    
    # Process all documents
    for i in range(0, total_docs, batch_size):
        batch = documents[i:i + batch_size]
        batch_texts = [doc["page_content"] for doc in batch]
        batch_metadatas = [doc["metadata"] for doc in batch]
        
        # Try to process with retries
        for retry in range(max_retries):
            try:
                # Generate embeddings with dynamic batch sizing
                embeddings_result, stats = batch_processor.embed_documents(batch_texts)
                
                # Add to vectorstore
                vectorstore.add_embeddings(
                    texts=batch_texts,
                    embeddings=embeddings_result,
                    metadatas=batch_metadatas,
                )
                
                # Update counts and progress
                processed_count += len(batch)
                if progress_bar:
                    progress_bar.update(len(batch))
                    progress_bar.set_description(
                        f"Processing: {processed_count}/{total_docs} "
                        f"(Batch size: {stats.final_batch_size}, "
                        f"Speed: {stats.items_per_second:.1f} items/s)"
                    )
                
                # Success, break retry loop
                break
                
            except Exception as e:
                print(f"Error processing batch (retry {retry+1}/{max_retries}): {str(e)}")
                
                if retry == max_retries - 1:
                    # Last retry failed
                    error_count += len(batch)
                    if progress_bar:
                        progress_bar.update(len(batch))
                
                # Wait before retry (exponential backoff)
                time.sleep(2 ** retry)
    
    # Close progress bar
    if progress_bar:
        progress_bar.close()
    
    # Calculate stats
    elapsed_time = time.time() - start_time
    docs_per_second = processed_count / elapsed_time if elapsed_time > 0 else 0
    
    print(f"\nProcessing complete!")
    print(f"Total documents: {total_docs}")
    print(f"Successfully processed: {processed_count}")
    print(f"Errors: {error_count}")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Average speed: {docs_per_second:.2f} documents per second")


def filter_by_category(
    vectorstore: HanaVectorStore,
    query: str,
    category: str,
    k: int = 5
):
    """Search with metadata filtering by category."""
    
    # Define filter for the specific category
    filter_dict = {"category": category}
    
    # Perform search
    results = vectorstore.similarity_search(
        query=query,
        k=k,
        filter=filter_dict
    )
    
    print(f"\nSearch results for '{query}' in category '{category}':")
    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"Content: {doc.page_content[:150]}...")
        print(f"Metadata: {doc.metadata}")


def main():
    # Create or connect to vector table
    vectorstore = create_vector_table_if_not_exists()
    
    # Load documents from CSV
    documents = load_documents_from_csv("documents.csv")
    print(f"Loaded {len(documents)} documents from CSV")
    
    # Process documents in batches
    process_documents_in_batches(
        vectorstore=vectorstore,
        documents=documents,
        batch_size=64,
        show_progress=True
    )
    
    # Perform a filtered search
    filter_by_category(
        vectorstore=vectorstore,
        query="What are the features of SAP HANA Cloud?",
        category="product_features"
    )


if __name__ == "__main__":
    main()