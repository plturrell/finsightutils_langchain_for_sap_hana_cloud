"""
Example demonstrating Arrow Flight integration with SAP HANA Cloud.

This example shows how to use the Apache Arrow Flight integration
for high-performance data transfer between SAP HANA Cloud and
GPU-accelerated vector operations.
"""

import argparse
import logging
import os
import time
from typing import Dict, List, Optional, Union

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check for required dependencies
try:
    import pyarrow as pa
    import pyarrow.flight as flight
    HAS_ARROW_FLIGHT = True
except ImportError:
    HAS_ARROW_FLIGHT = False
    logger.warning(
        "The pyarrow and pyarrow.flight packages are required for Arrow Flight integration. "
        "Install them with 'pip install pyarrow pyarrow.flight'."
    )

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning(
        "PyTorch is recommended for GPU acceleration. "
        "Install it with 'pip install torch'."
    )

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    logger.warning(
        "The sentence_transformers package is required for embedding generation. "
        "Install it with 'pip install sentence-transformers'."
    )

# Check if the Arrow Flight implementation is available
try:
    from langchain_hana.gpu import (
        ArrowFlightClient,
        HanaArrowFlightServer,
        start_arrow_flight_server,
        ArrowGpuMemoryManager,
        HanaArrowFlightVectorStore
    )
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.docstore.document import Document
    
    HAS_IMPLEMENTATION = True
except ImportError:
    HAS_IMPLEMENTATION = False
    logger.warning(
        "The Arrow Flight implementation could not be imported. "
        "Make sure the langchain_hana package is installed."
    )


def check_prerequisites():
    """Check if all prerequisites are met."""
    if not HAS_ARROW_FLIGHT:
        logger.error("PyArrow and PyArrow.Flight are required for this example.")
        return False
        
    if not HAS_TORCH:
        logger.warning("PyTorch is recommended for GPU acceleration.")
        
    if not HAS_SENTENCE_TRANSFORMERS:
        logger.error("SentenceTransformer is required for embedding generation.")
        return False
        
    if not HAS_IMPLEMENTATION:
        logger.error("Arrow Flight implementation is not available.")
        return False
        
    return True


def run_client_example(
    host: str = "localhost",
    port: int = 8815,
    table_name: str = "ARROW_FLIGHT_DEMO",
    username: Optional[str] = None,
    password: Optional[str] = None,
    use_tls: bool = False,
    device_id: int = 0
):
    """
    Run an example demonstrating the ArrowFlightClient.
    
    Args:
        host: SAP HANA host address
        port: Arrow Flight server port
        table_name: Table name for the demo
        username: SAP HANA username
        password: SAP HANA password
        use_tls: Whether to use TLS for secure connections
        device_id: GPU device ID
    """
    logger.info("Starting Arrow Flight client example")
    
    # Initialize client
    client = ArrowFlightClient(
        host=host,
        port=port,
        use_tls=use_tls,
        username=username,
        password=password
    )
    
    # Initialize GPU memory manager
    memory_manager = ArrowGpuMemoryManager(
        device_id=device_id,
        batch_size=1024
    )
    
    # Generate sample vectors
    dim = 768  # Common embedding dimension
    num_vectors = 1000
    
    logger.info(f"Generating {num_vectors} sample vectors with dimension {dim}")
    
    # Generate random vectors
    vectors = np.random.randn(num_vectors, dim).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)  # Normalize
    
    # Generate sample texts and metadata
    texts = [f"Sample document {i}" for i in range(num_vectors)]
    metadatas = [{"index": i, "source": "generated"} for i in range(num_vectors)]
    
    # Benchmark upload performance
    logger.info("Benchmarking vector upload performance...")
    
    start_time = time.time()
    ids = client.upload_vectors(
        table_name=table_name,
        vectors=vectors.tolist(),
        texts=texts,
        metadata=metadatas,
        batch_size=100
    )
    upload_time = time.time() - start_time
    
    logger.info(f"Uploaded {num_vectors} vectors in {upload_time:.2f} seconds "
                f"({num_vectors/upload_time:.2f} vectors/second)")
    
    # Benchmark search performance
    logger.info("Benchmarking similarity search performance...")
    
    # Generate a query vector
    query_vector = np.random.randn(dim).astype(np.float32)
    query_vector = query_vector / np.linalg.norm(query_vector)  # Normalize
    
    num_searches = 10
    k = 10
    
    start_time = time.time()
    for _ in range(num_searches):
        results = client.similarity_search(
            table_name=table_name,
            query_vector=query_vector.tolist(),
            k=k,
            include_vectors=True,
            distance_strategy="cosine"
        )
    search_time = time.time() - start_time
    
    logger.info(f"Performed {num_searches} searches in {search_time:.2f} seconds "
                f"({num_searches/search_time:.2f} searches/second)")
    
    # Demonstrate GPU-accelerated batch similarity search
    if HAS_TORCH and torch.cuda.is_available():
        logger.info("Demonstrating GPU-accelerated batch similarity search...")
        
        # Generate batch of query vectors
        num_queries = 100
        query_batch = np.random.randn(num_queries, dim).astype(np.float32)
        query_batch = query_batch / np.linalg.norm(query_batch, axis=1, keepdims=True)
        
        # Convert to PyArrow FixedSizeListArray
        query_vectors = memory_manager.vectors_to_fixed_size_list_array(query_batch)
        stored_vectors = memory_manager.vectors_to_fixed_size_list_array(vectors)
        
        start_time = time.time()
        distances, indices = memory_manager.batch_similarity_search(
            query_vectors=query_vectors,
            stored_vectors=stored_vectors,
            k=k,
            metric="cosine"
        )
        gpu_search_time = time.time() - start_time
        
        logger.info(f"Performed {num_queries} GPU-accelerated searches in {gpu_search_time:.2f} seconds "
                    f"({num_queries/gpu_search_time:.2f} searches/second)")
    
    # Cleanup
    logger.info("Cleaning up resources...")
    client.close()
    memory_manager.cleanup()
    
    logger.info("Arrow Flight client example completed successfully")


def run_vectorstore_example(
    host: str = "localhost",
    port: int = 8815,
    table_name: str = "ARROW_FLIGHT_VECTORSTORE",
    username: Optional[str] = None,
    password: Optional[str] = None,
    use_tls: bool = False,
    device_id: int = 0,
    model_name: str = "all-MiniLM-L6-v2"
):
    """
    Run an example demonstrating the HanaArrowFlightVectorStore.
    
    Args:
        host: SAP HANA host address
        port: Arrow Flight server port
        table_name: Table name for the demo
        username: SAP HANA username
        password: SAP HANA password
        use_tls: Whether to use TLS for secure connections
        device_id: GPU device ID
        model_name: Name of the embedding model
    """
    logger.info("Starting Arrow Flight vectorstore example")
    
    # Initialize embedding model
    logger.info(f"Initializing embedding model: {model_name}")
    
    embed_kwargs = {}
    if HAS_TORCH and torch.cuda.is_available():
        embed_kwargs["device"] = f"cuda:{device_id}"
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": embed_kwargs.get("device", "cpu")},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    # Initialize vectorstore
    logger.info(f"Initializing Arrow Flight vectorstore: {table_name}")
    
    vectorstore = HanaArrowFlightVectorStore(
        embedding=embeddings,
        host=host,
        port=port,
        table_name=table_name,
        username=username,
        password=password,
        use_tls=use_tls,
        device_id=device_id,
        batch_size=100,
        pre_delete_collection=True  # For demo purposes
    )
    
    # Create sample documents
    logger.info("Creating sample documents")
    
    num_docs = 100
    documents = [
        Document(
            page_content=f"This is a sample document {i} for testing Arrow Flight integration.",
            metadata={"index": i, "source": "generated", "category": f"category_{i % 5}"}
        )
        for i in range(num_docs)
    ]
    
    # Add documents to vectorstore
    logger.info(f"Adding {num_docs} documents to vectorstore")
    
    start_time = time.time()
    vectorstore.add_documents(documents)
    add_time = time.time() - start_time
    
    logger.info(f"Added {num_docs} documents in {add_time:.2f} seconds "
                f"({num_docs/add_time:.2f} documents/second)")
    
    # Perform similarity search
    logger.info("Performing similarity search")
    
    query = "sample document for testing"
    
    start_time = time.time()
    results = vectorstore.similarity_search(query, k=5)
    search_time = time.time() - start_time
    
    logger.info(f"Performed similarity search in {search_time:.2f} seconds")
    logger.info(f"Top result: {results[0].page_content}")
    
    # Perform similarity search with metadata filter
    logger.info("Performing similarity search with metadata filter")
    
    filter_dict = {"category": "category_2"}
    
    start_time = time.time()
    filtered_results = vectorstore.similarity_search(query, k=5, filter=filter_dict)
    filter_search_time = time.time() - start_time
    
    logger.info(f"Performed filtered similarity search in {filter_search_time:.2f} seconds")
    logger.info(f"Top filtered result: {filtered_results[0].page_content}")
    
    # Perform MMR search
    logger.info("Performing MMR search for diverse results")
    
    start_time = time.time()
    mmr_results = vectorstore.max_marginal_relevance_search(
        query, k=5, fetch_k=20, lambda_mult=0.5
    )
    mmr_time = time.time() - start_time
    
    logger.info(f"Performed MMR search in {mmr_time:.2f} seconds")
    logger.info(f"Top MMR result: {mmr_results[0].page_content}")
    
    # Update a document
    logger.info("Updating a document")
    
    # Get an existing document ID
    existing_docs = vectorstore.get([vectorstore.similarity_search(query, k=1)[0].metadata["index"]])
    doc_id = list(existing_docs.keys())[0]
    
    # Create updated document
    updated_doc = Document(
        page_content=f"This is an UPDATED document with ID {doc_id}",
        metadata={"index": int(doc_id), "source": "updated", "category": "updated"}
    )
    
    # Update document
    start_time = time.time()
    vectorstore.update_document(doc_id, updated_doc)
    update_time = time.time() - start_time
    
    logger.info(f"Updated document in {update_time:.2f} seconds")
    
    # Search for updated document
    updated_results = vectorstore.similarity_search("UPDATED document", k=1)
    logger.info(f"Updated document found: {updated_results[0].page_content}")
    
    # Cleanup
    logger.info("Cleaning up resources")
    
    # Close vectorstore (releases connections)
    vectorstore.aclose()
    
    logger.info("Arrow Flight vectorstore example completed successfully")


def run_arrow_flight_integration_example(
    mode: str = "client",
    host: str = "localhost",
    port: int = 8815,
    table_name: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    use_tls: bool = False,
    device_id: int = 0,
    model_name: str = "all-MiniLM-L6-v2"
):
    """
    Run Arrow Flight integration example.
    
    Args:
        mode: Example mode ("client" or "vectorstore")
        host: SAP HANA host address
        port: Arrow Flight server port
        table_name: Table name for the demo
        username: SAP HANA username
        password: SAP HANA password
        use_tls: Whether to use TLS for secure connections
        device_id: GPU device ID
        model_name: Name of the embedding model
    """
    # Check prerequisites
    if not check_prerequisites():
        logger.error("Prerequisites not met. Exiting.")
        return
    
    # Set default table name based on mode
    if table_name is None:
        table_name = f"ARROW_FLIGHT_{mode.upper()}_DEMO"
    
    # Run example based on mode
    if mode == "client":
        run_client_example(
            host=host,
            port=port,
            table_name=table_name,
            username=username,
            password=password,
            use_tls=use_tls,
            device_id=device_id
        )
    elif mode == "vectorstore":
        run_vectorstore_example(
            host=host,
            port=port,
            table_name=table_name,
            username=username,
            password=password,
            use_tls=use_tls,
            device_id=device_id,
            model_name=model_name
        )
    else:
        logger.error(f"Invalid mode: {mode}. Must be 'client' or 'vectorstore'.")


def main():
    """Parse command line arguments and run example."""
    parser = argparse.ArgumentParser(description="Arrow Flight Integration Example")
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["client", "vectorstore"],
        default="client",
        help="Example mode ('client' or 'vectorstore')"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="SAP HANA host address"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8815,
        help="Arrow Flight server port"
    )
    parser.add_argument(
        "--table-name",
        type=str,
        help="Table name for the demo"
    )
    parser.add_argument(
        "--username",
        type=str,
        help="SAP HANA username"
    )
    parser.add_argument(
        "--password",
        type=str,
        help="SAP HANA password"
    )
    parser.add_argument(
        "--use-tls",
        action="store_true",
        help="Use TLS for secure connections"
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="GPU device ID"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Name of the embedding model"
    )
    
    args = parser.parse_args()
    
    # Run example with parsed arguments
    run_arrow_flight_integration_example(
        mode=args.mode,
        host=args.host,
        port=args.port,
        table_name=args.table_name,
        username=args.username,
        password=args.password,
        use_tls=args.use_tls,
        device_id=args.device_id,
        model_name=args.model_name
    )


if __name__ == "__main__":
    main()