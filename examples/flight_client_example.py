"""
Example client for the Arrow Flight API.

This script demonstrates how to use the Arrow Flight API for
high-performance vector data transfer.
"""

import os
import json
import argparse
import numpy as np
import time
import pyarrow as pa
import pyarrow.flight as flight
import requests
from typing import Dict, List, Any, Optional

def get_flight_info(api_base_url: str = "http://localhost:8000") -> Dict[str, Any]:
    """Get Flight server information."""
    response = requests.get(f"{api_base_url}/flight/info")
    return response.json()

def list_collections(api_base_url: str = "http://localhost:8000") -> List[Dict[str, Any]]:
    """List available vector collections."""
    response = requests.get(f"{api_base_url}/flight/list")
    return response.json()["collections"]

def query_vectors(
    table_name: str, 
    filter_dict: Optional[Dict[str, Any]] = None,
    limit: int = 1000,
    offset: int = 0,
    api_base_url: str = "http://localhost:8000"
) -> pa.Table:
    """
    Query vectors using Arrow Flight protocol.
    
    Args:
        table_name: Name of the vector table
        filter_dict: Optional filter criteria
        limit: Maximum number of vectors to retrieve
        offset: Offset for pagination
        api_base_url: Base URL of the API
        
    Returns:
        Arrow Table containing the query results
    """
    # Create query request
    query_request = {
        "table_name": table_name,
        "filter": filter_dict,
        "limit": limit,
        "offset": offset
    }
    
    # Get ticket and location from API
    response = requests.post(f"{api_base_url}/flight/query", json=query_request)
    query_response = response.json()
    
    ticket = query_response["ticket"]
    location = query_response["location"]
    
    # Connect to Flight server
    client = flight.connect(location)
    
    # Create ticket object
    flight_ticket = flight.Ticket(ticket.encode())
    
    # Get reader for the ticket
    reader = client.do_get(flight_ticket)
    
    # Read all data
    table = reader.read_all()
    
    return table

def upload_vectors(
    table_name: str,
    documents: List[str],
    metadatas: List[Dict[str, Any]],
    vectors: List[List[float]],
    create_if_not_exists: bool = True,
    api_base_url: str = "http://localhost:8000"
) -> bool:
    """
    Upload vectors using Arrow Flight protocol.
    
    Args:
        table_name: Name of the vector table
        documents: List of document texts
        metadatas: List of metadata dictionaries
        vectors: List of vector embeddings
        create_if_not_exists: Whether to create the table if it doesn't exist
        api_base_url: Base URL of the API
        
    Returns:
        True if upload was successful
    """
    # Create upload request
    upload_request = {
        "table_name": table_name,
        "create_if_not_exists": create_if_not_exists
    }
    
    # Get descriptor and location from API
    response = requests.post(f"{api_base_url}/flight/upload", json=upload_request)
    upload_response = response.json()
    
    descriptor_str = upload_response["descriptor"]
    location = upload_response["location"]
    
    # Connect to Flight server
    client = flight.connect(location)
    
    # Create descriptor object
    flight_descriptor = flight.FlightDescriptor.for_command(descriptor_str.encode())
    
    # Create Arrow arrays from the data
    arrays = [
        pa.array(documents, type=pa.string()),
        pa.array([json.dumps(m) for m in metadatas], type=pa.string()),
        pa.array(vectors, type=pa.list_(pa.float32()))
    ]
    
    # Create Arrow table
    table = pa.Table.from_arrays(arrays, names=['document', 'metadata', 'vector'])
    
    # Create writer for the descriptor
    writer, _ = client.do_put(flight_descriptor, table.schema)
    
    # Write the table
    writer.write_table(table)
    writer.close()
    
    return True

def benchmark_flight_vs_rest(
    table_name: str, 
    num_vectors: int,
    vector_dim: int,
    api_base_url: str = "http://localhost:8000"
) -> Dict[str, Any]:
    """
    Benchmark Arrow Flight vs REST API for vector transfer.
    
    Args:
        table_name: Name of the vector table
        num_vectors: Number of vectors to use in the benchmark
        vector_dim: Dimension of the vectors
        api_base_url: Base URL of the API
        
    Returns:
        Dictionary with benchmark results
    """
    # Generate random vectors
    np.random.seed(42)
    vectors = [np.random.rand(vector_dim).tolist() for _ in range(num_vectors)]
    documents = [f"Document {i}" for i in range(num_vectors)]
    metadatas = [{"id": i, "benchmark": True} for i in range(num_vectors)]
    
    # PART 1: Upload benchmark
    
    # Upload using Flight
    flight_upload_start = time.time()
    upload_vectors(
        table_name=f"{table_name}_flight",
        documents=documents,
        metadatas=metadatas,
        vectors=vectors,
        api_base_url=api_base_url
    )
    flight_upload_time = time.time() - flight_upload_start
    
    # Upload using REST
    rest_upload_start = time.time()
    for i in range(num_vectors):
        requests.post(
            f"{api_base_url}/api/search",
            json={
                "query": documents[i],
                "metadata": metadatas[i],
                "vector": vectors[i]
            }
        )
    rest_upload_time = time.time() - rest_upload_start
    
    # PART 2: Query benchmark
    
    # Query using Flight
    flight_query_start = time.time()
    flight_results = query_vectors(
        table_name=f"{table_name}_flight",
        filter_dict={"benchmark": True},
        limit=num_vectors,
        api_base_url=api_base_url
    )
    flight_query_time = time.time() - flight_query_start
    
    # Query using REST
    rest_query_start = time.time()
    rest_response = requests.post(
        f"{api_base_url}/api/search",
        json={"query": "benchmark", "k": num_vectors}
    )
    rest_query_time = time.time() - rest_query_start
    
    # Compile results
    return {
        "num_vectors": num_vectors,
        "vector_dim": vector_dim,
        "upload": {
            "flight": {
                "time_seconds": flight_upload_time,
                "vectors_per_second": num_vectors / flight_upload_time
            },
            "rest": {
                "time_seconds": rest_upload_time,
                "vectors_per_second": num_vectors / rest_upload_time
            },
            "speedup": rest_upload_time / flight_upload_time
        },
        "query": {
            "flight": {
                "time_seconds": flight_query_time,
                "vectors_per_second": num_vectors / flight_query_time
            },
            "rest": {
                "time_seconds": rest_query_time,
                "vectors_per_second": num_vectors / rest_query_time
            },
            "speedup": rest_query_time / flight_query_time
        }
    }

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Arrow Flight client example")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--table", default="test_vectors", help="Vector table name")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--num-vectors", type=int, default=1000, help="Number of vectors for benchmark")
    parser.add_argument("--vector-dim", type=int, default=384, help="Vector dimension for benchmark")
    
    args = parser.parse_args()
    
    # Ensure Flight server is running
    try:
        info = get_flight_info(args.api_url)
        print(f"Flight server status: {info['status']}")
        print(f"Flight server location: {info['location']}")
    except Exception as e:
        print(f"Error connecting to Flight server: {str(e)}")
        print("Make sure the API server is running and the Flight server is started.")
        return
    
    # List available collections
    try:
        collections = list_collections(args.api_url)
        print(f"Available collections: {len(collections)}")
        for collection in collections:
            print(f"  - {collection['name']} ({collection['total_records']} vectors)")
    except Exception as e:
        print(f"Error listing collections: {str(e)}")
    
    # Run benchmark if requested
    if args.benchmark:
        print(f"\nRunning benchmark with {args.num_vectors} vectors of dimension {args.vector_dim}...")
        results = benchmark_flight_vs_rest(
            table_name=args.table,
            num_vectors=args.num_vectors,
            vector_dim=args.vector_dim,
            api_base_url=args.api_url
        )
        
        print("\nBenchmark Results:")
        print(f"Vectors: {results['num_vectors']}")
        print(f"Dimension: {results['vector_dim']}")
        print("\nUpload Performance:")
        print(f"  Flight: {results['upload']['flight']['time_seconds']:.4f}s ({results['upload']['flight']['vectors_per_second']:.2f} vectors/s)")
        print(f"  REST: {results['upload']['rest']['time_seconds']:.4f}s ({results['upload']['rest']['vectors_per_second']:.2f} vectors/s)")
        print(f"  Speedup: {results['upload']['speedup']:.2f}x")
        print("\nQuery Performance:")
        print(f"  Flight: {results['query']['flight']['time_seconds']:.4f}s ({results['query']['flight']['vectors_per_second']:.2f} vectors/s)")
        print(f"  REST: {results['query']['rest']['time_seconds']:.4f}s ({results['query']['rest']['vectors_per_second']:.2f} vectors/s)")
        print(f"  Speedup: {results['query']['speedup']:.2f}x")

if __name__ == "__main__":
    main()