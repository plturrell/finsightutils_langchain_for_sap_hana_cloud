#!/usr/bin/env python3
"""
Load testing script for SAP HANA Cloud LangChain integration API deployed on T4 GPU.

This script simulates concurrent requests to the API endpoints to evaluate
performance under load.

Usage:
    python load_test.py --url https://jupyter0-513syzm60.brevlab.com --endpoint embeddings --concurrent 10 --duration 60
"""

import argparse
import asyncio
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Optional, Tuple
import requests
import numpy as np

# Sample texts for embedding generation
SAMPLE_TEXTS = [
    "SAP HANA Cloud provides vector search capabilities for efficient similarity matching.",
    "The integration with LangChain enables developers to build powerful RAG applications.",
    "T4 GPUs offer significant performance improvements for embedding generation.",
    "TensorRT optimization can accelerate vector operations by up to 5x compared to CPU.",
    "SAP HANA Vector Engine supports HNSW indexing for fast approximate nearest neighbor search.",
    "Maximal Marginal Relevance (MMR) search helps ensure diversity in search results.",
    "SAP HANA Graph Engine enables knowledge graph querying through SPARQL.",
    "The T4 GPU has 16GB of GDDR6 memory and 2,560 CUDA cores for parallel processing.",
    "INT8 precision can provide the best performance but may sacrifice some accuracy.",
    "FP16 precision offers a good balance between performance and accuracy for most use cases."
]

# Sample queries for search operations
SAMPLE_QUERIES = [
    "How does vector search work in SAP HANA Cloud?",
    "What are the benefits of using T4 GPU for embeddings?",
    "How can I optimize TensorRT for best performance?",
    "What is the difference between similarity search and MMR search?",
    "How do I integrate LangChain with SAP HANA Cloud?",
    "What precision mode is best for T4 GPU?",
    "How much memory does a T4 GPU have?",
    "Can SAP HANA Cloud work with knowledge graphs?",
    "What batch size is optimal for T4 GPU?",
    "How do I handle errors in the SAP HANA Cloud LangChain integration?"
]

class LoadTester:
    """Class for load testing the SAP HANA Cloud LangChain integration API"""
    
    def __init__(
        self, 
        base_url: str,
        endpoint: str = "embeddings",
        concurrent: int = 5,
        duration: int = 60,
        auth_username: Optional[str] = None,
        auth_password: Optional[str] = None
    ):
        """
        Initialize the load tester
        
        Args:
            base_url: Base URL of the API
            endpoint: API endpoint to test
            concurrent: Number of concurrent requests
            duration: Test duration in seconds
            auth_username: Optional username for authentication
            auth_password: Optional password for authentication
        """
        self.base_url = base_url.rstrip("/")
        self.endpoint = endpoint
        self.concurrent = concurrent
        self.duration = duration
        self.auth = (auth_username, auth_password) if auth_username and auth_password else None
        
        # Results tracking
        self.results = {
            "request_count": 0,
            "success_count": 0,
            "error_count": 0,
            "response_times": [],
            "start_time": None,
            "end_time": None,
            "requests_per_second": 0.0,
            "avg_response_time": 0.0,
            "min_response_time": 0.0,
            "max_response_time": 0.0,
            "p50_response_time": 0.0,
            "p95_response_time": 0.0,
            "p99_response_time": 0.0
        }
    
    def run(self):
        """Run the load test"""
        print(f"Starting load test for {self.endpoint} endpoint")
        print(f"Base URL: {self.base_url}")
        print(f"Concurrent requests: {self.concurrent}")
        print(f"Test duration: {self.duration} seconds")
        print()
        
        self.results["start_time"] = time.time()
        
        # Create thread pool for concurrent requests
        with ThreadPoolExecutor(max_workers=self.concurrent) as executor:
            futures = []
            end_time = time.time() + self.duration
            
            # Submit initial batch of requests
            for _ in range(self.concurrent):
                futures.append(executor.submit(self._make_request))
            
            # Keep submitting requests until duration is reached
            while time.time() < end_time:
                # Check for completed futures
                done_futures = [f for f in futures if f.done()]
                futures = [f for f in futures if not f.done()]
                
                # Submit new requests to replace completed ones
                for _ in range(len(done_futures)):
                    if time.time() < end_time:
                        futures.append(executor.submit(self._make_request))
                
                # Sleep briefly to avoid CPU spinning
                time.sleep(0.01)
            
            # Wait for remaining futures to complete
            for f in futures:
                f.result()
        
        self.results["end_time"] = time.time()
        
        # Calculate statistics
        test_duration = self.results["end_time"] - self.results["start_time"]
        self.results["requests_per_second"] = self.results["request_count"] / test_duration
        
        if self.results["response_times"]:
            self.results["avg_response_time"] = np.mean(self.results["response_times"])
            self.results["min_response_time"] = np.min(self.results["response_times"])
            self.results["max_response_time"] = np.max(self.results["response_times"])
            self.results["p50_response_time"] = np.percentile(self.results["response_times"], 50)
            self.results["p95_response_time"] = np.percentile(self.results["response_times"], 95)
            self.results["p99_response_time"] = np.percentile(self.results["response_times"], 99)
        
        self._print_results()
        return self.results
    
    def _make_request(self):
        """Make a single request to the API endpoint"""
        url = f"{self.base_url}/api/{self.endpoint}"
        headers = {"Content-Type": "application/json"}
        payload = self._get_payload_for_endpoint()
        
        try:
            start_time = time.time()
            response = requests.post(url, json=payload, headers=headers, auth=self.auth, timeout=30)
            response_time = time.time() - start_time
            
            self.results["request_count"] += 1
            self.results["response_times"].append(response_time * 1000)  # Convert to ms
            
            if 200 <= response.status_code < 300:
                self.results["success_count"] += 1
            else:
                self.results["error_count"] += 1
                print(f"Error response: {response.status_code} - {response.text[:100]}")
                
            return response.status_code, response_time
            
        except Exception as e:
            self.results["request_count"] += 1
            self.results["error_count"] += 1
            print(f"Request error: {str(e)}")
            return None, 0
    
    def _get_payload_for_endpoint(self) -> Dict[str, Any]:
        """Get the appropriate payload for the specified endpoint"""
        if self.endpoint == "embeddings":
            # Random number of texts, between 1 and 10
            num_texts = random.randint(1, 10)
            return {
                "texts": random.sample(SAMPLE_TEXTS, num_texts),
                "use_tensorrt": True
            }
        elif self.endpoint == "vectorstore/search":
            return {
                "query": random.choice(SAMPLE_QUERIES),
                "k": random.randint(3, 10),
                "table_name": "T4_TEST_VECTORS",
                "filter": {"category": "technical"} if random.random() > 0.5 else None
            }
        elif self.endpoint == "vectorstore/mmr_search":
            return {
                "query": random.choice(SAMPLE_QUERIES),
                "k": random.randint(3, 10),
                "table_name": "T4_TEST_VECTORS",
                "use_mmr": True,
                "lambda_mult": random.uniform(0.3, 0.9)
            }
        else:
            # Default payload
            return {"query": "test"}
    
    def _print_results(self):
        """Print the test results"""
        print("\nLoad Test Results:")
        print(f"Endpoint: {self.endpoint}")
        print(f"Duration: {self.duration} seconds")
        print(f"Concurrent requests: {self.concurrent}")
        print(f"Total requests: {self.results['request_count']}")
        print(f"Successful requests: {self.results['success_count']}")
        print(f"Failed requests: {self.results['error_count']}")
        print(f"Requests per second: {self.results['requests_per_second']:.2f}")
        print(f"Average response time: {self.results['avg_response_time']:.2f} ms")
        print(f"Min response time: {self.results['min_response_time']:.2f} ms")
        print(f"Max response time: {self.results['max_response_time']:.2f} ms")
        print(f"50th percentile (median) response time: {self.results['p50_response_time']:.2f} ms")
        print(f"95th percentile response time: {self.results['p95_response_time']:.2f} ms")
        print(f"99th percentile response time: {self.results['p99_response_time']:.2f} ms")
        print()
        
        # Save results to file
        with open(f"load_test_{self.endpoint}_{self.concurrent}_{int(time.time())}.json", "w") as f:
            json.dump(self.results, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Load testing for SAP HANA Cloud LangChain integration API")
    parser.add_argument("--url", type=str, default="https://jupyter0-513syzm60.brevlab.com",
                        help="Base URL of the API")
    parser.add_argument("--endpoint", type=str, default="embeddings",
                        choices=["embeddings", "vectorstore/search", "vectorstore/mmr_search"],
                        help="API endpoint to test")
    parser.add_argument("--concurrent", type=int, default=5,
                        help="Number of concurrent requests")
    parser.add_argument("--duration", type=int, default=60,
                        help="Test duration in seconds")
    parser.add_argument("--username", type=str, help="Authentication username")
    parser.add_argument("--password", type=str, help="Authentication password")
    args = parser.parse_args()
    
    tester = LoadTester(
        base_url=args.url,
        endpoint=args.endpoint,
        concurrent=args.concurrent,
        duration=args.duration,
        auth_username=args.username,
        auth_password=args.password
    )
    
    tester.run()

if __name__ == "__main__":
    main()