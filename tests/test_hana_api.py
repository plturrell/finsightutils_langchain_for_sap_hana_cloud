#!/usr/bin/env python3
import requests
import json
import time
import sys

# Base URL for the deployed API
BASE_URL = "https://jupyter0-513syzm60.brevlab.com"

def test_health():
    """Test the health endpoint of the API"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        print(f"Health check status: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {response.json()}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"Connection error: {e}")
        return False

def test_embedding_generation():
    """Test the embedding generation endpoint"""
    try:
        payload = {
            "texts": ["This is a test sentence for embedding generation."]
        }
        response = requests.post(
            f"{BASE_URL}/api/embeddings", 
            json=payload,
            timeout=30
        )
        print(f"Embedding generation status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Embedding dimensions: {len(result['embeddings'][0])}")
            print(f"Processing time: {result.get('processing_time', 'N/A')} ms")
            if 'gpu_used' in result:
                print(f"GPU used: {result['gpu_used']}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"Connection error: {e}")
        return False

def test_similarity_search():
    """Test the similarity search endpoint"""
    try:
        payload = {
            "query": "How does SAP HANA Cloud integrate with LLMs?",
            "k": 3
        }
        response = requests.post(
            f"{BASE_URL}/api/search", 
            json=payload,
            timeout=30
        )
        print(f"Similarity search status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Number of results: {len(result['results'])}")
            if result['results']:
                print(f"First result snippet: {result['results'][0]['content'][:100]}...")
            print(f"Processing time: {result.get('processing_time', 'N/A')} ms")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"Connection error: {e}")
        return False

def test_mmr_search():
    """Test the MMR search endpoint"""
    try:
        payload = {
            "query": "Vector storage in SAP HANA Cloud",
            "k": 3,
            "lambda_mult": 0.7
        }
        response = requests.post(
            f"{BASE_URL}/api/mmr_search", 
            json=payload,
            timeout=30
        )
        print(f"MMR search status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Number of results: {len(result['results'])}")
            if result['results']:
                print(f"First result snippet: {result['results'][0]['content'][:100]}...")
            print(f"Processing time: {result.get('processing_time', 'N/A')} ms")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"Connection error: {e}")
        return False

def test_gpu_performance():
    """Test GPU performance with a batch of embeddings"""
    try:
        # Generate a larger batch to test GPU performance
        texts = [f"This is test sentence {i} for GPU performance testing." for i in range(100)]
        payload = {"texts": texts}
        
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/api/embeddings", 
            json=payload,
            timeout=60
        )
        end_time = time.time()
        
        print(f"GPU performance test status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            total_time = (end_time - start_time) * 1000  # Convert to ms
            print(f"Total request time: {total_time:.2f} ms")
            print(f"API reported processing time: {result.get('processing_time', 'N/A')} ms")
            print(f"Embeddings per second: {len(texts) / ((end_time - start_time)):.2f}")
            if 'gpu_used' in result:
                print(f"GPU used: {result['gpu_used']}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"Connection error: {e}")
        return False

def run_all_tests():
    """Run all tests and return the number of failures"""
    tests = [
        ("Health Check", test_health),
        ("Embedding Generation", test_embedding_generation),
        ("Similarity Search", test_similarity_search),
        ("MMR Search", test_mmr_search),
        ("GPU Performance", test_gpu_performance)
    ]
    
    failures = 0
    for name, test_func in tests:
        print(f"\n===== Testing {name} =====")
        try:
            result = test_func()
            if not result:
                failures += 1
                print(f"❌ {name} test failed")
            else:
                print(f"✅ {name} test passed")
        except Exception as e:
            failures += 1
            print(f"❌ {name} test failed with exception: {e}")
            
    return failures

if __name__ == "__main__":
    print("Starting tests for SAP HANA Cloud LangChain Integration")
    failures = run_all_tests()
    print(f"\nTest summary: {5 - failures}/5 tests passed")
    sys.exit(failures)