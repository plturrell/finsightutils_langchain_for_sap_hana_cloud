"""
Test script for the standardized API with versioned routers.

This script tests endpoints from both v1 and v2 to ensure the versioned router structure
is working correctly.
"""

import requests
import json
import time
import os
import sys
from typing import Dict, Any, List, Optional

# Configuration
API_BASE_URL = "http://localhost:8000"
API_TIMEOUT = 10  # seconds

# Test endpoints
TEST_ENDPOINTS = [
    # v1 endpoints
    {"method": "GET", "path": "/api/v1/health", "expected_status": 200},
    {"method": "GET", "path": "/api/v1/health/ping", "expected_status": 200},
    {"method": "GET", "path": "/api/v1/health/status", "expected_status": 200},
    {"method": "GET", "path": "/api/v1/financial-embeddings/status", "expected_status": 200},
    {"method": "POST", "path": "/api/v1/optimization/optimized-hyperparameters", 
     "payload": {
         "model_size": 10000000,
         "batch_size": 32,
         "dataset_size": 50000,
         "embedding_dimension": 768,
         "vocabulary_size": 30000,
         "max_sequence_length": 512
     },
     "expected_status": 200},
    
    # v2 endpoints
    {"method": "GET", "path": "/api/v2/health", "expected_status": 200},
    {"method": "GET", "path": "/api/v2/health/status", "expected_status": 200},
    {"method": "GET", "path": "/api/v2/gpu/info", "expected_status": 200},
    {"method": "GET", "path": "/api/v2/tensorrt/models", "expected_status": 200},
]


def test_endpoint(endpoint_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Test a single API endpoint.
    
    Args:
        endpoint_config: Endpoint configuration with method, path, payload, and expected_status
        
    Returns:
        Dict[str, Any]: Test result with status, response, and elapsed time
    """
    method = endpoint_config["method"]
    path = endpoint_config["path"]
    payload = endpoint_config.get("payload")
    expected_status = endpoint_config["expected_status"]
    
    url = f"{API_BASE_URL}{path}"
    
    try:
        start_time = time.time()
        
        if method == "GET":
            response = requests.get(url, timeout=API_TIMEOUT)
        elif method == "POST":
            response = requests.post(url, json=payload, timeout=API_TIMEOUT)
        elif method == "PUT":
            response = requests.put(url, json=payload, timeout=API_TIMEOUT)
        elif method == "DELETE":
            response = requests.delete(url, json=payload, timeout=API_TIMEOUT)
        else:
            return {
                "endpoint": path,
                "method": method,
                "status": "error",
                "message": f"Unsupported method: {method}",
                "elapsed_time": 0
            }
        
        elapsed_time = time.time() - start_time
        
        # Get response content
        try:
            response_json = response.json()
        except:
            response_json = {"raw": response.text}
        
        # Check if status code matches expected
        if response.status_code == expected_status:
            result = {
                "endpoint": path,
                "method": method,
                "status": "success",
                "status_code": response.status_code,
                "response": response_json,
                "elapsed_time": elapsed_time
            }
        else:
            result = {
                "endpoint": path,
                "method": method,
                "status": "fail",
                "expected_status": expected_status,
                "actual_status": response.status_code,
                "response": response_json,
                "elapsed_time": elapsed_time
            }
    except Exception as e:
        result = {
            "endpoint": path,
            "method": method,
            "status": "error",
            "message": str(e),
            "elapsed_time": time.time() - start_time
        }
    
    return result


def run_tests() -> Dict[str, Any]:
    """
    Run all API endpoint tests.
    
    Returns:
        Dict[str, Any]: Test results with summary and details
    """
    results = []
    success_count = 0
    fail_count = 0
    error_count = 0
    
    for endpoint_config in TEST_ENDPOINTS:
        result = test_endpoint(endpoint_config)
        results.append(result)
        
        if result["status"] == "success":
            success_count += 1
        elif result["status"] == "fail":
            fail_count += 1
        else:
            error_count += 1
    
    return {
        "summary": {
            "total": len(TEST_ENDPOINTS),
            "success": success_count,
            "fail": fail_count,
            "error": error_count,
            "success_rate": success_count / len(TEST_ENDPOINTS) if len(TEST_ENDPOINTS) > 0 else 0
        },
        "results": results
    }


def print_results(test_results: Dict[str, Any]) -> None:
    """
    Print test results in a formatted way.
    
    Args:
        test_results: Test results from run_tests()
    """
    summary = test_results["summary"]
    results = test_results["results"]
    
    print("\n====== API TEST RESULTS ======\n")
    print(f"Total endpoints tested: {summary['total']}")
    print(f"Success: {summary['success']}")
    print(f"Fail: {summary['fail']}")
    print(f"Error: {summary['error']}")
    print(f"Success rate: {summary['success_rate'] * 100:.1f}%\n")
    
    print("Detailed Results:")
    for result in results:
        status_symbol = "✅" if result["status"] == "success" else "❌"
        print(f"{status_symbol} {result['method']} {result['endpoint']}")
        
        if result["status"] == "success":
            print(f"  Status: {result['status_code']}")
        elif result["status"] == "fail":
            print(f"  Expected: {result['expected_status']}, Actual: {result['actual_status']}")
        else:
            print(f"  Error: {result['message']}")
        
        print(f"  Time: {result['elapsed_time']:.3f}s\n")


if __name__ == "__main__":
    print("Testing API with versioned router structure...")
    test_results = run_tests()
    print_results(test_results)
    
    # Save results to file
    with open("api_test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)
    
    # Exit with error code if any tests failed
    if test_results["summary"]["fail"] > 0 or test_results["summary"]["error"] > 0:
        sys.exit(1)
    else:
        sys.exit(0)