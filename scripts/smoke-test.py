#!/usr/bin/env python
"""
Smoke test script for verifying a deployed LangChain HANA integration API.
This script performs basic health checks and functionality tests.
"""

import argparse
import json
import sys
import time
import requests
from typing import Dict, Any, List, Optional

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Smoke test for LangChain HANA API")
    parser.add_argument("--api-url", required=True, help="Base URL of the API")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout in seconds")
    parser.add_argument("--retries", type=int, default=3, help="Number of retries for each test")
    parser.add_argument("--retry-delay", type=int, default=5, help="Delay between retries in seconds")
    return parser.parse_args()

class SmokeTestRunner:
    """Runs smoke tests against the deployed API."""
    
    def __init__(self, api_url: str, timeout: int = 30, retries: int = 3, retry_delay: int = 5):
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.retry_delay = retry_delay
        self.success_count = 0
        self.failure_count = 0
        self.tests_run = 0
    
    def run_test(self, name: str, endpoint: str, method: str = "GET", 
                 data: Optional[Dict[str, Any]] = None, 
                 expected_status: int = 200,
                 expected_keys: Optional[List[str]] = None) -> bool:
        """Run a single test with retries."""
        self.tests_run += 1
        print(f"\n[TEST {self.tests_run}] {name}")
        
        for attempt in range(1, self.retries + 1):
            try:
                if method.upper() == "GET":
                    response = requests.get(f"{self.api_url}/{endpoint.lstrip('/')}", 
                                          timeout=self.timeout)
                else:
                    response = requests.post(f"{self.api_url}/{endpoint.lstrip('/')}",
                                           json=data, 
                                           timeout=self.timeout)
                
                # Check status code
                if response.status_code != expected_status:
                    print(f"  Attempt {attempt}/{self.retries} - Failed: Expected status {expected_status}, got {response.status_code}")
                    if attempt < self.retries:
                        print(f"  Retrying in {self.retry_delay} seconds...")
                        time.sleep(self.retry_delay)
                        continue
                    self.failure_count += 1
                    return False
                
                # Check response content
                if expected_keys and response.headers.get("content-type") == "application/json":
                    content = response.json()
                    missing_keys = [key for key in expected_keys if key not in content]
                    if missing_keys:
                        print(f"  Attempt {attempt}/{self.retries} - Failed: Missing expected keys: {missing_keys}")
                        if attempt < self.retries:
                            print(f"  Retrying in {self.retry_delay} seconds...")
                            time.sleep(self.retry_delay)
                            continue
                        self.failure_count += 1
                        return False
                
                print(f"  Success: {response.status_code}")
                self.success_count += 1
                return True
                
            except Exception as e:
                print(f"  Attempt {attempt}/{self.retries} - Error: {str(e)}")
                if attempt < self.retries:
                    print(f"  Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    self.failure_count += 1
                    return False
        
        return False
    
    def run_all_tests(self) -> bool:
        """Run all smoke tests."""
        print(f"Running smoke tests against {self.api_url}\n")
        
        # Health check
        self.run_test("Health Check", "/health", "GET", 
                     expected_keys=["status", "version"])
        
        # Readiness check
        self.run_test("Readiness Check", "/health/ready", "GET", 
                     expected_keys=["status"])
        
        # Version check
        self.run_test("Version Check", "/version", "GET",
                     expected_keys=["version"])
        
        # Simple embedding test
        test_data = {
            "texts": ["This is a test sentence for embeddings."],
            "model": "all-MiniLM-L6-v2"
        }
        self.run_test("Basic Embeddings Test", "/embeddings", "POST", 
                     data=test_data,
                     expected_keys=["embeddings"])
        
        # Print summary
        print("\n--- Test Summary ---")
        print(f"Total tests: {self.tests_run}")
        print(f"Passed: {self.success_count}")
        print(f"Failed: {self.failure_count}")
        
        return self.failure_count == 0

def main():
    args = parse_args()
    runner = SmokeTestRunner(args.api_url, args.timeout, args.retries, args.retry_delay)
    success = runner.run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()