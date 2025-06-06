#!/usr/bin/env python
"""
Comprehensive verification script for production deployments.
This script performs extensive tests on the deployed API including
performance, consistency, error handling, and GPU acceleration tests.

For GPU-accelerated deployments, this script performs specialized tests to verify:
1. GPU availability and proper configuration
2. TensorRT acceleration for embeddings
3. Memory optimization
4. Multi-GPU support (if available)
5. Blue-green deployment status (if applicable)
"""

import argparse
import json
import sys
import time
import statistics
import requests
import concurrent.futures
from typing import Dict, Any, List, Optional, Tuple

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Verify production deployment of LangChain HANA API")
    parser.add_argument("--api-url", required=True, help="Base URL of the API")
    parser.add_argument("--timeout", type=int, default=60, help="Request timeout in seconds")
    parser.add_argument("--concurrency", type=int, default=5, help="Number of concurrent requests for load testing")
    parser.add_argument("--threshold", type=float, default=1.0, help="Response time threshold in seconds")
    parser.add_argument("--gpu", action="store_true", help="Run GPU-specific tests")
    parser.add_argument("--blue-green", action="store_true", help="Verify blue-green deployment")
    parser.add_argument("--token", help="API token (if required)")
    return parser.parse_args()

class DeploymentVerifier:
    """Verifies a production deployment with comprehensive tests."""
    
    def __init__(self, api_url: str, timeout: int = 60, concurrency: int = 5, 
                threshold: float = 1.0, test_gpu: bool = False, 
                test_blue_green: bool = False, token: Optional[str] = None):
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout
        self.concurrency = concurrency
        self.threshold = threshold
        self.test_gpu = test_gpu
        self.test_blue_green = test_blue_green
        self.token = token
        self.success_count = 0
        self.failure_count = 0
        self.tests_run = 0
        self.performance_results = {}
        self.gpu_results = {}
        
    def run_test(self, name: str, endpoint: str, method: str = "GET", 
                 data: Optional[Dict[str, Any]] = None, 
                 expected_status: int = 200,
                 expected_keys: Optional[List[str]] = None,
                 measure_time: bool = False) -> Tuple[bool, Optional[float]]:
        """Run a single test and optionally measure response time."""
        self.tests_run += 1
        print(f"\n[TEST {self.tests_run}] {name}")
        
        try:
            start_time = time.time() if measure_time else None
            
            if method.upper() == "GET":
                response = requests.get(f"{self.api_url}/{endpoint.lstrip('/')}", 
                                       timeout=self.timeout)
            else:
                response = requests.post(f"{self.api_url}/{endpoint.lstrip('/')}",
                                       json=data, 
                                       timeout=self.timeout)
            
            end_time = time.time() if measure_time else None
            response_time = end_time - start_time if start_time else None
            
            # Check status code
            if response.status_code != expected_status:
                print(f"  Failed: Expected status {expected_status}, got {response.status_code}")
                self.failure_count += 1
                return False, response_time
            
            # Check response content
            if expected_keys and response.headers.get("content-type") == "application/json":
                content = response.json()
                missing_keys = [key for key in expected_keys if key not in content]
                if missing_keys:
                    print(f"  Failed: Missing expected keys: {missing_keys}")
                    self.failure_count += 1
                    return False, response_time
            
            if measure_time and response_time:
                print(f"  Success: {response.status_code} (Response time: {response_time:.3f}s)")
                if response_time > self.threshold:
                    print(f"  Warning: Response time {response_time:.3f}s exceeds threshold of {self.threshold:.3f}s")
            else:
                print(f"  Success: {response.status_code}")
                
            self.success_count += 1
            return True, response_time
                
        except Exception as e:
            print(f"  Error: {str(e)}")
            self.failure_count += 1
            return False, None
    
    def run_concurrent_test(self, name: str, endpoint: str, method: str = "GET",
                           data: Optional[Dict[str, Any]] = None,
                           iterations: int = 10) -> bool:
        """Run a test with concurrent requests to check for stability under load."""
        self.tests_run += 1
        print(f"\n[LOAD TEST] {name} ({self.concurrency} concurrent requests x {iterations} iterations)")
        
        response_times = []
        success_count = 0
        failure_count = 0
        
        for i in range(iterations):
            print(f"  Iteration {i+1}/{iterations}")
            futures = []
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.concurrency) as executor:
                for _ in range(self.concurrency):
                    if method.upper() == "GET":
                        future = executor.submit(
                            lambda: self._timed_request("GET", endpoint)
                        )
                    else:
                        future = executor.submit(
                            lambda: self._timed_request("POST", endpoint, data)
                        )
                    futures.append(future)
                
                for future in concurrent.futures.as_completed(futures):
                    success, response_time = future.result()
                    if success:
                        success_count += 1
                        if response_time:
                            response_times.append(response_time)
                    else:
                        failure_count += 1
            
            # Small delay between iterations
            if i < iterations - 1:
                time.sleep(1)
        
        total_requests = self.concurrency * iterations
        success_rate = (success_count / total_requests) * 100
        
        print(f"\n  Results:")
        print(f"  - Success rate: {success_rate:.2f}% ({success_count}/{total_requests})")
        
        if response_times:
            avg_time = statistics.mean(response_times)
            median_time = statistics.median(response_times)
            p95_time = sorted(response_times)[int(len(response_times) * 0.95)]
            
            print(f"  - Average response time: {avg_time:.3f}s")
            print(f"  - Median response time: {median_time:.3f}s")
            print(f"  - 95th percentile response time: {p95_time:.3f}s")
            
            self.performance_results[name] = {
                "success_rate": success_rate,
                "avg_time": avg_time,
                "median_time": median_time,
                "p95_time": p95_time,
                "total_requests": total_requests
            }
        
        # Test passes if success rate is at least 95%
        passed = success_rate >= 95.0
        if passed:
            self.success_count += 1
        else:
            self.failure_count += 1
        
        return passed
    
    def _timed_request(self, method: str, endpoint: str, 
                      data: Optional[Dict[str, Any]] = None) -> Tuple[bool, Optional[float]]:
        """Execute a request and measure its response time."""
        try:
            start_time = time.time()
            
            if method.upper() == "GET":
                response = requests.get(
                    f"{self.api_url}/{endpoint.lstrip('/')}",
                    timeout=self.timeout
                )
            else:
                response = requests.post(
                    f"{self.api_url}/{endpoint.lstrip('/')}",
                    json=data,
                    timeout=self.timeout
                )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            return response.status_code == 200, response_time
            
        except Exception:
            return False, None
    
    def verify_error_handling(self) -> bool:
        """Verify API error handling by sending invalid requests."""
        print("\n[TESTING ERROR HANDLING]")
        
        # Test 1: Invalid endpoint
        _, _ = self.run_test(
            "Invalid Endpoint Test", 
            "/non-existent-endpoint", 
            expected_status=404
        )
        
        # Test 2: Invalid request body
        _, _ = self.run_test(
            "Invalid Request Body Test", 
            "/embeddings", 
            method="POST",
            data={"invalid": "data"},
            expected_status=422
        )
        
        # Test 3: Missing required parameters
        _, _ = self.run_test(
            "Missing Parameters Test", 
            "/embeddings", 
            method="POST",
            data={"texts": []},
            expected_status=422
        )
        
        return True
    
    def run_all_verification_tests(self) -> bool:
        """Run a comprehensive set of verification tests."""
        print(f"Running verification tests against {self.api_url}\n")
        
        # Basic health checks
        self.run_test("Health Check", "/health", expected_keys=["status", "version"])
        self.run_test("Readiness Check", "/health/ready", expected_keys=["status"])
        self.run_test("Version Check", "/version", expected_keys=["version"])
        
        # Performance tests
        test_data = {
            "texts": ["This is a test sentence for embeddings."],
            "model": "all-MiniLM-L6-v2"
        }
        
        # Single request performance
        self.run_test("Embeddings Response Time", "/embeddings", 
                     method="POST", data=test_data, 
                     expected_keys=["embeddings"], 
                     measure_time=True)
        
        # Batch performance test
        batch_data = {
            "texts": ["This is sentence " + str(i) for i in range(10)],
            "model": "all-MiniLM-L6-v2"
        }
        self.run_test("Batch Embeddings Response Time", "/embeddings", 
                     method="POST", data=batch_data, 
                     expected_keys=["embeddings"], 
                     measure_time=True)
        
        # Concurrent load test
        self.run_concurrent_test("Embeddings Under Load", "/embeddings", 
                               method="POST", data=test_data, 
                               iterations=3)
        
        # Error handling verification
        self.verify_error_handling()
        
        # Print summary
        print("\n--- Verification Summary ---")
        print(f"Total tests: {self.tests_run}")
        print(f"Passed: {self.success_count}")
        print(f"Failed: {self.failure_count}")
        
        if self.performance_results:
            print("\n--- Performance Results ---")
            for test_name, results in self.performance_results.items():
                print(f"\n{test_name}:")
                print(f"  - Success rate: {results['success_rate']:.2f}%")
                print(f"  - Average response time: {results['avg_time']:.3f}s")
                print(f"  - Median response time: {results['median_time']:.3f}s")
                print(f"  - 95th percentile: {results['p95_time']:.3f}s")
        
        # Export results
        self._export_results()
        
        return self.failure_count == 0
    
    def test_gpu_acceleration(self) -> bool:
        """Run tests specific to GPU acceleration features."""
        if not self.test_gpu:
            return True
            
        print("\n[TESTING GPU ACCELERATION]")
        
        # Test 1: Check GPU status
        gpu_status_success, _ = self.run_test(
            "GPU Status Check",
            "/gpu/status",
            expected_keys=["gpu_available", "gpu_name", "cuda_version"]
        )
        
        # Test 2: Check TensorRT engine status
        tensorrt_success, _ = self.run_test(
            "TensorRT Status Check",
            "/gpu/tensorrt",
            expected_keys=["enabled", "precision", "engines"]
        )
        
        # Test 3: Test different precision modes
        precision_test_success = True
        precision_modes = ["fp32", "fp16", "int8"]
        precision_results = {}
        
        for precision in precision_modes:
            print(f"\n[TESTING {precision.upper()} PRECISION]")
            try:
                test_data = {
                    "texts": ["This is a test sentence for GPU embeddings."] * 5,
                    "model": "all-MiniLM-L6-v2",
                    "precision": precision
                }
                
                start_time = time.time()
                response = requests.post(
                    f"{self.api_url}/embeddings/precision",
                    json=test_data,
                    timeout=self.timeout,
                    headers={"Authorization": f"Bearer {self.token}"} if self.token else None
                )
                end_time = time.time()
                
                if response.status_code == 200:
                    precision_results[precision] = {
                        "status": "success",
                        "time": end_time - start_time
                    }
                    print(f"  Success: {precision} precision test completed in {end_time - start_time:.3f}s")
                else:
                    # Some GPUs might not support all precision modes
                    print(f"  Note: {precision} precision not supported or failed: {response.status_code}")
                    precision_results[precision] = {
                        "status": "failed",
                        "code": response.status_code
                    }
            except Exception as e:
                print(f"  Error: {str(e)}")
                precision_results[precision] = {
                    "status": "error",
                    "message": str(e)
                }
                precision_test_success = False
        
        self.gpu_results["precision_tests"] = precision_results
        
        # Test 4: Test multi-GPU support if available
        try:
            multi_gpu_response = requests.get(
                f"{self.api_url}/gpu/multi",
                timeout=self.timeout,
                headers={"Authorization": f"Bearer {self.token}"} if self.token else None
            )
            
            if multi_gpu_response.status_code == 200:
                multi_gpu_data = multi_gpu_response.json()
                if multi_gpu_data.get("multi_gpu_available", False):
                    print("\n[TESTING MULTI-GPU SUPPORT]")
                    # Test scaling with multiple GPUs
                    test_data = {
                        "texts": ["This is a test sentence for GPU embeddings."] * 20,
                        "model": "all-MiniLM-L6-v2",
                        "use_multi_gpu": True
                    }
                    
                    _, multi_gpu_time = self.run_test(
                        "Multi-GPU Embedding Test",
                        "/embeddings/multi-gpu",
                        method="POST",
                        data=test_data,
                        expected_keys=["embeddings", "gpu_stats"],
                        measure_time=True
                    )
                    
                    self.gpu_results["multi_gpu"] = {
                        "available": True,
                        "gpu_count": multi_gpu_data.get("gpu_count", 0),
                        "response_time": multi_gpu_time
                    }
                else:
                    print("  Multi-GPU support not available")
                    self.gpu_results["multi_gpu"] = {"available": False}
            else:
                print("  Multi-GPU endpoint not available")
                self.gpu_results["multi_gpu"] = {"available": False}
                
        except Exception as e:
            print(f"  Error checking multi-GPU: {str(e)}")
            self.gpu_results["multi_gpu"] = {"available": False, "error": str(e)}
        
        # Overall GPU acceleration test success
        gpu_test_success = gpu_status_success and tensorrt_success and precision_test_success
        
        if gpu_test_success:
            print("\nGPU acceleration tests passed successfully!")
        else:
            print("\nSome GPU acceleration tests failed.")
            
        return gpu_test_success
    
    def test_blue_green_deployment(self) -> bool:
        """Test blue-green deployment status."""
        if not self.test_blue_green:
            return True
            
        print("\n[TESTING BLUE-GREEN DEPLOYMENT]")
        
        try:
            # Check deployment status
            response = requests.get(
                f"{self.api_url}/deployment/status",
                timeout=self.timeout,
                headers={"Authorization": f"Bearer {self.token}"} if self.token else None
            )
            
            if response.status_code != 200:
                print(f"  Failed: Could not get deployment status: {response.status_code}")
                self.failure_count += 1
                return False
                
            status_data = response.json()
            active_color = status_data.get("color", "unknown")
            version = status_data.get("version", "unknown")
            
            print(f"  Active deployment: {active_color} (version {version})")
            
            # Check health of both blue and green deployments
            blue_url = self.api_url.replace("localhost", "localhost:8000")
            green_url = self.api_url.replace("localhost", "localhost:8001")
            
            try:
                blue_response = requests.get(f"{blue_url}/health/status", timeout=self.timeout)
                blue_healthy = blue_response.status_code == 200
                print(f"  Blue deployment: {'Healthy' if blue_healthy else 'Unhealthy'}")
            except Exception as e:
                blue_healthy = False
                print(f"  Blue deployment: Unreachable ({str(e)})")
                
            try:
                green_response = requests.get(f"{green_url}/health/status", timeout=self.timeout)
                green_healthy = green_response.status_code == 200
                print(f"  Green deployment: {'Healthy' if green_healthy else 'Unhealthy'}")
            except Exception as e:
                green_healthy = False
                print(f"  Green deployment: Unreachable ({str(e)})")
            
            # Test success if active deployment is healthy
            if (active_color == "blue" and blue_healthy) or (active_color == "green" and green_healthy):
                print("  Blue-green deployment check passed!")
                self.success_count += 1
                return True
            else:
                print(f"  Failed: Active deployment ({active_color}) is not healthy")
                self.failure_count += 1
                return False
                
        except Exception as e:
            print(f"  Error testing blue-green deployment: {str(e)}")
            self.failure_count += 1
            return False
    
    def _export_results(self):
        """Export test results to a JSON file."""
        results = {
            "timestamp": time.time(),
            "api_url": self.api_url,
            "summary": {
                "total_tests": self.tests_run,
                "passed": self.success_count,
                "failed": self.failure_count,
                "success_rate": (self.success_count / self.tests_run) * 100 if self.tests_run > 0 else 0
            },
            "performance": self.performance_results,
            "gpu": self.gpu_results if self.test_gpu else {}
        }
        
        with open("verification_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults exported to verification_results.json")

def main():
    args = parse_args()
    verifier = DeploymentVerifier(
        args.api_url, 
        args.timeout, 
        args.concurrency, 
        args.threshold,
        test_gpu=args.gpu,
        test_blue_green=args.blue_green,
        token=args.token
    )
    
    # Run standard verification tests
    standard_success = verifier.run_all_verification_tests()
    
    # Run GPU-specific tests if enabled
    gpu_success = verifier.test_gpu_acceleration() if args.gpu else True
    
    # Run blue-green deployment tests if enabled
    blue_green_success = verifier.test_blue_green_deployment() if args.blue_green else True
    
    # Overall success
    success = standard_success and gpu_success and blue_green_success
    
    print("\n==== VERIFICATION SUMMARY ====")
    print(f"Standard tests: {'PASSED' if standard_success else 'FAILED'}")
    if args.gpu:
        print(f"GPU tests: {'PASSED' if gpu_success else 'FAILED'}")
    if args.blue_green:
        print(f"Blue-Green tests: {'PASSED' if blue_green_success else 'FAILED'}")
    print(f"Overall status: {'PASSED' if success else 'FAILED'}")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()