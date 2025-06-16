#!/usr/bin/env python3
"""
API Endpoint Verification Script

This script checks if all expected API endpoints required by the frontend
are available and responding correctly.
"""

import argparse
import json
import requests
import sys
from typing import Dict, List, Any, Tuple

# Colors for terminal output
class Colors:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"

# Expected endpoints and their methods
REQUIRED_ENDPOINTS = [
    # Health endpoints
    {"path": "/health", "method": "GET", "description": "System health check"},
    
    # Vector store endpoints
    {"path": "/texts", "method": "POST", "description": "Add texts to vector store"},
    {"path": "/query", "method": "POST", "description": "Query vector store"},
    {"path": "/query/vector", "method": "POST", "description": "Query by vector"},
    {"path": "/query/mmr", "method": "POST", "description": "MMR query"},
    {"path": "/delete", "method": "POST", "description": "Delete from vector store"},
    
    # Benchmark endpoints
    {"path": "/benchmark/status", "method": "GET", "description": "Benchmark status"},
    {"path": "/benchmark/gpu_info", "method": "GET", "description": "GPU info"},
    {"path": "/benchmark/embedding", "method": "POST", "description": "Embedding benchmark"},
    {"path": "/benchmark/search", "method": "POST", "description": "Vector search benchmark"},
    {"path": "/benchmark/tensorrt", "method": "POST", "description": "TensorRT benchmark"},
    {"path": "/benchmark/compare_embeddings", "method": "POST", "description": "Compare embeddings"},
    
    # GPU endpoints
    {"path": "/gpu/info", "method": "GET", "description": "GPU information"},
    
    # Developer endpoints
    {"path": "/developer/run", "method": "POST", "description": "Run flow"},
    {"path": "/developer/generate-code", "method": "POST", "description": "Generate code"},
    {"path": "/developer/flows", "method": "GET", "description": "List flows"},
    {"path": "/developer/flows", "method": "POST", "description": "Save flow"},
    {"path": "/developer/vectors", "method": "POST", "description": "Get vectors"},
    
    # Analytics endpoints (new)
    {"path": "/analytics/recent-queries", "method": "GET", "description": "Recent queries"},
    {"path": "/analytics/performance", "method": "GET", "description": "Performance stats"},
    {"path": "/analytics/query-count", "method": "GET", "description": "Query count"},
    {"path": "/analytics/response-time", "method": "GET", "description": "Average response time"},
    {"path": "/analytics/performance-comparison", "method": "GET", "description": "Performance comparison"},
    
    # Reasoning endpoints (new)
    {"path": "/reasoning/track", "method": "POST", "description": "Track reasoning path"},
    {"path": "/reasoning/transformation", "method": "POST", "description": "Track transformation"},
    {"path": "/reasoning/validate", "method": "POST", "description": "Validate reasoning"},
    {"path": "/reasoning/metrics", "method": "POST", "description": "Calculate metrics"},
    {"path": "/reasoning/feedback", "method": "POST", "description": "Submit feedback"},
    {"path": "/reasoning/fingerprint", "method": "POST", "description": "Get fingerprint"},
    {"path": "/reasoning/status", "method": "GET", "description": "Reasoning status"},
    
    # Data Pipeline endpoints (new)
    {"path": "/data-pipeline/create", "method": "POST", "description": "Create pipeline"},
    {"path": "/data-pipeline/register-source", "method": "POST", "description": "Register data source"},
    {"path": "/data-pipeline/register-intermediate", "method": "POST", "description": "Register intermediate stage"},
    {"path": "/data-pipeline/register-vector", "method": "POST", "description": "Register vector"},
    {"path": "/data-pipeline/register-rule", "method": "POST", "description": "Register transformation rule"},
    {"path": "/data-pipeline/get", "method": "POST", "description": "Get pipeline"},
    {"path": "/data-pipeline/lineage", "method": "POST", "description": "Get data lineage"},
    {"path": "/data-pipeline/reverse-map", "method": "POST", "description": "Get reverse map"},
    {"path": "/data-pipeline/list", "method": "GET", "description": "List pipelines"},
    {"path": "/data-pipeline/status", "method": "GET", "description": "Data pipeline status"},
    
    # Vector Operations endpoints (new)
    {"path": "/vector-operations/create", "method": "POST", "description": "Create vectors"},
    {"path": "/vector-operations/info", "method": "POST", "description": "Get vector info"},
    {"path": "/vector-operations/batch-embed", "method": "POST", "description": "Batch embed"},
    {"path": "/vector-operations/models", "method": "GET", "description": "List models"},
]

def check_endpoint(base_url: str, endpoint: Dict[str, str], timeout: int = 5) -> Tuple[bool, str, Any]:
    """
    Check if an endpoint is available and responding.
    
    Args:
        base_url: Base URL of the API
        endpoint: Dictionary with endpoint information
        timeout: Request timeout in seconds
    
    Returns:
        Tuple of (success, message, response_data)
    """
    url = f"{base_url}{endpoint['path']}"
    method = endpoint['method']
    description = endpoint['description']
    
    try:
        if method == "GET":
            response = requests.get(url, timeout=timeout)
        elif method == "POST":
            # For POST requests, we send minimal valid data
            # This might need customization based on endpoint requirements
            minimal_data = {}
            response = requests.post(url, json=minimal_data, timeout=timeout)
        else:
            return False, f"Unsupported method: {method}", None
        
        # Consider 2xx and 4xx as "available" (4xx can mean invalid input which is expected)
        # 5xx or connection errors suggest the endpoint isn't implemented
        if 200 <= response.status_code < 500:
            if 200 <= response.status_code < 300:
                status = "OK"
                try:
                    data = response.json()
                except ValueError:
                    data = response.text
                return True, f"{status} ({response.status_code})", data
            else:
                status = "Client Error"
                return True, f"{status} ({response.status_code}) - Endpoint exists but request was invalid", None
        else:
            status = "Server Error"
            return False, f"{status} ({response.status_code}) - Endpoint may not be implemented", None
            
    except requests.exceptions.ConnectionError:
        return False, "Connection Error - API server may be down", None
    except requests.exceptions.Timeout:
        return False, "Timeout - Request took too long", None
    except Exception as e:
        return False, f"Error: {str(e)}", None

def run_endpoint_check(base_url: str, verbose: bool = False) -> Tuple[int, int, List[Dict[str, Any]]]:
    """
    Run checks for all required endpoints.
    
    Args:
        base_url: Base URL of the API
        verbose: Whether to print detailed response information
    
    Returns:
        Tuple of (available_count, total_count, results)
    """
    results = []
    available_count = 0
    
    print(f"{Colors.BOLD}Checking API endpoints at {base_url}{Colors.RESET}")
    print(f"{'Endpoint':<40} {'Method':<7} {'Status':<30} {'Description'}")
    print("-" * 90)
    
    for endpoint in REQUIRED_ENDPOINTS:
        success, message, data = check_endpoint(base_url, endpoint)
        
        if success:
            status_color = Colors.GREEN
            available_count += 1
        else:
            status_color = Colors.RED
            
        print(f"{endpoint['path']:<40} {endpoint['method']:<7} {status_color}{message:<30}{Colors.RESET} {endpoint['description']}")
        
        if verbose and success and data:
            print(f"  {Colors.BLUE}Response:{Colors.RESET}")
            if isinstance(data, dict) or isinstance(data, list):
                print(f"  {json.dumps(data, indent=2)[:200]}...")
            else:
                print(f"  {str(data)[:200]}...")
                
        results.append({
            "endpoint": endpoint['path'],
            "method": endpoint['method'],
            "description": endpoint['description'],
            "available": success,
            "message": message,
            "data": data if success else None
        })
    
    return available_count, len(REQUIRED_ENDPOINTS), results

def main():
    parser = argparse.ArgumentParser(description='API Endpoint Verification Tool')
    parser.add_argument('--base-url', type=str, default='http://localhost:8000',
                        help='Base URL of the API (default: http://localhost:8000)')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed response information')
    parser.add_argument('--output', type=str,
                        help='Output results to JSON file')
    args = parser.parse_args()
    
    available_count, total_count, results = run_endpoint_check(args.base_url, args.verbose)
    
    # Print summary
    print("\n" + "-" * 90)
    percentage = (available_count / total_count) * 100
    if percentage >= 90:
        color = Colors.GREEN
    elif percentage >= 70:
        color = Colors.YELLOW
    else:
        color = Colors.RED
        
    print(f"{Colors.BOLD}Summary:{Colors.RESET} {color}{available_count}/{total_count} endpoints available ({percentage:.1f}%){Colors.RESET}")
    
    # Identify missing critical endpoints
    critical_paths = ['/health', '/texts', '/query', '/gpu/info']
    missing_critical = [r for r in results if not r['available'] and r['endpoint'] in critical_paths]
    
    if missing_critical:
        print(f"\n{Colors.RED}{Colors.BOLD}Warning: Critical endpoints are missing:{Colors.RESET}")
        for endpoint in missing_critical:
            print(f"  - {endpoint['endpoint']} ({endpoint['method']}): {endpoint['description']}")
    
    # Output to JSON if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump({
                "timestamp": import time; time.time(),
                "base_url": args.base_url,
                "available_count": available_count,
                "total_count": total_count,
                "percentage": percentage,
                "results": results
            }, f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    # Return non-zero exit code if too many endpoints are missing
    if percentage < 70:
        sys.exit(1)
    
    return 0

if __name__ == "__main__":
    main()