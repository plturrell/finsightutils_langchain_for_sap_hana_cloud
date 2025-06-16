#!/usr/bin/env python3
"""
Simple API Test Script to check basic endpoints
"""

import requests
import sys
import time
from typing import Dict, Any, List, Tuple

# Define colors for terminal output
class Colors:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"

# Basic endpoints to check
BASIC_ENDPOINTS = [
    {"path": "/", "method": "GET", "description": "Root endpoint"},
    {"path": "/health", "method": "GET", "description": "Health check"},
    {"path": "/health/ping", "method": "GET", "description": "Simple ping endpoint"},
    {"path": "/health/status", "method": "GET", "description": "Detailed health status"},
    {"path": "/gpu/info", "method": "GET", "description": "GPU information"}
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
            minimal_data = {}
            response = requests.post(url, json=minimal_data, timeout=timeout)
        else:
            return False, f"Unsupported method: {method}", None
        
        # Consider 2xx and 4xx as "available" (4xx can mean invalid input which is expected)
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

def run_basic_test(base_url: str) -> Tuple[int, int, List[Dict[str, Any]]]:
    """
    Run basic checks for essential endpoints.
    
    Args:
        base_url: Base URL of the API
    
    Returns:
        Tuple of (available_count, total_count, results)
    """
    results = []
    available_count = 0
    
    print(f"{Colors.BOLD}Checking basic API endpoints at {base_url}{Colors.RESET}")
    print(f"{'Endpoint':<30} {'Method':<7} {'Status':<30} {'Description'}")
    print("-" * 90)
    
    for endpoint in BASIC_ENDPOINTS:
        success, message, data = check_endpoint(base_url, endpoint)
        
        if success:
            status_color = Colors.GREEN
            available_count += 1
        else:
            status_color = Colors.RED
            
        print(f"{endpoint['path']:<30} {endpoint['method']:<7} {status_color}{message:<30}{Colors.RESET} {endpoint['description']}")
        
        if success and data:
            print(f"  {Colors.BLUE}Response:{Colors.RESET}")
            if isinstance(data, dict) or isinstance(data, list):
                print(f"  {str(data)[:200]}...")
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
    
    return available_count, len(BASIC_ENDPOINTS), results

def main():
    # Default URL for local testing
    base_url = "http://localhost:8000"
    
    # Allow override from command line
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    
    print(f"{Colors.BOLD}{Colors.BLUE}Testing API at: {base_url}{Colors.RESET}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 90)
    
    # Run basic checks
    available_count, total_count, results = run_basic_test(base_url)
    
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
    
    # Recommend next steps
    if percentage >= 90:
        print(f"\n{Colors.GREEN}✓ API is functioning correctly!{Colors.RESET}")
        print("Next steps:")
        print("1. Start the frontend container")
        print("2. Test the full system integration")
    elif percentage >= 50:
        print(f"\n{Colors.YELLOW}⚠ API is partially available.{Colors.RESET}")
        print("Recommended troubleshooting:")
        print("1. Check API logs for errors")
        print("2. Verify environment variables are set correctly")
    else:
        print(f"\n{Colors.RED}✗ API is mostly unavailable.{Colors.RESET}")
        print("Recommended troubleshooting:")
        print("1. Ensure the API container is running")
        print("2. Check network connectivity")
        print("3. Review API logs for startup errors")
    
    # Exit with appropriate code
    if percentage < 50:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()