#!/usr/bin/env python3
"""
API Diagnostics Tool for SAP HANA Cloud LangChain T4 GPU Integration

This script provides a comprehensive set of tests to diagnose issues with
the T4 GPU backend API, particularly focusing on connection issues and
500 server errors. It's designed to help pinpoint the exact cause of failures.

Usage:
    python api_diagnostics.py --url https://your-vercel-app.vercel.app [options]

Options:
    --url URL               The base URL of the API (required)
    --backend URL           The backend URL to test (optional)
    --username USERNAME     Username for authentication
    --password PASSWORD     Password for authentication
    --test-embeddings       Test the embeddings endpoint
    --test-metrics          Test the metrics endpoint
    --test-health           Test the health endpoint
    --debug                 Enable debug output
    --test-all              Run all tests
    --use-direct-proxy      Use the direct proxy endpoint for testing
"""

import argparse
import json
import os
import sys
import time
import traceback
import socket
import urllib.parse
import ssl
import requests
from typing import Dict, Any, List, Optional, Union

# Default values
DEFAULT_USERNAME = "admin"
DEFAULT_PASSWORD = "sap-hana-t4-admin"

# ANSI colors for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def print_color(text, color):
    """Print colored text"""
    print(f"{color}{text}{Colors.END}")

def print_success(text):
    """Print success message"""
    print_color(f"✅ SUCCESS: {text}", Colors.GREEN)

def print_warning(text):
    """Print warning message"""
    print_color(f"⚠️  WARNING: {text}", Colors.YELLOW)

def print_error(text):
    """Print error message"""
    print_color(f"❌ ERROR: {text}", Colors.RED)

def print_info(text):
    """Print info message"""
    print_color(f"ℹ️  INFO: {text}", Colors.BLUE)

def print_header(text):
    """Print header"""
    print("\n" + "=" * 80)
    print_color(f" {text} ", Colors.BOLD + Colors.UNDERLINE)
    print("=" * 80)

def print_separator():
    """Print separator line"""
    print("-" * 80)

def print_json(data):
    """Print JSON data"""
    print(json.dumps(data, indent=2))

def is_vercel_url(url):
    """Check if the URL is a Vercel deployment"""
    parsed = urllib.parse.urlparse(url)
    return parsed.netloc.endswith('.vercel.app')

# Basic connection testing functions
def test_dns_resolution(hostname):
    """Test DNS resolution for a hostname"""
    print_info(f"Testing DNS resolution for {hostname}...")
    try:
        start_time = time.time()
        ip_address = socket.gethostbyname(hostname)
        elapsed_time = (time.time() - start_time) * 1000  # Convert to ms
        
        print_success(f"DNS resolution successful: {hostname} -> {ip_address} ({elapsed_time:.2f}ms)")
        return {
            "success": True,
            "hostname": hostname,
            "ip_address": ip_address,
            "resolution_time_ms": elapsed_time
        }
    except socket.gaierror as e:
        print_error(f"DNS resolution failed: {hostname} -> {str(e)}")
        return {
            "success": False,
            "hostname": hostname,
            "error": str(e),
            "error_code": e.errno
        }
    except Exception as e:
        print_error(f"DNS resolution error: {str(e)}")
        return {
            "success": False,
            "hostname": hostname,
            "error": str(e)
        }

def test_tcp_connection(hostname, port, use_ssl=False, timeout=5):
    """Test TCP connection to a host and port"""
    ssl_text = "with SSL" if use_ssl else "without SSL"
    print_info(f"Testing TCP connection to {hostname}:{port} {ssl_text}...")
    try:
        start_time = time.time()
        
        # Create socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        
        # Wrap with SSL if needed
        if use_ssl:
            context = ssl.create_default_context()
            sock = context.wrap_socket(sock, server_hostname=hostname)
        
        # Connect
        sock.connect((hostname, port))
        elapsed_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Close connection
        sock.close()
        
        print_success(f"TCP connection successful: {hostname}:{port} {ssl_text} ({elapsed_time:.2f}ms)")
        return {
            "success": True,
            "hostname": hostname,
            "port": port,
            "ssl": use_ssl,
            "connection_time_ms": elapsed_time
        }
    except socket.timeout:
        print_error(f"TCP connection timed out: {hostname}:{port} {ssl_text}")
        return {
            "success": False,
            "hostname": hostname,
            "port": port,
            "ssl": use_ssl,
            "error": "Connection timed out"
        }
    except ssl.SSLError as e:
        print_error(f"SSL error connecting to {hostname}:{port}: {str(e)}")
        return {
            "success": False,
            "hostname": hostname,
            "port": port,
            "ssl": use_ssl,
            "error": str(e)
        }
    except ConnectionRefusedError:
        print_error(f"Connection refused: {hostname}:{port} {ssl_text}")
        return {
            "success": False,
            "hostname": hostname,
            "port": port,
            "ssl": use_ssl,
            "error": "Connection refused"
        }
    except Exception as e:
        print_error(f"TCP connection error: {str(e)}")
        return {
            "success": False,
            "hostname": hostname,
            "port": port,
            "ssl": use_ssl,
            "error": str(e)
        }

# API test functions
def login(base_url, username, password):
    """Login to the API and get an authentication token"""
    print_info(f"Logging in as {username}...")
    
    try:
        response = requests.post(
            f"{base_url}/api/auth/token",
            json={"username": username, "password": password},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            token = data.get("access_token")
            print_success(f"Login successful. Token received.")
            return token
        else:
            print_error(f"Login failed with status code {response.status_code}")
            print(f"Response: {response.text}")
            return None
    except Exception as e:
        print_error(f"Login error: {str(e)}")
        return None

def check_health(base_url, direct_proxy=False):
    """Check the health of the API"""
    print_info("Checking API health...")
    
    endpoint = "/debug-proxy/api/health" if direct_proxy else "/api/health"
    try:
        response = requests.get(
            f"{base_url}{endpoint}",
            timeout=10
        )
        
        if response.status_code == 200:
            print_success("Health check successful")
            data = response.json()
            print_json(data)
            return data
        else:
            print_error(f"Health check failed: {response.status_code}")
            print(f"Response: {response.text}")
            return None
    except Exception as e:
        print_error(f"Health check error: {str(e)}")
        return None

def test_embeddings(base_url, token=None, direct_proxy=False):
    """Test the embeddings endpoint"""
    print_info("Testing embeddings generation...")
    
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    endpoint = "/debug-proxy/api/embeddings" if direct_proxy else "/api/embeddings"
    try:
        response = requests.post(
            f"{base_url}{endpoint}",
            headers=headers,
            json={
                "texts": [
                    "This is a test embedding",
                    "Another test embedding"
                ],
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "use_tensorrt": True,
                "precision": "int8"
            },
            timeout=30
        )
        
        if response.status_code == 200:
            print_success("Embeddings generated successfully")
            data = response.json()
            print(f"Model: {data.get('model')}")
            print(f"Dimensions: {data.get('dimensions')}")
            print(f"Processing time: {data.get('processing_time_ms')} ms")
            print(f"GPU used: {data.get('gpu_used')}")
            print(f"Batch size: {data.get('batch_size_used')}")
            print(f"Number of embeddings: {len(data.get('embeddings', []))}")
            return data
        else:
            print_error(f"Failed to generate embeddings: {response.status_code}")
            print(f"Response: {response.text}")
            return None
    except Exception as e:
        print_error(f"Error generating embeddings: {str(e)}")
        return None

def test_metrics(base_url, token=None, direct_proxy=False):
    """Test the metrics endpoint"""
    print_info("Testing metrics endpoint...")
    
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    endpoint = "/debug-proxy/api/metrics" if direct_proxy else "/api/metrics"
    try:
        response = requests.get(
            f"{base_url}{endpoint}",
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            print_success("Metrics retrieved successfully")
            data = response.json()
            print_json(data)
            return data
        else:
            print_error(f"Failed to get metrics: {response.status_code}")
            print(f"Response: {response.text}")
            return None
    except Exception as e:
        print_error(f"Error getting metrics: {str(e)}")
        return None

def run_connection_diagnostics(base_url, direct_proxy=False):
    """Run connection diagnostics"""
    print_info("Running connection diagnostics...")
    
    endpoint = "/debug-proxy/connection-diagnostics" if direct_proxy else "/connection-diagnostics"
    try:
        # Check if the diagnostics endpoint is available
        response = requests.get(
            f"{base_url}{endpoint}",
            timeout=15
        )
        
        if response.status_code == 200:
            print_success("Connection diagnostics successful")
            data = response.json()
            print_json(data)
            return data
        else:
            print_warning(f"Connection diagnostics endpoint not available: {response.status_code}")
            
            # Fallback to manual diagnostics
            print_info("Falling back to manual connection diagnostics...")
            parsed = urllib.parse.urlparse(base_url)
            hostname = parsed.hostname
            port = parsed.port or (443 if parsed.scheme == 'https' else 80)
            use_ssl = parsed.scheme == 'https'
            
            results = {}
            results["dns"] = test_dns_resolution(hostname)
            results["tcp"] = test_tcp_connection(hostname, port, use_ssl=use_ssl)
            
            return results
    except Exception as e:
        print_error(f"Connection diagnostics error: {str(e)}")
        
        # Fallback to manual diagnostics
        print_info("Falling back to manual connection diagnostics...")
        parsed = urllib.parse.urlparse(base_url)
        hostname = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == 'https' else 80)
        use_ssl = parsed.scheme == 'https'
        
        results = {}
        results["dns"] = test_dns_resolution(hostname)
        results["tcp"] = test_tcp_connection(hostname, port, use_ssl=use_ssl)
        
        return results

def test_proxy_info(base_url):
    """Test the proxy info endpoint"""
    print_info("Testing proxy info endpoint...")
    
    try:
        response = requests.get(
            f"{base_url}/debug-proxy/proxy-info",
            timeout=10
        )
        
        if response.status_code == 200:
            print_success("Proxy info retrieved successfully")
            data = response.json()
            print_json(data)
            return data
        else:
            print_error(f"Failed to get proxy info: {response.status_code}")
            print(f"Response: {response.text}")
            return None
    except Exception as e:
        print_error(f"Error getting proxy info: {str(e)}")
        return None

def test_backend_api(backend_url):
    """Test the backend API directly"""
    print_info(f"Testing backend API directly at {backend_url}...")
    
    try:
        # Check health
        print_info("Checking backend health...")
        response = requests.get(
            f"{backend_url}/api/health",
            timeout=10
        )
        
        if response.status_code == 200:
            print_success("Backend health check successful")
            data = response.json()
            print_json(data)
        else:
            print_error(f"Backend health check failed: {response.status_code}")
            print(f"Response: {response.text}")
        
        print_separator()
        
        # Basic connection diagnostics
        parsed = urllib.parse.urlparse(backend_url)
        hostname = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == 'https' else 80)
        use_ssl = parsed.scheme == 'https'
        
        test_dns_resolution(hostname)
        test_tcp_connection(hostname, port, use_ssl=use_ssl)
        
        return True
    except Exception as e:
        print_error(f"Error testing backend API: {str(e)}")
        
        # Basic connection diagnostics
        parsed = urllib.parse.urlparse(backend_url)
        hostname = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == 'https' else 80)
        use_ssl = parsed.scheme == 'https'
        
        test_dns_resolution(hostname)
        test_tcp_connection(hostname, port, use_ssl=use_ssl)
        
        return False

def run_all_tests(args):
    """Run all tests"""
    print_header("API DIAGNOSTICS TOOL")
    print(f"Target API: {args.url}")
    
    if args.backend:
        print(f"Backend API: {args.backend}")
    
    print(f"Using direct proxy: {args.use_direct_proxy}")
    print_separator()
    
    # Test basic connectivity
    print_header("BASIC CONNECTIVITY TESTS")
    parsed = urllib.parse.urlparse(args.url)
    hostname = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == 'https' else 80)
    use_ssl = parsed.scheme == 'https'
    
    test_dns_resolution(hostname)
    test_tcp_connection(hostname, port, use_ssl=use_ssl)
    
    # Check if this is a Vercel deployment
    if is_vercel_url(args.url):
        print_info("Detected Vercel deployment")
    
    # Login
    print_header("AUTHENTICATION")
    token = login(args.url, args.username, args.password)
    
    # Health check
    print_header("HEALTH CHECK")
    check_health(args.url, args.use_direct_proxy)
    
    # Check proxy info if using direct proxy
    if args.use_direct_proxy:
        print_header("PROXY INFO")
        test_proxy_info(args.url)
    
    # Run connection diagnostics
    print_header("CONNECTION DIAGNOSTICS")
    run_connection_diagnostics(args.url, args.use_direct_proxy)
    
    # Test embeddings
    if args.test_embeddings or args.test_all:
        print_header("EMBEDDINGS TEST")
        test_embeddings(args.url, token, args.use_direct_proxy)
    
    # Test metrics
    if args.test_metrics or args.test_all:
        print_header("METRICS TEST")
        test_metrics(args.url, token, args.use_direct_proxy)
    
    # Test backend directly if provided
    if args.backend:
        print_header("DIRECT BACKEND TESTS")
        test_backend_api(args.backend)
    
    print_header("TEST SUMMARY")
    print_info("Testing completed. Review the results above for any issues.")
    if args.use_direct_proxy:
        print_info("You used the direct proxy endpoints, which provide more detailed error information.")
    else:
        print_info("For more detailed diagnostics, try running with --use-direct-proxy")

def main():
    parser = argparse.ArgumentParser(description="API Diagnostics Tool for SAP HANA Cloud LangChain T4 GPU Integration")
    
    parser.add_argument("--url", required=True, help="The base URL of the API")
    parser.add_argument("--backend", help="The backend URL to test directly")
    parser.add_argument("--username", default=DEFAULT_USERNAME, help="Username for authentication")
    parser.add_argument("--password", default=DEFAULT_PASSWORD, help="Password for authentication")
    parser.add_argument("--test-embeddings", action="store_true", help="Test the embeddings endpoint")
    parser.add_argument("--test-metrics", action="store_true", help="Test the metrics endpoint")
    parser.add_argument("--test-health", action="store_true", help="Test the health endpoint")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--test-all", action="store_true", help="Run all tests")
    parser.add_argument("--use-direct-proxy", action="store_true", help="Use the direct proxy endpoint for testing")
    
    args = parser.parse_args()
    
    run_all_tests(args)

if __name__ == "__main__":
    main()