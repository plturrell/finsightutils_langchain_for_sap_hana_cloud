#!/usr/bin/env python3
"""
API client for testing the Vercel integration

This script allows testing the API endpoints directly, bypassing the frontend.
It's useful for debugging deployment issues and verifying the API functionality.
"""

import argparse
import requests
import json
import sys
import time
from typing import Dict, Any, Optional, List, Union

# Default values
DEFAULT_URL = "http://localhost:8000"
DEFAULT_USERNAME = "admin"
DEFAULT_PASSWORD = "sap-hana-t4-admin"

def login(base_url: str, username: str, password: str) -> Optional[str]:
    """
    Login to the API and get an authentication token
    
    Args:
        base_url: Base URL of the API
        username: Username for authentication
        password: Password for authentication
        
    Returns:
        Optional[str]: Authentication token if successful, None otherwise
    """
    print(f"Logging in as {username}...")
    
    try:
        response = requests.post(
            f"{base_url}/api/auth/token",
            json={"username": username, "password": password},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            token = data.get("access_token")
            print("Login successful.")
            return token
        else:
            print(f"Login failed with status code {response.status_code}")
            print(f"Response: {response.text}")
            return None
    except Exception as e:
        print(f"Login error: {str(e)}")
        return None

def get_api_info(base_url: str) -> None:
    """
    Get API information from the root endpoint
    
    Args:
        base_url: Base URL of the API
    """
    print("Getting API information...")
    
    try:
        response = requests.get(f"{base_url}/api", timeout=10)
        if response.status_code == 200:
            print("API info retrieved successfully:")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"Failed to get API info: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error getting API info: {str(e)}")

def check_health(base_url: str) -> None:
    """
    Check the health of the API
    
    Args:
        base_url: Base URL of the API
    """
    print("Checking API health...")
    
    try:
        response = requests.get(f"{base_url}/api/health", timeout=10)
        if response.status_code == 200:
            print("Health check successful:")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"Health check failed: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error checking health: {str(e)}")

def generate_embeddings(base_url: str, token: Optional[str] = None) -> None:
    """
    Generate embeddings using the API
    
    Args:
        base_url: Base URL of the API
        token: Optional authentication token
    """
    print("Generating embeddings...")
    
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    try:
        response = requests.post(
            f"{base_url}/api/embeddings",
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
            print("Embeddings generated successfully:")
            data = response.json()
            print(f"Model: {data.get('model')}")
            print(f"Dimensions: {data.get('dimensions')}")
            print(f"Processing time: {data.get('processing_time_ms')} ms")
            print(f"GPU used: {data.get('gpu_used')}")
            print(f"Batch size: {data.get('batch_size_used')}")
            print(f"Number of embeddings: {len(data.get('embeddings', []))}")
        else:
            print(f"Failed to generate embeddings: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error generating embeddings: {str(e)}")

def get_metrics(base_url: str, token: Optional[str] = None) -> None:
    """
    Get performance metrics from the API
    
    Args:
        base_url: Base URL of the API
        token: Optional authentication token
    """
    print("Getting performance metrics...")
    
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    try:
        response = requests.get(
            f"{base_url}/api/metrics",
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            print("Metrics retrieved successfully:")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"Failed to get metrics: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error getting metrics: {str(e)}")

def test_all(base_url: str, username: str, password: str) -> None:
    """
    Run all tests
    
    Args:
        base_url: Base URL of the API
        username: Username for authentication
        password: Password for authentication
    """
    # Basic API info
    get_api_info(base_url)
    print("\n" + "-" * 50 + "\n")
    
    # Health check
    check_health(base_url)
    print("\n" + "-" * 50 + "\n")
    
    # Login
    token = login(base_url, username, password)
    print("\n" + "-" * 50 + "\n")
    
    # Generate embeddings
    generate_embeddings(base_url, token)
    print("\n" + "-" * 50 + "\n")
    
    # Get metrics
    get_metrics(base_url, token)

def main():
    parser = argparse.ArgumentParser(description="Test the Vercel API integration")
    
    parser.add_argument("--url", default=DEFAULT_URL, help="Base URL of the API")
    parser.add_argument("--username", default=DEFAULT_USERNAME, help="Username for authentication")
    parser.add_argument("--password", default=DEFAULT_PASSWORD, help="Password for authentication")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Get API information")
    
    # Health command
    health_parser = subparsers.add_parser("health", help="Check API health")
    
    # Login command
    login_parser = subparsers.add_parser("login", help="Test authentication")
    
    # Embeddings command
    embeddings_parser = subparsers.add_parser("embeddings", help="Generate embeddings")
    
    # Metrics command
    metrics_parser = subparsers.add_parser("metrics", help="Get performance metrics")
    
    # All command
    all_parser = subparsers.add_parser("all", help="Run all tests")
    
    args = parser.parse_args()
    
    if args.command == "info":
        get_api_info(args.url)
    elif args.command == "health":
        check_health(args.url)
    elif args.command == "login":
        token = login(args.url, args.username, args.password)
        if token:
            print(f"Token: {token}")
    elif args.command == "embeddings":
        token = login(args.url, args.username, args.password)
        generate_embeddings(args.url, token)
    elif args.command == "metrics":
        token = login(args.url, args.username, args.password)
        get_metrics(args.url, token)
    elif args.command == "all" or not args.command:
        test_all(args.url, args.username, args.password)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()