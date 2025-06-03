"""Example client for the SAP HANA Cloud Vector Store API."""

import json
import requests

# API base URL
BASE_URL = "http://localhost:8000"


def check_health():
    """Check the health of the API."""
    response = requests.get(f"{BASE_URL}/health")
    print(f"Health check: {response.status_code}")
    print(json.dumps(response.json(), indent=2))


def add_texts():
    """Add texts to the vector store."""
    texts = [
        "SAP HANA Cloud is a cloud-based database management system.",
        "Vector search enables semantic similarity searches.",
        "LangChain is a framework for building LLM-powered applications.",
        "Integration allows SAP HANA Cloud to be used as a vector store in LangChain.",
    ]
    
    metadatas = [
        {"source": "docs", "topic": "database"},
        {"source": "docs", "topic": "search"},
        {"source": "docs", "topic": "framework"},
        {"source": "docs", "topic": "integration"},
    ]
    
    payload = {
        "texts": texts,
        "metadatas": metadatas,
    }
    
    response = requests.post(f"{BASE_URL}/texts", json=payload)
    print(f"Add texts: {response.status_code}")
    print(json.dumps(response.json(), indent=2))


def query_text():
    """Query the vector store with text."""
    payload = {
        "query": "How does SAP HANA Cloud work with LangChain?",
        "k": 2,
        "filter": {"source": "docs"},
    }
    
    response = requests.post(f"{BASE_URL}/query", json=payload)
    print(f"Query: {response.status_code}")
    print(json.dumps(response.json(), indent=2))


def delete_documents():
    """Delete documents from the vector store."""
    payload = {
        "filter": {"topic": "database"},
    }
    
    response = requests.post(f"{BASE_URL}/delete", json=payload)
    print(f"Delete: {response.status_code}")
    print(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    print("=== Testing SAP HANA Cloud Vector Store API ===")
    
    # Check health
    print("\n=== Health Check ===")
    check_health()
    
    # Add texts
    print("\n=== Adding Texts ===")
    add_texts()
    
    # Query
    print("\n=== Querying ===")
    query_text()
    
    # Delete
    print("\n=== Deleting ===")
    delete_documents()
    
    print("\n=== Test Complete ===")