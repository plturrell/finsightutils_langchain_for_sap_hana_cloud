"""
Cross-platform deployment integration tests.

This module contains tests that verify the application works correctly
across different deployment platforms (NVIDIA, Together.ai, SAP BTP, Vercel).
"""

import os
import pytest
import requests
import json
from unittest import mock
from typing import Dict, Any, List, Optional

# Import test utilities
from tests.integration_tests.hana_test_utils import (
    get_test_connection,
    create_test_table,
    drop_test_table,
)
from tests.integration_tests.fake_embeddings import FakeEmbeddings

# Set environment variables for testing
os.environ["PLATFORM"] = "test"
os.environ["PLATFORM_SUPPORTS_GPU"] = "false"
os.environ["VERSION"] = "1.2.0-test"

# Import app for API testing
from api import app
from fastapi.testclient import TestClient

# Create test client
client = TestClient(app)

# Mock platform-specific behavior
class MockPlatformBehavior:
    """Class to mock platform-specific behavior for testing."""
    
    @staticmethod
    def set_platform(platform: str, gpu_support: bool = False):
        """Set platform for testing."""
        os.environ["PLATFORM"] = platform
        os.environ["PLATFORM_SUPPORTS_GPU"] = str(gpu_support).lower()
        
    @staticmethod
    def reset_platform():
        """Reset platform to test defaults."""
        os.environ["PLATFORM"] = "test"
        os.environ["PLATFORM_SUPPORTS_GPU"] = "false"

# Test fixtures
@pytest.fixture
def setup_test_env():
    """Set up test environment."""
    # Store original environment
    original_env = os.environ.copy()
    
    # Set test environment
    os.environ["PLATFORM"] = "test"
    os.environ["PLATFORM_SUPPORTS_GPU"] = "false"
    os.environ["VERSION"] = "1.2.0-test"
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)

@pytest.fixture
def mock_gpu_utils():
    """Mock GPU utils for testing."""
    with mock.patch("api.gpu_utils.is_gpu_available", return_value=True):
        with mock.patch("api.gpu_utils.get_gpu_info", return_value={
            "device_count": 2,
            "devices": [
                {
                    "name": "NVIDIA Tesla T4",
                    "memory_total": 16000000000,
                    "memory_free": 14000000000,
                    "memory_used": 2000000000
                },
                {
                    "name": "NVIDIA Tesla T4",
                    "memory_total": 16000000000,
                    "memory_free": 15000000000,
                    "memory_used": 1000000000
                }
            ]
        }):
            yield

@pytest.fixture
def mock_database():
    """Mock database connection for testing."""
    with mock.patch("api.database.get_db_connection") as mock_get_db:
        # Create mock connection
        mock_conn = mock.MagicMock()
        mock_cursor = mock.MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        
        # Set up cursor to return success for health check
        mock_cursor.execute.return_value = None
        mock_cursor.has_result_set.return_value = True
        mock_cursor.fetchall.return_value = [[1]]
        
        # Return mock connection
        mock_get_db.return_value = mock_conn
        yield mock_conn

# Tests for cross-platform functionality
@pytest.mark.parametrize("platform,gpu_support", [
    ("vercel", False),
    ("nvidia", True),
    ("together", True),
    ("sap_btp", True),
    ("unknown", False)
])
def test_health_endpoint_platform_detection(setup_test_env, mock_database, platform, gpu_support):
    """Test that health endpoint correctly detects and reports platform."""
    # Set platform for test
    MockPlatformBehavior.set_platform(platform, gpu_support)
    
    # Test health endpoint
    response = client.get("/api/health")
    assert response.status_code == 200
    
    # Check platform info is correctly reported
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"
    
    # Reset platform after test
    MockPlatformBehavior.reset_platform()

@pytest.mark.parametrize("platform", ["vercel", "nvidia", "together", "sap_btp"])
def test_deployment_info_endpoint(setup_test_env, platform):
    """Test deployment info endpoint for different platforms."""
    # Set platform for test
    MockPlatformBehavior.set_platform(platform, platform != "vercel")
    
    # Test deployment info endpoint
    response = client.get("/api/deployment/info")
    assert response.status_code == 200
    
    # Check deployment info
    data = response.json()
    assert data["deployment"] == platform
    assert data["version"] == "1.2.0-test"
    assert "features_enabled" in data
    assert "supports_gpu" in data
    assert data["supports_gpu"] == (platform != "vercel")
    
    # Reset platform after test
    MockPlatformBehavior.reset_platform()

@pytest.mark.parametrize("platform", ["vercel", "nvidia", "together", "sap_btp"])
def test_platform_headers(setup_test_env, platform):
    """Test that platform-specific headers are set correctly."""
    # Set platform for test
    MockPlatformBehavior.set_platform(platform)
    
    # Test any endpoint to check headers
    response = client.get("/")
    assert response.status_code == 200
    
    # Check headers
    assert "X-Platform" in response.headers
    assert response.headers["X-Platform"] == platform
    assert "X-API-Version" in response.headers
    assert response.headers["X-API-Version"] == "1.2.0-test"
    assert "X-Process-Time" in response.headers
    
    # Reset platform after test
    MockPlatformBehavior.reset_platform()

@pytest.mark.parametrize("platform", ["nvidia", "together", "sap_btp"])
def test_gpu_health_endpoint(setup_test_env, mock_gpu_utils, platform):
    """Test GPU health endpoint for platforms with GPU support."""
    # Set platform for test
    MockPlatformBehavior.set_platform(platform, True)
    
    # Test GPU health endpoint
    response = client.get("/api/health/gpu")
    assert response.status_code == 200
    
    # Check GPU info
    data = response.json()
    assert data["status"] == "ok"
    assert "gpu" in data
    assert data["gpu"]["available"] is True
    assert data["gpu"]["count"] == 2
    assert len(data["gpu"]["devices"]) == 2
    
    # Reset platform after test
    MockPlatformBehavior.reset_platform()

def test_vercel_gpu_health_endpoint(setup_test_env):
    """Test GPU health endpoint for Vercel (which doesn't support GPU)."""
    # Set platform for test
    MockPlatformBehavior.set_platform("vercel", False)
    
    # Test GPU health endpoint
    response = client.get("/api/health/gpu")
    assert response.status_code == 200
    
    # Check GPU info
    data = response.json()
    assert data["status"] == "unavailable"
    assert "gpu" in data
    assert data["gpu"]["available"] is False
    assert "error" in data["gpu"]
    assert "Vercel does not support GPU" in data["gpu"]["error"]
    
    # Reset platform after test
    MockPlatformBehavior.reset_platform()

@pytest.mark.parametrize("platform", ["vercel", "nvidia", "together", "sap_btp"])
def test_complete_health_check(setup_test_env, mock_database, mock_gpu_utils, platform):
    """Test complete health check endpoint for different platforms."""
    # Set platform for test
    MockPlatformBehavior.set_platform(platform, platform != "vercel")
    
    # Test complete health check endpoint
    response = client.get("/api/health/complete")
    assert response.status_code == 200
    
    # Check complete health info
    data = response.json()
    assert "status" in data
    assert "timestamp" in data
    assert "version" in data
    assert "environment" in data
    assert "database" in data
    assert "gpu" in data
    assert "system" in data
    assert "dependencies" in data
    assert "platform_info" in data
    
    # Platform-specific checks
    assert data["platform_info"]["platform"] == platform
    
    if platform == "vercel":
        assert data["gpu"]["available"] is False
        assert "error" in data["gpu"]
    else:
        assert data["gpu"]["available"] is True
        assert "count" in data["gpu"]
        assert "devices" in data["gpu"]
    
    # Reset platform after test
    MockPlatformBehavior.reset_platform()

@pytest.mark.parametrize("platform", ["vercel", "nvidia", "together", "sap_btp"])
def test_metrics_endpoint(setup_test_env, mock_database, platform):
    """Test Prometheus metrics endpoint for different platforms."""
    # Set platform for test
    MockPlatformBehavior.set_platform(platform, platform != "vercel")
    
    # Test metrics endpoint
    response = client.get("/api/health/metrics")
    assert response.status_code == 200
    
    # Check metrics format (Prometheus text format)
    metrics = response.text
    assert "# HELP" in metrics
    assert "# TYPE" in metrics
    assert "api_uptime_seconds" in metrics
    assert "api_memory_usage_percent" in metrics
    assert "api_database_connected" in metrics
    assert "api_gpu_available" in metrics
    
    # Check specific metrics based on platform
    if platform == "vercel":
        assert "api_gpu_available 0" in metrics
    else:
        assert "api_gpu_available 1" in metrics
    
    # Reset platform after test
    MockPlatformBehavior.reset_platform()

# Tests for error handling across platforms
@pytest.mark.parametrize("platform", ["vercel", "nvidia", "together", "sap_btp"])
def test_error_handling(setup_test_env, platform):
    """Test error handling across different platforms."""
    # Set platform for test
    MockPlatformBehavior.set_platform(platform)
    
    # Test invalid endpoint to trigger error
    response = client.get("/invalid_endpoint")
    assert response.status_code == 404
    
    # Check error response format
    data = response.json()
    assert "detail" in data
    
    # Reset platform after test
    MockPlatformBehavior.reset_platform()

# Performance tests across platforms
@pytest.mark.parametrize("platform", ["vercel", "nvidia", "together", "sap_btp"])
def test_platform_performance(setup_test_env, platform):
    """Test performance metrics across different platforms."""
    # Set platform for test
    MockPlatformBehavior.set_platform(platform, platform != "vercel")
    
    # Make multiple requests to measure performance
    start_times = []
    response_times = []
    for _ in range(5):
        # Record start time
        import time
        start_time = time.time()
        
        # Make request
        response = client.get("/api/health")
        assert response.status_code == 200
        
        # Record response time
        response_time = float(response.headers["X-Process-Time"])
        start_times.append(start_time)
        response_times.append(response_time)
    
    # Check performance metrics
    avg_response_time = sum(response_times) / len(response_times)
    max_response_time = max(response_times)
    min_response_time = min(response_times)
    
    # Log performance metrics
    print(f"Platform: {platform}")
    print(f"Average response time: {avg_response_time:.6f}s")
    print(f"Maximum response time: {max_response_time:.6f}s")
    print(f"Minimum response time: {min_response_time:.6f}s")
    
    # Reset platform after test
    MockPlatformBehavior.reset_platform()

# Platform-specific tests
@pytest.mark.parametrize("platform,gpu_support", [
    ("nvidia", True),
    ("together", True),
    ("sap_btp", True)
])
def test_gpu_platforms(setup_test_env, mock_gpu_utils, platform, gpu_support):
    """Test platforms with GPU support."""
    # Set platform for test
    MockPlatformBehavior.set_platform(platform, gpu_support)
    
    # Test GPU info endpoint
    response = client.get("/api/health/gpu")
    assert response.status_code == 200
    
    # Check GPU info
    data = response.json()
    assert data["gpu"]["available"] is True
    
    # Platform-specific checks
    if platform == "nvidia":
        assert data["gpu"]["tensorrt_available"] is True
    elif platform == "together":
        assert len(data["gpu"]["devices"]) == 1
        assert "Together.ai managed GPU" in data["gpu"]["devices"][0]["name"]
    
    # Reset platform after test
    MockPlatformBehavior.reset_platform()

# Simulate cross-platform database errors
@pytest.mark.parametrize("platform", ["vercel", "nvidia", "together", "sap_btp"])
def test_database_error_handling(setup_test_env, platform):
    """Test database error handling across different platforms."""
    # Set platform for test
    MockPlatformBehavior.set_platform(platform)
    
    with mock.patch("api.database.get_db_connection", side_effect=Exception("Database connection error")):
        # Test database health endpoint
        response = client.get("/api/health/database")
        assert response.status_code == 200
        
        # Check error response
        data = response.json()
        assert data["status"] == "error"
        assert "database" in data
        assert data["database"]["connected"] is False
        assert "error" in data["database"]
    
    # Reset platform after test
    MockPlatformBehavior.reset_platform()

# Test CORS headers for frontend integration
@pytest.mark.parametrize("platform", ["vercel", "nvidia", "together", "sap_btp"])
def test_cors_headers(setup_test_env, platform):
    """Test CORS headers for frontend integration across platforms."""
    # Set platform for test
    MockPlatformBehavior.set_platform(platform)
    
    # Test CORS preflight request
    response = client.options("/api/health", headers={
        "Access-Control-Request-Method": "GET",
        "Origin": "https://example.com"
    })
    assert response.status_code == 200
    
    # Check CORS headers
    assert "access-control-allow-origin" in response.headers
    assert response.headers["access-control-allow-origin"] == "*"
    assert "access-control-allow-methods" in response.headers
    assert "access-control-allow-headers" in response.headers
    
    # Reset platform after test
    MockPlatformBehavior.reset_platform()

if __name__ == "__main__":
    pytest.main(["-xvs", __file__])