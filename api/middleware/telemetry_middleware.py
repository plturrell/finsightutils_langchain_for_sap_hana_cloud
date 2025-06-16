"""
TelemetryMiddleware for the SAP HANA LangChain Integration API.

This middleware collects telemetry data about API usage, performance, and errors,
supporting observability through various metrics collection systems.
"""

import time
import logging
import os
import threading
import uuid
from collections import defaultdict, deque
from typing import Dict, List, Optional, Union, Callable, Deque, Any

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from ..config_standardized import get_standardized_settings

# Get settings
settings = get_standardized_settings()

# Setup logging
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import prometheus_client
    from prometheus_client import Counter, Histogram, Gauge
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not available, Prometheus metrics will be disabled")

try:
    from opentelemetry import trace, metrics
    from opentelemetry.trace import Span, StatusCode
    from opentelemetry.metrics import Meter
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    logger.warning("opentelemetry not available, OpenTelemetry metrics will be disabled")


class Metrics:
    """Class to manage metrics collection."""
    
    def __init__(self, enabled: bool = True):
        """
        Initialize metrics collection.
        
        Args:
            enabled: Whether metrics collection is enabled
        """
        self.enabled = enabled
        self.prometheus_enabled = enabled and PROMETHEUS_AVAILABLE and settings.telemetry.prometheus_enabled
        self.opentelemetry_enabled = enabled and OPENTELEMETRY_AVAILABLE and settings.telemetry.opentelemetry_enabled
        
        # In-memory metrics for when external systems are not available
        self.memory_metrics = {
            "request_count": defaultdict(int),
            "request_latency": defaultdict(list),
            "error_count": defaultdict(int),
            "status_codes": defaultdict(int),
        }
        
        # Keep a fixed-size queue of recent requests for reporting
        self.recent_requests: Deque[Dict[str, Any]] = deque(maxlen=100)
        
        # Initialize Prometheus metrics if available
        if self.prometheus_enabled:
            self._init_prometheus_metrics()
        
        # Initialize OpenTelemetry metrics if available
        if self.opentelemetry_enabled:
            self._init_opentelemetry_metrics()
        
        logger.info(f"Metrics initialized (enabled: {enabled}, prometheus: {self.prometheus_enabled}, opentelemetry: {self.opentelemetry_enabled})")
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics."""
        self.prom_requests = Counter(
            "api_requests_total",
            "Total number of API requests",
            ["method", "path", "endpoint"]
        )
        self.prom_responses = Counter(
            "api_responses_total",
            "Total number of API responses",
            ["method", "path", "endpoint", "status_code"]
        )
        self.prom_errors = Counter(
            "api_errors_total",
            "Total number of API errors",
            ["method", "path", "endpoint", "error_type"]
        )
        self.prom_latency = Histogram(
            "api_request_latency_seconds",
            "API request latency in seconds",
            ["method", "path", "endpoint"],
            buckets=[0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 30, 60]
        )
    
    def _init_opentelemetry_metrics(self):
        """Initialize OpenTelemetry metrics."""
        self.meter = metrics.get_meter("sap_hana_langchain_api")
        
        self.otel_requests_counter = self.meter.create_counter(
            "api.requests",
            description="Total number of API requests",
            unit="1"
        )
        self.otel_response_counter = self.meter.create_counter(
            "api.responses",
            description="Total number of API responses",
            unit="1"
        )
        self.otel_errors_counter = self.meter.create_counter(
            "api.errors",
            description="Total number of API errors",
            unit="1"
        )
        self.otel_latency_histogram = self.meter.create_histogram(
            "api.request.latency",
            description="API request latency",
            unit="s"
        )
    
    def record_request(self, method: str, path: str, endpoint: str):
        """
        Record an API request.
        
        Args:
            method: HTTP method
            path: Request path
            endpoint: Endpoint name
        """
        if not self.enabled:
            return
        
        # Record in memory metrics
        key = f"{method}:{path}"
        self.memory_metrics["request_count"][key] += 1
        
        # Record in Prometheus metrics
        if self.prometheus_enabled:
            self.prom_requests.labels(method=method, path=path, endpoint=endpoint).inc()
        
        # Record in OpenTelemetry metrics
        if self.opentelemetry_enabled:
            self.otel_requests_counter.add(
                1,
                {"method": method, "path": path, "endpoint": endpoint}
            )
    
    def record_response(
        self, method: str, path: str, endpoint: str, status_code: int, latency: float
    ):
        """
        Record an API response.
        
        Args:
            method: HTTP method
            path: Request path
            endpoint: Endpoint name
            status_code: HTTP status code
            latency: Request latency in seconds
        """
        if not self.enabled:
            return
        
        # Record in memory metrics
        key = f"{method}:{path}"
        self.memory_metrics["status_codes"][status_code] += 1
        self.memory_metrics["request_latency"][key].append(latency)
        
        # Keep only the last 100 latency measurements
        if len(self.memory_metrics["request_latency"][key]) > 100:
            self.memory_metrics["request_latency"][key].pop(0)
        
        # Record in recent requests
        self.recent_requests.append({
            "method": method,
            "path": path,
            "endpoint": endpoint,
            "status_code": status_code,
            "latency": latency,
            "timestamp": time.time()
        })
        
        # Record in Prometheus metrics
        if self.prometheus_enabled:
            self.prom_responses.labels(
                method=method, path=path, endpoint=endpoint, status_code=status_code
            ).inc()
            self.prom_latency.labels(
                method=method, path=path, endpoint=endpoint
            ).observe(latency)
        
        # Record in OpenTelemetry metrics
        if self.opentelemetry_enabled:
            self.otel_response_counter.add(
                1,
                {
                    "method": method,
                    "path": path,
                    "endpoint": endpoint,
                    "status_code": str(status_code)
                }
            )
            self.otel_latency_histogram.record(
                latency,
                {"method": method, "path": path, "endpoint": endpoint}
            )
    
    def record_error(
        self, method: str, path: str, endpoint: str, error_type: str, error: Exception
    ):
        """
        Record an API error.
        
        Args:
            method: HTTP method
            path: Request path
            endpoint: Endpoint name
            error_type: Error type
            error: Exception object
        """
        if not self.enabled:
            return
        
        # Record in memory metrics
        key = f"{method}:{path}"
        error_key = f"{key}:{error_type}"
        self.memory_metrics["error_count"][error_key] += 1
        
        # Record in Prometheus metrics
        if self.prometheus_enabled:
            self.prom_errors.labels(
                method=method, path=path, endpoint=endpoint, error_type=error_type
            ).inc()
        
        # Record in OpenTelemetry metrics
        if self.opentelemetry_enabled:
            self.otel_errors_counter.add(
                1,
                {
                    "method": method,
                    "path": path,
                    "endpoint": endpoint,
                    "error_type": error_type
                }
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current metrics.
        
        Returns:
            Dictionary of metrics
        """
        # Calculate aggregate metrics
        total_requests = sum(self.memory_metrics["request_count"].values())
        total_errors = sum(self.memory_metrics["error_count"].values())
        error_rate = total_errors / total_requests if total_requests > 0 else 0
        
        # Calculate latency statistics
        all_latencies = []
        for latencies in self.memory_metrics["request_latency"].values():
            all_latencies.extend(latencies)
        
        avg_latency = sum(all_latencies) / len(all_latencies) if all_latencies else 0
        max_latency = max(all_latencies) if all_latencies else 0
        
        # Calculate percentiles
        sorted_latencies = sorted(all_latencies)
        p50 = sorted_latencies[int(len(sorted_latencies) * 0.5)] if sorted_latencies else 0
        p90 = sorted_latencies[int(len(sorted_latencies) * 0.9)] if sorted_latencies else 0
        p95 = sorted_latencies[int(len(sorted_latencies) * 0.95)] if sorted_latencies else 0
        p99 = sorted_latencies[int(len(sorted_latencies) * 0.99)] if sorted_latencies else 0
        
        # Return metrics
        return {
            "total_requests": total_requests,
            "total_errors": total_errors,
            "error_rate": error_rate,
            "latency": {
                "avg": avg_latency,
                "max": max_latency,
                "p50": p50,
                "p90": p90,
                "p95": p95,
                "p99": p99,
            },
            "status_codes": dict(self.memory_metrics["status_codes"]),
            "request_count": dict(self.memory_metrics["request_count"]),
            "error_count": dict(self.memory_metrics["error_count"]),
            "recent_requests": list(self.recent_requests),
        }


class TelemetryMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting telemetry data about API usage."""
    
    def __init__(
        self,
        app: FastAPI,
        enabled: bool = None,
        metrics_enabled: bool = None,
        tracing_enabled: bool = None,
        exclude_paths: List[str] = None,
    ):
        """
        Initialize the middleware.
        
        Args:
            app: FastAPI application
            enabled: Whether telemetry is enabled
            metrics_enabled: Whether metrics collection is enabled
            tracing_enabled: Whether tracing is enabled
            exclude_paths: List of paths to exclude from telemetry
        """
        super().__init__(app)
        
        # Set defaults from settings if not provided
        self.enabled = enabled if enabled is not None else settings.telemetry.enabled
        self.metrics_enabled = metrics_enabled if metrics_enabled is not None else settings.telemetry.metrics_enabled
        self.tracing_enabled = tracing_enabled if tracing_enabled is not None else settings.telemetry.tracing_enabled
        self.exclude_paths = exclude_paths or settings.telemetry.exclude_paths
        
        # Add standard excluded paths
        self.exclude_paths.extend([
            "/metrics",
            "/health",
            "/api/health",
            "/api/v1/health",
            "/api/v2/health",
        ])
        
        # Initialize metrics
        self.metrics = Metrics(enabled=self.metrics_enabled)
        
        # Initialize tracer if OpenTelemetry is available
        self.tracer = None
        if self.tracing_enabled and OPENTELEMETRY_AVAILABLE:
            self.tracer = trace.get_tracer("sap_hana_langchain_api")
        
        logger.info(f"Telemetry middleware initialized (enabled: {self.enabled}, metrics: {self.metrics_enabled}, tracing: {self.tracing_enabled})")
    
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """
        Process the request and collect telemetry data.
        
        Args:
            request: FastAPI request object
            call_next: Next middleware or route handler
            
        Returns:
            Response from the next middleware or route handler
        """
        # Skip telemetry if disabled
        if not self.enabled:
            return await call_next(request)
        
        # Check if this path should be excluded
        path = request.url.path
        for excluded_path in self.exclude_paths:
            if path.startswith(excluded_path):
                return await call_next(request)
        
        # Extract request information
        method = request.method
        endpoint = request.scope.get("endpoint", None)
        endpoint_name = endpoint.__name__ if endpoint and hasattr(endpoint, "__name__") else "unknown"
        
        # Generate a span ID for tracing if not already present
        span_id = getattr(request.state, "span_id", str(uuid.uuid4()))
        request.state.span_id = span_id
        
        # Start a span if tracing is enabled
        span = None
        if self.tracer:
            span = self.tracer.start_span(f"{method} {path}")
            span.set_attribute("http.method", method)
            span.set_attribute("http.url", str(request.url))
            span.set_attribute("http.route", path)
            span.set_attribute("endpoint", endpoint_name)
            
            # Add request ID to span if available
            request_id = getattr(request.state, "request_id", None)
            if request_id:
                span.set_attribute("request_id", request_id)
        
        # Record the request
        self.metrics.record_request(method, path, endpoint_name)
        
        # Start timing
        start_time = time.time()
        
        try:
            # Process the request
            response = await call_next(request)
            
            # Calculate latency
            latency = time.time() - start_time
            
            # Record the response
            self.metrics.record_response(method, path, endpoint_name, response.status_code, latency)
            
            # Add tracing information to span
            if span:
                span.set_attribute("http.status_code", response.status_code)
                span.set_status(StatusCode.OK if response.status_code < 400 else StatusCode.ERROR)
                span.end()
            
            # Add latency header
            response.headers["X-Response-Time"] = f"{latency:.6f}"
            
            return response
        except Exception as e:
            # Record the error
            error_type = type(e).__name__
            self.metrics.record_error(method, path, endpoint_name, error_type, e)
            
            # Add error information to span
            if span:
                span.set_status(StatusCode.ERROR)
                span.record_exception(e)
                span.end()
            
            # Re-raise the exception
            raise
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current metrics.
        
        Returns:
            Dictionary of metrics
        """
        return self.metrics.get_metrics()


def setup_telemetry_middleware(
    app: FastAPI,
    enabled: bool = None,
    metrics_enabled: bool = None,
    tracing_enabled: bool = None,
    exclude_paths: List[str] = None,
) -> None:
    """
    Configure and add the telemetry middleware to the application.
    
    Args:
        app: FastAPI application
        enabled: Whether telemetry is enabled
        metrics_enabled: Whether metrics collection is enabled
        tracing_enabled: Whether tracing is enabled
        exclude_paths: List of paths to exclude from telemetry
    """
    middleware = TelemetryMiddleware(
        app,
        enabled=enabled,
        metrics_enabled=metrics_enabled,
        tracing_enabled=tracing_enabled,
        exclude_paths=exclude_paths,
    )
    
    # Expose Prometheus metrics endpoint if enabled
    if middleware.metrics_enabled and PROMETHEUS_AVAILABLE and settings.telemetry.prometheus_enabled:
        @app.get("/metrics", include_in_schema=False)
        def metrics():
            """Endpoint to expose Prometheus metrics."""
            return Response(
                content=prometheus_client.generate_latest(),
                media_type="text/plain"
            )
    
    # Expose telemetry metrics endpoint
    @app.get("/api/telemetry/metrics", include_in_schema=True)
    def telemetry_metrics():
        """Endpoint to expose telemetry metrics."""
        return middleware.get_metrics()
    
    # Add the middleware
    app.add_middleware(TelemetryMiddleware)
    
    return middleware