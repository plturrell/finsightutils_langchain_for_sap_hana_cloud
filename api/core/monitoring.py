"""
Monitoring module for SAP HANA Cloud LangChain Integration.

This module provides:
1. GPU monitoring endpoints and metrics collection
2. Prometheus metrics export
3. Health check endpoints
4. DCGM integration for comprehensive GPU monitoring
"""

import json
import logging
import os
import platform
import time
from typing import Dict, List, Optional, Union

import psutil
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from prometheus_client import REGISTRY, Counter, Gauge, Histogram, generate_latest

# Import GPU utilities conditionally
try:
    import torch
    import torch.cuda as cuda
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False

try:
    import pynvml
    from pynvml import nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex
    from pynvml import nvmlDeviceGetName, nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates
    HAS_NVML = True
    nvmlInit()
except ImportError:
    HAS_NVML = False

try:
    from opentelemetry import metrics
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader
    HAS_OPENTELEMETRY = True
except ImportError:
    HAS_OPENTELEMETRY = False

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/monitoring", tags=["Monitoring"])

# Define metrics
if HAS_OPENTELEMETRY:
    # Setup OpenTelemetry
    exporter = ConsoleMetricExporter()
    reader = PeriodicExportingMetricReader(exporter, export_interval_millis=15000)
    provider = MeterProvider(metric_readers=[reader])
    metrics.set_meter_provider(provider)
    meter = metrics.get_meter("sap_hana_langchain")

    # Define custom metrics
    embedding_request_counter = meter.create_counter(
        name="embedding_requests",
        description="Number of embedding requests",
        unit="requests",
    )
    
    vector_search_counter = meter.create_counter(
        name="vector_searches",
        description="Number of vector search operations",
        unit="searches",
    )
    
    embedding_processing_time = meter.create_histogram(
        name="embedding_processing_time",
        description="Time taken to process embedding requests",
        unit="seconds",
    )
    
    gpu_memory_usage = meter.create_gauge(
        name="gpu_memory_usage",
        description="GPU memory usage in MiB",
        unit="MiB",
    )
    
    gpu_utilization = meter.create_gauge(
        name="gpu_utilization",
        description="GPU utilization percentage",
        unit="percent",
    )
else:
    # Fallback to Prometheus client library
    embedding_request_counter = Counter(
        "embedding_requests_total",
        "Number of embedding requests",
        ["model", "batch_size"],
    )
    
    vector_search_counter = Counter(
        "vector_searches_total",
        "Number of vector search operations",
        ["filter_applied", "k"],
    )
    
    embedding_processing_time = Histogram(
        "embedding_processing_time_seconds",
        "Time taken to process embedding requests",
        ["model", "batch_size", "precision"],
        buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 30, 60],
    )
    
    gpu_memory_usage = Gauge(
        "gpu_memory_usage_mib",
        "GPU memory usage in MiB",
        ["device", "type"],
    )
    
    gpu_utilization = Gauge(
        "gpu_utilization_percent",
        "GPU utilization percentage",
        ["device", "type"],
    )


def get_system_info() -> Dict[str, Union[str, Dict]]:
    """Get system information."""
    system_info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "hostname": platform.node(),
        "cpu_count": os.cpu_count(),
        "memory_total": psutil.virtual_memory().total // (1024 * 1024),  # MB
        "memory_available": psutil.virtual_memory().available // (1024 * 1024),  # MB
        "memory_percent": psutil.virtual_memory().percent,
        "disk_usage": {
            "total": psutil.disk_usage("/").total // (1024 * 1024),  # MB
            "used": psutil.disk_usage("/").used // (1024 * 1024),  # MB
            "free": psutil.disk_usage("/").free // (1024 * 1024),  # MB
            "percent": psutil.disk_usage("/").percent,
        },
    }
    
    return system_info


def get_gpu_info() -> List[Dict[str, Union[str, int, float]]]:
    """Get GPU information."""
    gpu_info = []
    
    if HAS_CUDA:
        cuda_available = cuda.is_available()
        device_count = cuda.device_count() if cuda_available else 0
        
        for i in range(device_count):
            device_info = {
                "index": i,
                "name": cuda.get_device_name(i),
                "total_memory": cuda.get_device_properties(i).total_memory // (1024 * 1024),  # MB
                "compute_capability": f"{cuda.get_device_capability(i)[0]}.{cuda.get_device_capability(i)[1]}",
            }
            
            # Add NVML data if available
            if HAS_NVML:
                try:
                    handle = nvmlDeviceGetHandleByIndex(i)
                    device_info["nvml_name"] = nvmlDeviceGetName(handle)
                    memory_info = nvmlDeviceGetMemoryInfo(handle)
                    device_info["memory_used"] = memory_info.used // (1024 * 1024)  # MB
                    device_info["memory_free"] = memory_info.free // (1024 * 1024)  # MB
                    
                    util_rates = nvmlDeviceGetUtilizationRates(handle)
                    device_info["gpu_utilization"] = util_rates.gpu
                    device_info["memory_utilization"] = util_rates.memory
                    
                    # Update Prometheus metrics
                    gpu_memory_usage.labels(device=i, type=device_info["name"]).set(
                        device_info["memory_used"]
                    )
                    gpu_utilization.labels(device=i, type=device_info["name"]).set(
                        device_info["gpu_utilization"]
                    )
                except Exception as e:
                    logger.warning(f"Error getting NVML info: {e}")
            
            gpu_info.append(device_info)
    
    return gpu_info


@router.get("/health/ping")
def health_ping():
    """Simple health check endpoint."""
    return {"status": "ok", "timestamp": time.time()}


@router.get("/health/status")
def health_status():
    """Comprehensive health check endpoint."""
    gpu_available = HAS_CUDA and cuda.is_available()
    device_count = cuda.device_count() if gpu_available else 0
    
    status = {
        "status": "ok",
        "timestamp": time.time(),
        "services": {
            "api": {"status": "ok"},
            "gpu": {"status": "ok" if gpu_available else "unavailable", "device_count": device_count},
            "tensorrt": {"status": "ok" if os.environ.get("USE_TENSORRT", "true").lower() == "true" else "disabled"},
            "dcgm": {"status": "ok" if HAS_NVML else "unavailable"},
        },
        "system": get_system_info(),
        "uptime": time.time() - psutil.boot_time(),
    }
    
    return status


@router.get("/gpu/info")
def gpu_info():
    """Get GPU information."""
    if not HAS_CUDA:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="CUDA is not available in this environment",
        )
    
    gpu_available = cuda.is_available()
    if not gpu_available:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No CUDA-capable GPU is available",
        )
    
    return {
        "cuda_available": gpu_available,
        "device_count": cuda.device_count(),
        "current_device": cuda.current_device(),
        "devices": get_gpu_info(),
        "tensorrt_enabled": os.environ.get("USE_TENSORRT", "true").lower() == "true",
        "nvml_available": HAS_NVML,
    }


@router.get("/gpu/memory")
def gpu_memory():
    """Get detailed GPU memory information."""
    if not HAS_CUDA or not cuda.is_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="CUDA-capable GPU is not available",
        )
    
    memory_info = []
    device_count = cuda.device_count()
    
    for i in range(device_count):
        cuda.set_device(i)
        device_memory = {
            "device": i,
            "name": cuda.get_device_name(i),
            "total_memory": cuda.get_device_properties(i).total_memory // (1024 * 1024),  # MB
        }
        
        # Get memory allocated by torch
        device_memory["allocated_memory"] = cuda.memory_allocated(i) // (1024 * 1024)  # MB
        device_memory["cached_memory"] = cuda.memory_reserved(i) // (1024 * 1024)  # MB
        
        # Get NVML data if available
        if HAS_NVML:
            try:
                handle = nvmlDeviceGetHandleByIndex(i)
                memory_info_nvml = nvmlDeviceGetMemoryInfo(handle)
                device_memory["nvml_total"] = memory_info_nvml.total // (1024 * 1024)  # MB
                device_memory["nvml_used"] = memory_info_nvml.used // (1024 * 1024)  # MB
                device_memory["nvml_free"] = memory_info_nvml.free // (1024 * 1024)  # MB
            except Exception as e:
                logger.warning(f"Error getting NVML memory info: {e}")
        
        memory_info.append(device_memory)
    
    return {
        "memory_info": memory_info,
        "total_allocated": sum(m.get("allocated_memory", 0) for m in memory_info),
        "total_cached": sum(m.get("cached_memory", 0) for m in memory_info),
    }


@router.get("/metrics")
def metrics():
    """Export Prometheus metrics."""
    # Update system metrics
    if HAS_NVML:
        for i in range(nvmlDeviceGetCount()):
            try:
                handle = nvmlDeviceGetHandleByIndex(i)
                name = nvmlDeviceGetName(handle)
                memory_info = nvmlDeviceGetMemoryInfo(handle)
                util_rates = nvmlDeviceGetUtilizationRates(handle)
                
                gpu_memory_usage.labels(device=i, type=name).set(
                    memory_info.used // (1024 * 1024)
                )
                gpu_utilization.labels(device=i, type=name).set(util_rates.gpu)
            except Exception as e:
                logger.warning(f"Error updating GPU metrics: {e}")
    
    # Generate metrics output
    return Response(content=generate_latest(REGISTRY), media_type="text/plain")


# Function to track embedding request metrics
def track_embedding_request(
    model_name: str,
    batch_size: int,
    processing_time: float,
    precision: str = "fp16",
) -> None:
    """Track embedding request metrics."""
    if HAS_OPENTELEMETRY:
        embedding_request_counter.add(1, {"model": model_name, "batch_size": batch_size})
        embedding_processing_time.record(
            processing_time,
            {"model": model_name, "batch_size": batch_size, "precision": precision},
        )
    else:
        embedding_request_counter.labels(model=model_name, batch_size=batch_size).inc()
        embedding_processing_time.labels(
            model=model_name, batch_size=batch_size, precision=precision
        ).observe(processing_time)


# Function to track vector search metrics
def track_vector_search(
    filter_applied: bool,
    k: int,
) -> None:
    """Track vector search metrics."""
    if HAS_OPENTELEMETRY:
        vector_search_counter.add(
            1, {"filter_applied": str(filter_applied), "k": k}
        )
    else:
        vector_search_counter.labels(
            filter_applied=str(filter_applied), k=k
        ).inc()