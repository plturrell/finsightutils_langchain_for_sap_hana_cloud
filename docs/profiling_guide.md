# GPU Batch Size Profiling Guide

This guide explains how to use the profiling system to analyze and understand the performance characteristics of different batch sizes in GPU-accelerated embedding generation.

## Background

Our testing has revealed that in many scenarios, smaller batch sizes can outperform larger ones, which is counter-intuitive since larger batches should theoretically achieve better throughput by amortizing overhead costs. This profiling system helps identify the root causes of this anomaly.

## Potential Causes of Batch Size Performance Anomalies

1. **Memory Transfer Overhead**: Large batches require more memory transfers between host and device
2. **CUDA Synchronization Points**: Each batch requires synchronization between CPU and GPU
3. **Kernel Launch Overhead**: Multiple small kernels might be more efficient than fewer large ones
4. **Memory Padding Inefficiencies**: Variable-length inputs can cause padding waste in larger batches
5. **GPU Memory Fragmentation**: Larger allocations can lead to more fragmentation
6. **Cache Utilization**: Smaller batches might make better use of L1/L2 caches
7. **GPU Occupancy**: Different batch sizes affect how efficiently the GPU's compute units are utilized

## Profiling Tools

We've created several tools to help diagnose these issues:

### 1. GPU Profiler

The `GPUProfiler` class in `langchain_hana.monitoring.profiler` provides detailed timing and memory analysis of GPU operations.

```python
from langchain_hana.monitoring.profiler import GPUProfiler

# Create profiler
profiler = GPUProfiler(
    device_id=0,
    nvtx_ranges=True,  # Enable NVTX ranges for Nsight profiling
    memory_stats=True,  # Track memory statistics
    collect_cuda_events=True  # Use CUDA events for precise timing
)

# Profile a function
profiler.start_event("my_operation")
result = my_function()
event = profiler.end_event()

# Analyze results
print(f"Operation took {event.duration_ms} ms")
print(f"CUDA time: {event.cuda_time_ms} ms")
print(f"Memory used: {event.max_memory_allocated_mb} MB")
```

### 2. Batch Size Profiling

The `profile_embedding_model` function automates profiling across different batch sizes:

```python
from langchain_hana.monitoring.profiler import profile_embedding_model
from langchain_hana.gpu.tensorrt_embeddings import TensorRTEmbeddings

# Create embedding model
embedding_model = TensorRTEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    precision="fp16"
)

# Profile different batch sizes
results = profile_embedding_model(
    embedding_model=embedding_model,
    batch_sizes=[1, 2, 4, 8, 16, 32, 64, 128],
    text_lengths=[100],  # Can test different text lengths
    iterations=5,
    save_path="profile_results"
)

# Generate HTML report
from langchain_hana.monitoring.profiler import create_batch_size_comparison_report
create_batch_size_comparison_report(results, "batch_analysis.html")
```

### 3. Command-line Analysis Script

The `scripts/analyze_batch_performance.py` script provides a comprehensive analysis of batch size impact:

```bash
# Run full analysis
python scripts/analyze_batch_performance.py \
    --model all-MiniLM-L6-v2 \
    --batch-sizes 1 2 4 8 16 32 64 128 \
    --text-length 100 \
    --iterations 5 \
    --output-dir batch_analysis

# Run specific analyses
python scripts/analyze_batch_performance.py --memory-only
python scripts/analyze_batch_performance.py --kernel-only
python scripts/analyze_batch_performance.py --sync-only
```

## Analysis Results

The profiling tools generate several types of analysis:

1. **Throughput Analysis**: Items processed per second for each batch size
2. **Memory Analysis**: Memory usage, efficiency, and fragmentation
3. **Time Breakdown**: Where time is spent during processing
4. **Kernel Analysis**: CUDA kernel launch patterns and efficiency
5. **Bottleneck Identification**: Automatic detection of performance bottlenecks
6. **Recommendations**: Suggestions for optimizing performance

## HTML Report

The HTML report provides a visual representation of the analysis, including:

- Throughput comparison across batch sizes
- Memory usage and efficiency charts
- Time breakdown for different operations
- Identified bottlenecks and their severity
- Recommendations for optimal batch size and performance improvements

## Using NVIDIA Nsight for Deeper Analysis

For even deeper analysis, the profiling system can generate NVTX ranges that are compatible with NVIDIA Nsight Systems and Nsight Compute:

```bash
# Trace with NVTX ranges
python scripts/analyze_batch_performance.py --trace-only

# Capture with Nsight Systems
nsys profile -o batch_profile python scripts/analyze_batch_performance.py --trace-only
```

This allows for detailed kernel-level analysis and visualization of GPU execution patterns.

## Best Practices

1. **Start with a range of batch sizes**: Test batch sizes across a wide range (e.g., 1, 2, 4, 8, 16, 32, 64, 128)
2. **Run multiple iterations**: Performance can vary, so use multiple iterations for reliable results
3. **Consider variable-length inputs**: Test with both fixed and variable-length inputs
4. **Check GPU utilization**: Use `nvidia-smi` to monitor GPU utilization during profiling
5. **Test under realistic loads**: Profile under conditions similar to your production environment

## Optimizing Based on Results

After profiling, you can optimize your application by:

1. Setting the optimal batch size for your specific model and hardware
2. Using dynamic batch sizing to adapt to workload and available memory
3. Implementing custom padding strategies to reduce memory waste
4. Optimizing memory transfers and synchronization points
5. Using CUDA streams to overlap computation and data transfer

## Jupyter Notebook Example

See the `notebooks/batch_size_analysis.ipynb` notebook for an interactive example of batch size profiling and analysis.