# GPU Acceleration for SAP HANA Cloud Vector Search

This document provides a comprehensive guide to using GPU acceleration with the SAP HANA Cloud LangChain integration, focusing on optimizations for NVIDIA T4 GPUs and efficient vector operations.

## Overview

The GPU acceleration modules provide significant performance improvements for embedding generation and vector operations when working with SAP HANA Cloud's vector capabilities. These optimizations are especially valuable for:

- Large document collections (10,000+ documents)
- High-throughput search scenarios
- Real-time embedding generation
- Resource-constrained environments

Key features include:

1. **TensorRT-accelerated embedding generation**
2. **Tensor Core optimizations for NVIDIA T4 GPUs**
3. **Multi-GPU support with intelligent load balancing**
4. **Mixed-precision operations (FP32, FP16, INT8)**
5. **Memory-optimized vector serialization**
6. **Seamless integration with SAP HANA Cloud vectorstore**
7. **Comprehensive performance monitoring**

## Architecture

The GPU acceleration architecture consists of several integrated components:

```
┌─────────────────────────────────┐
│  HanaTensorRTVectorStore        │
│  (GPU-accelerated VectorStore)  │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐     ┌────────────────────────┐
│  HanaTensorRTEmbeddings         │     │ Vector Serialization   │
│  (GPU-accelerated Embeddings)   │────▶│ (Memory-optimized)     │
└───────────────┬─────────────────┘     └────────────────────────┘
                │
                ▼
┌─────────────────────────────────┐     ┌────────────────────────┐
│  TensorRTEmbeddings             │     │ Multi-GPU Manager      │
│  (Base TensorRT implementation) │────▶│ (Distributed workload) │
└───────────────┬─────────────────┘     └────────────────────────┘
                │
                ▼
┌─────────────────────────────────┐     ┌────────────────────────┐
│  Batch Processor                │     │ Tensor Core Optimizer  │
│  (Dynamic batch sizing)         │────▶│ (T4 GPU optimizations) │
└─────────────────────────────────┘     └────────────────────────┘
```

## Performance Characteristics

Performance improvements vary based on hardware, dataset size, and configuration, but typical gains include:

| Feature | Improvement | Notes |
|---------|-------------|-------|
| Embedding generation speed | 2-6x | Compared to CPU-based embedding |
| Memory efficiency | 2-4x | Using mixed precision (FP16/INT8) |
| Vector serialization | Up to 75% reduction | Using optimized serialization |
| Multi-GPU scaling | Near-linear | With proper batch configuration |
| Search throughput | 2-3x | For embedding-intensive workloads |

### T4 GPU-Specific Optimizations

The implementation includes specific optimizations for NVIDIA T4 GPUs:

- **Tensor Core utilization**: Aligned memory layouts for maximum Tensor Core efficiency
- **INT8 quantization**: Domain-specific calibration for minimal accuracy loss
- **Dynamic batch sizing**: Automatic adjustment based on available GPU memory
- **Mixed-precision inference**: FP16 and INT8 support for maximum throughput

## Installation Requirements

To use GPU acceleration, you need:

1. NVIDIA GPU with CUDA support (T4 recommended for best performance)
2. CUDA Toolkit 11.8+
3. PyTorch 2.0+ with CUDA support
4. TensorRT 8.0+
5. SAP HANA Cloud account

Install the required dependencies with:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tensorrt pycuda
```

## Basic Usage

### Creating GPU-Accelerated Embeddings

```python
from langchain_hana.gpu.hana_tensorrt_embeddings import HanaTensorRTEmbeddings

# Initialize GPU-accelerated embeddings
embeddings = HanaTensorRTEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",  # Embedding model
    precision="fp16",                                     # Precision mode (fp32, fp16, int8)
    multi_gpu=True,                                       # Use multiple GPUs if available
    max_batch_size=64                                     # Maximum batch size
)
```

### Creating GPU-Accelerated Vectorstore

```python
from langchain_hana.gpu.hana_tensorrt_vectorstore import HanaTensorRTVectorStore

# Create connection to SAP HANA Cloud
from hdbcli import dbapi
conn = dbapi.connect(address='your-hana-host', port=443, user='username', password='password')

# Initialize GPU-accelerated vectorstore
vectorstore = HanaTensorRTVectorStore(
    connection=conn,
    embedding=embeddings,
    table_name="GPU_ACCELERATED_VECTORS",
    batch_size=64,
    enable_performance_monitoring=True,
    # Optional: use HALF_VECTOR for reduced storage with FP16
    vector_column_type="HALF_VECTOR" if embeddings.precision == "fp16" else "REAL_VECTOR"
)
```

### Adding Documents

```python
# Add documents with GPU acceleration
documents = ["Document 1", "Document 2", "Document 3", ...]
metadatas = [{"category": "finance"}, {"category": "technology"}, {"category": "business"}, ...]

vectorstore.add_texts(documents, metadatas)
```

### Searching Documents

```python
# Search for similar documents (uses GPU for query embedding)
results = vectorstore.similarity_search(
    query="What is SAP HANA Cloud?",
    k=5,
    filter={"category": "technology"}
)

# Process results
for doc in results:
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
    print()
```

### Performance Monitoring

```python
# Get performance metrics
metrics = vectorstore.get_performance_metrics()
print(json.dumps(metrics, indent=2))

# Clear metrics if needed
vectorstore.clear_performance_metrics()
```

## Advanced Configuration

### Multi-GPU Setup

To enable multi-GPU processing:

```python
import os

# Enable multi-GPU support via environment variables (alternative to multi_gpu=True)
os.environ["MULTI_GPU_ENABLED"] = "true"
os.environ["MULTI_GPU_STRATEGY"] = "auto"  # Options: auto, round_robin, memory, utilization

# Create embeddings with multi-GPU support
embeddings = HanaTensorRTEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    multi_gpu=True
)

# Get multi-GPU status
from langchain_hana.gpu.multi_gpu_manager import get_multi_gpu_manager
manager = get_multi_gpu_manager()
print(manager.get_status())
```

### Precision Modes

Different precision modes offer varying tradeoffs between speed and accuracy:

| Precision | Speed | Accuracy | Memory Usage | Best For |
|-----------|-------|----------|--------------|----------|
| FP32      | Baseline | Highest | Highest | High-precision requirements |
| FP16      | 2-4x faster | Minimal loss | 50% of FP32 | General use, T4 GPUs |
| INT8      | 3-6x faster | ~0.5-2% loss | 25% of FP32 | Maximum throughput |

To select a precision mode:

```python
# FP16 (half-precision) - good balance of speed and accuracy for T4 GPUs
embeddings = HanaTensorRTEmbeddings(precision="fp16")

# INT8 (quantized) - maximum throughput, slight accuracy loss
embeddings = HanaTensorRTEmbeddings(precision="int8")

# Compare performance across precision modes
benchmark_results = embeddings.benchmark_precision_comparison()
```

### Tensor Core Optimization

For NVIDIA T4 GPUs, Tensor Core optimizations provide significant performance gains:

```python
# Create embeddings with Tensor Core optimizations
embeddings = HanaTensorRTEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    precision="fp16",
    enable_tensor_cores=True
)
```

### Custom Calibration for INT8 Quantization

When using INT8 precision, domain-specific calibration improves accuracy:

```python
from langchain_hana.gpu.calibration_datasets import create_enhanced_calibration_dataset

# Create domain-specific calibration dataset
calibration_data = create_enhanced_calibration_dataset(
    domains=["financial", "sap", "technical"],
    count=100
)

# Use custom calibration data
embeddings = HanaTensorRTEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    precision="int8",
    calibration_data=calibration_data
)
```

## Memory-Optimized Serialization

The implementation includes optimized vector serialization for efficient data transfer:

```python
from langchain_hana.gpu.vector_serialization import get_vector_memory_usage

# Calculate memory usage for different precision modes
memory_stats = get_vector_memory_usage(
    vector_dimension=384,  # Model embedding dimension
    num_vectors=100000,    # Number of vectors
    precision="fp16"       # Current precision
)

print(f"Memory usage with FP32: {memory_stats['memory_float32_mb']:.2f} MB")
print(f"Memory usage with FP16: {memory_stats['memory_float16_mb']:.2f} MB")
print(f"Memory usage with INT8: {memory_stats['memory_int8_mb']:.2f} MB")
```

## Dynamic Batch Processing

The dynamic batch processor automatically determines and adjusts batch sizes during processing based on:

1. Available GPU memory at runtime
2. Model size and memory requirements
3. Current processing performance
4. OOM (out-of-memory) recovery and adaptation

### Key Features

- **Runtime GPU Memory Detection**: Automatically detects available GPU memory
- **Model-Aware Batch Sizing**: Calculates optimal batch size based on model characteristics
- **Dynamic Adjustment**: Adapts batch size during processing for maximum throughput
- **Safety Margins**: Includes configurable safety margins to prevent OOM errors
- **Batch Splitting**: Automatically handles large requests by splitting into optimal batches
- **OOM Recovery**: Recovers gracefully from OOM errors by reducing batch size
- **Performance Monitoring**: Tracks processing statistics for optimization

### Custom Batch Processing

For advanced use cases, you can also use the batch processor directly:

```python
from langchain_hana.gpu.batch_processor import EmbeddingBatchProcessor, ModelMemoryProfile

# Define your embedding function
def embed_batch(batch_texts):
    # Your embedding generation code
    return embeddings

# Create model memory profile
profile = ModelMemoryProfile(
    model_name="your-model-name",
    embedding_dim=768,
    dtype="float16"
)

# Create batch processor
processor = EmbeddingBatchProcessor(
    embedding_fn=embed_batch,
    model_name="your-model-name",
    embedding_dim=768,
    device_id=0,
    initial_batch_size=32,
    min_batch_size=1,
    max_batch_size=128,
    safety_factor=0.8,
    oom_recovery_factor=0.5,
    dtype="float16",
    enable_caching=True
)

# Process documents with dynamic batching
embeddings, stats = processor.embed_documents(texts)

# Print statistics
print(f"Total time: {stats.total_time:.2f}s")
print(f"Items per second: {stats.items_per_second:.2f}")
print(f"Batch size adjustment: {stats.initial_batch_size} → {stats.final_batch_size}")
```

## Performance Optimization Tips

For maximum performance:

1. **Choose the right precision mode**:
   - For T4 GPUs, FP16 provides the best balance of speed and accuracy
   - Use INT8 when maximum throughput is required and slight accuracy loss is acceptable

2. **Optimize batch size**:
   - For T4 GPUs, optimal batch sizes are multiples of 8 (for FP16) or 16 (for INT8)
   - Larger batch sizes (64-128) provide better throughput for bulk operations
   - Smaller batch sizes (1-16) provide lower latency for interactive use

3. **Use multi-GPU when available**:
   - Enable `multi_gpu=True` when multiple GPUs are available
   - Processing is automatically distributed across available GPUs

4. **Match vector column type to precision mode**:
   - Use "HALF_VECTOR" with FP16 precision for reduced storage requirements
   - Use "REAL_VECTOR" with FP32 precision for maximum accuracy

5. **Create HNSW indexes for faster search**:
   - The vectorstore automatically creates an optimized index
   - Adjust index parameters for specific use cases:
     ```python
     vectorstore.create_hnsw_index(
         m=64,                 # Higher values improve recall but increase memory usage
         ef_construction=400,  # Higher values improve index quality but increase build time
         ef_search=100         # Higher values improve search quality but reduce speed
     )
     ```

## Troubleshooting

### CUDA Out of Memory Errors

If you encounter "CUDA out of memory" errors:

1. Reduce batch size (e.g., `batch_size=32` or lower)
2. Use a more memory-efficient precision mode (FP16 or INT8)
3. Ensure no other processes are using GPU memory
4. Consider using dynamic batch sizing:
   ```python
   os.environ["DYNAMIC_BATCH_SIZE"] = "true"
   ```

### Performance Issues

If performance is below expectations:

1. Check GPU utilization with `nvidia-smi`
2. Ensure TensorRT engine is properly built and cached
3. Verify that Tensor Core operations are enabled
4. Run benchmarks to identify bottlenecks:
   ```python
   benchmark_results = embeddings.benchmark()
   ```

### Missing GPU Features

If GPU features are not working:

1. Check CUDA availability: `torch.cuda.is_available()`
2. Verify TensorRT installation: `import tensorrt`
3. Check compute capability: `torch.cuda.get_device_capability(0)`
4. Ensure the correct CUDA version is installed

## Implementation Details

The GPU acceleration is implemented across several modules:

1. **hana_tensorrt_embeddings.py**: GPU-accelerated embedding generation
2. **hana_tensorrt_vectorstore.py**: Integration with SAP HANA vectorstore
3. **tensor_core_optimizer.py**: Tensor Core optimizations for T4 GPUs
4. **vector_serialization.py**: Memory-efficient vector serialization
5. **multi_gpu_manager.py**: Multi-GPU work distribution
6. **batch_processor.py**: Dynamic batch sizing for optimal performance
7. **calibration_datasets.py**: Domain-specific calibration for INT8 quantization

## Complete Example

Check the full example in [examples/hana_gpu_acceleration.py](../examples/hana_gpu_acceleration.py) that demonstrates:

1. Creating GPU-accelerated embeddings with TensorRT
2. Benchmarking different precision modes
3. Setting up a GPU-optimized HANA vectorstore
4. Adding documents with batched GPU processing
5. Running similarity searches with GPU acceleration
6. Monitoring performance metrics

## Integration with LangChain Components

The GPU-accelerated HANA vectorstore can be seamlessly integrated with LangChain's higher-level components like RetrievalQA, Agents, and Chains.

### RetrievalQA Example

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

# Create GPU-accelerated vectorstore (as shown in earlier examples)
vectorstore = HanaTensorRTVectorStore(
    connection=conn,
    embedding=embeddings,
    table_name="GPU_ACCELERATED_VECTORS",
    batch_size=64,
)

# Create a retriever from the vectorstore
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Create a language model
llm = ChatOpenAI(model="gpt-4")

# Create a RetrievalQA chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | llm
    | StrOutputParser()
)

# Use the chain
response = rag_chain.invoke("What is the role of vector databases in enterprise AI?")
print(response)
```

### Integration with Agents

```python
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain_openai import ChatOpenAI

# Create the vectorstore and retriever
embeddings = HanaTensorRTEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    precision="fp16"
)

vectorstore = HanaTensorRTVectorStore(
    connection=conn,
    embedding=embeddings,
    table_name="GPU_ACCELERATED_VECTORS"
)

# Create a search tool
tools = [
    Tool(
        name="VectorSearch",
        func=lambda q: vectorstore.similarity_search(q, k=5),
        description="Search for information in the vector database. Use for specific questions about products, services, or company information."
    )
]

# Create an agent
llm = ChatOpenAI(temperature=0, model="gpt-4")
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)

# Run the agent
agent.run("What are the key benefits of using SAP HANA Cloud for vector search?")
```

### Streaming Results with GPU Acceleration

You can also use streaming capabilities with GPU-accelerated processing:

```python
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Create the vectorstore (as shown above)

# Create a streaming LLM
llm = ChatOpenAI(
    model="gpt-4",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

# Create a retrieval chain with sources
chain = RetrievalQAWithSourcesChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever()
)

# Stream the response
chain({"question": "How can I implement semantic search with SAP HANA Cloud?"})
```

## Production Deployment Best Practices

When deploying GPU-accelerated SAP HANA vectorstore in production environments, consider these best practices:

### Resource Allocation

1. **GPU Selection**:
   - For high-throughput production: T4 or newer with at least 16GB VRAM
   - For cost-effective deployment: T4 with 8-16GB VRAM
   - For maximum performance: A100 or newer

2. **CPU Requirements**:
   - Minimum: 4 vCPUs
   - Recommended: 8+ vCPUs for preprocessing and I/O operations

3. **Memory Requirements**:
   - Minimum: 16GB RAM
   - Recommended: 32GB+ RAM for large datasets

### Containerization

When deploying with Docker:

```dockerfile
FROM nvcr.io/nvidia/pytorch:23.03-py3

# Install HANA client and SAP dependencies
RUN pip install hdbcli 'hana-ml>=2.17.24'

# Install TensorRT and our library
RUN pip install tensorrt pycuda
COPY . /app
WORKDIR /app
RUN pip install -e .

# Set environment variables
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
ENV DYNAMIC_BATCH_SIZE=true

# Entry point
CMD ["python", "app.py"]
```

Run with:
```bash
docker run --gpus all -p 8080:8080 -e HANA_HOST=your-host -e HANA_PORT=443 -e HANA_USER=user -e HANA_PASSWORD=password your-image
```

### Performance Monitoring

To monitor GPU usage in production:

1. **NVML Integration**:
   ```python
   from langchain_hana.gpu.performance_monitor import GPUPerformanceMonitor
   
   # Initialize monitor
   monitor = GPUPerformanceMonitor()
   
   # Start monitoring
   monitor.start()
   
   # Get current stats
   stats = monitor.get_current_stats()
   ```

2. **Prometheus Integration**:
   - The GPU performance metrics are exposed via Prometheus-compatible endpoints
   - Configure your Prometheus server to scrape these metrics
   - Create Grafana dashboards for visualization

### High Availability

For production deployments requiring high availability:

1. **Multi-Instance Deployment**:
   - Deploy multiple instances behind a load balancer
   - Use stateless design for easy scaling
   - Configure connection pooling for SAP HANA connections

2. **Graceful Degradation**:
   - Enable CPU fallback for GPU errors:
   ```python
   embeddings = HanaTensorRTEmbeddings(
       model_name="...",
       enable_cpu_fallback=True
   )
   ```

3. **Health Checks**:
   - Implement regular GPU health checks
   - Monitor CUDA errors and recover when possible
   - Implement circuit breakers for embedding services

### Security Considerations

1. **Model and Data Protection**:
   - TensorRT engines contain model weights; secure them appropriately
   - Use encrypted connections to SAP HANA Cloud
   - Implement proper authentication for API endpoints

2. **Resource Isolation**:
   - Use container resource limits to prevent resource exhaustion
   - Configure CUDA visible devices to control GPU access
   - Implement rate limiting for API endpoints

## Future Enhancements

Planned future enhancements include:

1. Real-time performance monitoring and auto-tuning
2. Advanced vector compression techniques
3. Integration with other embedding models (BERT, ADA, etc.)
4. Support for newer GPU architectures (A100, H100)
5. Enhanced multi-modal support (text, images, etc.)
6. Distributed processing across multiple nodes