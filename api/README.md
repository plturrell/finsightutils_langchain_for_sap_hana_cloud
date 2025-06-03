# SAP HANA Cloud Vector Store API with Advanced NVIDIA GPU Acceleration

A production-ready FastAPI application for SAP HANA Cloud vector store operations with advanced NVIDIA GPU acceleration features, featuring TensorRT optimization for maximum performance.

## Advanced GPU Acceleration Features

This API provides state-of-the-art NVIDIA GPU acceleration for SAP HANA Cloud vector store operations:

- **TensorRT Optimization**: Accelerate embedding operations with NVIDIA TensorRT for up to 3x faster inference
- **Multi-GPU Load Balancing**: Automatically distributes workloads across all available GPUs for maximum throughput
- **Dynamic Batch Size Adjustment**: Automatically optimizes batch sizes based on GPU memory to maximize performance
- **Memory Optimization**: Intelligent memory management for large embedding operations
- **Performance Benchmarking**: Built-in tools to compare CPU vs GPU performance across different configurations

## Features

- Secure connection to SAP HANA Cloud
- Vector store operations (add, query, delete)
- Similarity search with filtering
- Max Marginal Relevance (MMR) search with GPU acceleration
- **Advanced NVIDIA GPU Acceleration** for embeddings and vector operations
- Full error handling and logging
- Docker support for easy deployment

## Requirements

- Python 3.9+
- SAP HANA Cloud instance
- Docker and Docker Compose (optional)
- NVIDIA GPU with CUDA 12.x support (optional, for acceleration)

## Setup

### Environment Variables

Copy the example environment file and update with your settings:

```bash
cp .env.example .env
```

Edit the `.env` file with your SAP HANA Cloud credentials and configuration.

#### GPU Configuration

The following environment variables control GPU acceleration:

```
# GPU Configuration
GPU_ENABLED=true               # Enable/disable GPU acceleration
GPU_DEVICE=auto                # Device to use (auto, cuda, cpu)
GPU_BATCH_SIZE=32              # Batch size for GPU operations
GPU_EMBEDDING_MODEL=all-MiniLM-L6-v2  # Model for embeddings
USE_INTERNAL_EMBEDDINGS=true   # Use HANA internal embeddings by default
INTERNAL_EMBEDDING_MODEL_ID=SAP_NEB.20240715  # ID for internal embeddings

# TensorRT Configuration
USE_TENSORRT=true              # Enable TensorRT optimization
TENSORRT_PRECISION=fp16        # Precision for TensorRT (fp16, fp32)
TENSORRT_CACHE_DIR=/tmp/tensorrt_engines  # Cache directory for TensorRT engines
TENSORRT_DYNAMIC_SHAPES=true   # Use dynamic shapes for TensorRT optimization
```

### Installation

#### Local Development

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
uvicorn app:app --reload
```

#### Docker Deployment

```bash
# Build and start the container
docker-compose up -d

# View logs
docker-compose logs -f
```

For NVIDIA GPU support with Docker, ensure you have the NVIDIA Container Toolkit installed and use the following:

```bash
# Build and start with GPU support
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

#### NVIDIA NGC Container Deployment

For maximum performance with NVIDIA GPUs, we provide an optimized container in the NVIDIA NGC registry:

```bash
# Pull the container from NGC
docker pull nvcr.io/plturrell/sap-enhanced/langchain-hana-gpu:latest

# Run with GPU support and TensorRT optimization
docker run --gpus all -p 8000:8000 \
  -e HANA_HOST=your-hana-host \
  -e HANA_PORT=your-hana-port \
  -e HANA_USER=your-hana-user \
  -e HANA_PASSWORD=your-hana-password \
  -e GPU_ENABLED=true \
  -e USE_TENSORRT=true \
  nvcr.io/plturrell/sap-enhanced/langchain-hana-gpu:latest
```

#### Vercel Serverless Deployment

For lightweight, serverless deployment without GPU acceleration:

1. Fork this repository on GitHub
2. Connect your Vercel account to your GitHub account
3. Create a new project in Vercel and select your forked repository
4. Configure environment variables in Vercel:
   - HANA_HOST
   - HANA_PORT
   - HANA_USER
   - HANA_PASSWORD

The project includes a `vercel.json` configuration and a `vercel_handler.py` adapter for optimal serverless deployment.

```bash
# Or deploy directly from the CLI
vercel --prod
```

Note: Vercel serverless deployments do not support GPU acceleration. For GPU-accelerated deployments, use Docker or the NVIDIA NGC container.

## API Endpoints

### Health and Information

- `GET /health` - Check if the API and database connection are healthy, includes GPU status
- `GET /gpu/info` - Get detailed information about available GPUs

### Vector Store Operations

- `POST /texts` - Add texts to the vector store
- `POST /query` - Query the vector store by text
- `POST /query/vector` - Query the vector store by vector
- `POST /query/mmr` - Perform max marginal relevance search with GPU acceleration
- `POST /query/mmr/vector` - Perform max marginal relevance search by vector with GPU acceleration
- `POST /delete` - Delete documents from the vector store

### Benchmarking

- `POST /benchmark/embedding` - Run embedding benchmark
- `POST /benchmark/search` - Run vector search benchmark
- `POST /benchmark/tensorrt` - Run TensorRT vs PyTorch benchmark
- `GET /benchmark/status` - Get benchmark status
- `GET /benchmark/gpu_info` - Get detailed GPU information
- `POST /benchmark/compare_embeddings` - Compare different embedding approaches

## Examples

### Add Texts

```bash
curl -X POST "http://localhost:8000/texts" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["This is a sample text", "Another example document"],
    "metadatas": [{"source": "example1"}, {"source": "example2"}]
  }'
```

### Query

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "sample text",
    "k": 2,
    "filter": {"source": "example1"}
  }'
```

### Delete

```bash
curl -X POST "http://localhost:8000/delete" \
  -H "Content-Type: application/json" \
  -d '{
    "filter": {"source": "example1"}
  }'
```

### Get GPU Information

```bash
curl -X GET "http://localhost:8000/gpu/info"
```

### Run Embedding Benchmark

```bash
curl -X POST "http://localhost:8000/benchmark/embedding" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["This is a sample text for benchmarking embedding performance."],
    "count": 1000,
    "batch_size": 32
  }'
```

### Run TensorRT Benchmark

```bash
curl -X POST "http://localhost:8000/benchmark/tensorrt" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "all-MiniLM-L6-v2",
    "precision": "fp16",
    "batch_sizes": [1, 8, 32, 64, 128],
    "input_length": 128,
    "iterations": 100
  }'
```

## Advanced Features

### TensorRT Optimization

The API leverages NVIDIA TensorRT to accelerate embedding operations:
- Optimized model graphs with layer fusion and kernel autotuning
- Mixed precision support (FP16/FP32) for maximum performance
- Dynamic shape support for flexible batch sizes
- Cached optimization for fast startup
- Up to 3x faster inference compared to standard PyTorch

### Multi-GPU Load Balancing

The API automatically detects all available NVIDIA GPUs and distributes workloads across them using a sophisticated load balancing algorithm. This provides near-linear scaling with the number of GPUs for embedding operations.

### Dynamic Batch Size Adjustment

The system automatically determines the optimal batch size based on:
- Available GPU memory
- Workload characteristics
- Model size and complexity

This ensures maximum throughput while preventing out-of-memory errors.

### Memory Optimization

For large embedding operations, the API employs advanced memory optimization techniques:
- Tensor pooling for reusing allocated memory
- Progressive loading for handling datasets larger than GPU memory
- Automatic garbage collection and cache management
- Memory-aware processing to prevent memory fragmentation

### Performance Benchmarking

The built-in benchmarking tools allow you to:
- Compare TensorRT vs PyTorch performance
- Compare CPU vs GPU performance for your specific workload
- Identify optimal batch sizes for maximum throughput
- Measure performance across different configurations
- Track performance improvements over time

## Performance Considerations

When using GPU acceleration:

1. **TensorRT Optimization**: Enable TensorRT for maximum performance with `USE_TENSORRT=true`. For most modern NVIDIA GPUs, FP16 precision provides the best performance-accuracy tradeoff.
2. **Batch Size**: The default batch size is automatically optimized, but you can override it with the `GPU_BATCH_SIZE` parameter. With TensorRT, larger batch sizes often provide better throughput.
3. **Embedding Models**: Larger models provide better results but require more GPU memory. The default `all-MiniLM-L6-v2` provides a good balance.
4. **Multi-GPU Setup**: For multi-GPU environments, ensure all GPUs have similar specifications for optimal load balancing.
5. **Memory Management**: For very large datasets, consider using the progressive loading feature by submitting data in manageable chunks.
6. **NVIDIA NGC Container**: For production deployments, use the pre-optimized NGC container which includes TensorRT and all dependencies pre-configured for maximum performance.

## Error Handling

The API includes comprehensive error handling for:

- Database connection failures
- GPU-related issues with fallback to CPU
- Memory management issues with automatic recovery
- Invalid requests
- Internal server errors

All errors are properly logged and returned with appropriate HTTP status codes.

## Security Considerations

- Use environment variables for sensitive information
- Implement proper authentication in production
- Consider using HTTPS for production deployments

## Troubleshooting

### GPU Issues

If you experience issues with GPU acceleration:

1. Check GPU availability with the `/benchmark/gpu_info` endpoint
2. Ensure CUDA dependencies are properly installed
3. Set `GPU_ENABLED=false` to force CPU mode if needed
4. Check logs for any GPU-related errors
5. Run benchmarks to identify performance bottlenecks

### TensorRT Issues

If you experience issues with TensorRT optimization:

1. Check if TensorRT is properly installed with the `/benchmark/tensorrt` endpoint
2. Set `USE_TENSORRT=false` to disable TensorRT optimization
3. Try different precision settings with `TENSORRT_PRECISION=fp32` if you encounter numerical issues
4. Ensure the TensorRT cache directory is writable
5. Use the pre-optimized NGC container to avoid compatibility issues
6. For very large models, increase available GPU memory or reduce batch size

### Memory Management

If you encounter out-of-memory errors:

1. Reduce batch size with the `GPU_BATCH_SIZE` parameter
2. Use the progressive loading feature for large datasets
3. Enable dynamic batch sizing with `DYNAMIC_BATCHING=true`
4. Consider using a GPU with more memory or multiple GPUs
5. Disable TensorRT optimization for extremely large models