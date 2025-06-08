# NVIDIA NGC Blueprint: LangChain Integration for SAP HANA Cloud

This NGC Blueprint provides an optimized integration between LangChain and SAP HANA Cloud, accelerated by NVIDIA GPUs. It allows you to leverage HANA's vector database capabilities with significantly improved performance through GPU acceleration.

## Overview

The blueprint deploys a complete stack for LLM applications with SAP HANA Cloud:

- **FastAPI Backend**: GPU-accelerated API for vector operations
- **Web Frontend**: Interactive UI for search and visualization
- **NVIDIA GPU Acceleration**: TensorRT optimization for fast embedding generation
- **SAP HANA Cloud Integration**: Connects to your SAP HANA Cloud instance for vector storage and retrieval

## Requirements

- NVIDIA GPU: T4, A10, A100, or H100 (minimum 16GB VRAM recommended)
- Docker and Docker Compose
- SAP HANA Cloud instance (or use Test Mode for development)

## Quick Start

### 1. Set Environment Variables

Set your SAP HANA Cloud connection details:

```bash
export HANA_HOST=your-hana-host.hanacloud.ondemand.com
export HANA_PORT=443
export HANA_USER=your-username
export HANA_PASSWORD=your-password
export DEFAULT_TABLE_NAME=EMBEDDINGS
```

For testing without a real HANA instance:

```bash
export TEST_MODE=true
```

### 2. Start the Services

```bash
docker-compose -f ngc-blueprint.yml up -d
```

### 3. Access the Services

- **API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Web Frontend**: http://localhost:3000

## Features

### GPU Acceleration

- **TensorRT Optimization**: 3-10x faster embedding generation
- **Multi-GPU Support**: Automatic workload distribution across all available GPUs
- **Precision Options**: FP32, FP16, and INT8 support
- **Tensor Core Utilization**: Optimized for NVIDIA Tensor Cores

### Vector Database Integration

- **Vector Similarity Search**: Fast and accurate similarity search
- **Knowledge Graph Integration**: Graph-based retrieval
- **Metadata Filtering**: Filter search results based on metadata
- **MMR Search**: Maximal Marginal Relevance for diverse results

### API Capabilities

- **Vector Operations**: Embedding generation, similarity search, MMR search
- **Context-Aware Error Handling**: Detailed error messages with remediation suggestions
- **Performance Metrics**: GPU utilization, embedding throughput, search latency

### Frontend Features

- **Interactive Search**: Search interface with metadata filtering
- **Vector Visualization**: 3D visualization of embeddings (when enabled)
- **Result Exploration**: View and explore search results with relevance scores

## Configuration Options

The blueprint supports extensive configuration through environment variables:

### GPU Acceleration

- `GPU_ENABLED`: Enable/disable GPU acceleration (default: `true`)
- `USE_TENSORRT`: Enable/disable TensorRT optimization (default: `true`)
- `TENSORRT_PRECISION`: Precision for TensorRT (fp32, fp16, int8) (default: `fp16`)
- `ENABLE_MULTI_GPU`: Enable/disable multi-GPU support (default: `true`)
- `BATCH_SIZE`: Batch size for embedding generation (default: `32`)
- `MAX_BATCH_SIZE`: Maximum batch size (default: `128`)

### API Configuration

- `PORT`: API port (default: `8000`)
- `LOG_LEVEL`: Logging level (default: `INFO`)
- `ENABLE_CORS`: Enable/disable CORS (default: `true`)
- `TEST_MODE`: Run with mock implementations (default: `false`)

### Database Configuration

- `HANA_HOST`: SAP HANA Cloud host
- `HANA_PORT`: SAP HANA Cloud port (default: `443`)
- `HANA_USER`: SAP HANA Cloud username
- `HANA_PASSWORD`: SAP HANA Cloud password
- `DEFAULT_TABLE_NAME`: Default table for vector storage (default: `EMBEDDINGS`)

## Development Mode

For development without a real SAP HANA Cloud instance:

```bash
export TEST_MODE=true
docker-compose -f ngc-blueprint.yml up -d
```

This enables mock implementations of all dependencies, allowing you to develop and test without connecting to a real database.

## Advanced Usage

### Performance Testing

The API includes endpoints for performance testing:

- `/benchmark/gpu_info`: Information about available GPUs
- `/benchmark/embedding`: Benchmark embedding generation performance
- `/benchmark/search`: Benchmark vector search performance

### Customizing the Embedding Model

Change the embedding model by setting the `DEFAULT_EMBEDDING_MODEL` environment variable:

```bash
export DEFAULT_EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
```

### Scaling with Kubernetes

For production deployment in Kubernetes, use the provided Kubernetes configurations in the `config/kubernetes` directory.

## Resources

- [Documentation](https://github.com/plturrell/langchain-integration-for-sap-hana-cloud/tree/main/docs)
- [GitHub Repository](https://github.com/plturrell/langchain-integration-for-sap-hana-cloud)
- [Issue Tracker](https://github.com/plturrell/langchain-integration-for-sap-hana-cloud/issues)

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.