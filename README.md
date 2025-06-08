# Enhanced LangChain Integration for SAP HANA Cloud with NVIDIA GPU Acceleration

## About this project

This project provides an enhanced integration between [LangChain](https://github.com/langchain-ai/langchain) and [SAP HANA Cloud](https://www.sap.com/products/technology-platform/hana/cloud.html), optimized for NVIDIA GPUs. It allows you to leverage HANA's vector search, knowledge graph, and in-database capabilities as part of LLM-driven applications, with significant performance improvements through NVIDIA GPU acceleration.

> **Note**: This is an enhanced version of the [original SAP LangChain integration](https://github.com/SAP/langchain-integration-for-sap-hana-cloud) with additional features and optimizations.

## Key Features

### Original Features
- **Native Vector Store**: Utilize SAP HANA Cloud's vector database capabilities
- **VectorDB for Similarity Search**: Find relevant information efficiently
- **Embeddings Support**: Use HANA's built-in or external embedding models
- **Knowledge Graph Integration**: Leverage HANA's graph capabilities
- **Schema Configuration**: Customize database schema and table structure
- **Metadata Filtering**: Filter vector searches based on metadata

### Enhanced Extensions
- **NVIDIA GPU Acceleration**: Optimized for T4, A10, A100, and H100 GPUs
- **TensorRT Integration**: 3-10x faster inference with engine caching
- **Mobile-Responsive Frontend**: Accessibility-focused UI with dark mode
- **Interactive 3D Vector Visualization**: Explore embeddings visually
- **Flexible Deployment**: Docker, Kubernetes, Vercel, and NGC Blueprint options
- **Multi-GPU Support**: Automatic workload distribution across GPUs with dynamic load balancing
- **Context-Aware Error Handling**: Detailed error messages with remediation suggestions
- **Advanced Optimization Components**:
  - **Data Valuation (DVRL)**: Identify most valuable documents for retrieval
  - **Interpretable Embeddings (NAM)**: Understand which features contribute to search results
  - **Optimized Hyperparameters (opt_list)**: Data-driven learning rates and training schedules
  - **Model Compression (state_of_sparsity)**: Reduce memory footprint with minimal accuracy loss
- **Enterprise-Grade Additions**:
  - **Data Lineage Tracking**: Complete provenance information for embedding vectors
  - **Structured Audit Logging**: Compliance-focused logging for all operations
  - **Advanced Quantization**: INT8 and FP16 precision for faster inference
  - **Tensor Core Optimization**: Leverages NVIDIA Tensor Cores for 3-5x speedup
  - **Streaming Inference**: Support for real-time, continuous embedding generation

## Getting Started

### Prerequisites

- **Python Environment**: Python 3.9 or higher
- **SAP HANA Cloud**: Access to a running SAP HANA Cloud instance
- **NVIDIA GPU** (optional but recommended): For accelerated performance with CUDA support
- **Docker and Docker Compose** (optional): For containerized deployment

### Installation Options

#### Basic Installation (Original Functionality)

Install the core package:

```bash
pip install -U langchain-hana
```

#### Enhanced GPU-Accelerated Installation

For the full enhanced experience with GPU acceleration:

```bash
# Clone this repository
git clone https://github.com/yourusername/langchain-integration-for-sap-hana-cloud.git
cd langchain-integration-for-sap-hana-cloud

# Run the unified deployment script
./scripts/deployment/deploy-nvidia-stack.sh
```

The `deploy-nvidia-stack.sh` script orchestrates:
1. GitHub repository synchronization
2. NVIDIA T4 GPU backend deployment
3. Vercel frontend deployment with TensorRT optimization

## Project Structure

```
/
├── README.md                # Main comprehensive README
├── CONTRIBUTING.md          # Contribution guidelines
├── LICENSE                  # Original license file
├── pyproject.toml           # Python package configuration
├── requirements.txt         # Core dependencies
├── Makefile                 # Build and automation tasks
├── app.py                   # Main application entry point
├── .github/                 # GitHub-specific files
├── api/                     # Backend API code
├── frontend/                # Frontend code
├── langchain_hana/          # Core library implementation
│   ├── gpu/                 # GPU acceleration components
│   └── optimization/        # Advanced optimization components
├── docker/                  # Docker deployment files
│   ├── Dockerfile           # Standard Dockerfile
│   ├── Dockerfile.nvidia    # NVIDIA GPU-optimized Dockerfile
│   ├── docker-compose.yml   # Standard deployment
│   ├── docker-compose.nvidia.yml  # NVIDIA GPU deployment
│   ├── docker-compose.blue-green.yml  # Zero-downtime deployment
│   └── docker-compose.dev.yml  # Development environment
├── docs/                    # All documentation
│   ├── api/                 # API documentation
│   ├── deployment/          # Deployment guides
│   ├── development/         # Development guides
│   └── examples/            # Example code and usage
├── scripts/                 # All scripts
│   ├── deployment/          # Deployment scripts
│   ├── development/         # Development utilities
│   ├── testing/             # Testing scripts
│   ├── utils/               # Utility scripts
│   └── ci/                  # CI/CD scripts
├── config/                  # Configuration files
│   ├── env/                 # Environment templates
│   ├── kubernetes/          # Kubernetes configs
│   ├── vercel/              # Vercel configuration
│   ├── nvidia/              # NVIDIA-specific configs
│   └── terraform/           # Terraform configs
├── tests/                   # All tests
│   ├── unit_tests/          # Unit tests
│   ├── integration_tests/   # Integration tests
│   └── e2e_tests/           # End-to-end tests
└── examples/                # Example implementations
```

## Multi-GPU Support

The enhanced integration now includes comprehensive multi-GPU support for embedding generation, enabling significantly faster processing of large document collections. This feature distributes workloads intelligently across all available NVIDIA GPUs in your system.

### Key Multi-GPU Features

- **Dynamic Load Balancing**: Automatically distributes tasks based on GPU capabilities and current workload
- **Batch Processing Optimization**: Intelligently splits large batches across multiple GPUs
- **Priority-Based Scheduling**: Critical tasks (like query embeddings) are prioritized over background tasks
- **Real-Time Monitoring**: Tracks GPU utilization, memory usage, and task execution statistics
- **Automatic Failover**: Gracefully handles GPU failures by redistributing work
- **Cache Management**: Intelligent caching of embeddings with configurable persistence
- **TensorRT Integration**: Combines multi-GPU support with TensorRT optimization for maximum performance

### Using Multi-GPU Embeddings

```python
from langchain_hana import MultiGPUEmbeddings, HanaTensorRTMultiGPUEmbeddings
from langchain_core.embeddings import HuggingFaceEmbeddings
from langchain_hana.gpu.multi_gpu_manager import get_multi_gpu_manager

# Option 1: Wrap any embedding model for multi-GPU support
base_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
multi_gpu_embeddings = MultiGPUEmbeddings(
    base_embeddings=base_model,
    batch_size=32,
    enable_caching=True
)

# Option 2: Use the specialized TensorRT-optimized multi-GPU embeddings
trt_embeddings = HanaTensorRTMultiGPUEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    batch_size=64,
    use_fp16=True,
    enable_tensor_cores=True
)

# Generate embeddings across all available GPUs
texts = ["Text 1", "Text 2", "Text 3", ..., "Text 1000"]
embeddings = trt_embeddings.embed_documents(texts)
```

See the [examples/multi_gpu_embeddings_demo.py](examples/multi_gpu_embeddings_demo.py) for a complete benchmark and usage example.

## Testing

The project includes comprehensive testing to ensure reliability and performance:

### API Testing

To test all API endpoints with mock implementations:

```bash
# Run API endpoint tests
./run_api_tests.sh
```

This script creates a test environment with mock implementations for all external dependencies (HANA, embedding models, etc.) and validates all API endpoints.

### Unit Testing

Run unit tests with:

```bash
# Run all unit tests
pytest tests/unit_tests/

# Run specific test modules
pytest tests/unit_tests/test_gpu_embeddings.py
pytest tests/unit_tests/test_multi_gpu_manager.py
```

See [Testing Guide](docs/testing_api_endpoints.md) for more details on the testing infrastructure.

## Documentation

- [API Reference](docs/api/reference.md)
- [Deployment Guides](docs/deployment/)
  - [Docker Deployment](docs/deployment/docker.md)
  - [NVIDIA Deployment](docs/deployment/nvidia.md)
  - [NVIDIA T4 with Vercel](docs/deployment/nvidia-t4-vercel.md)
  - [NGC Blueprint](docs/deployment/ngc-blueprint.md)
- [Development Guides](docs/development/)
  - [Configuration Guide](docs/development/configuration.md)
  - [GitHub Synchronization](docs/development/github-sync.md)
- [Testing Guides](docs/)
  - [API Endpoint Testing](docs/testing_api_endpoints.md)
  - [Test Coverage Report](docs/test_coverage.md)
- [Advanced Features](docs/optimization/)
  - [Optimization Components Guide](docs/optimization/optimization_guide.md)
  - [Multi-GPU Deployment Guide](docs/optimization/multi_gpu_guide.md)

## License

This project maintains the original Apache 2.0 license. See [LICENSE](LICENSE) for details.
