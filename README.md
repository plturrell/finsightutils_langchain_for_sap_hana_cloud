[![REUSE status](https://api.reuse.software/badge/github.com/SAP/langchain-integration-for-sap-hana-cloud)](https://api.reuse.software/info/github.com/SAP/langchain-integration-for-sap-hana-cloud)

# LangChain integration for SAP HANA Cloud

## About this project

Integrates LangChain with SAP HANA Cloud to make use of vector search, knowledge graph, and further in-database capabilities as part of LLM-driven applications.

## Requirements and Setup

### Prerequisites

- **Python Environment**: Ensure you have Python 3.9 or higher installed.
- **SAP HANA Cloud**: Access to a running SAP HANA Cloud instance.


### Installation

Install the LangChain SAP HANA Cloud integration package using `pip`:

```bash
pip install -U langchain-hana
```

### Setting Up Vectorstore

The `HanaDB` class is used to connect to SAP HANA Cloud Vector Engine.

> **Important**:  You can use any embedding class that inherits from `langchain_core.embeddings.Embeddings`—**including** `HanaInternalEmbeddings`, which runs SAP HANA's `VECTOR_EMBEDDING()` function internally. See [SAP Help](https://help.sap.com/docs/hana-cloud-database/sap-hana-cloud-sap-hana-database-vector-engine-guide/vector-embedding-function-vector?locale=en-US) for more details.

Here's how to set up the connection and initialize the vector store:

```python
from langchain_hana import HanaDB, HanaInternalEmbeddings
from langchain_openai import OpenAIEmbeddings
from hdbcli import dbapi

# 1) HANA-internal embedding
internal_emb = HanaInternalEmbeddings(internal_embedding_model_id="SAP_NEB.20240715")
# 2) External embedding
external_emb = OpenAIEmbeddings()

# Establish the SAP HANA Cloud connection
connection = dbapi.connect(
    address="<hostname>",
    port=3<NN>MM,
    user="<username>",
    password="<password>",
    encrypt=True,  # Recommended for production
    sslValidateCertificate=True  # Recommended for production
)

# Initialize the HanaDB vector store
vectorstore = HanaDB(
    connection=connection,
    embedding=internal_emb,  # or external_emb
    table_name="<table_name>"  # Optional: Default is "EMBEDDINGS"
)
```

For detailed configuration options and recommended settings, see our [Configuration Guide](docs/configuration_guide.md).

## Advanced Features

This integration provides powerful advanced capabilities:

- **HNSW Vector Indexing**: Accelerate similarity searches with configurable indexing
- **Maximal Marginal Relevance**: Balance relevance and diversity in search results
- **Complex Metadata Filtering**: Filter results using rich query operators
- **Knowledge Graph Integration**: Query knowledge graphs with natural language
- **Asynchronous Operations**: High-throughput with async methods
- **Internal Embeddings**: Leverage SAP HANA's built-in embedding functions
- **Accurate Similarity Scoring**: Get precise vector similarity measurements for better ranking
- **Context-Aware Error Handling**: Detailed error messages with suggested fixes for common issues
- **GPU Acceleration**: High-performance embedding generation with NVIDIA GPU support
- **Dynamic Batch Processing**: Memory-aware batch sizing for optimal GPU throughput
- **TensorRT Optimization**: Accelerated inference with TensorRT engine compilation
- **Mixed Precision Support**: FP32, FP16, and INT8 precision options for optimal performance

For details on these and other advanced features, see our [Advanced Features Guide](docs/advanced_features.md) and [GPU Acceleration Guide](docs/gpu_acceleration.md).

### Intelligent Error Handling

The integration includes a sophisticated error handling system that provides context-aware error messages with suggested actions when problems occur:

```python
try:
    # This will use context-aware error handling internally
    results = vectorstore.similarity_search("my query", k=5)
except Exception as e:
    # Error message will include specific suggestions based on the operation
    # and error type, such as connection issues, permissions, or data format problems
    print(f"Error: {e}")
```

Error messages include:
- Detailed description of what went wrong
- Specific suggestions for fixing the issue
- Operation context (what you were trying to do)
- Possible underlying causes

This makes troubleshooting much easier, especially for database-specific issues or vector operation problems.

## Split Architecture with Frontend and Backend Components

This repository uses a split architecture with separate frontend and backend components for more flexible deployment options:

1. **Backend API**: A FastAPI application with GPU acceleration for vector operations
2. **Frontend**: A responsive web application with interactive visualizations

This architecture allows for various deployment scenarios:
- Frontend on Vercel with backend on Docker/Kubernetes
- Both components on Docker
- Frontend as static files with backend on a server
- Complete stack on NVIDIA NGC Blueprint platform

## NVIDIA NGC Blueprint Compatibility

This project is fully optimized for NVIDIA GPUs and complies with NVIDIA NGC Blueprint standards:

- **CUDA-Accelerated**: Leverages CUDA for high-performance embedding generation
- **TensorRT Optimized**: Uses TensorRT for maximum inference throughput
- **NGC Container Based**: Built on official NVIDIA NGC PyTorch containers
- **Multi-GPU Support**: Automatically scales across all available GPUs
- **NGC Blueprint Ready**: Includes complete NGC Blueprint configuration
- **Performance Optimized**: Tuned for T4, A10, A100, and H100 GPUs

The repository includes automated build scripts for NGC deployment:
```bash
# Build and push to NGC
./build_launchable.sh
```

### Backend API with Advanced NVIDIA GPU Acceleration

The backend API in the `api/` directory provides:

- Secure connection to SAP HANA Cloud
- Vector store operations (add, query, delete)
- Similarity search with filtering and MMR search
- **Advanced NVIDIA GPU Acceleration** for embeddings
- Automatic GPU detection with CPU fallback
- JWT authentication and error handling
- Comprehensive telemetry and metrics

#### Advanced GPU Acceleration

The API includes sophisticated GPU acceleration features:

- **Multi-GPU Load Balancing**: Automatically distributes workloads across GPUs
- **Dynamic Batch Size Adjustment**: Optimizes batch sizes based on GPU memory
- **TensorRT Optimization**: High-performance inference with TensorRT
- **Mixed Precision Support**: FP32, FP16, and INT8 precision options

### Responsive Frontend with Interactive Visualizations

The frontend provides:

- Mobile-first responsive design
- Accessibility features (dark mode, high contrast, screen reader support)
- Interactive 3D vector visualizations
- Advanced search interface
- Comprehensive error handling with suggestions
- Authentication and user management

### Deployment Options

#### Docker Deployment

Deploy using Docker Compose with separate files for backend and frontend:

```bash
# Deploy backend API only
docker-compose -f docker-compose.api.yml up -d

# Deploy frontend only
docker-compose -f docker-compose.frontend.yml up -d

# Deploy both with GPU acceleration
docker-compose -f docker-compose.api.yml -f docker-compose.gpu.yml -f docker-compose.frontend.yml up -d
```

#### Vercel Deployment

The frontend is optimized for deployment on Vercel:

1. Push code to GitHub
2. Create a new project on Vercel
3. Configure build settings and environment variables
4. Deploy

For complete deployment instructions, see the [Deployment Guide](docs/deployment_guide.md).

## Performance Considerations

When using GPU acceleration:

1. **Batch Size**: Adjust the `BATCH_SIZE` parameter to optimize for your GPU memory and performance needs
2. **TensorRT Precision**: Select the appropriate precision (FP16, FP32, INT8) based on your accuracy vs. performance tradeoff
3. **GPU Model Selection**: Performance varies significantly across different NVIDIA GPU models:
   - NVIDIA T4: Good for cost-effective inference (4-12x CPU performance)
   - NVIDIA A10: Excellent balanced performance (6-21x CPU performance)
   - NVIDIA A100: Highest throughput for large workloads (9-37x CPU performance)
4. **Hybrid Mode**: Supports both GPU acceleration and HANA's internal embedding capabilities

For detailed performance benchmarks and optimization guides, see our [NVIDIA Blueprint Compliance](docs/nvidia_blueprint_compliance.md) documentation.

## Development, CI/CD, and Infrastructure as Code

This project includes a complete CI/CD pipeline for automated testing, building, and deployment, plus Infrastructure as Code for reliable deployment to Kubernetes:

### CI/CD Pipeline

- GitHub Actions workflows for CI/CD pipelines
- Pre-commit hooks for code quality
- Automated testing across multiple Python versions
- Container building and publishing
- Automated deployment to cloud environments

### Infrastructure as Code with Terraform

The project uses Terraform to manage all infrastructure components:

- **Kubernetes Infrastructure**: Namespaces, deployments, services, secrets, HPA
- **NVIDIA GPU Support**: GPU-optimized deployments with resource management
- **Monitoring Stack**: Prometheus for metrics and Grafana for visualization
- **Environment Configurations**: Separate staging and production setups
- **Automated Management**: Infrastructure changes managed through CI/CD pipeline

### Local Development Setup

For developers, we recommend setting up the local development environment:

```bash
# Set up local development environment with pre-commit hooks
./scripts/setup_local_dev.sh
```

This setup script will:
1. Install pre-commit hooks and development dependencies
2. Configure git remotes for both the main SAP repository and @plturrell's SAP OpenSource Enhanced repository
3. Set up automatic synchronization between both repositories
4. Install the tag-and-release script for easy version management

### Infrastructure Management

For infrastructure management:

```bash
# Plan infrastructure changes for staging environment
./scripts/terraform_apply.sh

# Apply infrastructure changes to production
./scripts/terraform_apply.sh -e production -a apply
```

For detailed information about the setup, see our:
- [CI/CD Guide](docs/cicd_guide.md)
- [Infrastructure as Code Guide](docs/infrastructure_as_code.md)

## Support, Feedback, Contributing

This project is open to feature requests/suggestions, bug reports etc. via [GitHub issues](https://github.com/SAP/langchain-integration-for-sap-hana-cloud/issues). Contribution and feedback are encouraged and always welcome. For more information about how to contribute, the project structure, as well as additional contribution information, see our [Contribution Guidelines](CONTRIBUTING.md).

## Security

### Production Security Guide
For comprehensive security guidance when deploying in production environments, please review our [Security Guide](docs/security_guide.md) which covers:
- Secure connection configuration
- Credential management best practices
- Principle of least privilege implementation
- Network security recommendations
- Connection pooling for production
- Data security considerations
- Monitoring and alerting

### Security Disclosure
If you find any bug that may be a security problem, please follow our instructions at [in our security policy](https://github.com/SAP/langchain-integration-for-sap-hana-cloud/security/policy) on how to report it. Please do not create GitHub issues for security-related doubts or problems.

## Code of Conduct

We as members, contributors, and leaders pledge to make participation in our community a harassment-free experience for everyone. By participating in this project, you agree to abide by its [Code of Conduct](https://github.com/SAP/.github/blob/main/CODE_OF_CONDUCT.md) at all times.

## Licensing

Copyright 2025 SAP SE or an SAP affiliate company and langchain-integration-for-sap-hana-cloud contributors. Please see our [LICENSE](LICENSE) for copyright and license information. Detailed information including third-party components and their licensing/copyright information is available [via the REUSE tool](https://api.reuse.software/info/github.com/SAP/langchain-integration-for-sap-hana-cloud).