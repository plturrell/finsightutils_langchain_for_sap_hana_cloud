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
- **Multi-GPU Support**: Automatic workload distribution across GPUs
- **Context-Aware Error Handling**: Detailed error messages with remediation suggestions

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
│   └── gpu/                 # GPU acceleration components
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

## License

This project maintains the original Apache 2.0 license. See [LICENSE](LICENSE) for details.
