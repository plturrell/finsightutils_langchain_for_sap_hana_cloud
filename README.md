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
    password="<password>"
)

# Initialize the HanaDB vector store
vectorstore = HanaDB(
    connection=connection,
    embeddings=internal_emb,  # or external_emb
    table_name="<table_name>"  # Optional: Default is "EMBEDDINGS"
)
```

## FastAPI Integration with Advanced NVIDIA GPU Acceleration

This repository includes a production-ready FastAPI application for SAP HANA Cloud vector store operations in the `api/` directory. The API provides endpoints for all vector store operations with proper error handling and logging, and includes advanced NVIDIA GPU acceleration for high-performance embedding and vector operations.

### API Features

- Secure connection to SAP HANA Cloud
- Vector store operations (add, query, delete)
- Similarity search with filtering
- Max Marginal Relevance (MMR) search
- **Advanced NVIDIA GPU Acceleration** for embeddings and vector operations
- Automatic GPU detection with CPU fallback
- Docker support for easy deployment
- Performance benchmarking tools

### Advanced GPU Acceleration

The API includes sophisticated GPU acceleration features:

- **Multi-GPU Load Balancing**: Automatically distributes workloads across all available GPUs
- **Dynamic Batch Size Adjustment**: Optimizes batch sizes based on available GPU memory
- **Memory Optimization**: Advanced techniques for handling large embedding operations
- **Performance Benchmarking**: Built-in tools to compare CPU vs GPU performance

These features ensure maximum performance by:

- Embedding generation using sentence-transformers models
- Maximal Marginal Relevance calculations with CuPy
- Optimized batch processing for GPU efficiency
- Hybrid embedding mode (GPU + HANA internal)

### Running the API

```bash
# Navigate to the API directory
cd api

# Copy and configure environment variables
cp .env.example .env

# Install dependencies
pip install -r requirements.txt

# Run the API
uvicorn app:app --reload
```

For production deployment, use the provided Docker configuration:

```bash
# Build and run with Docker Compose (CPU mode)
docker-compose up -d

# Build and run with GPU acceleration
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

For more details, see the [API README](api/README.md).

## Performance Considerations

When using GPU acceleration:

1. **Batch Size**: Adjust the `GPU_BATCH_SIZE` parameter in the API config to optimize for your GPU memory and performance needs
2. **Embedding Models**: Choose the appropriate embedding model based on your quality vs. performance tradeoff
3. **Hybrid Mode**: The API supports using both GPU acceleration and HANA's internal embedding capabilities

## Development and CI/CD

This project includes a complete CI/CD pipeline for automated testing, building, and deployment:

- GitHub Actions workflows for CI/CD pipelines
- Pre-commit hooks for code quality
- Automated testing across multiple Python versions
- Container building and publishing
- Automated deployment to cloud environments

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

For detailed information about the CI/CD setup, see our [CI/CD Guide](docs/cicd_guide.md).

## Support, Feedback, Contributing

This project is open to feature requests/suggestions, bug reports etc. via [GitHub issues](https://github.com/SAP/langchain-integration-for-sap-hana-cloud/issues). Contribution and feedback are encouraged and always welcome. For more information about how to contribute, the project structure, as well as additional contribution information, see our [Contribution Guidelines](CONTRIBUTING.md).

## Security / Disclosure
If you find any bug that may be a security problem, please follow our instructions at [in our security policy](https://github.com/SAP/langchain-integration-for-sap-hana-cloud/security/policy) on how to report it. Please do not create GitHub issues for security-related doubts or problems.

## Code of Conduct

We as members, contributors, and leaders pledge to make participation in our community a harassment-free experience for everyone. By participating in this project, you agree to abide by its [Code of Conduct](https://github.com/SAP/.github/blob/main/CODE_OF_CONDUCT.md) at all times.

## Licensing

Copyright 2025 SAP SE or an SAP affiliate company and langchain-integration-for-sap-hana-cloud contributors. Please see our [LICENSE](LICENSE) for copyright and license information. Detailed information including third-party components and their licensing/copyright information is available [via the REUSE tool](https://api.reuse.software/info/github.com/SAP/langchain-integration-for-sap-hana-cloud).