# LangChain SAP HANA Integration Docker Configuration

This directory contains standardized Docker configuration for the LangChain Integration with SAP HANA Cloud.

## Directory Structure

```
.docker/
├── config/                  # Configuration files for services
│   └── prometheus/          # Prometheus monitoring configuration
├── compose/                 # Docker Compose files
│   ├── docker-compose.yml   # Main Docker Compose configuration
│   └── overrides/           # Environment-specific overrides
│       ├── docker-compose.dev.yml    # Development environment config
│       └── docker-compose.secure.yml # Security-enhanced config
├── scripts/                 # Utility scripts
│   ├── build.sh             # Build script for Docker images
│   └── run.sh               # Run script for Docker Compose
└── services/                # Service-specific Dockerfiles
    ├── api/                 # API service
    │   └── Dockerfile       # Dockerfile for the API service
    ├── arrow-flight/        # Arrow Flight service
    │   └── Dockerfile       # Dockerfile for the Arrow Flight service
    └── frontend/            # Frontend service
        └── Dockerfile       # Dockerfile for the frontend service
```

## Services

- **API**: Main API service for LangChain integration with SAP HANA
- **Arrow Flight**: High-performance data transfer service using Apache Arrow Flight
- **Frontend**: Web UI for interacting with the LangChain toolkit
- **Prometheus/Grafana**: Monitoring stack (optional)

## Building Docker Images

Use the provided build script to build Docker images:

```bash
# Build all images
.docker/scripts/build.sh

# Build specific services
.docker/scripts/build.sh -s api -s frontend

# Build with version tag
.docker/scripts/build.sh -v 1.0.0 

# Build and push to registry
.docker/scripts/build.sh --push

# Build secure versions of images
.docker/scripts/build.sh --secure
```

## Running Services

Use the provided run script to start services:

```bash
# Run all services
.docker/scripts/run.sh

# Run specific profile
.docker/scripts/run.sh -p api

# Run in detached mode
.docker/scripts/run.sh -d

# Run with environment file
.docker/scripts/run.sh -e .env.dev

# Run with enhanced security settings
.docker/scripts/run.sh --secure
```

## Available Profiles

- `full`: All services
- `api`: Only the API service
- `arrow-flight`: Only the Arrow Flight service
- `frontend`: Only the frontend service
- `monitoring`: Prometheus and Grafana
- `dev`: Development environment

## SAP HANA Integration

This Docker configuration is designed to work with SAP HANA Cloud. You can configure the connection using the following environment variables:

- `HANA_HOST`: SAP HANA host
- `HANA_PORT`: SAP HANA port (default: 443)
- `HANA_USER`: SAP HANA username
- `HANA_PASSWORD`: SAP HANA password
- `HANA_DATABASE`: SAP HANA database name
- `HANA_ENCRYPT`: Enable TLS encryption (default: true)
- `HANA_VALIDATE_CERT`: Validate TLS certificate (default: true)

## Arrow Flight Integration

The Arrow Flight service provides high-performance data transfer capabilities with the following features:

- Binary serialization for efficient data transfer
- Streaming capabilities for large datasets
- Low latency for real-time applications
- Efficient vector transfer for RAG applications

Environment variables:

- `FLIGHT_AUTO_START`: Auto-start Arrow Flight server (default: true)
- `FLIGHT_HOST`: Listening host (default: 0.0.0.0)
- `FLIGHT_PORT`: Listening port (default: 8815)
- `FLIGHT_TLS_ENABLED`: Enable TLS for Flight (secure mode only)

## Security Features

Enhanced security features are available by using the `--secure` flag:

- TLS encryption for all services
- JWT authentication
- Content Security Policy (CSP)
- Rate limiting
- Security headers
- Audit logging

## GitHub Actions Integration

A GitHub Actions workflow is included at `.github/workflows/docker-build.yml` that:
1. Builds Docker images for all services
2. Pushes images to GitHub Container Registry
3. Performs vulnerability scanning with Docker Scout
4. Creates appropriate tags based on Git branches/tags

## Development

For development:

```bash
# Start development environment
.docker/scripts/run.sh -p dev

# Access Jupyter notebook
open http://localhost:8888
```