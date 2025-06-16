# SAP HANA Cloud LangChain Integration with Apache Arrow Flight

This project integrates Apache Arrow Flight with SAP HANA Cloud LangChain for high-performance vector operations.

## Features

- Zero-copy data transfer between client and server
- GPU-accelerated vector search
- Efficient batch processing of vector operations
- Multi-GPU support for distributed workloads
- React frontend for visualization and interaction

## Quick Start

### Run with Docker Compose

```bash
# Run the unified application (API + Frontend)
docker-compose -f docker-compose.unified.yml up -d

# Alternatively, run with separate containers
docker-compose -f docker-compose.complete.yml up -d
```

### Access the Application

- **Frontend UI**: http://localhost:3000
- **API**: http://localhost:8000
- **Arrow Flight**: gRPC on localhost:8815

### Docker Images

- **Unified Image**: `finsightintelligence/langchain-sap-hana:unified`
- **API Only**: `finsightintelligence/langchain-sap-hana:arrow-flight`
- **Frontend Only**: `finsightintelligence/langchain-hana:frontend`

## Build and Deploy

```bash
# Build and push the unified image
./build_and_push_unified.sh

# Or build and push separate images
./build_and_push_arrow_flight.sh
```

## Configuration

### Environment Variables

- `FLIGHT_AUTO_START`: Set to "true" to automatically start the Flight server
- `FLIGHT_HOST`: Host address for the Flight server (default: "0.0.0.0")
- `FLIGHT_PORT`: Port for the Flight server (default: 8815)
- `REACT_APP_API_URL`: URL for the API (default: "http://localhost:8000")

## NVIDIA Blueprint Integration

This project is designed to work with NVIDIA Blueprint deployments. The unified Docker container includes GPU-accelerated API with Arrow Flight support and a React-based frontend.

## Documentation

For detailed documentation, see:

- [Arrow Flight Integration](./docs/deployment/arrow_flight_integration.md)
- [API Documentation](./api/README.md)
- [Frontend Documentation](./frontend/README.md)

## License

This project is licensed under the Apache License 2.0.