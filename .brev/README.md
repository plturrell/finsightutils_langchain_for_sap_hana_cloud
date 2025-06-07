# SAP HANA Cloud LangChain Integration with NVIDIA GPU Acceleration

This Brev LaunchPad configuration deploys a GPU-accelerated vector search service for SAP HANA Cloud integration.

## Features

- TensorRT-optimized embedding generation
- Multi-GPU support with dynamic load balancing
- NVIDIA Triton Server for scalable inference
- DCGM monitoring for GPU performance tracking
- Comprehensive security measures
- Prometheus-based metrics and monitoring

## Getting Started

1. Deploy the blueprint with your SAP HANA Cloud credentials
2. Access the API at the public endpoint: `http://<deployment-url>:8000`
3. Access the frontend at the public endpoint: `http://<deployment-url>:3000`
4. Monitor GPU performance with Prometheus: `http://<deployment-url>:9090`

## Security Considerations

- JWT authentication is required for API access
- Container security scanning is performed during deployment
- Services run as non-root users when possible
- CORS is restricted to specified origins

## Performance Optimization

- TensorRT engines are calibrated and cached for optimal performance
- NVIDIA DALI accelerates data loading operations
- Dynamic batch sizing based on GPU memory availability
- Multi-GPU support distributes workload across available GPUs

## API Documentation

The API documentation is available at `/docs` on the API endpoint.

## Monitoring

- GPU metrics: `http://<deployment-url>:9400/metrics`
- Prometheus: `http://<deployment-url>:9090`
- API health: `http://<deployment-url>:8000/health/ping`
- Triton metrics: `http://<deployment-url>:8002/metrics`

## Environment Variables

| Variable | Description | Default | Required |
| --- | --- | --- | --- |
| HANA_HOST | SAP HANA Cloud host | - | Yes |
| HANA_PORT | SAP HANA Cloud port | 443 | No |
| HANA_USER | SAP HANA Cloud username | - | Yes |
| HANA_PASSWORD | SAP HANA Cloud password | - | Yes |
| JWT_SECRET | JWT authentication secret key | - | Yes |
| GPU_ENABLED | Enable GPU acceleration | true | No |
| USE_TENSORRT | Enable TensorRT optimization | true | No |
| TENSORRT_PRECISION | TensorRT precision (fp32, fp16, int8) | fp16 | No |
| BATCH_SIZE | Default batch size | 32 | No |
| MAX_BATCH_SIZE | Maximum batch size | 128 | No |
| ENABLE_MULTI_GPU | Enable multi-GPU processing | true | No |
| DALI_ENABLED | Enable NVIDIA DALI for data loading | true | No |
| USE_TRANSFORMER_ENGINE | Enable Transformer Engine | true | No |

## Services

- **API**: LangChain integration API service
- **Frontend**: Web interface for API interaction
- **Triton Server**: NVIDIA Triton Inference Server
- **DCGM Exporter**: GPU metrics exporter
- **Prometheus**: Metrics collection and monitoring

## Repository Structure

- **api/**: API source code
- **frontend/**: Frontend source code
- **langchain_hana/**: LangChain integration library
- **scripts/**: Utility scripts
- **docker/**: Docker configuration files
EOF < /dev/null