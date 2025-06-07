# NVIDIA Deployment Guide for SAP HANA Cloud LangChain Integration

This guide explains how to deploy the SAP HANA Cloud LangChain Integration on NVIDIA GPU-enabled infrastructure for maximum performance.

## System Requirements

- NVIDIA GPU with at least 8GB memory (T4, V100, A100, etc.)
- NVIDIA Driver: 525.60.13 or newer
- NVIDIA Container Toolkit (nvidia-docker2)
- Docker and Docker Compose
- 50GB+ storage space
- 32GB+ system memory

## Quick Start Deployment

1. Clone the repository
2. Run the deployment script:
   ```bash
   ./deploy_nvidia.sh
   ```
3. Follow the prompts to configure your SAP HANA Cloud connection
4. Access the application at http://localhost:8000

## Deployment Architecture

The deployment consists of several containerized services:

- **API Service**: FastAPI application for SAP HANA Cloud LangChain integration
- **Frontend**: Web interface for interacting with the API
- **Triton Server**: NVIDIA Triton Inference Server for optimized model serving
- **DCGM Exporter**: NVIDIA GPU metrics exporter
- **Prometheus**: Metrics collection and monitoring
- **Continuous Learning**: Parameter optimization service

![Deployment Architecture](docs/nvidia_deployment_architecture.png)

## Configuration Options

The deployment can be configured using environment variables in the `.env.nvidia` file:

### GPU Optimization
- `USE_TENSORRT`: Enable TensorRT optimization (default: true)
- `TENSORRT_PRECISION`: TensorRT precision mode - fp32, fp16, int8 (default: fp16)
- `BATCH_SIZE`: Default batch size for operations (default: 32)
- `MAX_BATCH_SIZE`: Maximum batch size for operations (default: 128)
- `ENABLE_MULTI_GPU`: Enable multi-GPU processing (default: true)
- `GPU_MEMORY_FRACTION`: Fraction of GPU memory to use (default: 0.9)
- `DALI_ENABLED`: Enable NVIDIA DALI for accelerated data loading (default: true)
- `USE_TRANSFORMER_ENGINE`: Enable NVIDIA Transformer Engine optimizations (default: true)
- `AUTO_TUNE_ENABLED`: Enable automatic parameter tuning (default: true)

### SAP HANA Connection
- `HANA_HOST`: SAP HANA Cloud host (required)
- `HANA_PORT`: SAP HANA Cloud port (default: 443)
- `HANA_USER`: SAP HANA Cloud username (required)
- `HANA_PASSWORD`: SAP HANA Cloud password (required)
- `DEFAULT_TABLE_NAME`: Default table for vector store (default: EMBEDDINGS)

### API Configuration
- `LOG_LEVEL`: Logging level (default: INFO)
- `ENABLE_CORS`: Enable CORS for API requests (default: true)
- `CORS_ORIGINS`: Allowed CORS origins (default: https://example.com,http://localhost:3000)
- `JWT_SECRET`: JWT authentication secret key (required)

## Monitoring

The deployment includes comprehensive monitoring for all services:

### Health Checks
- API Health: http://localhost:8000/health/ping
- API Status: http://localhost:8000/health/status
- GPU Info: http://localhost:8000/monitoring/gpu/info
- GPU Memory: http://localhost:8000/monitoring/gpu/memory

### Metrics
- Prometheus UI: http://localhost:9090
- DCGM GPU Metrics: http://localhost:9400/metrics
- API Metrics: http://localhost:8000/monitoring/metrics
- Triton Metrics: http://localhost:8002/metrics

## Troubleshooting

### Service Fails to Start

Check the logs for each service:
```bash
docker compose -f docker/docker-compose.nvidia.yml logs api
docker compose -f docker/docker-compose.nvidia.yml logs dcgm-exporter
docker compose -f docker/docker-compose.nvidia.yml logs triton-server
```

### GPU Not Detected

Check NVIDIA driver installation:
```bash
nvidia-smi
```

Verify NVIDIA Container Toolkit is properly installed:
```bash
docker info | grep -i runtime
```

### API Performance Issues

1. Check GPU utilization:
   ```bash
   curl http://localhost:9400/metrics | grep DCGM_FI_DEV_GPU_UTIL
   ```

2. Examine TensorRT engine status:
   ```bash
   docker exec sap-hana-langchain-api-nvidia ls -la /app/trt_engines
   ```

3. Check API logs for warnings or errors:
   ```bash
   docker compose -f docker/docker-compose.nvidia.yml logs api | grep -i warning
   ```

4. Verify auto-tuning has completed:
   ```bash
   docker exec sap-hana-langchain-api-nvidia cat /app/config/auto_tuned_config.json
   ```

## Advanced Configuration

### Custom TensorRT Optimization

To customize TensorRT optimization:

1. Edit the auto-tune configuration:
   ```bash
   docker exec -it sap-hana-langchain-api-nvidia vi /app/config/auto_tuned_config.json
   ```

2. Restart the API service:
   ```bash
   docker compose -f docker/docker-compose.nvidia.yml restart api
   ```

### Multi-GPU Setup

For multi-GPU environments:

1. Ensure `ENABLE_MULTI_GPU=true` in your `.env.nvidia` file
2. Edit the Docker Compose file to allocate GPUs:
   ```yaml
   deploy:
     resources:
       reservations:
         devices:
           - driver: nvidia
             count: all  # Use all available GPUs
             capabilities: [gpu]
   ```

## Performance Benchmarks

Typical performance metrics on different GPU hardware:

| GPU Model | Batch Size | Embedding Throughput | Memory Usage |
|-----------|------------|----------------------|--------------|
| T4        | 32         | ~2000 texts/sec      | ~6 GB        |
| V100      | 64         | ~5000 texts/sec      | ~10 GB       |
| A100      | 128        | ~10000 texts/sec     | ~20 GB       |

## Continuous Learning

The deployment includes a continuous learning service that automatically optimizes configuration parameters based on observed performance. The service:

1. Monitors system performance metrics
2. Experiments with different parameter values
3. Learns optimal configurations for your specific workload
4. Saves optimized configurations for future use

To view the current learned configuration:
```bash
docker exec sap-hana-langchain-continuous-learning cat /app/config/learned/best_parameters.json
```

## Security Considerations

1. Always set a strong `JWT_SECRET` for API authentication
2. Restrict CORS origins to trusted domains
3. Use a dedicated database user with appropriate permissions
4. Keep all services behind a secure reverse proxy in production

## Further Resources

- [SAP HANA Cloud Documentation](https://help.sap.com/docs/hana-cloud)
- [LangChain Documentation](https://python.langchain.com/docs/get_started)
- [NVIDIA TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)
- [NVIDIA Triton Server Documentation](https://github.com/triton-inference-server/server)