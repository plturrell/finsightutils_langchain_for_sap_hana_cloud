# SAP HANA Cloud LangChain Integration for NVIDIA Brev LaunchPad

This blueprint deploys a GPU-accelerated vector search service for SAP HANA Cloud integration using NVIDIA GPUs on Brev LaunchPad.

## Features

- **TensorRT-optimized embedding generation**: Leverage NVIDIA TensorRT for up to 5x faster embedding generation
- **Multi-GPU support**: Scale across multiple GPUs for higher throughput
- **Precision controls**: FP32, FP16, and INT8 precision options for optimal performance/accuracy tradeoffs
- **Dynamic batch sizing**: Automatically adjust batch sizes based on available GPU memory
- **SAP HANA Cloud integration**: Connect directly to your SAP HANA Cloud instance
- **Vector similarity search**: Fast semantic search using cosine similarity
- **Web UI**: Visualization interface for exploring vector embeddings

## Requirements

- SAP HANA Cloud instance with credentials
- NVIDIA GPU (T4 or better recommended)
- Brev LaunchPad account

## Quick Start

1. Deploy this blueprint on Brev LaunchPad
2. Configure the required environment variables:
   - `HANA_HOST`: Your SAP HANA Cloud host
   - `HANA_PORT`: Your SAP HANA Cloud port (usually 443)
   - `HANA_USER`: Your SAP HANA Cloud username
   - `HANA_PASSWORD`: Your SAP HANA Cloud password
3. Access the API at the provided endpoint
4. Access the frontend at the provided endpoint

## Configuration Options

### GPU Acceleration

- `GPU_ENABLED`: Enable GPU acceleration (default: true)
- `USE_TENSORRT`: Enable TensorRT optimization (default: true)
- `TENSORRT_PRECISION`: Precision for TensorRT (fp32, fp16, int8) (default: fp16)
- `BATCH_SIZE`: Initial batch size for embedding generation (default: 32)
- `MAX_BATCH_SIZE`: Maximum batch size (default: 128)
- `ENABLE_MULTI_GPU`: Enable multi-GPU support (default: true)
- `GPU_MEMORY_FRACTION`: Fraction of GPU memory to use (default: 0.9)

### SAP HANA Cloud

- `DEFAULT_TABLE_NAME`: Default table for vector store (default: EMBEDDINGS)
- `DB_MAX_CONNECTIONS`: Maximum database connections (default: 5)
- `DB_CONNECTION_TIMEOUT`: Connection timeout in seconds (default: 600)

## API Endpoints

- `/docs`: API documentation (Swagger UI)
- `/health/ping`: Health check endpoint
- `/gpu/info`: GPU information endpoint
- `/embeddings`: Generate embeddings
- `/texts`: Add texts to vector store
- `/query`: Query the vector store
- `/query/mmr`: Query with Maximal Marginal Relevance

## Troubleshooting

- **GPU not detected**: Ensure the Brev LaunchPad environment has GPU support
- **Connection errors**: Check your SAP HANA Cloud credentials
- **Performance issues**: Try adjusting batch size and precision settings

## Support

For issues or questions, please open an issue on the [GitHub repository](https://github.com/plturrell/finsightutils_langchain_for_sap_hana_cloud).