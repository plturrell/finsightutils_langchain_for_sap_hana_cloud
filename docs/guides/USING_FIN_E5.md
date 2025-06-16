# Using FinMTEB/Fin-E5 with SAP HANA Cloud

This document explains how to use the high-quality FinMTEB/Fin-E5 embedding model with SAP HANA Cloud for financial document processing.

## About FinMTEB/Fin-E5

FinMTEB/Fin-E5 is a state-of-the-art financial embedding model released in 2025. It is based on the Mistral-7B architecture and has been specifically trained on financial data. This model achieves a **0.6767 average score** on the FinMTEB benchmark, showing superior performance on financial text.

Key advantages:
- 7B parameters for high-quality embeddings
- Optimized for financial domain understanding
- Excellent performance on financial classification and similarity tasks
- 4.5% improvement over general-purpose models

## System Requirements

To run FinMTEB/Fin-E5 efficiently, you need:
- **CPU**: At least 8 cores recommended
- **RAM**: Minimum 16GB, 32GB+ recommended
- **GPU**: NVIDIA GPU with 16GB+ VRAM recommended (not required but highly beneficial)
- **Disk**: At least 15GB free space for model files
- **Python**: 3.8 or higher

## Quick Start

We've configured the system to use FinMTEB/Fin-E5 by default. The simplest way to use it is with the `run_fin_e5.sh` script:

```bash
# Add documents to the system
./run_fin_e5.sh add --input-file documents.json

# Process queries
./run_fin_e5.sh query --input-file queries.json --output-file results.json

# Check system health
./run_fin_e5.sh health

# Get performance metrics
./run_fin_e5.sh metrics
```

## First Run

On first run, the system will:
1. Download the FinMTEB/Fin-E5 model (approximately 14GB)
2. Cache it locally in `./financial_models`
3. Use the cached model for all subsequent operations

This initial download may take 5-20 minutes depending on your internet connection.

## Advanced Configuration

### Disable GPU

If you're experiencing GPU memory issues, you can force CPU-only mode:

```bash
CUDA_VISIBLE_DEVICES="" ./run_fin_e5.sh query --input-file queries.json
```

### Custom Model Directory

To store the model in a custom location:

```bash
./run_fin_e5.sh query --input-file queries.json --models-dir /path/to/models
```

### Connecting to Different SAP HANA Instance

Edit the `run_fin_e5.sh` script to update the connection parameters:

```bash
HANA_HOST="your-host.hana.ondemand.com"
HANA_PORT=443
HANA_USER="your-username"
HANA_PASSWORD="your-password"
```

## Performance Optimization

For optimal performance with FinMTEB/Fin-E5:

1. **GPU Acceleration**: Use a CUDA-capable GPU with at least 16GB VRAM
2. **Batch Processing**: Process documents in batches rather than individually
3. **Caching**: Enable semantic caching for similar queries
4. **Mixed Precision**: The system automatically uses mixed precision (FP16) on compatible GPUs

## Monitoring

Monitor system performance with:

```bash
./run_fin_e5.sh metrics
```

Key metrics to watch:
- Embedding generation time
- Query processing time
- Cache hit rate
- GPU memory usage (if applicable)

## Troubleshooting

### Model Download Issues

If you encounter issues downloading the model:

1. Check your internet connection
2. Ensure you have sufficient disk space
3. Try downloading manually:
   ```bash
   python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('FinMTEB/Fin-E5', cache_folder='./financial_models/FinMTEB/Fin-E5')"
   ```

### Out of Memory Errors

If you see CUDA out of memory errors:

1. Force CPU mode as described above
2. Reduce batch size with `--batch-size 8` (default is 32)
3. Try the smaller model with `--model-name FinMTEB/Fin-E5-small`

### Slow Performance

If performance is slow:

1. Check if you're using GPU acceleration
2. Enable caching if not already enabled
3. Consider using a more efficient model for less critical applications