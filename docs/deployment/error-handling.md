# Error Handling and Troubleshooting Guide

The SAP HANA Cloud LangChain integration includes a sophisticated error handling system that provides context-aware error messages, detailed diagnostics, and suggested remediation steps. This guide explains how to effectively use and configure the error handling features.

## Table of Contents

1. [Error Handling Architecture](#error-handling-architecture)
2. [Common Error Categories](#common-error-categories)
3. [Configuring Error Detail Levels](#configuring-error-detail-levels)
4. [Handling Database Errors](#handling-database-errors)
5. [GPU-Related Error Handling](#gpu-related-error-handling)
6. [Error Logging and Monitoring](#error-logging-and-monitoring)
7. [Error Codes Reference](#error-codes-reference)
8. [Troubleshooting Common Issues](#troubleshooting-common-issues)

## Error Handling Architecture

The error handling system consists of several key components:

1. **Context-aware error handler**: `handle_database_error()` in `error_utils.py`
2. **Error category detection**: Categorizes errors based on operation and error text
3. **Remediation suggestions**: Provides targeted advice for resolving issues
4. **Detailed error formatting**: Formats errors with multiple detail levels
5. **Logging integration**: Records errors with appropriate context

The system is designed to:
- Provide meaningful error messages instead of generic database errors
- Include context about the operation being performed
- Suggest specific actions to resolve the issue
- Support different detail levels for development vs. production

## Common Error Categories

The error handling system categorizes errors into the following types:

| Category | Description | Examples |
|----------|-------------|----------|
| `connection` | Database connection issues | Connection timeout, authentication failures |
| `table_creation` | Issues with creating tables | Invalid schema, permission denied |
| `add_texts` | Problems adding documents | Invalid vector format, metadata errors |
| `embedding_generation` | Embedding creation failures | Model not found, out of memory |
| `similarity_search` | Vector search issues | Invalid query format, HNSW index errors |
| `index_creation` | Vector index creation problems | Invalid parameters, duplicate index |
| `general` | Other database errors | General SQL syntax errors |

## Configuring Error Detail Levels

The error detail level can be configured through environment variables:

```bash
# Options: minimal, standard, verbose
export ERROR_DETAIL_LEVEL=standard

# Options: true, false
export INCLUDE_SUGGESTIONS=true
```

Detail levels include:
- **minimal**: Basic error message without context or suggestions
- **standard**: Error message with context and common suggestions
- **verbose**: Detailed error information with all available context, SQL statements, and comprehensive suggestions

In production environments, you might want to set `ERROR_DETAIL_LEVEL=minimal` and log the verbose details for internal debugging.

## Handling Database Errors

### Example 1: Connection Error

```python
from langchain_hana.error_utils import handle_database_error

try:
    # Attempt database connection
    connection = dbapi.connect(...)
except dbapi.Error as e:
    additional_context = {
        "host": host,
        "port": port,
        "user": "******",  # Redacted for security
    }
    handle_database_error(e, "connection", additional_context)
```

Error output:
```
Error connecting to SAP HANA Cloud: Connection refused

Issue: The application cannot establish a connection to the SAP HANA database.

Possible causes:
1. Incorrect host or port in connection string
2. Network connectivity issues
3. Firewall blocking the connection
4. Database instance is not running

Suggestions:
1. Verify your connection parameters (host: sap-hana.example.com, port: 443)
2. Check network connectivity to the database
3. Verify the database instance is running
4. Check firewall rules to ensure the port is open
```

### Example 2: Vector Table Error

```python
try:
    # Create vector table
    cursor.execute(create_table_sql)
except dbapi.Error as e:
    additional_context = {
        "table_name": table_name,
        "vector_column_type": vector_column_type,
        "vector_column_length": vector_column_length
    }
    handle_database_error(e, "table_creation", additional_context)
```

Error output:
```
Error creating vector table: Invalid data type 'HALF_VECTOR' for column

Issue: Your database does not support the HALF_VECTOR data type.

Your current instance version: 2024.2.0 (QRC 1/2024)
Required version: 2025.15 (QRC 2/2025)

Solutions:
1. Use 'REAL_VECTOR' instead (supported in older versions)
2. Upgrade your SAP HANA Cloud instance to 2025.15 (QRC 2/2025) or newer
3. Contact your SAP HANA Cloud administrator to check for available vector types
```

## GPU-Related Error Handling

GPU-related errors include specialized handling for:

1. **CUDA initialization errors**
2. **Out of memory (OOM) errors**
3. **TensorRT engine compilation failures**
4. **Missing GPU dependencies**

Example of a GPU dependency error:

```python
try:
    from langchain_hana.embeddings import HanaTensorRTMultiGPUEmbeddings
    embeddings = HanaTensorRTMultiGPUEmbeddings()
except ImportError as e:
    print(f"Error: {str(e)}")
```

Error output:
```
Error initializing TensorRT embeddings: No module named 'tensorrt'.

Make sure TensorRT and related dependencies are installed:
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu118 tensorrt>=8.6.0 pycuda>=2022.2
```

### Automatic Recovery from GPU Errors

The system includes several automatic recovery mechanisms:

1. **Dynamic batch size adjustment**: Automatically reduces batch size when OOM errors occur
2. **GPU failover**: Automatically switches to another GPU if one fails
3. **CPU fallback**: Falls back to CPU execution if all GPUs fail
4. **Precision downgrading**: Attempts lower precision if higher precision fails

## Error Logging and Monitoring

### Structured Error Logging

Errors are logged in a structured format for easier analysis:

```json
{
  "timestamp": "2024-07-08T15:42:31Z",
  "level": "ERROR",
  "operation": "embedding_generation",
  "error_type": "CudaOutOfMemoryError",
  "message": "CUDA out of memory. Tried to allocate 2.00 GiB",
  "context": {
    "batch_size": 128,
    "model": "sentence-transformers/all-mpnet-base-v2",
    "gpu_id": 0,
    "gpu_memory_mb": 16384,
    "document_count": 1024
  },
  "resolution": "Batch size automatically reduced to 64"
}
```

### Configuring Error Logging

Error logging can be configured with:

```python
import logging
from langchain_hana.error_utils import configure_error_logging

configure_error_logging(
    log_file="/path/to/errors.log",
    log_level=logging.ERROR,
    format_json=True,
    include_context=True
)
```

## Error Codes Reference

The system uses standardized error codes to identify specific issues:

| Code | Category | Description | Remediation |
|------|----------|-------------|------------|
| `LCHANA-CONN-001` | Connection | Authentication failed | Verify credentials |
| `LCHANA-CONN-002` | Connection | Host not found | Verify host name and network |
| `LCHANA-CONN-003` | Connection | Connection timeout | Check network latency and firewall |
| `LCHANA-VEC-001` | Vectorstore | Column type not supported | Use supported vector type or upgrade |
| `LCHANA-VEC-002` | Vectorstore | Vector dimension mismatch | Ensure consistent dimensions |
| `LCHANA-VEC-003` | Vectorstore | Invalid metadata format | Fix metadata structure |
| `LCHANA-EMB-001` | Embeddings | Model not found | Verify model name and path |
| `LCHANA-EMB-002` | Embeddings | GPU out of memory | Reduce batch size |
| `LCHANA-GPU-001` | GPU | CUDA not available | Install CUDA dependencies |
| `LCHANA-GPU-002` | GPU | TensorRT compilation failed | Check model compatibility |

## Troubleshooting Common Issues

### Connection Issues

**Problem**: Unable to connect to SAP HANA Cloud

**Diagnostic Steps**:
1. Verify connection parameters (host, port, user, password)
2. Test network connectivity: `ping <host>`
3. Check if the port is open: `telnet <host> <port>`
4. Verify credentials with HANA SQL client

**Solution**:
- Ensure you're using the correct host/port
- Check if you need to connect through a VPN
- Use strong encryption for password (especially in CI/CD)

### Vector Type Errors

**Problem**: Unsupported vector type errors

**Diagnostic Steps**:
1. Check your HANA Cloud version: `SELECT CLOUD_VERSION FROM SYS.M_DATABASE`
2. Verify available datatypes: `SELECT TYPE_NAME FROM SYS.DATA_TYPES`

**Solution**:
- For HANA Cloud QRC 1/2024 or newer: Use `REAL_VECTOR`
- For HANA Cloud QRC 2/2025 or newer: Use `HALF_VECTOR` or `REAL_VECTOR`
- For older versions: Use a custom serialization format with BLOB type

### GPU Acceleration Issues

**Problem**: GPU acceleration not working

**Diagnostic Steps**:
1. Verify GPU availability: `nvidia-smi`
2. Check dependencies: `python -c "from langchain_hana.gpu.imports import get_gpu_features_status; print(get_gpu_features_status())"`
3. Test CUDA: `python -c "import torch; print(torch.cuda.is_available())"`

**Solution**:
- Install required dependencies: `pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu118`
- Verify NVIDIA driver installation
- Set environment variables: `MULTI_GPU_ENABLED=true`, `USE_TENSORRT=true`

### Embedding Generation Performance

**Problem**: Slow embedding generation

**Diagnostic Steps**:
1. Check batch size: Is it too small or too large?
2. Verify GPU utilization: `nvidia-smi dmon`
3. Check precision mode: Is FP16/INT8 enabled?

**Solution**:
- Increase batch size incrementally (try 32, 64, 128)
- Enable TensorRT optimization: `USE_TENSORRT=true`
- Use lower precision: `TENSORRT_PRECISION=fp16` or `int8`
- Enable Tensor Cores: `ENABLE_TENSOR_CORES=true`

### Memory Usage Issues

**Problem**: Out of memory errors during embedding generation

**Diagnostic Steps**:
1. Monitor GPU memory: `nvidia-smi -l 1`
2. Check batch size and model dimensions
3. Verify if dynamic batching is enabled

**Solution**:
- Reduce batch size
- Enable dynamic batching: `ENABLE_DYNAMIC_BATCHING=true`
- Use a smaller embedding model
- Enable memory optimization: `OPTIMIZE_MEMORY_USAGE=true`

## Getting Help

If you encounter issues that aren't covered by this guide:

1. Check the [GitHub issues](https://github.com/yourusername/langchain-integration-for-sap-hana-cloud/issues) for similar problems
2. Collect diagnostic information:
   ```bash
   python -m langchain_hana.diagnostics > diagnostics.log
   ```
3. Open a new GitHub issue with your diagnostics log and detailed reproduction steps