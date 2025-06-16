# SAP HANA Cloud LangChain Integration Testing

This repository contains testing scripts and documentation for the SAP HANA Cloud LangChain integration deployed on an NVIDIA T4 GPU in Brev Cloud.

## Overview

The SAP HANA Cloud LangChain integration enables vector search, knowledge graph querying, and other LLM-driven applications using SAP HANA Cloud's database capabilities. The deployment on an NVIDIA T4 GPU provides accelerated performance for embedding generation and vector operations.

## Testing Scripts

This repository includes the following testing resources:

1. **`create_test_data.py`** - Generates sample documents, test queries, and mock performance data for testing
2. **`tests/test_tensorrt_t4.py`** - Tests TensorRT optimizations specifically for T4 GPU
3. **`docs/T4_GPU_TESTING_PLAN.md`** - Comprehensive testing plan with notebook templates
4. **`tests/test_hana_api.py`** - Script for testing the deployed API endpoints
5. **`run_tests.sh`** - Shell script to run the automated tests
6. **`run_automated_tests.py`** - Main test orchestrator

## Prerequisites

- Access to the deployed Jupyter Lab instance at `https://jupyter0-513syzm60.brevlab.com`
- NVIDIA Enterprise authentication credentials
- SAP HANA Cloud connection details
- Python 3.7+ with required packages

## Getting Started

1. **Generate test data**:
   ```bash
   python create_test_data.py
   ```

2. **Upload testing scripts to the Jupyter instance**:
   Access the Jupyter Lab interface and upload the testing scripts and data.

3. **Test TensorRT optimization**:
   ```bash
   python tests/test_tensorrt_t4.py --precision fp16 --output results.json
   ```

4. **Run all tests**:
   ```bash
   ./run_tests.sh --all
   ```

5. **Follow the testing plan**:
   Open `docs/T4_GPU_TESTING_PLAN.md` and follow the structured testing approach.

## Testing Areas

The testing covers the following key areas:

1. **Environment Verification** - Validate GPU detection, drivers, and packages
2. **TensorRT Optimization** - Test embedding generation with TensorRT
3. **Core Functionality** - Test vector store operations
4. **GPU Acceleration** - Benchmark performance with T4 GPU
5. **Multi-Backend Deployment** - Test configuration switching
6. **Error Handling** - Test recovery and graceful degradation

## Performance Benchmarking

For each test, we collect the following metrics:

- **Latency** - Time to complete operations
- **Throughput** - Operations per second
- **Memory Usage** - Peak memory consumption
- **GPU Utilization** - Percentage of GPU compute resources used
- **CPU vs. GPU Speedup** - Comparative performance gains

## T4 GPU Optimization

The T4 GPU has the following specifications:
- 16GB GDDR6 memory
- 2,560 CUDA cores
- 320 Tensor cores
- 65W power consumption

Optimization recommendations:
- Use FP16 precision for the best balance of performance and accuracy
- Adjust batch sizes based on memory constraints (typically 64-128 for embedding models)
- Use TensorRT for maximum performance

## Important Notes

1. **Authentication**: The Jupyter instance requires NVIDIA Enterprise authentication through Cloudflare Access. Make sure you have valid credentials.

2. **Database Connection**: Ensure you have the correct connection parameters for the SAP HANA Cloud instance.

3. **GPU Memory Management**: T4 GPUs have 16GB of memory. Monitor memory usage to prevent out-of-memory errors.

4. **TensorRT Engine Building**: Building TensorRT engines can take several minutes. The engines are cached for future use.

## Reporting Issues

If you encounter any issues during testing, please document them with:
- Steps to reproduce
- Error messages
- Environment details
- Performance metrics

## Next Steps

After testing is complete:
1. Optimize configuration based on test results
2. Document best practices for T4 GPU deployment
3. Create deployment guides for different environments
4. Implement any fixes for identified issues