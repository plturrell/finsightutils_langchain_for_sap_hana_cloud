# LangChain Integration with SAP HANA Cloud - Examples

This directory contains example scripts demonstrating how to use LangChain with SAP HANA Cloud's vector database capabilities. These examples showcase various integration patterns and use cases for building AI applications with SAP HANA Cloud as the vector store backend.

## NEW: Financial Embeddings and Fine-Tuning

We've added new examples for financial domain-specific embeddings and fine-tuning capabilities:

### Custom Fine-Tuning (`custom_fine_tuning.py`)

Demonstrates how to generate custom training data from your own financial documents and fine-tune the FinMTEB/Fin-E5 model for your specific domain.

```bash
python custom_fine_tuning.py --documents-file ../documents.json --train-output custom_train.json --val-output custom_val.json
```

This script:
- Loads your financial documents from a JSON file
- Generates training pairs using TF-IDF similarity and metadata matching
- Creates representative query-document pairs
- Splits data into training and validation sets
- Saves the data in the format required for fine-tuning

After running this script, you can fine-tune the model with:

```bash
python ../finetune_fin_e5.py --train-file custom_train.json --val-file custom_val.json --training-format pairs
```

For more details on fine-tuning, see the [`FINE_TUNING_GUIDE.md`](../FINE_TUNING_GUIDE.md) file.

## Prerequisites

- SAP HANA Cloud instance with connection details (host, port, username, password)
- Python 3.8+ environment
- Required Python packages:
  - `langchain` and `langchain_core`
  - `langchain_hana` (this package)
  - `sentence-transformers` (for embedding generation)
  - `hdbcli` (SAP HANA Python client)
  - `torch` (for GPU acceleration examples)
  - `openai` (for RAG example with LLM)
  - `scikit-learn` (for custom fine-tuning example)

## Connection Configuration

All examples look for SAP HANA Cloud connection details in the following locations (in order):

1. JSON file: `connection.json` in the current directory or `config/connection.json`
2. Environment variables: `HANA_HOST`, `HANA_PORT`, `HANA_USER`, `HANA_PASSWORD`

Example connection.json format:
```json
{
  "address": "your-hana-instance.hanacloud.ondemand.com",
  "port": 443,
  "user": "DBADMIN",
  "password": "your-password"
}
```

## Examples

### 1. Basic Quickstart (`langchain_hana_quickstart.py`)

A simple example demonstrating the core functionality of the LangChain integration with SAP HANA Cloud:

- Connecting to SAP HANA Cloud
- Creating a vector store
- Adding documents with metadata
- Performing similarity search
- Filtering search results
- Using Maximal Marginal Relevance (MMR) for diverse results
- Updating documents

```bash
python langchain_hana_quickstart.py
```

### 2. GPU-Accelerated Integration (`langchain_hana_gpu_quickstart.py`)

An advanced example showing how to leverage GPU acceleration for high-performance vector operations:

- TensorRT-optimized embeddings with HanaTensorRTEmbeddings
- GPU-accelerated vector store with HanaGPUVectorStore
- Parallel processing with multi-GPU support
- Performance benchmarking and monitoring
- Asynchronous operations for improved throughput

```bash
python langchain_hana_gpu_quickstart.py
```

### 3. Retrieval-Augmented Generation (RAG) Example (`langchain_hana_rag_example.py`)

A complete RAG application using SAP HANA Cloud as the vector database:

- Document preparation and text splitting
- Storing document chunks with metadata
- Creating a retriever with filtering capabilities
- Building a RAG chain with LangChain and an LLM
- Answering questions based on retrieved context

```bash
python langchain_hana_rag_example.py
```

## Additional Examples

### Direct API Testing (`direct_test.py`)

Tests direct connections to the SAP HANA Cloud API without using LangChain.

### Multi-GPU Embedding Demo (`multi_gpu_embeddings_demo.py`)

Demonstrates how to use multiple GPUs for parallel embedding generation.

### Update Operations (`update_operations.py`)

Shows advanced document update and upsert operations with the vector store.

## Usage Notes

- These examples create temporary tables that are cleaned up at the end of execution
- For production use, you may want to persist the vector store tables
- Performance will vary based on your SAP HANA Cloud instance size and capabilities
- GPU acceleration requires NVIDIA GPUs with CUDA support

## Troubleshooting

If you encounter connection issues:
- Verify your SAP HANA Cloud instance is running
- Check that your connection parameters are correct
- Ensure your network allows connections to the SAP HANA Cloud port

For GPU acceleration issues:
- Verify CUDA is installed and configured correctly
- Check GPU compatibility with TensorRT
- Monitor GPU memory usage for potential out-of-memory errors