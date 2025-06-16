# LangChain Integration for SAP HANA Cloud - Usage Guide

This guide provides instructions for setting up and using the LangChain integration with SAP HANA Cloud.

## Setup

1. **Configure your SAP HANA Cloud connection:**

   Copy the template configuration file and add your connection details:
   ```bash
   cp config/connection.json.template config/connection.json
   ```
   
   Edit `config/connection.json` with your SAP HANA Cloud credentials.

   Alternatively, set environment variables:
   ```bash
   export HANA_HOST=your-hana-instance.hanacloud.ondemand.com
   export HANA_PORT=443
   export HANA_USER=DBADMIN
   export HANA_PASSWORD=your-password
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Basic Usage

The main integration class is available in `app.py`. Here's how to use it:

```python
from app import SAP_HANA_Langchain_Integration

# Initialize the integration
integration = SAP_HANA_Langchain_Integration()

# Add documents
documents = [
    "SAP HANA Cloud is a cloud-based database management system.",
    "LangChain is a framework for developing LLM applications."
]

metadata = [
    {"source": "SAP Documentation", "category": "database"},
    {"source": "LangChain Documentation", "category": "framework"}
]

integration.add_documents(documents, metadata)

# Search for similar documents
results = integration.search("What is SAP HANA Cloud?", k=2)

# Print results
for doc in results:
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
    print()

# Clean up
integration.close()
```

## Advanced Features

### Filtered Search

```python
# Search only within documents with specific metadata
filtered_results = integration.search(
    "database capabilities",
    k=3,
    filter={"category": "database"}
)
```

### Diverse Search (MMR)

```python
# Get diverse results using Maximal Marginal Relevance
diverse_results = integration.diverse_search(
    "cloud database",
    k=3,
    fetch_k=10,
    lambda_mult=0.7  # Balance between relevance and diversity
)
```

### Updating Documents

```python
# Update documents matching a filter
integration.update_documents(
    texts=["Updated content about SAP HANA Cloud."],
    filter={"source": "SAP Documentation"}
)
```

### Deleting Documents

```python
# Delete documents matching a filter
integration.delete_documents(
    filter={"category": "outdated"}
)
```

## Testing the Integration

Run the basic example included in the app:

```bash
python app.py
```

This will:
1. Connect to your SAP HANA Cloud instance
2. Create a vector store table
3. Add sample documents
4. Perform searches and display results

## Troubleshooting

- **Connection Issues**: Verify your SAP HANA Cloud instance is running and accessible. Check that your connection parameters are correct.

- **Missing Dependencies**: Ensure all required packages are installed. The key dependencies are `langchain`, `sentence-transformers`, and `hdbcli`.

- **Table Permissions**: The user specified in your connection details needs permission to create and modify tables in the SAP HANA database.

- **Performance Issues**: For large document collections, consider using the GPU-accelerated version of the integration found in the examples directory.

## Next Steps

Once you have the basic integration working, you can:

1. Integrate with your application's document storage
2. Connect to an LLM for RAG (Retrieval-Augmented Generation) applications
3. Scale up with the GPU-accelerated implementation for better performance
4. Create custom retrieval strategies based on your specific use case