# SAP HANA Cloud Quick Start Guide

This guide will help you get started with SAP HANA Cloud integration for LangChain.

## Step 1: Set Up HANA Credentials

Run the credentials setup script:

```bash
./setup_hana_credentials.sh
```

You'll be prompted to enter:
- SAP HANA Host (e.g., your-hana-host.hanacloud.ondemand.com)
- SAP HANA Port (default: 443)
- SAP HANA User (default: SYSTEM)
- SAP HANA Password
- Default Table Name (default: EMBEDDINGS)

## Step 2: Test HANA Connection

Run the connection test script:

```bash
python test_hana_connection.py
```

This script will:
- Load credentials from the .env file
- Attempt to connect to your HANA Cloud instance
- Run a simple query to verify the connection
- Display version information and available schemas

## Step 3: Set Up HANA Tables for Vector Storage

Run the table setup script:

```bash
python setup_hana_tables.py
```

This script will:
- Create a schema if it doesn't exist
- Create a table for storing document embeddings
- Set up vector indexing for similarity search

## Step 4: Start the API with HANA Integration

Start the API service:

```bash
./brev_deploy.sh
```

This will:
- Install required dependencies
- Set up the API in test mode
- Connect to your HANA Cloud instance
- Start the service on port 8000

To check the API status:

```bash
curl http://localhost:8000/health/ping
```

## Step 5: Configure Real HANA Connection

Once the API is running, configure it to use your real HANA connection:

```bash
./configure_hana.sh
```

Follow the prompts to enter your HANA credentials again, and the script will:
- Create configuration files
- Test the connection
- Provide instructions for restarting the API with the new configuration

## Using in Jupyter Notebook

Once the setup is complete, you can use the LangChain HANA integration in your Jupyter notebooks:

```python
from langchain_hana import HanaVectorStore
from langchain.embeddings import HuggingFaceEmbeddings

# Initialize embeddings model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize HANA vector store
vector_store = HanaVectorStore(
    host=os.environ.get("HANA_HOST"),
    port=int(os.environ.get("HANA_PORT")),
    user=os.environ.get("HANA_USER"),
    password=os.environ.get("HANA_PASSWORD"),
    table_name=os.environ.get("DEFAULT_TABLE_NAME"),
    embeddings=embeddings
)

# Add documents
vector_store.add_texts(
    ["SAP HANA is a high-performance in-memory database", 
     "LangChain provides a framework for LLM applications",
     "Vector databases are optimized for similarity search"]
)

# Perform similarity search
results = vector_store.similarity_search("What is SAP HANA?", k=1)
print(results[0].page_content)
```

## Troubleshooting

If you encounter issues:

1. Check connection parameters:
   - Verify the host, port, user, and password are correct
   - Ensure your user has the necessary permissions

2. Verify network access:
   - Check that your instance can connect to the HANA Cloud endpoint
   - Ensure firewall rules allow outbound connections to your HANA instance

3. Check logs:
   - API logs are in the `logs` directory
   - Look for specific error messages related to HANA connection