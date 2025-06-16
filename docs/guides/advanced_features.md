# Advanced Features Guide

This guide covers advanced features and techniques for the LangChain SAP HANA Cloud integration.

## HNSW Vector Indexing

Hierarchical Navigable Small World (HNSW) indexing significantly accelerates similarity searches in large vector datasets.

### Creating an HNSW Index

```python
# Create with default parameters
vectorstore.create_hnsw_index()

# Create with custom parameters
vectorstore.create_hnsw_index(
    m=64,                # Number of connections per node (higher = more accurate, more memory)
    ef_construction=200, # Search width during construction (higher = more accurate, slower build)
    ef_search=100,       # Search width during query (higher = more accurate, slower query)
    index_name="my_custom_index"  # Custom index name
)
```

### HNSW Parameter Tuning

| Parameter | Description | Default | Recommendation |
|-----------|-------------|---------|---------------|
| `m` | Connections per node | Default by HANA | 32-128 (balance of speed and accuracy) |
| `ef_construction` | Build-time search width | Default by HANA | 100-500 (higher for more accuracy) |
| `ef_search` | Query-time search width | Default by HANA | 50-200 (higher for more accuracy) |

## Maximal Marginal Relevance (MMR)

MMR balances relevance with diversity in search results, reducing redundancy.

```python
# Regular similarity search (may return similar documents)
results = vectorstore.similarity_search("quantum computing", k=5)

# MMR search (balances relevance and diversity)
diverse_results = vectorstore.max_marginal_relevance_search(
    "quantum computing",
    k=5,              # Number of documents to return
    fetch_k=20,       # Number of documents to consider
    lambda_mult=0.7   # Balance between relevance (1.0) and diversity (0.0)
)
```

## Metadata Filtering

Filter search results based on metadata attributes:

```python
# Simple equality filter
results = vectorstore.similarity_search(
    "quantum computing",
    filter={"category": "physics"}
)

# Numeric comparison filter
results = vectorstore.similarity_search(
    "quantum computing",
    filter={"year": {"$gte": 2020}}
)

# Complex logical filter
results = vectorstore.similarity_search(
    "quantum computing",
    filter={
        "$and": [
            {"category": "physics"},
            {"$or": [
                {"year": {"$gte": 2020}},
                {"importance": {"$gt": 8}}
            ]}
        ]
    }
)
```

### Available Filter Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `$eq` | Equal to | `{"field": {"$eq": value}}` |
| `$ne` | Not equal to | `{"field": {"$ne": value}}` |
| `$lt` | Less than | `{"field": {"$lt": value}}` |
| `$lte` | Less than or equal | `{"field": {"$lte": value}}` |
| `$gt` | Greater than | `{"field": {"$gt": value}}` |
| `$gte` | Greater than or equal | `{"field": {"$gte": value}}` |
| `$in` | In array | `{"field": {"$in": [value1, value2]}}` |
| `$nin` | Not in array | `{"field": {"$nin": [value1, value2]}}` |
| `$contains` | Text contains | `{"field": {"$contains": "text"}}` |
| `$like` | SQL LIKE pattern | `{"field": {"$like": "%pattern%"}}` |
| `$between` | Between values | `{"field": {"$between": [min, max]}}` |

## Using Specific Metadata Columns

For frequently filtered metadata fields, use specific columns for better performance:

```python
vectorstore = HanaDB(
    connection=connection,
    embedding=embedding,
    specific_metadata_columns=["category", "year", "author"]
)
```

## Internal Embeddings

SAP HANA Cloud provides built-in embedding functions for improved performance:

```python
from langchain_hana import HanaInternalEmbeddings

# Initialize with internal embedding model
internal_emb = HanaInternalEmbeddings(internal_embedding_model_id="SAP_NEB.20240715")

# Use with vector store
vectorstore = HanaDB(
    connection=connection,
    embedding=internal_emb
)

# Add documents
vectorstore.add_texts(texts, metadatas)

# Search documents
results = vectorstore.similarity_search("query text")
```

## Knowledge Graph Integration

Query knowledge graphs using SPARQL and natural language:

```python
from langchain_hana import HanaRdfGraph, HanaSparqlQAChain
from langchain_openai import ChatOpenAI

# Connect to the RDF graph
graph = HanaRdfGraph(
    connection=connection,
    graph_uri="http://example.com/mygraph",
    auto_extract_ontology=True  # Automatically extract schema
)

# Create a QA chain
qa_chain = HanaSparqlQAChain.from_llm(
    llm=ChatOpenAI(),
    graph=graph,
    allow_dangerous_requests=True  # Required acknowledgment
)

# Ask questions
response = qa_chain.invoke("Who is the CEO of SAP?")
print(response["result"])
```

## Asynchronous Operations

For high-throughput applications, use asynchronous methods:

```python
import asyncio

async def search_async():
    # Asynchronous MMR search
    docs = await vectorstore.amax_marginal_relevance_search(
        "quantum computing",
        k=5,
        fetch_k=20
    )
    return docs

# Delete documents asynchronously
await vectorstore.adelete(filter={"category": "outdated"})
```

## Vector Data Types

SAP HANA Cloud supports different vector data types:

```python
# Standard 32-bit float vectors (default)
vectorstore = HanaDB(
    connection=connection,
    embedding=embedding,
    vector_column_type="REAL_VECTOR"
)

# Memory-efficient 16-bit float vectors (if supported by your HANA version)
vectorstore = HanaDB(
    connection=connection,
    embedding=embedding,
    vector_column_type="HALF_VECTOR"
)
```

## Connection Pooling

For production applications, implement connection pooling:

```python
import threading
from queue import Queue
from hdbcli import dbapi

class HanaConnectionPool:
    def __init__(self, pool_size=10, **connection_params):
        self.connection_params = connection_params
        self.pool = Queue(maxsize=pool_size)
        self.size = 0
        self.max_size = pool_size
        self.lock = threading.Lock()
        
    def get_connection(self):
        if not self.pool.empty():
            return self.pool.get()
            
        with self.lock:
            if self.size < self.max_size:
                conn = dbapi.connect(**self.connection_params)
                self.size += 1
                return conn
            else:
                return self.pool.get(block=True)
    
    def return_connection(self, connection):
        self.pool.put(connection)

# Create a connection pool
pool = HanaConnectionPool(
    pool_size=5,
    address="your-hana-hostname.hanacloud.ondemand.com",
    port=443,
    user="your_user",
    password="your_password",
    encrypt=True
)

# Use a connection from the pool
conn = pool.get_connection()
try:
    vectorstore = HanaDB(connection=conn, embedding=embedding)
    # Use vectorstore...
finally:
    # Return connection to the pool
    pool.return_connection(conn)
```

## Batch Operations

For improved performance when adding multiple documents:

```python
# Adding texts in a single batch operation
texts = ["Document 1", "Document 2", "Document 3", ...]
metadatas = [{"source": "source1"}, {"source": "source2"}, {"source": "source3"}, ...]

# Add them all at once
vectorstore.add_texts(texts, metadatas)
```

## Error Handling and Retries

Implement robust error handling for production applications:

```python
import time
from hdbcli import dbapi

max_retries = 3
retry_delay = 2  # seconds

for attempt in range(max_retries):
    try:
        # Attempt database operation
        results = vectorstore.similarity_search("quantum computing", k=5)
        break  # Success, exit the retry loop
    except dbapi.Error as e:
        if attempt < max_retries - 1:
            print(f"Database error: {e}. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff
        else:
            print(f"Failed after {max_retries} attempts: {e}")
            raise
```

## Advanced SPARQL Queries

For complex knowledge graph queries:

```python
# Define a custom SPARQL query
custom_query = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX ex: <http://example.org/>

SELECT ?person ?name ?email
WHERE {
    ?person rdf:type ex:Employee .
    ?person ex:name ?name .
    ?person ex:email ?email .
    ?person ex:department ?dept .
    ?dept ex:name "Research" .
}
ORDER BY ?name
LIMIT 10
"""

# Execute the query
result = graph.query(custom_query)
print(result)
```

## Conclusion

These advanced features enable sophisticated AI applications powered by SAP HANA Cloud's vector and graph capabilities. For production deployments, also refer to our [Security Guide](security_guide.md) and [Configuration Guide](configuration_guide.md).