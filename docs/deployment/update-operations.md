# Update Operations Guide

This guide explains how to use the update operations in the SAP HANA Cloud LangChain Integration.

## Overview

The integration now supports complete CRUD operations (Create, Read, Update, Delete) for document management in the SAP HANA vector store. This includes:

- **Create**: Add new documents with `add_texts()`
- **Read**: Retrieve documents with `similarity_search()`
- **Update**: Update existing documents with `update_texts()`
- **Delete**: Remove documents with `delete()`
- **Upsert**: Add or update documents with `upsert_texts()`

## Core Methods

### Update Documents

```python
vectorstore.update_texts(
    texts=["Updated document content"],
    filter={"metadata_field": "value_to_match"},
    metadatas=[{"source": "updated_source.txt"}],
    update_embeddings=True
)
```

This method updates documents that match the specified filter criteria:

- `texts`: New text content for the documents
- `filter`: Filter criteria to identify which documents to update
- `metadatas`: New metadata for the documents (optional)
- `update_embeddings`: Whether to regenerate embeddings (default: True)

The update operation supports both:
- Using internal HANA embedding generation for improved performance
- Using external embedding models for greater flexibility

### Upsert Documents

```python
vectorstore.upsert_texts(
    texts=["Document content"],
    metadatas=[{"source": "source.txt"}],
    filter={"metadata_field": "value_to_match"}
)
```

This method provides an "upsert" operation (update if exists, insert if not):

- If documents matching the filter exist, they are updated
- If no matching documents exist, a new document is added
- If no filter is provided, a new document is always added

### Delete Documents

```python
vectorstore.delete(
    filter={"metadata_field": "value_to_match"}
)
```

This method deletes documents that match the specified filter criteria.

## API Endpoints

The API provides RESTful endpoints for each operation:

### Update Documents

```http
PUT /api/documents/update
Content-Type: application/json

{
  "text": "Updated document content",
  "metadata": {"source": "updated_source.txt"},
  "filter": {"category": "article"},
  "update_embeddings": true
}
```

### Upsert Documents

```http
POST /api/documents/upsert
Content-Type: application/json

{
  "text": "Document content",
  "metadata": {"source": "source.txt"},
  "filter": {"category": "article"}
}
```

### Delete Documents

```http
DELETE /api/documents/delete
Content-Type: application/json

{
  "filter": {"category": "article"}
}
```

## Implementation Details

### Update Process

The update operation follows this process:

1. Documents matching the filter criteria are identified
2. For each matching document:
   - The text content is replaced with the new text
   - The metadata is updated with the new metadata (if provided)
   - The embedding is regenerated (if requested)

### Embedding Generation

Depending on configuration, embeddings can be generated using:

- **HANA Internal Embedding**: Using SAP HANA's `VECTOR_EMBEDDING` function
  - More efficient as embedding generation happens in-database
  - No need to transfer data between application and database
  - Uses the same model for all operations

- **External Embedding**: Using Python-based embedding models
  - More flexible, supporting any embedding model
  - Generated in Python and then sent to the database
  - Useful for specialized embedding models

### Performance Considerations

- Updating embeddings adds computational overhead but ensures search accuracy
- For metadata-only updates, set `update_embeddings=False` for better performance
- Batch updates are more efficient than individual updates
- Internal HANA embeddings are more efficient for large-scale updates

## Examples

### Example 1: Update Document Content and Regenerate Embeddings

```python
from langchain_hana import HanaDB
from langchain_core.embeddings import HuggingFaceEmbeddings

# Initialize vectorstore
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = HanaDB(
    connection=connection,
    embedding=embedding_model,
    table_name="EMBEDDINGS"
)

# Update documents with category "finance"
vectorstore.update_texts(
    texts=["Updated financial report for Q2 2023"],
    filter={"category": "finance", "quarter": "Q2"},
    metadatas=[{"updated_date": "2023-06-30", "version": "2.0"}]
)
```

### Example 2: Update Metadata Without Regenerating Embeddings

```python
# Update only metadata without changing embeddings
vectorstore.update_texts(
    texts=["Financial report for Q2 2023"],  # Keep original text
    filter={"document_id": "fin-2023-q2"},
    metadatas=[{"status": "approved", "approved_by": "finance_committee"}],
    update_embeddings=False  # Don't regenerate embeddings
)
```

### Example 3: Upsert Documents

```python
# Add if not exists, update if exists
vectorstore.upsert_texts(
    texts=["Updated compliance policy for 2023"],
    metadatas=[{"department": "legal", "effective_date": "2023-01-01"}],
    filter={"policy_id": "compliance-2023"}
)
```

## Error Handling

The update operations include robust error handling:

- Database connection issues are detected and reported
- Filter validation ensures proper filter criteria
- Transaction handling ensures data consistency
- Detailed error messages with operation context
- Suggestions for resolving common errors

## Conclusion

The update operations complete the CRUD functionality in the SAP HANA Cloud LangChain integration, providing a comprehensive solution for managing documents in the vector store. These operations are essential for maintaining up-to-date information in long-lived applications where document content and metadata need to evolve over time.