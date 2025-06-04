# Security Guide for LangChain SAP HANA Cloud Integration

This document provides comprehensive guidance for securing your SAP HANA Cloud integration with LangChain in production environments.

## Database Connection Security

### Secure Connection Parameters

Always use these secure connection parameters when connecting to SAP HANA Cloud in production:

```python
from hdbcli import dbapi

# Secure connection configuration
connection = dbapi.connect(
    address="<hostname>",
    port=3<NN>MM,
    user="<username>",
    password="<password>",
    encrypt=True,  # Enable encryption for all communication
    sslValidateCertificate=True,  # Validate server certificates
    sslTrustStore="/path/to/truststore.pem",  # Specify trusted certificates
    connectTimeout=15,  # Set connection timeout (seconds)
    reconnect=True,  # Enable automatic reconnection
)
```

### Credential Management

**NEVER hardcode credentials in your application code**. Instead:

1. **Environment Variables**: Store credentials in environment variables, but ensure they're properly secured:
   ```python
   import os
   from hdbcli import dbapi
   
   connection = dbapi.connect(
       address=os.environ.get("HANA_DB_ADDRESS"),
       port=os.environ.get("HANA_DB_PORT"),
       user=os.environ.get("HANA_DB_USER"),
       password=os.environ.get("HANA_DB_PASSWORD"),
       encrypt=True,
       sslValidateCertificate=True
   )
   ```

2. **Secret Management Systems**: Use cloud provider secret management:
   - AWS Secrets Manager
   - Google Secret Manager
   - Azure Key Vault
   - HashiCorp Vault

   Example with AWS Secrets Manager:
   ```python
   import boto3
   import json
   from hdbcli import dbapi
   
   secrets_client = boto3.client('secretsmanager')
   secret_response = secrets_client.get_secret_value(SecretId='hana-credentials')
   secret = json.loads(secret_response['SecretString'])
   
   connection = dbapi.connect(
       address=secret['address'],
       port=secret['port'],
       user=secret['username'],
       password=secret['password'],
       encrypt=True,
       sslValidateCertificate=True
   )
   ```

3. **Certificate-Based Authentication**: For highest security, use certificate-based authentication:
   ```python
   connection = dbapi.connect(
       address="<hostname>",
       port=3<NN>MM,
       user="<username>",
       sslKeyStore="/path/to/client.pem",
       sslTrustStore="/path/to/truststore.pem",
       encrypt=True,
       sslValidateCertificate=True
   )
   ```

## Principle of Least Privilege

Create dedicated database users with minimal permissions required:

1. **Read-Only Access**: For retrieval operations only:
   ```sql
   CREATE USER langchain_retrieval_user PASSWORD "StrongPassword123!";
   GRANT SELECT ON SCHEMA your_schema TO langchain_retrieval_user;
   ```

2. **Read-Write Access**: For full vector store operations:
   ```sql
   CREATE USER langchain_vector_user PASSWORD "StrongPassword123!";
   GRANT SELECT, INSERT, UPDATE, DELETE ON SCHEMA your_schema TO langchain_vector_user;
   ```

3. **Schema-Only Access**: Restrict access to specific schemas:
   ```sql
   GRANT SELECT ON SCHEMA your_vector_schema TO langchain_user;
   ```

## Network Security

1. **IP Whitelisting**: Restrict database access to specific IP addresses
2. **VPC/Private Networks**: Run your application and database in a private network
3. **TLS Encryption**: Always use TLS 1.2+ for all connections

## Connection Pooling

For production applications, implement connection pooling:

```python
import threading
from hdbcli import dbapi
from queue import Queue

class ConnectionPool:
    def __init__(self, max_connections=10, **connection_params):
        self.connection_params = connection_params
        self.max_connections = max_connections
        self.pool = Queue(maxsize=max_connections)
        self.size = 0
        self.lock = threading.Lock()
        
    def get_connection(self):
        if not self.pool.empty():
            return self.pool.get()
        
        with self.lock:
            if self.size < self.max_connections:
                connection = dbapi.connect(**self.connection_params)
                self.size += 1
                return connection
            else:
                return self.pool.get(block=True)
    
    def return_connection(self, connection):
        self.pool.put(connection)
```

## Data Security

### Sensitive Data

Be cautious about what data is stored in vector embeddings:

1. **Data Classification**: Classify data before storing in vectors
2. **PII/PHI**: Don't embed personally identifiable or protected health information
3. **Data Anonymization**: Anonymize sensitive data before embedding
4. **Data Tokenization**: Replace sensitive values with tokens

### Logging and Auditing

Implement comprehensive logging:

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("hana_vector_operations.log"),
        logging.StreamHandler()
    ]
)

# Log database operations
logger = logging.getLogger("langchain_hana")

try:
    # Execute database operation
    result = vectorstore.similarity_search("query", filter={"security_level": "public"})
    logger.info(f"Search completed successfully with {len(result)} results")
except Exception as e:
    logger.error(f"Search operation failed: {str(e)}")
```

## Container Security

When deploying in containers:

1. **Non-Root User**: Run the container as a non-root user
2. **Read-Only Filesystem**: Mount the filesystem as read-only where possible
3. **Resource Limits**: Set memory and CPU limits
4. **No Privileged Mode**: Avoid running containers in privileged mode

Example Docker configuration:

```dockerfile
FROM python:3.9-slim

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
WORKDIR /app
COPY . .

# Set ownership
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Run the application
CMD ["python", "app.py"]
```

## API Security

If exposing functionality via API:

1. **Authentication**: Implement OAuth2, API keys, or JWT authentication
2. **Rate Limiting**: Protect against DoS attacks
3. **Input Validation**: Validate all inputs before processing
4. **HTTPS Only**: Enforce HTTPS for all communications

## Regular Security Practices

1. **Dependency Scanning**: Regularly scan for vulnerabilities in dependencies
2. **Credential Rotation**: Rotate database credentials regularly
3. **Security Updates**: Keep all components updated with security patches
4. **Penetration Testing**: Conduct regular security testing

## Monitoring and Alerting

Implement monitoring to detect security incidents:

1. **Failed Login Attempts**: Monitor and alert on multiple failed logins
2. **Unusual Query Patterns**: Detect abnormal database access patterns
3. **Resource Utilization**: Monitor for unexpected resource usage

## Conclusion

Security is a continuous process. Regularly review and update your security practices as new threats emerge and as your application evolves.

For additional guidance, refer to:
- [SAP HANA Cloud Security Guide](https://help.sap.com/docs/hana-cloud/sap-hana-cloud-administration-guide/security)
- [LangChain Security Documentation](https://python.langchain.com/docs/security)