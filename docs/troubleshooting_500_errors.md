# Troubleshooting 500 Server Errors in SAP HANA Cloud LangChain T4 GPU Integration

This guide provides comprehensive information on diagnosing and fixing 500 server errors in the Vercel deployment of the SAP HANA Cloud LangChain T4 GPU Integration.

## New Diagnostic Tools

We've implemented several new tools to help diagnose 500 server errors:

### 1. Enhanced Debug Proxy

The enhanced debug proxy provides detailed error reporting, request tracing, and connection diagnostics to help pinpoint the exact cause of 500 errors.

- Endpoint: `/enhanced-debug`
- Key features:
  - Detailed request/response logging
  - Connection diagnostics
  - Memory usage tracking
  - Configurable timeouts
  - Exception tracing

### 2. Direct Proxy with Connection Validation

The direct proxy has been enhanced with detailed connection validation to diagnose backend connectivity issues.

- Endpoint: `/debug-proxy`
- Key diagnostics:
  - DNS resolution testing
  - TCP connection testing
  - SSL/TLS handshake validation
  - HTTP request testing
  - Comprehensive error reporting

### 3. API Diagnostics Script

A dedicated Python script for testing the API and diagnosing specific endpoint failures.

- Usage: `./scripts/api_diagnostics.py --url <your-vercel-url> --test-all --use-direct-proxy`
- Features:
  - Tests all API endpoints
  - Connection diagnostics
  - Authentication testing
  - Embeddings generation testing
  - Metrics testing
  - Detailed error reporting

### 4. Configurable Timeouts

Timeout values are now configurable through environment variables to accommodate slower connections and operations.

- Default timeout: 30 seconds
- Health check timeout: 10 seconds
- Embeddings timeout: 60 seconds
- Search timeout: 45 seconds
- Authentication timeout: 15 seconds
- Connection test timeout: 5 seconds

## Common 500 Error Causes and Solutions

### 1. Backend Connection Issues

**Symptoms**:
- 503 Service Unavailable errors
- "Unable to connect to the T4 GPU backend"
- Backend service timeouts

**Diagnosis**:
- Check `/debug-proxy/connection-diagnostics` for detailed connection information
- Verify DNS resolution, TCP connection, and SSL/TLS handshake

**Solutions**:
- Verify the T4_GPU_BACKEND_URL is correct in vercel.json
- Ensure the backend is running and accessible
- Check for network restrictions between Vercel and the backend
- Try increasing timeout values for slow connections

### 2. Request Timeout Issues

**Symptoms**:
- 504 Gateway Timeout errors
- "Backend service timed out"
- Long-running operations (like embeddings generation) failing

**Diagnosis**:
- Check the timeout values in vercel.json
- Use `/enhanced-debug` to see which operation is timing out
- Test with different timeout values

**Solutions**:
- Increase the relevant timeout value in vercel.json
- For embeddings, use a smaller batch size or fewer texts
- For large operations, consider splitting into smaller chunks

### 3. Authentication Issues

**Symptoms**:
- 401 Unauthorized errors
- "Authentication required" errors
- Login failures

**Diagnosis**:
- Check if the JWT token is being properly sent in requests
- Verify the JWT_SECRET value in vercel.json
- Test authentication using the API diagnostics script

**Solutions**:
- Ensure JWT_SECRET is consistent between frontend and backend
- For testing, set ENVIRONMENT to "development" to disable authentication
- Verify correct credentials are being used (admin/sap-hana-t4-admin, demo/sap-hana-t4-demo, user/sap-hana-t4-user)

### 4. Pydantic Version Compatibility

**Symptoms**:
- AttributeError: 'EmbeddingRequest' object has no attribute 'model_dump'
- Version mismatch errors

**Diagnosis**:
- Check the logs for Pydantic-related errors
- Verify the Pydantic version in requirements.txt

**Solutions**:
- The code has been updated to support both Pydantic v1 and v2
- Ensure you're using the latest code with compatibility fixes

### 5. Backend Processing Errors

**Symptoms**:
- 500 Internal Server Error from the backend
- Error in embedding generation or vector search
- GPU-related errors

**Diagnosis**:
- Use the direct proxy to get more detailed error messages
- Check the backend logs for GPU-related errors
- Test with smaller input data

**Solutions**:
- Verify GPU is available and working on the backend
- Try disabling TensorRT by setting USE_TENSORRT to "false"
- For embedding generation, use a smaller batch size or fewer texts

## Deployment and Testing

### Deployment Script

Use the provided deployment script to deploy and test the changes:

```bash
./scripts/deploy_and_test.sh
```

This script:
- Updates the backend URL if needed
- Configures timeout and environment settings
- Deploys to Vercel
- Runs diagnostics tests
- Provides debugging URLs and commands

### Manual Testing

To manually test the API:

1. Test the health endpoint:
   ```
   https://your-vercel-url.vercel.app/api/health
   ```

2. Test the connection diagnostics:
   ```
   https://your-vercel-url.vercel.app/debug-proxy/connection-diagnostics
   ```

3. Test with the API diagnostics script:
   ```bash
   ./scripts/api_diagnostics.py --url https://your-vercel-url.vercel.app --test-all --use-direct-proxy
   ```

## Environment Variables

The following environment variables can be configured in vercel.json:

| Variable | Description | Default Value |
|----------|-------------|---------------|
| T4_GPU_BACKEND_URL | URL of the T4 GPU backend | https://jupyter0-513syzm60.brevlab.com |
| ENVIRONMENT | Environment (development, production) | development |
| DEFAULT_TIMEOUT | Default request timeout in seconds | 30 |
| HEALTH_CHECK_TIMEOUT | Health check timeout in seconds | 10 |
| EMBEDDING_TIMEOUT | Embeddings endpoint timeout in seconds | 60 |
| SEARCH_TIMEOUT | Search endpoint timeout in seconds | 45 |
| AUTH_TIMEOUT | Authentication timeout in seconds | 15 |
| CONNECTION_TEST_TIMEOUT | Connection test timeout in seconds | 5 |
| LOG_LEVEL | Logging level (DEBUG, INFO, WARNING, ERROR) | DEBUG |
| ENABLE_MEMORY_TRACKING | Enable memory usage tracking | true |
| ENABLE_DETAILED_LOGGING | Enable detailed request/response logging | true |

## Troubleshooting Workflow

When encountering 500 server errors, follow this workflow:

1. **Basic connectivity check**:
   - Check the health endpoint: `/api/health`
   - Check the proxy health: `/debug-proxy/proxy-health`

2. **Detailed diagnostics**:
   - Run connection diagnostics: `/debug-proxy/connection-diagnostics`
   - Use the enhanced debug proxy: `/enhanced-debug`

3. **Run the API diagnostics script**:
   ```bash
   ./scripts/api_diagnostics.py --url https://your-vercel-url.vercel.app --test-all --use-direct-proxy
   ```

4. **Check Vercel logs**:
   ```bash
   vercel logs https://your-vercel-url.vercel.app
   ```

5. **Adjust environment variables** if needed:
   - Update timeout values
   - Set ENVIRONMENT to "development" for more detailed errors
   - Set LOG_LEVEL to "DEBUG" for verbose logging

6. **Re-deploy** with updated settings:
   ```bash
   ./scripts/deploy_and_test.sh
   ```

By following this guide, you should be able to diagnose and fix most 500 server errors in the SAP HANA Cloud LangChain T4 GPU Integration.