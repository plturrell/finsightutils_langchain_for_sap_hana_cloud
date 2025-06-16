# Vercel Deployment Guide

This guide provides comprehensive instructions for deploying the SAP HANA Cloud LangChain Integration API to Vercel's serverless platform, including both backend API and frontend components.

## Overview

Vercel provides a serverless deployment option for the full application, which is ideal for:
- Development and testing environments
- Lower-traffic production use cases
- Scenarios where GPU acceleration is not required
- Cost-effective deployments
- Quick demos and proof-of-concept deployments

Our deployment includes:
- REST API with vector search capabilities
- React-based frontend for searching and visualization
- Context-aware error handling
- CORS support for cross-domain access
- Rate limiting for protection against abuse

## Prerequisites

- [Vercel account](https://vercel.com/signup)
- [GitHub account](https://github.com/signup) with repository access
- SAP HANA Cloud instance credentials
- Node.js 16+ (for local frontend development)
- Python 3.9+ (for local API development)

## Limitations

Before deploying to Vercel, be aware of these limitations:

1. **No GPU Acceleration**: Vercel doesn't support GPU hardware, so TensorRT and CUDA features are disabled.
2. **Function Timeout**: Functions have a 60-second execution limit (configured as maxDuration).
3. **Memory Constraints**: Limited to 1024MB RAM per function.
4. **Cold Starts**: Serverless functions experience cold starts after periods of inactivity.
5. **Statelessness**: No persistent storage between function invocations.
6. **Deployment Size**: There's a limit on total deployment size (50MB for the entire project).

## Deployment Steps

### 1. Fork or Clone the Repository

```bash
git clone https://github.com/plturrell/langchain-integration-for-sap-hana-cloud.git
cd langchain-integration-for-sap-hana-cloud
```

### 2. Configure for Vercel Deployment

#### Update Requirements for Vercel

Ensure that `api/requirements-vercel.txt` contains the minimal dependencies for Vercel:

```
fastapi==0.109.2
uvicorn==0.27.1
python-dotenv==1.0.1
pydantic==2.6.1
starlette==0.36.3
```

#### Verify Configuration Files

Ensure these key configuration files are properly set up:

1. **vercel.json** - Configures the build process and routing:
   ```json
   {
     "version": 2,
     "builds": [
       {
         "src": "api/index.py",
         "use": "@vercel/python",
         "config": {
           "runtime": "python3.9",
           "maxDuration": 60
         }
       },
       {
         "src": "frontend/package.json",
         "use": "@vercel/static-build",
         "config": {
           "distDir": "frontend/build",
           "buildCommand": "cd frontend && npm install && npm run build"
         }
       }
     ],
     "routes": [
       {
         "src": "/api/(.*)",
         "dest": "/api/index.py"
       },
       {
         "src": "/static/(.*)",
         "dest": "/frontend/build/static/$1"
       },
       {
         "src": "/(.*)",
         "dest": "/frontend/build/index.html"
       }
     ],
     "env": {
       "API_HOST": "0.0.0.0",
       "API_PORT": "8000",
       "LOG_LEVEL": "INFO"
     },
     "headers": [
       {
         "source": "/api/(.*)",
         "headers": [
           { "key": "Access-Control-Allow-Credentials", "value": "true" },
           { "key": "Access-Control-Allow-Origin", "value": "*" },
           { "key": "Access-Control-Allow-Methods", "value": "GET,OPTIONS,PATCH,DELETE,POST,PUT" },
           { "key": "Access-Control-Allow-Headers", "value": "X-CSRF-Token, X-Requested-With, Accept, Accept-Version, Content-Length, Content-MD5, Content-Type, Date, X-Api-Version" }
         ]
       }
     ]
   }
   ```

2. **api/index.py** - Entry point for the API:
   - Should use FastAPI with CORS middleware
   - Include mock implementations for environments without database access
   - Implement proper error handling for serverless context

3. **api/vercel_middleware.py** - Error handling middleware for Vercel:
   - Implements context-aware error responses
   - Handles rate limiting for protection
   - Provides detailed logging

### 3. Connect to Vercel

#### Option A: Using Vercel CLI (Recommended)

The CLI method gives more control and better visibility into the deployment process:

```bash
# Install Vercel CLI
npm install -g vercel

# Login to Vercel
vercel login

# Deploy to Vercel (from project root)
vercel
```

When prompted:
- Select "No" for default settings
- Configure your project settings as shown in step 4
- Set up environment variables when prompted

For production deployment:
```bash
vercel --prod
```

#### Option B: Using Vercel Web Interface

1. Go to [Vercel Dashboard](https://vercel.com/dashboard)
2. Click "Add New" > "Project"
3. Import your GitHub repository
4. Select the "langchain-integration-for-sap-hana-cloud" repository
5. Configure the project (see next section)

### 4. Configure Environment Variables

Add the following environment variables in the Vercel dashboard:

| Variable | Description | Example | Required |
|----------|-------------|---------|----------|
| `HANA_HOST` | SAP HANA Cloud host | `myhana.hanacloud.ondemand.com` | Yes |
| `HANA_PORT` | SAP HANA Cloud port | `443` | Yes |
| `HANA_USER` | SAP HANA Cloud username | `SYSTEM` | Yes |
| `HANA_PASSWORD` | SAP HANA Cloud password | `********` | Yes |
| `HANA_ENCRYPT` | Use SSL/TLS for database connection | `true` | No |
| `HANA_SSL_VALIDATE_CERT` | Validate SSL certificates | `true` | No |
| `LOG_LEVEL` | Logging level | `INFO` | No |
| `ENVIRONMENT` | Deployment environment | `production` | No |
| `USE_INTERNAL_EMBEDDINGS` | Use SAP HANA's internal embeddings | `false` | No |
| `EMBEDDING_MODEL` | Model for embeddings | `all-MiniLM-L6-v2` | No |
| `ENABLE_ERROR_CONTEXT` | Enable context-aware errors | `true` | No |
| `ENABLE_RATE_LIMITING` | Enable API rate limiting | `true` | No |

> **Security Note**: All sensitive variables like `HANA_PASSWORD` should be marked as "Environment Variables" (not "System Environment Variables") and set to "Encrypted".

### 5. Configure Project Settings

In the Vercel project settings:

1. **Framework Preset**: Select "Other"
2. **Root Directory**: Leave as default (project root)
3. **Build Command**: Override with this command if needed:
   ```
   pip install -r api/requirements-vercel.txt && cd frontend && npm install && npm run build
   ```
4. **Output Directory**: Leave blank (handled by vercel.json)
5. **Install Command**: `pip install -r api/requirements-vercel.txt`
6. **Development Command**: Leave blank

### 6. Deploy

Click "Deploy" and wait for the build to complete. Vercel will provide a deployment URL once finished.

## Testing the Full Deployment

After deployment, you can test both the API and frontend:

### Testing the API

```bash
# Check health endpoint
curl https://your-deployment-url.vercel.app/api/health

# Check feature information
curl https://your-deployment-url.vercel.app/api/feature/vector-similarity
```

### Testing the Frontend

Simply visit your Vercel deployment URL in a browser:
```
https://your-deployment-url.vercel.app/
```

The frontend should load and be able to communicate with the API endpoints.

## Advanced Features

### 1. Context-Aware Error Handling

The deployment includes sophisticated error handling that provides:

- Operation-specific error messages
- Suggested actions for resolution
- Intelligent detection of error types
- Client-friendly JSON responses

Example error response:
```json
{
  "error": "connection_failed",
  "message": "Connection to database failed: timeout connecting to server",
  "context": {
    "operation": "similarity_search",
    "request_id": "1653890122-12345",
    "suggestion": "Check your database connection settings and ensure the database is accessible from Vercel's IP range",
    "processing_time": 1.234
  }
}
```

### 2. Rate Limiting

The API includes rate limiting to protect against abuse:

- Default: 60 requests per minute for general endpoints
- Search: 20 requests per minute for vector search endpoints
- Sliding window implementation
- IP-based rate limiting
- Custom headers with limit information

### 3. CORS Support

Built-in CORS support allows:

- Cross-domain access to the API
- Support for credentialed requests
- Preflight request handling
- Custom header access

### 4. Mock Mode

When database connection isn't available (like in preview deployments), the API falls back to mock mode:

- Returns sample responses for search queries
- Simulates vector operations
- Provides realistic responses for testing
- Clearly indicates mock mode in responses

Enable mock mode by setting:
```
ENABLE_MOCK_MODE=true
```

## Monitoring and Logs

Monitor your deployment through the Vercel dashboard:

1. **Function Logs**: View real-time logs for debugging
   - Access via: Project > Settings > Functions > Logs
   - Filter by status code, path or date

2. **Usage Statistics**: Monitor API usage and performance
   - View request counts, response times, and error rates
   - Track cold starts and function durations

3. **Error Tracking**: View and troubleshoot errors
   - Automatic error grouping
   - Stack traces and request contexts
   - Error frequency and impact

## Custom Domains

To use a custom domain:

1. Go to your project in the Vercel dashboard
2. Navigate to "Settings" > "Domains"
3. Add your domain and follow the verification steps
4. Configure DNS settings as instructed
5. Vercel automatically provisions SSL certificates

## CI/CD Integration

For automated deployments:

1. Connect your GitHub repository to Vercel
2. Vercel automatically deploys when changes are pushed to the main branch
3. Create Preview Deployments for pull requests
4. Configure deployment protection rules if needed

### GitHub Actions Integration

Add this workflow file to `.github/workflows/vercel-deploy.yml` for more control:

```yaml
name: Deploy to Vercel

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Install Vercel CLI
        run: npm install -g vercel
        
      - name: Deploy to Vercel
        run: vercel --prod --token ${{ secrets.VERCEL_TOKEN }}
        env:
          VERCEL_ORG_ID: ${{ secrets.VERCEL_ORG_ID }}
          VERCEL_PROJECT_ID: ${{ secrets.VERCEL_PROJECT_ID }}
```

## Troubleshooting

### Common Issues

1. **Deployment Failures**:
   - Check build logs for errors in the Vercel dashboard
   - Verify requirements file has compatible versions
   - Ensure frontend build completes successfully
   - Check for size limits (50MB deployment size limit)

2. **Function Timeouts**:
   - Look for "Function Execution Timeout" errors in logs
   - Optimize vector search to use smaller fetch sizes
   - Use smaller embedding models
   - Consider implementing pagination for large result sets

3. **Database Connection Issues**:
   - Verify HANA credentials are correct
   - Check if HANA instance allows connections from Vercel's IP ranges
   - Test with mock mode enabled to verify API functionality
   - Check for SSL/TLS certificate validation issues

4. **CORS Problems**:
   - Verify CORS headers are being set correctly
   - Check browser console for CORS errors
   - Ensure frontend is using the correct API URL
   - Test with a CORS browser extension

5. **Frontend Build Failures**:
   - Check for Node.js version compatibility
   - Verify npm dependencies resolve correctly
   - Look for JavaScript build errors in logs
   - Consider using a separate frontend deployment if needed

### Diagnostic Commands

Use these commands to diagnose issues:

```bash
# Check API health
curl https://your-deployment-url.vercel.app/api/health

# Get API version and status
curl https://your-deployment-url.vercel.app/api/deployment/info

# Test vector search functionality (mock mode)
curl -X POST -H "Content-Type: application/json" \
  -d '{"query":"test query","k":2}' \
  https://your-deployment-url.vercel.app/api/search
```

## Production Optimization Tips

For production deployments:

1. **Domain Configuration**:
   - Use a custom domain with proper SSL
   - Configure caching headers for static assets

2. **Security**:
   - Add API authentication
   - Enable rate limiting
   - Consider using Vercel Teams for IP allowlisting

3. **Performance**:
   - Optimize frontend bundle size
   - Use smaller embedding models
   - Implement proper caching strategies

4. **Reliability**:
   - Set up monitoring and alerts
   - Configure error tracking
   - Implement retry mechanisms for API calls

## Alternative Deployment Options

For high-throughput or GPU-accelerated deployments, consider these alternatives:

1. **[NGC Container Deployment](./nvidia_deployment.md)** - For GPU acceleration
2. **[VM Deployment](./vm_setup_guide.md)** - For more control and persistent resources
3. **[Docker Deployment](../docker-compose.yml)** - For containerized deployment with custom hardware

## Further Resources

- [FastAPI on Vercel](https://vercel.com/guides/using-fastapi-on-vercel)
- [React on Vercel](https://vercel.com/guides/deploying-react-with-vercel)
- [Vercel Environment Variables](https://vercel.com/docs/concepts/projects/environment-variables)
- [Vercel Serverless Functions](https://vercel.com/docs/concepts/functions/serverless-functions)
- [SAP HANA Cloud Documentation](https://help.sap.com/docs/HANA_CLOUD)