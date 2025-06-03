# Vercel Deployment Guide

This guide provides instructions for deploying the SAP HANA Cloud LangChain Integration API to Vercel's serverless platform.

## Overview

Vercel provides a serverless deployment option for the API, which is ideal for:
- Development and testing environments
- Lower-traffic production use cases
- Scenarios where GPU acceleration is not required
- Cost-effective deployments

## Prerequisites

- [Vercel account](https://vercel.com/signup)
- [GitHub account](https://github.com/signup) with repository access
- SAP HANA Cloud instance credentials

## Limitations

Before deploying to Vercel, be aware of these limitations:

1. **No GPU Acceleration**: Vercel doesn't support GPU hardware, so TensorRT and CUDA features are disabled.
2. **Function Timeout**: Functions have a 60-second execution limit (configured as maxDuration).
3. **Memory Constraints**: Limited to 1024MB RAM per function.
4. **Cold Starts**: Serverless functions experience cold starts after periods of inactivity.
5. **Statelessness**: No persistent storage between function invocations.

## Deployment Steps

### 1. Fork or Clone the Repository

```bash
git clone https://github.com/plturrell/langchain-integration-for-sap-hana-cloud.git
cd langchain-integration-for-sap-hana-cloud
```

### 2. Connect to Vercel

#### Option A: Using Vercel CLI

```bash
# Install Vercel CLI
npm install -g vercel

# Login to Vercel
vercel login

# Deploy to Vercel
vercel
```

#### Option B: Using Vercel Web Interface

1. Go to [Vercel Dashboard](https://vercel.com/dashboard)
2. Click "Add New" > "Project"
3. Import your GitHub repository
4. Select the "langchain-integration-for-sap-hana-cloud" repository
5. Configure the project (see next section)

### 3. Configure Environment Variables

Add the following environment variables in the Vercel dashboard:

| Variable | Description | Example |
|----------|-------------|---------|
| `HANA_HOST` | SAP HANA Cloud host | `myhana.hanacloud.ondemand.com` |
| `HANA_PORT` | SAP HANA Cloud port | `443` |
| `HANA_USER` | SAP HANA Cloud username | `SYSTEM` |
| `HANA_PASSWORD` | SAP HANA Cloud password | `********` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `EMBEDDING_MODEL` | Model for embeddings | `all-MiniLM-L6-v2` |

> **Security Note**: All sensitive variables like `HANA_PASSWORD` should be marked as "Environment Variables" (not "System Environment Variables") and set to "Encrypted".

### 4. Configure Project Settings

In the Vercel project settings:

1. **Framework Preset**: Select "Other"
2. **Root Directory**: Leave as default (project root)
3. **Build Command**: `pip install -r api/requirements-vercel.txt`
4. **Output Directory**: Leave blank
5. **Install Command**: `pip install -r api/requirements-vercel.txt`
6. **Development Command**: Leave blank

### 5. Deploy

Click "Deploy" and wait for the build to complete. Vercel will provide a deployment URL once finished.

## Testing the Deployment

After deployment, you can test the API using the provided URL:

```bash
# Check health endpoint
curl https://your-deployment-url.vercel.app/__health

# Access API documentation
# Visit: https://your-deployment-url.vercel.app/docs
```

## Optimizing for Serverless

The project includes several optimizations for Vercel's serverless environment:

1. **Custom Handler**: `vercel_handler.py` adapts the FastAPI app for serverless execution
2. **Reduced Dependencies**: `requirements-vercel.txt` includes only necessary packages
3. **Disabled GPU Features**: GPU acceleration is automatically disabled
4. **Response Time Limits**: Functions are configured to respond within the 60-second limit
5. **Error Handling**: Enhanced error handling for serverless environment

## Monitoring and Logs

Monitor your deployment through the Vercel dashboard:

1. **Function Logs**: View real-time logs for debugging
2. **Usage Statistics**: Monitor API usage and performance
3. **Error Tracking**: View and troubleshoot errors

## Custom Domains

To use a custom domain:

1. Go to your project in the Vercel dashboard
2. Navigate to "Settings" > "Domains"
3. Add your domain and follow the verification steps

## CI/CD Integration

For automated deployments:

1. Connect your GitHub repository to Vercel
2. Vercel automatically deploys when changes are pushed to the main branch
3. Create Preview Deployments for pull requests

## Troubleshooting

### Common Issues

1. **Deployment Failures**:
   - Check build logs for errors
   - Verify requirements file is correct
   - Ensure environment variables are set properly

2. **Function Timeouts**:
   - Optimize queries for faster execution
   - Split complex operations into multiple API calls
   - Consider using smaller embedding models

3. **Database Connection Issues**:
   - Verify HANA credentials are correct
   - Check if HANA instance allows connections from Vercel's IP ranges
   - Test connection with a simple query

## Next Steps

- Set up a custom domain
- Configure API authentication
- Set up monitoring and alerts
- Integrate with frontend applications

For high-throughput or GPU-accelerated deployments, consider using the [NGC Container Deployment](./nvidia_deployment.md) option instead.