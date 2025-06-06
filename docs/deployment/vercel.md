# Vercel Frontend Deployment Guide

This document provides detailed instructions for deploying the frontend of the SAP HANA Cloud LangChain Integration to Vercel.

## Prerequisites

- [Node.js](https://nodejs.org/) (version 16 or later)
- [Vercel CLI](https://vercel.com/docs/cli) (optional for local deployment)
- A Vercel account
- Backend already deployed and accessible

## Manual Deployment

### 1. Prepare the Project

Ensure your frontend code is ready for deployment:

```bash
cd frontend
npm install
npm run build
```

### 2. Configure Vercel JSON

The `vercel.frontend.json` file contains the configuration for the Vercel deployment. Update the `env` section with your backend URL:

```json
"env": {
  "REACT_APP_API_URL": "https://your-backend-url.example.com",
  "REACT_APP_API_VERSION": "1.0.0",
  "REACT_APP_ENVIRONMENT": "production"
}
```

### 3. Deploy to Vercel

#### Using Vercel CLI

If you have the Vercel CLI installed:

```bash
cd frontend
vercel --prod
```

When prompted, provide the following:
- Project name (or accept the suggested one)
- Directory to deploy (default: current directory)
- Link to existing project: Yes (if already created)
- Environment variables (will be prompted to enter)

#### Using Vercel Dashboard

1. Go to [Vercel Dashboard](https://vercel.com/dashboard)
2. Click "New Project"
3. Import your GitHub repository
4. Configure the project:
   - Root directory: `frontend`
   - Build command: `npm run build`
   - Output directory: `build`
5. Add environment variables:
   - `REACT_APP_API_URL`: Your backend URL
   - `REACT_APP_API_VERSION`: API version (e.g., "1.0.0")
   - `REACT_APP_ENVIRONMENT`: "production"
6. Click "Deploy"

### 4. Verify Deployment

After deployment, Vercel will provide a URL for your frontend. Verify it's working by:
- Opening the URL in your browser
- Testing the connection to the backend
- Checking that features like embeddings and search work correctly

## GitHub Integration for Automatic Deployments

For a more streamlined workflow, you can connect your GitHub repository to Vercel:

1. In the Vercel dashboard, go to your project settings
2. Under "Git Integration", click "Connect" if not already connected
3. Select the repository containing your frontend code
4. Configure the project as described in the manual deployment section
5. Set up production and preview branches

With GitHub integration, every push to your repository will trigger a new deployment:
- Push to main branch: Production deployment
- Push to other branches: Preview deployment

## Environment-Specific Configuration

The frontend supports different configurations for development and production:

- `.env`: Development configuration (local development)
- `.env.production`: Production configuration (Vercel deployment)

You can update these files to match your specific requirements.

## Custom Domains

To use a custom domain for your frontend:

1. In the Vercel dashboard, go to your project
2. Click on "Settings" > "Domains"
3. Add your custom domain and follow the DNS configuration instructions

## Troubleshooting

### CORS Issues

If you encounter CORS issues when connecting to the backend:

1. Verify the backend CORS configuration includes your frontend URL
2. Check that the backend `connection.json` has the correct frontend URL
3. Make sure the frontend's `.env.production` file has the correct backend URL

### Build Failures

If the Vercel build fails:

1. Check the build logs in the Vercel dashboard
2. Ensure all dependencies are properly defined in `package.json`
3. Verify that the build command and output directory are correctly configured

### Connection Issues

If the frontend can't connect to the backend:

1. Check that the backend is running and accessible
2. Verify the `REACT_APP_API_URL` environment variable is correctly set
3. Test the backend URL directly to ensure it's responding

### Slow Performance

If the frontend performance is slow:

1. Check the backend's GPU acceleration is properly configured
2. Verify the connection latency between frontend and backend
3. Consider implementing caching for frequently accessed data