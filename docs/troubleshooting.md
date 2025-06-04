# Troubleshooting Guide for SAP HANA Cloud LangChain T4 GPU Integration

This guide helps you troubleshoot common issues with the SAP HANA Cloud LangChain integration, particularly focusing on the Vercel frontend and T4 GPU backend integration.

## 500 Server Error

If you encounter a 500 server error when making API requests:

### Problem: Backend Connection Issues

**Error Message**: "Backend service unavailable" or "Unable to connect to the T4 GPU backend"

**Possible Causes**:
1. The T4 GPU backend URL is incorrect or the service is down
2. Network connectivity issues between Vercel and the T4 GPU backend
3. The T4 GPU backend is still initializing (especially after a cold start)

**Solutions**:
1. Verify the backend URL in your `vercel.json` file:
   ```json
   "env": {
     "T4_GPU_BACKEND_URL": "https://your-backend-url.com"
   }
   ```

2. Use the update script to set the correct backend URL:
   ```bash
   ./scripts/update_backend_url.sh https://your-correct-backend-url.com
   ```

3. Check if the backend is running by accessing its health endpoint directly:
   ```bash
   curl https://your-backend-url.com/api/health
   ```

4. If the backend is on Brev Cloud, check if the notebook/instance is running

5. Redeploy to Vercel after updating the backend URL:
   ```bash
   ./scripts/deploy_to_vercel.sh
   ```

### Problem: Pydantic Version Compatibility

**Error Message**: "AttributeError: 'EmbeddingRequest' object has no attribute 'model_dump'"

**Possible Causes**:
1. Using Pydantic v2 methods with Pydantic v1 installed
2. Version mismatch between local development and Vercel

**Solutions**:
1. The code has been updated to support both Pydantic v1 and v2. Make sure you're using the latest code.
2. Specify the Pydantic version in your requirements.txt:
   ```
   pydantic>=1.10.0,<3.0.0
   ```

### Problem: Authentication Issues

**Error Message**: "Authentication required" (401 error)

**Possible Causes**:
1. Missing or invalid JWT token
2. Authentication enabled in production but not logging in first

**Solutions**:
1. Make sure to login using the credentials provided:
   - Username: admin, demo, or user
   - Password: sap-hana-t4-admin, sap-hana-t4-demo, or sap-hana-t4-user
2. Check that the JWT token is being properly stored and sent in requests
3. For testing, you can set ENVIRONMENT to "development" to disable authentication

## Testing the API Locally

To test the API locally before deploying to Vercel:

1. Run the local test script:
   ```bash
   ./scripts/test_api_locally.sh
   ```

2. The API will be available at http://localhost:8000

3. Test the root endpoint:
   ```bash
   curl http://localhost:8000/
   ```

4. Test the authentication:
   ```bash
   curl -X POST http://localhost:8000/api/auth/token \
     -H "Content-Type: application/json" \
     -d '{"username":"admin","password":"sap-hana-t4-admin"}'
   ```

## Vercel Deployment Issues

### Problem: Routing Issues

**Error Message**: "404 Not Found" when accessing API endpoints

**Possible Causes**:
1. Incorrect Vercel routing configuration
2. File path issues in the Vercel builds

**Solutions**:
1. Check your `vercel.json` routing configuration:
   ```json
   "routes": [
     {
       "src": "/api",
       "dest": "api/vercel_integration.py"
     },
     {
       "src": "/api/(.*)",
       "dest": "api/vercel_integration.py"
     },
     {
       "src": "/(.*)",
       "dest": "/frontend/index.html"
     }
   ]
   ```

2. Make sure the file paths in your `vercel.json` builds section match your actual file structure

3. Try manually setting the project directory in your Vercel deployment command:
   ```bash
   vercel --cwd /path/to/project
   ```

## Debugging Connection to T4 GPU Backend

To debug connection issues between Vercel and the T4 GPU backend:

1. Check the health endpoint on your deployed Vercel app:
   ```
   https://your-vercel-app.vercel.app/api/health
   ```

2. Look for the backend status in the response

3. If the status is "unreachable", check:
   - Is the T4 GPU backend URL correct?
   - Is the T4 GPU backend running?
   - Are there any network restrictions between Vercel and Brev Cloud?

4. Try changing the timeout in `vercel_integration.py` to a longer value for the health check:
   ```python
   backend_response = requests.get(
       f"{T4_GPU_BACKEND_URL}/api/health",
       timeout=10  # Increase from 5 to 10 seconds
   )
   ```

## Additional Resources

- Check the [GPU Acceleration Guide](gpu_acceleration.md) for more information on the T4 GPU backend
- For authentication issues, refer to the [Security Guide](security_guide.md)
- For deployment options, see the [Deployment Guide](deployment_guide.md)

If you're still experiencing issues, please open an issue on GitHub with detailed information about the problem.