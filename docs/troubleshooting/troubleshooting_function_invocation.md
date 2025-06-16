# Troubleshooting Vercel FUNCTION_INVOCATION_FAILED Errors

This guide specifically addresses the "FUNCTION_INVOCATION_FAILED" error that can occur when deploying Python serverless functions to Vercel.

## Common Error Patterns

The typical error message looks like this:

```
A server error has occurred
FUNCTION_INVOCATION_FAILED
sin1::hf82s-1749041842066-42f2b7318797
```

This error indicates that Vercel attempted to execute your function but failed during the invocation process.

## Root Causes and Solutions

### 1. Missing Dependencies

**Symptoms:** The function works locally but fails on Vercel.

**Solution:**
- Ensure all required packages are listed in `requirements.txt` with appropriate version constraints
- Check for non-Python dependencies that might be required
- For the SAP HANA LangChain integration, make sure these are included:
  ```
  fastapi>=0.100.0,<1.0.0
  uvicorn>=0.22.0,<1.0.0
  requests>=2.31.0,<3.0.0
  pyjwt>=2.8.0,<3.0.0
  pydantic>=1.10.0,<3.0.0
  python-multipart>=0.0.6,<1.0.0
  python-dotenv>=1.0.0,<2.0.0
  psutil>=5.9.0,<6.0.0
  dnspython>=2.3.0,<3.0.0
  ```

### 2. Python Version Incompatibility

**Symptoms:** Code uses features not available in Vercel's Python runtime.

**Solution:**
- Vercel serverless functions use Python 3.9 by default
- Avoid using features introduced in Python 3.10+ (such as match/case statements)
- Test with the minimal_test.py endpoint to verify Python compatibility

### 3. Function Size Limits

**Symptoms:** Only complex functions fail, simpler ones work.

**Solution:**
- Vercel has a 50MB limit for functions
- Check if your function includes large dependencies or assets
- Use the minimal_test.py endpoint to verify basic functionality
- Consider using external storage for large assets

### 4. Syntax or Runtime Errors

**Symptoms:** The function runs fine locally but fails on Vercel.

**Solution:**
- Check for platform-specific code that might fail on Vercel's Linux environment
- Look for differences in file paths between local and Vercel environments
- Ensure environment variables are properly configured
- Use try/except blocks with detailed error logging

### 5. Memory or Timeout Limits

**Symptoms:** Functions fail only with large requests or after running for some time.

**Solution:**
- Vercel Hobby tier has a maximum of 1024MB RAM per function
- Functions have a maximum execution time of 10 seconds (Hobby tier)
- Optimize memory usage for large operations
- Break down long-running tasks into smaller functions

## Diagnostic Tools

### Minimal Test API

We've created a minimal test API with basic endpoints to help diagnose function invocation issues:

- `/minimal-test` - Basic endpoint to verify function execution
- `/minimal-test/environment` - Shows Python version and environment variables
- `/minimal-test/sys-path` - Shows Python sys.path to help diagnose import issues

Use the provided script to test these endpoints:
```bash
./scripts/test_minimal_api.sh -u <your-vercel-url>
```

### Vercel Logs

Check Vercel logs for detailed error information:
```bash
vercel logs <deployment-url>
```

Use the `--function=api/minimal_test.py` flag to view logs for a specific function:
```bash
vercel logs <deployment-url> --function=api/minimal_test.py
```

### Local Testing

Test the function locally before deploying:
```bash
cd api
python -m uvicorn minimal_test:app --reload
```

## Protection Bypass for Testing

When troubleshooting, you can bypass Vercel's function protection using a protection bypass header:

```bash
curl -H "x-vercel-protection-bypass: jyuthfgjugjuiytioytkkilytkgjhkui" https://your-deployment-url.vercel.app/minimal-test
```

## Step-by-Step Troubleshooting Process

1. **Deploy the minimal test API first**
   - Verify that basic function invocation works
   - If this fails, focus on fundamental configuration issues

2. **Check dependencies**
   - Ensure all required packages are in requirements.txt
   - Verify versions are compatible with Python 3.9

3. **Check environment variables**
   - Verify all required environment variables are set in vercel.json
   - Check for sensitive values that might be missing

4. **Examine Vercel logs**
   - Look for specific error messages
   - Check for syntax errors, import errors, or runtime exceptions

5. **Incrementally add complexity**
   - Once the minimal API works, add more complex functionality
   - Test each addition to isolate the problematic code

## Additional Resources

- [Vercel Serverless Functions Documentation](https://vercel.com/docs/functions/serverless-functions)
- [Vercel Python Runtime Documentation](https://vercel.com/docs/functions/runtimes/python)
- [FastAPI on Vercel Guide](https://vercel.com/guides/deploying-fastapi-with-vercel)