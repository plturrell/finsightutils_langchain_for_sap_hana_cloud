# Quick Start Guide: Fixing Vercel 500 Errors

This guide provides step-by-step instructions to diagnose and fix the 500 server errors encountered in your Vercel deployment.

## Step 1: Test with Minimal API

The minimal test API helps isolate Vercel function invocation issues from application-specific issues.

```bash
# Run the minimal test deployment script
chmod +x scripts/deploy_minimal_test.sh
./scripts/deploy_minimal_test.sh

# Test the minimal API endpoints
chmod +x scripts/test_minimal_api.sh
./scripts/test_minimal_api.sh
```

If the minimal API works, proceed to Step 2. If it fails, check:
- Python version compatibility (Vercel uses Python 3.9)
- Syntax errors in minimal_test.py
- Missing dependencies in requirements.txt

## Step 2: Deploy Full Application

Once the minimal API works, deploy the full application:

```bash
# Run the no-token deployment script
chmod +x scripts/deploy_vercel_no_token.sh
./scripts/deploy_vercel_no_token.sh
```

This will deploy the full application to Vercel and save the deployment URL to `deployment_url.txt`.

## Step 3: Test All Endpoints

Test all endpoints using the protection bypass token:

```bash
# Set the deployment URL and bypass token
DEPLOYMENT_URL=$(cat deployment_url.txt)
BYPASS_TOKEN="jyuthfgjugjuiytioytkkilytkgjhkui"

# Test the endpoints
curl -H "x-vercel-protection-bypass: $BYPASS_TOKEN" "$DEPLOYMENT_URL/minimal-test"
curl -H "x-vercel-protection-bypass: $BYPASS_TOKEN" "$DEPLOYMENT_URL/enhanced-debug"
curl -H "x-vercel-protection-bypass: $BYPASS_TOKEN" "$DEPLOYMENT_URL/debug-proxy/proxy-health"
curl -H "x-vercel-protection-bypass: $BYPASS_TOKEN" "$DEPLOYMENT_URL/api/health"
```

## Step 4: Check Vercel Logs

If you encounter errors, check the Vercel logs:

```bash
# Install Vercel CLI if not already installed
npm install -g vercel

# Check logs for a specific function
vercel logs $DEPLOYMENT_URL --function=api/minimal_test.py
vercel logs $DEPLOYMENT_URL --function=api/enhanced_debug_proxy.py
vercel logs $DEPLOYMENT_URL --function=api/direct_proxy.py
vercel logs $DEPLOYMENT_URL --function=api/vercel_integration.py
```

## Step 5: Run Diagnostics

Use the enhanced diagnostic tools to identify issues:

```bash
# Run the API diagnostics script
chmod +x scripts/api_diagnostics.py
./scripts/api_diagnostics.py --url $DEPLOYMENT_URL --test-all --use-direct-proxy
```

## Common Issues and Solutions

### Function Invocation Failed

**Symptoms:** Error message "A server error has occurred FUNCTION_INVOCATION_FAILED"

**Solutions:**
1. Check requirements.txt for missing dependencies
2. Verify Python compatibility with Vercel (Python 3.9)
3. Check for syntax errors in your code
4. Look for large dependencies that exceed Vercel's size limits

### Connection Timeouts

**Symptoms:** Requests to the backend timeout or fail with connection errors

**Solutions:**
1. Verify the backend URL is correct in vercel.json
2. Check if the backend is running and accessible
3. Increase timeout values in vercel.json
4. Test connection using the debug proxy

### Authentication Issues

**Symptoms:** 401 Unauthorized errors when accessing endpoints

**Solutions:**
1. Set REQUIRE_AUTH=false in vercel.json
2. Use the protection bypass token in your requests
3. Check JWT configuration in vercel_integration.py

## Additional Resources

For more detailed information, refer to:

- `docs/troubleshooting_500_errors.md`: General troubleshooting guide
- `docs/troubleshooting_function_invocation.md`: Vercel-specific issues
- `DEPLOYMENT_SUMMARY.md`: Overview of deployment architecture
- `IMPROVEMENTS.md`: Summary of improvements made

## Getting Help

If you continue to experience issues, the most useful information to provide is:

1. The output of `./scripts/test_minimal_api.sh`
2. Vercel logs from the failing function
3. The specific error messages you're encountering
4. Your vercel.json configuration