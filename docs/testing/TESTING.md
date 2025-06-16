# Testing SAP HANA Cloud and DataSphere Connections

This guide explains how to test real connections to SAP HANA Cloud and DataSphere before deploying with Terraform.

## Prerequisites

- SAP HANA Cloud instance with credentials
- SAP DataSphere tenant with OAuth client credentials
- Python 3.8+ (for local testing)
- Docker (optional, for containerized testing)
- kubectl with access to your Kubernetes cluster (optional, for cluster testing)

## 1. Set Up Your Credentials

First, set up your credentials using one of these methods:

### Option A: Use Environment Variables

```bash
# SAP HANA Cloud credentials
export HANA_HOST=your-hana-hostname.hanacloud.ondemand.com
export HANA_PORT=443  # or your custom port
export HANA_USER=your-username
export HANA_PASSWORD=your-password

# SAP DataSphere credentials
export DATASPHERE_CLIENT_ID=sb-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx!bxxxx|client!bxxxx
export DATASPHERE_CLIENT_SECRET=your-client-secret
export DATASPHERE_AUTH_URL=https://your-tenant.authentication.eu10.hana.ondemand.com/oauth/authorize
export DATASPHERE_TOKEN_URL=https://your-tenant.authentication.eu10.hana.ondemand.com/oauth/token
export DATASPHERE_API_URL=https://your-tenant.eu10.hcs.cloud.sap/api/v1
```

### Option B: Create a .env File

Create a `.env` file with the following content:

```
# SAP HANA Cloud credentials
HANA_HOST=your-hana-hostname.hanacloud.ondemand.com
HANA_PORT=443
HANA_USER=your-username
HANA_PASSWORD=your-password

# SAP DataSphere credentials
DATASPHERE_CLIENT_ID=sb-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx!bxxxx|client!bxxxx
DATASPHERE_CLIENT_SECRET=your-client-secret
DATASPHERE_AUTH_URL=https://your-tenant.authentication.eu10.hana.ondemand.com/oauth/authorize
DATASPHERE_TOKEN_URL=https://your-tenant.authentication.eu10.hana.ondemand.com/oauth/token
DATASPHERE_API_URL=https://your-tenant.eu10.hcs.cloud.sap/api/v1
```

### Option C: Convert from Terraform Variables

If you've already set up Terraform variables, convert them to a .env file:

```bash
# Convert credentials.tfvars to .env
./scripts/generate_env_from_tfvars.py
```

## 2. Install Required Dependencies

For local testing, install the required Python packages:

```bash
pip install hdbcli requests requests-oauthlib
```

## 3. Run the Connection Tests

### Method 1: Comprehensive Testing (Recommended)

This method performs real-world comprehensive tests with actual queries and operations:

```bash
# Test both HANA and DataSphere with detailed checks
python scripts/comprehensive_test.py --all

# Test SAP HANA with specific schema and custom query
python scripts/comprehensive_test.py --test-hana --schema YOUR_SCHEMA --query "SELECT * FROM SYS.TABLES LIMIT 5"

# Test DataSphere with specific space
python scripts/comprehensive_test.py --test-datasphere --space-id YOUR_SPACE_ID

# Save results to custom location
python scripts/comprehensive_test.py --all --output hana_ds_test_results.json
```

The comprehensive test performs:

- **For HANA**: 
  - Connection verification
  - Version query
  - System information retrieval
  - Schema access testing
  - Custom query execution

- **For DataSphere**:
  - OAuth authentication
  - Spaces listing
  - Space details retrieval
  - Assets listing within spaces

### Method 2: Basic Connection Test

This method tests just the connectivity:

```bash
# Test both HANA and DataSphere connections
python scripts/test_connections.py --all

# Test only HANA connection
python scripts/test_connections.py --test-hana

# Test only DataSphere connection
python scripts/test_connections.py --test-datasphere
```

### Method 3: Using the Helper Script

This method automatically handles environment variables from a .env file:

```bash
# Test both connections
./scripts/verify_credentials.sh

# Test only HANA connection
./scripts/verify_credentials.sh --hana-only

# Test only DataSphere connection
./scripts/verify_credentials.sh --datasphere-only
```

### Method 4: Using Docker (No Local Dependencies)

This method runs the tests in a Docker container, avoiding the need to install dependencies locally:

```bash
# Test using current environment variables
./scripts/test_connections_docker.sh

# Test using a .env file
./scripts/test_connections_docker.sh --env-file .env

# Test only HANA connection
./scripts/test_connections_docker.sh --hana-only --env-file .env
```

## 4. Understanding the Test Results

The tests perform the following real-world operations:

### SAP HANA Cloud Test:
1. Establishes a real connection to your HANA Cloud instance
2. Authenticates with your provided credentials
3. Executes a version query to verify the connection
4. Reports the actual SAP HANA version if successful

### SAP DataSphere Test:
1. Obtains a real OAuth token from the authentication endpoint
2. Uses the token to make an actual API call to the DataSphere API
3. Retrieves the list of spaces from your DataSphere tenant
4. Reports the number of spaces found if successful

## 5. Troubleshooting

### Common HANA Connection Issues:
- **Connection Timeout**: Check firewall rules and network connectivity
- **Authentication Failure**: Verify username and password
- **SSL/TLS Errors**: Check if your HANA instance requires specific certificates

### Common DataSphere Issues:
- **OAuth Token Failure**: Verify client ID and secret
- **API Access Denied**: Check if the OAuth client has appropriate scopes
- **URL Errors**: Ensure you're using the correct tenant-specific URLs

## Next Steps

After successfully testing the connections:

1. Deploy your infrastructure with Terraform
2. Verify connections from within the deployed environment using:
   ```bash
   ./scripts/k8s_test_connections.sh --env-file .env
   ```