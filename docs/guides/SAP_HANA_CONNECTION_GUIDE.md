# SAP HANA Cloud Connection Guide

This guide provides instructions for setting up and troubleshooting connections to SAP HANA Cloud for the LangChain integration.

## ✅ Successful Connection Configuration

We've confirmed that the following minimal configuration works for connecting to the SAP HANA Cloud instance:

```json
{
  "address": "d93a8739-44a8-4845-bef3-8ec724dea2ce.hana.prod-us10.hanacloud.ondemand.com",
  "port": 443,
  "user": "DBADMIN",
  "password": "Initial@1",
  "encrypt": true,
  "sslValidateCertificate": false
}
```

**Key findings**:
- The SAP HANA Cloud instance requires SSL encryption
- SSL certificate validation should be disabled for this instance
- Additional parameters like `connectTimeout` and `communicationTimeout` caused connection issues

## Initial Setup

1. **Create a virtual environment and install dependencies**:
   ```bash
   # Create a virtual environment
   python3 -m venv venv
   
   # Activate the virtual environment
   source venv/bin/activate
   
   # Install required packages
   pip install hdbcli langchain numpy torch
   ```

2. **Verify requirements are installed**:
   ```bash
   python check_packages.py
   ```
   
   Make sure all packages show as installed, especially `hdbcli`.

## Testing the Connection

1. **Run the connection test**:
   ```bash
   python test_connection.py
   ```

2. **For troubleshooting, use the enhanced test script**:
   ```bash
   python enhanced_test_connection.py
   ```

## Connection Troubleshooting

If you experience connection issues, try these steps in order:

1. **Start with minimal parameters**:
   ```json
   {
     "address": "your-host.hana.prod-region.hanacloud.ondemand.com",
     "port": 443,
     "user": "your-username",
     "password": "your-password"
   }
   ```

2. **Add SSL parameters if needed**:
   ```json
   {
     "address": "your-host.hana.prod-region.hanacloud.ondemand.com",
     "port": 443,
     "user": "your-username",
     "password": "your-password",
     "encrypt": true,
     "sslValidateCertificate": false
   }
   ```

3. **Try the simplified connection test**:
   ```bash
   python test_simplified_connection.py
   ```

## Common Connection Issues and Solutions

### Connection Timeout

If you see a timeout error (e.g., `-10709, 'Connect failed (connect timeout expired)'`):

1. **Try with minimal parameters** as shown above
2. **Check network connectivity**:
   ```bash
   nc -zv your-host.hana.prod-region.hanacloud.ondemand.com 443
   ```
3. **Verify instance is running** in the SAP BTP Cockpit

### SSL/TLS Issues

If you see SSL/TLS errors:

1. **Always use encryption** with `"encrypt": true`
2. **Disable certificate validation** with `"sslValidateCertificate": false`
3. **Check if your organization uses custom certificates**

### Authentication Errors

If you see authentication failures:

1. **Double-check credentials** (username/password)
2. **Verify database user exists** and has appropriate permissions
3. **Check if password needs to be reset**

## Connection Parameters Reference

### Essential Parameters (Recommended)

| Parameter | Description | Recommended Value |
|-----------|-------------|------------------|
| `address` | SAP HANA Cloud hostname | Required |
| `port` | SAP HANA Cloud port | 443 |
| `user` | Database username | Required |
| `password` | Database password | Required |
| `encrypt` | Use SSL/TLS encryption | `true` |
| `sslValidateCertificate` | Validate SSL certificates | `false` |

### Optional Parameters (Use with caution)

| Parameter | Description | Notes |
|-----------|-------------|-------|
| `connectTimeout` | Connection timeout (seconds) | Only add if connection works with minimal parameters |
| `communicationTimeout` | Query timeout (milliseconds) | Only add if connection works with minimal parameters |
| `reconnect` | Auto-reconnect if connection lost | Only add if connection works with minimal parameters |

## Next Steps

Once connection is established:

1. **Set up vector tables** for embedding storage
2. **Configure connection pooling** for production use
3. **Implement error handling** for specific SAP HANA error codes

## Support

If connection issues persist:

1. Check SAP HANA Cloud documentation for specific connection requirements
2. Verify network access from your current location to the SAP HANA Cloud instance
3. Contact SAP support if the connection issues continue