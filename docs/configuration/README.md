# Configuration Documentation

This directory contains all the configuration-related documentation for the SAP HANA Cloud LangChain Integration with NVIDIA GPU Acceleration.

## Overview

The SAP HANA Cloud LangChain Integration provides various configuration options to customize its behavior for different environments and use cases. This index will help you navigate through the various configuration resources.

## Configuration Resources

* [Configuration Guide](configuration_guide.md) - Comprehensive guide to configuring the system
* [Updated Configuration Guide](updated_configuration_guide.md) - Updated guide with the latest configuration options
* [Development Configuration](../development/configuration.md) - Configuration for development environments

## Configuration Categories

The configuration is organized into the following categories:

### 1. Database Configuration

Configuration for the SAP HANA Cloud connection.

* **Environment Variables**:
  * `HANA_HOST`: SAP HANA Cloud host
  * `HANA_PORT`: SAP HANA Cloud port
  * `HANA_USER`: SAP HANA Cloud username
  * `HANA_PASSWORD`: SAP HANA Cloud password
  * `DB_SCHEMA`: Database schema
  * `DB_TABLE`: Database table
  * `DB_CONNECTION_POOL_SIZE`: Connection pool size

* **Configuration File**: `/config/connection.json`

### 2. API Configuration

Configuration for the API server.

* **Environment Variables**:
  * `API_HOST`: API host
  * `API_PORT`: API port
  * `API_WORKERS`: Number of API workers
  * `API_TIMEOUT`: API timeout in seconds
  * `API_DEBUG`: Enable/disable debug mode
  * `API_KEY`: API key for authentication
  * `ENABLE_CORS`: Enable/disable CORS
  * `CORS_ORIGINS`: Allowed CORS origins

* **Configuration File**: `/api/config.py`

### 3. GPU Configuration

Configuration for GPU acceleration.

* **Environment Variables**:
  * `USE_GPU`: Enable/disable GPU acceleration
  * `GPU_DEVICE_ID`: GPU device ID
  * `USE_TENSORRT`: Enable/disable TensorRT optimization
  * `USE_FP16`: Enable/disable FP16 precision
  * `BATCH_SIZE`: Default batch size
  * `MAX_SEQUENCE_LENGTH`: Maximum sequence length
  * `ENABLE_TENSOR_CORES`: Enable/disable Tensor Core optimization
  * `MULTI_GPU`: Enable/disable multi-GPU support

* **Configuration File**: `/api/gpu/config.py`

### 4. Frontend Configuration

Configuration for the frontend.

* **Environment Variables**:
  * `BACKEND_URL`: Backend API URL
  * `AUTH_ENABLED`: Enable/disable authentication
  * `ENVIRONMENT`: Current environment (staging/production)

* **Configuration File**: `/frontend/.env.production`

### 5. Logging Configuration

Configuration for logging.

* **Environment Variables**:
  * `LOG_LEVEL`: Logging level
  * `LOG_FORMAT`: Logging format
  * `LOG_FILE`: Log file path
  * `ENABLE_SENTRY`: Enable/disable Sentry integration
  * `SENTRY_DSN`: Sentry DSN

* **Configuration File**: `/api/core/logging.py`

### 6. Monitoring Configuration

Configuration for monitoring.

* **Environment Variables**:
  * `ENABLE_PROMETHEUS`: Enable/disable Prometheus metrics
  * `PROMETHEUS_PORT`: Prometheus metrics port
  * `ENABLE_HEALTH_CHECKS`: Enable/disable health checks
  * `HEALTH_CHECK_INTERVAL`: Health check interval in seconds

* **Configuration Files**:
  * `/api/core/monitoring.py`
  * `/config/prometheus/prometheus.yml`

## Configuration Methods

The project supports multiple configuration methods:

### 1. Environment Variables

Environment variables are the primary method for configuration. They can be set in the following ways:

* Directly in the environment
* In a `.env` file
* In Docker Compose environment configuration
* In Kubernetes ConfigMaps and Secrets

### 2. Configuration Files

Configuration files provide more structured configuration options:

* `/config/connection.json`: Database connection configuration
* `/config/unified-config.yaml`: Unified configuration for all components

### 3. Command Line Arguments

Some components support command line arguments for configuration:

* `api/start.sh`: Script for starting the API server with command line arguments
* `scripts/run_api_local.sh`: Script for running the API locally with command line arguments

## Environment-Specific Configuration

The project supports different configurations for different environments:

### Development

* Detailed logging
* Debug mode enabled
* Local database connection
* CORS allowed for all origins

### Staging

* Standard logging
* Performance monitoring enabled
* Connection to staging database
* CORS allowed for specific origins

### Production

* Minimal logging (warnings and errors only)
* Comprehensive monitoring
* Connection to production database
* CORS restricted to production origins
* Enhanced security measures

## Best Practices

When configuring the system, consider these best practices:

1. **Use Environment Variables for Secrets**:
   - Store sensitive information like passwords and API keys in environment variables
   - Never hardcode secrets in configuration files

2. **Use Different Configurations for Different Environments**:
   - Create separate configuration files for development, staging, and production
   - Use environment-specific environment variables

3. **Document Configuration Changes**:
   - Keep track of configuration changes
   - Document the purpose of each configuration option

4. **Validate Configuration**:
   - Implement validation for configuration options
   - Provide clear error messages for invalid configuration

5. **Use Sensible Defaults**:
   - Provide sensible defaults for all configuration options
   - Document the default values

## Troubleshooting Configuration Issues

If you encounter configuration issues:

1. Check the logs for configuration-related errors
2. Verify that all required environment variables are set
3. Check the configuration files for syntax errors
4. Use the `api/core/auto_config.py` module to dump the current configuration
5. Consult the troubleshooting guide for common configuration issues