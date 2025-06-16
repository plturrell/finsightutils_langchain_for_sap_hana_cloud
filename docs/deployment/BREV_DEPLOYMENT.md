# Deploying on Brev Container with Python and CUDA

This guide explains how to deploy the SAP HANA Cloud LangChain Integration on Brev Container with Python and CUDA preinstalled.

## Prerequisites

- Brev environment with Python and CUDA preinstalled
- Git access to this repository

## Deployment Steps

1. **Clone the repository**

   ```bash
   git clone https://github.com/plturrell/finsightutils_langchain_for_sap_hana_cloud.git
   cd finsightutils_langchain_for_sap_hana_cloud
   ```

2. **Run the deployment script**

   ```bash
   # Make the script executable if needed
   chmod +x brev_deploy.sh
   
   # Run the deployment script
   ./brev_deploy.sh
   ```

   This script:
   - Checks the environment and GPU availability
   - Installs all dependencies
   - Sets up test mode to run without SAP HANA Cloud connection
   - Runs the API with comprehensive logging
   - Saves logs to the `logs` directory

3. **Check API health**

   ```bash
   # In a separate terminal, check the API health
   chmod +x check_api_health.sh
   ./check_api_health.sh
   ```

   The health check script will test various endpoints to ensure the API is working correctly.

## Troubleshooting

If you encounter issues, follow these steps:

1. **Check the logs**

   The deployment script creates detailed logs in the `logs` directory. Review them for any errors.

   ```bash
   ls -la logs/
   cat logs/deployment_YYYYMMDD_HHMMSS.log
   ```

2. **Verify API is running**

   ```bash
   ps aux | grep uvicorn
   curl http://localhost:8000/health/ping
   ```

3. **Common issues and solutions**

   - **Missing dependencies**: Make sure all requirements are installed
     ```bash
     pip install --no-cache-dir -r requirements.txt
     pip install --no-cache-dir -r api/requirements.txt
     ```

   - **Permission issues**: Ensure scripts are executable
     ```bash
     chmod +x *.sh
     ```

   - **Port conflicts**: If port 8000 is in use, modify the port in `brev_deploy.sh`
     ```bash
     # Change the port number in the script
     export PORT=8001
     python -m uvicorn app:app --host 0.0.0.0 --port 8001 --log-level debug --reload
     ```

   - **GPU not detected**: Verify CUDA is available
     ```bash
     nvidia-smi
     python -c "import torch; print(torch.cuda.is_available())"
     ```

## Running in Production Mode

After successfully deploying the API in test mode, you can configure it to connect to a real SAP HANA Cloud instance:

1. **Run the configuration script**

   ```bash
   ./configure_hana.sh
   ```

   This interactive script will:
   - Prompt you for SAP HANA Cloud connection details
   - Create the necessary configuration files (.env and config/connection.json)
   - Test the connection to verify credentials
   - Provide instructions for restarting the API with the new configuration

2. **Restart the API with the new configuration**

   ```bash
   # Stop the current API instance
   pkill -f "uvicorn app:app"
   
   # Restart with the new configuration
   nohup ./brev_deploy.sh > logs/api_$(date +%Y%m%d_%H%M%S).log 2>&1 &
   ```

3. **Verify the connection**

   ```bash
   # Check the database connection status
   curl http://localhost:8000/health/database
   ```

   You should see a response with "status": "ok" if the connection is successful.

## Monitoring

The API provides comprehensive monitoring endpoints:

- `/health/ping`: Simple health check
- `/health/status`: Basic status information
- `/health/complete`: Detailed health status of all components
- `/gpu/info`: GPU availability and information
- `/health/metrics`: Prometheus-format metrics

Use these endpoints to monitor the API's health and performance.