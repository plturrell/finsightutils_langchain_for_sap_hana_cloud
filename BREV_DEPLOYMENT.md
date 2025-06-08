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

To run with an actual SAP HANA Cloud connection:

1. **Create an environment file**

   ```bash
   cat > .env << EOF
   HANA_HOST=your_hana_host.hanacloud.ondemand.com
   HANA_PORT=443
   HANA_USER=your_username
   HANA_PASSWORD=your_password
   TEST_MODE=false
   EOF
   ```

2. **Modify the deployment script**

   Edit `brev_deploy.sh` to load credentials from the .env file and set `TEST_MODE=false`.

## Monitoring

The API provides comprehensive monitoring endpoints:

- `/health/ping`: Simple health check
- `/health/status`: Basic status information
- `/health/complete`: Detailed health status of all components
- `/gpu/info`: GPU availability and information
- `/health/metrics`: Prometheus-format metrics

Use these endpoints to monitor the API's health and performance.