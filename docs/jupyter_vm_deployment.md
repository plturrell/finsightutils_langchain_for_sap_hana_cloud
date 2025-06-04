# Deploying to Jupyter VM with T4 GPU

This guide explains how to deploy the SAP HANA Cloud LangChain integration to a VM with Jupyter and NVIDIA T4 GPU.

## Prerequisites

- VM with Jupyter and NVIDIA T4 GPU
- Docker and Docker Compose installed on the VM
- NVIDIA Container Toolkit installed on the VM
- SSH access to the VM

## Option 1: Automated Deployment

The easiest way to deploy is using the automated deployment script:

```bash
./deploy_to_jupyter_vm.sh -h <vm-hostname> -u <username> -i <identity-file>
```

For example:

```bash
./deploy_to_jupyter_vm.sh -h jupyter0-4ckg1m6x0.brevlab.com -u ubuntu -i ~/.ssh/id_rsa
```

This script will:
1. Package the necessary files
2. Copy them to your VM
3. Run the deployment script on the VM
4. Start the API and frontend services with Docker Compose

## Option 2: Manual Deployment

If you prefer to deploy manually:

1. Copy the project files to your VM:

   ```bash
   scp -r ./api ./frontend ./langchain_hana ./deploy_to_vm.sh <username>@<vm-hostname>:/home/<username>/langchain-integration-for-sap-hana-cloud/
   ```

2. SSH into your VM:

   ```bash
   ssh <username>@<vm-hostname>
   ```

3. Navigate to the project directory and run the deployment script:

   ```bash
   cd /home/<username>/langchain-integration-for-sap-hana-cloud/
   chmod +x deploy_to_vm.sh
   ./deploy_to_vm.sh
   ```

## Accessing the Services

After deployment, you can access:

- API: `http://<vm-hostname>:8000`
- API Documentation: `http://<vm-hostname>:8000/docs`
- Frontend: `http://<vm-hostname>:3000`
- GPU Info: `http://<vm-hostname>:8000/benchmark/gpu_info`

## Managing the Services

To view logs:

```bash
docker-compose -f api/docker-compose.yml logs -f
```

To stop the services:

```bash
docker-compose -f api/docker-compose.yml down
```

To restart the services:

```bash
docker-compose -f api/docker-compose.yml restart
```

## Troubleshooting

### GPU Issues

If you encounter GPU-related issues:

1. Verify that NVIDIA drivers are installed:
   ```bash
   nvidia-smi
   ```

2. Check that NVIDIA Container Toolkit is installed:
   ```bash
   docker info | grep -i nvidia
   ```

3. Inspect the logs for GPU-related errors:
   ```bash
   docker-compose -f api/docker-compose.yml logs api
   ```

### Connection Issues

If your frontend can't connect to the API:

1. Verify that both services are running:
   ```bash
   docker-compose -f api/docker-compose.yml ps
   ```

2. Check if the API is accessible:
   ```bash
   curl http://localhost:8000/api/health
   ```

3. Verify network settings:
   ```bash
   docker network inspect api_default
   ```

### Port Conflicts

If you have port conflicts, you can modify the `docker-compose.yml` file to use different ports:

```yaml
services:
  api:
    ports:
      - "8001:8000"  # Change 8001 to an available port
  
  frontend:
    ports:
      - "3001:3000"  # Change 3001 to an available port
```