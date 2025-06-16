#!/bin/bash

# Launch script with pre-configured SAP HANA Cloud credentials

# Set SAP HANA Cloud credentials
export HANA_HOST=d93a8739-44a8-4845-bef3-8ec724dea2ce.hana.prod-us10.hanacloud.ondemand.com
export HANA_PORT=443
export HANA_USER=DBADMIN
export HANA_PASSWORD=Initial@1
export DEFAULT_TABLE_NAME=EMBEDDINGS

# Set GPU configuration (adjust based on your hardware)
export GPU_ENABLED=true
export USE_TENSORRT=true
export TENSORRT_PRECISION=fp16
export ENABLE_MULTI_GPU=true

# Launch the NGC Blueprint
echo "Starting SAP HANA Cloud LangChain Integration with NGC Blueprint..."
echo "Using SAP HANA Cloud host: $HANA_HOST"

docker compose -f ngc-blueprint.yml up -d

echo "Services are starting..."
echo "API will be available at: http://localhost:8000"
echo "API Documentation: http://localhost:8000/docs"
echo "Frontend (if enabled): http://localhost:3000"