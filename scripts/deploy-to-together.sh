#!/bin/bash
set -e

# Deploy to Together.ai
# This script deploys the backend to Together.ai

echo "Deploying backend to Together.ai"

# Check if TOGETHER_API_KEY is set
if [ -z "$TOGETHER_API_KEY" ]; then
    echo "Error: TOGETHER_API_KEY is not set"
    exit 1
fi

# Install required dependencies
pip install -q requests packaging

# Create deployment package
echo "Creating deployment package"
DEPLOYMENT_DIR="deployment_together"
rm -rf $DEPLOYMENT_DIR
mkdir -p $DEPLOYMENT_DIR

# Copy necessary files
cp -r api/* $DEPLOYMENT_DIR/
cp -r langchain_hana $DEPLOYMENT_DIR/
cp requirements.txt $DEPLOYMENT_DIR/

# Create together.yaml configuration
cat > $DEPLOYMENT_DIR/together.yaml << EOL
name: langchain-hana-integration
version: 1.0.0
description: SAP HANA Cloud LangChain Integration
runtime: python3.10
build:
  python_version: "3.10"
  python_packages:
    - langchain-hana
    - fastapi
    - uvicorn
    - together
entrypoint: uvicorn app:app --host 0.0.0.0 --port 8000
environment:
  - name: TOGETHER_API_KEY
    value: \${TOGETHER_API_KEY}
  - name: BACKEND_PLATFORM
    value: "together_ai"
  - name: HANA_HOST
    value: \${HANA_HOST}
  - name: HANA_PORT
    value: \${HANA_PORT}
  - name: HANA_USER
    value: \${HANA_USER}
  - name: HANA_PASSWORD
    value: \${HANA_PASSWORD}
EOL

# Create deployment script using Python
cat > deploy.py << EOL
import os
import json
import requests
import tarfile
import time
from pathlib import Path

# Configuration
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")
API_URL = "https://api.together.xyz/v1/deployments"
DEPLOYMENT_DIR = "$DEPLOYMENT_DIR"
DEPLOYMENT_NAME = "langchain-hana-integration"

# Create deployment package
print("Preparing deployment package...")
archive_path = "deployment.tar.gz"
with tarfile.open(archive_path, "w:gz") as tar:
    for item in Path(DEPLOYMENT_DIR).glob("**/*"):
        if item.is_file():
            tar.add(item, arcname=str(item.relative_to(DEPLOYMENT_DIR)))

# Upload deployment package
print("Uploading deployment package...")
with open(archive_path, "rb") as f:
    files = {"file": (archive_path, f)}
    headers = {"Authorization": f"Bearer {TOGETHER_API_KEY}"}
    response = requests.post(
        f"{API_URL}/upload",
        headers=headers,
        files=files
    )
    
    if response.status_code != 200:
        print(f"Error uploading deployment package: {response.text}")
        exit(1)
    
    upload_id = response.json().get("upload_id")
    print(f"Upload successful, ID: {upload_id}")

# Check for existing deployment
print("Checking for existing deployment...")
headers = {"Authorization": f"Bearer {TOGETHER_API_KEY}"}
response = requests.get(
    f"{API_URL}/list",
    headers=headers
)

deployment_exists = False
deployment_id = None

if response.status_code == 200:
    deployments = response.json().get("deployments", [])
    for deployment in deployments:
        if deployment.get("name") == DEPLOYMENT_NAME:
            deployment_exists = True
            deployment_id = deployment.get("id")
            print(f"Found existing deployment: {deployment_id}")
            break

# Create or update deployment
if deployment_exists:
    print(f"Updating existing deployment: {deployment_id}")
    response = requests.post(
        f"{API_URL}/update",
        headers=headers,
        json={
            "deployment_id": deployment_id,
            "upload_id": upload_id
        }
    )
else:
    print("Creating new deployment")
    response = requests.post(
        f"{API_URL}/create",
        headers=headers,
        json={
            "name": DEPLOYMENT_NAME,
            "upload_id": upload_id,
            "instance_type": "cpu-small",
            "num_instances": 1
        }
    )

if response.status_code not in [200, 201, 202]:
    print(f"Error creating/updating deployment: {response.text}")
    exit(1)

result = response.json()
deployment_id = result.get("deployment_id") or deployment_id
print(f"Deployment successful: {deployment_id}")

# Wait for deployment to be ready
print("Waiting for deployment to be ready...")
max_retries = 30
retries = 0
deployment_url = None

while retries < max_retries:
    response = requests.get(
        f"{API_URL}/get",
        headers=headers,
        params={"deployment_id": deployment_id}
    )
    
    if response.status_code == 200:
        status = response.json().get("status")
        if status == "RUNNING":
            deployment_url = response.json().get("url")
            print(f"Deployment is ready: {deployment_url}")
            break
        print(f"Deployment status: {status}")
    else:
        print(f"Error checking deployment status: {response.text}")
    
    retries += 1
    time.sleep(10)

if deployment_url:
    print(f"Deployment successful!")
    print(f"Deployment URL: {deployment_url}")
    
    # Save deployment URL to a file
    with open("deployment_url.txt", "w") as f:
        f.write(deployment_url)
else:
    print("Deployment not ready after maximum retries")
EOL

# Run the deployment script
echo "Running deployment script"
python deploy.py

echo "Deployment to Together.ai completed"