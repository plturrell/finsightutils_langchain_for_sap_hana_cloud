# Extensibility Guide

This document outlines how to extend the SAP HANA Cloud LangChain Integration to support additional deployment platforms beyond the default NVIDIA GPU backend + Vercel frontend configuration.

## Architecture Overview

The system is designed with a modular architecture that separates:

1. **Core Backend Logic**: API and integration with SAP HANA Cloud
2. **Infrastructure Configuration**: Deployment and runtime environment
3. **Frontend Application**: User interface and interaction
4. **Connection Layer**: Communication between frontend and backend

This separation makes it possible to extend the system to support additional deployment platforms without changing the core functionality.

## Adding Support for SAP BTP

### Backend on SAP BTP

To deploy the backend on SAP BTP (Cloud Foundry):

1. Create a `manifest.yml` file in the project root:

```yaml
applications:
- name: sap-hana-langchain-backend
  memory: 2G
  instances: 1
  path: .
  buildpacks:
    - python_buildpack
  command: python -m uvicorn api.app:app --host 0.0.0.0 --port $PORT
  env:
    PLATFORM: sap_btp
    GPU_ENABLED: false
    USE_INTERNAL_EMBEDDINGS: true
```

2. Create a `runtime.txt` file specifying the Python version:

```
python-3.9.x
```

3. Update `requirements.txt` to ensure compatibility with SAP BTP

4. Deploy using the CF CLI:

```bash
cf login
cf push
```

### Frontend on SAP BTP

To deploy the frontend on SAP BTP (HTML5 Application Repository):

1. Add a build script for BTP in `frontend/package.json`:

```json
"scripts": {
  "build:btp": "REACT_APP_API_URL=$BTP_BACKEND_URL react-scripts build",
}
```

2. Create an HTML5 application descriptor:

```yaml
# manifest.yml
applications:
- name: sap-hana-langchain-ui
  memory: 256M
  path: frontend/build
  buildpacks:
    - staticfile_buildpack
```

3. Configure destinations in SAP BTP cockpit to connect to the backend

## Adding Support for Kubernetes

### Backend on Kubernetes

1. Create Kubernetes deployment files in `kubernetes/backend/`:

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sap-hana-langchain-backend
spec:
  replicas: 1
  selector:
    matchLabels:
      app: sap-hana-langchain-backend
  template:
    metadata:
      labels:
        app: sap-hana-langchain-backend
    spec:
      containers:
      - name: backend
        image: ${YOUR_REGISTRY}/sap-hana-langchain-backend:latest
        ports:
        - containerPort: 8000
        env:
        - name: PLATFORM
          value: kubernetes
        # Add other environment variables as needed
```

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: sap-hana-langchain-backend
spec:
  selector:
    app: sap-hana-langchain-backend
  ports:
  - port: 8000
    targetPort: 8000
  type: ClusterIP
```

2. For GPU support, add NVIDIA GPU resource requests:

```yaml
spec:
  template:
    spec:
      containers:
      - name: backend
        resources:
          limits:
            nvidia.com/gpu: 1
```

### Frontend on Kubernetes

1. Create Kubernetes deployment files in `kubernetes/frontend/`:

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sap-hana-langchain-frontend
spec:
  replicas: 2
  selector:
    matchLabels:
      app: sap-hana-langchain-frontend
  template:
    metadata:
      labels:
        app: sap-hana-langchain-frontend
    spec:
      containers:
      - name: frontend
        image: ${YOUR_REGISTRY}/sap-hana-langchain-frontend:latest
        ports:
        - containerPort: 80
        env:
        - name: REACT_APP_API_URL
          value: http://sap-hana-langchain-backend:8000
```

## Extending for Other Cloud Providers

### AWS Configuration

To deploy on AWS:

1. Create AWS-specific configuration files in an `aws/` directory
2. Use AWS ECS with GPU support for the backend
3. Use AWS Amplify or S3+CloudFront for the frontend
4. Configure AWS-specific environment variables

### Azure Configuration

To deploy on Azure:

1. Create Azure-specific configuration files in an `azure/` directory
2. Use Azure Container Instances with GPU support for the backend
3. Use Azure Static Web Apps for the frontend
4. Configure Azure-specific environment variables

## Making Custom Configurations

The system is designed to be flexible and customizable. To create a custom configuration:

1. Create a new directory for your custom configuration (e.g., `custom/`)
2. Create a `docker-compose.custom.yml` file based on the existing `docker-compose.backend.yml`
3. Adjust environment variables, volumes, and networking as needed
4. Create a custom Dockerfile if necessary
5. Update the frontend configuration to connect to your custom backend

## Best Practices for Extensions

When extending the system:

1. **Maintain API Compatibility**: Ensure your extension implements the same API endpoints
2. **Environment Variables**: Use environment variables for configuration
3. **Connection Configuration**: Update the connection configuration to match your deployment
4. **Documentation**: Document your extension thoroughly
5. **Testing**: Test all functionality in the new environment
6. **Security**: Ensure secure connections between frontend and backend
7. **Monitoring**: Implement appropriate monitoring for your environment

## Extension Points

The main extension points in the system are:

1. **Deployment Configuration**: How the system is deployed (Docker, Kubernetes, cloud-specific)
2. **Environment Configuration**: Environment-specific settings
3. **Connection Layer**: How frontend and backend communicate
4. **Authentication**: How users authenticate with the system
5. **Monitoring**: How the system is monitored