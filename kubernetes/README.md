# Kubernetes Configuration for LangChain SAP HANA Integration

This directory contains Kubernetes configuration files for deploying the LangChain SAP HANA integration backend to Kubernetes clusters with NVIDIA GPU support.

## Directory Structure

- `staging/`: Configuration files for the staging environment
- `production/`: Configuration files for the production environment

Each environment directory contains:

- `namespace.yaml`: Kubernetes namespace definition
- `configmap.yaml`: Non-sensitive configuration
- `secrets.yaml`: Sensitive configuration (with placeholder values)
- `deployment.yaml`: Pod deployment configuration with GPU support
- `service.yaml`: Service definition for external access
- `hpa.yaml`: Horizontal Pod Autoscaler (production only)

## Deployment

### Prerequisites

- Kubernetes cluster with NVIDIA GPU support
- `kubectl` configured to access the cluster
- Proper secrets configured in your CI/CD pipeline

### Manual Deployment

To manually deploy to a specific environment:

```bash
# Deploy to staging
kubectl apply -f kubernetes/staging/

# Deploy to production
kubectl apply -f kubernetes/production/
```

### Automated Deployment

The CI/CD pipeline automatically deploys to:
- Staging when changes are pushed to the `main` branch
- Production when version tags (e.g., `v1.0.0`) are pushed

## Configuration Details

### Resource Requirements

- **Staging**:
  - 1 NVIDIA GPU
  - 4Gi memory
  - 1 CPU core
  - 2 replicas

- **Production**:
  - 1 NVIDIA GPU
  - 8Gi memory
  - 2 CPU cores
  - 3+ replicas (auto-scaled)

### Autoscaling (Production)

The Horizontal Pod Autoscaler (HPA) is configured to:
- Maintain minimum 3 replicas
- Scale up to 10 replicas based on:
  - CPU utilization (75% threshold)
  - Memory utilization (75% threshold)
- Include a 5-minute stabilization window for scale-down operations

### Security

- Namespaces provide isolation between environments
- Secrets are managed by the CI/CD pipeline
- ConfigMaps store non-sensitive configuration
- RBAC is configured for minimal required permissions

## Monitoring

The deployment includes:

- Health check endpoints (`/health`, `/health/ready`)
- Prometheus metrics endpoint (`/metrics`)
- Kubernetes liveness and readiness probes

## Troubleshooting

To check the status of deployments:

```bash
# List pods
kubectl get pods -n langchain-hana-production

# View logs
kubectl logs -n langchain-hana-production deployment/langchain-hana-api

# Describe deployment
kubectl describe deployment -n langchain-hana-production langchain-hana-api

# Check auto-scaling status
kubectl get hpa -n langchain-hana-production
```