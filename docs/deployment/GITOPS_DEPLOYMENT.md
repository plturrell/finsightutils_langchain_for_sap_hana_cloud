# GitOps Deployment Guide for LangChain SAP HANA Cloud Integration

This guide outlines how to deploy the LangChain SAP HANA Cloud Integration using GitOps principles with tools like Flux or ArgoCD. GitOps uses Git as the single source of truth for declarative infrastructure and applications.

## Prerequisites

- Kubernetes cluster (EKS, GKE, AKS, or on-premises)
- kubectl configured for your cluster
- Helm v3.x
- Flux or ArgoCD installed on your cluster
- SAP BTP account with SAP HANA Cloud instance
- Docker Hub or private container registry access

## Repository Structure for GitOps

For GitOps deployment, your repository should include:

```
deploy/
├── base/                      # Base Kubernetes manifests
│   ├── deployment.yaml        # API service deployment
│   ├── service.yaml           # Service definitions
│   ├── configmap.yaml         # Configuration
│   ├── secret.yaml.template   # Template for secrets
│   └── kustomization.yaml     # Kustomize base configuration
│
└── overlays/                  # Environment-specific configurations
    ├── development/           # Development environment
    │   ├── kustomization.yaml # Dev-specific customizations
    │   └── patch.yaml         # Patches for development
    │
    ├── staging/              # Staging environment
    │   ├── kustomization.yaml
    │   └── patch.yaml
    │
    └── production/           # Production environment
        ├── kustomization.yaml
        ├── patch.yaml
        ├── hpa.yaml          # Horizontal Pod Autoscaler
        └── pdb.yaml          # Pod Disruption Budget
```

## Setting Up Flux

### 1. Install Flux CLI

```bash
curl -s https://fluxcd.io/install.sh | bash
```

### 2. Bootstrap Flux on your Kubernetes cluster

```bash
flux bootstrap github \
  --owner=<your-github-username> \
  --repository=langchain-hana-gitops \
  --branch=main \
  --path=./clusters/production \
  --personal
```

### 3. Create a GitRepository resource

```yaml
# langchain-hana-source.yaml
apiVersion: source.toolkit.fluxcd.io/v1beta2
kind: GitRepository
metadata:
  name: langchain-hana
  namespace: flux-system
spec:
  interval: 1m
  url: https://github.com/your-org/langchain-integration-for-sap-hana-cloud
  ref:
    branch: main
```

### 4. Create a Kustomization resource

```yaml
# langchain-hana-kustomization.yaml
apiVersion: kustomize.toolkit.fluxcd.io/v1beta2
kind: Kustomization
metadata:
  name: langchain-hana
  namespace: flux-system
spec:
  interval: 5m
  path: ./deploy/overlays/production
  prune: true
  sourceRef:
    kind: GitRepository
    name: langchain-hana
  validation: client
  healthChecks:
    - apiVersion: apps/v1
      kind: Deployment
      name: langchain-hana-api
      namespace: langchain-hana
```

### 5. Apply the configuration

```bash
kubectl apply -f langchain-hana-source.yaml
kubectl apply -f langchain-hana-kustomization.yaml
```

## Setting Up ArgoCD

### 1. Create an Application resource

```yaml
# langchain-hana-application.yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: langchain-hana
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/your-org/langchain-integration-for-sap-hana-cloud.git
    targetRevision: HEAD
    path: deploy/overlays/production
  destination:
    server: https://kubernetes.default.svc
    namespace: langchain-hana
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
    - CreateNamespace=true
```

### 2. Apply the ArgoCD Application

```bash
kubectl apply -f langchain-hana-application.yaml
```

## SAP HANA Cloud Configuration

The LangChain SAP HANA Cloud Integration requires connection to a SAP HANA Cloud instance. Store your connection details securely:

### Using Kubernetes Secrets

```bash
kubectl create secret generic hana-cloud-credentials \
  --from-literal=host=<your-hana-hostname> \
  --from-literal=port=<your-hana-port> \
  --from-literal=user=<your-hana-user> \
  --from-literal=password=<your-hana-password> \
  --namespace langchain-hana
```

For GitOps, use a sealed secrets approach:

```bash
# Using kubeseal (Sealed Secrets)
kubectl create secret generic hana-cloud-credentials \
  --from-literal=host=<your-hana-hostname> \
  --from-literal=port=<your-hana-port> \
  --from-literal=user=<your-hana-user> \
  --from-literal=password=<your-hana-password> \
  --namespace langchain-hana \
  --dry-run=client -o yaml | \
kubeseal > deploy/base/sealed-hana-credentials.yaml
```

## GPU Configuration for Vector Operations

For deployments using GPU acceleration:

```yaml
# deploy/overlays/production/gpu-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: langchain-hana-gpu-config
data:
  GPU_ENABLED: "true"
  GPU_MEMORY_FRACTION: "0.85"
  VECTOR_BATCH_SIZE: "64"
  TENSORRT_ENABLED: "true"
  OPTIMIZATION_LEVEL: "3"
```

And reference it in your deployment patch:

```yaml
# deploy/overlays/production/patch.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: langchain-hana-api
spec:
  template:
    spec:
      containers:
      - name: langchain-hana-api
        resources:
          limits:
            nvidia.com/gpu: 1
        envFrom:
        - configMapRef:
            name: langchain-hana-gpu-config
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-tesla-t4  # Adjust based on your cloud provider
```

## Using Helm for Deployment

For more complex deployments, consider using the Helm chart:

```bash
# Install using Helm
helm upgrade --install langchain-hana ./deploy/helm \
  --namespace langchain-hana \
  --create-namespace \
  --set hanaCloud.host=${HANA_HOST} \
  --set hanaCloud.port=${HANA_PORT} \
  --set hanaCloud.user=${HANA_USER} \
  --set hanaCloud.password=${HANA_PASSWORD} \
  --set gpu.enabled=true \
  --set gpu.type=nvidia-tesla-t4 \
  --set replicas=3
```

## Monitoring and Observability

Deploy Prometheus and Grafana for monitoring:

```yaml
# prometheus-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: langchain-hana-prometheus-config
data:
  prometheus.yml: |
    scrape_configs:
      - job_name: 'langchain-hana'
        kubernetes_sd_configs:
          - role: pod
        relabel_configs:
          - source_labels: [__meta_kubernetes_pod_label_app]
            regex: langchain-hana
            action: keep
```

A pre-configured Grafana dashboard is available in `deploy/monitoring/grafana-dashboard.json`.

## CI/CD Integration with GitOps

Implement a CI/CD pipeline with GitHub Actions that integrates with your GitOps workflow:

1. On push to main or PR:
   - Run tests
   - Build Docker image
   - Push to registry
   - Update deployment manifests with new image tag
   - Commit changes to GitOps repository
2. Flux/ArgoCD automatically detects and applies changes

Example GitHub Actions workflow is available in `.github/workflows/ci-cd.yml`.

## SAP BTP Deployment

For SAP Business Technology Platform deployment:

```bash
# Deploy to SAP BTP using cf CLI
cf push langchain-hana-api \
  -f manifest.yml \
  --var hana_host=${HANA_HOST} \
  --var hana_port=${HANA_PORT} \
  --var hana_user=${HANA_USER} \
  --var hana_password=${HANA_PASSWORD}
```

## Best Practices

1. **Immutable Infrastructure**: Never modify running containers; rebuild and redeploy
2. **Progressive Delivery**: Use Flagger for canary deployments
3. **Infrastructure as Code**: Store all configuration in Git
4. **Automation**: Automate everything, from testing to deployment
5. **Observability**: Implement comprehensive monitoring and alerting
6. **Secrets Management**: Use a secure solution for managing secrets
7. **Resource Limits**: Set appropriate CPU and memory limits

## Security Considerations

1. Implement network policies to restrict traffic
2. Use Pod Security Policies/Security Contexts
3. Regularly scan container images for vulnerabilities
4. Rotate SAP HANA Cloud credentials regularly
5. Use RBAC to limit permissions

## Troubleshooting

Common issues and their solutions:

- **SAP HANA Connection Failures**: Verify credentials and network connectivity
- **Vector Operation Performance**: Check GPU utilization and batch size settings
- **Memory Issues**: Adjust memory limits and vector batch sizes
- **Resource Constraints**: Monitor resource usage and adjust limits and requests accordingly

## Disaster Recovery

1. **Database Backups**: Ensure regular backups of your SAP HANA Cloud instance
2. **Configuration Backups**: Store all configuration in Git
3. **Deployment Snapshots**: Create deployment snapshots for quick recovery
4. **Failover Procedures**: Document and practice failover procedures

## Next Steps

- Set up continuous monitoring and alerting
- Implement auto-scaling based on query load
- Configure backup and restore procedures
- Establish a regular update and patching schedule