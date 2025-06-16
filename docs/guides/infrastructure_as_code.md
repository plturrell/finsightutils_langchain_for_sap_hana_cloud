# Infrastructure as Code with Terraform

This document provides an overview of the Infrastructure as Code (IaC) implementation using Terraform for the LangChain HANA Cloud Integration project.

## Overview

We use Terraform to manage and provision the Kubernetes infrastructure required for the application. This approach provides several benefits:

1. **Consistency**: Infrastructure is defined as code, ensuring consistent environments
2. **Version Control**: Infrastructure changes are tracked in git
3. **Automation**: Infrastructure changes can be automated through CI/CD pipelines
4. **Documentation**: The infrastructure is self-documenting through code
5. **Repeatability**: Environments can be easily replicated or recreated

## Architecture

Our Terraform implementation manages the following components:

### Core Infrastructure
- Kubernetes namespaces
- Deployments with GPU support
- Services for API access
- ConfigMaps for non-sensitive configuration
- Secrets for sensitive information
- Horizontal Pod Autoscaler (HPA) for scaling

### Monitoring Infrastructure
- Prometheus for metrics collection
- Grafana for visualization and dashboards
- Pre-configured dashboards for API monitoring

### Backup Infrastructure
- Scheduled backups via CronJobs
- Persistent storage for backup data
- Backup and restore scripts
- Secure handling of database credentials

### Network Security
- Network policies for ingress/egress control
- Namespace isolation
- Fine-grained access control
- CIDR-based restrictions for external services

## Directory Structure

```
terraform/
├── modules/
│   ├── kubernetes/
│   │   ├── main.tf           # Core Kubernetes resources
│   │   ├── variables.tf      # Input variables
│   │   ├── outputs.tf        # Output values
│   │   └── secrets.tf        # Secret management
│   ├── monitoring/
│   │   ├── main.tf           # Prometheus and Grafana resources
│   │   ├── variables.tf      # Input variables
│   │   └── outputs.tf        # Output values
│   ├── backup/
│   │   ├── main.tf           # Backup CronJobs and storage
│   │   ├── variables.tf      # Input variables
│   │   └── outputs.tf        # Output values
│   └── network/
│       ├── main.tf           # Network policies
│       └── variables.tf      # Input variables
├── environments/
│   ├── staging/
│   │   └── main.tf           # Staging-specific configuration
│   └── production/
│       └── main.tf           # Production-specific configuration
├── main.tf                   # Main configuration
├── variables.tf              # Common variables
├── credentials.tfvars.example # Example credentials template
└── README.md                 # Documentation
```

## Environment-Specific Configurations

We maintain separate configurations for staging and production environments:

### Staging
- Lower resource requirements
- Single replica by default
- Configured for development and testing
- Network policies disabled for easier debugging
- Backup system disabled to reduce resource usage
- Less restrictive security settings

### Production
- Higher resource requirements
- Multiple replicas for high availability
- Stricter autoscaling policies
- Optimized for performance
- Daily automated backups
- Strict network policies for enhanced security
- CIDR-based restrictions for external services
- Premium storage class for reliability

## CI/CD Integration

The Terraform configuration is integrated with our CI/CD pipeline:

1. **Pull Requests**: 
   - Terraform plans are generated and posted as comments
   - Format and validation checks are performed

2. **Merges to Staging Branch**:
   - Terraform applies changes to the staging environment automatically

3. **Merges to Main Branch**:
   - Terraform applies changes to the production environment automatically

## GPU Support

Our Terraform configuration includes specific support for NVIDIA GPUs:

1. Resource limits for GPU allocation
2. Node selectors for GPU-enabled nodes
3. Configuration for TensorRT optimization

## Monitoring

The monitoring module deploys:

1. **Prometheus**:
   - Collects metrics from the API service
   - Configured with appropriate retention and storage
   - Custom scrape configurations for the API service

2. **Grafana**:
   - Pre-configured dashboards for API monitoring
   - Visualizations for request rates and latencies
   - Integration with Prometheus data source

## Backup System

The backup module provides automated data protection:

1. **Scheduled Backups**:
   - Configurable cron schedule (daily by default)
   - Full database backups
   - Compressed storage for efficiency

2. **Storage Management**:
   - Persistent volume claims for reliable storage
   - Configurable storage size and class
   - Environment-specific storage configurations

3. **Backup Scripts**:
   - Pre-configured backup and restore scripts
   - Automated backup manifests and metadata
   - Error handling and reporting

4. **Security**:
   - Secure handling of database credentials
   - Isolated namespace for backup operations
   - Resource limits for backup jobs

## Network Policies

The network module enhances security through:

1. **Ingress Control**:
   - Default deny-all policy
   - Explicit allowance for necessary services
   - Namespace-based access control
   - CIDR-based filtering for external access

2. **Egress Control**:
   - Restricted outbound connections
   - Explicit allowance for external services
   - Protocol and port-specific rules
   - DNS resolution allowance

3. **Environment-Specific Settings**:
   - Production: Strict policies with explicit CIDRs
   - Staging: More permissive for easier development

## Security Considerations

While implementing Infrastructure as Code, we've considered several security aspects:

1. **RBAC**:
   - Resources are namespaced for proper isolation
   - Service accounts with minimal permissions

2. **Resource Isolation**:
   - Clear separation between environments
   - Dedicated namespaces for different components

## Secrets Management

### Supported Credentials

Our Terraform configuration includes secure management for the following credential types:

1. **SAP HANA Cloud Credentials**:
   - Host, port, username, and password
   - Stored as Kubernetes secrets
   - Mounted as environment variables in the API container

2. **SAP DataSphere Credentials**:
   - Client ID and client secret
   - Authentication and token URLs
   - API URL
   - Stored as Kubernetes secrets
   - Mounted as environment variables in the API container

3. **API Secrets**:
   - General API keys and authentication tokens
   - Stored as Kubernetes secrets

4. **Monitoring Credentials**:
   - Grafana admin password
   - Stored securely in the Helm release

### Secure Credential Management

For local development and manual deployments, we provide a `credentials.tfvars.example` template that should be copied to a local `credentials.tfvars` file (which is excluded from git):

```bash
cp credentials.tfvars.example credentials.tfvars
```

For production environments, these should be populated using a secure method such as:

1. Using a CI/CD pipeline to inject secrets from a secure storage
2. Using External Secrets Operator to sync secrets from external sources
3. Using HashiCorp Vault for secret management

Example of securely loading credentials in a CI/CD pipeline:

```yaml
- name: Set up Terraform credentials
  run: |
    cat > credentials.tfvars << EOF
    hana_credentials = {
      host     = "${{ secrets.HANA_HOST }}"
      port     = "${{ secrets.HANA_PORT }}"
      user     = "${{ secrets.HANA_USER }}"
      password = "${{ secrets.HANA_PASSWORD }}"
    }
    
    datasphere_credentials = {
      client_id     = "${{ secrets.DATASPHERE_CLIENT_ID }}"
      client_secret = "${{ secrets.DATASPHERE_CLIENT_SECRET }}"
      auth_url      = "${{ secrets.DATASPHERE_AUTH_URL }}"
      token_url     = "${{ secrets.DATASPHERE_TOKEN_URL }}"
      api_url       = "${{ secrets.DATASPHERE_API_URL }}"
    }
    EOF
```

## Getting Started

To work with the Terraform configuration:

1. **Prerequisites**:
   - Terraform v1.0.0 or later
   - kubectl configured with access to your Kubernetes cluster
   - Helm v3.0.0 or later

2. **Initialize Terraform**:
   ```bash
   cd terraform/environments/staging
   terraform init
   ```

3. **Plan Changes**:
   ```bash
   terraform plan -out=tfplan
   ```

4. **Apply Changes**:
   ```bash
   terraform apply tfplan
   ```

5. **Destroy Infrastructure** (if needed):
   ```bash
   terraform destroy
   ```

## Best Practices

When working with our Terraform configuration:

1. Always run `terraform plan` before applying changes
2. Keep environment-specific values in the appropriate environment directories
3. Avoid hardcoding sensitive information
4. Use meaningful variable names and include descriptions
5. Add comments to explain complex or non-obvious configurations
6. Test changes in staging before applying to production

## Future Improvements

Potential enhancements to our Infrastructure as Code implementation:

1. **Secrets Management**:
   - Integration with HashiCorp Vault for secrets management
   - External Secrets Operator integration for cloud provider secrets
   - Automated credential rotation

2. **State Management**:
   - Remote state storage with locking (S3, GCS, or Azure Blob)
   - State encryption for enhanced security
   - Terraform Cloud integration for collaborative workflows

3. **Additional Modules**:
   - Disaster recovery automation
   - Multi-region deployment support
   - Blue-green deployment capabilities
   - Canary deployment support

4. **Advanced Security**:
   - Pod Security Policies or Pod Security Standards
   - Service Mesh integration (Istio/Linkerd)
   - Automated security scanning in CI/CD

5. **Cost Optimization**:
   - Spot instance support for non-critical components
   - Resource utilization monitoring and adjustment
   - Automated scaling based on time schedules

6. **Cloud Provider Integration**:
   - Cloud-specific storage classes
   - Managed database services integration
   - Cloud-native monitoring integration