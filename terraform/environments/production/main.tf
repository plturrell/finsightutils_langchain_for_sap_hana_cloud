module "production" {
  source = "../../"
  
  # Core configuration
  environment       = "production"
  namespace         = "langchain-hana-production"
  replicas          = 3
  image_repository  = "langchain-hana-api"
  image_tag         = "stable"
  
  # GPU configuration
  gpu_enabled       = true
  nvidia_gpu_count  = 1
  
  # Resource limits
  memory_limit      = "8Gi"
  cpu_limit         = "4"
  memory_request    = "4Gi"
  cpu_request       = "1"
  
  # API configuration
  api_config = {
    tensorrt_enabled = true
    batch_size       = 64
    timeout_seconds  = 30
  }
  
  # Autoscaling configuration
  autoscaling = {
    enabled      = true
    min_replicas = 3
    max_replicas = 10
    cpu_target   = 60
  }
  
  # Monitoring configuration
  enable_monitoring = true
  
  # Backup configuration
  enable_backup       = true
  backup_schedule     = "0 1 * * *"  # Daily at 1 AM
  backup_storage_size = "50Gi"       # Larger storage for production
  storage_class       = "premium"    # Use premium storage for production
  
  # Network policy configuration
  enable_network_policies = true
  frontend_namespace      = "langchain-hana-frontend-production"
  
  # Security configuration - restrict network access in production
  hana_cidr       = "10.0.0.0/16"  # Example: Restrict to corporate network
  datasphere_cidr = "10.0.0.0/16"  # Example: Restrict to corporate network
  oauth_cidr      = "10.0.0.0/16"  # Example: Restrict to corporate network
  
  # Additional allowed ingress sources (e.g., VPN, office networks)
  allowed_ingress_sources = [
    "192.168.0.0/16",    # Example: Corporate VPN
    "203.0.113.0/24"     # Example: Office network
  ]
  
  # Additional allowed egress destinations
  allowed_egress_destinations = [
    {
      cidr     = "10.1.0.0/16"
      port     = 443
      protocol = "TCP"
    },
    {
      cidr     = "10.2.0.0/16"
      port     = 1521
      protocol = "TCP"
    }
  ]
}