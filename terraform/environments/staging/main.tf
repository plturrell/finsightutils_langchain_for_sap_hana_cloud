module "staging" {
  source = "../../"
  
  # Core configuration
  environment       = "staging"
  namespace         = "langchain-hana-staging"
  replicas          = 1
  image_repository  = "langchain-hana-api"
  image_tag         = "latest"
  
  # GPU configuration
  gpu_enabled       = true
  nvidia_gpu_count  = 1
  
  # Resource limits
  memory_limit      = "4Gi"
  cpu_limit         = "2"
  memory_request    = "1Gi"
  cpu_request       = "0.5"
  
  # API configuration
  api_config = {
    tensorrt_enabled = true
    batch_size       = 32
    timeout_seconds  = 60
  }
  
  # Autoscaling configuration
  autoscaling = {
    enabled      = true
    min_replicas = 1
    max_replicas = 3
    cpu_target   = 70
  }
  
  # Monitoring configuration
  enable_monitoring = true
  
  # Backup configuration - disabled for staging
  enable_backup     = false
  
  # Network policy configuration - more permissive in staging
  enable_network_policies = false
  frontend_namespace      = "langchain-hana-frontend-staging"
  
  # Security configuration - more permissive in staging
  hana_cidr       = "0.0.0.0/0"  # Allow all in staging
  datasphere_cidr = "0.0.0.0/0"  # Allow all in staging
  oauth_cidr      = "0.0.0.0/0"  # Allow all in staging
}