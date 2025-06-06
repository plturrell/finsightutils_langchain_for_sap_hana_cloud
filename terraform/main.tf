terraform {
  required_providers {
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.10"
    }
  }
  
  # Optional: Configure backend for state storage
  # backend "s3" {
  #   bucket = "terraform-state-langchain-hana"
  #   key    = "terraform.tfstate"
  #   region = "us-east-1"
  # }
}

provider "kubernetes" {
  config_path    = var.kube_config_path
  config_context = var.kube_context
}

provider "helm" {
  kubernetes {
    config_path    = var.kube_config_path
    config_context = var.kube_context
  }
}

module "kubernetes_infrastructure" {
  source = "./modules/kubernetes"
  
  environment       = var.environment
  namespace         = var.namespace
  replicas          = var.replicas
  image_repository  = var.image_repository
  image_tag         = var.image_tag
  gpu_enabled       = var.gpu_enabled
  nvidia_gpu_count  = var.nvidia_gpu_count
  memory_limit      = var.memory_limit
  cpu_limit         = var.cpu_limit
  memory_request    = var.memory_request
  cpu_request       = var.cpu_request
  
  # Pass other variables to the module as needed
  api_config            = var.api_config
  autoscaling           = var.autoscaling
  
  # Pass credentials
  hana_credentials      = var.hana_credentials
  datasphere_credentials = var.datasphere_credentials
}

# Optional: Monitoring stack with Prometheus and Grafana
module "monitoring" {
  count  = var.enable_monitoring ? 1 : 0
  source = "./modules/monitoring"
  
  namespace = "${var.namespace}-monitoring"
  environment = var.environment
}

# Optional: Backup system for HANA data
module "backup" {
  count  = var.enable_backup ? 1 : 0
  source = "./modules/backup"
  
  namespace = "${var.namespace}-backup"
  environment = var.environment
  
  # HANA connection details for backup
  hana_host     = var.hana_credentials.host
  hana_port     = var.hana_credentials.port
  hana_user     = var.hana_credentials.user
  hana_password = var.hana_credentials.password
  
  # Backup configuration
  backup_schedule    = var.backup_schedule
  backup_storage_size = var.backup_storage_size
  storage_class      = var.storage_class
}

# Network policies for enhanced security
module "network" {
  source = "./modules/network"
  
  namespace = var.namespace
  enable_network_policies = var.enable_network_policies
  
  # Configuration
  frontend_namespace = var.frontend_namespace
  monitoring_namespace = var.enable_monitoring ? module.monitoring[0].namespace : ""
  enable_monitoring = var.enable_monitoring
  
  # CIDR ranges for external services
  hana_cidr       = var.hana_cidr
  datasphere_cidr = var.datasphere_cidr
  oauth_cidr      = var.oauth_cidr
  
  # Port configuration
  hana_port = tonumber(var.hana_credentials.port)
  
  # Custom ingress/egress rules
  allowed_ingress_sources = var.allowed_ingress_sources
  allowed_egress_destinations = var.allowed_egress_destinations
}