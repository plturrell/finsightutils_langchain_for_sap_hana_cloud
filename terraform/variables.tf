variable "kube_config_path" {
  description = "Path to the Kubernetes config file"
  type        = string
  default     = "~/.kube/config"
}

variable "kube_context" {
  description = "Kubernetes context to use"
  type        = string
  default     = ""
}

variable "environment" {
  description = "Deployment environment (staging or production)"
  type        = string
  validation {
    condition     = contains(["staging", "production"], var.environment)
    error_message = "Environment must be either 'staging' or 'production'."
  }
}

variable "namespace" {
  description = "Kubernetes namespace for deployment"
  type        = string
  default     = "langchain-hana"
}

variable "replicas" {
  description = "Number of API replicas to deploy"
  type        = number
  default     = 2
}

variable "image_repository" {
  description = "Docker image repository for the API"
  type        = string
  default     = "langchain-hana-api"
}

variable "image_tag" {
  description = "Docker image tag for the API"
  type        = string
  default     = "latest"
}

variable "gpu_enabled" {
  description = "Whether to enable GPU resources"
  type        = bool
  default     = true
}

variable "nvidia_gpu_count" {
  description = "Number of NVIDIA GPUs to allocate per pod"
  type        = number
  default     = 1
}

variable "memory_limit" {
  description = "Memory limit for API containers"
  type        = string
  default     = "4Gi"
}

variable "cpu_limit" {
  description = "CPU limit for API containers"
  type        = string
  default     = "2"
}

variable "memory_request" {
  description = "Memory request for API containers"
  type        = string
  default     = "2Gi"
}

variable "cpu_request" {
  description = "CPU request for API containers"
  type        = string
  default     = "0.5"
}

variable "api_config" {
  description = "Configuration for the API"
  type = object({
    tensorrt_enabled = bool
    batch_size       = number
    timeout_seconds  = number
  })
  default = {
    tensorrt_enabled = true
    batch_size       = 32
    timeout_seconds  = 60
  }
}

variable "autoscaling" {
  description = "Autoscaling configuration"
  type = object({
    enabled     = bool
    min_replicas = number
    max_replicas = number
    cpu_target   = number
  })
  default = {
    enabled      = true
    min_replicas = 2
    max_replicas = 10
    cpu_target   = 70
  }
}

variable "enable_monitoring" {
  description = "Whether to deploy monitoring stack"
  type        = bool
  default     = true
}

variable "enable_backup" {
  description = "Whether to deploy backup system"
  type        = bool
  default     = false
}

variable "backup_schedule" {
  description = "Cron schedule for backups"
  type        = string
  default     = "0 1 * * *"  # Daily at 1 AM
}

variable "backup_storage_size" {
  description = "Size of the persistent volume for backups"
  type        = string
  default     = "10Gi"
}

variable "storage_class" {
  description = "Storage class for persistent volumes"
  type        = string
  default     = "standard"
}

variable "enable_network_policies" {
  description = "Whether to enable network policies"
  type        = bool
  default     = false
}

variable "frontend_namespace" {
  description = "Namespace where frontend is deployed"
  type        = string
  default     = ""
}

variable "hana_cidr" {
  description = "CIDR block for SAP HANA Cloud"
  type        = string
  default     = "0.0.0.0/0"  # Should be restricted in production
}

variable "datasphere_cidr" {
  description = "CIDR block for SAP DataSphere"
  type        = string
  default     = "0.0.0.0/0"  # Should be restricted in production
}

variable "oauth_cidr" {
  description = "CIDR block for OAuth server"
  type        = string
  default     = "0.0.0.0/0"  # Should be restricted in production
}

variable "allowed_ingress_sources" {
  description = "List of CIDRs allowed for ingress"
  type        = list(string)
  default     = []
}

variable "allowed_egress_destinations" {
  description = "List of allowed egress destinations"
  type = list(object({
    cidr     = string
    port     = number
    protocol = string
  }))
  default = []
}

variable "hana_credentials" {
  description = "SAP HANA Cloud credentials"
  type = object({
    host     = string
    port     = string
    user     = string
    password = string
  })
  sensitive = true
  # No defaults for sensitive information
}

variable "datasphere_credentials" {
  description = "SAP DataSphere credentials"
  type = object({
    client_id     = string
    client_secret = string
    auth_url      = string
    token_url     = string
    api_url       = string
  })
  sensitive = true
  # No defaults for sensitive information
}