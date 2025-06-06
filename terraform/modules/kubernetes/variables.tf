variable "environment" {
  description = "Deployment environment (staging or production)"
  type        = string
}

variable "namespace" {
  description = "Kubernetes namespace for deployment"
  type        = string
}

variable "replicas" {
  description = "Number of API replicas to deploy"
  type        = number
}

variable "image_repository" {
  description = "Docker image repository for the API"
  type        = string
}

variable "image_tag" {
  description = "Docker image tag for the API"
  type        = string
}

variable "gpu_enabled" {
  description = "Whether to enable GPU resources"
  type        = bool
}

variable "nvidia_gpu_count" {
  description = "Number of NVIDIA GPUs to allocate per pod"
  type        = number
}

variable "memory_limit" {
  description = "Memory limit for API containers"
  type        = string
}

variable "cpu_limit" {
  description = "CPU limit for API containers"
  type        = string
}

variable "memory_request" {
  description = "Memory request for API containers"
  type        = string
}

variable "cpu_request" {
  description = "CPU request for API containers"
  type        = string
}

variable "api_config" {
  description = "Configuration for the API"
  type = object({
    tensorrt_enabled = bool
    batch_size       = number
    timeout_seconds  = number
  })
}

variable "autoscaling" {
  description = "Autoscaling configuration"
  type = object({
    enabled      = bool
    min_replicas = number
    max_replicas = number
    cpu_target   = number
  })
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
}