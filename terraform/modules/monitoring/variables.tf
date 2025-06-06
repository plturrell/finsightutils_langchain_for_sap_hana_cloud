variable "namespace" {
  description = "Kubernetes namespace for monitoring stack"
  type        = string
  default     = "monitoring"
}

variable "environment" {
  description = "Deployment environment (staging or production)"
  type        = string
  default     = "staging"
}

variable "prometheus_chart_version" {
  description = "Prometheus Helm chart version"
  type        = string
  default     = "15.10.1"
}

variable "grafana_chart_version" {
  description = "Grafana Helm chart version"
  type        = string
  default     = "6.50.7"
}

variable "prometheus_storage_size" {
  description = "Prometheus server storage size"
  type        = string
  default     = "8Gi"
}

variable "prometheus_retention" {
  description = "Prometheus retention period"
  type        = string
  default     = "15d"
}

variable "alertmanager_storage_size" {
  description = "Alertmanager storage size"
  type        = string
  default     = "2Gi"
}

variable "grafana_storage_size" {
  description = "Grafana storage size"
  type        = string
  default     = "2Gi"
}

variable "grafana_admin_password" {
  description = "Grafana admin password"
  type        = string
  default     = "admin" # Should be overridden in production
  sensitive   = true
}