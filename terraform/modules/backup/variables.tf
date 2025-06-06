variable "namespace" {
  description = "Kubernetes namespace for backup resources"
  type        = string
  default     = "langchain-hana-backup"
}

variable "environment" {
  description = "Deployment environment (staging or production)"
  type        = string
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
  description = "Storage class for backup PVC"
  type        = string
  default     = "standard"
}

variable "backup_image" {
  description = "Docker image for backup jobs"
  type        = string
  default     = "saplabs/hanatools:latest"
}

variable "hana_host" {
  description = "SAP HANA Cloud host"
  type        = string
}

variable "hana_port" {
  description = "SAP HANA Cloud port"
  type        = string
}

variable "hana_user" {
  description = "SAP HANA Cloud user"
  type        = string
}

variable "hana_password" {
  description = "SAP HANA Cloud password"
  type        = string
  sensitive   = true
}