variable "namespace" {
  description = "Kubernetes namespace to apply network policies to"
  type        = string
}

variable "enable_network_policies" {
  description = "Whether to enable network policies"
  type        = bool
  default     = true
}

variable "frontend_namespace" {
  description = "Namespace where frontend is deployed"
  type        = string
  default     = ""
}

variable "monitoring_namespace" {
  description = "Namespace where monitoring is deployed"
  type        = string
  default     = "monitoring"
}

variable "enable_monitoring" {
  description = "Whether monitoring is enabled"
  type        = bool
  default     = true
}

variable "hana_cidr" {
  description = "CIDR block for SAP HANA Cloud"
  type        = string
  default     = "0.0.0.0/0"  # Should be restricted in production
}

variable "hana_port" {
  description = "Port for SAP HANA Cloud"
  type        = number
  default     = 443
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