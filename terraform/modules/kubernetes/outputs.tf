output "namespace" {
  description = "The Kubernetes namespace created"
  value       = kubernetes_namespace.langchain_hana.metadata[0].name
}

output "api_service_name" {
  description = "The name of the API service"
  value       = kubernetes_service.api.metadata[0].name
}

output "api_service_cluster_ip" {
  description = "The cluster IP of the API service"
  value       = kubernetes_service.api.spec[0].cluster_ip
}

output "deployment_name" {
  description = "The name of the API deployment"
  value       = kubernetes_deployment.api.metadata[0].name
}

output "config_map_name" {
  description = "The name of the config map"
  value       = kubernetes_config_map.api_config.metadata[0].name
}

output "secret_name" {
  description = "The name of the secret"
  value       = kubernetes_secret.api_secrets.metadata[0].name
}