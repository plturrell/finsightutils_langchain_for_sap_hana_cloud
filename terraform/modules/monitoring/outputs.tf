output "prometheus_server_endpoint" {
  description = "The endpoint for the Prometheus server"
  value       = "http://prometheus-server.${kubernetes_namespace.monitoring.metadata[0].name}.svc.cluster.local"
}

output "grafana_endpoint" {
  description = "The endpoint for Grafana"
  value       = "http://grafana.${kubernetes_namespace.monitoring.metadata[0].name}.svc.cluster.local"
}

output "namespace" {
  description = "The Kubernetes namespace for monitoring"
  value       = kubernetes_namespace.monitoring.metadata[0].name
}