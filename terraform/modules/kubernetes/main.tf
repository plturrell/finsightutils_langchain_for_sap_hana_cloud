resource "kubernetes_namespace" "langchain_hana" {
  metadata {
    name = var.namespace
    
    labels = {
      environment = var.environment
      app         = "langchain-hana"
    }
  }
}

# ConfigMap for non-sensitive configuration
resource "kubernetes_config_map" "api_config" {
  metadata {
    name      = "langchain-hana-api-config"
    namespace = kubernetes_namespace.langchain_hana.metadata[0].name
  }

  data = {
    "TENSORRT_ENABLED"       = tostring(var.api_config.tensorrt_enabled)
    "EMBEDDING_BATCH_SIZE"   = tostring(var.api_config.batch_size)
    "REQUEST_TIMEOUT"        = tostring(var.api_config.timeout_seconds)
    "ENVIRONMENT"            = var.environment
    "GPU_ENABLED"            = tostring(var.gpu_enabled)
  }
}

# Secret for sensitive configuration
resource "kubernetes_secret" "api_secrets" {
  metadata {
    name      = "langchain-hana-api-secrets"
    namespace = kubernetes_namespace.langchain_hana.metadata[0].name
  }

  data = {
    # These are placeholders and should be populated via CI/CD or externally
    "HANA_HOST"           = "placeholder"
    "HANA_PORT"           = "placeholder"
    "HANA_USER"           = "placeholder"
    "HANA_PASSWORD"       = "placeholder"
    "API_KEY"             = "placeholder"
  }

  type = "Opaque"
}

# Deployment for the API
resource "kubernetes_deployment" "api" {
  metadata {
    name      = "langchain-hana-api"
    namespace = kubernetes_namespace.langchain_hana.metadata[0].name
    
    labels = {
      app         = "langchain-hana-api"
      environment = var.environment
    }
  }

  spec {
    replicas = var.autoscaling.enabled ? var.autoscaling.min_replicas : var.replicas

    selector {
      match_labels = {
        app = "langchain-hana-api"
      }
    }

    template {
      metadata {
        labels = {
          app         = "langchain-hana-api"
          environment = var.environment
        }
      }

      spec {
        container {
          name  = "api"
          image = "${var.image_repository}:${var.image_tag}"
          
          port {
            container_port = 8000
            name           = "http"
          }
          
          env_from {
            config_map_ref {
              name = kubernetes_config_map.api_config.metadata[0].name
            }
          }
          
          env_from {
            secret_ref {
              name = kubernetes_secret.api_secrets.metadata[0].name
            }
          }
          
          env_from {
            secret_ref {
              name = kubernetes_secret.hana_credentials.metadata[0].name
            }
          }
          
          env_from {
            secret_ref {
              name = kubernetes_secret.datasphere_credentials.metadata[0].name
            }
          }
          
          resources {
            limits = {
              "cpu"    = var.cpu_limit
              "memory" = var.memory_limit
              "nvidia.com/gpu" = var.gpu_enabled ? var.nvidia_gpu_count : null
            }
            requests = {
              "cpu"    = var.cpu_request
              "memory" = var.memory_request
            }
          }
          
          liveness_probe {
            http_get {
              path = "/health"
              port = "http"
            }
            initial_delay_seconds = 30
            period_seconds        = 10
          }
          
          readiness_probe {
            http_get {
              path = "/health"
              port = "http"
            }
            initial_delay_seconds = 5
            period_seconds        = 5
          }
        }
        
        # Add node selector for GPU nodes if needed
        dynamic "node_selector" {
          for_each = var.gpu_enabled ? [1] : []
          content {
            "accelerator" = "nvidia-gpu"
          }
        }
      }
    }
  }
}

# Service for the API
resource "kubernetes_service" "api" {
  metadata {
    name      = "langchain-hana-api"
    namespace = kubernetes_namespace.langchain_hana.metadata[0].name
    
    labels = {
      app         = "langchain-hana-api"
      environment = var.environment
    }
  }
  
  spec {
    selector = {
      app = "langchain-hana-api"
    }
    
    port {
      port        = 80
      target_port = 8000
      protocol    = "TCP"
      name        = "http"
    }
    
    type = "ClusterIP"
  }
}

# Horizontal Pod Autoscaler
resource "kubernetes_horizontal_pod_autoscaler" "api" {
  count = var.autoscaling.enabled ? 1 : 0
  
  metadata {
    name      = "langchain-hana-api"
    namespace = kubernetes_namespace.langchain_hana.metadata[0].name
  }
  
  spec {
    scale_target_ref {
      api_version = "apps/v1"
      kind        = "Deployment"
      name        = kubernetes_deployment.api.metadata[0].name
    }
    
    min_replicas = var.autoscaling.min_replicas
    max_replicas = var.autoscaling.max_replicas
    
    metric {
      type = "Resource"
      resource {
        name = "cpu"
        target {
          type                = "Utilization"
          average_utilization = var.autoscaling.cpu_target
        }
      }
    }
  }
}