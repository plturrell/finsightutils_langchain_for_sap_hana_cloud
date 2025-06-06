# Network Policies for enhanced security

# Default deny all ingress and egress
resource "kubernetes_network_policy" "default_deny" {
  count = var.enable_network_policies ? 1 : 0

  metadata {
    name      = "default-deny"
    namespace = var.namespace
  }

  spec {
    pod_selector {}
    
    policy_types = ["Ingress", "Egress"]
  }
}

# Allow API ingress from specified sources
resource "kubernetes_network_policy" "api_ingress" {
  count = var.enable_network_policies ? 1 : 0

  metadata {
    name      = "api-ingress"
    namespace = var.namespace
  }

  spec {
    pod_selector {
      match_labels = {
        app = "langchain-hana-api"
      }
    }
    
    ingress {
      from {
        namespace_selector {
          match_labels = {
            name = "kube-system"
          }
        }
      }
      
      from {
        namespace_selector {
          match_labels = {
            name = "ingress-nginx"
          }
        }
      }
      
      # Allow ingress from frontend namespace if specified
      dynamic "from" {
        for_each = var.frontend_namespace != "" ? [1] : []
        content {
          namespace_selector {
            match_labels = {
              name = var.frontend_namespace
            }
          }
        }
      }
      
      # Allow custom ingress sources
      dynamic "from" {
        for_each = var.allowed_ingress_sources
        content {
          ip_block {
            cidr = from.value
          }
        }
      }
      
      ports {
        port     = 8000
        protocol = "TCP"
      }
    }
    
    policy_types = ["Ingress"]
  }
}

# Allow API egress to necessary services
resource "kubernetes_network_policy" "api_egress" {
  count = var.enable_network_policies ? 1 : 0

  metadata {
    name      = "api-egress"
    namespace = var.namespace
  }

  spec {
    pod_selector {
      match_labels = {
        app = "langchain-hana-api"
      }
    }
    
    # Allow DNS resolution
    egress {
      to {
        namespace_selector {
          match_labels = {
            name = "kube-system"
          }
        }
        pod_selector {
          match_labels = {
            k8s-app = "kube-dns"
          }
        }
      }
      
      ports {
        port     = 53
        protocol = "UDP"
      }
      
      ports {
        port     = 53
        protocol = "TCP"
      }
    }
    
    # Allow egress to HANA Cloud
    egress {
      to {
        ip_block {
          cidr = var.hana_cidr
        }
      }
      
      ports {
        port     = var.hana_port
        protocol = "TCP"
      }
    }
    
    # Allow egress to DataSphere
    egress {
      to {
        ip_block {
          cidr = var.datasphere_cidr
        }
      }
      
      ports {
        port     = 443
        protocol = "TCP"
      }
    }
    
    # Allow egress to OAuth server
    egress {
      to {
        ip_block {
          cidr = var.oauth_cidr
        }
      }
      
      ports {
        port     = 443
        protocol = "TCP"
      }
    }
    
    # Allow custom egress destinations
    dynamic "egress" {
      for_each = var.allowed_egress_destinations
      content {
        to {
          ip_block {
            cidr = egress.value.cidr
          }
        }
        
        ports {
          port     = egress.value.port
          protocol = egress.value.protocol
        }
      }
    }
    
    policy_types = ["Egress"]
  }
}

# Allow monitoring ingress
resource "kubernetes_network_policy" "monitoring_ingress" {
  count = var.enable_network_policies && var.enable_monitoring ? 1 : 0

  metadata {
    name      = "monitoring-ingress"
    namespace = var.namespace
  }

  spec {
    pod_selector {
      match_labels = {
        app = "langchain-hana-api"
      }
    }
    
    ingress {
      from {
        namespace_selector {
          match_labels = {
            name = var.monitoring_namespace
          }
        }
      }
      
      ports {
        port     = 8000
        protocol = "TCP"
      }
    }
    
    policy_types = ["Ingress"]
  }
}