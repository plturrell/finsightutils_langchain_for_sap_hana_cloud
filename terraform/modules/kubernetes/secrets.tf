resource "kubernetes_secret" "hana_credentials" {
  metadata {
    name      = "hana-credentials"
    namespace = kubernetes_namespace.langchain_hana.metadata[0].name
    
    labels = {
      app         = "langchain-hana-api"
      environment = var.environment
    }
  }

  data = {
    "HANA_HOST"     = var.hana_credentials.host
    "HANA_PORT"     = var.hana_credentials.port
    "HANA_USER"     = var.hana_credentials.user
    "HANA_PASSWORD" = var.hana_credentials.password
  }

  type = "Opaque"
}

resource "kubernetes_secret" "datasphere_credentials" {
  metadata {
    name      = "datasphere-credentials"
    namespace = kubernetes_namespace.langchain_hana.metadata[0].name
    
    labels = {
      app         = "langchain-hana-api"
      environment = var.environment
    }
  }

  data = {
    "DATASPHERE_CLIENT_ID"     = var.datasphere_credentials.client_id
    "DATASPHERE_CLIENT_SECRET" = var.datasphere_credentials.client_secret
    "DATASPHERE_AUTH_URL"      = var.datasphere_credentials.auth_url
    "DATASPHERE_TOKEN_URL"     = var.datasphere_credentials.token_url
    "DATASPHERE_API_URL"       = var.datasphere_credentials.api_url
  }

  type = "Opaque"
}