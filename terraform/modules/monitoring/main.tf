resource "kubernetes_namespace" "monitoring" {
  metadata {
    name = var.namespace
    
    labels = {
      name = "monitoring"
    }
  }
}

# Helm release for Prometheus
resource "helm_release" "prometheus" {
  name       = "prometheus"
  repository = "https://prometheus-community.github.io/helm-charts"
  chart      = "prometheus"
  namespace  = kubernetes_namespace.monitoring.metadata[0].name
  version    = var.prometheus_chart_version

  set {
    name  = "server.persistentVolume.size"
    value = var.prometheus_storage_size
  }
  
  set {
    name  = "server.retention"
    value = var.prometheus_retention
  }
  
  values = [
    <<-EOT
    alertmanager:
      enabled: true
      persistentVolume:
        enabled: true
        size: ${var.alertmanager_storage_size}
    server:
      global:
        scrape_interval: 15s
        evaluation_interval: 15s
      extraScrapeConfigs: |
        - job_name: 'langchain-hana-api'
          kubernetes_sd_configs:
          - role: service
            namespaces:
              names:
              - langchain-hana-${var.environment}
          relabel_configs:
          - source_labels: [__meta_kubernetes_service_label_app]
            regex: langchain-hana-api
            action: keep
          - source_labels: [__meta_kubernetes_service_port_name]
            regex: http
            action: keep
          metrics_path: /metrics
    EOT
  ]
}

# Helm release for Grafana
resource "helm_release" "grafana" {
  name       = "grafana"
  repository = "https://grafana.github.io/helm-charts"
  chart      = "grafana"
  namespace  = kubernetes_namespace.monitoring.metadata[0].name
  version    = var.grafana_chart_version
  
  depends_on = [helm_release.prometheus]

  set {
    name  = "persistence.enabled"
    value = "true"
  }
  
  set {
    name  = "persistence.size"
    value = var.grafana_storage_size
  }
  
  set {
    name  = "adminPassword"
    value = var.grafana_admin_password
  }
  
  values = [
    <<-EOT
    datasources:
      datasources.yaml:
        apiVersion: 1
        datasources:
        - name: Prometheus
          type: prometheus
          url: http://prometheus-server.${kubernetes_namespace.monitoring.metadata[0].name}.svc.cluster.local
          access: proxy
          isDefault: true
    
    dashboardProviders:
      dashboardproviders.yaml:
        apiVersion: 1
        providers:
        - name: 'default'
          orgId: 1
          folder: ''
          type: file
          disableDeletion: false
          editable: true
          options:
            path: /var/lib/grafana/dashboards/default
    
    dashboards:
      default:
        langchain-hana-api:
          json: |
            {
              "annotations": {
                "list": [
                  {
                    "builtIn": 1,
                    "datasource": "-- Grafana --",
                    "enable": true,
                    "hide": true,
                    "iconColor": "rgba(0, 211, 255, 1)",
                    "name": "Annotations & Alerts",
                    "type": "dashboard"
                  }
                ]
              },
              "editable": true,
              "gnetId": null,
              "graphTooltip": 0,
              "id": 1,
              "links": [],
              "panels": [
                {
                  "aliasColors": {},
                  "bars": false,
                  "dashLength": 10,
                  "dashes": false,
                  "datasource": "Prometheus",
                  "fieldConfig": {
                    "defaults": {
                      "custom": {}
                    },
                    "overrides": []
                  },
                  "fill": 1,
                  "fillGradient": 0,
                  "gridPos": {
                    "h": 8,
                    "w": 12,
                    "x": 0,
                    "y": 0
                  },
                  "hiddenSeries": false,
                  "id": 2,
                  "legend": {
                    "avg": false,
                    "current": false,
                    "max": false,
                    "min": false,
                    "show": true,
                    "total": false,
                    "values": false
                  },
                  "lines": true,
                  "linewidth": 1,
                  "nullPointMode": "null",
                  "options": {
                    "dataLinks": []
                  },
                  "percentage": false,
                  "pointradius": 2,
                  "points": false,
                  "renderer": "flot",
                  "seriesOverrides": [],
                  "spaceLength": 10,
                  "stack": false,
                  "steppedLine": false,
                  "targets": [
                    {
                      "expr": "rate(http_requests_total[5m])",
                      "interval": "",
                      "legendFormat": "",
                      "refId": "A"
                    }
                  ],
                  "thresholds": [],
                  "timeFrom": null,
                  "timeRegions": [],
                  "timeShift": null,
                  "title": "HTTP Request Rate",
                  "tooltip": {
                    "shared": true,
                    "sort": 0,
                    "value_type": "individual"
                  },
                  "type": "graph",
                  "xaxis": {
                    "buckets": null,
                    "mode": "time",
                    "name": null,
                    "show": true,
                    "values": []
                  },
                  "yaxes": [
                    {
                      "format": "short",
                      "label": null,
                      "logBase": 1,
                      "max": null,
                      "min": null,
                      "show": true
                    },
                    {
                      "format": "short",
                      "label": null,
                      "logBase": 1,
                      "max": null,
                      "min": null,
                      "show": true
                    }
                  ],
                  "yaxis": {
                    "align": false,
                    "alignLevel": null
                  }
                },
                {
                  "aliasColors": {},
                  "bars": false,
                  "dashLength": 10,
                  "dashes": false,
                  "datasource": "Prometheus",
                  "fieldConfig": {
                    "defaults": {
                      "custom": {}
                    },
                    "overrides": []
                  },
                  "fill": 1,
                  "fillGradient": 0,
                  "gridPos": {
                    "h": 8,
                    "w": 12,
                    "x": 12,
                    "y": 0
                  },
                  "hiddenSeries": false,
                  "id": 3,
                  "legend": {
                    "avg": false,
                    "current": false,
                    "max": false,
                    "min": false,
                    "show": true,
                    "total": false,
                    "values": false
                  },
                  "lines": true,
                  "linewidth": 1,
                  "nullPointMode": "null",
                  "options": {
                    "dataLinks": []
                  },
                  "percentage": false,
                  "pointradius": 2,
                  "points": false,
                  "renderer": "flot",
                  "seriesOverrides": [],
                  "spaceLength": 10,
                  "stack": false,
                  "steppedLine": false,
                  "targets": [
                    {
                      "expr": "histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))",
                      "interval": "",
                      "legendFormat": "",
                      "refId": "A"
                    }
                  ],
                  "thresholds": [],
                  "timeFrom": null,
                  "timeRegions": [],
                  "timeShift": null,
                  "title": "HTTP Request Duration (95th percentile)",
                  "tooltip": {
                    "shared": true,
                    "sort": 0,
                    "value_type": "individual"
                  },
                  "type": "graph",
                  "xaxis": {
                    "buckets": null,
                    "mode": "time",
                    "name": null,
                    "show": true,
                    "values": []
                  },
                  "yaxes": [
                    {
                      "format": "s",
                      "label": null,
                      "logBase": 1,
                      "max": null,
                      "min": null,
                      "show": true
                    },
                    {
                      "format": "short",
                      "label": null,
                      "logBase": 1,
                      "max": null,
                      "min": null,
                      "show": true
                    }
                  ],
                  "yaxis": {
                    "align": false,
                    "alignLevel": null
                  }
                }
              ],
              "schemaVersion": 25,
              "style": "dark",
              "tags": [],
              "templating": {
                "list": []
              },
              "time": {
                "from": "now-6h",
                "to": "now"
              },
              "timepicker": {
                "refresh_intervals": [
                  "5s",
                  "10s",
                  "30s",
                  "1m",
                  "5m",
                  "15m",
                  "30m",
                  "1h",
                  "2h",
                  "1d"
                ]
              },
              "timezone": "",
              "title": "LangChain HANA API Dashboard",
              "uid": "langchain-hana-api",
              "version": 1
            }
    EOT
  ]
}