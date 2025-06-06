resource "kubernetes_namespace" "backup" {
  metadata {
    name = var.namespace
    
    labels = {
      environment = var.environment
      app         = "langchain-hana-backup"
    }
  }
}

# Persistent Volume Claim for backups
resource "kubernetes_persistent_volume_claim" "backup_pvc" {
  metadata {
    name      = "backup-pvc"
    namespace = kubernetes_namespace.backup.metadata[0].name
    
    labels = {
      app         = "langchain-hana-backup"
      environment = var.environment
    }
  }
  
  spec {
    access_modes = ["ReadWriteOnce"]
    
    resources {
      requests = {
        storage = var.backup_storage_size
      }
    }
    
    storage_class_name = var.storage_class
  }
}

# ConfigMap for backup scripts
resource "kubernetes_config_map" "backup_scripts" {
  metadata {
    name      = "backup-scripts"
    namespace = kubernetes_namespace.backup.metadata[0].name
  }

  data = {
    "backup.sh" = <<-EOT
      #!/bin/bash
      # Backup script for SAP HANA Cloud data
      
      set -e
      
      # Configuration
      BACKUP_DIR="/backups"
      TIMESTAMP=$(date +%Y%m%d_%H%M%S)
      HANA_HOST="${var.hana_host}"
      HANA_PORT="${var.hana_port}"
      HANA_USER="${var.hana_user}"
      
      # Create backup directory
      mkdir -p "$BACKUP_DIR/$TIMESTAMP"
      
      # Run backup
      echo "Starting backup at $TIMESTAMP"
      
      # Connect to HANA and perform backup
      echo "Connecting to HANA at $HANA_HOST:$HANA_PORT"
      hdbsql -n "$HANA_HOST:$HANA_PORT" -u "$HANA_USER" -p "$HANA_PASSWORD" \
        -i 00 "BACKUP DATA USING FILE ('$BACKUP_DIR/$TIMESTAMP/backup')"
      
      # Create backup manifest
      echo "Creating backup manifest"
      cat > "$BACKUP_DIR/$TIMESTAMP/manifest.json" << EOF
      {
        "timestamp": "$TIMESTAMP",
        "host": "$HANA_HOST",
        "port": "$HANA_PORT",
        "database": "SYSTEMDB",
        "type": "full",
        "status": "completed"
      }
      EOF
      
      # Compress backup
      echo "Compressing backup"
      tar -czf "$BACKUP_DIR/$TIMESTAMP.tar.gz" -C "$BACKUP_DIR" "$TIMESTAMP"
      
      # Clean up temporary files
      rm -rf "$BACKUP_DIR/$TIMESTAMP"
      
      echo "Backup completed successfully"
    EOT
    
    "restore.sh" = <<-EOT
      #!/bin/bash
      # Restore script for SAP HANA Cloud data
      
      set -e
      
      # Configuration
      BACKUP_DIR="/backups"
      HANA_HOST="${var.hana_host}"
      HANA_PORT="${var.hana_port}"
      HANA_USER="${var.hana_user}"
      
      # Check for backup file argument
      if [ -z "$1" ]; then
        echo "Error: Backup file name required"
        echo "Usage: $0 <backup_file>"
        exit 1
      fi
      
      BACKUP_FILE="$1"
      TEMP_DIR=$(mktemp -d)
      
      # Extract backup
      echo "Extracting backup $BACKUP_FILE"
      tar -xzf "$BACKUP_DIR/$BACKUP_FILE" -C "$TEMP_DIR"
      
      # Find backup directory
      BACKUP_TIMESTAMP=$(ls "$TEMP_DIR")
      
      # Connect to HANA and perform restore
      echo "Connecting to HANA at $HANA_HOST:$HANA_PORT"
      hdbsql -n "$HANA_HOST:$HANA_PORT" -u "$HANA_USER" -p "$HANA_PASSWORD" \
        -i 00 "RECOVER DATABASE UNTIL TIMESTAMP '$BACKUP_TIMESTAMP' USING FILE ('$TEMP_DIR/$BACKUP_TIMESTAMP/backup')"
      
      # Clean up
      rm -rf "$TEMP_DIR"
      
      echo "Restore completed successfully"
    EOT
  }
}

# CronJob for scheduled backups
resource "kubernetes_cron_job_v1" "backup_job" {
  metadata {
    name      = "hana-backup"
    namespace = kubernetes_namespace.backup.metadata[0].name
  }

  spec {
    schedule                      = var.backup_schedule
    concurrency_policy            = "Forbid"
    successful_jobs_history_limit = 5
    failed_jobs_history_limit     = 3
    
    job_template {
      metadata {
        labels = {
          app         = "langchain-hana-backup"
          environment = var.environment
        }
      }
      
      spec {
        template {
          metadata {
            labels = {
              app         = "langchain-hana-backup"
              environment = var.environment
            }
          }
          
          spec {
            container {
              name    = "backup"
              image   = var.backup_image
              command = ["/scripts/backup.sh"]
              
              env {
                name = "HANA_PASSWORD"
                value_from {
                  secret_key_ref {
                    name = kubernetes_secret.backup_secrets.metadata[0].name
                    key  = "HANA_PASSWORD"
                  }
                }
              }
              
              volume_mount {
                name       = "backup-volume"
                mount_path = "/backups"
              }
              
              volume_mount {
                name       = "scripts-volume"
                mount_path = "/scripts"
              }
              
              resources {
                limits = {
                  cpu    = "1"
                  memory = "1Gi"
                }
                requests = {
                  cpu    = "200m"
                  memory = "512Mi"
                }
              }
            }
            
            volume {
              name = "backup-volume"
              persistent_volume_claim {
                claim_name = kubernetes_persistent_volume_claim.backup_pvc.metadata[0].name
              }
            }
            
            volume {
              name = "scripts-volume"
              config_map {
                name = kubernetes_config_map.backup_scripts.metadata[0].name
                default_mode = "0755"
              }
            }
            
            restart_policy = "OnFailure"
          }
        }
      }
    }
  }
}

# Secret for backup credentials
resource "kubernetes_secret" "backup_secrets" {
  metadata {
    name      = "backup-secrets"
    namespace = kubernetes_namespace.backup.metadata[0].name
  }

  data = {
    "HANA_PASSWORD" = var.hana_password
  }

  type = "Opaque"
}