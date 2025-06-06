output "namespace" {
  description = "The Kubernetes namespace for backups"
  value       = kubernetes_namespace.backup.metadata[0].name
}

output "backup_pvc_name" {
  description = "The name of the PVC used for backups"
  value       = kubernetes_persistent_volume_claim.backup_pvc.metadata[0].name
}

output "backup_job_name" {
  description = "The name of the backup cronjob"
  value       = kubernetes_cron_job_v1.backup_job.metadata[0].name
}