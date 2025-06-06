#!/usr/bin/env python3
"""
Script to convert Kubernetes YAML manifests to Terraform configuration.
This helps when migrating from kubectl-based deployments to Terraform.
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
import re

def indent(text, spaces=2):
    """Indent text by the specified number of spaces."""
    if not text:
        return ""
    lines = text.split('\n')
    return '\n'.join(' ' * spaces + line for line in lines)

def sanitize_resource_name(name):
    """Convert resource name to a valid Terraform identifier."""
    # Replace invalid characters with underscores
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    # Ensure it starts with a letter or underscore
    if sanitized and sanitized[0].isdigit():
        sanitized = f"_{sanitized}"
    return sanitized

def get_resource_name(kind, name):
    """Get a unique Terraform resource name."""
    kind_lower = kind.lower()
    name_sanitized = sanitize_resource_name(name)
    return f"{kind_lower}_{name_sanitized}"

def yaml_to_terraform(yaml_content):
    """Convert YAML content to Terraform configuration."""
    try:
        documents = list(yaml.safe_load_all(yaml_content))
        terraform_configs = []
        
        for doc in documents:
            if not doc:
                continue
                
            kind = doc.get('kind')
            if not kind:
                continue
                
            metadata = doc.get('metadata', {})
            name = metadata.get('name', 'unnamed')
            
            resource_name = get_resource_name(kind, name)
            terraform_type = kind_to_terraform_type(kind)
            
            if not terraform_type:
                terraform_configs.append(f"# Unsupported resource type: {kind} (name: {name})")
                continue
            
            terraform_config = generate_terraform_resource(terraform_type, resource_name, doc)
            terraform_configs.append(terraform_config)
        
        return '\n\n'.join(terraform_configs)
    except Exception as e:
        return f"# Error converting YAML to Terraform: {str(e)}"

def kind_to_terraform_type(kind):
    """Map Kubernetes resource kind to Terraform resource type."""
    kind_map = {
        'Namespace': 'kubernetes_namespace',
        'Service': 'kubernetes_service',
        'Deployment': 'kubernetes_deployment',
        'StatefulSet': 'kubernetes_stateful_set',
        'ConfigMap': 'kubernetes_config_map',
        'Secret': 'kubernetes_secret',
        'PersistentVolumeClaim': 'kubernetes_persistent_volume_claim',
        'PersistentVolume': 'kubernetes_persistent_volume',
        'Ingress': 'kubernetes_ingress',
        'NetworkPolicy': 'kubernetes_network_policy',
        'ServiceAccount': 'kubernetes_service_account',
        'Role': 'kubernetes_role',
        'RoleBinding': 'kubernetes_role_binding',
        'ClusterRole': 'kubernetes_cluster_role',
        'ClusterRoleBinding': 'kubernetes_cluster_role_binding',
        'HorizontalPodAutoscaler': 'kubernetes_horizontal_pod_autoscaler',
        'CronJob': 'kubernetes_cron_job_v1',
        'Job': 'kubernetes_job'
    }
    return kind_map.get(kind)

def generate_terraform_resource(resource_type, resource_name, data):
    """Generate Terraform resource configuration."""
    result = [f'resource "{resource_type}" "{resource_name}" {{']
    
    # Process metadata
    metadata = data.get('metadata', {})
    result.append('  metadata {')
    
    if 'name' in metadata:
        result.append(f'    name = "{metadata["name"]}"')
    
    if 'namespace' in metadata:
        # Use a variable reference for namespace
        result.append('    namespace = var.namespace')
    
    if 'labels' in metadata and metadata['labels']:
        result.append('    labels = {')
        for k, v in metadata['labels'].items():
            result.append(f'      {k} = "{v}"')
        result.append('    }')
    
    if 'annotations' in metadata and metadata['annotations']:
        result.append('    annotations = {')
        for k, v in metadata['annotations'].items():
            result.append(f'      "{k}" = "{v}"')
        result.append('    }')
    
    result.append('  }')
    
    # Process spec
    if 'spec' in data:
        spec_tf = process_spec(resource_type, data['spec'])
        result.append(spec_tf)
    
    # Process data (for ConfigMap and Secret)
    if 'data' in data and resource_type in ['kubernetes_config_map', 'kubernetes_secret']:
        result.append('  data = {')
        for k, v in data['data'].items():
            # Handle multiline strings
            if isinstance(v, str) and '\n' in v:
                result.append(f'    {k} = <<-EOT')
                result.append(indent(v, 6))
                result.append('    EOT')
            else:
                result.append(f'    {k} = "{v}"')
        result.append('  }')
    
    # Add type for secrets
    if resource_type == 'kubernetes_secret' and 'type' in data:
        result.append(f'  type = "{data["type"]}"')
    
    result.append('}')
    
    return '\n'.join(result)

def process_spec(resource_type, spec):
    """Process resource spec."""
    if resource_type == 'kubernetes_deployment':
        return process_deployment_spec(spec)
    elif resource_type == 'kubernetes_service':
        return process_service_spec(spec)
    elif resource_type == 'kubernetes_persistent_volume_claim':
        return process_pvc_spec(spec)
    elif resource_type == 'kubernetes_config_map':
        return ""  # ConfigMap data is handled separately
    elif resource_type == 'kubernetes_secret':
        return ""  # Secret data is handled separately
    elif resource_type == 'kubernetes_namespace':
        return ""  # Namespace doesn't have a spec
    elif resource_type == 'kubernetes_ingress':
        return process_ingress_spec(spec)
    elif resource_type == 'kubernetes_horizontal_pod_autoscaler':
        return process_hpa_spec(spec)
    elif resource_type == 'kubernetes_network_policy':
        return process_network_policy_spec(spec)
    elif resource_type == 'kubernetes_cron_job_v1':
        return process_cronjob_spec(spec)
    else:
        return f"  # Spec processing not implemented for {resource_type}\n  # spec = {spec}"

def process_deployment_spec(spec):
    """Process Deployment spec."""
    result = ['  spec {']
    
    # Replicas - use a variable
    if 'replicas' in spec:
        result.append('    replicas = var.replicas')
    
    # Selector
    if 'selector' in spec:
        result.append('    selector {')
        if 'matchLabels' in spec['selector']:
            result.append('      match_labels = {')
            for k, v in spec['selector']['matchLabels'].items():
                result.append(f'        {k} = "{v}"')
            result.append('      }')
        result.append('    }')
    
    # Template
    if 'template' in spec:
        result.append('    template {')
        
        # Template metadata
        if 'metadata' in spec['template']:
            result.append('      metadata {')
            
            if 'labels' in spec['template']['metadata']:
                result.append('        labels = {')
                for k, v in spec['template']['metadata']['labels'].items():
                    result.append(f'          {k} = "{v}"')
                result.append('        }')
            
            if 'annotations' in spec['template']['metadata']:
                result.append('        annotations = {')
                for k, v in spec['template']['metadata']['annotations'].items():
                    result.append(f'          "{k}" = "{v}"')
                result.append('        }')
            
            result.append('      }')
        
        # Template spec
        if 'spec' in spec['template']:
            result.append('      spec {')
            
            # Containers
            if 'containers' in spec['template']['spec']:
                for container in spec['template']['spec']['containers']:
                    result.append('        container {')
                    result.append(f'          name = "{container["name"]}"')
                    
                    # Image
                    if 'image' in container:
                        # Use variable references for image and tag
                        result.append('          image = "${var.image_repository}:${var.image_tag}"')
                    
                    # Command
                    if 'command' in container:
                        result.append('          command = [')
                        for cmd in container['command']:
                            result.append(f'            "{cmd}",')
                        result.append('          ]')
                    
                    # Args
                    if 'args' in container:
                        result.append('          args = [')
                        for arg in container['args']:
                            result.append(f'            "{arg}",')
                        result.append('          ]')
                    
                    # Ports
                    if 'ports' in container:
                        for port in container['ports']:
                            result.append('          port {')
                            if 'containerPort' in port:
                                result.append(f'            container_port = {port["containerPort"]}')
                            if 'name' in port:
                                result.append(f'            name = "{port["name"]}"')
                            if 'protocol' in port:
                                result.append(f'            protocol = "{port["protocol"]}"')
                            result.append('          }')
                    
                    # Environment variables
                    if 'env' in container:
                        for env in container['env']:
                            result.append('          env {')
                            result.append(f'            name = "{env["name"]}"')
                            
                            if 'value' in env:
                                result.append(f'            value = "{env["value"]}"')
                            elif 'valueFrom' in env:
                                result.append('            value_from {')
                                
                                if 'configMapKeyRef' in env['valueFrom']:
                                    result.append('              config_map_key_ref {')
                                    ref = env['valueFrom']['configMapKeyRef']
                                    result.append(f'                name = "{ref["name"]}"')
                                    result.append(f'                key = "{ref["key"]}"')
                                    result.append('              }')
                                
                                if 'secretKeyRef' in env['valueFrom']:
                                    result.append('              secret_key_ref {')
                                    ref = env['valueFrom']['secretKeyRef']
                                    result.append(f'                name = "{ref["name"]}"')
                                    result.append(f'                key = "{ref["key"]}"')
                                    result.append('              }')
                                
                                result.append('            }')
                            
                            result.append('          }')
                    
                    # Environment from
                    if 'envFrom' in container:
                        for env_from in container['envFrom']:
                            if 'configMapRef' in env_from:
                                result.append('          env_from {')
                                result.append('            config_map_ref {')
                                result.append(f'              name = "{env_from["configMapRef"]["name"]}"')
                                result.append('            }')
                                result.append('          }')
                            
                            if 'secretRef' in env_from:
                                result.append('          env_from {')
                                result.append('            secret_ref {')
                                result.append(f'              name = "{env_from["secretRef"]["name"]}"')
                                result.append('            }')
                                result.append('          }')
                    
                    # Resources
                    if 'resources' in container:
                        result.append('          resources {')
                        
                        if 'limits' in container['resources']:
                            result.append('            limits = {')
                            for k, v in container['resources']['limits'].items():
                                if k == 'nvidia.com/gpu':
                                    result.append('              "nvidia.com/gpu" = var.gpu_enabled ? var.nvidia_gpu_count : null')
                                else:
                                    result.append(f'              "{k}" = "{v}"')
                            result.append('            }')
                        
                        if 'requests' in container['resources']:
                            result.append('            requests = {')
                            for k, v in container['resources']['requests'].items():
                                result.append(f'              "{k}" = "{v}"')
                            result.append('            }')
                        
                        result.append('          }')
                    
                    # Volume mounts
                    if 'volumeMounts' in container:
                        for mount in container['volumeMounts']:
                            result.append('          volume_mount {')
                            result.append(f'            name = "{mount["name"]}"')
                            result.append(f'            mount_path = "{mount["mountPath"]}"')
                            if 'subPath' in mount:
                                result.append(f'            sub_path = "{mount["subPath"]}"')
                            if 'readOnly' in mount:
                                result.append(f'            read_only = {str(mount["readOnly"]).lower()}')
                            result.append('          }')
                    
                    # Readiness probe
                    if 'readinessProbe' in container:
                        probe = container['readinessProbe']
                        result.append('          readiness_probe {')
                        
                        if 'httpGet' in probe:
                            result.append('            http_get {')
                            http = probe['httpGet']
                            if 'path' in http:
                                result.append(f'              path = "{http["path"]}"')
                            if 'port' in http:
                                if isinstance(http['port'], int):
                                    result.append(f'              port = {http["port"]}')
                                else:
                                    result.append(f'              port = "{http["port"]}"')
                            if 'scheme' in http:
                                result.append(f'              scheme = "{http["scheme"]}"')
                            result.append('            }')
                        
                        if 'tcpSocket' in probe:
                            result.append('            tcp_socket {')
                            tcp = probe['tcpSocket']
                            if 'port' in tcp:
                                if isinstance(tcp['port'], int):
                                    result.append(f'              port = {tcp["port"]}')
                                else:
                                    result.append(f'              port = "{tcp["port"]}"')
                            result.append('            }')
                        
                        if 'initialDelaySeconds' in probe:
                            result.append(f'            initial_delay_seconds = {probe["initialDelaySeconds"]}')
                        if 'periodSeconds' in probe:
                            result.append(f'            period_seconds = {probe["periodSeconds"]}')
                        if 'timeoutSeconds' in probe:
                            result.append(f'            timeout_seconds = {probe["timeoutSeconds"]}')
                        if 'successThreshold' in probe:
                            result.append(f'            success_threshold = {probe["successThreshold"]}')
                        if 'failureThreshold' in probe:
                            result.append(f'            failure_threshold = {probe["failureThreshold"]}')
                        
                        result.append('          }')
                    
                    # Liveness probe
                    if 'livenessProbe' in container:
                        probe = container['livenessProbe']
                        result.append('          liveness_probe {')
                        
                        if 'httpGet' in probe:
                            result.append('            http_get {')
                            http = probe['httpGet']
                            if 'path' in http:
                                result.append(f'              path = "{http["path"]}"')
                            if 'port' in http:
                                if isinstance(http['port'], int):
                                    result.append(f'              port = {http["port"]}')
                                else:
                                    result.append(f'              port = "{http["port"]}"')
                            if 'scheme' in http:
                                result.append(f'              scheme = "{http["scheme"]}"')
                            result.append('            }')
                        
                        if 'tcpSocket' in probe:
                            result.append('            tcp_socket {')
                            tcp = probe['tcpSocket']
                            if 'port' in tcp:
                                if isinstance(tcp['port'], int):
                                    result.append(f'              port = {tcp["port"]}')
                                else:
                                    result.append(f'              port = "{tcp["port"]}"')
                            result.append('            }')
                        
                        if 'initialDelaySeconds' in probe:
                            result.append(f'            initial_delay_seconds = {probe["initialDelaySeconds"]}')
                        if 'periodSeconds' in probe:
                            result.append(f'            period_seconds = {probe["periodSeconds"]}')
                        if 'timeoutSeconds' in probe:
                            result.append(f'            timeout_seconds = {probe["timeoutSeconds"]}')
                        if 'successThreshold' in probe:
                            result.append(f'            success_threshold = {probe["successThreshold"]}')
                        if 'failureThreshold' in probe:
                            result.append(f'            failure_threshold = {probe["failureThreshold"]}')
                        
                        result.append('          }')
                    
                    result.append('        }')
            
            # Volumes
            if 'volumes' in spec['template']['spec']:
                for volume in spec['template']['spec']['volumes']:
                    result.append('        volume {')
                    result.append(f'          name = "{volume["name"]}"')
                    
                    if 'configMap' in volume:
                        result.append('          config_map {')
                        result.append(f'            name = "{volume["configMap"]["name"]}"')
                        if 'defaultMode' in volume['configMap']:
                            result.append(f'            default_mode = "{volume["configMap"]["defaultMode"]}"')
                        result.append('          }')
                    
                    if 'secret' in volume:
                        result.append('          secret {')
                        result.append(f'            secret_name = "{volume["secret"]["secretName"]}"')
                        if 'defaultMode' in volume['secret']:
                            result.append(f'            default_mode = "{volume["secret"]["defaultMode"]}"')
                        result.append('          }')
                    
                    if 'persistentVolumeClaim' in volume:
                        result.append('          persistent_volume_claim {')
                        result.append(f'            claim_name = "{volume["persistentVolumeClaim"]["claimName"]}"')
                        if 'readOnly' in volume['persistentVolumeClaim']:
                            result.append(f'            read_only = {str(volume["persistentVolumeClaim"]["readOnly"]).lower()}')
                        result.append('          }')
                    
                    if 'emptyDir' in volume:
                        result.append('          empty_dir {')
                        if 'medium' in volume['emptyDir']:
                            result.append(f'            medium = "{volume["emptyDir"]["medium"]}"')
                        if 'sizeLimit' in volume['emptyDir']:
                            result.append(f'            size_limit = "{volume["emptyDir"]["sizeLimit"]}"')
                        result.append('          }')
                    
                    if 'hostPath' in volume:
                        result.append('          host_path {')
                        result.append(f'            path = "{volume["hostPath"]["path"]}"')
                        if 'type' in volume['hostPath']:
                            result.append(f'            type = "{volume["hostPath"]["type"]}"')
                        result.append('          }')
                    
                    result.append('        }')
            
            # Node selector
            if 'nodeSelector' in spec['template']['spec']:
                result.append('        dynamic "node_selector" {')
                result.append('          for_each = var.gpu_enabled ? [1] : []')
                result.append('          content {')
                for k, v in spec['template']['spec']['nodeSelector'].items():
                    result.append(f'            "{k}" = "{v}"')
                result.append('          }')
                result.append('        }')
            
            # Restart policy
            if 'restartPolicy' in spec['template']['spec']:
                result.append(f'        restart_policy = "{spec["template"]["spec"]["restartPolicy"]}"')
            
            # Service account
            if 'serviceAccountName' in spec['template']['spec']:
                result.append(f'        service_account_name = "{spec["template"]["spec"]["serviceAccountName"]}"')
            
            result.append('      }')
        
        result.append('    }')
    
    result.append('  }')
    
    return '\n'.join(result)

def process_service_spec(spec):
    """Process Service spec."""
    result = ['  spec {']
    
    # Selector
    if 'selector' in spec:
        result.append('    selector = {')
        for k, v in spec['selector'].items():
            result.append(f'      {k} = "{v}"')
        result.append('    }')
    
    # Ports
    if 'ports' in spec:
        for port in spec['ports']:
            result.append('    port {')
            if 'port' in port:
                result.append(f'      port = {port["port"]}')
            if 'targetPort' in port:
                if isinstance(port['targetPort'], int):
                    result.append(f'      target_port = {port["targetPort"]}')
                else:
                    result.append(f'      target_port = "{port["targetPort"]}"')
            if 'protocol' in port:
                result.append(f'      protocol = "{port["protocol"]}"')
            if 'name' in port:
                result.append(f'      name = "{port["name"]}"')
            result.append('    }')
    
    # Type
    if 'type' in spec:
        result.append(f'    type = "{spec["type"]}"')
    
    # ClusterIP
    if 'clusterIP' in spec:
        result.append(f'    cluster_ip = "{spec["clusterIP"]}"')
    
    # External name
    if 'externalName' in spec:
        result.append(f'    external_name = "{spec["externalName"]}"')
    
    # Session affinity
    if 'sessionAffinity' in spec:
        result.append(f'    session_affinity = "{spec["sessionAffinity"]}"')
    
    result.append('  }')
    
    return '\n'.join(result)

def process_pvc_spec(spec):
    """Process PersistentVolumeClaim spec."""
    result = ['  spec {']
    
    # Access modes
    if 'accessModes' in spec:
        result.append('    access_modes = [')
        for mode in spec['accessModes']:
            result.append(f'      "{mode}",')
        result.append('    ]')
    
    # Storage class
    if 'storageClassName' in spec:
        result.append(f'    storage_class_name = "{spec["storageClassName"]}"')
    
    # Volume name
    if 'volumeName' in spec:
        result.append(f'    volume_name = "{spec["volumeName"]}"')
    
    # Resources
    if 'resources' in spec:
        result.append('    resources {')
        
        if 'requests' in spec['resources']:
            result.append('      requests = {')
            for k, v in spec['resources']['requests'].items():
                result.append(f'        "{k}" = "{v}"')
            result.append('      }')
        
        if 'limits' in spec['resources']:
            result.append('      limits = {')
            for k, v in spec['resources']['limits'].items():
                result.append(f'        "{k}" = "{v}"')
            result.append('      }')
        
        result.append('    }')
    
    result.append('  }')
    
    return '\n'.join(result)

def process_ingress_spec(spec):
    """Process Ingress spec."""
    result = ['  spec {']
    
    # TLS
    if 'tls' in spec:
        for tls in spec['tls']:
            result.append('    tls {')
            
            if 'hosts' in tls:
                result.append('      hosts = [')
                for host in tls['hosts']:
                    result.append(f'        "{host}",')
                result.append('      ]')
            
            if 'secretName' in tls:
                result.append(f'      secret_name = "{tls["secretName"]}"')
            
            result.append('    }')
    
    # Rules
    if 'rules' in spec:
        for rule in spec['rules']:
            result.append('    rule {')
            
            if 'host' in rule:
                result.append(f'      host = "{rule["host"]}"')
            
            if 'http' in rule:
                result.append('      http {')
                
                if 'paths' in rule['http']:
                    for path in rule['http']['paths']:
                        result.append('        path {')
                        
                        if 'path' in path:
                            result.append(f'          path = "{path["path"]}"')
                        
                        if 'pathType' in path:
                            result.append(f'          path_type = "{path["pathType"]}"')
                        
                        if 'backend' in path:
                            result.append('          backend {')
                            
                            if 'serviceName' in path['backend']:
                                result.append('            service_name = "{path["backend"]["serviceName"]}"')
                            
                            if 'servicePort' in path['backend']:
                                if isinstance(path['backend']['servicePort'], int):
                                    result.append(f'              service_port = {path["backend"]["servicePort"]}')
                                else:
                                    result.append(f'              service_port = "{path["backend"]["servicePort"]}"')
                            
                            result.append('          }')
                        
                        result.append('        }')
                
                result.append('      }')
            
            result.append('    }')
    
    result.append('  }')
    
    return '\n'.join(result)

def process_hpa_spec(spec):
    """Process HorizontalPodAutoscaler spec."""
    result = ['  spec {']
    
    # Scale target ref
    if 'scaleTargetRef' in spec:
        result.append('    scale_target_ref {')
        
        if 'apiVersion' in spec['scaleTargetRef']:
            result.append(f'      api_version = "{spec["scaleTargetRef"]["apiVersion"]}"')
        
        if 'kind' in spec['scaleTargetRef']:
            result.append(f'      kind = "{spec["scaleTargetRef"]["kind"]}"')
        
        if 'name' in spec['scaleTargetRef']:
            result.append(f'      name = "{spec["scaleTargetRef"]["name"]}"')
        
        result.append('    }')
    
    # Min replicas
    if 'minReplicas' in spec:
        result.append('    min_replicas = var.autoscaling.min_replicas')
    
    # Max replicas
    if 'maxReplicas' in spec:
        result.append('    max_replicas = var.autoscaling.max_replicas')
    
    # Metrics
    if 'metrics' in spec:
        for metric in spec['metrics']:
            result.append('    metric {')
            
            if 'type' in metric:
                result.append(f'      type = "{metric["type"]}"')
            
            if 'resource' in metric:
                result.append('      resource {')
                
                if 'name' in metric['resource']:
                    result.append(f'        name = "{metric["resource"]["name"]}"')
                
                if 'target' in metric['resource']:
                    result.append('        target {')
                    
                    if 'type' in metric['resource']['target']:
                        result.append(f'          type = "{metric["resource"]["target"]["type"]}"')
                    
                    if 'averageUtilization' in metric['resource']['target']:
                        result.append('          average_utilization = var.autoscaling.cpu_target')
                    
                    if 'averageValue' in metric['resource']['target']:
                        result.append(f'          average_value = "{metric["resource"]["target"]["averageValue"]}"')
                    
                    if 'value' in metric['resource']['target']:
                        result.append(f'          value = "{metric["resource"]["target"]["value"]}"')
                    
                    result.append('        }')
                
                result.append('      }')
            
            result.append('    }')
    
    result.append('  }')
    
    return '\n'.join(result)

def process_network_policy_spec(spec):
    """Process NetworkPolicy spec."""
    result = ['  spec {']
    
    # Pod selector
    if 'podSelector' in spec:
        result.append('    pod_selector {')
        
        if 'matchLabels' in spec['podSelector']:
            result.append('      match_labels = {')
            for k, v in spec['podSelector']['matchLabels'].items():
                result.append(f'        {k} = "{v}"')
            result.append('      }')
        
        result.append('    }')
    
    # Ingress
    if 'ingress' in spec:
        for ingress in spec['ingress']:
            result.append('    ingress {')
            
            if 'from' in ingress:
                for from_rule in ingress['from']:
                    
                    if 'ipBlock' in from_rule:
                        result.append('      from {')
                        result.append('        ip_block {')
                        
                        if 'cidr' in from_rule['ipBlock']:
                            result.append(f'          cidr = "{from_rule["ipBlock"]["cidr"]}"')
                        
                        if 'except' in from_rule['ipBlock']:
                            result.append('          except = [')
                            for cidr in from_rule['ipBlock']['except']:
                                result.append(f'            "{cidr}",')
                            result.append('          ]')
                        
                        result.append('        }')
                        result.append('      }')
                    
                    if 'namespaceSelector' in from_rule:
                        result.append('      from {')
                        result.append('        namespace_selector {')
                        
                        if 'matchLabels' in from_rule['namespaceSelector']:
                            result.append('          match_labels = {')
                            for k, v in from_rule['namespaceSelector']['matchLabels'].items():
                                result.append(f'            {k} = "{v}"')
                            result.append('          }')
                        
                        result.append('        }')
                        result.append('      }')
                    
                    if 'podSelector' in from_rule:
                        result.append('      from {')
                        result.append('        pod_selector {')
                        
                        if 'matchLabels' in from_rule['podSelector']:
                            result.append('          match_labels = {')
                            for k, v in from_rule['podSelector']['matchLabels'].items():
                                result.append(f'            {k} = "{v}"')
                            result.append('          }')
                        
                        result.append('        }')
                        result.append('      }')
            
            if 'ports' in ingress:
                for port in ingress['ports']:
                    result.append('      ports {')
                    
                    if 'protocol' in port:
                        result.append(f'        protocol = "{port["protocol"]}"')
                    
                    if 'port' in port:
                        if isinstance(port['port'], int):
                            result.append(f'        port = {port["port"]}')
                        else:
                            result.append(f'        port = "{port["port"]}"')
                    
                    result.append('      }')
            
            result.append('    }')
    
    # Egress
    if 'egress' in spec:
        for egress in spec['egress']:
            result.append('    egress {')
            
            if 'to' in egress:
                for to_rule in egress['to']:
                    
                    if 'ipBlock' in to_rule:
                        result.append('      to {')
                        result.append('        ip_block {')
                        
                        if 'cidr' in to_rule['ipBlock']:
                            result.append(f'          cidr = "{to_rule["ipBlock"]["cidr"]}"')
                        
                        if 'except' in to_rule['ipBlock']:
                            result.append('          except = [')
                            for cidr in to_rule['ipBlock']['except']:
                                result.append(f'            "{cidr}",')
                            result.append('          ]')
                        
                        result.append('        }')
                        result.append('      }')
                    
                    if 'namespaceSelector' in to_rule:
                        result.append('      to {')
                        result.append('        namespace_selector {')
                        
                        if 'matchLabels' in to_rule['namespaceSelector']:
                            result.append('          match_labels = {')
                            for k, v in to_rule['namespaceSelector']['matchLabels'].items():
                                result.append(f'            {k} = "{v}"')
                            result.append('          }')
                        
                        result.append('        }')
                        result.append('      }')
                    
                    if 'podSelector' in to_rule:
                        result.append('      to {')
                        result.append('        pod_selector {')
                        
                        if 'matchLabels' in to_rule['podSelector']:
                            result.append('          match_labels = {')
                            for k, v in to_rule['podSelector']['matchLabels'].items():
                                result.append(f'            {k} = "{v}"')
                            result.append('          }')
                        
                        result.append('        }')
                        result.append('      }')
            
            if 'ports' in egress:
                for port in egress['ports']:
                    result.append('      ports {')
                    
                    if 'protocol' in port:
                        result.append(f'        protocol = "{port["protocol"]}"')
                    
                    if 'port' in port:
                        if isinstance(port['port'], int):
                            result.append(f'        port = {port["port"]}')
                        else:
                            result.append(f'        port = "{port["port"]}"')
                    
                    result.append('      }')
            
            result.append('    }')
    
    # Policy types
    if 'policyTypes' in spec:
        result.append('    policy_types = [')
        for policy_type in spec['policyTypes']:
            result.append(f'      "{policy_type}",')
        result.append('    ]')
    
    result.append('  }')
    
    return '\n'.join(result)

def process_cronjob_spec(spec):
    """Process CronJob spec."""
    result = ['  spec {']
    
    # Schedule
    if 'schedule' in spec:
        result.append('    schedule = var.backup_schedule')
    
    # Concurrency policy
    if 'concurrencyPolicy' in spec:
        result.append(f'    concurrency_policy = "{spec["concurrencyPolicy"]}"')
    
    # Failed jobs history limit
    if 'failedJobsHistoryLimit' in spec:
        result.append(f'    failed_jobs_history_limit = {spec["failedJobsHistoryLimit"]}')
    
    # Successful jobs history limit
    if 'successfulJobsHistoryLimit' in spec:
        result.append(f'    successful_jobs_history_limit = {spec["successfulJobsHistoryLimit"]}')
    
    # Job template
    if 'jobTemplate' in spec:
        result.append('    job_template {')
        
        if 'metadata' in spec['jobTemplate']:
            result.append('      metadata {')
            
            if 'labels' in spec['jobTemplate']['metadata']:
                result.append('        labels = {')
                for k, v in spec['jobTemplate']['metadata']['labels'].items():
                    result.append(f'          {k} = "{v}"')
                result.append('        }')
            
            result.append('      }')
        
        if 'spec' in spec['jobTemplate']:
            result.append('      spec {')
            
            if 'template' in spec['jobTemplate']['spec']:
                result.append('        template {')
                
                if 'metadata' in spec['jobTemplate']['spec']['template']:
                    result.append('          metadata {')
                    
                    if 'labels' in spec['jobTemplate']['spec']['template']['metadata']:
                        result.append('            labels = {')
                        for k, v in spec['jobTemplate']['spec']['template']['metadata']['labels'].items():
                            result.append(f'              {k} = "{v}"')
                        result.append('            }')
                    
                    result.append('          }')
                
                if 'spec' in spec['jobTemplate']['spec']['template']:
                    template_spec = spec['jobTemplate']['spec']['template']['spec']
                    result.append('          spec {')
                    
                    # Containers
                    if 'containers' in template_spec:
                        for container in template_spec['containers']:
                            result.append('            container {')
                            result.append(f'              name = "{container["name"]}"')
                            
                            # Image
                            if 'image' in container:
                                result.append(f'              image = "{container["image"]}"')
                            
                            # Command
                            if 'command' in container:
                                result.append('              command = [')
                                for cmd in container['command']:
                                    result.append(f'                "{cmd}",')
                                result.append('              ]')
                            
                            # Args
                            if 'args' in container:
                                result.append('              args = [')
                                for arg in container['args']:
                                    result.append(f'                "{arg}",')
                                result.append('              ]')
                            
                            # Environment variables
                            if 'env' in container:
                                for env in container['env']:
                                    result.append('              env {')
                                    result.append(f'                name = "{env["name"]}"')
                                    
                                    if 'value' in env:
                                        result.append(f'                value = "{env["value"]}"')
                                    elif 'valueFrom' in env:
                                        result.append('                value_from {')
                                        
                                        if 'configMapKeyRef' in env['valueFrom']:
                                            result.append('                  config_map_key_ref {')
                                            ref = env['valueFrom']['configMapKeyRef']
                                            result.append(f'                    name = "{ref["name"]}"')
                                            result.append(f'                    key = "{ref["key"]}"')
                                            result.append('                  }')
                                        
                                        if 'secretKeyRef' in env['valueFrom']:
                                            result.append('                  secret_key_ref {')
                                            ref = env['valueFrom']['secretKeyRef']
                                            result.append(f'                    name = "{ref["name"]}"')
                                            result.append(f'                    key = "{ref["key"]}"')
                                            result.append('                  }')
                                        
                                        result.append('                }')
                                    
                                    result.append('              }')
                            
                            # Volume mounts
                            if 'volumeMounts' in container:
                                for mount in container['volumeMounts']:
                                    result.append('              volume_mount {')
                                    result.append(f'                name = "{mount["name"]}"')
                                    result.append(f'                mount_path = "{mount["mountPath"]}"')
                                    if 'subPath' in mount:
                                        result.append(f'                sub_path = "{mount["subPath"]}"')
                                    if 'readOnly' in mount:
                                        result.append(f'                read_only = {str(mount["readOnly"]).lower()}')
                                    result.append('              }')
                            
                            # Resources
                            if 'resources' in container:
                                result.append('              resources {')
                                
                                if 'limits' in container['resources']:
                                    result.append('                limits = {')
                                    for k, v in container['resources']['limits'].items():
                                        result.append(f'                  "{k}" = "{v}"')
                                    result.append('                }')
                                
                                if 'requests' in container['resources']:
                                    result.append('                requests = {')
                                    for k, v in container['resources']['requests'].items():
                                        result.append(f'                  "{k}" = "{v}"')
                                    result.append('                }')
                                
                                result.append('              }')
                            
                            result.append('            }')
                    
                    # Volumes
                    if 'volumes' in template_spec:
                        for volume in template_spec['volumes']:
                            result.append('            volume {')
                            result.append(f'              name = "{volume["name"]}"')
                            
                            if 'configMap' in volume:
                                result.append('              config_map {')
                                result.append(f'                name = "{volume["configMap"]["name"]}"')
                                if 'defaultMode' in volume['configMap']:
                                    result.append(f'                default_mode = "{volume["configMap"]["defaultMode"]}"')
                                result.append('              }')
                            
                            if 'secret' in volume:
                                result.append('              secret {')
                                result.append(f'                secret_name = "{volume["secret"]["secretName"]}"')
                                if 'defaultMode' in volume['secret']:
                                    result.append(f'                default_mode = "{volume["secret"]["defaultMode"]}"')
                                result.append('              }')
                            
                            if 'persistentVolumeClaim' in volume:
                                result.append('              persistent_volume_claim {')
                                result.append(f'                claim_name = "{volume["persistentVolumeClaim"]["claimName"]}"')
                                if 'readOnly' in volume['persistentVolumeClaim']:
                                    result.append(f'                read_only = {str(volume["persistentVolumeClaim"]["readOnly"]).lower()}')
                                result.append('              }')
                            
                            if 'emptyDir' in volume:
                                result.append('              empty_dir {')
                                if 'medium' in volume['emptyDir']:
                                    result.append(f'                medium = "{volume["emptyDir"]["medium"]}"')
                                if 'sizeLimit' in volume['emptyDir']:
                                    result.append(f'                size_limit = "{volume["emptyDir"]["sizeLimit"]}"')
                                result.append('              }')
                            
                            result.append('            }')
                    
                    # Restart policy
                    if 'restartPolicy' in template_spec:
                        result.append(f'            restart_policy = "{template_spec["restartPolicy"]}"')
                    
                    # Service account
                    if 'serviceAccountName' in template_spec:
                        result.append(f'            service_account_name = "{template_spec["serviceAccountName"]}"')
                    
                    result.append('          }')
                
                result.append('        }')
            
            result.append('      }')
        
        result.append('    }')
    
    result.append('  }')
    
    return '\n'.join(result)

def process_files(input_files, output_dir):
    """Process multiple YAML files and convert them to Terraform."""
    os.makedirs(output_dir, exist_ok=True)
    
    for input_file in input_files:
        if not os.path.isfile(input_file):
            print(f"Warning: Input file {input_file} does not exist or is not a file.")
            continue
        
        try:
            with open(input_file, 'r') as f:
                yaml_content = f.read()
            
            terraform_content = yaml_to_terraform(yaml_content)
            
            base_name = os.path.basename(input_file)
            tf_file = os.path.splitext(base_name)[0] + '.tf'
            output_file = os.path.join(output_dir, tf_file)
            
            with open(output_file, 'w') as f:
                f.write(f"# Generated from {input_file}\n\n")
                f.write(terraform_content)
            
            print(f"Converted {input_file} to {output_file}")
        except Exception as e:
            print(f"Error processing file {input_file}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Convert Kubernetes YAML manifests to Terraform configuration")
    parser.add_argument('--input', '-i', nargs='+', help='Input YAML file(s) or directory')
    parser.add_argument('--output-dir', '-o', default='terraform_output', help='Output directory for Terraform files')
    parser.add_argument('--recursive', '-r', action='store_true', help='Recursively process directories')
    
    args = parser.parse_args()
    
    if not args.input:
        parser.print_help()
        return 1
    
    # Collect input files
    input_files = []
    for path in args.input:
        if os.path.isfile(path):
            input_files.append(path)
        elif os.path.isdir(path):
            if args.recursive:
                for root, _, files in os.walk(path):
                    for file in files:
                        if file.endswith(('.yaml', '.yml')):
                            input_files.append(os.path.join(root, file))
            else:
                for file in os.listdir(path):
                    if file.endswith(('.yaml', '.yml')):
                        input_files.append(os.path.join(path, file))
        else:
            print(f"Warning: {path} is not a valid file or directory.")
    
    if not input_files:
        print("No YAML files found to process.")
        return 1
    
    process_files(input_files, args.output_dir)
    return 0

if __name__ == "__main__":
    sys.exit(main())