#!/usr/bin/env python3
"""
Script to generate Terraform code from existing Kubernetes manifests.
This helps with migrating from static Kubernetes YAML files to Terraform.
"""

import os
import sys
import yaml
import argparse
from pathlib import Path


def indent(text, spaces=2):
    """Indent text by specified number of spaces."""
    return '\n'.join(' ' * spaces + line for line in text.split('\n'))


def sanitize_name(name):
    """Sanitize resource name for Terraform use."""
    return name.replace('-', '_').replace('.', '_').lower()


def process_metadata(metadata, indent_level=2):
    """Process Kubernetes metadata into Terraform format."""
    result = []
    spaces = ' ' * indent_level
    
    result.append(f"{spaces}metadata {{")
    
    if 'name' in metadata:
        result.append(f"{spaces}  name = \"{metadata['name']}\"")
    
    if 'namespace' in metadata:
        result.append(f"{spaces}  namespace = var.namespace")
    
    if 'labels' in metadata:
        result.append(f"{spaces}  labels = {{")
        for key, value in metadata['labels'].items():
            result.append(f"{spaces}    {key} = \"{value}\"")
        result.append(f"{spaces}  }}")
    
    if 'annotations' in metadata:
        result.append(f"{spaces}  annotations = {{")
        for key, value in metadata['annotations'].items():
            result.append(f"{spaces}    \"{key}\" = \"{value}\"")
        result.append(f"{spaces}  }}")
    
    result.append(f"{spaces}}}")
    
    return '\n'.join(result)


def process_deployment(deployment):
    """Convert Deployment YAML to Terraform."""
    name = deployment['metadata']['name']
    resource_name = sanitize_name(name)
    
    result = [f"resource \"kubernetes_deployment\" \"{resource_name}\" {{"]
    
    # Add metadata
    result.append(process_metadata(deployment['metadata']))
    
    # Start spec
    result.append("  spec {")
    
    # Replicas
    if 'replicas' in deployment['spec']:
        result.append(f"    replicas = var.replicas")
    
    # Selector
    if 'selector' in deployment['spec']:
        result.append("    selector {")
        if 'matchLabels' in deployment['spec']['selector']:
            result.append("      match_labels = {")
            for key, value in deployment['spec']['selector']['matchLabels'].items():
                result.append(f"        {key} = \"{value}\"")
            result.append("      }")
        result.append("    }")
    
    # Template
    if 'template' in deployment['spec']:
        result.append("    template {")
        
        # Template metadata
        if 'metadata' in deployment['spec']['template']:
            result.append(process_metadata(deployment['spec']['template']['metadata'], 6))
        
        # Template spec
        if 'spec' in deployment['spec']['template']:
            result.append("      spec {")
            
            # Containers
            if 'containers' in deployment['spec']['template']['spec']:
                for container in deployment['spec']['template']['spec']['containers']:
                    result.append("        container {")
                    result.append(f"          name  = \"{container['name']}\"")
                    result.append(f"          image = var.image_repository + \":\" + var.image_tag")
                    
                    # Ports
                    if 'ports' in container:
                        for port in container['ports']:
                            result.append("          port {")
                            result.append(f"            container_port = {port['containerPort']}")
                            if 'name' in port:
                                result.append(f"            name = \"{port['name']}\"")
                            result.append("          }")
                    
                    # Environment variables
                    if 'env' in container:
                        for env in container['env']:
                            result.append("          env {")
                            result.append(f"            name = \"{env['name']}\"")
                            if 'value' in env:
                                result.append(f"            value = \"{env['value']}\"")
                            elif 'valueFrom' in env:
                                result.append("            value_from {")
                                if 'configMapKeyRef' in env['valueFrom']:
                                    result.append("              config_map_key_ref {")
                                    result.append(f"                name = \"{env['valueFrom']['configMapKeyRef']['name']}\"")
                                    result.append(f"                key = \"{env['valueFrom']['configMapKeyRef']['key']}\"")
                                    result.append("              }")
                                elif 'secretKeyRef' in env['valueFrom']:
                                    result.append("              secret_key_ref {")
                                    result.append(f"                name = \"{env['valueFrom']['secretKeyRef']['name']}\"")
                                    result.append(f"                key = \"{env['valueFrom']['secretKeyRef']['key']}\"")
                                    result.append("              }")
                                result.append("            }")
                            result.append("          }")
                    
                    # Resources
                    if 'resources' in container:
                        result.append("          resources {")
                        if 'limits' in container['resources']:
                            result.append("            limits = {")
                            for key, value in container['resources']['limits'].items():
                                if key == 'nvidia.com/gpu':
                                    result.append(f"              \"{key}\" = var.gpu_enabled ? var.nvidia_gpu_count : null")
                                else:
                                    result.append(f"              \"{key}\" = \"{value}\"")
                            result.append("            }")
                        if 'requests' in container['resources']:
                            result.append("            requests = {")
                            for key, value in container['resources']['requests'].items():
                                result.append(f"              \"{key}\" = \"{value}\"")
                            result.append("            }")
                        result.append("          }")
                    
                    # Health checks
                    if 'livenessProbe' in container:
                        result.append("          liveness_probe {")
                        probe = container['livenessProbe']
                        if 'httpGet' in probe:
                            result.append("            http_get {")
                            result.append(f"              path = \"{probe['httpGet']['path']}\"")
                            result.append(f"              port = \"{probe['httpGet']['port']}\"")
                            result.append("            }")
                        if 'initialDelaySeconds' in probe:
                            result.append(f"            initial_delay_seconds = {probe['initialDelaySeconds']}")
                        if 'periodSeconds' in probe:
                            result.append(f"            period_seconds = {probe['periodSeconds']}")
                        result.append("          }")
                    
                    if 'readinessProbe' in container:
                        result.append("          readiness_probe {")
                        probe = container['readinessProbe']
                        if 'httpGet' in probe:
                            result.append("            http_get {")
                            result.append(f"              path = \"{probe['httpGet']['path']}\"")
                            result.append(f"              port = \"{probe['httpGet']['port']}\"")
                            result.append("            }")
                        if 'initialDelaySeconds' in probe:
                            result.append(f"            initial_delay_seconds = {probe['initialDelaySeconds']}")
                        if 'periodSeconds' in probe:
                            result.append(f"            period_seconds = {probe['periodSeconds']}")
                        result.append("          }")
                    
                    result.append("        }")
            
            # Node selector
            if 'nodeSelector' in deployment['spec']['template']['spec']:
                result.append("        dynamic \"node_selector\" {")
                result.append("          for_each = var.gpu_enabled ? [1] : []")
                result.append("          content {")
                for key, value in deployment['spec']['template']['spec']['nodeSelector'].items():
                    result.append(f"            \"{key}\" = \"{value}\"")
                result.append("          }")
                result.append("        }")
            
            result.append("      }")
        
        result.append("    }")
    
    result.append("  }")
    result.append("}")
    
    return '\n'.join(result)


def process_service(service):
    """Convert Service YAML to Terraform."""
    name = service['metadata']['name']
    resource_name = sanitize_name(name)
    
    result = [f"resource \"kubernetes_service\" \"{resource_name}\" {{"]
    
    # Add metadata
    result.append(process_metadata(service['metadata']))
    
    # Start spec
    result.append("  spec {")
    
    # Selector
    if 'selector' in service['spec']:
        result.append("    selector = {")
        for key, value in service['spec']['selector'].items():
            result.append(f"      {key} = \"{value}\"")
        result.append("    }")
    
    # Ports
    if 'ports' in service['spec']:
        for port in service['spec']['ports']:
            result.append("    port {")
            result.append(f"      port = {port['port']}")
            if 'targetPort' in port:
                result.append(f"      target_port = {port['targetPort']}")
            if 'protocol' in port:
                result.append(f"      protocol = \"{port['protocol']}\"")
            if 'name' in port:
                result.append(f"      name = \"{port['name']}\"")
            result.append("    }")
    
    # Type
    if 'type' in service['spec']:
        result.append(f"    type = \"{service['spec']['type']}\"")
    
    result.append("  }")
    result.append("}")
    
    return '\n'.join(result)


def process_configmap(configmap):
    """Convert ConfigMap YAML to Terraform."""
    name = configmap['metadata']['name']
    resource_name = sanitize_name(name)
    
    result = [f"resource \"kubernetes_config_map\" \"{resource_name}\" {{"]
    
    # Add metadata
    result.append(process_metadata(configmap['metadata']))
    
    # Data
    if 'data' in configmap:
        result.append("  data = {")
        for key, value in configmap['data'].items():
            result.append(f"    \"{key}\" = \"{value}\"")
        result.append("  }")
    
    result.append("}")
    
    return '\n'.join(result)


def process_secret(secret):
    """Convert Secret YAML to Terraform."""
    name = secret['metadata']['name']
    resource_name = sanitize_name(name)
    
    result = [f"resource \"kubernetes_secret\" \"{resource_name}\" {{"]
    
    # Add metadata
    result.append(process_metadata(secret['metadata']))
    
    # Data
    if 'data' in secret:
        result.append("  data = {")
        for key, value in secret['data'].items():
            result.append(f"    \"{key}\" = \"placeholder\"  # Replace with var.{key.lower()}")
        result.append("  }")
    
    # Type
    if 'type' in secret:
        result.append(f"  type = \"{secret['type']}\"")
    else:
        result.append("  type = \"Opaque\"")
    
    result.append("}")
    
    return '\n'.join(result)


def process_hpa(hpa):
    """Convert HorizontalPodAutoscaler YAML to Terraform."""
    name = hpa['metadata']['name']
    resource_name = sanitize_name(name)
    
    result = [f"resource \"kubernetes_horizontal_pod_autoscaler\" \"{resource_name}\" {{"]
    
    # Add metadata
    result.append(process_metadata(hpa['metadata']))
    
    # Spec
    result.append("  spec {")
    
    # Scale target ref
    if 'scaleTargetRef' in hpa['spec']:
        result.append("    scale_target_ref {")
        if 'apiVersion' in hpa['spec']['scaleTargetRef']:
            result.append(f"      api_version = \"{hpa['spec']['scaleTargetRef']['apiVersion']}\"")
        if 'kind' in hpa['spec']['scaleTargetRef']:
            result.append(f"      kind = \"{hpa['spec']['scaleTargetRef']['kind']}\"")
        if 'name' in hpa['spec']['scaleTargetRef']:
            result.append(f"      name = \"{hpa['spec']['scaleTargetRef']['name']}\"")
        result.append("    }")
    
    # Min/max replicas
    if 'minReplicas' in hpa['spec']:
        result.append(f"    min_replicas = var.autoscaling.min_replicas")
    if 'maxReplicas' in hpa['spec']:
        result.append(f"    max_replicas = var.autoscaling.max_replicas")
    
    # Metrics
    if 'metrics' in hpa['spec']:
        for metric in hpa['spec']['metrics']:
            result.append("    metric {")
            result.append(f"      type = \"{metric['type']}\"")
            
            if metric['type'] == 'Resource':
                result.append("      resource {")
                result.append(f"        name = \"{metric['resource']['name']}\"")
                
                if 'target' in metric['resource']:
                    result.append("        target {")
                    result.append(f"          type = \"{metric['resource']['target']['type']}\"")
                    if 'averageUtilization' in metric['resource']['target']:
                        result.append(f"          average_utilization = var.autoscaling.cpu_target")
                    result.append("        }")
                
                result.append("      }")
            
            result.append("    }")
    
    result.append("  }")
    result.append("}")
    
    return '\n'.join(result)


def process_resource(resource):
    """Process a Kubernetes resource and convert to Terraform."""
    kind = resource.get('kind', '')
    
    if kind == 'Deployment':
        return process_deployment(resource)
    elif kind == 'Service':
        return process_service(resource)
    elif kind == 'ConfigMap':
        return process_configmap(resource)
    elif kind == 'Secret':
        return process_secret(resource)
    elif kind == 'HorizontalPodAutoscaler':
        return process_hpa(resource)
    else:
        return f"# Unsupported resource type: {kind}"


def process_yaml_file(file_path):
    """Process a YAML file containing Kubernetes resources."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Split YAML file with multiple documents
        documents = list(yaml.safe_load_all(content))
        
        results = []
        for doc in documents:
            if doc:  # Skip empty documents
                results.append(process_resource(doc))
        
        return '\n\n'.join(results)
    except Exception as e:
        return f"# Error processing {file_path}: {str(e)}"


def main():
    parser = argparse.ArgumentParser(description='Convert Kubernetes YAML to Terraform')
    parser.add_argument('input_dir', help='Directory containing Kubernetes YAML files')
    parser.add_argument('output_dir', help='Directory to write Terraform files')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Input directory does not exist: {input_dir}")
        return 1
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    
    yaml_files = list(input_dir.glob('**/*.yaml')) + list(input_dir.glob('**/*.yml'))
    
    for yaml_file in yaml_files:
        relative_path = yaml_file.relative_to(input_dir)
        output_file = output_dir / relative_path.with_suffix('.tf')
        
        # Create parent directories if they don't exist
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        terraform_code = process_yaml_file(yaml_file)
        
        with open(output_file, 'w') as f:
            f.write(f"# Generated from {yaml_file}\n\n")
            f.write(terraform_code)
        
        print(f"Processed {yaml_file} -> {output_file}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())