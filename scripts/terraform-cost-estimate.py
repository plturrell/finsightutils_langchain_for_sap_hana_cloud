#!/usr/bin/env python3
"""
Script to estimate costs for Terraform-managed infrastructure.
This is a simple estimation tool for understanding resource usage.
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
import csv
import datetime

# Define cost constants (very approximate)
COST_ESTIMATES = {
    # Compute costs per month
    "cpu": {
        "request": 20.0,  # $ per CPU core
        "limit": 0.0      # No additional cost for limits
    },
    "memory": {
        "request": 5.0,   # $ per GB
        "limit": 0.0      # No additional cost for limits
    },
    "gpu": {
        "nvidia": 200.0   # $ per GPU
    },
    # Storage costs per month
    "storage": {
        "standard": 0.1,  # $ per GB
        "premium": 0.2    # $ per GB
    },
    # Additional services
    "monitoring": {
        "prometheus": 30.0,
        "grafana": 10.0
    },
    # Network costs
    "network": {
        "base": 50.0      # Base cost for network infrastructure
    }
}

def get_terraform_plan(plan_file=None, env_dir=None):
    """Get Terraform plan output as JSON."""
    try:
        if plan_file:
            # Read from existing plan file
            with open(plan_file, 'r') as f:
                plan_json = json.load(f)
        elif env_dir:
            # Generate a new plan
            cwd = os.getcwd()
            os.chdir(env_dir)
            
            # Initialize Terraform if needed
            subprocess.run(["terraform", "init"], check=True, capture_output=True)
            
            # Create plan and capture as JSON
            result = subprocess.run(
                ["terraform", "plan", "-out=tfplan"],
                check=True, capture_output=True, text=True
            )
            
            # Convert plan to JSON
            result = subprocess.run(
                ["terraform", "show", "-json", "tfplan"],
                check=True, capture_output=True, text=True
            )
            
            plan_json = json.loads(result.stdout)
            
            # Clean up
            subprocess.run(["rm", "tfplan"], check=True)
            os.chdir(cwd)
        else:
            raise ValueError("Either plan_file or env_dir must be provided")
        
        return plan_json
    except Exception as e:
        print(f"Error getting Terraform plan: {str(e)}")
        return None

def parse_size_to_gb(size_str):
    """Parse size string (like '4Gi') to GB float."""
    if not size_str:
        return 0.0
    
    size_str = size_str.strip().lower()
    
    # Extract number and unit
    if size_str[-2:] == 'gi':
        return float(size_str[:-2])
    elif size_str[-2:] == 'mi':
        return float(size_str[:-2]) / 1024.0
    elif size_str[-1:] == 'g':
        return float(size_str[:-1])
    elif size_str[-1:] == 'm':
        return float(size_str[:-1]) / 1024.0
    else:
        try:
            return float(size_str)
        except ValueError:
            return 0.0

def estimate_resource_costs(plan_json):
    """Estimate costs based on resources in the plan."""
    if not plan_json or 'resource_changes' not in plan_json:
        return None
    
    cost_estimate = {
        "compute": {
            "cpu": 0.0,
            "memory": 0.0,
            "gpu": 0.0,
            "subtotal": 0.0
        },
        "storage": {
            "standard": 0.0,
            "premium": 0.0,
            "subtotal": 0.0
        },
        "services": {
            "monitoring": 0.0,
            "networking": 0.0,
            "subtotal": 0.0
        },
        "total": 0.0
    }
    
    resources = {}
    
    # Extract resources
    for change in plan_json['resource_changes']:
        if 'change' not in change or 'after' not in change['change']:
            continue
        
        resource_type = change['type']
        resource_name = change['name']
        
        after = change['change']['after']
        
        # Process Kubernetes resources
        if resource_type == 'kubernetes_deployment':
            # Extract container resources
            try:
                spec = after['spec'][0]
                template = spec['template'][0]
                pod_spec = template['spec'][0]
                
                for container in pod_spec.get('container', []):
                    resources_spec = container.get('resources', [{}])[0]
                    
                    # CPU requests and limits
                    requests = resources_spec.get('requests', {})
                    limits = resources_spec.get('limits', {})
                    
                    cpu_request = float(requests.get('cpu', '0').replace('m', '')) / 1000.0 if 'm' in requests.get('cpu', '0') else float(requests.get('cpu', '0'))
                    memory_request_gb = parse_size_to_gb(requests.get('memory', '0'))
                    
                    # GPU allocation
                    gpu_count = int(limits.get('nvidia.com/gpu', 0))
                    
                    # Replicas
                    replicas = int(spec.get('replicas', 1))
                    
                    # Calculate costs
                    cost_estimate['compute']['cpu'] += cpu_request * COST_ESTIMATES['cpu']['request'] * replicas
                    cost_estimate['compute']['memory'] += memory_request_gb * COST_ESTIMATES['memory']['request'] * replicas
                    cost_estimate['compute']['gpu'] += gpu_count * COST_ESTIMATES['gpu']['nvidia'] * replicas
            except (KeyError, IndexError):
                pass
        
        # Process persistent volume claims
        elif resource_type == 'kubernetes_persistent_volume_claim':
            try:
                spec = after['spec'][0]
                resources = spec.get('resources', [{}])[0]
                requests = resources.get('requests', {})
                storage_size_gb = parse_size_to_gb(requests.get('storage', '0'))
                storage_class = spec.get('storage_class_name', 'standard')
                
                if storage_class == 'premium':
                    cost_estimate['storage']['premium'] += storage_size_gb * COST_ESTIMATES['storage']['premium']
                else:
                    cost_estimate['storage']['standard'] += storage_size_gb * COST_ESTIMATES['storage']['standard']
            except (KeyError, IndexError):
                pass
        
        # Process monitoring resources
        elif resource_type == 'helm_release' and after.get('name') in ['prometheus', 'grafana']:
            if after.get('name') == 'prometheus':
                cost_estimate['services']['monitoring'] += COST_ESTIMATES['monitoring']['prometheus']
            elif after.get('name') == 'grafana':
                cost_estimate['services']['monitoring'] += COST_ESTIMATES['monitoring']['grafana']
    
    # Network costs
    if cost_estimate['compute']['cpu'] > 0:  # If we have compute resources, add network cost
        cost_estimate['services']['networking'] = COST_ESTIMATES['network']['base']
    
    # Calculate subtotals
    cost_estimate['compute']['subtotal'] = (
        cost_estimate['compute']['cpu'] +
        cost_estimate['compute']['memory'] +
        cost_estimate['compute']['gpu']
    )
    
    cost_estimate['storage']['subtotal'] = (
        cost_estimate['storage']['standard'] +
        cost_estimate['storage']['premium']
    )
    
    cost_estimate['services']['subtotal'] = (
        cost_estimate['services']['monitoring'] +
        cost_estimate['services']['networking']
    )
    
    # Calculate total
    cost_estimate['total'] = (
        cost_estimate['compute']['subtotal'] +
        cost_estimate['storage']['subtotal'] +
        cost_estimate['services']['subtotal']
    )
    
    return cost_estimate

def print_cost_estimate(cost_estimate, output_format='text', output_file=None):
    """Print the cost estimate in the specified format."""
    if not cost_estimate:
        print("No cost estimate available.")
        return
    
    if output_format == 'text':
        print("\n===== Monthly Cost Estimate =====\n")
        
        print("Compute Resources:")
        print(f"  CPU:               ${cost_estimate['compute']['cpu']:.2f}")
        print(f"  Memory:            ${cost_estimate['compute']['memory']:.2f}")
        print(f"  GPU:               ${cost_estimate['compute']['gpu']:.2f}")
        print(f"  Subtotal:          ${cost_estimate['compute']['subtotal']:.2f}")
        
        print("\nStorage Resources:")
        print(f"  Standard Storage:  ${cost_estimate['storage']['standard']:.2f}")
        print(f"  Premium Storage:   ${cost_estimate['storage']['premium']:.2f}")
        print(f"  Subtotal:          ${cost_estimate['storage']['subtotal']:.2f}")
        
        print("\nAdditional Services:")
        print(f"  Monitoring:        ${cost_estimate['services']['monitoring']:.2f}")
        print(f"  Networking:        ${cost_estimate['services']['networking']:.2f}")
        print(f"  Subtotal:          ${cost_estimate['services']['subtotal']:.2f}")
        
        print("\n------------------------------")
        print(f"TOTAL MONTHLY COST:  ${cost_estimate['total']:.2f}")
        print("==============================\n")
        
        print("NOTE: This is an approximation based on generic cloud costs.")
        print("      Actual costs may vary based on your specific provider and usage patterns.")
    
    elif output_format == 'json':
        result = {
            "timestamp": datetime.datetime.now().isoformat(),
            "cost_estimate": cost_estimate,
            "disclaimer": "This is an approximation based on generic cloud costs. Actual costs may vary."
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Cost estimate saved to {output_file}")
        else:
            print(json.dumps(result, indent=2))
    
    elif output_format == 'csv':
        rows = [
            ["Category", "Resource", "Monthly Cost ($)"],
            ["Compute", "CPU", cost_estimate['compute']['cpu']],
            ["Compute", "Memory", cost_estimate['compute']['memory']],
            ["Compute", "GPU", cost_estimate['compute']['gpu']],
            ["Compute", "Subtotal", cost_estimate['compute']['subtotal']],
            ["Storage", "Standard Storage", cost_estimate['storage']['standard']],
            ["Storage", "Premium Storage", cost_estimate['storage']['premium']],
            ["Storage", "Subtotal", cost_estimate['storage']['subtotal']],
            ["Services", "Monitoring", cost_estimate['services']['monitoring']],
            ["Services", "Networking", cost_estimate['services']['networking']],
            ["Services", "Subtotal", cost_estimate['services']['subtotal']],
            ["", "TOTAL", cost_estimate['total']]
        ]
        
        if output_file:
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(rows)
            print(f"Cost estimate saved to {output_file}")
        else:
            for row in rows:
                print(','.join(str(cell) for cell in row))

def main():
    parser = argparse.ArgumentParser(description="Estimate costs for Terraform-managed infrastructure")
    parser.add_argument('--plan-file', help='Path to Terraform plan JSON file')
    parser.add_argument('--environment', choices=['staging', 'production'], help='Environment to estimate costs for')
    parser.add_argument('--format', choices=['text', 'json', 'csv'], default='text', help='Output format')
    parser.add_argument('--output', help='Output file (for JSON or CSV formats)')
    
    args = parser.parse_args()
    
    # Get Terraform project root
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent
    terraform_root = project_root / 'terraform'
    
    if not args.plan_file and not args.environment:
        parser.error("Either --plan-file or --environment must be specified")
    
    # Get the plan
    plan_json = None
    if args.plan_file:
        plan_json = get_terraform_plan(plan_file=args.plan_file)
    elif args.environment:
        env_dir = terraform_root / 'environments' / args.environment
        if not env_dir.exists():
            parser.error(f"Environment directory not found: {env_dir}")
        
        plan_json = get_terraform_plan(env_dir=env_dir)
    
    if not plan_json:
        print("Failed to get Terraform plan.")
        return 1
    
    # Estimate costs
    cost_estimate = estimate_resource_costs(plan_json)
    
    if not cost_estimate:
        print("Failed to estimate costs.")
        return 1
    
    # Print results
    print_cost_estimate(cost_estimate, args.format, args.output)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())