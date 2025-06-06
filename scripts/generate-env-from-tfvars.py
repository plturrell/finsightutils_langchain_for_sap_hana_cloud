#!/usr/bin/env python3
"""
Generate .env file from Terraform credentials.tfvars file.
This helps with using the same credentials for both Terraform and the application.
"""

import os
import sys
import re
import argparse
from pathlib import Path


def parse_tfvars(file_path):
    """Parse a Terraform .tfvars file and extract credential values."""
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist.")
        return None
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract credentials using regex
    creds = {}
    
    # Parse HANA credentials
    hana_match = re.search(r'hana_credentials\s*=\s*{([^}]*)}', content, re.DOTALL)
    if hana_match:
        hana_block = hana_match.group(1)
        
        # Extract individual values
        host_match = re.search(r'host\s*=\s*"([^"]*)"', hana_block)
        port_match = re.search(r'port\s*=\s*"([^"]*)"', hana_block)
        user_match = re.search(r'user\s*=\s*"([^"]*)"', hana_block)
        password_match = re.search(r'password\s*=\s*"([^"]*)"', hana_block)
        
        if host_match:
            creds['HANA_HOST'] = host_match.group(1)
        if port_match:
            creds['HANA_PORT'] = port_match.group(1)
        if user_match:
            creds['HANA_USER'] = user_match.group(1)
        if password_match:
            creds['HANA_PASSWORD'] = password_match.group(1)
    
    # Parse DataSphere credentials
    ds_match = re.search(r'datasphere_credentials\s*=\s*{([^}]*)}', content, re.DOTALL)
    if ds_match:
        ds_block = ds_match.group(1)
        
        # Extract individual values
        client_id_match = re.search(r'client_id\s*=\s*"([^"]*)"', ds_block)
        client_secret_match = re.search(r'client_secret\s*=\s*"([^"]*)"', ds_block)
        auth_url_match = re.search(r'auth_url\s*=\s*"([^"]*)"', ds_block)
        token_url_match = re.search(r'token_url\s*=\s*"([^"]*)"', ds_block)
        api_url_match = re.search(r'api_url\s*=\s*"([^"]*)"', ds_block)
        
        if client_id_match:
            creds['DATASPHERE_CLIENT_ID'] = client_id_match.group(1)
        if client_secret_match:
            creds['DATASPHERE_CLIENT_SECRET'] = client_secret_match.group(1)
        if auth_url_match:
            creds['DATASPHERE_AUTH_URL'] = auth_url_match.group(1)
        if token_url_match:
            creds['DATASPHERE_TOKEN_URL'] = token_url_match.group(1)
        if api_url_match:
            creds['DATASPHERE_API_URL'] = api_url_match.group(1)
    
    return creds


def generate_env_file(creds, output_file):
    """Generate a .env file from the extracted credentials."""
    if not creds:
        return False
    
    try:
        with open(output_file, 'w') as f:
            f.write("# Generated from Terraform credentials.tfvars\n")
            f.write("# DO NOT COMMIT THIS FILE!\n\n")
            
            # Write HANA credentials
            f.write("# SAP HANA Cloud credentials\n")
            for key in ['HANA_HOST', 'HANA_PORT', 'HANA_USER', 'HANA_PASSWORD']:
                if key in creds:
                    f.write(f"{key}={creds[key]}\n")
            
            f.write("\n# SAP DataSphere credentials\n")
            for key in ['DATASPHERE_CLIENT_ID', 'DATASPHERE_CLIENT_SECRET', 
                        'DATASPHERE_AUTH_URL', 'DATASPHERE_TOKEN_URL', 'DATASPHERE_API_URL']:
                if key in creds:
                    f.write(f"{key}={creds[key]}\n")
        
        return True
    except Exception as e:
        print(f"Error writing .env file: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Generate .env file from Terraform credentials.tfvars")
    parser.add_argument('--input', '-i', default='terraform/credentials.tfvars', 
                        help='Path to credentials.tfvars file (default: terraform/credentials.tfvars)')
    parser.add_argument('--output', '-o', default='.env',
                        help='Path to output .env file (default: .env)')
    parser.add_argument('--force', '-f', action='store_true',
                        help='Overwrite output file if it exists')
    
    args = parser.parse_args()
    
    # Get absolute paths
    script_dir = Path(__file__).parent.absolute()
    project_dir = script_dir.parent
    
    # Resolve input path
    input_path = args.input
    if not os.path.isabs(input_path):
        input_path = os.path.join(project_dir, input_path)
    
    # Resolve output path
    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(project_dir, output_path)
    
    # Check if output file exists
    if os.path.exists(output_path) and not args.force:
        print(f"Error: Output file {output_path} already exists. Use --force to overwrite.")
        return 1
    
    # Parse tfvars file
    print(f"Parsing Terraform variables from {input_path}")
    creds = parse_tfvars(input_path)
    
    if not creds:
        print("Error: Failed to parse credentials from tfvars file.")
        return 1
    
    # Generate .env file
    print(f"Generating .env file at {output_path}")
    if generate_env_file(creds, output_path):
        print(f"Successfully generated .env file with {len(creds)} environment variables.")
        print("IMPORTANT: Do not commit this file to version control!")
        return 0
    else:
        print("Error: Failed to generate .env file.")
        return 1


if __name__ == "__main__":
    sys.exit(main())