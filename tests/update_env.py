#!/usr/bin/env python3
"""
Script to update HANA credentials in .env file
"""

import os
from dotenv import load_dotenv, set_key

def update_env():
    """Update HANA credentials in .env file"""
    # Create .env file if it doesn't exist
    env_path = ".env"
    if not os.path.exists(env_path):
        with open(env_path, "w") as f:
            f.write("HANA_HOST=\n")
            f.write("HANA_PORT=443\n")
            f.write("HANA_USER=\n")
            f.write("HANA_PASSWORD=\n")
            f.write("DEFAULT_TABLE_NAME=EMBEDDINGS\n")
            f.write("TEST_MODE=false\n")
            f.write("ENABLE_CORS=true\n")
            f.write("LOG_LEVEL=INFO\n")
    
    # Load existing .env file
    load_dotenv(env_path)
    
    # Get user input for credentials
    print("Please enter your SAP HANA Cloud credentials:")
    hana_host = input("HANA Host (e.g., abc123-xyz.hanacloud.ondemand.com): ")
    hana_port = input(f"HANA Port [{os.getenv('HANA_PORT', '443')}]: ") or os.getenv('HANA_PORT', '443')
    hana_user = input("HANA User: ")
    hana_password = input("HANA Password: ")
    table_name = input(f"Default Table Name [{os.getenv('DEFAULT_TABLE_NAME', 'EMBEDDINGS')}]: ") or os.getenv('DEFAULT_TABLE_NAME', 'EMBEDDINGS')
    
    # Update .env file
    set_key(env_path, "HANA_HOST", hana_host)
    set_key(env_path, "HANA_PORT", hana_port)
    set_key(env_path, "HANA_USER", hana_user)
    set_key(env_path, "HANA_PASSWORD", hana_password)
    set_key(env_path, "DEFAULT_TABLE_NAME", table_name)
    set_key(env_path, "TEST_MODE", "false")
    
    print(f"Credentials updated in {env_path}")
    print("To use these credentials in your application, restart your application")

if __name__ == "__main__":
    update_env()