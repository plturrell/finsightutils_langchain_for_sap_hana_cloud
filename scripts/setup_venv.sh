#!/bin/bash
# Setup virtual environment and install required dependencies for SAP HANA Cloud integration

# Exit on error
set -e

echo "Setting up virtual environment for SAP HANA Cloud integration..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install required packages
echo "Installing required packages..."
pip install -r requirements.txt

# Verify installation
echo "Verifying installation..."
python check_packages.py

echo "Setup complete. Use 'source venv/bin/activate' to activate the virtual environment."
echo "Then run 'python enhanced_test_connection.py' to test the connection to SAP HANA Cloud."