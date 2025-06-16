#\!/bin/bash
# Setup script for LangChain Integration for SAP HANA Cloud

echo "Setting up LangChain Integration for SAP HANA Cloud..."

# Ensure pip is available
if \! command -v pip3 &> /dev/null; then
    echo "Error: pip3 is not installed. Please install Python 3 and pip first."
    exit 1
fi

# Install required packages
echo "Installing required packages..."
pip3 install -r requirements.txt

# Create cache directory
echo "Creating cache directory..."
mkdir -p cache

echo "Setup complete\!"
echo ""
echo "Next steps:"
echo "1. Run the connection test: python3 test_connection.py"
echo "2. Run the data insights generator: python3 data_insights_generator.py"
EOF < /dev/null