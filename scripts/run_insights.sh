#\!/bin/bash
# Run the Data Insights Generator

# Check for OpenAI API key
if [ -z "$OPENAI_API_KEY" ]; then
  echo "Warning: OPENAI_API_KEY environment variable not set."
  echo "Some features will be limited without an OpenAI API key."
  echo ""
fi

# Run connection test first
echo "Testing connection to SAP HANA Cloud..."
python3 test_connection.py
if [ $? -ne 0 ]; then
  echo "Connection test failed. Please check your configuration."
  exit 1
fi

# Run data insights generator
echo ""
echo "Starting Data Insights Generator..."
python3 data_insights_generator.py "$@"
EOF < /dev/null