#!/bin/bash

# Run the standardized API with versioned routes
echo "Starting the standardized API with versioned routes..."

# Set environment variables
export API_HOST="0.0.0.0"
export API_PORT="8000"
export API_LOG_LEVEL="INFO"
export ENVIRONMENT="development"

# Run the API using uvicorn
python -m uvicorn main_standardized:app --host "$API_HOST" --port "$API_PORT" --reload

echo "API server stopped."