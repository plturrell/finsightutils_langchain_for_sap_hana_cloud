#!/bin/sh

# Enhanced entrypoint script for production environment
# Handles environment variable injection and runtime configuration

# Default values
REACT_APP_API_URL=${REACT_APP_API_URL:-http://localhost:8000}
FLIGHT_URL=${FLIGHT_URL:-grpc://localhost:8815}

echo "Configuring frontend with:"
echo "API URL: $REACT_APP_API_URL"
echo "Flight URL: $FLIGHT_URL"

# Update nginx config with the API URL
sed -i "s|REACT_APP_API_URL|${REACT_APP_API_URL}|g" /etc/nginx/conf.d/nginx.conf

# Update runtime environment variables for React in the index.html
if [ -f /usr/share/nginx/html/index.html ]; then
  # Define the script to inject environment variables
  ENV_SCRIPT="<script>window.env = { 
    REACT_APP_API_URL: '${REACT_APP_API_URL}',
    FLIGHT_URL: '${FLIGHT_URL}',
    NODE_ENV: 'production',
    VERSION: '$(cat /usr/share/nginx/html/version.txt 2>/dev/null || echo "1.0.0")',
    BUILD_TIME: '$(date -u +"%Y-%m-%dT%H:%M:%SZ")'
  };</script>"
  
  # Insert the script right before the closing </head> tag
  # Use a different delimiter for sed to avoid issues with the / in URLs
  sed -i "s|</head>|${ENV_SCRIPT}</head>|g" /usr/share/nginx/html/index.html
  
  echo "Environment variables injected into index.html"
fi

# Generate runtime configuration file
cat > /usr/share/nginx/html/config.json << EOF
{
  "apiUrl": "${REACT_APP_API_URL}",
  "flightUrl": "${FLIGHT_URL}",
  "environment": "production",
  "features": {
    "tensorRT": true,
    "multiGPU": true,
    "zeroShot": true,
    "vectorVisualization": true
  }
}
EOF

echo "Runtime configuration generated at /usr/share/nginx/html/config.json"

# Test nginx config
echo "Testing Nginx configuration..."
nginx -t

# Start nginx
echo "Starting Nginx server..."
exec "$@"