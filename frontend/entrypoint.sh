#!/bin/sh

# Replace environment variables in the Nginx configuration
REACT_APP_API_URL=${REACT_APP_API_URL:-http://localhost:8000}

# Update nginx config with the API URL
sed -i "s|REACT_APP_API_URL|${REACT_APP_API_URL}|g" /etc/nginx/conf.d/nginx.conf

# Update runtime environment variables for React in the index.html
if [ -f /usr/share/nginx/html/index.html ]; then
  # Define the script to inject environment variables
  ENV_SCRIPT="<script>window.env = { REACT_APP_API_URL: '${REACT_APP_API_URL}' };</script>"
  
  # Insert the script right before the closing </head> tag
  sed -i "s|</head>|${ENV_SCRIPT}</head>|g" /usr/share/nginx/html/index.html
fi

# Start nginx
exec "$@"