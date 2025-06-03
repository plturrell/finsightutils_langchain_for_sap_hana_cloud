#!/bin/bash
# Script to deploy the API to Vercel

set -e

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "Vercel CLI not found. Installing..."
    npm install -g vercel
fi

# Check if user is logged in to Vercel
if ! vercel whoami &> /dev/null; then
    echo "Please log in to Vercel:"
    vercel login
fi

# Check if .env file exists for environment variables
if [ ! -f .env.vercel ]; then
    echo "Creating .env.vercel template..."
    cat > .env.vercel << EOL
HANA_HOST=your-hana-host.hanacloud.ondemand.com
HANA_PORT=443
HANA_USER=your_username
HANA_PASSWORD=your_password
LOG_LEVEL=INFO
EMBEDDING_MODEL=all-MiniLM-L6-v2
EOL
    echo "Please update .env.vercel with your SAP HANA Cloud credentials"
    exit 1
fi

# Create Vercel configuration if not exists
if [ ! -f vercel.json ]; then
    echo "Error: vercel.json not found. Please create it using the template."
    exit 1
fi

# Deploy to Vercel
echo "Deploying to Vercel..."
vercel --env-file .env.vercel --prod

echo "Deployment complete!"
echo "Visit your project in the Vercel dashboard to view details and monitor performance."