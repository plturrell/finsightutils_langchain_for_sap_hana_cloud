#!/bin/bash

# Script to launch the complete SAP HANA Cloud LangChain Integration demo
# with React frontend and Arrow Flight backend

echo "🚀 Starting SAP HANA Cloud LangChain Integration Demo"
echo "======================================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
  echo "❌ Docker is not running. Please start Docker and try again."
  exit 1
fi

# Check for docker-compose
if ! command -v docker-compose > /dev/null 2>&1; then
  echo "❌ docker-compose is not installed. Please install docker-compose and try again."
  exit 1
fi

# Print configuration information
echo "🔧 Configuration:"
echo "  - Frontend port: 3000"
echo "  - Backend API port: 8000"
echo "  - Arrow Flight port: 8815"
echo "  - Using test mode: Yes (no database connection required)"
echo ""

# Ask for confirmation
read -p "Continue with this configuration? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
  echo "Aborted."
  exit 1
fi

# Clean up any existing containers
echo "🧹 Cleaning up existing containers..."
docker-compose -f docker-compose.frontend.yml down -v

# Pull the latest images
echo "📥 Pulling latest container images..."
docker-compose -f docker-compose.frontend.yml pull

# Build and start the containers
echo "🏗️ Building and starting containers..."
docker-compose -f docker-compose.frontend.yml up -d --build

# Wait for services to start
echo "⏳ Waiting for services to start..."
echo "  This may take a minute or two for the first startup."
echo ""

# Initialize countdown
TIMEOUT=60
while [ $TIMEOUT -gt 0 ]; do
  echo -ne "  Waiting for backend... $TIMEOUT seconds remaining\r"
  
  # Check if backend is healthy
  if curl -s http://localhost:8000/health/ping | grep -q "pong"; then
    echo -e "\n✅ Backend is ready!"
    break
  fi
  
  sleep 1
  TIMEOUT=$((TIMEOUT-1))
done

if [ $TIMEOUT -eq 0 ]; then
  echo -e "\n❌ Backend did not start within the timeout period."
  echo "Please check the logs with: docker-compose -f docker-compose.frontend.yml logs backend"
  exit 1
fi

# Verify Flight server is running
echo "🔍 Verifying Arrow Flight server..."
if curl -s http://localhost:8000/flight/info | grep -q "running"; then
  echo "✅ Arrow Flight server is running"
else
  echo "⚠️ Arrow Flight server might not be fully initialized."
  echo "Starting it now..."
  curl -s -X POST http://localhost:8000/flight/start > /dev/null
  sleep 5
fi

# Check frontend is accessible
echo "🔍 Verifying frontend is accessible..."
if curl -s -o /dev/null -w "%{http_code}" http://localhost:3000 | grep -q "200"; then
  echo "✅ Frontend is accessible"
else
  echo "⚠️ Frontend might need more time to initialize."
  echo "Please wait a moment and try accessing http://localhost:3000 manually."
fi

echo ""
echo "🎉 Demo environment is ready!"
echo "======================================================="
echo "📊 Dashboard: http://localhost:3000"
echo "🔍 API: http://localhost:8000"
echo "✈️ Arrow Flight: http://localhost:8815"
echo ""
echo "📝 Useful commands:"
echo "  - View logs: docker-compose -f docker-compose.frontend.yml logs -f"
echo "  - Stop demo: docker-compose -f docker-compose.frontend.yml down"
echo "  - Restart demo: ./run_frontend_demo.sh"
echo ""
echo "📚 Documentation: See FRONTEND_ARROW_INTEGRATION.md for details"