#!/bin/bash

# Test end-to-end integration of React frontend with Arrow Flight backend
echo "Starting end-to-end integration test..."

# Set up error handling
set -e

# Start Docker Compose with the frontend configuration
echo "Starting Docker containers..."
docker-compose -f docker-compose.frontend.yml down -v
docker-compose -f docker-compose.frontend.yml up -d

# Wait for services to be fully available
echo "Waiting for services to start..."
sleep 15

# Check if backend is running
echo "Checking backend health..."
BACKEND_HEALTH=$(curl -s http://localhost:8000/health)
if [[ $BACKEND_HEALTH == *"status"*"ok"* ]]; then
  echo "✅ Backend is healthy"
else
  echo "❌ Backend health check failed"
  echo $BACKEND_HEALTH
  exit 1
fi

# Check if Flight server is running
echo "Checking Arrow Flight server..."
FLIGHT_INFO=$(curl -s http://localhost:8000/flight/info)
if [[ $FLIGHT_INFO == *"host"*"status"*"running"* ]]; then
  echo "✅ Flight server is running"
else
  echo "❌ Flight server check failed"
  echo $FLIGHT_INFO
  exit 1
fi

# Check if frontend is accessible
echo "Checking frontend..."
FRONTEND_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:3000)
if [[ $FRONTEND_RESPONSE == "200" ]]; then
  echo "✅ Frontend is accessible"
else
  echo "❌ Frontend check failed with status code $FRONTEND_RESPONSE"
  exit 1
fi

echo "All services are running successfully!"
echo "Frontend: http://localhost:3000"
echo "Backend API: http://localhost:8000"
echo "Arrow Flight: http://localhost:8815"
echo ""
echo "Run the following command to view logs:"
echo "docker-compose -f docker-compose.frontend.yml logs -f"
echo ""
echo "Run the following command to stop the services:"
echo "docker-compose -f docker-compose.frontend.yml down"