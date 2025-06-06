#!/bin/bash
set -e

echo "Starting Blue-Green deployment health checker..."

# Run the health check script
exec python /app/healthcheck.py