FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY healthcheck.py .
COPY entrypoint.sh .

# Make entrypoint executable
RUN chmod +x entrypoint.sh

# Set environment variables
ENV CHECK_INTERVAL=30
ENV BLUE_URL=http://api-blue:8000/health/status
ENV GREEN_URL=http://api-green:8000/health/status
ENV TRAEFIK_API_URL=http://traefik:8080/api

# Run the healthcheck service
ENTRYPOINT ["/app/entrypoint.sh"]