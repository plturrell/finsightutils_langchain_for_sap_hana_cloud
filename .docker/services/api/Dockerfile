# Stage 1: Builder
FROM python:3.10-slim as builder

ARG VERSION=1.0.0
ARG BUILD_DATE=unknown
ARG GIT_COMMIT=unknown
ARG TARGETPLATFORM="linux/amd64"
ARG ENABLE_GPU=false

# Set environment variables for build stage
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Copy requirements files
COPY requirements.txt .
COPY api/requirements.txt api_requirements.txt
COPY security-requirements.txt security_requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools>=78.1.1 && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r api_requirements.txt && \
    pip install --no-cache-dir -r security_requirements.txt && \
    pip install --no-cache-dir hdbcli rdflib && \
    pip install --no-cache-dir numpy>=1.24.0 scipy pandas scikit-learn langchain>=0.1.0 langchain-core>=0.1.0 fastapi>=0.111.1 starlette>=0.40.0

# Install optional GPU dependencies if enabled
RUN if [ "$ENABLE_GPU" = "true" ]; then \
        pip install --no-cache-dir torch>=2.0.0 nvidia-cuda-runtime-cu12 && \
        pip install --no-cache-dir sentence-transformers>=2.2.2; \
    else \
        pip install --no-cache-dir torch>=2.0.0 --index-url https://download.pytorch.org/whl/cpu && \
        pip install --no-cache-dir sentence-transformers>=2.2.2; \
    fi

# Stage 2: Runtime
FROM python:3.10-slim

ARG VERSION=1.0.0
ARG BUILD_DATE=unknown
ARG GIT_COMMIT=unknown
ARG TARGETPLATFORM="linux/amd64"

# Set environment variables for runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONPATH=/app

# Set image metadata
LABEL org.opencontainers.image.title="LangChain SAP HANA Integration API" \
      org.opencontainers.image.description="API Service for LangChain Integration with SAP HANA Cloud" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${GIT_COMMIT}" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.architecture="${TARGETPLATFORM}"

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install security patches
RUN apt-get update && apt-get upgrade -y && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create a non-root user and set permissions
RUN groupadd -r appuser && useradd -r -g appuser -m -s /bin/bash appuser

# Set up working directory
WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages/
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=appuser:appuser . /app/

# Create required directories
RUN mkdir -p /app/cache /app/logs && \
    chown -R appuser:appuser /app/cache /app/logs

# Create a simple health check file
RUN echo 'from fastapi import APIRouter\n\nrouter = APIRouter()\n\n@router.get("/health")\ndef health():\n    return {"status": "ok", "message": "API is healthy"}\n' > /app/api/health.py && \
    chown appuser:appuser /app/api/health.py

# Switch to the non-root user
USER appuser

# Set environment variables
ENV TEST_MODE=false \
    ENABLE_CORS=true \
    LOG_LEVEL=INFO \
    FLIGHT_AUTO_START=true \
    FLIGHT_HOST=0.0.0.0 \
    FLIGHT_PORT=8815

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000
EXPOSE 8815

# Create entrypoint script
RUN echo '#!/bin/bash\n\
mkdir -p /app/logs\n\
touch /app/logs/api.log\n\
echo "Starting API server..."\n\
exec uvicorn api.core.main:app --host 0.0.0.0 --port 8000 --log-level ${LOG_LEVEL:-info}\n\
' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]