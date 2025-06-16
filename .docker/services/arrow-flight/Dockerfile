# Stage 1: Builder
FROM python:3.10-slim as builder

ARG VERSION=1.0.0
ARG BUILD_DATE=unknown
ARG GIT_COMMIT=unknown
ARG TARGETPLATFORM="linux/amd64"

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

# Install Python dependencies with focus on Arrow Flight
RUN pip install --no-cache-dir --upgrade pip setuptools && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r api_requirements.txt && \
    pip install --no-cache-dir pyarrow>=10.0.0 fastapi uvicorn

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
LABEL org.opencontainers.image.title="LangChain SAP HANA Arrow Flight Service" \
      org.opencontainers.image.description="Arrow Flight Service for LangChain Integration with SAP HANA Cloud" \
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

# Copy only necessary application code for Arrow Flight
COPY --chown=appuser:appuser api/core /app/api/core
COPY --chown=appuser:appuser api/routes/flight.py /app/api/routes/flight.py
COPY --chown=appuser:appuser api/models/flight_models.py /app/api/models/flight_models.py
COPY --chown=appuser:appuser api/utils /app/api/utils
COPY --chown=appuser:appuser api/__init__.py /app/api/__init__.py
COPY --chown=appuser:appuser api/database.py /app/api/database.py
COPY --chown=appuser:appuser api/db.py /app/api/db.py
COPY --chown=appuser:appuser api/config.py /app/api/config.py
COPY --chown=appuser:appuser api/settings.py /app/api/settings.py
COPY --chown=appuser:appuser api/version.py /app/api/version.py
COPY --chown=appuser:appuser langchain_hana/gpu/arrow_flight_client.py /app/langchain_hana/gpu/arrow_flight_client.py
COPY --chown=appuser:appuser langchain_hana/gpu/arrow_flight_server.py /app/langchain_hana/gpu/arrow_flight_server.py
COPY --chown=appuser:appuser langchain_hana/gpu/arrow_gpu_memory_manager.py /app/langchain_hana/gpu/arrow_gpu_memory_manager.py
COPY --chown=appuser:appuser langchain_hana/gpu/arrow_flight_vectorstore.py /app/langchain_hana/gpu/arrow_flight_vectorstore.py
COPY --chown=appuser:appuser langchain_hana/gpu/arrow_flight_multi_gpu.py /app/langchain_hana/gpu/arrow_flight_multi_gpu.py
COPY --chown=appuser:appuser langchain_hana/gpu/vector_serialization.py /app/langchain_hana/gpu/vector_serialization.py
COPY --chown=appuser:appuser langchain_hana/gpu/utils.py /app/langchain_hana/gpu/utils.py
COPY --chown=appuser:appuser langchain_hana/gpu/__init__.py /app/langchain_hana/gpu/__init__.py
COPY --chown=appuser:appuser langchain_hana/__init__.py /app/langchain_hana/__init__.py
COPY --chown=appuser:appuser langchain_hana/error_utils.py /app/langchain_hana/error_utils.py

# Create required directories
RUN mkdir -p /app/langchain_hana/vectorstores /app/logs && \
    chown -R appuser:appuser /app/langchain_hana/vectorstores /app/logs

# Create a health check file
RUN echo 'from fastapi import APIRouter\n\nrouter = APIRouter()\n\n@router.get("/health")\ndef health():\n    return {"status": "ok", "message": "Arrow Flight API is healthy"}\n' > /app/api/health.py && \
    chown appuser:appuser /app/api/health.py

# Switch to the non-root user
USER appuser

# Set environment variables
ENV FLIGHT_HOST=0.0.0.0 \
    FLIGHT_PORT=8815 \
    LOG_LEVEL=INFO \
    ENABLE_CORS=true

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000
EXPOSE 8815

# Create entrypoint script
RUN echo '#!/bin/bash\n\
mkdir -p /app/logs\n\
touch /app/logs/flight.log\n\
echo "Starting Arrow Flight server..."\n\
exec uvicorn api.core.main:app --host 0.0.0.0 --port 8000 --log-level ${LOG_LEVEL:-info}\n\
' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]