# Stage 1: Build stage
FROM node:18-alpine as builder

ARG VERSION=1.0.0
ARG BUILD_DATE=unknown
ARG GIT_COMMIT=unknown
ARG TARGETPLATFORM="linux/amd64"
ARG API_URL=http://localhost:8000

# Set working directory
WORKDIR /app

# Copy package.json and package-lock.json
COPY frontend/package*.json ./

# Install dependencies
RUN npm ci

# Copy source code
COPY frontend/ ./

# Set environment variables
ENV REACT_APP_API_URL=${API_URL} \
    REACT_APP_VERSION=${VERSION} \
    REACT_APP_COMMIT=${GIT_COMMIT} \
    NODE_ENV=production

# Build the application
RUN npm run build

# Stage 2: Production stage with Nginx
FROM nginx:alpine

ARG VERSION=1.0.0
ARG BUILD_DATE=unknown
ARG GIT_COMMIT=unknown
ARG TARGETPLATFORM="linux/amd64"

# Set image metadata
LABEL org.opencontainers.image.title="LangChain SAP HANA Frontend" \
      org.opencontainers.image.description="Frontend UI for LangChain Integration with SAP HANA Cloud" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${GIT_COMMIT}" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.architecture="${TARGETPLATFORM}"

# Install dependencies
RUN apk add --no-cache bash curl

# Copy the built app from the builder stage
COPY --from=builder /app/build /usr/share/nginx/html

# Copy nginx configuration
COPY nginx/nginx.conf /etc/nginx/conf.d/default.conf

# Create a health check script
RUN echo '#!/bin/bash\n\
if curl -s -f http://localhost/health > /dev/null; then\n\
    exit 0\n\
else\n\
    exit 1\n\
fi' > /usr/local/bin/health_check.sh && \
    chmod +x /usr/local/bin/health_check.sh

# Create a simple health check endpoint
RUN mkdir -p /usr/share/nginx/html/health && \
    echo "OK" > /usr/share/nginx/html/health/index.html

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD /usr/local/bin/health_check.sh

# Expose ports
EXPOSE 80

# Create entrypoint script to handle env vars
RUN echo '#!/bin/bash\n\
# Replace environment variables in the frontend app\n\
if [ -n "$API_URL" ]; then\n\
    find /usr/share/nginx/html -type f -name "*.js" -exec sed -i "s|__API_URL__|$API_URL|g" {} \\;\n\
fi\n\
\n\
# Start Nginx\n\
nginx -g "daemon off;"\n\
' > /docker-entrypoint.sh && chmod +x /docker-entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/docker-entrypoint.sh"]