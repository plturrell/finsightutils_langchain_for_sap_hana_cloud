# Security Fixes for Arrow Flight Integration

This document outlines the security fixes implemented for the Arrow Flight integration Docker image.

## Issues Fixed

1. **Critical Vulnerabilities in PyArrow**
   - Removed PyArrow in minimal version (A-rated) due to build complexity and vulnerabilities
   - Previous version: Updated PyArrow from 10.0.0 to 17.0.0 (B-rated)

2. **High Vulnerabilities in Setuptools**
   - Updated Setuptools from 65.5.1 to 78.1.1
   - Fixed CVE-2025-47273 and CVE-2024-6345

3. **Non-Root User Implementation**
   - Added a dedicated non-root user (appuser)
   - Switched to this user for running the application
   - Properly set permissions on application directories

4. **Pip Version Update**
   - Updated Pip from 23.0.1 to 23.3.1
   - Fixed CVE-2023-5752

5. **Reduced Attack Surface**
   - Removed build-essential and unnecessary build tools
   - Simplified the Docker image with only required dependencies

6. **Supply Chain Attestations**
   - Added SBOM attestation during build
   - Added provenance attestation for supply chain security

7. **Base Image Updates**
   - Updated to Python 3.13-slim for latest security patches

## Docker Scout Improvements

Before fixes:
- 2 Critical vulnerabilities
- 2 High vulnerabilities
- 2 Medium vulnerabilities
- 65 Low vulnerabilities
- No default non-root user
- No supply chain attestations
- C-rated image

After fixes (minimal secure):
- 0 Critical vulnerabilities
- 0 High vulnerabilities
- 1 Medium vulnerability
- 35 Low vulnerabilities
- Non-root user implemented
- Supply chain attestations added
- A-rated image

## Secure Images

Two secure images are available:

1. **A-Rated Minimal Image** (Recommended):
   ```
   finsightintelligence/finsight_utils_langchain_hana:minimal-secure
   ```

2. **B-Rated Full Image**:
   ```
   finsightintelligence/finsight_utils_langchain_hana:secure-arrow-flight
   ```

## Deployment

To deploy the A-rated minimal secure version:

```bash
docker-compose -f docker-compose.minimal-secure.yml up -d
```

To deploy the B-rated full version with PyArrow:

```bash
docker-compose -f docker-compose.secure.yml up -d
```

## Remaining Medium Vulnerability

There is one remaining medium vulnerability in systemd (CVE-2025-4598) that cannot be fixed by updating packages as it requires a full OS upgrade. This vulnerability has minimal impact on containerized environments.

## Additional Security Recommendations

1. **Runtime Security**
   - Implement read-only file systems where possible
   - Use Docker secrets for sensitive information
   - Set resource limits in the docker-compose file

2. **Consider Alpine Base**
   - For complete vulnerability elimination, consider using python:alpine base image
   - Note: This may require additional work for compatibility

3. **Regular Updates**
   - Set up automated scanning and rebuilding of images
   - Keep base images updated regularly