# Security Scan Summary

Initial template created: 2025-06-09

## Current Security Status

This is an initial template. Actual scan results will be populated automatically by the GitHub Actions workflow.

### CPU Image: finsightintelligence/finsight_utils_langchain_hana:cpu-latest

Pending first automated scan.

### GPU Image: finsightintelligence/finsight_utils_langchain_hana:gpu-latest

Pending first automated scan.

## Known Vulnerabilities Being Addressed

- **setuptools < 78.1.1**: High severity vulnerabilities including CVE-2025-47273 and CVE-2024-6345
- **starlette < 0.40.0**: High severity vulnerability CVE-2024-47874
- **Running as root**: Security best practice violation

## Implemented Fixes

- Updated dependencies in security-requirements.txt
- Non-root user execution in Docker containers
- System package updates during build
- Reduced container attack surface
