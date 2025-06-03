# CI/CD Guide for langchain-integration-for-sap-hana-cloud

This document provides an overview of the Continuous Integration and Continuous Deployment (CI/CD) setup for this project.

## Overview

The CI/CD pipeline automates the following processes:

1. **Continuous Integration (CI)**:
   - Code linting and static analysis
   - Type checking
   - Unit tests
   - Package building and validation
   - Docker image building

2. **Continuous Deployment (CD)**:
   - Package publishing to PyPI
   - Docker image publishing to GitHub Container Registry
   - Deployment to cloud environments

## Local Development Setup

### Prerequisites

- Python 3.9+ installed
- Git installed
- Access to the GitHub repository

### Setting Up Local Development Environment

Run the setup script to install pre-commit hooks and development dependencies:

```bash
./scripts/setup_local_dev.sh
```

This script:
- Installs pre-commit and required hooks
- Sets up development dependencies
- Configures git hooks for pre-commit and pre-push

### Pre-commit Hooks

The following pre-commit hooks are configured:

- **Code Formatting**:
  - trailing-whitespace: Trims trailing whitespace
  - end-of-file-fixer: Ensures files end with a newline
  - ruff: Lints and fixes Python code issues
  - isort: Sorts imports

- **Code Quality**:
  - check-ast: Validates Python syntax
  - mypy: Performs static type checking
  - pytest-check: Runs unit tests

- **Security**:
  - detect-private-key: Prevents committing private keys
  - check-merge-conflict: Detects unresolved merge conflicts

### Git Workflow

Follow these best practices for working with git:

1. **Remote Repositories**:
   - **origin**: The main SAP repository (https://github.com/SAP/langchain-integration-for-sap-hana-cloud)
   - **enhanced**: @plturrell's enhanced repository (https://github.com/plturrell/langchain-integration-for-sap-hana-cloud)
   - Both repositories are synchronized automatically with the provided scripts

2. **Branch Strategy**:
   - `main`: Production-ready code
   - `dev`: Development branch for integration
   - Feature branches: Named like `feature/your-feature-name`
   - Bug fix branches: Named like `fix/issue-description`

3. **Commit Messages**:
   - Use clear, descriptive commit messages
   - Start with a verb (Add, Fix, Update, etc.)
   - Reference issue numbers if applicable

4. **Pull Requests**:
   - Create pull requests to merge into `dev` or `main`
   - Ensure CI checks pass before requesting review
   - Request reviews from team members

5. **Automatic Synchronization**:
   - Commits are automatically pushed to both repositories via git hooks
   - Version tags are pushed to both repositories via the release script

## CI Pipeline

The CI pipeline is triggered on:
- Push to `main` and `dev` branches
- Pull requests to `main` and `dev` branches
- Manual trigger via GitHub Actions interface

### CI Jobs

1. **Lint**:
   - Runs code linting with ruff
   - Performs type checking with mypy

2. **Test**:
   - Runs unit tests with pytest
   - Tests across multiple Python versions (3.9, 3.10, 3.11)
   - Generates code coverage reports

3. **Build Package**:
   - Builds Python package
   - Validates package structure
   - Uploads package artifact

4. **Build API**:
   - Builds Docker image for API
   - Pushes to GitHub Container Registry (on push to `main` or `dev`)

## CD Pipeline

The CD pipeline is triggered on:
- Push of version tags (e.g., `v1.0.0`)
- Manual trigger via GitHub Actions interface

### CD Jobs

1. **Publish Package**:
   - Builds Python package
   - Publishes to PyPI

2. **Deploy API**:
   - Builds and tags Docker image with version
   - Pushes to GitHub Container Registry

3. **Deploy to Cloud**:
   - Deploys to specified environment (staging/production)
   - Can be customized for your cloud provider (AWS, GCP, Azure)

## Versioning

This project follows semantic versioning (SEMVER):
- **MAJOR**: Incompatible API changes
- **MINOR**: Added functionality (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

To create a new release:

```bash
# Use the provided script to tag and push to both repositories
./scripts/tag_release.sh 1.0.0
```

This script will:
1. Create an annotated tag (v1.0.0)
2. Push the tag to both the main SAP repository and @plturrell's repository
3. Trigger the CD pipeline in both repositories

## Environments

The deployment workflow supports multiple environments:
- **Development**: For testing new features
- **Staging**: For pre-release validation
- **Production**: For end-user access

## Security Considerations

- Secrets are stored in GitHub Secrets
- No hardcoded credentials in the codebase
- Regular dependency updates
- Pre-commit hooks to detect security issues

## Troubleshooting

### Common Issues

1. **CI Failures**:
   - Check the GitHub Actions logs for details
   - Run pre-commit hooks locally to catch issues before pushing

2. **CD Failures**:
   - Verify secrets and environment variables are correctly set
   - Check if the tag follows semantic versioning format

3. **Local Development Issues**:
   - Run `pre-commit run --all-files` to debug hook issues
   - Update dependencies with `pip install -e .`

### Getting Help

If you encounter issues with the CI/CD pipeline:
1. Check the GitHub Actions logs
2. Review this documentation
3. Create an issue in the GitHub repository