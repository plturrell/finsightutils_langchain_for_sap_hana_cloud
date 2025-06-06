# Deployment Scripts

This directory contains scripts for deploying the application to various environments.

## Available Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `deploy-nvidia-stack.sh` | Deploy full NVIDIA stack | `./deploy-nvidia-stack.sh` |
| `deploy-nvidia-vercel.sh` | Deploy with NVIDIA backend and Vercel frontend | `./deploy-nvidia-vercel.sh` |
| `deploy-to-nvidia-t4.sh` | Deploy to NVIDIA T4 GPU instance | `./deploy-to-nvidia-t4.sh` |
| `deploy-to-vercel.sh` | Deploy to Vercel | `./deploy-to-vercel.sh` |
| `build-nvidia-local.sh` | Build NVIDIA container locally | `./build-nvidia-local.sh` |

## NVIDIA Stack Deployment

The `deploy-nvidia-stack.sh` script orchestrates the entire deployment process:

1. GitHub repository synchronization
2. NVIDIA T4 GPU backend deployment
3. Vercel frontend deployment with TensorRT optimization

```bash
./deploy-nvidia-stack.sh [--skip-github] [--skip-backend] [--skip-frontend] [--backend-url URL]
```

### Options

| Option | Description |
|--------|-------------|
| `--skip-github` | Skip GitHub synchronization step |
| `--skip-backend` | Skip NVIDIA T4 backend deployment |
| `--skip-frontend` | Skip Vercel frontend deployment |
| `--backend-url` | Specify backend URL (default: auto-detected) |
| `--help` | Show help message |

## NVIDIA Backend with Vercel Frontend

The `deploy-nvidia-vercel.sh` script deploys the frontend to Vercel with proper configuration for NVIDIA T4 GPU backend:

```bash
BACKEND_URL=https://your-backend-url.example.com ./deploy-nvidia-vercel.sh
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `BACKEND_URL` | T4 GPU backend URL | `https://jupyter0-513syzm60.brevlab.com` |
| `VERCEL_PROJECT_NAME` | Vercel project name | `sap-hana-langchain-t4` |
| `ENVIRONMENT` | Deployment environment | `production` |
| `TENSORRT_ENABLED` | Enable TensorRT | `true` |
| `TENSORRT_PRECISION` | TensorRT precision | `int8` |

## NVIDIA T4 GPU Deployment

The `deploy-to-nvidia-t4.sh` script deploys the backend to an NVIDIA T4 GPU instance:

```bash
./deploy-to-nvidia-t4.sh
```

## Build NVIDIA Container Locally

The `build-nvidia-local.sh` script builds the NVIDIA container locally:

```bash
./build-nvidia-local.sh [--base-image IMAGE] [--tag TAG]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--base-image` | Base NVIDIA image | `nvcr.io/nvidia/pytorch:23.12-py3` |
| `--tag` | Image tag | `langchain-hana-nvidia:latest` |
| `--push` | Push to registry | `false` |
| `--registry` | Registry URL | None |