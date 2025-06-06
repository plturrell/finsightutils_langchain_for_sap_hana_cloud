# Development Scripts

This directory contains scripts for development workflows and environment setup.

## Available Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `setup-github-repo.sh` | Set up GitHub repository | `./setup-github-repo.sh --repo-url https://github.com/username/repo.git` |
| `setup-github-sync.sh` | Configure GitHub synchronization | `./setup-github-sync.sh --repo-url https://github.com/username/repo.git` |
| `sync-to-github.sh` | Synchronize with GitHub | `./sync-to-github.sh [--push-only\|--pull-only]` |
| `setup-local-dev.sh` | Set up local development environment | `./setup-local-dev.sh` |

## GitHub Repository Setup

The `setup-github-repo.sh` script sets up a GitHub remote for the project and pushes the initial commit.

```bash
./setup-github-repo.sh --repo-url https://github.com/username/repo.git --token YOUR_GITHUB_TOKEN
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--repo-url` | GitHub repository URL (required) | - |
| `--branch` | Branch name | `main` |
| `--remote` | Remote name | `github` |
| `--token` | GitHub personal access token | - |
| `--force` | Force push to remote | `false` |

## GitHub Synchronization

The `sync-to-github.sh` script synchronizes your local repository with a GitHub remote.

```bash
./sync-to-github.sh [--push-only|--pull-only]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--remote` | Remote name | `github` |
| `--branch` | Branch name | `main` |
| `--message` | Commit message | `Auto-sync: $(date)` |
| `--push-only` | Only push changes, don't pull | `false` |
| `--pull-only` | Only pull changes, don't push | `false` |
| `--force` | Force push (use with caution) | `false` |

## Local Development Environment

The `setup-local-dev.sh` script sets up your local development environment.

```bash
./setup-local-dev.sh
```

This script:
- Installs required dependencies
- Creates virtual environment
- Sets up configuration files
- Configures development tools