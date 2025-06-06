# GitHub Synchronization Guide

This guide explains how to synchronize your enhanced SAP HANA Cloud LangChain Integration with a GitHub repository.

## Initial Setup

To set up GitHub synchronization for the first time:

```bash
# Navigate to your project directory
cd /path/to/langchain-integration-for-sap-hana-cloud

# Run the setup script
./scripts/setup_github_sync.sh --repo-url https://github.com/yourusername/your-repo.git --token YOUR_GITHUB_TOKEN
```

### Setup Options

| Option | Description | Default |
|--------|-------------|---------|
| `--repo-url` | GitHub repository URL (required) | - |
| `--branch` | Branch name | `main` |
| `--remote` | Remote name | `github` |
| `--token` | GitHub personal access token | - |

## Synchronizing Changes

Once set up, you can synchronize changes with GitHub using:

```bash
# Basic synchronization (pull and push)
./scripts/sync_to_github.sh

# Push local changes only
./scripts/sync_to_github.sh --push-only

# Pull remote changes only
./scripts/sync_to_github.sh --pull-only

# Custom commit message
./scripts/sync_to_github.sh --message "Update TensorRT optimizations"
```

### Sync Options

| Option | Description | Default |
|--------|-------------|---------|
| `--remote` | Remote name | `github` |
| `--branch` | Branch name | `main` |
| `--message` | Commit message | `Auto-sync: $(date)` |
| `--push-only` | Only push changes, don't pull | `false` |
| `--pull-only` | Only pull changes, don't push | `false` |
| `--force` | Force push (use with caution) | `false` |

## Setting Up Automatic Synchronization

You can set up automatic synchronization using cron (on Linux/macOS) or Task Scheduler (on Windows).

### Using Cron (Linux/macOS)

1. Open your crontab file:
   ```bash
   crontab -e
   ```

2. Add a line to run the sync script periodically (e.g., every 30 minutes):
   ```
   */30 * * * * cd /path/to/langchain-integration-for-sap-hana-cloud && ./scripts/sync_to_github.sh
   ```

### Using Task Scheduler (Windows)

1. Create a batch script called `sync_github.bat`:
   ```batch
   @echo off
   cd /path/to/langchain-integration-for-sap-hana-cloud
   bash scripts/sync_to_github.sh
   ```

2. Open Task Scheduler and create a new task to run this batch file at your desired interval.

## GitHub Actions Integration

For more advanced synchronization, you can use GitHub Actions to automatically sync changes when they're pushed to GitHub.

1. Create a `.github/workflows/sync.yml` file in your repository:

```yaml
name: Sync Changes

on:
  push:
    branches: [main]
  schedule:
    - cron: '0 */6 * * *'  # Run every 6 hours

jobs:
  sync:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
        with:
          fetch-depth: 0
          
      - name: Configure Git
        run: |
          git config --global user.name "GitHub Actions Bot"
          git config --global user.email "actions@github.com"
          
      - name: Sync Changes
        run: |
          # Add your synchronization commands here
          # For example, syncing with another repo or service
```

## Troubleshooting

### Authentication Issues

If you encounter authentication issues:

1. Ensure your GitHub token has the necessary permissions (repo scope)
2. Check that the token is correctly included in the repository URL
3. Verify the token hasn't expired

To update your token:

```bash
./scripts/setup_github_sync.sh --repo-url https://github.com/yourusername/your-repo.git --token YOUR_NEW_TOKEN
```

### Merge Conflicts

If you encounter merge conflicts:

1. Pull the changes without trying to merge:
   ```bash
   git fetch github main
   ```

2. View the differences:
   ```bash
   git diff github/main
   ```

3. Resolve conflicts manually:
   ```bash
   git pull github main
   # Resolve conflicts in the affected files
   git add .
   git commit -m "Resolve merge conflicts"
   ```

### Force Push

If you need to overwrite remote changes (use with caution):

```bash
./scripts/sync_to_github.sh --force
```

## Best Practices

- **Regular Syncing**: Sync frequently to minimize merge conflicts
- **Pull Before Push**: Always pull changes before pushing to avoid conflicts
- **Descriptive Messages**: Use clear commit messages to track changes
- **Branch Management**: Create feature branches for major changes
- **Token Security**: Store GitHub tokens securely and rotate them regularly