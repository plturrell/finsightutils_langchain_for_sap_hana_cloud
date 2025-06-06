#!/bin/bash

# Create archive directory
ARCHIVE_DIR="./archived_deployment_files"
mkdir -p "$ARCHIVE_DIR"

# Files to archive (old deployment configurations)
FILES_TO_ARCHIVE=(
  "docker-compose.cpu.yml"
  "docker-compose.nvidia.yml"
  "docker-compose.yml"
  "Dockerfile.nvidia"
  "deploy_t4_app.sh"
  "deploy_t4_backend.sh"
  "deploy_to_jupyter_vm.sh"
  "deploy_to_vm.sh"
  "deploy_vercel_frontend.sh"
  "vercel.json.bak"
  ".env.frontend.vercel.prod"
  ".env.vercel"
)

# Move files to archive directory
for file in "${FILES_TO_ARCHIVE[@]}"; do
  if [ -f "$file" ]; then
    echo "Moving $file to $ARCHIVE_DIR/"
    mv "$file" "$ARCHIVE_DIR/"
  else
    echo "File $file not found, skipping"
  fi
done

echo "Deployment files have been archived to $ARCHIVE_DIR/"
echo "New deployment configuration uses:"
echo "  - backend/Dockerfile.nvidia for NVIDIA GPU backend"
echo "  - docker-compose.backend.yml for backend deployment"
echo "  - vercel.frontend.json for Vercel frontend deployment"

# Create .gitignore in archived directory
echo "# Archived deployment files - not used in current deployment" > "$ARCHIVE_DIR/.gitignore"
echo "*" >> "$ARCHIVE_DIR/.gitignore"
echo "!.gitignore" >> "$ARCHIVE_DIR/.gitignore"
echo "!README.md" >> "$ARCHIVE_DIR/.gitignore"

# Create README in archived directory
cat > "$ARCHIVE_DIR/README.md" << 'EOF'
# Archived Deployment Files

This directory contains archived deployment configuration files that are no longer used in the current deployment setup.

The current deployment uses:
- `backend/Dockerfile.nvidia` - Docker configuration for NVIDIA GPU backend
- `docker-compose.backend.yml` - Docker Compose configuration for backend deployment
- `vercel.frontend.json` - Vercel configuration for frontend deployment

These archived files are kept for reference purposes only.
EOF

echo "Done!"