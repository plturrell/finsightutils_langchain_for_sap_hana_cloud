#!/bin/bash

# Setup script for local development environment
# This script installs pre-commit hooks and other development tools
# and configures GitHub remote repositories

set -e

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 could not be found. Please install Python 3 and try again."
    exit 1
fi

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    echo "pip could not be found. Please install pip and try again."
    exit 1
fi

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "git could not be found. Please install git and try again."
    exit 1
fi

# Install pre-commit
echo "Installing pre-commit..."
pip install pre-commit

# Install development dependencies
echo "Installing development dependencies..."
pip install -e ".[dev,test,lint]"

# Install pre-commit hooks
echo "Setting up pre-commit hooks..."
pre-commit install

# Create git hooks directory if it doesn't exist
mkdir -p .git/hooks

# Create pre-push hook
echo "Creating pre-push hook..."
cat > .git/hooks/pre-push << 'EOF'
#!/bin/bash

# Run tests before pushing
echo "Running tests before push..."
pytest tests/unit_tests/

# Check exit code
if [ $? -ne 0 ]; then
    echo "Tests failed. Push aborted."
    exit 1
fi

echo "Tests passed. Continuing with push."
exit 0
EOF

# Make pre-push hook executable
chmod +x .git/hooks/pre-push

# Setup git remotes for synchronization
echo "Setting up git remotes for synchronization..."

# Check if we're in a git repository
if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    echo "This is not a git repository. Initializing git..."
    git init
    git add .
    git commit -m "Initial commit"
fi

# Get current remotes
CURRENT_REMOTES=$(git remote)

# Add the main SAP repository if not already present
if ! echo "$CURRENT_REMOTES" | grep -q "origin"; then
    echo "Adding SAP remote repository as 'origin'..."
    git remote add origin https://github.com/SAP/langchain-integration-for-sap-hana-cloud.git
fi

# Add @plturrell's enhanced repository if not already present
if ! echo "$CURRENT_REMOTES" | grep -q "enhanced"; then
    echo "Adding @plturrell's enhanced repository as 'enhanced'..."
    git remote add enhanced https://github.com/plturrell/langchain-integration-for-sap-hana-cloud.git
fi

# Update all remotes
echo "Fetching updates from all remotes..."
git fetch --all

# Create a post-commit hook to automatically sync with remotes
echo "Creating post-commit hook for auto-sync..."
cat > .git/hooks/post-commit << 'EOF'
#!/bin/bash

# Auto-sync to remotes after commit
echo "Auto-syncing with remote repositories..."

# Push to origin (SAP repository)
git push origin HEAD || echo "Warning: Could not push to origin"

# Push to enhanced (@plturrell's repository)
git push enhanced HEAD || echo "Warning: Could not push to enhanced"

echo "Auto-sync complete"
EOF

# Make post-commit hook executable
chmod +x .git/hooks/post-commit

# Create a tag-and-release script
echo "Creating tag-and-release script..."
cat > ./scripts/tag_release.sh << 'EOF'
#!/bin/bash

# Script to create a new version tag and trigger the CD pipeline

# Check if version argument is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 VERSION"
    echo "Example: $0 1.0.0"
    exit 1
fi

VERSION="v$1"

# Validate version format
if ! [[ $VERSION =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Error: Version must be in format vX.Y.Z"
    exit 1
fi

# Create tag
echo "Creating tag $VERSION..."
git tag -a "$VERSION" -m "Release $VERSION"

# Push tag to origin
echo "Pushing tag to origin..."
git push origin "$VERSION"

# Push tag to enhanced
echo "Pushing tag to enhanced..."
git push enhanced "$VERSION"

echo "Release $VERSION has been tagged and pushed to both repositories."
echo "The CD pipeline should now be triggered automatically."
EOF

# Make tag-and-release script executable
chmod +x ./scripts/tag_release.sh

echo "Local development environment setup complete!"
echo "Pre-commit hooks are now installed and will run automatically on commit."
echo "Unit tests will run automatically before pushing."
echo "Changes will automatically sync to both repositories after commits."
echo ""
echo "To create a new release:"
echo "  ./scripts/tag_release.sh VERSION"
echo "  Example: ./scripts/tag_release.sh 1.0.0"