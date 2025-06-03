#!/bin/bash

# Script to set up git remotes for synchronization with both repositories
# This simplified script focuses only on the git configuration

set -e

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

# Remove existing remotes if they exist
git remote remove origin 2>/dev/null || echo "No existing origin remote to remove"
git remote remove enhanced 2>/dev/null || echo "No existing enhanced remote to remove"

# Add the SAP repository as origin
echo "Adding SAP repository as 'origin'..."
git remote add origin https://github.com/SAP/langchain-integration-for-sap-hana-cloud.git

# Add @plturrell's fork as enhanced
echo "Adding @plturrell's enhanced repository as 'enhanced'..."
git remote add enhanced https://github.com/plturrell/langchain-integration-for-sap-hana-cloud.git

# Update all remotes
echo "Fetching updates from all remotes..."
git fetch --all || echo "Warning: Could not fetch from remotes. This is expected if you're offline."

# Create a post-commit hook to automatically sync with remotes
echo "Creating post-commit hook for auto-sync..."
mkdir -p .git/hooks
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

echo "Git synchronization setup complete!"
echo "Changes will automatically sync to both repositories after commits."
echo ""
echo "To create a new release:"
echo "  ./scripts/tag_release.sh VERSION"
echo "  Example: ./scripts/tag_release.sh 1.0.0"