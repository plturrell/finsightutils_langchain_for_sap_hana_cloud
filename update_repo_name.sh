#!/bin/bash
# Script to update remote URLs after repository rename

set -e

echo "Updating repository remote URLs to finsightutils_langchain_for_sap_hana_cloud..."

# Save current branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

# Update the enhanced remote URL (your personal fork)
git remote set-url enhanced https://github.com/plturrell/finsightutils_langchain_for_sap_hana_cloud.git

# Verify the changes
echo "Updated remote URLs:"
git remote -v

echo ""
echo "Next steps:"
echo "1. Go to GitHub and rename your repository to 'finsightutils_langchain_for_sap_hana_cloud'"
echo "   at: https://github.com/plturrell/langchain-integration-for-sap-hana-cloud/settings"
echo "2. If you also have a fork of the SAP repository, you may want to update it as well"
echo ""
echo "After renaming on GitHub, test pushing to the new repository:"
echo "git push enhanced $CURRENT_BRANCH"
echo ""

# Update project references
if grep -q "langchain-integration-for-sap-hana-cloud" ./docker/Dockerfile.nvidia; then
    echo "Updating Docker label references..."
    sed -i.bak 's/langchain-integration-for-sap-hana-cloud/finsightutils_langchain_for_sap_hana_cloud/g' ./docker/Dockerfile.nvidia
    rm -f ./docker/Dockerfile.nvidia.bak
fi

# Update any other files that might reference the old name
echo "Checking for other references to update..."
FILES_TO_CHECK=$(grep -l "langchain-integration-for-sap-hana-cloud" $(git ls-files | grep -v "\.git") 2>/dev/null || true)

if [ -n "$FILES_TO_CHECK" ]; then
    echo "Found references in the following files:"
    echo "$FILES_TO_CHECK"
    echo ""
    echo "You may want to update these references manually."
fi

echo "Done!"