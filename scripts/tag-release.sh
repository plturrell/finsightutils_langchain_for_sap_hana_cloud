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
git push origin "$VERSION" || echo "Warning: Could not push tag to origin. You'll need to push manually."

# Push tag to enhanced
echo "Pushing tag to enhanced..."
git push enhanced "$VERSION" || echo "Warning: Could not push tag to enhanced. You'll need to push manually."

echo "Release $VERSION has been tagged and pushed to both repositories."
echo "The CD pipeline should now be triggered automatically."