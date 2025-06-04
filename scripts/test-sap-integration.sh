#!/bin/bash

# This script tests that the SAP libraries integration works correctly

# Use environment variables with defaults for more flexibility
# These can be overridden by setting the variables before running the script
FINSIGHT_ROOT=${FINSIGHT_ROOT:-"/Users/apple/projects/finsight"}
FINSIGHTSAP_ROOT=${FINSIGHTSAP_ROOT:-"/Users/apple/projects/finsightsap"}

echo "Testing SAP libraries integration..."
echo "Using FINSIGHT_ROOT: $FINSIGHT_ROOT"
echo "Using FINSIGHTSAP_ROOT: $FINSIGHTSAP_ROOT"

# Check if the symlinks directories exist
if [ ! -d "$FINSIGHT_ROOT/finsightdata/external/sap-symlinks" ]; then
  echo "ERROR: Symlinks directory in finsightdata doesn't exist!"
  exit 1
fi

if [ ! -d "$FINSIGHT_ROOT/finsightdeep/external/sap-symlinks" ]; then
  echo "ERROR: Symlinks directory in finsightdeep doesn't exist!"
  exit 1
fi

if [ ! -d "$FINSIGHT_ROOT/finsightexperience/external/sap-symlinks" ]; then
  echo "ERROR: Symlinks directory in finsightexperience doesn't exist!"
  exit 1
fi

# Check finsightdata symlinks
echo "Checking finsightdata symlinks..."

libraries=(
  "cloud-sdk-js"
  "ucx"
  "python-pyodata"
  "odata-vocabularies"
  "langchain-integration"
  "graphql"
  "ord"
  "odata-library"
)

for lib in "${libraries[@]}"; do
  if [ ! -L "$FINSIGHT_ROOT/finsightdata/external/sap-symlinks/$lib" ]; then
    echo "ERROR: $lib symlink doesn't exist in finsightdata!"
    exit 1
  fi

  LINK_TARGET=$(readlink "$FINSIGHT_ROOT/finsightdata/external/sap-symlinks/$lib")
  if [ "$LINK_TARGET" != "$FINSIGHTSAP_ROOT/$lib" ]; then
    echo "ERROR: $lib symlink points to incorrect location: $LINK_TARGET"
    exit 1
  fi

  if [ ! -d "$FINSIGHTSAP_ROOT/$lib" ]; then
    echo "ERROR: $lib target directory doesn't exist in finsightsap!"
    exit 1
  fi

  echo "✅ $lib: OK"
done

# Check finsightdeep symlinks
echo "Checking finsightdeep symlinks..."

if [ ! -L "$FINSIGHT_ROOT/finsightdeep/external/sap-symlinks/generative-ai-toolkit-for-sap-hana-cloud" ]; then
  echo "ERROR: generative-ai-toolkit-for-sap-hana-cloud symlink doesn't exist in finsightdeep!"
  exit 1
fi

LINK_TARGET=$(readlink "$FINSIGHT_ROOT/finsightdeep/external/sap-symlinks/generative-ai-toolkit-for-sap-hana-cloud")
if [ "$LINK_TARGET" != "$FINSIGHTSAP_ROOT/generative-ai-toolkit-for-sap-hana-cloud" ]; then
  echo "ERROR: generative-ai-toolkit-for-sap-hana-cloud symlink points to incorrect location: $LINK_TARGET"
  exit 1
fi

if [ ! -d "$FINSIGHTSAP_ROOT/generative-ai-toolkit-for-sap-hana-cloud" ]; then
  echo "ERROR: generative-ai-toolkit-for-sap-hana-cloud target directory doesn't exist in finsightsap!"
  exit 1
fi

echo "✅ generative-ai-toolkit-for-sap-hana-cloud: OK"

# Check finsightexperience symlinks
echo "Checking finsightexperience symlinks..."

if [ ! -L "$FINSIGHT_ROOT/finsightexperience/external/sap-symlinks/fundamental-styles" ]; then
  echo "ERROR: fundamental-styles symlink doesn't exist in finsightexperience!"
  exit 1
fi

LINK_TARGET=$(readlink "$FINSIGHT_ROOT/finsightexperience/external/sap-symlinks/fundamental-styles")
if [ "$LINK_TARGET" != "$FINSIGHTSAP_ROOT/fundamental-styles" ]; then
  echo "ERROR: fundamental-styles symlink points to incorrect location: $LINK_TARGET"
  exit 1
fi

if [ ! -d "$FINSIGHTSAP_ROOT/fundamental-styles" ]; then
  echo "ERROR: fundamental-styles target directory doesn't exist in finsightsap!"
  exit 1
fi

echo "✅ fundamental-styles: OK"

# Check that README files exist for all libraries
echo "Checking README files..."

for lib in "${libraries[@]}" "fundamental-styles" "generative-ai-toolkit-for-sap-hana-cloud"; do
  if [ ! -f "$FINSIGHTSAP_ROOT/$lib/README.md" ]; then
    echo "WARNING: $lib README.md file doesn't exist!"
  else
    echo "✅ $lib README: OK"
  fi
done

echo "All tests passed! SAP libraries integration is working correctly."
exit 0