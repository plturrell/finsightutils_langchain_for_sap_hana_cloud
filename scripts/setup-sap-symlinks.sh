#!/bin/bash

# This script sets up symbolic links to all SAP libraries in the finsightsap repository

# Use environment variables with defaults for more flexibility
# These can be overridden by setting the variables before running the script
FINSIGHT_ROOT=${FINSIGHT_ROOT:-"/Users/apple/projects/finsight"}
FINSIGHTSAP_ROOT=${FINSIGHTSAP_ROOT:-"/Users/apple/projects/finsightsap"}

echo "Setting up SAP library symlinks..."
echo "Using FINSIGHT_ROOT: $FINSIGHT_ROOT"
echo "Using FINSIGHTSAP_ROOT: $FINSIGHTSAP_ROOT"

# Create symlinks directories if they don't exist
mkdir -p "$FINSIGHT_ROOT/finsightdata/external/sap-symlinks"
mkdir -p "$FINSIGHT_ROOT/finsightdeep/external/sap-symlinks"
mkdir -p "$FINSIGHT_ROOT/finsightexperience/external/sap-symlinks"

# Remove any existing symlinks
rm -rf "$FINSIGHT_ROOT/finsightdata/external/sap-symlinks/"*
rm -rf "$FINSIGHT_ROOT/finsightdeep/external/sap-symlinks/"*
rm -rf "$FINSIGHT_ROOT/finsightexperience/external/sap-symlinks/"*

# Create symlinks for finsightdata
ln -s "$FINSIGHTSAP_ROOT/cloud-sdk-js" "$FINSIGHT_ROOT/finsightdata/external/sap-symlinks/cloud-sdk-js"
ln -s "$FINSIGHTSAP_ROOT/ucx" "$FINSIGHT_ROOT/finsightdata/external/sap-symlinks/ucx"
ln -s "$FINSIGHTSAP_ROOT/python-pyodata" "$FINSIGHT_ROOT/finsightdata/external/sap-symlinks/python-pyodata"
ln -s "$FINSIGHTSAP_ROOT/odata-vocabularies" "$FINSIGHT_ROOT/finsightdata/external/sap-symlinks/odata-vocabularies"
ln -s "$FINSIGHTSAP_ROOT/langchain-integration-for-sap-hana-cloud" "$FINSIGHT_ROOT/finsightdata/external/sap-symlinks/langchain-integration-for-sap-hana-cloud"
ln -s "$FINSIGHTSAP_ROOT/graphql" "$FINSIGHT_ROOT/finsightdata/external/sap-symlinks/graphql"
ln -s "$FINSIGHTSAP_ROOT/ord" "$FINSIGHT_ROOT/finsightdata/external/sap-symlinks/ord"
ln -s "$FINSIGHTSAP_ROOT/odata-library" "$FINSIGHT_ROOT/finsightdata/external/sap-symlinks/odata-library"
echo "✅ Symlinks created in finsightdata"

# Create symlinks for finsightdeep
ln -s "$FINSIGHTSAP_ROOT/generative-ai-toolkit-for-sap-hana-cloud" "$FINSIGHT_ROOT/finsightdeep/external/sap-symlinks/generative-ai-toolkit-for-sap-hana-cloud"
echo "✅ Symlinks created in finsightdeep"

# Create symlinks for finsightexperience
ln -s "$FINSIGHTSAP_ROOT/fundamental-styles" "$FINSIGHT_ROOT/finsightexperience/external/sap-symlinks/fundamental-styles"
echo "✅ Symlinks created in finsightexperience"

echo "All SAP library symlinks have been set up successfully."
echo "Use the libraries through the symlinks in the external/sap-symlinks directories."
echo ""
echo "To test the integration, run: $FINSIGHTSAP_ROOT/scripts/test-sap-integration.sh"