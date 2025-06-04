# FinSight SAP Integration Scripts

This directory contains scripts for managing SAP libraries integration in the FinSight platform.

## Available Scripts

### 1. setup-all-sap-symlinks.sh

Sets up symbolic links to all SAP libraries in the finsightsap repository.

Usage:
```bash
# Default usage
./setup-all-sap-symlinks.sh

# With custom paths
FINSIGHT_ROOT=/path/to/finsight FINSIGHTSAP_ROOT=/path/to/finsightsap ./setup-all-sap-symlinks.sh
```

This script:
- Creates symlinks directories in each FinSight component
- Sets up symbolic links for each SAP library in the appropriate component
- Cleans up any existing symlinks before creating new ones

### 2. test-sap-integration.sh

Tests that the SAP libraries integration works correctly.

Usage:
```bash
# Default usage
./test-sap-integration.sh

# With custom paths
FINSIGHT_ROOT=/path/to/finsight FINSIGHTSAP_ROOT=/path/to/finsightsap ./test-sap-integration.sh
```

This script:
- Checks if the symlinks directories exist
- Verifies that all library symlinks are set up correctly
- Confirms that the target directories exist in finsightsap
- Validates that README files exist for all libraries

## Environment Variables

Both scripts support the following environment variables:

- `FINSIGHT_ROOT`: Path to the finsight repository (default: `/Users/apple/projects/finsight`)
- `FINSIGHTSAP_ROOT`: Path to the finsightsap repository (default: `/Users/apple/projects/finsightsap`)