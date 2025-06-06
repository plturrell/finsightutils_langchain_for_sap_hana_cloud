"""Version information module.

This module provides consistent access to version information
across the application. It reads the version from the VERSION
file and provides additional version metadata.
"""

import os
from typing import Dict

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Read version from VERSION file
VERSION_FILE = os.path.join(BASE_DIR, "VERSION")

try:
    if os.path.exists(VERSION_FILE):
        with open(VERSION_FILE, "r") as f:
            VERSION = f.read().strip()
    else:
        VERSION = "1.0.0"  # Default version if file not found
except Exception:
    VERSION = "1.0.0"  # Default version if file cannot be read

# Get version components
VERSION_PARTS = VERSION.split(".")
VERSION_MAJOR = VERSION_PARTS[0] if len(VERSION_PARTS) > 0 else "0"
VERSION_MINOR = VERSION_PARTS[1] if len(VERSION_PARTS) > 1 else "0"
VERSION_PATCH = VERSION_PARTS[2] if len(VERSION_PARTS) > 2 else "0"

# Check for environment overrides
ENV_VERSION = os.environ.get("VERSION")
if ENV_VERSION:
    VERSION = ENV_VERSION

# Additional version metadata
BUILD_ID = os.environ.get("BUILD_ID", "")
COMMIT_ID = os.environ.get("COMMIT_ID", "")
BUILD_DATE = os.environ.get("BUILD_DATE", "")


def get_version_info() -> Dict[str, str]:
    """
    Get complete version information.
    
    Returns:
        Dict[str, str]: Dictionary with version information.
    """
    return {
        "version": VERSION,
        "major": VERSION_MAJOR,
        "minor": VERSION_MINOR, 
        "patch": VERSION_PATCH,
        "build_id": BUILD_ID,
        "commit_id": COMMIT_ID,
        "build_date": BUILD_DATE,
    }


def get_version() -> str:
    """
    Get the current version string.
    
    Returns:
        str: Version string.
    """
    return VERSION