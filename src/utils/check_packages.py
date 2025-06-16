#!/usr/bin/env python
"""
Check if required packages are installed.
"""

import sys
import importlib

required_packages = [
    "hdbcli",
    "langchain",
    "numpy",
    "torch"
]

for package in required_packages:
    try:
        importlib.import_module(package)
        print(f"✓ {package} is installed")
    except ImportError:
        print(f"✗ {package} is NOT installed")

try:
    from hdbcli import dbapi
    print(f"✓ hdbcli.dbapi module is available")
    print(f"  hdbcli version: {dbapi.__version__ if hasattr(dbapi, '__version__') else 'unknown'}")
except ImportError:
    print(f"✗ hdbcli.dbapi module is NOT available")