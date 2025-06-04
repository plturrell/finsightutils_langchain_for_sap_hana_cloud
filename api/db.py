"""Database connection utility."""

import os
from typing import Optional
from hdbcli import dbapi

_connection = None

def get_db_connection() -> dbapi.Connection:
    """Get a HANA database connection.
    
    Returns:
        A HANA database connection.
    """
    global _connection
    
    if _connection is not None and _connection.isconnected():
        return _connection
    
    # Connection parameters
    host = os.environ.get("HANA_HOST")
    port = os.environ.get("HANA_PORT")
    user = os.environ.get("HANA_USER")
    password = os.environ.get("HANA_PASSWORD")
    
    # Connect to HANA
    _connection = dbapi.connect(
        address=host,
        port=port,
        user=user,
        password=password,
        encrypt=True,
        sslValidateCertificate=False,
    )
    
    return _connection