"""Database connection module."""

import logging
from typing import Optional

from hdbcli import dbapi

from config import config

logger = logging.getLogger(__name__)


class DatabaseConnectionError(Exception):
    """Exception raised for database connection errors."""
    pass


class Database:
    """Database connection handler."""
    
    _connection: Optional[dbapi.Connection] = None
    
    @classmethod
    def get_connection(cls) -> dbapi.Connection:
        """
        Get a connection to the database.
        
        Returns:
            dbapi.Connection: Connection to the database.
            
        Raises:
            DatabaseConnectionError: If the connection fails.
        """
        if cls._connection is None:
            try:
                logger.info("Establishing connection to SAP HANA Cloud...")
                cls._connection = dbapi.connect(
                    address=config.db.host,
                    port=config.db.port,
                    user=config.db.user,
                    password=config.db.password,
                )
                logger.info("Connection to SAP HANA Cloud established successfully")
            except dbapi.Error as e:
                logger.error(f"Failed to connect to SAP HANA Cloud: {str(e)}")
                raise DatabaseConnectionError(f"Failed to connect to database: {str(e)}")
        
        return cls._connection
    
    @classmethod
    def close_connection(cls) -> None:
        """Close the database connection."""
        if cls._connection is not None:
            logger.info("Closing connection to SAP HANA Cloud...")
            cls._connection.close()
            cls._connection = None
            logger.info("Connection to SAP HANA Cloud closed successfully")


def get_db_connection() -> dbapi.Connection:
    """
    Get a database connection.
    
    Returns:
        dbapi.Connection: Connection to the database.
    """
    return Database.get_connection()