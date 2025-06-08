"""Database connection module with connection pooling and advanced error handling."""

import logging
import os
import time
from contextlib import contextmanager
from typing import Dict, Optional, Generator
import threading

from hdbcli import dbapi

from config import config

logger = logging.getLogger(__name__)


class DatabaseConnectionError(Exception):
    """Exception raised for database connection errors."""
    pass


class ConnectionPool:
    """
    Database connection pool with automatic cleanup and health check.
    
    This class maintains a pool of database connections and ensures they remain
    healthy by performing periodic checks and reconnecting if necessary.
    """
    
    def __init__(self, max_connections: int = 5, connection_timeout: int = 600):
        """
        Initialize the connection pool.
        
        Args:
            max_connections: Maximum number of connections in the pool
            connection_timeout: Connection timeout in seconds
        """
        self.max_connections = max_connections
        self.connection_timeout = connection_timeout
        self.pool: Dict[threading.Thread, Dict] = {}
        self.lock = threading.RLock()
    
    def _create_connection(self) -> dbapi.Connection:
        """Create a new database connection."""
        try:
            logger.info("Creating new connection to SAP HANA Cloud...")
            connection = dbapi.connect(
                address=config.db.host,
                port=config.db.port,
                user=config.db.user,
                password=config.db.password,
                encrypt=config.db.encrypt,
                sslValidateCertificate=config.db.ssl_validate_cert,
            )
            logger.debug("Connection to SAP HANA Cloud established successfully")
            return connection
        except dbapi.Error as e:
            logger.error(f"Failed to connect to SAP HANA Cloud: {str(e)}")
            raise DatabaseConnectionError(f"Failed to connect to database: {str(e)}")
    
    def get_connection(self) -> dbapi.Connection:
        """
        Get a connection from the pool or create a new one.
        
        Returns:
            dbapi.Connection: Database connection.
            
        Raises:
            DatabaseConnectionError: If the connection fails.
        """
        current_thread = threading.current_thread()
        
        with self.lock:
            # Check if the current thread already has a connection
            if current_thread in self.pool:
                conn_data = self.pool[current_thread]
                
                # Check if the connection is still valid
                if time.time() - conn_data["last_used"] > self.connection_timeout:
                    logger.info("Connection timed out, creating new connection")
                    try:
                        conn_data["connection"].close()
                    except Exception:
                        pass  # Ignore errors when closing timed-out connections
                    
                    # Create new connection
                    conn_data["connection"] = self._create_connection()
                
                # Update last used time
                conn_data["last_used"] = time.time()
                return conn_data["connection"]
            
            # Clean up any connections from threads that no longer exist
            self._cleanup()
            
            # Create a new connection if the pool isn't full
            if len(self.pool) < self.max_connections:
                connection = self._create_connection()
                self.pool[current_thread] = {
                    "connection": connection,
                    "last_used": time.time()
                }
                return connection
            
            # If we reach here, the pool is full
            logger.warning(f"Connection pool is full ({self.max_connections}), reusing oldest connection")
            oldest_thread = min(self.pool, key=lambda t: self.pool[t]["last_used"])
            conn_data = self.pool[oldest_thread]
            
            # Close the old connection
            try:
                conn_data["connection"].close()
            except Exception:
                pass  # Ignore errors when closing old connections
            
            # Create a new connection and reassign to current thread
            del self.pool[oldest_thread]
            connection = self._create_connection()
            self.pool[current_thread] = {
                "connection": connection,
                "last_used": time.time()
            }
            return connection
    
    def _cleanup(self) -> None:
        """Clean up connections from threads that no longer exist."""
        active_threads = {t.ident for t in threading.enumerate()}
        
        to_remove = []
        for thread in self.pool:
            if thread.ident not in active_threads:
                to_remove.append(thread)
        
        for thread in to_remove:
            try:
                self.pool[thread]["connection"].close()
            except Exception:
                pass  # Ignore errors when closing abandoned connections
            del self.pool[thread]
    
    def close_all(self) -> None:
        """Close all connections in the pool."""
        with self.lock:
            for thread, conn_data in list(self.pool.items()):
                try:
                    conn_data["connection"].close()
                except Exception as e:
                    logger.warning(f"Error closing connection: {str(e)}")
                del self.pool[thread]
            logger.info("All database connections closed")


# Create a global connection pool
connection_pool = ConnectionPool(
    max_connections=int(os.getenv("DB_MAX_CONNECTIONS", "5")),
    connection_timeout=int(os.getenv("DB_CONNECTION_TIMEOUT", "600"))
)


def get_db_connection() -> dbapi.Connection:
    """
    Get a database connection from the connection pool.
    
    Returns:
        dbapi.Connection: Database connection.
        
    Raises:
        DatabaseConnectionError: If the connection fails.
    """
    return connection_pool.get_connection()


@contextmanager
def get_db_connection_context() -> Generator[dbapi.Connection, None, None]:
    """
    Context manager for database connections.
    
    Yields:
        Generator[dbapi.Connection, None, None]: Database connection.
        
    Raises:
        DatabaseConnectionError: If the connection fails.
    """
    connection = get_db_connection()
    try:
        yield connection
    finally:
        # We don't close the connection here as it's managed by the pool
        pass


def get_vectorstore():
    """
    Get a vector store instance with the default configuration.
    
    Returns:
        HanaDB: A configured vector store instance.
        
    Raises:
        HTTPException: If the vector store initialization fails.
    """
    from fastapi import HTTPException
    from langchain_hana import HanaDB
    from langchain_hana.utils import DistanceStrategy
    
    try:
        # Get database connection
        connection = get_db_connection()
        
        # Get embedding model from configuration
        from api.services import get_embedding_model
        embedding_model = get_embedding_model()
        
        # Create vector store instance
        vectorstore = HanaDB(
            connection=connection,
            embedding=embedding_model,
            distance_strategy=DistanceStrategy.COSINE,
            table_name=config.vector.table_name,
            enable_lineage=config.vector.enable_lineage,
            enable_audit_logging=config.vector.enable_audit_logging,
            audit_log_to_console=True,
        )
        
        return vectorstore
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "vectorstore_initialization_error",
                "message": f"Failed to initialize vector store: {str(e)}",
                "context": {
                    "table_name": config.vector.table_name,
                    "suggestion": "Check database connection and table configuration"
                }
            }
        )