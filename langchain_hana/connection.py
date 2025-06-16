"""
Connection Utilities for SAP HANA Cloud

This module provides utility functions for creating and managing connections
to SAP HANA Cloud databases, including production-ready features like:
- Connection pooling
- Automatic reconnection
- Robust error handling
- Connection verification
"""

import os
import logging
import time
import threading
import queue
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, Union, List, Callable

from hdbcli import dbapi

# Configure logging
logger = logging.getLogger(__name__)

# Global connection pool
_connection_pools = {}

def create_connection(
    host: Optional[str] = None,
    port: Optional[int] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    encrypt: bool = True,
    validate_cert: bool = False,
    connection_timeout: int = 30,
    reconnect_attempts: int = 3,
    reconnect_delay: int = 5,
    **additional_params
) -> dbapi.Connection:
    """
    Create a connection to SAP HANA Cloud with production-grade reliability features.
    
    This function supports multiple ways to provide connection parameters:
    1. Explicit parameters passed to the function
    2. Environment variables (HANA_HOST, HANA_PORT, HANA_USER, HANA_PASSWORD)
    3. Additional parameters passed as keyword arguments
    
    Args:
        host: Hostname or IP address of the SAP HANA Cloud instance
        port: Port number (default is 443 for HANA Cloud)
        user: Username for authentication
        password: Password for authentication
        encrypt: Whether to use encrypted connection (default True)
        validate_cert: Whether to validate SSL certificates (default False)
        connection_timeout: Timeout in seconds for connection attempts (default 30)
        reconnect_attempts: Number of reconnection attempts (default 3)
        reconnect_delay: Delay in seconds between reconnection attempts (default 5)
        **additional_params: Additional parameters to pass to the connection
            
    Returns:
        A HANA database connection object
        
    Raises:
        ValueError: If required connection parameters are missing
        ConnectionError: If connection fails after all retry attempts
        dbapi.Error: If a database-specific error occurs
    """
    # Get parameters from environment variables if not provided
    connection_params = {
        "address": host or os.environ.get("HANA_HOST"),
        "port": port or int(os.environ.get("HANA_PORT", "443")),
        "user": user or os.environ.get("HANA_USER"),
        "password": password or os.environ.get("HANA_PASSWORD"),
        "encrypt": encrypt,
        "sslValidateCertificate": validate_cert,
        "timeout": connection_timeout,
    }
    
    # Add additional parameters
    connection_params.update(additional_params)
    
    # Check for required parameters
    required_params = ["address", "port", "user", "password"]
    missing_params = [param for param in required_params if not connection_params.get(param)]
    
    if missing_params:
        raise ValueError(
            f"Missing required connection parameters: {', '.join(missing_params)}. "
            f"Please provide them as function arguments or set the corresponding "
            f"environment variables (HANA_HOST, HANA_PORT, HANA_USER, HANA_PASSWORD)."
        )
    
    # Log connection attempt
    logger.info(f"Connecting to SAP HANA at {connection_params['address']}:{connection_params['port']}")
    
    # Implement connection retry logic
    attempt = 0
    last_exception = None
    
    while attempt < reconnect_attempts:
        try:
            # Connect to SAP HANA
            connection = dbapi.connect(**connection_params)
            
            # Test the connection with a simple query
            cursor = connection.cursor()
            cursor.execute("SELECT 1 FROM SYS.DUMMY")
            cursor.close()
            
            logger.info("Connected successfully to SAP HANA Cloud")
            return connection
            
        except dbapi.Error as e:
            last_exception = e
            attempt += 1
            
            error_msg = f"Connection attempt {attempt}/{reconnect_attempts} failed: {str(e)}"
            if attempt < reconnect_attempts:
                logger.warning(f"{error_msg}, retrying in {reconnect_delay} seconds...")
                import time
                time.sleep(reconnect_delay)
            else:
                logger.error(f"{error_msg}, no more retry attempts.")
        
        except Exception as e:
            logger.error(f"Unexpected error connecting to SAP HANA: {str(e)}")
            raise ConnectionError(f"Failed to connect to SAP HANA: {str(e)}") from e
    
    # If we get here, all connection attempts failed
    raise ConnectionError(
        f"Failed to connect to SAP HANA after {reconnect_attempts} attempts: {str(last_exception)}"
    ) from last_exception


class ConnectionPool:
    """
    A thread-safe connection pool for SAP HANA Cloud.
    
    This class provides a pool of database connections that can be reused,
    reducing the overhead of creating new connections for each operation.
    It includes features like:
    - Connection validation before returning to clients
    - Automatic reconnection for stale connections
    - Connection health check
    - Connection timeout and max age enforcement
    """
    
    def __init__(
        self,
        min_connections: int = 1,
        max_connections: int = 10,
        connection_timeout: int = 30,
        connection_max_age: int = 3600,  # 1 hour
        health_check_interval: int = 300,  # 5 minutes
        **connection_params
    ):
        """
        Initialize a new connection pool.
        
        Args:
            min_connections: Minimum number of connections to keep in the pool (default 1)
            max_connections: Maximum number of connections allowed in the pool (default 10)
            connection_timeout: Timeout in seconds for acquiring a connection (default 30)
            connection_max_age: Maximum age of a connection in seconds before recycling (default 3600)
            health_check_interval: Interval in seconds for health checks (default 300)
            **connection_params: Connection parameters to pass to create_connection
        """
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.connection_timeout = connection_timeout
        self.connection_max_age = connection_max_age
        self.health_check_interval = health_check_interval
        self.connection_params = connection_params
        
        # Connection pool
        self._pool = queue.Queue(maxsize=max_connections)
        self._active_connections = 0
        self._lock = threading.RLock()
        
        # Connection metadata (for tracking age and health)
        self._connection_metadata = {}
        
        # Initialize the pool with min_connections
        self._initialize_pool()
        
        # Start health check thread
        self._stop_health_check = threading.Event()
        self._health_check_thread = threading.Thread(
            target=self._health_check_loop, 
            daemon=True
        )
        self._health_check_thread.start()
    
    def _initialize_pool(self):
        """Initialize the pool with the minimum number of connections."""
        for _ in range(self.min_connections):
            try:
                with self._lock:
                    if self._active_connections < self.max_connections:
                        conn = self._create_new_connection()
                        if conn:
                            self._pool.put(conn)
            except Exception as e:
                logger.error(f"Error initializing connection pool: {str(e)}")
    
    def _create_new_connection(self):
        """Create a new connection and track its metadata."""
        try:
            conn = create_connection(**self.connection_params)
            self._active_connections += 1
            
            # Track metadata
            self._connection_metadata[id(conn)] = {
                'created_at': datetime.now(),
                'last_used': datetime.now(),
                'healthy': True
            }
            
            return conn
        except Exception as e:
            logger.error(f"Error creating new connection: {str(e)}")
            return None
    
    def _check_connection_health(self, conn):
        """Check if a connection is healthy by executing a simple query."""
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM SYS.DUMMY")
            cursor.close()
            return True
        except Exception:
            return False
    
    def _health_check_loop(self):
        """Periodically check the health of connections in the pool."""
        while not self._stop_health_check.is_set():
            try:
                time.sleep(self.health_check_interval)
                self._check_all_connections()
            except Exception as e:
                logger.error(f"Error in health check loop: {str(e)}")
    
    def _check_all_connections(self):
        """Check the health of all connections in the pool."""
        try:
            # Get all connections from the pool
            connections = []
            pool_size = self._pool.qsize()
            
            for _ in range(pool_size):
                try:
                    conn = self._pool.get(block=False)
                    connections.append(conn)
                except queue.Empty:
                    break
            
            # Check each connection and return healthy ones to the pool
            for conn in connections:
                conn_id = id(conn)
                metadata = self._connection_metadata.get(conn_id, {})
                
                # Check if connection is too old
                if metadata.get('created_at'):
                    age = (datetime.now() - metadata['created_at']).total_seconds()
                    if age > self.connection_max_age:
                        self._close_connection(conn)
                        with self._lock:
                            if self._active_connections < self.max_connections:
                                new_conn = self._create_new_connection()
                                if new_conn:
                                    self._pool.put(new_conn)
                        continue
                
                # Check connection health
                is_healthy = self._check_connection_health(conn)
                
                if is_healthy:
                    # Return healthy connection to the pool
                    self._pool.put(conn)
                    metadata['healthy'] = True
                    metadata['last_used'] = datetime.now()
                else:
                    # Close unhealthy connection and create a new one
                    self._close_connection(conn)
                    with self._lock:
                        if self._active_connections < self.max_connections:
                            new_conn = self._create_new_connection()
                            if new_conn:
                                self._pool.put(new_conn)
        
        except Exception as e:
            logger.error(f"Error checking connections: {str(e)}")
    
    def _close_connection(self, conn):
        """Close a connection and clean up metadata."""
        try:
            conn.close()
        except Exception as e:
            logger.error(f"Error closing connection: {str(e)}")
        
        with self._lock:
            self._active_connections -= 1
            conn_id = id(conn)
            if conn_id in self._connection_metadata:
                del self._connection_metadata[conn_id]
    
    def get_connection(self):
        """
        Get a connection from the pool.
        
        Returns:
            A database connection
            
        Raises:
            ConnectionError: If unable to get a connection within the timeout
        """
        start_time = time.time()
        
        while True:
            # Check if we've exceeded the timeout
            if time.time() - start_time > self.connection_timeout:
                raise ConnectionError(f"Timeout getting connection from pool after {self.connection_timeout} seconds")
            
            try:
                # Try to get a connection from the pool
                conn = self._pool.get(block=True, timeout=1.0)
                conn_id = id(conn)
                
                # Check if the connection is still valid
                metadata = self._connection_metadata.get(conn_id, {})
                
                if metadata.get('healthy', False) and self._check_connection_health(conn):
                    # Update last used time
                    metadata['last_used'] = datetime.now()
                    return conn
                else:
                    # Close the unhealthy connection
                    self._close_connection(conn)
                    
                    # Create a new connection if needed
                    with self._lock:
                        if self._active_connections < self.max_connections:
                            new_conn = self._create_new_connection()
                            if new_conn:
                                return new_conn
            
            except queue.Empty:
                # If the pool is empty, create a new connection if possible
                with self._lock:
                    if self._active_connections < self.max_connections:
                        new_conn = self._create_new_connection()
                        if new_conn:
                            return new_conn
    
    def release_connection(self, conn):
        """
        Release a connection back to the pool.
        
        Args:
            conn: The connection to release
        """
        conn_id = id(conn)
        
        # Check if the connection is still valid
        if conn_id in self._connection_metadata and self._check_connection_health(conn):
            # Update metadata
            self._connection_metadata[conn_id]['last_used'] = datetime.now()
            
            try:
                self._pool.put(conn, block=False)
            except queue.Full:
                # If the pool is full, close the connection
                self._close_connection(conn)
        else:
            # Close invalid connection
            self._close_connection(conn)
    
    def close(self):
        """Close all connections and shut down the pool."""
        # Stop health check thread
        self._stop_health_check.set()
        if self._health_check_thread.is_alive():
            self._health_check_thread.join(timeout=5.0)
        
        # Close all connections in the pool
        while not self._pool.empty():
            try:
                conn = self._pool.get(block=False)
                self._close_connection(conn)
            except queue.Empty:
                break

def test_connection(connection: dbapi.Connection) -> Tuple[bool, Dict[str, Any]]:
    """
    Test a SAP HANA Cloud connection and retrieve database information.
    
    This function tests the connection by running a simple query and
    collects information about the database.
    
    Args:
        connection: SAP HANA database connection
        
    Returns:
        Tuple containing:
            - Boolean indicating if the connection is valid
            - Dictionary with database information (version, current schema, etc.)
    """
    info = {}
    
    try:
        cursor = connection.cursor()
        
        # Get database version
        cursor.execute("SELECT VERSION FROM SYS.M_DATABASE")
        version = cursor.fetchone()[0]
        info["version"] = version
        
        # Get current schema
        cursor.execute("SELECT CURRENT_SCHEMA FROM SYS.DUMMY")
        current_schema = cursor.fetchone()[0]
        info["current_schema"] = current_schema
        
        # Get database name
        cursor.execute("SELECT DATABASE_NAME FROM SYS.M_DATABASE")
        database_name = cursor.fetchone()[0]
        info["database_name"] = database_name
        
        # Check if vector capabilities are available
        try:
            cursor.execute("SELECT COUNT(*) FROM SYS.DATA_TYPES WHERE TYPE_NAME = 'REAL_VECTOR'")
            vector_support = cursor.fetchone()[0] > 0
            info["vector_support"] = vector_support
        except:
            info["vector_support"] = False
        
        cursor.close()
        return True, info
    except dbapi.Error as e:
        logger.error(f"Connection test failed: {str(e)}")
        return False, {"error": str(e)}

def create_connection_from_json(config_path: str) -> dbapi.Connection:
    """
    Create a connection from a JSON configuration file.
    
    Args:
        config_path: Path to the JSON configuration file
        
    Returns:
        A HANA database connection object
        
    Raises:
        FileNotFoundError: If the configuration file does not exist
        ValueError: If required parameters are missing
        dbapi.Error: If connection fails
    """
    import json
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return create_connection(
        host=config.get("host"),
        port=config.get("port"),
        user=config.get("user"),
        password=config.get("password"),
        encrypt=config.get("encrypt", True),
        validate_cert=config.get("validate_cert", False),
        **{k: v for k, v in config.items() if k not in 
           ["host", "port", "user", "password", "encrypt", "validate_cert"]}
    )

def get_connection(
    connection_or_config: Union[dbapi.Connection, Dict[str, Any], str, None] = None
) -> dbapi.Connection:
    """
    Get a database connection from various input types.
    
    This utility function handles different ways of providing connection information:
    - Existing connection object
    - Dictionary of connection parameters
    - Path to a JSON configuration file
    - None (use environment variables)
    
    Args:
        connection_or_config: Connection object, parameters dict, config file path, or None
        
    Returns:
        A HANA database connection object
        
    Raises:
        ValueError: If the input type is not supported or required parameters are missing
        dbapi.Error: If connection fails
    """
    if connection_or_config is None:
        # Use environment variables
        return create_connection()
    
    if isinstance(connection_or_config, dbapi.Connection):
        # Already a connection object
        return connection_or_config
    
    if isinstance(connection_or_config, dict):
        # Dictionary of connection parameters
        return create_connection(**connection_or_config)
    
    if isinstance(connection_or_config, str):
        # Path to JSON configuration file
        return create_connection_from_json(connection_or_config)
    
    raise ValueError(
        f"Unsupported connection type: {type(connection_or_config)}. "
        f"Expected: dbapi.Connection, dict, str, or None"
    )


def create_connection_pool(
    pool_name: str = "default",
    min_connections: int = 1,
    max_connections: int = 10,
    **connection_params
) -> ConnectionPool:
    """
    Create a new connection pool or return an existing one with the given name.
    
    Args:
        pool_name: Name of the pool (default: "default")
        min_connections: Minimum number of connections in the pool
        max_connections: Maximum number of connections in the pool
        **connection_params: Connection parameters to pass to create_connection
        
    Returns:
        ConnectionPool: A connection pool instance
    """
    global _connection_pools
    
    # If the pool already exists, return it
    if pool_name in _connection_pools:
        return _connection_pools[pool_name]
    
    # Create a new pool
    pool = ConnectionPool(
        min_connections=min_connections,
        max_connections=max_connections,
        **connection_params
    )
    
    # Store the pool in the global registry
    _connection_pools[pool_name] = pool
    
    logger.info(f"Created new connection pool '{pool_name}' with {min_connections}-{max_connections} connections")
    
    return pool


def get_connection_pool(pool_name: str = "default") -> Optional[ConnectionPool]:
    """
    Get an existing connection pool by name.
    
    Args:
        pool_name: Name of the pool (default: "default")
        
    Returns:
        ConnectionPool: The connection pool instance, or None if not found
    """
    return _connection_pools.get(pool_name)


def close_all_connection_pools():
    """Close all connection pools."""
    global _connection_pools
    
    for pool_name, pool in list(_connection_pools.items()):
        try:
            logger.info(f"Closing connection pool '{pool_name}'")
            pool.close()
        except Exception as e:
            logger.error(f"Error closing connection pool '{pool_name}': {str(e)}")
    
    _connection_pools.clear()