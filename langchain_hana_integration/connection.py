"""
Connection management module for SAP HANA Cloud integration.

This module provides robust connection handling with connection pooling,
reconnection logic, and proper error management for production environments.
"""

import os
import json
import time
import logging
import threading
from typing import Dict, Any, Optional, List, Union, Callable
from contextlib import contextmanager

from hdbcli import dbapi
from hdbcli.dbapi import Connection, Cursor
from langchain_hana_integration.exceptions import ConnectionError, ConfigurationError

logger = logging.getLogger(__name__)

# Thread-local storage for connection pools
_connection_pools = threading.local()

class ConnectionPool:
    """
    Production-grade connection pool for SAP HANA Cloud.
    
    Features:
    - Connection pooling with min/max connections
    - Automatic connection health checks
    - Connection timeout handling
    - Reconnection with exponential backoff
    - Connection validation before returning to caller
    """
    
    def __init__(
        self,
        connection_params: Dict[str, Any],
        min_connections: int = 1,
        max_connections: int = 10,
        connection_timeout: int = 30,
        validation_interval: int = 300,
        reconnect_attempts: int = 3
    ):
        """
        Initialize a new connection pool.
        
        Args:
            connection_params: Dictionary with connection parameters
            min_connections: Minimum number of connections to maintain
            max_connections: Maximum number of connections allowed
            connection_timeout: Timeout in seconds for connection attempts
            validation_interval: Time in seconds between connection validations
            reconnect_attempts: Number of reconnection attempts before giving up
        """
        self.connection_params = connection_params
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.connection_timeout = connection_timeout
        self.validation_interval = validation_interval
        self.reconnect_attempts = reconnect_attempts
        
        # Internal state
        self._available_connections: List[Connection] = []
        self._in_use_connections: Dict[int, Connection] = {}
        self._lock = threading.RLock()
        self._last_validation_time = 0
        
        # Initialize the pool with minimum connections
        self._initialize_pool()
        
        # Start validation thread
        self._start_validation_thread()
    
    def _initialize_pool(self) -> None:
        """Initialize the connection pool with minimum connections."""
        try:
            with self._lock:
                for _ in range(self.min_connections):
                    connection = self._create_new_connection()
                    if connection:
                        self._available_connections.append(connection)
            
            logger.info(f"Initialized connection pool with {len(self._available_connections)} connections")
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            raise ConnectionError(f"Failed to initialize connection pool: {e}")
    
    def _create_new_connection(self) -> Optional[Connection]:
        """Create a new connection to SAP HANA Cloud with retry logic."""
        for attempt in range(self.reconnect_attempts):
            try:
                logger.debug(f"Attempting to create new connection (attempt {attempt+1}/{self.reconnect_attempts})")
                
                connection = dbapi.connect(
                    address=self.connection_params["address"],
                    port=self.connection_params["port"],
                    user=self.connection_params["user"],
                    password=self.connection_params["password"],
                    encrypt=self.connection_params.get("encrypt", True),
                    sslValidateCertificate=self.connection_params.get("sslValidateCertificate", False),
                    connectTimeout=self.connection_timeout
                )
                
                logger.debug("Successfully created new connection")
                return connection
            except dbapi.Error as e:
                backoff = min(2 ** attempt, 60)  # Exponential backoff with 60s max
                logger.warning(f"Connection attempt {attempt+1} failed: {e}. Retrying in {backoff}s...")
                time.sleep(backoff)
        
        logger.error(f"Failed to create connection after {self.reconnect_attempts} attempts")
        return None
    
    def _validate_connection(self, connection: Connection) -> bool:
        """Validate that a connection is still active and healthy."""
        try:
            cursor = connection.cursor()
            cursor.execute("SELECT 1 FROM DUMMY")
            cursor.close()
            return True
        except Exception as e:
            logger.warning(f"Connection validation failed: {e}")
            return False
    
    def _start_validation_thread(self) -> None:
        """Start a background thread for periodic connection validation."""
        def validation_worker():
            while True:
                try:
                    time.sleep(self.validation_interval)
                    self._validate_all_connections()
                except Exception as e:
                    logger.error(f"Error in validation thread: {e}")
        
        thread = threading.Thread(target=validation_worker, daemon=True)
        thread.start()
        logger.debug("Started connection validation thread")
    
    def _validate_all_connections(self) -> None:
        """Validate all connections in the pool and replace invalid ones."""
        with self._lock:
            # Track the current time
            self._last_validation_time = time.time()
            
            # Validate available connections
            valid_connections = []
            for conn in self._available_connections:
                if self._validate_connection(conn):
                    valid_connections.append(conn)
                else:
                    try:
                        conn.close()
                    except:
                        pass  # Ignore errors during close
            
            # Replace invalid connections
            invalid_count = len(self._available_connections) - len(valid_connections)
            self._available_connections = valid_connections
            
            # Create new connections to maintain minimum pool size
            for _ in range(invalid_count):
                if len(self._available_connections) < self.min_connections:
                    conn = self._create_new_connection()
                    if conn:
                        self._available_connections.append(conn)
            
            logger.debug(f"Validated connections. Replaced {invalid_count} invalid connections.")
    
    def get_connection(self) -> Connection:
        """
        Get a connection from the pool.
        
        Returns:
            A validated SAP HANA database connection
            
        Raises:
            ConnectionError: If no connection could be obtained
        """
        with self._lock:
            # If we have available connections, use one
            if self._available_connections:
                connection = self._available_connections.pop()
                
                # Validate the connection before returning it
                if self._validate_connection(connection):
                    self._in_use_connections[id(connection)] = connection
                    return connection
                else:
                    # If validation fails, try to create a new one
                    try:
                        connection.close()
                    except:
                        pass  # Ignore close errors
                    
                    connection = self._create_new_connection()
                    if connection:
                        self._in_use_connections[id(connection)] = connection
                        return connection
            
            # If we're below max connections, create a new one
            if len(self._in_use_connections) < self.max_connections:
                connection = self._create_new_connection()
                if connection:
                    self._in_use_connections[id(connection)] = connection
                    return connection
            
            # If we get here, we're out of connections and at max capacity
            raise ConnectionError(
                f"Could not obtain connection. Pool exhausted with {len(self._in_use_connections)} connections in use."
            )
    
    def return_connection(self, connection: Connection) -> None:
        """Return a connection to the pool."""
        with self._lock:
            # Remove from in-use tracking
            conn_id = id(connection)
            if conn_id in self._in_use_connections:
                del self._in_use_connections[conn_id]
            
            # Check if still valid before returning to pool
            if self._validate_connection(connection):
                # Don't exceed max connections
                if len(self._available_connections) < self.max_connections:
                    self._available_connections.append(connection)
                else:
                    try:
                        connection.close()
                    except:
                        pass  # Ignore close errors
            else:
                # Invalid connection, close and don't return to pool
                try:
                    connection.close()
                except:
                    pass  # Ignore close errors
    
    def close_all(self) -> None:
        """Close all connections in the pool."""
        with self._lock:
            # Close available connections
            for conn in self._available_connections:
                try:
                    conn.close()
                except:
                    pass  # Ignore close errors
            
            # Close in-use connections
            for conn in self._in_use_connections.values():
                try:
                    conn.close()
                except:
                    pass  # Ignore close errors
            
            # Reset internal state
            self._available_connections = []
            self._in_use_connections = {}
            
            logger.info("Closed all connections in the pool")

def create_connection_pool(
    connection_params: Dict[str, Any] = None,
    config_path: str = None,
    pool_name: str = "default",
    min_connections: int = 1,
    max_connections: int = 10
) -> ConnectionPool:
    """
    Create or retrieve a connection pool.
    
    Args:
        connection_params: Dictionary with connection parameters
        config_path: Path to configuration file
        pool_name: Name for this connection pool
        min_connections: Minimum number of connections
        max_connections: Maximum number of connections
        
    Returns:
        ConnectionPool instance
        
    Raises:
        ConfigurationError: If connection parameters are invalid
    """
    # Get connection parameters
    params = _get_connection_params(connection_params, config_path)
    
    # Check if we already have a pool with this name
    if not hasattr(_connection_pools, "pools"):
        _connection_pools.pools = {}
    
    # Create new pool if needed
    if pool_name not in _connection_pools.pools:
        logger.info(f"Creating new connection pool: {pool_name}")
        _connection_pools.pools[pool_name] = ConnectionPool(
            connection_params=params,
            min_connections=min_connections,
            max_connections=max_connections
        )
    
    return _connection_pools.pools[pool_name]

@contextmanager
def get_connection(pool_name: str = "default") -> Connection:
    """
    Context manager for getting a connection from a pool.
    
    Args:
        pool_name: Name of the connection pool
        
    Yields:
        SAP HANA database connection
        
    Raises:
        ConnectionError: If the specified pool doesn't exist
    """
    if not hasattr(_connection_pools, "pools") or pool_name not in _connection_pools.pools:
        raise ConnectionError(f"Connection pool '{pool_name}' does not exist")
    
    pool = _connection_pools.pools[pool_name]
    connection = pool.get_connection()
    
    try:
        yield connection
    finally:
        pool.return_connection(connection)

def _get_connection_params(
    connection_params: Dict[str, Any] = None,
    config_path: str = None
) -> Dict[str, Any]:
    """
    Get connection parameters from params, config file, or environment.
    
    Args:
        connection_params: Dictionary with connection parameters
        config_path: Path to configuration file
        
    Returns:
        Dictionary with connection parameters
        
    Raises:
        ConfigurationError: If connection parameters are invalid
    """
    # Priority 1: Use provided parameters
    if connection_params:
        params = connection_params.copy()
    else:
        # Priority 2: Load from config file
        params = _load_config_file(config_path)
        
        # Priority 3: Use environment variables
        if not params:
            params = {
                "address": os.environ.get("HANA_HOST"),
                "port": int(os.environ.get("HANA_PORT", "443")),
                "user": os.environ.get("HANA_USER"),
                "password": os.environ.get("HANA_PASSWORD"),
                "encrypt": os.environ.get("HANA_ENCRYPT", "True").lower() == "true",
                "sslValidateCertificate": os.environ.get("HANA_VALIDATE_CERT", "False").lower() == "true",
            }
    
    # Validate required parameters
    required_params = ["address", "port", "user", "password"]
    missing_params = [param for param in required_params if not params.get(param)]
    
    if missing_params:
        raise ConfigurationError(f"Missing required connection parameters: {', '.join(missing_params)}")
    
    # Ensure port is an integer
    if "port" in params and not isinstance(params["port"], int):
        try:
            params["port"] = int(params["port"])
        except (ValueError, TypeError):
            raise ConfigurationError(f"Invalid port value: {params['port']}. Must be an integer.")
    
    return params

def _load_config_file(config_path: str = None) -> Dict[str, Any]:
    """Load connection configuration from a JSON file."""
    if config_path is None:
        # Check common locations for the configuration file
        possible_paths = [
            "connection.json",
            "config/connection.json",
            "../config/connection.json",
            os.path.join(os.path.dirname(__file__), "../config/connection.json")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break
    
    if config_path and os.path.exists(config_path):
        logger.info(f"Loading connection configuration from {config_path}")
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in configuration file {config_path}: {e}")
            raise ConfigurationError(f"Invalid JSON in configuration file: {e}")
        except Exception as e:
            logger.error(f"Error reading configuration file {config_path}: {e}")
            raise ConfigurationError(f"Error reading configuration file: {e}")
    
    return {}

def close_all_pools() -> None:
    """Close all connection pools."""
    if hasattr(_connection_pools, "pools"):
        for name, pool in _connection_pools.pools.items():
            logger.info(f"Closing connection pool: {name}")
            pool.close_all()
        
        _connection_pools.pools = {}