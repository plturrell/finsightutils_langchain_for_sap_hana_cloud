"""
Apache Arrow Flight server implementation for SAP HANA Cloud.

This module provides a high-performance data transfer layer between
SAP HANA Cloud and GPU-accelerated vector operations using Apache Arrow Flight.
"""

import concurrent.futures
import json
import logging
import os
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

try:
    import pyarrow as pa
    import pyarrow.flight as flight
    HAS_ARROW_FLIGHT = True
except ImportError:
    HAS_ARROW_FLIGHT = False

try:
    from hdbcli import dbapi
    HAS_HDBCLI = True
except ImportError:
    HAS_HDBCLI = False

from ..error_utils import wrap_hana_errors

logger = logging.getLogger(__name__)


class HanaArrowFlightServer(flight.FlightServerBase):
    """
    Arrow Flight server implementation for SAP HANA Cloud.
    
    This server acts as a bridge between SAP HANA Cloud and Arrow Flight clients,
    providing optimized data transfer for vector embeddings and search operations.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8815,
        location: Optional[str] = None,
        hana_host: Optional[str] = None,
        hana_port: Optional[int] = None,
        hana_user: Optional[str] = None,
        hana_password: Optional[str] = None,
        hana_connection_args: Optional[Dict[str, Any]] = None,
        auth_handler: Optional[flight.ServerAuthHandler] = None,
        tls_certificates: Optional[List[Tuple[bytes, bytes]]] = None,
        verify_client: Optional[bool] = False,
        root_certificates: Optional[bytes] = None,
        middleware: Optional[List[flight.ServerMiddleware]] = None,
        vector_column_type: str = "REAL_VECTOR",
    ):
        """
        Initialize the Arrow Flight server for SAP HANA.
        
        Args:
            host: Server host address (default: localhost)
            port: Server port (default: 8815)
            location: Optional location URI override
            hana_host: SAP HANA host address
            hana_port: SAP HANA port
            hana_user: SAP HANA username
            hana_password: SAP HANA password
            hana_connection_args: Additional SAP HANA connection arguments
            auth_handler: Optional authentication handler
            tls_certificates: Optional TLS certificates for secure connections
            verify_client: Whether to verify client certificates
            root_certificates: Optional root certificates for client verification
            middleware: Optional server middleware components
            vector_column_type: Type of vector column in SAP HANA ("REAL_VECTOR" or "HALF_VECTOR")
        
        Raises:
            ImportError: If required dependencies are not installed
        """
        if not HAS_ARROW_FLIGHT:
            raise ImportError(
                "The pyarrow and pyarrow.flight packages are required for Arrow Flight integration. "
                "Install them with 'pip install pyarrow pyarrow.flight'."
            )
        
        if not HAS_HDBCLI:
            raise ImportError(
                "The hdbcli package is required for SAP HANA connectivity. "
                "Install it with 'pip install hdbcli'."
            )
        
        super().__init__(
            location or f"grpc://{host}:{port}",
            auth_handler=auth_handler,
            tls_certificates=tls_certificates,
            verify_client=verify_client,
            root_certificates=root_certificates,
            middleware=middleware
        )
        
        # Store configuration
        self.host = host
        self.port = port
        self.hana_host = hana_host
        self.hana_port = hana_port
        self.hana_user = hana_user
        self.hana_password = hana_password
        self.hana_connection_args = hana_connection_args or {}
        self.vector_column_type = vector_column_type
        
        # Initialize connection pool
        self._connection_pool = concurrent.futures.ThreadPoolExecutor(max_workers=10)
        self._db_connections = {}
        
        # Register known endpoints
        self._registered_tables = {}
        
        # Initialize basic authentication handler if not provided
        if auth_handler is None and hana_user and hana_password:
            self._auth_handler = self._create_basic_auth_handler()
    
    def _create_basic_auth_handler(self) -> flight.ServerAuthHandler:
        """Create a basic authentication handler that delegates to SAP HANA."""
        class HanaBasicAuthHandler(flight.ServerAuthHandler):
            def __init__(self, server):
                self.server = server
                self._basic_auth = flight.BasicAuth(self.server.hana_user, self.server.hana_password)
            
            def authenticate(self, client_auth, outgoing_auth):
                auth = flight.BasicAuth.deserialize(client_auth)
                if auth.username != self.server.hana_user or auth.password != self.server.hana_password:
                    raise flight.FlightUnauthenticatedError("Invalid username or password")
                outgoing_auth.write(self._basic_auth.serialize())
            
            def is_valid(self, token):
                return token == self._basic_auth.serialize()
        
        return HanaBasicAuthHandler(self)
    
    @wrap_hana_errors
    def _get_db_connection(self) -> Any:
        """Get a database connection from the pool."""
        import threading
        thread_id = threading.get_ident()
        
        if thread_id not in self._db_connections:
            conn_args = {
                "address": self.hana_host,
                "port": self.hana_port,
                "user": self.hana_user,
                "password": self.hana_password,
                **self.hana_connection_args
            }
            
            try:
                connection = dbapi.connect(**conn_args)
                self._db_connections[thread_id] = connection
                logger.info(f"Created new database connection for thread {thread_id}")
                return connection
            except Exception as e:
                logger.error(f"Failed to establish database connection: {str(e)}")
                raise ConnectionError(f"Database connection failed: {str(e)}")
        
        return self._db_connections[thread_id]
    
    def _register_table(self, table_name: str, schema: pa.Schema):
        """Register a table with its schema."""
        self._registered_tables[table_name] = schema
    
    def _serialize_binary_format(self, values: List[float]) -> bytes:
        """Convert a list of floats to binary format."""
        import struct
        
        if self.vector_column_type == "HALF_VECTOR":
            # 2-byte half-precision float serialization
            return struct.pack(f"<I{len(values)}e", len(values), *values)
        else:
            # 4-byte float serialization (standard FVECS format)
            return struct.pack(f"<I{len(values)}f", len(values), *values)
    
    def _deserialize_binary_format(self, binary_data: bytes) -> List[float]:
        """Convert binary data to a list of floats."""
        import struct
        
        # Read 4-byte dimension header
        dim = struct.unpack("<I", binary_data[:4])[0]
        
        if self.vector_column_type == "HALF_VECTOR":
            # 2-byte half-precision float deserialization
            values = struct.unpack(f"<{dim}e", binary_data[4:4+dim*2])
        else:
            # 4-byte float deserialization
            values = struct.unpack(f"<{dim}f", binary_data[4:4+dim*4])
            
        return list(values)
    
    def _execute_query(self, query: str, params: Optional[List[Any]] = None) -> List[Tuple]:
        """Execute a query against SAP HANA."""
        conn = self._get_db_connection()
        cursor = conn.cursor()
        
        try:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
                
            result = cursor.fetchall() if cursor.description else []
            return result
        finally:
            cursor.close()
    
    def _get_table_schema(self, table_name: str) -> pa.Schema:
        """Get Arrow schema for a table from SAP HANA."""
        # Check if schema is already registered
        if table_name in self._registered_tables:
            return self._registered_tables[table_name]
        
        # Query table schema from SAP HANA
        query = f"""
        SELECT COLUMN_NAME, DATA_TYPE_NAME
        FROM TABLE_COLUMNS
        WHERE TABLE_NAME = ?
        ORDER BY POSITION
        """
        
        result = self._execute_query(query, [table_name.upper()])
        
        if not result:
            raise ValueError(f"Table {table_name} not found")
        
        # Map SAP HANA types to Arrow types
        fields = []
        for col_name, col_type in result:
            arrow_type = None
            
            if col_type in ("VARCHAR", "NVARCHAR", "CHAR", "NCHAR", "TEXT"):
                arrow_type = pa.string()
            elif col_type == "INTEGER":
                arrow_type = pa.int32()
            elif col_type == "BIGINT":
                arrow_type = pa.int64()
            elif col_type in ("REAL", "FLOAT"):
                arrow_type = pa.float32()
            elif col_type == "DOUBLE":
                arrow_type = pa.float64()
            elif col_type == "BOOLEAN":
                arrow_type = pa.bool_()
            elif col_type in ("BLOB", "CLOB"):
                arrow_type = pa.binary()
            elif col_type in ("DATE", "TIME", "TIMESTAMP"):
                arrow_type = pa.timestamp('us')
            elif col_type in ("REAL_VECTOR", "HALF_VECTOR", "VARBINARY"):
                # For vector types, we'll use binary for storage and fixed-size list for interface
                arrow_type = pa.binary()
            else:
                # Default to binary for unknown types
                arrow_type = pa.binary()
                
            fields.append(pa.field(col_name.lower(), arrow_type))
        
        schema = pa.schema(fields)
        self._registered_tables[table_name] = schema
        return schema
    
    def _vector_data_to_record_batch(
        self, 
        data: List[Tuple],
        table_name: str,
        include_vectors: bool = True
    ) -> pa.RecordBatch:
        """Convert database query results to an Arrow RecordBatch."""
        schema = self._get_table_schema(table_name)
        field_names = [field.name for field in schema]
        
        # Prepare arrays for each column
        arrays = []
        for i, field in enumerate(schema):
            col_values = [row[i] for row in data]
            
            if field.type == pa.binary() and i < len(data[0]) and isinstance(data[0][i], bytes):
                # Check if this might be a vector column
                try:
                    # Try to deserialize the first non-null value
                    for val in col_values:
                        if val is not None:
                            vector = self._deserialize_binary_format(val)
                            dim = len(vector)
                            
                            # If successful, convert all values to fixed-size lists
                            if include_vectors:
                                vectors = []
                                for v in col_values:
                                    if v is None:
                                        vectors.append([0.0] * dim)
                                    else:
                                        vectors.append(self._deserialize_binary_format(v))
                                
                                # Create fixed-size list array
                                vectors_np = np.array(vectors, dtype=np.float32)
                                array = pa.FixedSizeListArray.from_arrays(
                                    pa.array(vectors_np.flatten()), 
                                    vectors_np.shape[1]
                                )
                                arrays.append(array)
                            else:
                                # Skip vector column if not requested
                                arrays.append(pa.array([None] * len(col_values)))
                            break
                    continue
                except Exception:
                    # Not a vector column, handle as normal binary
                    pass
            
            # Handle regular column
            arrays.append(pa.array(col_values))
        
        return pa.RecordBatch.from_arrays(arrays, field_names)
    
    def _parse_command(self, command: bytes) -> Dict[str, Any]:
        """Parse a command from a FlightDescriptor."""
        try:
            return json.loads(command.decode('utf-8'))
        except json.JSONDecodeError:
            raise ValueError("Invalid command format")
    
    def _get_similarity_search_query(
        self,
        table_name: str,
        query_vector: List[float],
        k: int,
        filter_query: Optional[str],
        distance_strategy: str
    ) -> Tuple[str, List[Any]]:
        """Generate a similarity search query for SAP HANA."""
        # Serialize query vector
        vector_binary = self._serialize_binary_format(query_vector)
        
        # Determine distance function
        if distance_strategy.lower() == "cosine":
            distance_func = "COSINE_SIMILARITY"
        elif distance_strategy.lower() == "l2":
            distance_func = "L2DISTANCE"
        elif distance_strategy.lower() == "dot":
            distance_func = "DOT_PRODUCT"
        else:
            raise ValueError(f"Unsupported distance strategy: {distance_strategy}")
        
        # Base query
        query = f"""
        SELECT *,
               {distance_func}(VEC_VECTOR, ?) AS SCORE
        FROM {table_name}
        """
        
        # Add filter if provided
        if filter_query:
            query += f" WHERE {filter_query}"
        
        # Add order by and limit
        query += f"""
        ORDER BY SCORE {"DESC" if distance_strategy.lower() == "dot" else "ASC"}
        LIMIT {k}
        """
        
        return query, [vector_binary]
    
    # Arrow Flight protocol implementation methods
    
    def list_flights(self, context, criteria):
        """List available flights (tables)."""
        try:
            # Query available tables with vector columns
            query = """
            SELECT DISTINCT TABLE_NAME 
            FROM TABLE_COLUMNS 
            WHERE DATA_TYPE_NAME IN ('VARBINARY', 'BLOB')
            ORDER BY TABLE_NAME
            """
            
            result = self._execute_query(query)
            
            # Create FlightInfo objects for each table
            for table_name, in result:
                try:
                    schema = self._get_table_schema(table_name)
                    descriptor = flight.FlightDescriptor.for_path(table_name.lower())
                    
                    endpoint = flight.FlightEndpoint(
                        ticket=flight.Ticket(table_name.lower().encode('utf-8')),
                        locations=[flight.Location(self.location)]
                    )
                    
                    info = flight.FlightInfo(
                        schema=schema,
                        descriptor=descriptor,
                        endpoints=[endpoint],
                        total_records=-1,
                        total_bytes=-1
                    )
                    
                    yield info
                except Exception as e:
                    logger.warning(f"Error getting schema for table {table_name}: {str(e)}")
        except Exception as e:
            logger.error(f"Error listing flights: {str(e)}")
            raise flight.FlightServerError(f"Failed to list flights: {str(e)}")
    
    def get_flight_info(self, context, descriptor):
        """Get information about a specific flight."""
        try:
            if descriptor.path:
                # This is a table name request
                table_name = descriptor.path[0].decode('utf-8')
                schema = self._get_table_schema(table_name)
                
                endpoint = flight.FlightEndpoint(
                    ticket=flight.Ticket(table_name.encode('utf-8')),
                    locations=[flight.Location(self.location)]
                )
                
                return flight.FlightInfo(
                    schema=schema,
                    descriptor=descriptor,
                    endpoints=[endpoint],
                    total_records=-1,
                    total_bytes=-1
                )
            
            elif descriptor.command:
                # This is a command request (like a search)
                command_data = self._parse_command(descriptor.command)
                
                if "table" in command_data and "query_vector" in command_data:
                    # This is a similarity search command
                    table_name = command_data["table"]
                    schema = self._get_table_schema(table_name)
                    
                    # Add score field to schema
                    fields = list(schema.fields)
                    fields.append(pa.field("score", pa.float32()))
                    schema = pa.schema(fields)
                    
                    endpoint = flight.FlightEndpoint(
                        ticket=flight.Ticket(descriptor.command),
                        locations=[flight.Location(self.location)]
                    )
                    
                    return flight.FlightInfo(
                        schema=schema,
                        descriptor=descriptor,
                        endpoints=[endpoint],
                        total_records=-1,
                        total_bytes=-1
                    )
                
                else:
                    raise ValueError("Unsupported command format")
            
            else:
                raise ValueError("Invalid flight descriptor")
                
        except Exception as e:
            logger.error(f"Error getting flight info: {str(e)}")
            raise flight.FlightServerError(f"Failed to get flight info: {str(e)}")
    
    def do_put(self, context, descriptor, reader, writer):
        """Handle PUT operations (inserting data)."""
        try:
            if not descriptor.path:
                raise ValueError("Table name not specified in descriptor path")
                
            table_name = descriptor.path[0].decode('utf-8')
            schema = reader.schema
            
            # Check if table exists
            try:
                self._get_table_schema(table_name)
            except ValueError:
                # Table doesn't exist, create it
                logger.warning(f"Table {table_name} not found, would need to create it")
                # TODO: Implement table creation logic
                raise NotImplementedError("Automatic table creation not yet implemented")
            
            # Process record batches
            conn = self._get_db_connection()
            cursor = conn.cursor()
            
            try:
                for batch in reader:
                    # Convert batch to rows
                    rows = []
                    for i in range(batch.num_rows):
                        row = {}
                        for j, col_name in enumerate(schema.names):
                            # Special handling for vector column
                            if isinstance(batch.column(j).type, pa.FixedSizeListType):
                                # Convert fixed-size list to binary
                                vector = batch.column(j)[i].values.to_numpy()
                                row[col_name] = self._serialize_binary_format(vector.tolist())
                            else:
                                row[col_name] = batch.column(j)[i].as_py()
                        
                        rows.append(row)
                    
                    if rows:
                        # Generate INSERT statement
                        columns = ", ".join(rows[0].keys())
                        placeholders = ", ".join(["?"] * len(rows[0]))
                        
                        insert_query = f"""
                        INSERT INTO {table_name} ({columns})
                        VALUES ({placeholders})
                        """
                        
                        # Execute batch insert
                        cursor.executemany(
                            insert_query,
                            [[row[col] for col in rows[0].keys()] for row in rows]
                        )
                        
                        conn.commit()
                        logger.info(f"Inserted {len(rows)} rows into {table_name}")
            finally:
                cursor.close()
                
        except Exception as e:
            logger.error(f"Error in do_put: {str(e)}")
            raise flight.FlightServerError(f"Failed to put data: {str(e)}")
    
    def do_get(self, context, ticket):
        """Handle GET operations (retrieving data)."""
        try:
            ticket_bytes = ticket.ticket
            
            if ticket_bytes.startswith(b"{"):
                # This is a command ticket (like a search)
                command_data = self._parse_command(ticket_bytes)
                
                if "table" in command_data and "query_vector" in command_data:
                    # This is a similarity search command
                    table_name = command_data["table"]
                    query_vector = command_data["query_vector"]
                    k = command_data.get("k", 4)
                    filter_query = command_data.get("filter")
                    include_metadata = command_data.get("include_metadata", True)
                    include_vectors = command_data.get("include_vectors", False)
                    distance_strategy = command_data.get("distance_strategy", "cosine")
                    
                    # Generate and execute query
                    query, params = self._get_similarity_search_query(
                        table_name, query_vector, k, filter_query, distance_strategy
                    )
                    
                    result = self._execute_query(query, params)
                    
                    # Convert to record batch and yield
                    if result:
                        batch = self._vector_data_to_record_batch(
                            result, table_name, include_vectors=include_vectors
                        )
                        yield batch
            
            else:
                # This is a table name ticket
                table_name = ticket_bytes.decode('utf-8')
                
                # Query all data from the table
                query = f"SELECT * FROM {table_name}"
                result = self._execute_query(query)
                
                # Convert to record batch and yield in chunks
                chunk_size = 1000
                for i in range(0, len(result), chunk_size):
                    chunk = result[i:i+chunk_size]
                    batch = self._vector_data_to_record_batch(chunk, table_name)
                    yield batch
                    
        except Exception as e:
            logger.error(f"Error in do_get: {str(e)}")
            raise flight.FlightServerError(f"Failed to get data: {str(e)}")
    
    def do_action(self, context, action):
        """Handle custom actions."""
        try:
            if action.type == "ping":
                # Simple ping action to check server health
                yield flight.Result(b"pong")
                
            elif action.type == "register_table":
                # Register a table with its schema
                data = json.loads(action.body.to_pybytes().decode('utf-8'))
                table_name = data["table_name"]
                
                # Get table schema
                schema = self._get_table_schema(table_name)
                self._register_table(table_name, schema)
                
                yield flight.Result(b"Table registered successfully")
                
            else:
                raise NotImplementedError(f"Action {action.type} not implemented")
                
        except Exception as e:
            logger.error(f"Error in do_action: {str(e)}")
            raise flight.FlightServerError(f"Action failed: {str(e)}")
    
    def list_actions(self, context):
        """List available custom actions."""
        return [
            flight.ActionType("ping", "Check if the server is responsive"),
            flight.ActionType("register_table", "Register a table with its schema")
        ]
    
    def serve(self):
        """Start the Arrow Flight server."""
        try:
            # Initialize location
            self.init_location()
            
            # Log server details
            logger.info(f"Starting Arrow Flight server at {self.location}")
            logger.info(f"Connected to SAP HANA at {self.hana_host}")
            
            # Start server
            self.run()
            
        except Exception as e:
            logger.error(f"Error starting Arrow Flight server: {str(e)}")
            raise RuntimeError(f"Failed to start server: {str(e)}")
    
    def shutdown(self):
        """Shutdown the server and clean up resources."""
        try:
            # Close database connections
            for conn in self._db_connections.values():
                conn.close()
            
            # Shutdown connection pool
            self._connection_pool.shutdown()
            
            # Shutdown server
            super().shutdown()
            
            logger.info("Arrow Flight server shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during server shutdown: {str(e)}")


def start_arrow_flight_server(
    host: str = "localhost",
    port: int = 8815,
    hana_host: str = "localhost",
    hana_port: int = 30015,
    hana_user: str = "SYSTEM",
    hana_password: str = "",
    vector_column_type: str = "REAL_VECTOR",
    use_tls: bool = False,
    tls_cert_file: Optional[str] = None,
    tls_key_file: Optional[str] = None,
):
    """
    Start an Arrow Flight server for SAP HANA.
    
    Args:
        host: Server host address
        port: Server port
        hana_host: SAP HANA host address
        hana_port: SAP HANA port
        hana_user: SAP HANA username
        hana_password: SAP HANA password
        vector_column_type: Type of vector column in SAP HANA
        use_tls: Whether to use TLS
        tls_cert_file: Path to TLS certificate file
        tls_key_file: Path to TLS key file
        
    Returns:
        The server instance
    """
    # Configure TLS if requested
    tls_certificates = None
    if use_tls and tls_cert_file and tls_key_file:
        with open(tls_cert_file, "rb") as cert_file:
            cert = cert_file.read()
        with open(tls_key_file, "rb") as key_file:
            key = key_file.read()
        tls_certificates = [(cert, key)]
    
    # Create and start server
    location = None
    if use_tls:
        location = f"grpc+tls://{host}:{port}"
    
    server = HanaArrowFlightServer(
        host=host,
        port=port,
        location=location,
        hana_host=hana_host,
        hana_port=hana_port,
        hana_user=hana_user,
        hana_password=hana_password,
        tls_certificates=tls_certificates,
        vector_column_type=vector_column_type,
    )
    
    server.serve()
    return server