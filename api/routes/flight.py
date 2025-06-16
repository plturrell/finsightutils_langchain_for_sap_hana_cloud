"""
Arrow Flight API for efficient vector data transfer.

This module implements the Apache Arrow Flight protocol for the SAP HANA Cloud
vector store, enabling high-performance transfer of vector embeddings between
the API and clients.

It supports both single-GPU and multi-GPU operations for improved performance
and scalability.
"""

import logging
import json
import numpy as np
import pyarrow as pa
import pyarrow.flight as flight
from typing import Dict, List, Any, Optional, Union, Tuple
import threading
import time
import os

from fastapi import APIRouter, HTTPException, Depends, Request, Response
from pydantic import BaseModel, Field

from database import get_db_connection
from models.flight_models import (
    FlightQueryRequest,
    FlightQueryResponse,
    FlightUploadRequest,
    FlightUploadResponse,
    FlightListResponse,
    FlightInfoResponse,
    FlightMultiGPUConfig,
    FlightMultiGPURequest,
    FlightMultiGPUResponse
)

# Import multi-GPU support if available
try:
    from langchain_hana.gpu import (
        ArrowFlightMultiGPUManager,
        ArrowGpuMemoryManager,
        HAS_ARROW_FLIGHT
    )
    HAS_MULTI_GPU_SUPPORT = True
except ImportError:
    HAS_MULTI_GPU_SUPPORT = False

# Import GPU manager
try:
    from api.multi_gpu import get_gpu_manager
    HAS_GPU_MANAGER = True
except ImportError:
    HAS_GPU_MANAGER = False

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/flight", tags=["flight"])

class FlightServer(flight.FlightServerBase):
    """
    Arrow Flight server implementation for vector data transfer.
    
    This class provides high-performance data transfer capabilities for 
    vector embeddings stored in SAP HANA Cloud.
    """
    
    def __init__(self, host="localhost", port=8815, location=None, 
                 middleware=None, options=None, **kwargs):
        """Initialize the Flight server."""
        super(FlightServer, self).__init__(
            host=host, port=port, location=location, 
            middleware=middleware, options=options, **kwargs
        )
        self.flights = {}
        self.db_lock = threading.Lock()
        
        # Flight location, combining host and port
        self.location = location or f"grpc://{host}:{port}"
        
    def _get_db_connection(self):
        """Get a database connection with thread safety."""
        with self.db_lock:
            return get_db_connection()
    
    def list_flights(self, context, criteria):
        """
        List available vector collections/tables.
        
        Args:
            context: Flight server context
            criteria: Optional criteria to filter the list
            
        Yields:
            FlightInfo objects describing available vector collections
        """
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            
            # Query to get vector tables from the database
            query = """
            SELECT TABLE_NAME 
            FROM SYS.TABLE_COLUMNS 
            WHERE DATA_TYPE_NAME IN ('REAL_VECTOR', 'HALF_VECTOR')
            GROUP BY TABLE_NAME
            """
            
            cursor.execute(query)
            tables = cursor.fetchall()
            
            for table in tables:
                table_name = table[0]
                
                # Create a descriptor for this table
                descriptor = flight.FlightDescriptor.for_path(table_name.encode())
                
                # Get the schema of the table
                schema_query = f"""
                SELECT COLUMN_NAME, DATA_TYPE_NAME
                FROM SYS.TABLE_COLUMNS
                WHERE TABLE_NAME = '{table_name}'
                """
                
                cursor.execute(schema_query)
                columns = cursor.fetchall()
                
                # Create a basic schema (will be refined in get_schema)
                fields = []
                for col in columns:
                    if col[1] in ('REAL_VECTOR', 'HALF_VECTOR'):
                        fields.append(pa.field(col[0], pa.list_(pa.float32())))
                    elif col[1] == 'NCLOB':
                        fields.append(pa.field(col[0], pa.string()))
                    else:
                        fields.append(pa.field(col[0], pa.string()))
                
                schema = pa.schema(fields)
                
                # Create endpoints
                endpoints = [flight.FlightEndpoint(
                    ticket=flight.Ticket(table_name.encode()),
                    locations=[flight.Location(self.location)]
                )]
                
                # Count rows in the table
                count_query = f"SELECT COUNT(*) FROM \"{table_name}\""
                cursor.execute(count_query)
                total_bytes = cursor.fetchone()[0]
                
                # Create and yield FlightInfo
                flight_info = flight.FlightInfo(
                    schema=schema,
                    descriptor=descriptor,
                    endpoints=endpoints,
                    total_records=total_bytes,
                    total_bytes=total_bytes * 100  # Rough estimate
                )
                
                yield flight_info
                
        except Exception as e:
            logger.error(f"Error in list_flights: {str(e)}")
            raise flight.FlightServerError(f"Error listing flights: {str(e)}")
    
    def get_flight_info(self, context, descriptor):
        """
        Get information about a specific vector collection.
        
        Args:
            context: Flight server context
            descriptor: FlightDescriptor identifying the collection
            
        Returns:
            FlightInfo object with details about the collection
        """
        try:
            # Extract table name from descriptor
            if descriptor.type == flight.DescriptorType.PATH:
                table_name = descriptor.path[0].decode()
            else:
                table_info = json.loads(descriptor.command)
                table_name = table_info.get("table")
            
            if not table_name:
                raise flight.FlightServerError("Missing table name in descriptor")
            
            conn = self._get_db_connection()
            cursor = conn.cursor()
            
            # Check if table exists
            check_query = """
            SELECT COUNT(*) 
            FROM SYS.TABLES 
            WHERE TABLE_NAME = ? AND SCHEMA_NAME = CURRENT_SCHEMA
            """
            cursor.execute(check_query, (table_name,))
            if cursor.fetchone()[0] == 0:
                raise flight.FlightServerError(f"Table {table_name} not found")
            
            # Get the schema of the table
            schema_query = f"""
            SELECT COLUMN_NAME, DATA_TYPE_NAME
            FROM SYS.TABLE_COLUMNS
            WHERE TABLE_NAME = '{table_name}'
            """
            
            cursor.execute(schema_query)
            columns = cursor.fetchall()
            
            # Create schema
            fields = []
            for col in columns:
                if col[1] in ('REAL_VECTOR', 'HALF_VECTOR'):
                    fields.append(pa.field(col[0], pa.list_(pa.float32())))
                elif col[1] == 'NCLOB':
                    fields.append(pa.field(col[0], pa.string()))
                else:
                    fields.append(pa.field(col[0], pa.string()))
            
            schema = pa.schema(fields)
            
            # Create endpoints
            endpoints = [flight.FlightEndpoint(
                ticket=flight.Ticket(table_name.encode()),
                locations=[flight.Location(self.location)]
            )]
            
            # Count rows in the table
            count_query = f"SELECT COUNT(*) FROM \"{table_name}\""
            cursor.execute(count_query)
            total_records = cursor.fetchone()[0]
            
            # Create FlightInfo
            return flight.FlightInfo(
                schema=schema,
                descriptor=descriptor,
                endpoints=endpoints,
                total_records=total_records,
                total_bytes=total_records * 100  # Rough estimate
            )
            
        except flight.FlightServerError:
            raise
        except Exception as e:
            logger.error(f"Error in get_flight_info: {str(e)}")
            raise flight.FlightServerError(f"Error getting flight info: {str(e)}")
    
    def do_get(self, context, ticket):
        """
        Retrieve vectors from the database.
        
        Args:
            context: Flight server context
            ticket: Ticket identifying what to retrieve
            
        Returns:
            FlightDataStream of vector data
        """
        try:
            # Parse the ticket
            if isinstance(ticket.ticket, bytes):
                ticket_info = ticket.ticket.decode()
                try:
                    # Try to parse as JSON
                    ticket_data = json.loads(ticket_info)
                    table_name = ticket_data.get("table")
                    filter_dict = ticket_data.get("filter", {})
                    limit = ticket_data.get("limit", 1000)
                    offset = ticket_data.get("offset", 0)
                except json.JSONDecodeError:
                    # Simple string ticket
                    table_name = ticket_info
                    filter_dict = {}
                    limit = 1000
                    offset = 0
            else:
                raise flight.FlightServerError("Invalid ticket format")
            
            if not table_name:
                raise flight.FlightServerError("Missing table name in ticket")
            
            conn = self._get_db_connection()
            cursor = conn.cursor()
            
            # Build the filter SQL if needed
            filter_sql = ""
            filter_params = []
            
            if filter_dict:
                filter_parts = []
                for key, value in filter_dict.items():
                    if key == 'content':
                        filter_parts.append(f"DOCUMENT LIKE ?")
                        filter_params.append(f"%{value}%")
                    elif isinstance(value, list):
                        placeholders = ','.join(['?'] * len(value))
                        filter_parts.append(f"JSON_VALUE(METADATA, '$.{key}') IN ({placeholders})")
                        filter_params.extend(value)
                    else:
                        filter_parts.append(f"JSON_VALUE(METADATA, '$.{key}') = ?")
                        filter_params.append(value)
                
                if filter_parts:
                    filter_sql = "WHERE " + " AND ".join(filter_parts)
            
            # Query to get the vectors
            query = f"""
            SELECT ID, DOCUMENT, METADATA, VECTOR 
            FROM "{table_name}"
            {filter_sql}
            LIMIT {limit} OFFSET {offset}
            """
            
            cursor.execute(query, filter_params)
            rows = cursor.fetchall()
            
            # Create batches for Arrow
            ids = []
            documents = []
            metadatas = []
            vectors = []
            
            for row in rows:
                vector_id = row[0]
                document = row[1]
                metadata = row[2] if row[2] else "{}"
                vector = row[3]
                
                ids.append(vector_id)
                documents.append(document)
                metadatas.append(metadata)
                
                # Convert vector to list of floats
                if isinstance(vector, bytes):
                    # Parse vector from binary
                    import struct
                    # The first 4 bytes contain the vector dimension
                    dim = struct.unpack_from("<I", vector, 0)[0]
                    # The rest of the bytes contain the vector values
                    vector_data = list(struct.unpack_from(f"<{dim}f", vector, 4))
                    vectors.append(vector_data)
                else:
                    # Handle other formats if needed
                    vectors.append([])
            
            # Create Arrow table
            batch = pa.RecordBatch.from_arrays(
                [
                    pa.array(ids),
                    pa.array(documents),
                    pa.array(metadatas),
                    pa.array(vectors, type=pa.list_(pa.float32()))
                ],
                names=['id', 'document', 'metadata', 'vector']
            )
            
            # Return as a record batch stream
            return flight.RecordBatchStream(batch)
            
        except flight.FlightServerError:
            raise
        except Exception as e:
            logger.error(f"Error in do_get: {str(e)}")
            raise flight.FlightServerError(f"Error retrieving data: {str(e)}")
    
    def do_put(self, context, descriptor, reader, writer):
        """
        Store vectors in the database.
        
        Args:
            context: Flight server context
            descriptor: FlightDescriptor identifying where to store
            reader: FlightDataStream containing the data
            writer: PutResult writer
            
        Returns:
            None
        """
        try:
            # Extract table name from descriptor
            if descriptor.type == flight.DescriptorType.PATH:
                table_name = descriptor.path[0].decode()
            else:
                command_info = json.loads(descriptor.command)
                table_name = command_info.get("table")
                
            if not table_name:
                raise flight.FlightServerError("Missing table name in descriptor")
            
            conn = self._get_db_connection()
            cursor = conn.cursor()
            
            # Check if table exists
            check_query = """
            SELECT COUNT(*) 
            FROM SYS.TABLES 
            WHERE TABLE_NAME = ? AND SCHEMA_NAME = CURRENT_SCHEMA
            """
            cursor.execute(check_query, (table_name,))
            if cursor.fetchone()[0] == 0:
                # Table doesn't exist, create it
                create_table_query = f"""
                CREATE TABLE "{table_name}" (
                    "ID" INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                    "DOCUMENT" NCLOB,
                    "METADATA" NCLOB,
                    "VECTOR" REAL_VECTOR
                )
                """
                cursor.execute(create_table_query)
                conn.commit()
            
            # Read all the data
            data = reader.read_all()
            
            # Insert data into the table
            insert_query = f"""
            INSERT INTO "{table_name}" (
                "DOCUMENT", 
                "METADATA", 
                "VECTOR"
            ) VALUES (?, ?, ?)
            """
            
            batch_size = 100
            total_rows = data.num_rows
            total_inserted = 0
            
            for batch_start in range(0, total_rows, batch_size):
                batch_end = min(batch_start + batch_size, total_rows)
                batch_data = []
                
                for i in range(batch_start, batch_end):
                    document = data.column('document')[i].as_py()
                    metadata = data.column('metadata')[i].as_py()
                    vector = data.column('vector')[i].as_py()
                    
                    # Convert vector to binary format
                    import struct
                    vector_binary = struct.pack(f"<I{len(vector)}f", len(vector), *vector)
                    
                    batch_data.append((document, metadata, vector_binary))
                
                cursor.executemany(insert_query, batch_data)
                conn.commit()
                
                total_inserted += (batch_end - batch_start)
                writer.puts_done()
                
            return None
            
        except flight.FlightServerError:
            raise
        except Exception as e:
            logger.error(f"Error in do_put: {str(e)}")
            raise flight.FlightServerError(f"Error storing data: {str(e)}")

# Initialize the Flight server
flight_server = None
flight_server_thread = None

def get_flight_server():
    """Get or create the Flight server instance."""
    global flight_server
    if flight_server is None:
        # Get configuration from environment variables
        host = os.environ.get("FLIGHT_HOST", "localhost")
        port = int(os.environ.get("FLIGHT_PORT", "8815"))
        
        flight_server = FlightServer(
            host=host,
            port=port,
            location=f"grpc://{host}:{port}"
        )
    return flight_server

def start_flight_server():
    """Start the Flight server in a separate thread."""
    global flight_server_thread
    if flight_server_thread is None or not flight_server_thread.is_alive():
        server = get_flight_server()
        flight_server_thread = threading.Thread(
            target=server.serve,
            daemon=True
        )
        flight_server_thread.start()
        
        # Wait for server to start
        time.sleep(2)
        logger.info(f"Flight server started at {server.location}")
    return flight_server_thread

# FastAPI endpoints to interact with Flight server

@router.get("/info", response_model=FlightInfoResponse)
async def flight_info():
    """Get information about the Flight server."""
    try:
        server = get_flight_server()
        return FlightInfoResponse(
            host=server.host,
            port=server.port,
            location=server.location,
            status="running" if flight_server_thread and flight_server_thread.is_alive() else "stopped"
        )
    except Exception as e:
        logger.error(f"Error getting Flight server info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting Flight server info: {str(e)}")

@router.post("/start", response_model=FlightInfoResponse)
async def start_server():
    """Start the Flight server if it's not already running."""
    try:
        start_flight_server()
        server = get_flight_server()
        return FlightInfoResponse(
            host=server.host,
            port=server.port,
            location=server.location,
            status="running"
        )
    except Exception as e:
        logger.error(f"Error starting Flight server: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting Flight server: {str(e)}")

@router.post("/query", response_model=FlightQueryResponse)
async def flight_query(request: FlightQueryRequest):
    """
    Create a Flight query for retrieving vectors.
    
    Args:
        request: Query parameters
        
    Returns:
        FlightQueryResponse: Response with Flight ticket and location
    """
    try:
        server = get_flight_server()
        
        # Ensure server is running
        if not flight_server_thread or not flight_server_thread.is_alive():
            start_flight_server()
        
        # Create ticket for the query
        ticket_data = {
            "table": request.table_name,
            "filter": request.filter,
            "limit": request.limit,
            "offset": request.offset
        }
        
        ticket = json.dumps(ticket_data)
        
        return FlightQueryResponse(
            ticket=ticket,
            location=server.location,
            schema=None  # Schema will be determined by client
        )
    except Exception as e:
        logger.error(f"Error creating Flight query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating Flight query: {str(e)}")

@router.post("/upload", response_model=FlightUploadResponse)
async def flight_upload(request: FlightUploadRequest):
    """
    Create a Flight descriptor for uploading vectors.
    
    Args:
        request: Upload parameters
        
    Returns:
        FlightUploadResponse: Response with Flight descriptor and location
    """
    try:
        server = get_flight_server()
        
        # Ensure server is running
        if not flight_server_thread or not flight_server_thread.is_alive():
            start_flight_server()
        
        # Create descriptor for the upload
        descriptor_data = {
            "table": request.table_name,
            "create_if_not_exists": request.create_if_not_exists
        }
        
        descriptor = json.dumps(descriptor_data)
        
        return FlightUploadResponse(
            descriptor=descriptor,
            location=server.location
        )
    except Exception as e:
        logger.error(f"Error creating Flight upload descriptor: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating Flight upload descriptor: {str(e)}")

@router.get("/list", response_model=FlightListResponse)
async def flight_list():
    """
    List available vector collections.
    
    Returns:
        FlightListResponse: Response with list of available collections
    """
    try:
        server = get_flight_server()
        
        # Ensure server is running
        if not flight_server_thread or not flight_server_thread.is_alive():
            start_flight_server()
        
        # Create client to connect to our server
        import pyarrow.flight as flight
        client = flight.connect(server.location)
        
        # List available flights
        collections = []
        for flight_info in client.list_flights():
            if flight_info.descriptor.type == flight.DescriptorType.PATH:
                collection_name = flight_info.descriptor.path[0].decode()
                collections.append({
                    "name": collection_name,
                    "total_records": flight_info.total_records
                })
        
        return FlightListResponse(
            collections=collections,
            location=server.location
        )
    except Exception as e:
        logger.error(f"Error listing Flight collections: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing Flight collections: {str(e)}")

# Initialize the Flight server when the module loads
def init_flight_server():
    """Initialize the Flight server when the API starts."""
    if os.environ.get("FLIGHT_AUTO_START", "true").lower() in ("true", "1", "yes"):
        try:
            start_flight_server()
            logger.info("Flight server initialized at startup")
        except Exception as e:
            logger.error(f"Error initializing Flight server: {str(e)}")

# Multi-GPU support utilities
multi_gpu_manager = None

def get_multi_gpu_manager(config: FlightMultiGPUConfig = None) -> Optional[ArrowFlightMultiGPUManager]:
    """
    Get or create the multi-GPU manager.
    
    Args:
        config: Optional multi-GPU configuration
        
    Returns:
        Multi-GPU manager or None if not available
    """
    global multi_gpu_manager
    
    if not HAS_MULTI_GPU_SUPPORT:
        logger.warning("Multi-GPU support is not available")
        return None
    
    if multi_gpu_manager is None:
        # Get server and its connection details
        server = get_flight_server()
        host = server.host
        port = server.port
        
        # Extract configuration parameters
        if config is None:
            config = FlightMultiGPUConfig()
        
        # Get GPU IDs to use
        gpu_ids = config.gpu_ids
        if gpu_ids is None and HAS_GPU_MANAGER:
            # Use GPU manager to get available GPUs
            gpu_manager = get_gpu_manager()
            if not gpu_manager.cpu_only and gpu_manager.device_count > 0:
                gpu_ids = gpu_manager.devices
            else:
                logger.warning("No GPUs available for multi-GPU operations")
                return None
        
        # Create flight clients for each GPU
        flight_clients = [
            ArrowFlightClient(
                host=host,
                port=port,
                use_tls=False
            )
            for _ in range(len(gpu_ids))
        ]
        
        # Create multi-GPU manager
        try:
            multi_gpu_manager = ArrowFlightMultiGPUManager(
                flight_clients=flight_clients,
                gpu_ids=gpu_ids,
                batch_size=config.batch_size,
                memory_fraction=config.memory_fraction,
                distribution_strategy=config.distribution_strategy
            )
            logger.info(f"Created multi-GPU manager with {len(gpu_ids)} GPUs: {gpu_ids}")
        except Exception as e:
            logger.error(f"Error creating multi-GPU manager: {str(e)}")
            return None
    
    return multi_gpu_manager


@router.post("/multi-gpu/info", response_model=FlightMultiGPUResponse)
async def multi_gpu_info(request: FlightMultiGPURequest):
    """
    Get information about multi-GPU Flight operations.
    
    Args:
        request: Multi-GPU request parameters
        
    Returns:
        FlightMultiGPUResponse: Response with multi-GPU configuration and metrics
    """
    try:
        if not HAS_MULTI_GPU_SUPPORT:
            raise HTTPException(
                status_code=400,
                detail="Multi-GPU support is not available"
            )
        
        # Get or create multi-GPU manager
        mgpu_manager = get_multi_gpu_manager(request.config)
        if mgpu_manager is None:
            raise HTTPException(
                status_code=400,
                detail="Failed to initialize multi-GPU manager"
            )
        
        # Get optimal batch sizes for each GPU
        batch_sizes = mgpu_manager.get_optimal_batch_sizes()
        
        # Get server information
        server = get_flight_server()
        
        # Create response
        return FlightMultiGPUResponse(
            gpu_ids=mgpu_manager.gpu_ids,
            num_gpus=mgpu_manager.num_gpus,
            batch_sizes=batch_sizes,
            location=server.location,
            config=request.config or FlightMultiGPUConfig()
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting multi-GPU info: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting multi-GPU info: {str(e)}"
        )


@router.post("/multi-gpu/search", response_model=FlightQueryResponse)
async def multi_gpu_search(request: FlightMultiGPURequest):
    """
    Create a Flight query for multi-GPU similarity search.
    
    Args:
        request: Multi-GPU request parameters
        
    Returns:
        FlightQueryResponse: Response with Flight ticket and location
    """
    try:
        if not HAS_MULTI_GPU_SUPPORT:
            raise HTTPException(
                status_code=400,
                detail="Multi-GPU support is not available"
            )
        
        # Get or create multi-GPU manager
        mgpu_manager = get_multi_gpu_manager(request.config)
        if mgpu_manager is None:
            raise HTTPException(
                status_code=400,
                detail="Failed to initialize multi-GPU manager"
            )
        
        # Get server information
        server = get_flight_server()
        
        # Create ticket with multi-GPU configuration
        ticket_data = {
            "table": request.table_name,
            "multi_gpu": True,
            "gpu_ids": mgpu_manager.gpu_ids,
            "distribution_strategy": request.config.distribution_strategy if request.config else "round_robin"
        }
        
        # Create ticket
        ticket = json.dumps(ticket_data)
        
        # Create response
        return FlightQueryResponse(
            ticket=ticket,
            location=server.location,
            schema=None
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating multi-GPU search query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error creating multi-GPU search query: {str(e)}"
        )


@router.post("/multi-gpu/upload", response_model=FlightUploadResponse)
async def multi_gpu_upload(request: FlightMultiGPURequest):
    """
    Create a Flight descriptor for multi-GPU vector upload.
    
    Args:
        request: Multi-GPU request parameters
        
    Returns:
        FlightUploadResponse: Response with Flight descriptor and location
    """
    try:
        if not HAS_MULTI_GPU_SUPPORT:
            raise HTTPException(
                status_code=400,
                detail="Multi-GPU support is not available"
            )
        
        # Get or create multi-GPU manager
        mgpu_manager = get_multi_gpu_manager(request.config)
        if mgpu_manager is None:
            raise HTTPException(
                status_code=400,
                detail="Failed to initialize multi-GPU manager"
            )
        
        # Get server information
        server = get_flight_server()
        
        # Create descriptor with multi-GPU configuration
        descriptor_data = {
            "table": request.table_name,
            "multi_gpu": True,
            "gpu_ids": mgpu_manager.gpu_ids,
            "create_if_not_exists": True,
            "distribution_strategy": request.config.distribution_strategy if request.config else "round_robin"
        }
        
        # Create descriptor
        descriptor = json.dumps(descriptor_data)
        
        # Create response
        return FlightUploadResponse(
            descriptor=descriptor,
            location=server.location
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating multi-GPU upload descriptor: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error creating multi-GPU upload descriptor: {str(e)}"
        )


# Call init function when module loads
init_flight_server()