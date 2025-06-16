"""
Apache Arrow Flight client implementation for SAP HANA Cloud.

This module provides a high-performance data transfer layer between
SAP HANA Cloud and GPU-accelerated vector operations using Apache Arrow Flight.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

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


class ArrowFlightClient:
    """
    Client for high-performance data transfer between SAP HANA and vector operations using Arrow Flight.
    
    This client provides optimized data transfer for vector embeddings and search operations,
    leveraging Apache Arrow's columnar format and Flight RPC framework for efficient
    serialization and zero-copy operations where possible.
    """
    
    def __init__(
        self,
        host: str,
        port: int = 8815,
        use_tls: bool = False,
        username: Optional[str] = None,
        password: Optional[str] = None,
        connection: Optional[Any] = None,
        connection_args: Optional[Dict[str, Any]] = None,
        location: Optional[str] = None,
    ):
        """
        Initialize the Arrow Flight client for SAP HANA.
        
        Args:
            host: SAP HANA host address
            port: Arrow Flight server port (default: 8815)
            use_tls: Whether to use TLS for secure connections
            username: SAP HANA username
            password: SAP HANA password
            connection: Existing SAP HANA connection (optional)
            connection_args: Additional connection arguments for SAP HANA
            location: Optional Arrow Flight location URI
        
        Raises:
            ImportError: If required dependencies are not installed
            ConnectionError: If unable to establish connection
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
        
        self.host = host
        self.port = port
        self.use_tls = use_tls
        self.username = username
        self.password = password
        self.connection_args = connection_args or {}
        
        # Setup Arrow Flight connection
        self._setup_flight_connection(location)
        
        # Setup SAP HANA connection (needed for operations not yet supported by Arrow Flight)
        self._db_connection = connection
        if self._db_connection is None:
            self._db_connection = self._create_db_connection()
    
    def _setup_flight_connection(self, location: Optional[str] = None):
        """Set up the Arrow Flight connection."""
        # Create location URI if not provided
        if location is None:
            scheme = "grpc+tls" if self.use_tls else "grpc"
            location = f"{scheme}://{self.host}:{self.port}"
        
        try:
            # Create Flight client
            self.location = location
            self.client = flight.FlightClient(location)
            
            # Set up authentication if username/password provided
            if self.username and self.password:
                self.client.authenticate(
                    flight.BasicAuth(self.username, self.password)
                )
                
            logger.info(f"Established Arrow Flight connection to {location}")
        except Exception as e:
            logger.error(f"Failed to establish Arrow Flight connection: {str(e)}")
            raise ConnectionError(f"Arrow Flight connection failed: {str(e)}")
    
    @wrap_hana_errors
    def _create_db_connection(self) -> Any:
        """Create a fallback direct database connection to SAP HANA."""
        conn_args = {
            "address": self.host,
            "user": self.username,
            "password": self.password,
            **self.connection_args
        }
        
        try:
            connection = dbapi.connect(**conn_args)
            logger.info(f"Established fallback database connection to {self.host}")
            return connection
        except Exception as e:
            logger.error(f"Failed to establish database connection: {str(e)}")
            raise ConnectionError(f"Database connection failed: {str(e)}")
    
    def get_flight_info(self, descriptor: Union[str, bytes, flight.FlightDescriptor]) -> flight.FlightInfo:
        """
        Get information about a specific flight.
        
        Args:
            descriptor: Flight descriptor (table name, query, or command)
            
        Returns:
            FlightInfo object containing schema and endpoints
        """
        if isinstance(descriptor, str):
            descriptor = flight.FlightDescriptor.for_path(descriptor)
        elif isinstance(descriptor, bytes):
            descriptor = flight.FlightDescriptor.for_command(descriptor)
            
        try:
            return self.client.get_flight_info(descriptor)
        except Exception as e:
            logger.error(f"Failed to get flight info: {str(e)}")
            raise RuntimeError(f"Failed to get flight info: {str(e)}")
    
    def _vector_to_record_batch(
        self, 
        vectors: List[List[float]], 
        metadata: Optional[List[Dict[str, Any]]] = None,
        texts: Optional[List[str]] = None,
        ids: Optional[List[str]] = None
    ) -> pa.RecordBatch:
        """
        Convert vectors, texts, and metadata to an Arrow RecordBatch.
        
        Args:
            vectors: List of embedding vectors
            metadata: Optional list of metadata dictionaries
            texts: Optional list of text strings
            ids: Optional list of IDs
            
        Returns:
            Arrow RecordBatch containing the data
        """
        # Convert vectors to numpy array then to Arrow array
        vectors_np = np.array(vectors, dtype=np.float32)
        vectors_arrow = pa.FixedSizeListArray.from_arrays(
            pa.array(vectors_np.flatten()), 
            vectors_np.shape[1]
        )
        
        # Create field dictionary
        fields = {"vector": vectors_arrow}
        
        # Add texts if provided
        if texts is not None:
            fields["text"] = pa.array(texts)
            
        # Add ids if provided
        if ids is not None:
            fields["id"] = pa.array(ids)
        elif texts is not None:
            # Generate UUIDs if IDs not provided but texts are
            fields["id"] = pa.array([str(uuid.uuid4()) for _ in range(len(texts))])
            
        # Handle metadata if provided
        if metadata is not None:
            # Convert metadata to binary JSON for now (could be optimized further)
            import json
            fields["metadata"] = pa.array([json.dumps(m).encode('utf-8') for m in metadata])
        
        # Create RecordBatch from fields
        return pa.RecordBatch.from_pydict(fields)
    
    def _record_batch_to_vectors(
        self, 
        batch: pa.RecordBatch
    ) -> Tuple[List[List[float]], Optional[List[str]], Optional[List[Dict[str, Any]]], Optional[List[str]]]:
        """
        Convert an Arrow RecordBatch to vectors, texts, metadata, and IDs.
        
        Args:
            batch: Arrow RecordBatch to convert
            
        Returns:
            Tuple of (vectors, texts, metadata, ids)
        """
        # Extract vectors
        vectors_arrow = batch.column('vector')
        vectors_np = np.array(vectors_arrow.flatten())
        vectors_list = vectors_np.reshape(-1, vectors_arrow.type.list_size).tolist()
        
        # Extract texts if available
        texts = None
        if 'text' in batch.schema.names:
            texts = batch.column('text').to_pylist()
            
        # Extract metadata if available
        metadata = None
        if 'metadata' in batch.schema.names:
            import json
            metadata = [json.loads(m.decode('utf-8')) for m in batch.column('metadata').to_pylist()]
            
        # Extract IDs if available
        ids = None
        if 'id' in batch.schema.names:
            ids = batch.column('id').to_pylist()
            
        return vectors_list, texts, metadata, ids
    
    def upload_vectors(
        self, 
        table_name: str,
        vectors: List[List[float]],
        texts: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 1000
    ) -> List[str]:
        """
        Upload vectors to the specified table using Arrow Flight.
        
        Args:
            table_name: Target table name
            vectors: List of embedding vectors
            texts: Optional list of text strings
            metadata: Optional list of metadata dictionaries
            ids: Optional list of IDs
            batch_size: Number of vectors to upload in each batch
            
        Returns:
            List of generated or provided IDs
        """
        if len(vectors) == 0:
            return []
        
        # Determine total number of batches
        num_vectors = len(vectors)
        num_batches = (num_vectors + batch_size - 1) // batch_size
        
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(num_vectors)]
        
        # Process in batches
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_vectors)
            
            # Extract batch data
            batch_vectors = vectors[start_idx:end_idx]
            batch_ids = ids[start_idx:end_idx]
            
            batch_texts = None
            if texts is not None:
                batch_texts = texts[start_idx:end_idx]
                
            batch_metadata = None
            if metadata is not None:
                batch_metadata = metadata[start_idx:end_idx]
            
            # Convert to Arrow RecordBatch
            record_batch = self._vector_to_record_batch(
                batch_vectors, batch_metadata, batch_texts, batch_ids
            )
            
            # Create upload descriptor
            descriptor = flight.FlightDescriptor.for_path(table_name)
            
            try:
                # Upload batch
                writer, _ = self.client.do_put(descriptor, record_batch.schema)
                writer.write_batch(record_batch)
                writer.close()
                
                logger.info(f"Uploaded batch {i+1}/{num_batches} ({len(batch_vectors)} vectors) to {table_name}")
            except Exception as e:
                logger.error(f"Failed to upload vectors: {str(e)}")
                raise RuntimeError(f"Failed to upload vectors: {str(e)}")
        
        return ids
    
    def similarity_search(
        self, 
        table_name: str,
        query_vector: List[float],
        k: int = 4,
        filter_query: Optional[str] = None,
        include_metadata: bool = True,
        include_vectors: bool = False,
        distance_strategy: str = "cosine"
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search using Arrow Flight.
        
        Args:
            table_name: Table to search
            query_vector: Query embedding vector
            k: Number of results to return
            filter_query: Optional SQL WHERE clause for filtering results
            include_metadata: Whether to include metadata in results
            include_vectors: Whether to include vectors in results
            distance_strategy: Distance strategy ("cosine", "l2", "dot")
            
        Returns:
            List of results with scores and optional metadata/vectors
        """
        # Construct query command
        query_data = {
            "table": table_name,
            "query_vector": query_vector,
            "k": k,
            "filter": filter_query,
            "include_metadata": include_metadata,
            "include_vectors": include_vectors,
            "distance_strategy": distance_strategy
        }
        
        # Serialize query to JSON
        import json
        command = json.dumps(query_data).encode('utf-8')
        
        # Create descriptor for search command
        descriptor = flight.FlightDescriptor.for_command(command)
        
        try:
            # Get flight info for results
            info = self.client.get_flight_info(descriptor)
            
            # Fetch results
            results = []
            for endpoint in info.endpoints:
                # Get ticket for this endpoint
                ticket = endpoint.ticket
                
                # Use the ticket to get the result data
                reader = self.client.do_get(ticket)
                
                # Process all batches from this endpoint
                while True:
                    try:
                        batch, metadata = reader.read_chunk()
                        if batch is None:
                            break
                            
                        # Convert Arrow batch to Python objects
                        vectors, texts, metadata_list, ids = self._record_batch_to_vectors(batch)
                        
                        # Extract scores (should be in the batch)
                        scores = batch.column('score').to_pylist() if 'score' in batch.schema.names else None
                        
                        # Combine results
                        for i in range(len(vectors) if include_vectors else len(ids)):
                            result = {"id": ids[i]}
                            
                            if texts is not None:
                                result["text"] = texts[i]
                                
                            if metadata_list is not None:
                                result["metadata"] = metadata_list[i]
                                
                            if scores is not None:
                                result["score"] = scores[i]
                                
                            if include_vectors:
                                result["vector"] = vectors[i]
                                
                            results.append(result)
                            
                    except StopIteration:
                        break
                        
            # Sort results by score if scores are available
            if results and "score" in results[0]:
                results.sort(key=lambda x: x["score"], reverse=(distance_strategy == "dot"))
                
            return results[:k]  # Ensure we return exactly k results
            
        except Exception as e:
            logger.error(f"Failed to perform similarity search: {str(e)}")
            raise RuntimeError(f"Failed to perform similarity search: {str(e)}")
    
    def list_flights(self) -> List[flight.FlightInfo]:
        """
        List all available flights (tables/datasets).
        
        Returns:
            List of FlightInfo objects for available flights
        """
        try:
            return list(self.client.list_flights())
        except Exception as e:
            logger.error(f"Failed to list flights: {str(e)}")
            raise RuntimeError(f"Failed to list flights: {str(e)}")
    
    def get_schema(self, descriptor: Union[str, bytes, flight.FlightDescriptor]) -> pa.Schema:
        """
        Get the schema for a specific flight.
        
        Args:
            descriptor: Flight descriptor (table name, query, or command)
            
        Returns:
            Arrow Schema object
        """
        if isinstance(descriptor, str):
            descriptor = flight.FlightDescriptor.for_path(descriptor)
        elif isinstance(descriptor, bytes):
            descriptor = flight.FlightDescriptor.for_command(descriptor)
            
        try:
            info = self.client.get_flight_info(descriptor)
            return info.schema
        except Exception as e:
            logger.error(f"Failed to get schema: {str(e)}")
            raise RuntimeError(f"Failed to get schema: {str(e)}")
    
    def close(self):
        """Close the client connections."""
        try:
            if hasattr(self, 'client') and self.client is not None:
                self.client.close()
            
            if hasattr(self, '_db_connection') and self._db_connection is not None:
                self._db_connection.close()
                
            logger.info("Closed Arrow Flight and database connections")
        except Exception as e:
            logger.warning(f"Error while closing connections: {str(e)}")