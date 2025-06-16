"""
Multi-GPU integration for Apache Arrow Flight in SAP HANA Cloud.

This module extends the Arrow Flight integration to work with multiple GPUs,
enabling distributed processing and load balancing for improved performance.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import numpy as np

try:
    import pyarrow as pa
    import pyarrow.flight as flight
    HAS_ARROW_FLIGHT = True
except ImportError:
    HAS_ARROW_FLIGHT = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .arrow_flight_client import ArrowFlightClient
from .arrow_gpu_memory_manager import ArrowGpuMemoryManager
from .multi_gpu_manager import EnhancedMultiGPUManager
from .vector_serialization import (
    vectors_to_arrow_batch,
    arrow_batch_to_vectors,
    serialize_arrow_batch,
    deserialize_arrow_batch
)

logger = logging.getLogger(__name__)


class ArrowFlightMultiGPUManager:
    """
    Multi-GPU manager for Apache Arrow Flight integration.
    
    This class enables distributed processing of vector operations across multiple GPUs,
    leveraging Arrow Flight for efficient data transfer and the multi-GPU system for
    load balancing and parallelization.
    """
    
    def __init__(
        self,
        flight_clients: Optional[List[ArrowFlightClient]] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        use_tls: bool = False,
        username: Optional[str] = None,
        password: Optional[str] = None,
        connection_args: Optional[Dict[str, Any]] = None,
        gpu_ids: Optional[List[int]] = None,
        batch_size: int = 1000,
        memory_fraction: float = 0.8,
        distribution_strategy: str = "round_robin",
    ):
        """
        Initialize the Arrow Flight Multi-GPU Manager.
        
        Args:
            flight_clients: Optional list of pre-configured Arrow Flight clients
            host: SAP HANA host address (if clients not provided)
            port: Arrow Flight server port (if clients not provided)
            use_tls: Whether to use TLS for secure connections
            username: SAP HANA username
            password: SAP HANA password
            connection_args: Additional connection arguments
            gpu_ids: List of GPU device IDs to use
            batch_size: Default batch size for operations
            memory_fraction: Maximum fraction of GPU memory to use
            distribution_strategy: Strategy for distributing work ("round_robin", "memory_based", "model_based")
            
        Raises:
            ImportError: If required dependencies are not installed
            RuntimeError: If no GPUs are available
        """
        if not HAS_ARROW_FLIGHT:
            raise ImportError(
                "The pyarrow and pyarrow.flight packages are required for Arrow Flight integration. "
                "Install them with 'pip install pyarrow pyarrow.flight'."
            )
        
        if not HAS_TORCH:
            raise ImportError(
                "PyTorch is required for GPU operations. "
                "Install it with 'pip install torch'."
            )
        
        # Initialize Multi-GPU Manager
        self.multi_gpu_manager = EnhancedMultiGPUManager(
            gpu_ids=gpu_ids,
            distribution_strategy=distribution_strategy
        )
        
        if not self.multi_gpu_manager.gpus:
            raise RuntimeError("No GPUs available for multi-GPU operations")
        
        self.gpu_ids = self.multi_gpu_manager.gpu_ids
        self.num_gpus = len(self.gpu_ids)
        
        # Initialize ArrowFlightClient instances for each GPU
        if flight_clients is not None and len(flight_clients) > 0:
            # Use provided clients
            self.flight_clients = flight_clients
            if len(self.flight_clients) != self.num_gpus:
                logger.warning(
                    f"Number of provided flight clients ({len(self.flight_clients)}) "
                    f"does not match number of GPUs ({self.num_gpus})"
                )
        else:
            # Create new clients
            if host is None:
                raise ValueError("Host must be provided if flight_clients is not provided")
                
            self.flight_clients = [
                ArrowFlightClient(
                    host=host,
                    port=port,
                    use_tls=use_tls,
                    username=username,
                    password=password,
                    connection_args=connection_args
                )
                for _ in range(self.num_gpus)
            ]
        
        # Initialize GPU memory managers for each GPU
        self.memory_managers = [
            ArrowGpuMemoryManager(
                device_id=gpu_id,
                max_memory_fraction=memory_fraction,
                batch_size=batch_size
            )
            for gpu_id in self.gpu_ids
        ]
        
        # Set default batch size
        self.batch_size = batch_size
        
        # Initialize thread pool for parallel operations
        self.executor = ThreadPoolExecutor(max_workers=self.num_gpus)
        
        logger.info(f"Initialized Arrow Flight Multi-GPU Manager with {self.num_gpus} GPUs: {self.gpu_ids}")
    
    def get_optimal_batch_sizes(self) -> List[int]:
        """
        Get optimal batch sizes for each GPU based on available memory.
        
        Returns:
            List of optimal batch sizes for each GPU
        """
        return [
            memory_manager.get_optimal_batch_size()
            for memory_manager in self.memory_managers
        ]
    
    def distribute_batch(
        self,
        vectors: List[List[float]],
        texts: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Distribute a batch of vectors across multiple GPUs.
        
        Args:
            vectors: List of vectors to distribute
            texts: Optional list of text strings
            metadata: Optional list of metadata dictionaries
            ids: Optional list of IDs
            
        Returns:
            List of dictionaries containing the distributed batches
        """
        if not vectors:
            return []
        
        # Get number of vectors
        num_vectors = len(vectors)
        
        # Get optimal batch size for each GPU
        batch_sizes = self.get_optimal_batch_sizes()
        
        # Calculate distribution based on strategy
        gpu_assignments = self.multi_gpu_manager.get_work_distribution(num_vectors, batch_sizes)
        
        # Prepare distributed batches
        distributed_batches = []
        start_idx = 0
        
        for gpu_idx, num_items in enumerate(gpu_assignments):
            if num_items == 0:
                continue
                
            # Calculate end index for this batch
            end_idx = start_idx + num_items
            
            # Extract batch data
            batch_vectors = vectors[start_idx:end_idx]
            
            batch_data = {
                "gpu_idx": gpu_idx,
                "gpu_id": self.gpu_ids[gpu_idx],
                "vectors": batch_vectors,
                "start_idx": start_idx,
                "end_idx": end_idx,
            }
            
            # Add optional data if provided
            if texts is not None:
                batch_data["texts"] = texts[start_idx:end_idx]
                
            if metadata is not None:
                batch_data["metadata"] = metadata[start_idx:end_idx]
                
            if ids is not None:
                batch_data["ids"] = ids[start_idx:end_idx]
            
            distributed_batches.append(batch_data)
            start_idx = end_idx
        
        return distributed_batches
    
    def upload_vectors_multi_gpu(
        self,
        table_name: str,
        vectors: List[List[float]],
        texts: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Upload vectors to SAP HANA using multiple GPUs for processing.
        
        Args:
            table_name: Target table name
            vectors: List of vectors to upload
            texts: Optional list of text strings
            metadata: Optional list of metadata dictionaries
            ids: Optional list of IDs
            
        Returns:
            List of generated or provided IDs
        """
        if not vectors:
            return []
        
        # Distribute vectors across GPUs
        distributed_batches = self.distribute_batch(vectors, texts, metadata, ids)
        
        if not distributed_batches:
            logger.warning("No batches to process after distribution")
            return []
        
        # Process batches in parallel
        futures = []
        for batch in distributed_batches:
            gpu_idx = batch["gpu_idx"]
            client = self.flight_clients[gpu_idx]
            
            # Create a function for processing this batch
            def process_batch(batch_data):
                batch_vectors = batch_data["vectors"]
                batch_texts = batch_data.get("texts")
                batch_metadata = batch_data.get("metadata")
                batch_ids = batch_data.get("ids")
                
                # Upload batch
                return client.upload_vectors(
                    table_name=table_name,
                    vectors=batch_vectors,
                    texts=batch_texts,
                    metadata=batch_metadata,
                    ids=batch_ids,
                    batch_size=self.batch_size
                )
            
            # Submit to thread pool
            future = self.executor.submit(process_batch, batch)
            futures.append((future, batch["start_idx"], batch["end_idx"]))
        
        # Collect results
        all_ids = [None] * len(vectors)
        for future, start_idx, end_idx in futures:
            try:
                batch_ids = future.result()
                # Place IDs in the correct positions
                for i, idx in enumerate(range(start_idx, end_idx)):
                    if idx < len(all_ids):
                        all_ids[idx] = batch_ids[i]
            except Exception as e:
                logger.error(f"Error processing batch: {str(e)}")
        
        return [id_val for id_val in all_ids if id_val is not None]
    
    def similarity_search_multi_gpu(
        self,
        table_name: str,
        query_vectors: List[List[float]],
        k: int = 4,
        filter_query: Optional[str] = None,
        include_metadata: bool = True,
        include_vectors: bool = False,
        distance_strategy: str = "cosine",
    ) -> List[List[Dict[str, Any]]]:
        """
        Perform similarity search using multiple GPUs for processing.
        
        Args:
            table_name: Table to search
            query_vectors: List of query vectors
            k: Number of results to return for each query
            filter_query: Optional SQL WHERE clause for filtering results
            include_metadata: Whether to include metadata in results
            include_vectors: Whether to include vectors in results
            distance_strategy: Distance strategy ("cosine", "l2", "dot")
            
        Returns:
            List of results for each query vector
        """
        if not query_vectors:
            return []
        
        # Distribute query vectors across GPUs
        distributed_batches = self.distribute_batch(query_vectors)
        
        if not distributed_batches:
            logger.warning("No batches to process after distribution")
            return []
        
        # Process batches in parallel
        futures = []
        for batch in distributed_batches:
            gpu_idx = batch["gpu_idx"]
            client = self.flight_clients[gpu_idx]
            
            # Create a function for processing this batch
            def process_batch(batch_data):
                batch_vectors = batch_data["vectors"]
                results = []
                
                # Process each query vector
                for query_vector in batch_vectors:
                    # Perform similarity search
                    query_results = client.similarity_search(
                        table_name=table_name,
                        query_vector=query_vector,
                        k=k,
                        filter_query=filter_query,
                        include_metadata=include_metadata,
                        include_vectors=include_vectors,
                        distance_strategy=distance_strategy
                    )
                    results.append(query_results)
                
                return results
            
            # Submit to thread pool
            future = self.executor.submit(process_batch, batch)
            futures.append((future, batch["start_idx"], batch["end_idx"]))
        
        # Collect results
        all_results = [None] * len(query_vectors)
        for future, start_idx, end_idx in futures:
            try:
                batch_results = future.result()
                # Place results in the correct positions
                for i, idx in enumerate(range(start_idx, end_idx)):
                    if idx < len(all_results):
                        all_results[idx] = batch_results[i]
            except Exception as e:
                logger.error(f"Error processing batch: {str(e)}")
        
        # Filter out None results
        return [results for results in all_results if results is not None]
    
    def batch_similarity_search_multi_gpu(
        self,
        query_vectors: List[List[float]],
        stored_vectors: List[List[float]],
        k: int = 4,
        metric: str = "cosine",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform batch similarity search using multiple GPUs.
        
        This method distributes the query vectors across multiple GPUs and
        performs similarity search against the same set of stored vectors.
        
        Args:
            query_vectors: List of query vectors
            stored_vectors: List of stored vectors to search against
            k: Number of results to return for each query
            metric: Distance metric ("cosine", "l2", "dot")
            
        Returns:
            Tuple of (distances, indices) as numpy arrays
        """
        if not query_vectors or not stored_vectors:
            return np.array([]), np.array([])
        
        # Convert to numpy arrays for easier manipulation
        query_np = np.array(query_vectors, dtype=np.float32)
        stored_np = np.array(stored_vectors, dtype=np.float32)
        
        # Distribute query vectors across GPUs
        num_queries = len(query_vectors)
        batch_sizes = self.get_optimal_batch_sizes()
        gpu_assignments = self.multi_gpu_manager.get_work_distribution(num_queries, batch_sizes)
        
        # Process batches in parallel
        futures = []
        start_idx = 0
        
        for gpu_idx, num_items in enumerate(gpu_assignments):
            if num_items == 0:
                continue
                
            # Calculate end index for this batch
            end_idx = start_idx + num_items
            
            # Extract batch data
            batch_queries = query_np[start_idx:end_idx]
            
            # Create a function for processing this batch
            def process_batch(gpu_idx, batch_queries, start_idx, end_idx):
                memory_manager = self.memory_managers[gpu_idx]
                device_id = self.gpu_ids[gpu_idx]
                
                # Convert to Arrow arrays
                query_array = memory_manager.vectors_to_fixed_size_list_array(batch_queries)
                stored_array = memory_manager.vectors_to_fixed_size_list_array(stored_np)
                
                # Perform batch similarity search
                distances, indices = memory_manager.batch_similarity_search(
                    query_vectors=query_array,
                    stored_vectors=stored_array,
                    k=min(k, len(stored_vectors)),
                    metric=metric
                )
                
                # Convert to numpy arrays and return
                return (
                    distances.cpu().numpy() if hasattr(distances, 'cpu') else distances,
                    indices.cpu().numpy() if hasattr(indices, 'cpu') else indices,
                    start_idx,
                    end_idx
                )
            
            # Submit to thread pool
            future = self.executor.submit(
                process_batch, gpu_idx, batch_queries, start_idx, end_idx
            )
            futures.append(future)
            start_idx = end_idx
        
        # Prepare result arrays
        all_distances = np.zeros((num_queries, k), dtype=np.float32)
        all_indices = np.zeros((num_queries, k), dtype=np.int64)
        
        # Collect results
        for future in futures:
            try:
                distances, indices, start_idx, end_idx = future.result()
                # Place results in the correct positions
                all_distances[start_idx:end_idx] = distances
                all_indices[start_idx:end_idx] = indices
            except Exception as e:
                logger.error(f"Error in batch similarity search: {str(e)}")
        
        return all_distances, all_indices
    
    def close(self):
        """Close all connections and resources."""
        # Close Arrow Flight clients
        for client in self.flight_clients:
            try:
                client.close()
            except Exception as e:
                logger.warning(f"Error closing Arrow Flight client: {str(e)}")
        
        # Clean up memory managers
        for memory_manager in self.memory_managers:
            try:
                memory_manager.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up memory manager: {str(e)}")
        
        # Shutdown thread pool
        self.executor.shutdown()
        
        logger.info("Closed all Arrow Flight Multi-GPU Manager resources")