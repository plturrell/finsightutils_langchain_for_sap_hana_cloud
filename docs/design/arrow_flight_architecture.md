# Apache Arrow Flight Integration Architecture

## Overview

This document outlines the architecture for integrating Apache Arrow Flight into the SAP HANA Cloud LangChain Integration, focusing on optimizing data transfer between SAP HANA Cloud and the GPU acceleration layer. The design aims to minimize serialization/deserialization overhead and enable zero-copy transfers where possible.

## Current Architecture

The existing data transfer architecture has several bottlenecks:

1. **Custom Binary Serialization**: Vectors are serialized/deserialized using custom binary formats in `_serialize_binary_format()` and `_deserialize_binary_format()` methods, causing CPU overhead.

2. **Multiple Data Copies**: Data moves through multiple copies during the pipeline:
   - SAP HANA → Python (as binary) → Deserialized to lists → Converted to NumPy/PyTorch → GPU memory

3. **Inefficient Batch Processing**: The current batching mechanism requires full materialization of data in Python memory.

4. **Limited Parallelism**: Data transfer and GPU processing operations are not fully pipelined.

## Proposed Architecture with Apache Arrow Flight

The proposed architecture introduces Apache Arrow Flight as the data transport layer with several key components:

### 1. Arrow Flight Server for SAP HANA Cloud

```
┌───────────────────────────────────┐
│      SAP HANA Cloud Instance      │
│                                   │
│  ┌───────────────────────────┐    │
│  │     Native SQL Executor   │    │
│  └───────────┬───────────────┘    │
│              │                    │
│  ┌───────────▼───────────────┐    │
│  │  Arrow Flight Server      │    │
│  │  - Vector Query Handler   │    │
│  │  - Columnar Data Adapter  │    │
│  └───────────┬───────────────┘    │
└──────────────┼──────────────────┬─┘
               │                  │
               ▼                  ▼
     ┌─────────────────┐  ┌────────────────┐
     │ Metadata Flight │  │ Vector Flight  │
     │     Stream      │  │    Stream      │
     └─────────┬───────┘  └────────┬───────┘
               │                   │
┌──────────────┼───────────────────┼────────┐
│              │                   │        │
│  ┌───────────▼───────────────────▼─────┐  │
│  │       Arrow Flight Client           │  │
│  └───────────────────┬────────────────┘  │
│                      │                    │
│  ┌───────────────────▼────────────────┐  │
│  │      GPU-Aware Arrow Memory        │  │
│  │      - Zero-copy GPU transfer      │  │
│  └───────────────────┬────────────────┘  │
│                      │                    │
│  ┌───────────────────▼────────────────┐  │
│  │      Multi-GPU Manager              │  │
│  │      - Batch distribution           │  │
│  └────────────────────────────────────┘  │
│           GPU Acceleration Layer          │
└───────────────────────────────────────────┘
```

### 2. Core Components

#### 2.1 Arrow Flight Server Adapter for SAP HANA

This component runs inside or alongside SAP HANA Cloud and exposes an Arrow Flight interface for vector operations:

```python
class HanaFlightServer(flight.FlightServerBase):
    def __init__(self, connection_config, **kwargs):
        super().__init__(**kwargs)
        self.connection_config = connection_config
        self.connection_pool = ConnectionPool(connection_config)
    
    def get_flight_info(self, context, descriptor):
        # Handle metadata about vector data
        cmd = json.loads(descriptor.command)
        table_name = cmd.get("table_name")
        # Return flight info with schema and endpoints
        
    def do_get(self, context, ticket):
        # Retrieve vector data based on ticket
        cmd = json.loads(ticket.ticket)
        query = cmd.get("query")
        table_name = cmd.get("table_name")
        filter = cmd.get("filter")
        
        # Execute query against HANA and stream results as Arrow data
        conn = self.connection_pool.get_connection()
        try:
            # Convert HANA results directly to Arrow RecordBatch
            # Stream as Arrow Flight data
        finally:
            self.connection_pool.release_connection(conn)
```

#### 2.2 Arrow Flight Client for Vector Operations

Client component that replaces the current vector serialization/deserialization:

```python
class ArrowFlightVectorClient:
    def __init__(self, location, **kwargs):
        self.client = flight.FlightClient(location)
        # Setup authentication if needed
        
    def get_vectors(self, table_name, filter=None, columns=None):
        # Create descriptor for the request
        descriptor = flight.FlightDescriptor.for_command(
            json.dumps({
                "table_name": table_name,
                "filter": filter,
                "columns": columns
            })
        )
        
        # Get flight info and endpoints
        info = self.client.get_flight_info(descriptor)
        
        # Stream data from the endpoints
        for endpoint in info.endpoints:
            # Get data from the endpoint
            reader = self.client.do_get(endpoint.ticket)
            
            # Process record batches
            for batch in reader:
                yield batch
```

#### 2.3 GPU-Aware Arrow Memory Manager

Component that manages Arrow data and enables zero-copy transfer to GPU:

```python
class GPUArrowMemoryManager:
    def __init__(self, device_id=0):
        self.device_id = device_id
        self.device = torch.device(f"cuda:{device_id}")
        
    def record_batch_to_gpu_tensor(self, batch):
        # Get vector data from Arrow RecordBatch
        vector_array = batch.column('vector').to_numpy()
        
        # Use CUDA IPC if available for zero-copy
        if hasattr(torch.cuda, 'from_numpy'):
            # Direct transfer to GPU with zero-copy when possible
            return torch.cuda.from_numpy(vector_array).to(self.device)
        else:
            # Fallback path with memory copy
            return torch.tensor(vector_array, device=self.device)
            
    def compute_similarity(self, query_vector, document_vectors):
        # Compute similarity on GPU directly from Arrow data
        # This avoids unnecessary data copies
```

#### 2.4 Multi-GPU Arrow Distribution Manager

Component that distributes Arrow data across multiple GPUs:

```python
class MultiGPUArrowManager:
    def __init__(self, gpu_ids=None):
        self.gpu_ids = gpu_ids or range(torch.cuda.device_count())
        self.memory_managers = [
            GPUArrowMemoryManager(device_id=gpu_id) for gpu_id in self.gpu_ids
        ]
        
    def distribute_batch(self, batch):
        # Split the Arrow RecordBatch across GPUs
        # Each GPU gets a portion of the data
        # Implements load balancing based on GPU memory and capability
```

#### 2.5 Vector Format Adapters

Components to convert between Arrow format and HANA vector formats:

```python
def hana_vector_to_arrow_array(vector_binary):
    # Convert HANA REAL_VECTOR/HALF_VECTOR to Arrow Array
    # Extract dimension and values from binary format
    dim = struct.unpack_from("<I", vector_binary, 0)[0]
    values = list(struct.unpack_from(f"<{dim}f", vector_binary, 4))
    
    # Create Arrow Array from values
    return pa.array(values, type=pa.float32())
    
def arrow_array_to_hana_vector(array, vector_type="REAL_VECTOR"):
    # Convert Arrow Array to HANA vector format
    values = array.to_numpy()
    
    if vector_type == "HALF_VECTOR":
        # 2-byte half-precision float serialization
        return struct.pack(f"<I{len(values)}e", len(values), *values)
    else:
        # 4-byte float serialization (standard FVECS format)
        return struct.pack(f"<I{len(values)}f", len(values), *values)
```

### 3. Key Data Flows

#### 3.1 Similarity Search Flow

1. Client constructs an Arrow Flight request with query vector and filter criteria
2. Arrow Flight Server executes optimized query against SAP HANA
3. Results stream directly as Arrow RecordBatches to the client
4. GPU-Aware Memory Manager transfers data to GPU with minimal copying
5. GPU operations execute on the data directly

#### 3.2 Batch Document Addition Flow

1. Client batches documents and embeddings as Arrow RecordBatches
2. Arrow Flight Client streams batches to the server
3. Server writes data efficiently to SAP HANA using bulk operations
4. Status updates stream back to client

### 4. Performance Optimizations

#### 4.1 Zero-Copy Operations

The architecture enables zero-copy operations through several mechanisms:

1. **Direct GPU Transfer**: Arrow data can be transferred directly to GPU memory without intermediate CPU copies.

2. **Pinned Memory**: For operations that require CPU-GPU transfer, pinned memory is used to optimize transfer speed.

3. **Stream Processing**: Data is processed in streams rather than materializing entire result sets in memory.

#### 4.2 Batching and Pipelining

1. **Dynamic Batching**: Batch sizes adapt based on available GPU memory and processing capabilities.

2. **Operation Pipelining**: Data transfer and GPU processing are pipelined to hide latency.

3. **Asynchronous Processing**: Non-blocking operations allow concurrent execution of data transfer and computation.

#### 4.3 Vectorized Operations

1. **Columnar Processing**: Arrow's columnar format enables vectorized operations on embeddings.

2. **Predicate Pushdown**: Filter conditions are pushed down to SAP HANA for optimal execution.

3. **Projection Pushdown**: Only required columns are retrieved from the database.

## Implementation Strategy

The implementation will follow a phased approach:

### Phase 1: Core Arrow Flight Integration

1. Implement Arrow Flight Client for vector operations
2. Develop Arrow schema definitions for vector data
3. Create adapters for converting between HANA and Arrow formats
4. Implement basic vector similarity search with Arrow

### Phase 2: GPU Optimization

1. Develop GPU-aware Arrow memory management
2. Implement zero-copy data transfer where possible
3. Optimize similarity computation with Arrow data
4. Add profiling and benchmarking

### Phase 3: Multi-GPU and Advanced Features

1. Extend Arrow Flight to support multi-GPU distribution
2. Implement parallel query execution
3. Add advanced features like streaming MMR search
4. Optimize for production workloads

## Compatibility Considerations

1. **Backward Compatibility**: The implementation will maintain API compatibility with existing code.

2. **Graceful Fallback**: Systems without Arrow Flight support will fall back to the current implementation.

3. **Progressive Enhancement**: Features will be added incrementally to ensure stability.

## Performance Expectations

Based on similar integrations in other systems, we anticipate the following performance improvements:

1. **Data Transfer Speed**: 3-5x faster data transfer between SAP HANA and GPU layer
2. **Memory Usage**: 30-50% reduction in memory overhead during vector operations
3. **Query Latency**: 2-3x reduction in end-to-end query latency for large result sets
4. **Batch Processing**: 4-8x throughput improvement for large batch operations

## Next Steps

1. **Prototype Implementation**: Develop a proof-of-concept for key components
2. **Performance Benchmarking**: Establish baseline metrics for comparison
3. **Integration Planning**: Create detailed plan for integrating with existing codebase
4. **Documentation**: Prepare developer documentation for the new architecture