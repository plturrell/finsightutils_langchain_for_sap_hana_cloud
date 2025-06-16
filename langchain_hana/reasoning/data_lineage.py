"""
Data Lineage Tracking for SAP HANA Vector Knowledge System

This module provides a comprehensive data lineage tracking system for the SAP HANA
Vector Knowledge System. It tracks the flow of data from raw text through the embedding
pipeline and into the vector store, enabling transparency and accountability.

Key features:
- Track transformation history of data through the embedding pipeline
- Record input-output relationships at each stage
- Store provenance information in SAP HANA
- Visualize data lineage graphs
- Query lineage information for any vector
"""

import time
import uuid
import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple, Set

from hdbcli import dbapi

logger = logging.getLogger(__name__)


class LineageEvent:
    """
    Represents a single event in the data lineage history.
    
    An event captures a transformation or operation that was applied to data,
    including the inputs, outputs, operation details, and metadata.
    """
    
    def __init__(
        self,
        event_type: str,
        operation_name: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        metadata: Dict[str, Any],
        timestamp: Optional[float] = None,
        event_id: Optional[str] = None,
        parent_event_ids: Optional[List[str]] = None,
    ):
        """
        Initialize a lineage event.
        
        Args:
            event_type: Type of event (e.g., 'preprocessing', 'embedding', 'storage')
            operation_name: Name of the operation performed
            inputs: Input data for the operation
            outputs: Output data from the operation
            metadata: Additional metadata about the operation
            timestamp: Event timestamp (default: current time)
            event_id: Unique ID for the event (default: generated UUID)
            parent_event_ids: IDs of parent events (for tracking dependencies)
        """
        self.event_type = event_type
        self.operation_name = operation_name
        self.inputs = inputs
        self.outputs = outputs
        self.metadata = metadata
        self.timestamp = timestamp or time.time()
        self.event_id = event_id or str(uuid.uuid4())
        self.parent_event_ids = parent_event_ids or []
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the event to a dictionary.
        
        Returns:
            Dictionary representation of the event
        """
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "operation_name": self.operation_name,
            "timestamp": self.timestamp,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "metadata": self.metadata,
            "parent_event_ids": self.parent_event_ids,
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'LineageEvent':
        """
        Create a LineageEvent from a dictionary.
        
        Args:
            data: Dictionary representation of the event
            
        Returns:
            LineageEvent instance
        """
        return LineageEvent(
            event_type=data.get("event_type", ""),
            operation_name=data.get("operation_name", ""),
            inputs=data.get("inputs", {}),
            outputs=data.get("outputs", {}),
            metadata=data.get("metadata", {}),
            timestamp=data.get("timestamp"),
            event_id=data.get("event_id"),
            parent_event_ids=data.get("parent_event_ids", []),
        )


class LineageGraph:
    """
    A graph representing the lineage of data through the system.
    
    The graph consists of nodes (events) and edges (dependencies between events),
    and can be queried to understand the provenance of any piece of data.
    """
    
    def __init__(self):
        """
        Initialize a lineage graph.
        """
        self.events: Dict[str, LineageEvent] = {}
        self.edges: Dict[str, Set[str]] = {}  # event_id -> set of child event_ids
        self.reverse_edges: Dict[str, Set[str]] = {}  # event_id -> set of parent event_ids
    
    def add_event(self, event: LineageEvent) -> None:
        """
        Add an event to the lineage graph.
        
        Args:
            event: LineageEvent to add
        """
        # Add the event
        self.events[event.event_id] = event
        
        # Add edges for parent-child relationships
        for parent_id in event.parent_event_ids:
            if parent_id not in self.edges:
                self.edges[parent_id] = set()
            self.edges[parent_id].add(event.event_id)
            
            if event.event_id not in self.reverse_edges:
                self.reverse_edges[event.event_id] = set()
            self.reverse_edges[event.event_id].add(parent_id)
    
    def get_event(self, event_id: str) -> Optional[LineageEvent]:
        """
        Get an event by ID.
        
        Args:
            event_id: ID of the event to get
            
        Returns:
            LineageEvent if found, None otherwise
        """
        return self.events.get(event_id)
    
    def get_parents(self, event_id: str) -> List[LineageEvent]:
        """
        Get the parent events of an event.
        
        Args:
            event_id: ID of the event to get parents for
            
        Returns:
            List of parent events
        """
        parent_ids = self.reverse_edges.get(event_id, set())
        return [self.events[pid] for pid in parent_ids if pid in self.events]
    
    def get_children(self, event_id: str) -> List[LineageEvent]:
        """
        Get the child events of an event.
        
        Args:
            event_id: ID of the event to get children for
            
        Returns:
            List of child events
        """
        child_ids = self.edges.get(event_id, set())
        return [self.events[cid] for cid in child_ids if cid in self.events]
    
    def get_ancestry(self, event_id: str) -> List[LineageEvent]:
        """
        Get the full ancestry of an event (all ancestors).
        
        Args:
            event_id: ID of the event to get ancestry for
            
        Returns:
            List of ancestor events, from oldest to newest
        """
        ancestors = []
        visited = set()
        
        def visit_ancestors(eid: str) -> None:
            if eid in visited:
                return
            visited.add(eid)
            
            for parent in self.get_parents(eid):
                visit_ancestors(parent.event_id)
                if parent.event_id not in [a.event_id for a in ancestors]:
                    ancestors.append(parent)
        
        visit_ancestors(event_id)
        
        # Sort by timestamp
        ancestors.sort(key=lambda evt: evt.timestamp)
        return ancestors
    
    def get_descendants(self, event_id: str) -> List[LineageEvent]:
        """
        Get the full set of descendants of an event.
        
        Args:
            event_id: ID of the event to get descendants for
            
        Returns:
            List of descendant events, from oldest to newest
        """
        descendants = []
        visited = set()
        
        def visit_descendants(eid: str) -> None:
            if eid in visited:
                return
            visited.add(eid)
            
            for child in self.get_children(eid):
                if child.event_id not in [d.event_id for d in descendants]:
                    descendants.append(child)
                visit_descendants(child.event_id)
        
        visit_descendants(event_id)
        
        # Sort by timestamp
        descendants.sort(key=lambda evt: evt.timestamp)
        return descendants
    
    def find_events_by_type(self, event_type: str) -> List[LineageEvent]:
        """
        Find events by type.
        
        Args:
            event_type: Type of events to find
            
        Returns:
            List of matching events
        """
        return [evt for evt in self.events.values() if evt.event_type == event_type]
    
    def find_events_by_operation(self, operation_name: str) -> List[LineageEvent]:
        """
        Find events by operation name.
        
        Args:
            operation_name: Name of the operation to find events for
            
        Returns:
            List of matching events
        """
        return [evt for evt in self.events.values() if evt.operation_name == operation_name]
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the lineage graph to a dictionary.
        
        Returns:
            Dictionary representation of the graph
        """
        return {
            "events": {eid: evt.to_dict() for eid, evt in self.events.items()},
            "edges": {src: list(targets) for src, targets in self.edges.items()},
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'LineageGraph':
        """
        Create a LineageGraph from a dictionary.
        
        Args:
            data: Dictionary representation of the graph
            
        Returns:
            LineageGraph instance
        """
        graph = LineageGraph()
        
        # Add events
        for eid, evt_data in data.get("events", {}).items():
            graph.add_event(LineageEvent.from_dict(evt_data))
        
        # Add edges (the reverse edges will be automatically created)
        for src, targets in data.get("edges", {}).items():
            if src not in graph.edges:
                graph.edges[src] = set()
            for target in targets:
                graph.edges[src].add(target)
                
                if target not in graph.reverse_edges:
                    graph.reverse_edges[target] = set()
                graph.reverse_edges[target].add(src)
        
        return graph


class LineageTracker:
    """
    Tracks data lineage through the system.
    
    This class provides methods for recording lineage events, querying lineage
    information, and storing/retrieving lineage data from SAP HANA.
    """
    
    def __init__(
        self,
        connection: Optional[dbapi.Connection] = None,
        schema_name: Optional[str] = None,
        table_name: str = "DATA_LINEAGE",
        enable_persistence: bool = True,
    ):
        """
        Initialize a lineage tracker.
        
        Args:
            connection: SAP HANA database connection (required for persistence)
            schema_name: Database schema to use
            table_name: Name of the table to store lineage data
            enable_persistence: Whether to persist lineage data to the database
        """
        self.connection = connection
        self.schema_name = schema_name
        self.table_name = table_name
        self.enable_persistence = enable_persistence
        self.lineage_graph = LineageGraph()
        
        # Initialize lineage table if needed
        if enable_persistence and connection:
            self._initialize_lineage_table()
    
    def _initialize_lineage_table(self) -> None:
        """
        Initialize the lineage table in the database.
        """
        if not self.connection:
            logger.warning("Cannot initialize lineage table: no connection provided")
            return
        
        full_table_name = (
            f"{self.schema_name}.{self.table_name}" 
            if self.schema_name 
            else self.table_name
        )
        
        try:
            cursor = self.connection.cursor()
            
            # Create schema if it doesn't exist
            if self.schema_name:
                try:
                    cursor.execute(f"CREATE SCHEMA {self.schema_name}")
                    logger.info(f"Created schema {self.schema_name}")
                except Exception as e:
                    # Ignore error if schema already exists
                    if "exists" not in str(e).lower():
                        logger.warning(f"Error creating schema: {str(e)}")
            
            # Create lineage table if it doesn't exist
            table_sql = f"""
            CREATE TABLE IF NOT EXISTS {full_table_name} (
                EVENT_ID VARCHAR(100) PRIMARY KEY,
                EVENT_TYPE VARCHAR(100),
                OPERATION_NAME VARCHAR(256),
                TIMESTAMP TIMESTAMP,
                INPUTS NCLOB,
                OUTPUTS NCLOB,
                METADATA NCLOB,
                PARENT_EVENT_IDS NCLOB
            )
            """
            cursor.execute(table_sql)
            
            # Create index on event type and timestamp
            try:
                cursor.execute(f"CREATE INDEX IDX_{self.table_name}_TYPE_TIME ON {full_table_name}(EVENT_TYPE, TIMESTAMP)")
            except Exception as e:
                # Ignore error if index already exists
                if "exists" not in str(e).lower():
                    logger.warning(f"Error creating index: {str(e)}")
            
            # Create index on operation name
            try:
                cursor.execute(f"CREATE INDEX IDX_{self.table_name}_OPERATION ON {full_table_name}(OPERATION_NAME)")
            except Exception as e:
                # Ignore error if index already exists
                if "exists" not in str(e).lower():
                    logger.warning(f"Error creating index: {str(e)}")
                    
            logger.info(f"Initialized lineage table {full_table_name}")
        except Exception as e:
            logger.error(f"Error initializing lineage table: {str(e)}")
        finally:
            cursor.close()
    
    def record_event(
        self,
        event_type: str,
        operation_name: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        metadata: Dict[str, Any],
        parent_event_ids: Optional[List[str]] = None,
    ) -> str:
        """
        Record a lineage event.
        
        Args:
            event_type: Type of event
            operation_name: Name of the operation
            inputs: Input data
            outputs: Output data
            metadata: Additional metadata
            parent_event_ids: IDs of parent events
            
        Returns:
            ID of the recorded event
        """
        # Create the event
        event = LineageEvent(
            event_type=event_type,
            operation_name=operation_name,
            inputs=inputs,
            outputs=outputs,
            metadata=metadata,
            parent_event_ids=parent_event_ids or [],
        )
        
        # Add to in-memory graph
        self.lineage_graph.add_event(event)
        
        # Persist to database if enabled
        if self.enable_persistence and self.connection:
            self._persist_event(event)
        
        return event.event_id
    
    def _persist_event(self, event: LineageEvent) -> None:
        """
        Persist an event to the database.
        
        Args:
            event: LineageEvent to persist
        """
        if not self.connection:
            return
        
        full_table_name = (
            f"{self.schema_name}.{self.table_name}" 
            if self.schema_name 
            else self.table_name
        )
        
        try:
            cursor = self.connection.cursor()
            
            # Serialize JSON fields
            inputs_json = json.dumps(event.inputs)
            outputs_json = json.dumps(event.outputs)
            metadata_json = json.dumps(event.metadata)
            parent_ids_json = json.dumps(event.parent_event_ids)
            
            # Insert the event
            insert_sql = f"""
            INSERT INTO {full_table_name} (
                EVENT_ID, EVENT_TYPE, OPERATION_NAME, TIMESTAMP,
                INPUTS, OUTPUTS, METADATA, PARENT_EVENT_IDS
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            cursor.execute(
                insert_sql,
                (
                    event.event_id,
                    event.event_type,
                    event.operation_name,
                    event.timestamp,
                    inputs_json,
                    outputs_json,
                    metadata_json,
                    parent_ids_json,
                )
            )
            
            # Commit the transaction
            self.connection.commit()
        except Exception as e:
            logger.error(f"Error persisting lineage event: {str(e)}")
            try:
                self.connection.rollback()
            except:
                pass
        finally:
            cursor.close()
    
    def load_events_from_db(self, limit: int = 1000) -> None:
        """
        Load lineage events from the database into the in-memory graph.
        
        Args:
            limit: Maximum number of events to load
        """
        if not self.connection:
            logger.warning("Cannot load events: no connection provided")
            return
        
        full_table_name = (
            f"{self.schema_name}.{self.table_name}" 
            if self.schema_name 
            else self.table_name
        )
        
        try:
            cursor = self.connection.cursor()
            
            # Query events ordered by timestamp
            query_sql = f"""
            SELECT EVENT_ID, EVENT_TYPE, OPERATION_NAME, TIMESTAMP,
                   INPUTS, OUTPUTS, METADATA, PARENT_EVENT_IDS
            FROM {full_table_name}
            ORDER BY TIMESTAMP
            LIMIT {limit}
            """
            
            cursor.execute(query_sql)
            
            # Process results
            count = 0
            for row in cursor.fetchall():
                event_id, event_type, operation_name, timestamp = row[0:4]
                inputs_json, outputs_json, metadata_json, parent_ids_json = row[4:8]
                
                # Parse JSON fields
                try:
                    inputs = json.loads(inputs_json)
                    outputs = json.loads(outputs_json)
                    metadata = json.loads(metadata_json)
                    parent_ids = json.loads(parent_ids_json)
                except:
                    # If JSON parsing fails, use empty dictionaries/lists
                    inputs, outputs, metadata = {}, {}, {}
                    parent_ids = []
                
                # Create and add the event
                event = LineageEvent(
                    event_id=event_id,
                    event_type=event_type,
                    operation_name=operation_name,
                    timestamp=timestamp,
                    inputs=inputs,
                    outputs=outputs,
                    metadata=metadata,
                    parent_event_ids=parent_ids,
                )
                
                self.lineage_graph.add_event(event)
                count += 1
            
            logger.info(f"Loaded {count} lineage events from the database")
        except Exception as e:
            logger.error(f"Error loading lineage events: {str(e)}")
        finally:
            cursor.close()
    
    def get_event(self, event_id: str) -> Optional[LineageEvent]:
        """
        Get an event by ID.
        
        Args:
            event_id: ID of the event to get
            
        Returns:
            LineageEvent if found, None otherwise
        """
        return self.lineage_graph.get_event(event_id)
    
    def get_ancestry(self, event_id: str) -> List[LineageEvent]:
        """
        Get the ancestry of an event.
        
        Args:
            event_id: ID of the event to get ancestry for
            
        Returns:
            List of ancestor events
        """
        return self.lineage_graph.get_ancestry(event_id)
    
    def get_descendants(self, event_id: str) -> List[LineageEvent]:
        """
        Get the descendants of an event.
        
        Args:
            event_id: ID of the event to get descendants for
            
        Returns:
            List of descendant events
        """
        return self.lineage_graph.get_descendants(event_id)
    
    def find_events_by_type(self, event_type: str) -> List[LineageEvent]:
        """
        Find events by type.
        
        Args:
            event_type: Type of events to find
            
        Returns:
            List of matching events
        """
        return self.lineage_graph.find_events_by_type(event_type)
    
    def find_events_by_operation(self, operation_name: str) -> List[LineageEvent]:
        """
        Find events by operation name.
        
        Args:
            operation_name: Name of the operation to find events for
            
        Returns:
            List of matching events
        """
        return self.lineage_graph.find_events_by_operation(operation_name)
    
    def export_graph(self) -> Dict[str, Any]:
        """
        Export the lineage graph as a dictionary.
        
        Returns:
            Dictionary representation of the graph
        """
        return self.lineage_graph.to_dict()
    
    def import_graph(self, data: Dict[str, Any]) -> None:
        """
        Import a lineage graph from a dictionary.
        
        Args:
            data: Dictionary representation of the graph
        """
        self.lineage_graph = LineageGraph.from_dict(data)
    
    def record_embedding_pipeline_event(
        self,
        original_texts: List[str],
        embedding_vectors: List[List[float]],
        pipeline_metadata: Dict[str, Any],
        parent_event_ids: Optional[List[str]] = None,
    ) -> str:
        """
        Record a lineage event for an embedding pipeline operation.
        
        Args:
            original_texts: Original texts that were embedded
            embedding_vectors: Final embedding vectors
            pipeline_metadata: Metadata about the pipeline operation
            parent_event_ids: IDs of parent events
            
        Returns:
            ID of the recorded event
        """
        # Record the event
        return self.record_event(
            event_type="embedding_pipeline",
            operation_name="generate_embeddings",
            inputs={"texts": [text[:100] + "..." if len(text) > 100 else text for text in original_texts]},
            outputs={"vector_count": len(embedding_vectors), "vector_dimension": len(embedding_vectors[0]) if embedding_vectors else 0},
            metadata=pipeline_metadata,
            parent_event_ids=parent_event_ids,
        )
    
    def record_vector_storage_event(
        self,
        table_name: str,
        vector_ids: List[str],
        embedding_event_id: str,
        storage_metadata: Dict[str, Any],
    ) -> str:
        """
        Record a lineage event for a vector storage operation.
        
        Args:
            table_name: Name of the table where vectors were stored
            vector_ids: IDs of the stored vectors
            embedding_event_id: ID of the embedding event that generated the vectors
            storage_metadata: Metadata about the storage operation
            
        Returns:
            ID of the recorded event
        """
        # Record the event
        return self.record_event(
            event_type="vector_storage",
            operation_name="store_vectors",
            inputs={"embedding_event_id": embedding_event_id},
            outputs={"table_name": table_name, "vector_ids": vector_ids},
            metadata=storage_metadata,
            parent_event_ids=[embedding_event_id],
        )
    
    def record_query_event(
        self,
        query_text: str,
        query_embedding: List[float],
        query_metadata: Dict[str, Any],
    ) -> str:
        """
        Record a lineage event for a query operation.
        
        Args:
            query_text: Text of the query
            query_embedding: Embedding vector for the query
            query_metadata: Metadata about the query operation
            
        Returns:
            ID of the recorded event
        """
        # Record the event
        return self.record_event(
            event_type="query",
            operation_name="vector_search",
            inputs={"query_text": query_text},
            outputs={"vector_dimension": len(query_embedding)},
            metadata=query_metadata,
            parent_event_ids=[],
        )
    
    def record_result_event(
        self,
        query_event_id: str,
        result_vectors: List[List[float]],
        result_metadata: Dict[str, Any],
    ) -> str:
        """
        Record a lineage event for a query result.
        
        Args:
            query_event_id: ID of the query event
            result_vectors: Embedding vectors for the results
            result_metadata: Metadata about the results
            
        Returns:
            ID of the recorded event
        """
        # Record the event
        return self.record_event(
            event_type="query_result",
            operation_name="search_results",
            inputs={"query_event_id": query_event_id},
            outputs={"result_count": len(result_vectors)},
            metadata=result_metadata,
            parent_event_ids=[query_event_id],
        )