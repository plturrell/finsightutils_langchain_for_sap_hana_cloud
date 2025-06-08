"""Data lineage tracking for embedding vectors in SAP HANA Cloud.

This module provides utilities for tracking the lineage of embedding vectors, including:
- Source document provenance
- Embedding model metadata
- Transformation history
- Usage tracking
- Versioning information

The lineage tracking system is designed to be integrated with the HanaDB vectorstore
to provide comprehensive auditing and compliance capabilities.
"""

import json
import uuid
import logging
import datetime
from typing import Dict, List, Optional, Any, Union

logger = logging.getLogger(__name__)

class LineageRecord:
    """
    A record containing lineage information for a document and its embedding vector.
    
    This class stores detailed provenance information about a document and its embedding,
    enabling compliance reporting, audit trails, and data governance.
    
    Attributes:
        id (str): Unique identifier for the lineage record
        document_id (str): Identifier for the document (if available)
        document_hash (str): Hash of the document content for integrity verification
        embedding_model (str): Name or identifier of the embedding model used
        embedding_model_version (str): Version of the embedding model
        embedding_timestamp (str): When the embedding was generated
        embedding_dimension (int): Dimension of the embedding vector
        source_system (str): System or application that originated the document
        source_location (str): Location of the original document (URL, file path, etc.)
        transformation_history (List[Dict]): Record of transformations applied
        access_history (List[Dict]): Record of access events
        metadata (Dict): Additional metadata about the document and embedding
    """
    
    def __init__(
        self,
        document_content: str,
        embedding_model: str,
        embedding_dimension: int,
        document_id: Optional[str] = None,
        source_system: Optional[str] = None,
        source_location: Optional[str] = None,
        embedding_model_version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a new lineage record.
        
        Args:
            document_content: The text content of the document
            embedding_model: Name or identifier of the embedding model
            embedding_dimension: Dimension of the embedding vector
            document_id: Optional identifier for the document
            source_system: Optional system or application that originated the document
            source_location: Optional location of the original document
            embedding_model_version: Optional version of the embedding model
            metadata: Optional additional metadata
        """
        # Generate a unique ID for this lineage record
        self.id = str(uuid.uuid4())
        
        # Document information
        self.document_id = document_id or self.id
        self.document_hash = self._hash_content(document_content)
        
        # Embedding information
        self.embedding_model = embedding_model
        self.embedding_model_version = embedding_model_version
        self.embedding_timestamp = datetime.datetime.now().isoformat()
        self.embedding_dimension = embedding_dimension
        
        # Source information
        self.source_system = source_system or "langchain_hana"
        self.source_location = source_location
        
        # History tracking
        self.transformation_history = []
        self.access_history = []
        
        # Additional metadata
        self.metadata = metadata or {}
        
        # Initial creation event
        self._record_creation_event()
    
    def _hash_content(self, content: str) -> str:
        """
        Generate a hash of the document content for integrity verification.
        
        Args:
            content: The document content to hash
            
        Returns:
            A string hash of the content
        """
        import hashlib
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def _record_creation_event(self) -> None:
        """Record the initial creation of this lineage record."""
        creation_event = {
            "event_type": "creation",
            "timestamp": self.embedding_timestamp,
            "details": {
                "embedding_model": self.embedding_model,
                "embedding_dimension": self.embedding_dimension,
                "source_system": self.source_system
            }
        }
        self.transformation_history.append(creation_event)
    
    def record_transformation(self, 
                             transformation_type: str, 
                             details: Dict[str, Any]) -> None:
        """
        Record a transformation applied to the document or embedding.
        
        Args:
            transformation_type: Type of transformation (e.g., "normalization", "filtering")
            details: Details about the transformation
        """
        transformation_event = {
            "event_type": transformation_type,
            "timestamp": datetime.datetime.now().isoformat(),
            "details": details
        }
        self.transformation_history.append(transformation_event)
    
    def record_access(self, 
                     access_type: str, 
                     user_id: Optional[str] = None,
                     application: Optional[str] = None,
                     details: Optional[Dict[str, Any]] = None) -> None:
        """
        Record an access event for the document or embedding.
        
        Args:
            access_type: Type of access (e.g., "search", "retrieval", "update")
            user_id: Optional ID of the user accessing the data
            application: Optional name of the application accessing the data
            details: Optional additional details about the access
        """
        access_event = {
            "event_type": access_type,
            "timestamp": datetime.datetime.now().isoformat(),
            "user_id": user_id,
            "application": application,
            "details": details or {}
        }
        self.access_history.append(access_event)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the lineage record to a dictionary.
        
        Returns:
            A dictionary representation of the lineage record
        """
        return {
            "id": self.id,
            "document_id": self.document_id,
            "document_hash": self.document_hash,
            "embedding_model": self.embedding_model,
            "embedding_model_version": self.embedding_model_version,
            "embedding_timestamp": self.embedding_timestamp,
            "embedding_dimension": self.embedding_dimension,
            "source_system": self.source_system,
            "source_location": self.source_location,
            "transformation_history": self.transformation_history,
            "access_history": self.access_history,
            "metadata": self.metadata
        }
    
    def to_json(self) -> str:
        """
        Convert the lineage record to a JSON string.
        
        Returns:
            A JSON string representation of the lineage record
        """
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LineageRecord':
        """
        Create a lineage record from a dictionary.
        
        Args:
            data: Dictionary containing lineage record data
            
        Returns:
            A LineageRecord instance
        """
        # Create a minimal instance
        instance = cls(
            document_content="", # This is just a placeholder
            embedding_model=data["embedding_model"],
            embedding_dimension=data["embedding_dimension"]
        )
        
        # Update with all the data
        instance.id = data["id"]
        instance.document_id = data["document_id"]
        instance.document_hash = data["document_hash"]
        instance.embedding_model_version = data.get("embedding_model_version")
        instance.embedding_timestamp = data["embedding_timestamp"]
        instance.source_system = data["source_system"]
        instance.source_location = data.get("source_location")
        instance.transformation_history = data["transformation_history"]
        instance.access_history = data["access_history"]
        instance.metadata = data["metadata"]
        
        return instance
    
    @classmethod
    def from_json(cls, json_str: str) -> 'LineageRecord':
        """
        Create a lineage record from a JSON string.
        
        Args:
            json_str: JSON string containing lineage record data
            
        Returns:
            A LineageRecord instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)


class LineageManager:
    """
    Manager for tracking and storing lineage records.
    
    This class provides methods for creating, updating, and retrieving lineage records,
    as well as for querying the lineage database.
    """
    
    def __init__(self, connection):
        """
        Initialize the lineage manager.
        
        Args:
            connection: Database connection for storing lineage records
        """
        self.connection = connection
        self._initialize_tables()
    
    def _initialize_tables(self) -> None:
        """Initialize the necessary database tables for lineage tracking."""
        try:
            cursor = self.connection.cursor()
            
            # Create table for lineage records
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS "EMBEDDING_LINEAGE" (
                    "LINEAGE_ID" VARCHAR(255) PRIMARY KEY,
                    "DOCUMENT_ID" VARCHAR(255),
                    "DOCUMENT_HASH" VARCHAR(64),
                    "EMBEDDING_MODEL" NVARCHAR(255),
                    "EMBEDDING_MODEL_VERSION" NVARCHAR(255),
                    "EMBEDDING_TIMESTAMP" TIMESTAMP,
                    "EMBEDDING_DIMENSION" INTEGER,
                    "SOURCE_SYSTEM" NVARCHAR(255),
                    "SOURCE_LOCATION" NVARCHAR(1000),
                    "TRANSFORMATION_HISTORY" NCLOB,
                    "ACCESS_HISTORY" NCLOB,
                    "METADATA" NCLOB
                )
            """)
            
            # Create index on document_id for faster lookups
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS "IDX_LINEAGE_DOC_ID" ON "EMBEDDING_LINEAGE" ("DOCUMENT_ID")
            """)
            
            cursor.close()
            logger.info("Initialized lineage tables")
        except Exception as e:
            logger.error(f"Error initializing lineage tables: {str(e)}")
            raise
    
    def create_lineage_record(self, 
                             document_content: str,
                             embedding_model: str,
                             embedding_dimension: int,
                             **kwargs) -> LineageRecord:
        """
        Create a new lineage record.
        
        Args:
            document_content: The text content of the document
            embedding_model: Name or identifier of the embedding model
            embedding_dimension: Dimension of the embedding vector
            **kwargs: Additional arguments to pass to LineageRecord constructor
            
        Returns:
            A new LineageRecord instance
        """
        record = LineageRecord(
            document_content=document_content,
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension,
            **kwargs
        )
        self.save_lineage_record(record)
        return record
    
    def save_lineage_record(self, record: LineageRecord) -> None:
        """
        Save a lineage record to the database.
        
        Args:
            record: The LineageRecord to save
        """
        try:
            cursor = self.connection.cursor()
            
            # Convert history and metadata to JSON strings
            transformation_history_json = json.dumps(record.transformation_history)
            access_history_json = json.dumps(record.access_history)
            metadata_json = json.dumps(record.metadata)
            
            # Insert or update the record
            cursor.execute("""
                MERGE INTO "EMBEDDING_LINEAGE" AS target
                USING (SELECT ? AS "LINEAGE_ID") AS source
                ON target."LINEAGE_ID" = source."LINEAGE_ID"
                WHEN MATCHED THEN
                    UPDATE SET
                        "DOCUMENT_ID" = ?,
                        "DOCUMENT_HASH" = ?,
                        "EMBEDDING_MODEL" = ?,
                        "EMBEDDING_MODEL_VERSION" = ?,
                        "EMBEDDING_TIMESTAMP" = ?,
                        "EMBEDDING_DIMENSION" = ?,
                        "SOURCE_SYSTEM" = ?,
                        "SOURCE_LOCATION" = ?,
                        "TRANSFORMATION_HISTORY" = ?,
                        "ACCESS_HISTORY" = ?,
                        "METADATA" = ?
                WHEN NOT MATCHED THEN
                    INSERT (
                        "LINEAGE_ID",
                        "DOCUMENT_ID",
                        "DOCUMENT_HASH",
                        "EMBEDDING_MODEL",
                        "EMBEDDING_MODEL_VERSION",
                        "EMBEDDING_TIMESTAMP",
                        "EMBEDDING_DIMENSION",
                        "SOURCE_SYSTEM",
                        "SOURCE_LOCATION",
                        "TRANSFORMATION_HISTORY",
                        "ACCESS_HISTORY",
                        "METADATA"
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.id,
                record.document_id,
                record.document_hash,
                record.embedding_model,
                record.embedding_model_version,
                record.embedding_timestamp,
                record.embedding_dimension,
                record.source_system,
                record.source_location,
                transformation_history_json,
                access_history_json,
                metadata_json,
                # Values for INSERT
                record.id,
                record.document_id,
                record.document_hash,
                record.embedding_model,
                record.embedding_model_version,
                record.embedding_timestamp,
                record.embedding_dimension,
                record.source_system,
                record.source_location,
                transformation_history_json,
                access_history_json,
                metadata_json
            ))
            
            cursor.close()
        except Exception as e:
            logger.error(f"Error saving lineage record: {str(e)}")
            raise
    
    def get_lineage_record(self, lineage_id: str) -> Optional[LineageRecord]:
        """
        Retrieve a lineage record by ID.
        
        Args:
            lineage_id: The ID of the lineage record to retrieve
            
        Returns:
            The LineageRecord if found, None otherwise
        """
        try:
            cursor = self.connection.cursor()
            
            cursor.execute("""
                SELECT
                    "LINEAGE_ID",
                    "DOCUMENT_ID",
                    "DOCUMENT_HASH",
                    "EMBEDDING_MODEL",
                    "EMBEDDING_MODEL_VERSION",
                    "EMBEDDING_TIMESTAMP",
                    "EMBEDDING_DIMENSION",
                    "SOURCE_SYSTEM",
                    "SOURCE_LOCATION",
                    "TRANSFORMATION_HISTORY",
                    "ACCESS_HISTORY",
                    "METADATA"
                FROM "EMBEDDING_LINEAGE"
                WHERE "LINEAGE_ID" = ?
            """, (lineage_id,))
            
            row = cursor.fetchone()
            cursor.close()
            
            if row:
                # Convert JSON strings back to objects
                transformation_history = json.loads(row[9])
                access_history = json.loads(row[10])
                metadata = json.loads(row[11])
                
                # Create and return a LineageRecord instance
                data = {
                    "id": row[0],
                    "document_id": row[1],
                    "document_hash": row[2],
                    "embedding_model": row[3],
                    "embedding_model_version": row[4],
                    "embedding_timestamp": row[5].isoformat() if row[5] else None,
                    "embedding_dimension": row[6],
                    "source_system": row[7],
                    "source_location": row[8],
                    "transformation_history": transformation_history,
                    "access_history": access_history,
                    "metadata": metadata
                }
                
                return LineageRecord.from_dict(data)
            
            return None
        except Exception as e:
            logger.error(f"Error retrieving lineage record: {str(e)}")
            raise
    
    def find_lineage_records_by_document_id(self, document_id: str) -> List[LineageRecord]:
        """
        Find all lineage records for a specific document.
        
        Args:
            document_id: The document ID to search for
            
        Returns:
            A list of matching LineageRecord instances
        """
        try:
            cursor = self.connection.cursor()
            
            cursor.execute("""
                SELECT
                    "LINEAGE_ID",
                    "DOCUMENT_ID",
                    "DOCUMENT_HASH",
                    "EMBEDDING_MODEL",
                    "EMBEDDING_MODEL_VERSION",
                    "EMBEDDING_TIMESTAMP",
                    "EMBEDDING_DIMENSION",
                    "SOURCE_SYSTEM",
                    "SOURCE_LOCATION",
                    "TRANSFORMATION_HISTORY",
                    "ACCESS_HISTORY",
                    "METADATA"
                FROM "EMBEDDING_LINEAGE"
                WHERE "DOCUMENT_ID" = ?
            """, (document_id,))
            
            results = []
            for row in cursor.fetchall():
                # Convert JSON strings back to objects
                transformation_history = json.loads(row[9])
                access_history = json.loads(row[10])
                metadata = json.loads(row[11])
                
                # Create a LineageRecord instance
                data = {
                    "id": row[0],
                    "document_id": row[1],
                    "document_hash": row[2],
                    "embedding_model": row[3],
                    "embedding_model_version": row[4],
                    "embedding_timestamp": row[5].isoformat() if row[5] else None,
                    "embedding_dimension": row[6],
                    "source_system": row[7],
                    "source_location": row[8],
                    "transformation_history": transformation_history,
                    "access_history": access_history,
                    "metadata": metadata
                }
                
                results.append(LineageRecord.from_dict(data))
            
            cursor.close()
            return results
        except Exception as e:
            logger.error(f"Error finding lineage records: {str(e)}")
            raise
    
    def record_access(self, lineage_id: str, access_type: str, **kwargs) -> None:
        """
        Record an access event for a lineage record.
        
        Args:
            lineage_id: The ID of the lineage record
            access_type: Type of access (e.g., "search", "retrieval", "update")
            **kwargs: Additional arguments to pass to record_access method
        """
        record = self.get_lineage_record(lineage_id)
        if record:
            record.record_access(access_type, **kwargs)
            self.save_lineage_record(record)
    
    def record_transformation(self, lineage_id: str, transformation_type: str, details: Dict[str, Any]) -> None:
        """
        Record a transformation for a lineage record.
        
        Args:
            lineage_id: The ID of the lineage record
            transformation_type: Type of transformation
            details: Details about the transformation
        """
        record = self.get_lineage_record(lineage_id)
        if record:
            record.record_transformation(transformation_type, details)
            self.save_lineage_record(record)
    
    def get_audit_report(self, 
                        start_date: Optional[str] = None, 
                        end_date: Optional[str] = None,
                        user_id: Optional[str] = None,
                        application: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate an audit report for lineage records.
        
        Args:
            start_date: Optional start date for the report (ISO format)
            end_date: Optional end date for the report (ISO format)
            user_id: Optional user ID to filter by
            application: Optional application name to filter by
            
        Returns:
            A dictionary containing the audit report
        """
        try:
            cursor = self.connection.cursor()
            
            # Base query
            query = """
                SELECT
                    "LINEAGE_ID",
                    "DOCUMENT_ID",
                    "EMBEDDING_MODEL",
                    "EMBEDDING_TIMESTAMP",
                    "SOURCE_SYSTEM",
                    "ACCESS_HISTORY"
                FROM "EMBEDDING_LINEAGE"
                WHERE 1=1
            """
            
            params = []
            
            # Add date filters if provided
            if start_date:
                query += " AND \"EMBEDDING_TIMESTAMP\" >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND \"EMBEDDING_TIMESTAMP\" <= ?"
                params.append(end_date)
            
            cursor.execute(query, params)
            
            # Process results
            access_events = []
            for row in cursor.fetchall():
                lineage_id = row[0]
                document_id = row[1]
                embedding_model = row[2]
                embedding_timestamp = row[3].isoformat() if row[3] else None
                source_system = row[4]
                access_history = json.loads(row[5])
                
                # Filter access events by user_id and application if provided
                for event in access_history:
                    if user_id and event.get("user_id") != user_id:
                        continue
                    
                    if application and event.get("application") != application:
                        continue
                    
                    # Add document info to the event
                    event["lineage_id"] = lineage_id
                    event["document_id"] = document_id
                    event["embedding_model"] = embedding_model
                    event["embedding_timestamp"] = embedding_timestamp
                    event["source_system"] = source_system
                    
                    access_events.append(event)
            
            cursor.close()
            
            # Sort events by timestamp
            access_events.sort(key=lambda e: e["timestamp"])
            
            # Compile report
            report = {
                "start_date": start_date,
                "end_date": end_date,
                "user_id": user_id,
                "application": application,
                "total_events": len(access_events),
                "events": access_events
            }
            
            return report
        except Exception as e:
            logger.error(f"Error generating audit report: {str(e)}")
            raise


# Decorator for adding lineage tracking to vectorstore methods
def track_lineage(method):
    """
    Decorator that adds lineage tracking to vectorstore methods.
    
    This decorator can be applied to methods like add_texts and similarity_search
    to automatically record lineage information.
    
    Args:
        method: The method to decorate
    
    Returns:
        The decorated method
    """
    def wrapper(self, *args, **kwargs):
        # Check if lineage tracking is enabled
        if not hasattr(self, "lineage_manager") or self.lineage_manager is None:
            # Lineage tracking not enabled, just call the original method
            return method(self, *args, **kwargs)
        
        # Get method name and args for lineage tracking
        method_name = method.__name__
        
        # Record appropriate lineage information based on the method
        if method_name == "add_texts":
            # For add_texts, create lineage records for the new documents
            texts = args[0]
            metadatas = kwargs.get("metadatas") or (args[1] if len(args) > 1 else None)
            
            # Create lineage records
            for i, text in enumerate(texts):
                metadata = metadatas[i] if metadatas else {}
                
                # Create lineage record
                embedding_model = self.embedding.__class__.__name__
                embedding_dimension = self._get_embedding_dimension()
                
                lineage_record = self.lineage_manager.create_lineage_record(
                    document_content=text,
                    embedding_model=embedding_model,
                    embedding_dimension=embedding_dimension,
                    metadata=metadata,
                    source_system=metadata.get("source_system"),
                    source_location=metadata.get("source_location")
                )
                
                # Store lineage ID in document metadata
                if metadatas and i < len(metadatas):
                    if metadatas[i] is None:
                        metadatas[i] = {}
                    metadatas[i]["lineage_id"] = lineage_record.id
        
        elif method_name in ["similarity_search", "similarity_search_with_score", 
                           "max_marginal_relevance_search"]:
            # For search methods, record access events
            query = args[0] if len(args) > 0 else kwargs.get("query")
            filter_dict = kwargs.get("filter")
            
            # Get user and application info from context if available
            user_id = getattr(self, "current_user_id", None)
            application = getattr(self, "current_application", None)
            
            # Record query in access history for all documents that will be returned
            access_details = {
                "query": query,
                "filter": filter_dict,
                "method": method_name
            }
        
        # Call the original method
        result = method(self, *args, **kwargs)
        
        # Update lineage records for search results if applicable
        if method_name in ["similarity_search", "similarity_search_with_score"]:
            if result:
                # For similarity_search and similarity_search_with_score, process results
                docs = result if method_name == "similarity_search" else [doc for doc, _ in result]
                
                for doc in docs:
                    lineage_id = doc.metadata.get("lineage_id")
                    if lineage_id:
                        self.lineage_manager.record_access(
                            lineage_id=lineage_id,
                            access_type="search",
                            user_id=user_id,
                            application=application,
                            details=access_details
                        )
        
        elif method_name == "max_marginal_relevance_search":
            # For MMR search, process results
            docs = result
            for doc in docs:
                lineage_id = doc.metadata.get("lineage_id")
                if lineage_id:
                    self.lineage_manager.record_access(
                        lineage_id=lineage_id,
                        access_type="mmr_search",
                        user_id=user_id,
                        application=application,
                        details=access_details
                    )
        
        return result
    
    return wrapper