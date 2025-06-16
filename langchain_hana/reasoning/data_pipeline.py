"""
Data Pipeline module for tracking the complete life cycle of data from HANA tables to vectors and back.

This module provides components for visualizing and interacting with the entire data transformation
process, from raw HANA tables through intermediate representations to vector embeddings and back.
"""

import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

# Setup logging
logger = logging.getLogger(__name__)

class DataSourceMetadata:
    """Metadata about a HANA table or data source."""
    
    def __init__(
        self,
        source_id: str,
        schema_name: str,
        table_name: str,
        column_metadata: Dict[str, Dict[str, Any]],
        row_count: int,
        sample_data: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Initialize data source metadata.
        
        Args:
            source_id: Unique identifier for the data source
            schema_name: HANA schema name
            table_name: HANA table name
            column_metadata: Metadata for each column including data type, constraints, etc.
            row_count: Number of rows in the table
            sample_data: Sample data from the table (optional)
        """
        self.source_id = source_id
        self.schema_name = schema_name
        self.table_name = table_name
        self.column_metadata = column_metadata
        self.row_count = row_count
        self.sample_data = sample_data
        self.created_at = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "source_id": self.source_id,
            "schema_name": self.schema_name,
            "table_name": self.table_name,
            "column_metadata": self.column_metadata,
            "row_count": self.row_count,
            "sample_data": self.sample_data[:5] if self.sample_data else None,
            "created_at": self.created_at,
        }
    
    @classmethod
    def from_connection(
        cls,
        connection,
        schema_name: str,
        table_name: str,
        include_sample: bool = True,
        sample_size: int = 5
    ) -> "DataSourceMetadata":
        """
        Create metadata from a HANA connection and table information.
        
        Args:
            connection: HANA database connection
            schema_name: HANA schema name
            table_name: HANA table name
            include_sample: Whether to include sample data
            sample_size: Number of sample rows to include
            
        Returns:
            DataSourceMetadata object
        """
        source_id = str(uuid.uuid4())
        
        # Get column metadata
        cursor = connection.cursor()
        cursor.execute(f"""
            SELECT COLUMN_NAME, DATA_TYPE_NAME, LENGTH, SCALE, IS_NULLABLE
            FROM TABLE_COLUMNS
            WHERE SCHEMA_NAME = '{schema_name}' AND TABLE_NAME = '{table_name}'
            ORDER BY POSITION
        """)
        
        columns = cursor.fetchall()
        column_metadata = {}
        
        for col in columns:
            column_name, data_type, length, scale, is_nullable = col
            column_metadata[column_name] = {
                "data_type": data_type,
                "length": length,
                "scale": scale,
                "is_nullable": is_nullable == "TRUE",
            }
        
        # Get row count
        cursor.execute(f"""
            SELECT COUNT(*) FROM "{schema_name}"."{table_name}"
        """)
        row_count = cursor.fetchone()[0]
        
        # Get sample data if requested
        sample_data = None
        if include_sample:
            column_names = list(column_metadata.keys())
            cols_str = ", ".join([f'"{c}"' for c in column_names])
            
            cursor.execute(f"""
                SELECT {cols_str} FROM "{schema_name}"."{table_name}" LIMIT {sample_size}
            """)
            
            rows = cursor.fetchall()
            sample_data = []
            
            for row in rows:
                row_dict = {}
                for i, col_name in enumerate(column_names):
                    # Handle binary data and convert to string representation if needed
                    if isinstance(row[i], bytes):
                        row_dict[col_name] = f"<binary data: {len(row[i])} bytes>"
                    else:
                        row_dict[col_name] = row[i]
                sample_data.append(row_dict)
        
        return cls(
            source_id=source_id,
            schema_name=schema_name,
            table_name=table_name,
            column_metadata=column_metadata,
            row_count=row_count,
            sample_data=sample_data,
        )


class IntermediateRepresentation:
    """Representation of data at an intermediate stage of the transformation pipeline."""
    
    def __init__(
        self,
        stage_id: str,
        stage_name: str,
        stage_description: str,
        source_id: str,
        column_mapping: Dict[str, List[str]],
        data_sample: Optional[List[Dict[str, Any]]] = None,
        processing_metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize intermediate representation.
        
        Args:
            stage_id: Unique identifier for this transformation stage
            stage_name: Name of the transformation stage
            stage_description: Description of what happens in this stage
            source_id: ID of the source data
            column_mapping: Mapping of output columns to input columns/transformations
            data_sample: Sample data after transformation
            processing_metadata: Metadata about the processing (timing, etc.)
        """
        self.stage_id = stage_id
        self.stage_name = stage_name
        self.stage_description = stage_description
        self.source_id = source_id
        self.column_mapping = column_mapping
        self.data_sample = data_sample
        self.processing_metadata = processing_metadata or {}
        self.created_at = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "stage_id": self.stage_id,
            "stage_name": self.stage_name,
            "stage_description": self.stage_description,
            "source_id": self.source_id,
            "column_mapping": self.column_mapping,
            "data_sample": self.data_sample[:5] if self.data_sample else None,
            "processing_metadata": self.processing_metadata,
            "created_at": self.created_at,
        }


class VectorRepresentation:
    """Representation of data after vectorization."""
    
    def __init__(
        self,
        vector_id: str,
        source_id: str,
        model_name: str,
        vector_dimensions: int,
        vector_sample: Optional[List[float]] = None,
        original_text: Optional[str] = None,
        processing_metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize vector representation.
        
        Args:
            vector_id: Unique identifier for this vector
            source_id: ID of the source data
            model_name: Name of the embedding model used
            vector_dimensions: Dimensionality of the vector
            vector_sample: Sample of the vector data (typically truncated)
            original_text: Original text that was vectorized
            processing_metadata: Metadata about the processing
        """
        self.vector_id = vector_id
        self.source_id = source_id
        self.model_name = model_name
        self.vector_dimensions = vector_dimensions
        self.vector_sample = vector_sample
        self.original_text = original_text
        self.processing_metadata = processing_metadata or {}
        self.created_at = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "vector_id": self.vector_id,
            "source_id": self.source_id,
            "model_name": self.model_name,
            "vector_dimensions": self.vector_dimensions,
            "vector_sample": self.vector_sample[:10] if self.vector_sample else None,
            "original_text": (self.original_text[:200] + "..." if len(self.original_text) > 200 else self.original_text) if self.original_text else None,
            "processing_metadata": self.processing_metadata,
            "created_at": self.created_at,
        }


class TransformationRule:
    """Rule for transforming data from one representation to another."""
    
    def __init__(
        self,
        rule_id: str,
        rule_name: str,
        rule_description: str,
        input_columns: List[str],
        output_columns: List[str],
        transformation_type: str,
        transformation_params: Dict[str, Any],
    ):
        """
        Initialize transformation rule.
        
        Args:
            rule_id: Unique identifier for this rule
            rule_name: Name of the rule
            rule_description: Description of what the rule does
            input_columns: List of input column names
            output_columns: List of output column names
            transformation_type: Type of transformation (join, filter, aggregate, etc.)
            transformation_params: Parameters for the transformation
        """
        self.rule_id = rule_id
        self.rule_name = rule_name
        self.rule_description = rule_description
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.transformation_type = transformation_type
        self.transformation_params = transformation_params
        self.created_at = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "rule_description": self.rule_description,
            "input_columns": self.input_columns,
            "output_columns": self.output_columns,
            "transformation_type": self.transformation_type,
            "transformation_params": self.transformation_params,
            "created_at": self.created_at,
        }


class DataPipeline:
    """
    Tracks the complete data transformation pipeline from HANA tables to vectors.
    
    This class provides methods for tracking and visualizing the entire data pipeline,
    from raw tables to vector embeddings, including all intermediate transformations.
    """
    
    def __init__(self, connection=None):
        """
        Initialize data pipeline tracker.
        
        Args:
            connection: HANA database connection (optional)
        """
        self.connection = connection
        self.data_sources = {}
        self.intermediate_stages = {}
        self.vector_representations = {}
        self.transformation_rules = {}
        self.pipeline_id = str(uuid.uuid4())
    
    def register_data_source(
        self,
        schema_name: str,
        table_name: str,
        include_sample: bool = True,
        sample_size: int = 5,
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Register a HANA table as a data source.
        
        Args:
            schema_name: HANA schema name
            table_name: HANA table name
            include_sample: Whether to include sample data
            sample_size: Number of sample rows to include
            custom_metadata: Additional metadata to include
            
        Returns:
            source_id: Unique identifier for the data source
        """
        if not self.connection:
            raise ValueError("Database connection is required to register a data source")
        
        source_metadata = DataSourceMetadata.from_connection(
            self.connection,
            schema_name,
            table_name,
            include_sample,
            sample_size
        )
        
        if custom_metadata:
            source_metadata.processing_metadata = custom_metadata
        
        self.data_sources[source_metadata.source_id] = source_metadata
        return source_metadata.source_id
    
    def register_intermediate_stage(
        self,
        stage_name: str,
        stage_description: str,
        source_id: str,
        column_mapping: Dict[str, List[str]],
        data_sample: Optional[List[Dict[str, Any]]] = None,
        processing_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Register an intermediate transformation stage.
        
        Args:
            stage_name: Name of the transformation stage
            stage_description: Description of what happens in this stage
            source_id: ID of the source data
            column_mapping: Mapping of output columns to input columns/transformations
            data_sample: Sample data after transformation
            processing_metadata: Metadata about the processing
            
        Returns:
            stage_id: Unique identifier for the stage
        """
        stage_id = str(uuid.uuid4())
        
        stage = IntermediateRepresentation(
            stage_id=stage_id,
            stage_name=stage_name,
            stage_description=stage_description,
            source_id=source_id,
            column_mapping=column_mapping,
            data_sample=data_sample,
            processing_metadata=processing_metadata
        )
        
        self.intermediate_stages[stage_id] = stage
        return stage_id
    
    def register_vector_representation(
        self,
        source_id: str,
        model_name: str,
        vector_dimensions: int,
        vector_sample: Optional[List[float]] = None,
        original_text: Optional[str] = None,
        processing_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Register a vector representation.
        
        Args:
            source_id: ID of the source data
            model_name: Name of the embedding model used
            vector_dimensions: Dimensionality of the vector
            vector_sample: Sample of the vector data
            original_text: Original text that was vectorized
            processing_metadata: Metadata about the processing
            
        Returns:
            vector_id: Unique identifier for the vector
        """
        vector_id = str(uuid.uuid4())
        
        vector = VectorRepresentation(
            vector_id=vector_id,
            source_id=source_id,
            model_name=model_name,
            vector_dimensions=vector_dimensions,
            vector_sample=vector_sample,
            original_text=original_text,
            processing_metadata=processing_metadata
        )
        
        self.vector_representations[vector_id] = vector
        return vector_id
    
    def register_transformation_rule(
        self,
        rule_name: str,
        rule_description: str,
        input_columns: List[str],
        output_columns: List[str],
        transformation_type: str,
        transformation_params: Dict[str, Any]
    ) -> str:
        """
        Register a transformation rule.
        
        Args:
            rule_name: Name of the rule
            rule_description: Description of what the rule does
            input_columns: List of input column names
            output_columns: List of output column names
            transformation_type: Type of transformation
            transformation_params: Parameters for the transformation
            
        Returns:
            rule_id: Unique identifier for the rule
        """
        rule_id = str(uuid.uuid4())
        
        rule = TransformationRule(
            rule_id=rule_id,
            rule_name=rule_name,
            rule_description=rule_description,
            input_columns=input_columns,
            output_columns=output_columns,
            transformation_type=transformation_type,
            transformation_params=transformation_params
        )
        
        self.transformation_rules[rule_id] = rule
        return rule_id
    
    def get_complete_pipeline(self, source_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get the complete data pipeline visualization data.
        
        Args:
            source_id: Filter to a specific source (optional)
            
        Returns:
            Dictionary with the complete pipeline data
        """
        sources = {}
        intermediate = {}
        vectors = {}
        rules = {}
        
        # Filter by source_id if provided
        if source_id:
            if source_id in self.data_sources:
                sources[source_id] = self.data_sources[source_id].to_dict()
            
            for stage_id, stage in self.intermediate_stages.items():
                if stage.source_id == source_id:
                    intermediate[stage_id] = stage.to_dict()
            
            for vector_id, vector in self.vector_representations.items():
                if vector.source_id == source_id:
                    vectors[vector_id] = vector.to_dict()
        else:
            # Include all data
            sources = {src_id: src.to_dict() for src_id, src in self.data_sources.items()}
            intermediate = {stg_id: stg.to_dict() for stg_id, stg in self.intermediate_stages.items()}
            vectors = {vec_id: vec.to_dict() for vec_id, vec in self.vector_representations.items()}
        
        # Always include all transformation rules
        rules = {rule_id: rule.to_dict() for rule_id, rule in self.transformation_rules.items()}
        
        return {
            "pipeline_id": self.pipeline_id,
            "data_sources": sources,
            "intermediate_stages": intermediate,
            "vector_representations": vectors,
            "transformation_rules": rules,
            "created_at": time.time(),
        }
    
    def get_data_lineage(self, vector_id: str) -> Dict[str, Any]:
        """
        Get data lineage for a specific vector.
        
        Args:
            vector_id: ID of the vector
            
        Returns:
            Dictionary with lineage information
        """
        if vector_id not in self.vector_representations:
            raise ValueError(f"Vector ID {vector_id} not found")
        
        vector = self.vector_representations[vector_id]
        source_id = vector.source_id
        
        # Find all intermediate stages related to this source
        related_stages = []
        for stage_id, stage in self.intermediate_stages.items():
            if stage.source_id == source_id:
                related_stages.append(stage.to_dict())
        
        # Sort stages by creation time
        related_stages.sort(key=lambda x: x["created_at"])
        
        # Get the source data
        source_data = self.data_sources.get(source_id)
        if not source_data:
            source_data = {"source_id": source_id, "not_found": True}
        else:
            source_data = source_data.to_dict()
        
        return {
            "vector_id": vector_id,
            "vector_data": vector.to_dict(),
            "source_data": source_data,
            "transformation_stages": related_stages,
            "created_at": time.time(),
        }
    
    def get_reverse_mapping(self, vector_id: str, similarity_threshold: float = 0.8) -> Dict[str, Any]:
        """
        Get the reverse mapping from vector back to source data.
        
        Args:
            vector_id: ID of the vector
            similarity_threshold: Threshold for vector similarity
            
        Returns:
            Dictionary with reverse mapping information
        """
        if vector_id not in self.vector_representations:
            raise ValueError(f"Vector ID {vector_id} not found")
        
        vector = self.vector_representations[vector_id]
        source_id = vector.source_id
        
        # Find the source data
        source_data = self.data_sources.get(source_id)
        if not source_data:
            return {
                "vector_id": vector_id,
                "error": f"Source data with ID {source_id} not found",
            }
        
        # Mock similarity calculation - in real implementation would query HANA
        # for similar vectors using the similarity_threshold
        similar_vectors = []
        for vec_id, vec in self.vector_representations.items():
            if vec_id != vector_id and vec.source_id == source_id:
                # In a real implementation, would calculate actual similarity
                similarity = 0.9  # Mock similarity score
                if similarity >= similarity_threshold:
                    similar_vectors.append({
                        "vector_id": vec_id,
                        "similarity": similarity,
                        "vector_data": vec.to_dict(),
                    })
        
        return {
            "vector_id": vector_id,
            "source_data": source_data.to_dict(),
            "similar_vectors": similar_vectors,
            "threshold": similarity_threshold,
            "created_at": time.time(),
        }


class DataPipelineManager:
    """
    Manages data pipelines and provides methods for accessing them.
    """
    
    _instance = None
    _pipelines = {}
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def create_pipeline(self, connection=None) -> str:
        """
        Create a new data pipeline.
        
        Args:
            connection: HANA database connection (optional)
            
        Returns:
            pipeline_id: Unique identifier for the pipeline
        """
        pipeline = DataPipeline(connection)
        self._pipelines[pipeline.pipeline_id] = pipeline
        return pipeline.pipeline_id
    
    def get_pipeline(self, pipeline_id: str) -> DataPipeline:
        """
        Get a pipeline by ID.
        
        Args:
            pipeline_id: ID of the pipeline
            
        Returns:
            DataPipeline object
        """
        if pipeline_id not in self._pipelines:
            raise ValueError(f"Pipeline ID {pipeline_id} not found")
        
        return self._pipelines[pipeline_id]
    
    def list_pipelines(self) -> List[Dict[str, Any]]:
        """
        List all pipelines.
        
        Returns:
            List of pipeline summary dictionaries
        """
        return [
            {
                "pipeline_id": p_id,
                "created_at": pipeline.created_at if hasattr(pipeline, "created_at") else time.time(),
                "data_sources_count": len(pipeline.data_sources),
                "vectors_count": len(pipeline.vector_representations),
            }
            for p_id, pipeline in self._pipelines.items()
        ]
    
    def delete_pipeline(self, pipeline_id: str) -> bool:
        """
        Delete a pipeline.
        
        Args:
            pipeline_id: ID of the pipeline
            
        Returns:
            Success flag
        """
        if pipeline_id in self._pipelines:
            del self._pipelines[pipeline_id]
            return True
        return False