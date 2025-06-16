"""
Transformation tracking module for monitoring data-to-vector transformations.

This module provides tools for tracking how source data transforms into vector
embeddings, capturing information about the transformation process at each step.
"""

import uuid
import time
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class TransformationStage:
    """
    Represents a single stage in the data-to-vector transformation process.
    
    Captures the input, transformation logic, output, and metadata about a
    stage in the data transformation pipeline.
    """
    
    def __init__(
        self,
        stage_id: str,
        stage_name: str,
        input_data: Any,
        output_data: Any,
        transformation_parameters: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
    ):
        """
        Initialize a transformation stage.
        
        Args:
            stage_id: Unique identifier for this stage
            stage_name: Name of the transformation stage
            input_data: Input data for this stage
            output_data: Output data from this stage
            transformation_parameters: Parameters used for the transformation
            metadata: Additional metadata about the stage
            timestamp: Time when this stage was executed
        """
        self.stage_id = stage_id
        self.stage_name = stage_name
        self.input_data = input_data
        self.output_data = output_data
        self.transformation_parameters = transformation_parameters
        self.metadata = metadata or {}
        self.timestamp = timestamp or time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the transformation stage to a dictionary."""
        return {
            "stage_id": self.stage_id,
            "stage_name": self.stage_name,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "transformation_parameters": self.transformation_parameters,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TransformationStage":
        """Create a transformation stage from a dictionary."""
        return cls(
            stage_id=data["stage_id"],
            stage_name=data["stage_name"],
            input_data=data["input_data"],
            output_data=data["output_data"],
            transformation_parameters=data["transformation_parameters"],
            metadata=data.get("metadata", {}),
            timestamp=data.get("timestamp", time.time()),
        )


class TransformationPipeline:
    """
    Represents a complete data-to-vector transformation pipeline.
    
    Tracks the sequence of transformation stages from source data to vector.
    """
    
    def __init__(
        self,
        pipeline_id: str,
        source_data_id: str,
        source_data_type: str,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
    ):
        """
        Initialize a transformation pipeline.
        
        Args:
            pipeline_id: Unique identifier for this pipeline
            source_data_id: Identifier for the source data
            source_data_type: Type of the source data
            metadata: Additional metadata about the pipeline
            timestamp: Time when this pipeline was created
        """
        self.pipeline_id = pipeline_id
        self.source_data_id = source_data_id
        self.source_data_type = source_data_type
        self.metadata = metadata or {}
        self.timestamp = timestamp or time.time()
        self.stages: List[TransformationStage] = []
        self.final_vector_id: Optional[str] = None
    
    def add_stage(self, stage: TransformationStage) -> None:
        """
        Add a transformation stage to the pipeline.
        
        Args:
            stage: The transformation stage to add
        """
        self.stages.append(stage)
    
    def set_final_vector_id(self, vector_id: str) -> None:
        """
        Set the ID of the final vector produced by this pipeline.
        
        Args:
            vector_id: ID of the final vector
        """
        self.final_vector_id = vector_id
    
    def get_stage(self, stage_id: str) -> Optional[TransformationStage]:
        """
        Get a transformation stage by ID.
        
        Args:
            stage_id: The ID of the stage to retrieve
            
        Returns:
            The transformation stage or None if not found
        """
        for stage in self.stages:
            if stage.stage_id == stage_id:
                return stage
        return None
    
    def get_stage_by_name(self, stage_name: str) -> Optional[TransformationStage]:
        """
        Get a transformation stage by name.
        
        Args:
            stage_name: The name of the stage to retrieve
            
        Returns:
            The first transformation stage with the given name or None if not found
        """
        for stage in self.stages:
            if stage.stage_name == stage_name:
                return stage
        return None
    
    def get_stages_by_name(self, stage_name: str) -> List[TransformationStage]:
        """
        Get all transformation stages with a specific name.
        
        Args:
            stage_name: The name of the stages to retrieve
            
        Returns:
            List of transformation stages with the specified name
        """
        return [stage for stage in self.stages if stage.stage_name == stage_name]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the transformation pipeline to a dictionary."""
        return {
            "pipeline_id": self.pipeline_id,
            "source_data_id": self.source_data_id,
            "source_data_type": self.source_data_type,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "stages": [stage.to_dict() for stage in self.stages],
            "final_vector_id": self.final_vector_id,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TransformationPipeline":
        """Create a transformation pipeline from a dictionary."""
        pipeline = cls(
            pipeline_id=data["pipeline_id"],
            source_data_id=data["source_data_id"],
            source_data_type=data["source_data_type"],
            metadata=data.get("metadata", {}),
            timestamp=data.get("timestamp", time.time()),
        )
        
        # Add stages
        for stage_data in data.get("stages", []):
            stage = TransformationStage.from_dict(stage_data)
            pipeline.add_stage(stage)
        
        # Set final vector ID if available
        if "final_vector_id" in data:
            pipeline.final_vector_id = data["final_vector_id"]
        
        return pipeline


class TransformationTracker:
    """
    Tracks data-to-vector transformations.
    
    Provides tools for creating, tracking, and analyzing how source data
    transforms into vector embeddings.
    """
    
    def __init__(self, storage_backend=None):
        """
        Initialize a transformation tracker.
        
        Args:
            storage_backend: Optional backend for storing transformation pipelines
        """
        self.active_pipelines: Dict[str, TransformationPipeline] = {}
        self.storage_backend = storage_backend
    
    def create_pipeline(
        self,
        source_data_id: str,
        source_data_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a new transformation pipeline.
        
        Args:
            source_data_id: Identifier for the source data
            source_data_type: Type of the source data
            metadata: Additional metadata about the pipeline
            
        Returns:
            The ID of the newly created transformation pipeline
        """
        pipeline_id = str(uuid.uuid4())
        self.active_pipelines[pipeline_id] = TransformationPipeline(
            pipeline_id=pipeline_id,
            source_data_id=source_data_id,
            source_data_type=source_data_type,
            metadata=metadata,
        )
        return pipeline_id
    
    def add_stage(
        self,
        pipeline_id: str,
        stage_name: str,
        input_data: Any,
        output_data: Any,
        transformation_parameters: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add a stage to a transformation pipeline.
        
        Args:
            pipeline_id: ID of the transformation pipeline
            stage_name: Name of the transformation stage
            input_data: Input data for this stage
            output_data: Output data from this stage
            transformation_parameters: Parameters used for the transformation
            metadata: Additional metadata about the stage
            
        Returns:
            The ID of the newly created stage
            
        Raises:
            ValueError: If the pipeline_id is not found
        """
        if pipeline_id not in self.active_pipelines:
            raise ValueError(f"Transformation pipeline {pipeline_id} not found")
        
        stage_id = str(uuid.uuid4())
        stage = TransformationStage(
            stage_id=stage_id,
            stage_name=stage_name,
            input_data=input_data,
            output_data=output_data,
            transformation_parameters=transformation_parameters,
            metadata=metadata,
        )
        
        self.active_pipelines[pipeline_id].add_stage(stage)
        return stage_id
    
    def set_final_vector(self, pipeline_id: str, vector_id: str) -> None:
        """
        Set the final vector for a transformation pipeline.
        
        Args:
            pipeline_id: ID of the transformation pipeline
            vector_id: ID of the final vector
            
        Raises:
            ValueError: If the pipeline_id is not found
        """
        if pipeline_id not in self.active_pipelines:
            raise ValueError(f"Transformation pipeline {pipeline_id} not found")
        
        self.active_pipelines[pipeline_id].set_final_vector_id(vector_id)
    
    def get_pipeline(self, pipeline_id: str) -> Optional[TransformationPipeline]:
        """
        Get a transformation pipeline by ID.
        
        Args:
            pipeline_id: The ID of the pipeline to retrieve
            
        Returns:
            The transformation pipeline or None if not found
        """
        if pipeline_id in self.active_pipelines:
            return self.active_pipelines[pipeline_id]
        
        # Try to load from storage backend
        if self.storage_backend:
            pipeline_data = self.storage_backend.load_pipeline(pipeline_id)
            if pipeline_data:
                pipeline = TransformationPipeline.from_dict(pipeline_data)
                self.active_pipelines[pipeline_id] = pipeline
                return pipeline
        
        return None
    
    def save_pipeline(self, pipeline_id: str) -> bool:
        """
        Save a transformation pipeline to the storage backend.
        
        Args:
            pipeline_id: The ID of the pipeline to save
            
        Returns:
            True if the pipeline was saved successfully, False otherwise
        """
        if pipeline_id not in self.active_pipelines:
            logger.warning(f"Transformation pipeline {pipeline_id} not found for saving")
            return False
        
        if self.storage_backend:
            pipeline_data = self.active_pipelines[pipeline_id].to_dict()
            return self.storage_backend.save_pipeline(pipeline_id, pipeline_data)
        
        logger.warning("No storage backend configured for saving transformation pipeline")
        return False
    
    def get_pipelines_by_source_data(self, source_data_id: str) -> List[TransformationPipeline]:
        """
        Get all transformation pipelines for a specific source data.
        
        Args:
            source_data_id: The ID of the source data
            
        Returns:
            List of transformation pipelines for the specified source data
        """
        return [
            pipeline for pipeline in self.active_pipelines.values()
            if pipeline.source_data_id == source_data_id
        ]
    
    def get_pipeline_by_vector_id(self, vector_id: str) -> Optional[TransformationPipeline]:
        """
        Get the transformation pipeline that produced a specific vector.
        
        Args:
            vector_id: The ID of the vector
            
        Returns:
            The transformation pipeline or None if not found
        """
        for pipeline in self.active_pipelines.values():
            if pipeline.final_vector_id == vector_id:
                return pipeline
        
        # Try to load from storage backend
        if self.storage_backend:
            pipeline_id = self.storage_backend.get_pipeline_by_vector_id(vector_id)
            if pipeline_id:
                return self.get_pipeline(pipeline_id)
        
        return None
    
    def analyze_pipeline(self, pipeline_id: str) -> Dict[str, Any]:
        """
        Analyze a transformation pipeline for insights.
        
        Args:
            pipeline_id: The ID of the pipeline to analyze
            
        Returns:
            Analysis results
            
        Raises:
            ValueError: If the pipeline_id is not found
        """
        pipeline = self.get_pipeline(pipeline_id)
        if not pipeline:
            raise ValueError(f"Transformation pipeline {pipeline_id} not found for analysis")
        
        # Get basic pipeline information
        result = {
            "pipeline_id": pipeline.pipeline_id,
            "source_data_id": pipeline.source_data_id,
            "source_data_type": pipeline.source_data_type,
            "created_at": pipeline.timestamp,
            "final_vector_id": pipeline.final_vector_id,
            "stage_count": len(pipeline.stages),
        }
        
        # Analyze stages
        stages_by_name = {}
        for stage in pipeline.stages:
            if stage.stage_name not in stages_by_name:
                stages_by_name[stage.stage_name] = []
            stages_by_name[stage.stage_name].append(stage)
        
        result["stages"] = {
            name: len(stages) for name, stages in stages_by_name.items()
        }
        
        # Calculate data reduction/expansion at each stage
        data_size_changes = []
        
        for i, stage in enumerate(pipeline.stages):
            # Try to calculate input and output data sizes
            try:
                input_size = self._estimate_data_size(stage.input_data)
                output_size = self._estimate_data_size(stage.output_data)
                
                if input_size and output_size:
                    change_factor = output_size / input_size if input_size > 0 else 0
                    data_size_changes.append({
                        "stage_id": stage.stage_id,
                        "stage_name": stage.stage_name,
                        "input_size": input_size,
                        "output_size": output_size,
                        "change_factor": change_factor,
                    })
            except Exception as e:
                logger.warning(f"Error calculating data size for stage {stage.stage_id}: {e}")
        
        result["data_size_changes"] = data_size_changes
        
        # Calculate total transformation time
        if pipeline.stages:
            start_time = min(stage.timestamp for stage in pipeline.stages)
            end_time = max(stage.timestamp for stage in pipeline.stages)
            result["total_transformation_time"] = end_time - start_time
        
        return result
    
    def _estimate_data_size(self, data: Any) -> Optional[int]:
        """
        Estimate the size of data in bytes.
        
        Args:
            data: The data to estimate the size of
            
        Returns:
            Estimated size in bytes or None if size cannot be determined
        """
        if data is None:
            return 0
        
        try:
            # Handle common types
            if isinstance(data, (str, bytes)):
                return len(data)
            elif isinstance(data, np.ndarray):
                return data.nbytes
            elif isinstance(data, (list, tuple)):
                return sum(self._estimate_data_size(item) or 0 for item in data)
            elif isinstance(data, dict):
                return sum(
                    (self._estimate_data_size(k) or 0) + (self._estimate_data_size(v) or 0)
                    for k, v in data.items()
                )
            else:
                # For other types, use object size approximation
                import sys
                return sys.getsizeof(data)
        except Exception:
            return None