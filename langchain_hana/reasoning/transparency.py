"""
Reasoning transparency module for tracking LLM reasoning through vector data.

This module implements SimpleQA-inspired reasoning path tracking to provide
visibility into how language models navigate and reason with vector data.
"""

import uuid
import time
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class ReasoningStep:
    """
    Represents a single step in a reasoning path.
    
    Captures the input, reasoning, output, and metadata about a single
    step in the model's reasoning process.
    """
    
    def __init__(
        self,
        step_id: str,
        step_type: str,
        input_data: Any,
        output_data: Any,
        reasoning: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        vectors_accessed: Optional[List[str]] = None,
        confidence: Optional[float] = None,
        timestamp: Optional[float] = None,
        parent_step_id: Optional[str] = None,
    ):
        """
        Initialize a reasoning step.
        
        Args:
            step_id: Unique identifier for this step
            step_type: Type of reasoning step (e.g., 'retrieval', 'comparison', 'inference')
            input_data: Input data for this step
            output_data: Output data from this step
            reasoning: Text description of the reasoning applied
            metadata: Additional metadata about the step
            vectors_accessed: IDs of vectors accessed during this step
            confidence: Confidence score for this step (0-1)
            timestamp: Time when this step was executed
            parent_step_id: ID of the parent step
        """
        self.step_id = step_id
        self.step_type = step_type
        self.input_data = input_data
        self.output_data = output_data
        self.reasoning = reasoning
        self.metadata = metadata or {}
        self.vectors_accessed = vectors_accessed or []
        self.confidence = confidence
        self.timestamp = timestamp or time.time()
        self.parent_step_id = parent_step_id
        self.child_step_ids: List[str] = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the reasoning step to a dictionary."""
        return {
            "step_id": self.step_id,
            "step_type": self.step_type,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "reasoning": self.reasoning,
            "metadata": self.metadata,
            "vectors_accessed": self.vectors_accessed,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "parent_step_id": self.parent_step_id,
            "child_step_ids": self.child_step_ids,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReasoningStep":
        """Create a reasoning step from a dictionary."""
        step = cls(
            step_id=data["step_id"],
            step_type=data["step_type"],
            input_data=data["input_data"],
            output_data=data["output_data"],
            reasoning=data.get("reasoning"),
            metadata=data.get("metadata", {}),
            vectors_accessed=data.get("vectors_accessed", []),
            confidence=data.get("confidence"),
            timestamp=data.get("timestamp", time.time()),
            parent_step_id=data.get("parent_step_id"),
        )
        step.child_step_ids = data.get("child_step_ids", [])
        return step


class ReasoningPath:
    """
    Represents a complete reasoning path.
    
    Tracks the sequence of reasoning steps from query to answer.
    """
    
    def __init__(
        self,
        path_id: str,
        query: str,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
    ):
        """
        Initialize a reasoning path.
        
        Args:
            path_id: Unique identifier for this reasoning path
            query: The original query that initiated this reasoning path
            metadata: Additional metadata about the path
            timestamp: Time when this path was created
        """
        self.path_id = path_id
        self.query = query
        self.metadata = metadata or {}
        self.timestamp = timestamp or time.time()
        self.steps: Dict[str, ReasoningStep] = {}
        self.root_step_ids: List[str] = []
    
    def add_step(self, step: ReasoningStep) -> None:
        """
        Add a reasoning step to the path.
        
        Args:
            step: The reasoning step to add
        """
        self.steps[step.step_id] = step
        
        if step.parent_step_id:
            # Add this step as a child of its parent
            if step.parent_step_id in self.steps:
                self.steps[step.parent_step_id].child_step_ids.append(step.step_id)
            else:
                logger.warning(
                    f"Parent step {step.parent_step_id} not found for step {step.step_id}"
                )
        else:
            # This is a root step
            self.root_step_ids.append(step.step_id)
    
    def get_step(self, step_id: str) -> Optional[ReasoningStep]:
        """
        Get a reasoning step by ID.
        
        Args:
            step_id: The ID of the step to retrieve
            
        Returns:
            The reasoning step or None if not found
        """
        return self.steps.get(step_id)
    
    def get_steps_by_type(self, step_type: str) -> List[ReasoningStep]:
        """
        Get all reasoning steps of a specific type.
        
        Args:
            step_type: The type of steps to retrieve
            
        Returns:
            List of reasoning steps of the specified type
        """
        return [step for step in self.steps.values() if step.step_type == step_type]
    
    def get_final_steps(self) -> List[ReasoningStep]:
        """
        Get the final steps in the reasoning path.
        
        Returns:
            List of steps that have no children
        """
        return [
            step for step in self.steps.values() if not step.child_step_ids
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the reasoning path to a dictionary."""
        return {
            "path_id": self.path_id,
            "query": self.query,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "steps": {step_id: step.to_dict() for step_id, step in self.steps.items()},
            "root_step_ids": self.root_step_ids,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReasoningPath":
        """Create a reasoning path from a dictionary."""
        path = cls(
            path_id=data["path_id"],
            query=data["query"],
            metadata=data.get("metadata", {}),
            timestamp=data.get("timestamp", time.time()),
        )
        
        # First, create all steps without setting parent-child relationships
        for step_id, step_data in data.get("steps", {}).items():
            step = ReasoningStep.from_dict(step_data)
            path.steps[step_id] = step
        
        # Then, set root step IDs
        path.root_step_ids = data.get("root_step_ids", [])
        
        return path


class ReasoningPathTracker:
    """
    Tracks reasoning paths through vector data.
    
    Provides tools for creating, tracking, and analyzing how models reason
    through vector data for specific queries.
    """
    
    def __init__(self, storage_backend=None):
        """
        Initialize a reasoning path tracker.
        
        Args:
            storage_backend: Optional backend for storing reasoning paths
        """
        self.active_paths: Dict[str, ReasoningPath] = {}
        self.storage_backend = storage_backend
    
    def create_path(
        self, query: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new reasoning path.
        
        Args:
            query: The query that initiated this reasoning path
            metadata: Additional metadata about the path
            
        Returns:
            The ID of the newly created reasoning path
        """
        path_id = str(uuid.uuid4())
        self.active_paths[path_id] = ReasoningPath(
            path_id=path_id,
            query=query,
            metadata=metadata,
        )
        return path_id
    
    def add_step(
        self,
        path_id: str,
        step_type: str,
        input_data: Any,
        output_data: Any,
        reasoning: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        vectors_accessed: Optional[List[str]] = None,
        confidence: Optional[float] = None,
        parent_step_id: Optional[str] = None,
    ) -> str:
        """
        Add a step to a reasoning path.
        
        Args:
            path_id: ID of the reasoning path
            step_type: Type of reasoning step
            input_data: Input data for this step
            output_data: Output data from this step
            reasoning: Text description of the reasoning applied
            metadata: Additional metadata about the step
            vectors_accessed: IDs of vectors accessed during this step
            confidence: Confidence score for this step (0-1)
            parent_step_id: ID of the parent step
            
        Returns:
            The ID of the newly created step
            
        Raises:
            ValueError: If the path_id is not found
        """
        if path_id not in self.active_paths:
            raise ValueError(f"Reasoning path {path_id} not found")
        
        step_id = str(uuid.uuid4())
        step = ReasoningStep(
            step_id=step_id,
            step_type=step_type,
            input_data=input_data,
            output_data=output_data,
            reasoning=reasoning,
            metadata=metadata,
            vectors_accessed=vectors_accessed,
            confidence=confidence,
            parent_step_id=parent_step_id,
        )
        
        self.active_paths[path_id].add_step(step)
        return step_id
    
    def get_path(self, path_id: str) -> Optional[ReasoningPath]:
        """
        Get a reasoning path by ID.
        
        Args:
            path_id: The ID of the path to retrieve
            
        Returns:
            The reasoning path or None if not found
        """
        if path_id in self.active_paths:
            return self.active_paths[path_id]
        
        # Try to load from storage backend
        if self.storage_backend:
            path_data = self.storage_backend.load_path(path_id)
            if path_data:
                path = ReasoningPath.from_dict(path_data)
                self.active_paths[path_id] = path
                return path
        
        return None
    
    def save_path(self, path_id: str) -> bool:
        """
        Save a reasoning path to the storage backend.
        
        Args:
            path_id: The ID of the path to save
            
        Returns:
            True if the path was saved successfully, False otherwise
        """
        if path_id not in self.active_paths:
            logger.warning(f"Reasoning path {path_id} not found for saving")
            return False
        
        if self.storage_backend:
            path_data = self.active_paths[path_id].to_dict()
            return self.storage_backend.save_path(path_id, path_data)
        
        logger.warning("No storage backend configured for saving reasoning path")
        return False
    
    def analyze_path(self, path_id: str) -> Dict[str, Any]:
        """
        Analyze a reasoning path for insights.
        
        Args:
            path_id: The ID of the path to analyze
            
        Returns:
            Analysis results including:
            - step_count: Total number of steps
            - step_types: Distribution of step types
            - confidence: Average confidence across steps
            - vectors_accessed: List of accessed vector IDs
            - complexity: Estimated reasoning complexity
            
        Raises:
            ValueError: If the path_id is not found
        """
        path = self.get_path(path_id)
        if not path:
            raise ValueError(f"Reasoning path {path_id} not found for analysis")
        
        steps = list(path.steps.values())
        
        # Calculate basic statistics
        step_count = len(steps)
        step_types = {}
        for step in steps:
            step_types[step.step_type] = step_types.get(step.step_type, 0) + 1
        
        # Calculate average confidence
        confidence_values = [step.confidence for step in steps if step.confidence is not None]
        avg_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else None
        
        # Collect accessed vectors
        vectors_accessed = set()
        for step in steps:
            vectors_accessed.update(step.vectors_accessed)
        
        # Estimate reasoning complexity
        # - Number of steps
        # - Maximum path depth
        # - Branching factor
        max_depth = 0
        visited = set()
        
        def calculate_depth(step_id, current_depth):
            nonlocal max_depth
            if step_id in visited:
                return
            visited.add(step_id)
            max_depth = max(max_depth, current_depth)
            
            step = path.steps.get(step_id)
            if step:
                for child_id in step.child_step_ids:
                    calculate_depth(child_id, current_depth + 1)
        
        for root_id in path.root_step_ids:
            calculate_depth(root_id, 0)
        
        # Calculate branching factor
        branching_factors = []
        for step in steps:
            if step.child_step_ids:
                branching_factors.append(len(step.child_step_ids))
        
        avg_branching = sum(branching_factors) / len(branching_factors) if branching_factors else 0
        
        # Create complexity score (simplified)
        complexity = (max_depth + 1) * avg_branching if avg_branching > 0 else max_depth + 1
        
        return {
            "step_count": step_count,
            "step_types": step_types,
            "confidence": avg_confidence,
            "vectors_accessed": list(vectors_accessed),
            "complexity": {
                "score": complexity,
                "max_depth": max_depth,
                "avg_branching": avg_branching,
            },
            "query": path.query,
            "timestamp": path.timestamp,
        }
    
    def compare_paths(self, path_id_1: str, path_id_2: str) -> Dict[str, Any]:
        """
        Compare two reasoning paths.
        
        Args:
            path_id_1: ID of the first reasoning path
            path_id_2: ID of the second reasoning path
            
        Returns:
            Comparison results
            
        Raises:
            ValueError: If either path_id is not found
        """
        path1 = self.get_path(path_id_1)
        path2 = self.get_path(path_id_2)
        
        if not path1:
            raise ValueError(f"Reasoning path {path_id_1} not found for comparison")
        if not path2:
            raise ValueError(f"Reasoning path {path_id_2} not found for comparison")
        
        # Get analyses for both paths
        analysis1 = self.analyze_path(path_id_1)
        analysis2 = self.analyze_path(path_id_2)
        
        # Compare vector usage
        vectors1 = set(analysis1["vectors_accessed"])
        vectors2 = set(analysis2["vectors_accessed"])
        
        common_vectors = vectors1.intersection(vectors2)
        unique_vectors1 = vectors1 - vectors2
        unique_vectors2 = vectors2 - vectors1
        
        # Compare step types
        step_types1 = set(analysis1["step_types"].keys())
        step_types2 = set(analysis2["step_types"].keys())
        
        common_step_types = step_types1.intersection(step_types2)
        unique_step_types1 = step_types1 - step_types2
        unique_step_types2 = step_types2 - step_types1
        
        # Compare complexity
        complexity_diff = analysis1["complexity"]["score"] - analysis2["complexity"]["score"]
        
        return {
            "vector_overlap": {
                "common_count": len(common_vectors),
                "common_percent": len(common_vectors) / len(vectors1.union(vectors2)) * 100 if vectors1.union(vectors2) else 0,
                "unique_to_path1": list(unique_vectors1),
                "unique_to_path2": list(unique_vectors2),
            },
            "step_type_comparison": {
                "common_types": list(common_step_types),
                "unique_to_path1": list(unique_step_types1),
                "unique_to_path2": list(unique_step_types2),
            },
            "complexity_comparison": {
                "path1_complexity": analysis1["complexity"]["score"],
                "path2_complexity": analysis2["complexity"]["score"],
                "difference": complexity_diff,
                "percent_difference": abs(complexity_diff) / ((analysis1["complexity"]["score"] + analysis2["complexity"]["score"]) / 2) * 100 if (analysis1["complexity"]["score"] + analysis2["complexity"]["score"]) > 0 else 0,
            },
            "path1_analysis": analysis1,
            "path2_analysis": analysis2,
        }