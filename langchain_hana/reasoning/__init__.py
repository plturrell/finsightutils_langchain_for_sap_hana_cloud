"""
Reasoning transparency and validation framework for SAP HANA vector operations.

This module provides tools for tracking, analyzing, and visualizing how
meaning transforms through the vector embedding process and how language
models reason with the resulting vectors.
"""

from langchain_hana.reasoning.transparency import ReasoningPathTracker
from langchain_hana.reasoning.validation import ReasoningValidator
from langchain_hana.reasoning.transformation import TransformationTracker
from langchain_hana.reasoning.metrics import InformationPreservationMetrics
from langchain_hana.reasoning.fingerprinting import InformationFingerprint

# Import feedback-related classes
from langchain_hana.reasoning.feedback import (
    UserFeedbackCollector,
    FeedbackItem,
    EmbeddingFeedback,
    RetrievalFeedback,
    ReasoningFeedback,
    TransformationFeedback,
    FeedbackStorage,
    InMemoryFeedbackStorage,
    HanaFeedbackStorage,
    FeedbackProcessor,
    FeedbackVisualization,
)

# Import transparent pipeline classes
from langchain_hana.reasoning.transparent_pipeline import (
    TransparentEmbeddingPipeline,
    TextPreprocessingStage,
    ModelEmbeddingStage,
    VectorPostprocessingStage,
    EmbeddingFingerprint,
)

# Import lineage-related classes
from langchain_hana.reasoning.data_lineage import (
    LineageTracker,
    LineageEvent,
    LineageGraph,
)

# Import factuality-related classes
from langchain_hana.reasoning.factuality import (
    SchemaFactualityBenchmark,
    FactualityEvaluator,
    HanaSchemaQuestionGenerator,
    AuditStyleEvaluator,
    FactualityGrade,
)

__all__ = [
    # Reasoning transparency
    "ReasoningPathTracker",
    "ReasoningValidator",
    "TransformationTracker",
    "InformationPreservationMetrics",
    "InformationFingerprint",
    
    # Feedback collection
    "UserFeedbackCollector",
    "FeedbackItem",
    "EmbeddingFeedback",
    "RetrievalFeedback",
    "ReasoningFeedback",
    "TransformationFeedback",
    "FeedbackStorage",
    "InMemoryFeedbackStorage",
    "HanaFeedbackStorage",
    "FeedbackProcessor",
    "FeedbackVisualization",
    
    # Transparent pipeline
    "TransparentEmbeddingPipeline",
    "TextPreprocessingStage",
    "ModelEmbeddingStage",
    "VectorPostprocessingStage",
    "EmbeddingFingerprint",
    
    # Data lineage
    "LineageTracker",
    "LineageEvent",
    "LineageGraph",
    
    # Factuality measurement
    "SchemaFactualityBenchmark",
    "FactualityEvaluator",
    "HanaSchemaQuestionGenerator",
    "AuditStyleEvaluator",
    "FactualityGrade",
]