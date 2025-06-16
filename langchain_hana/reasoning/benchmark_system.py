"""
Comprehensive benchmarking system for SAP HANA data representations.

This module provides tools for generating, storing, and analyzing benchmarks
that compare different data representation methods (relational, vector, OWL/ontology)
using question-answer pairs to measure factual accuracy.
"""

import uuid
import time
import json
import logging
import datetime
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable

import numpy as np
from hdbcli import dbapi
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.embeddings import Embeddings

from langchain_hana.reasoning.factuality import (
    SchemaFactualityBenchmark,
    FactualityEvaluator,
    FactualityGrade,
)

logger = logging.getLogger(__name__)


class RepresentationType(str, Enum):
    """Enumeration of data representation types."""
    RELATIONAL = "relational"
    VECTOR = "vector"
    ONTOLOGY = "ontology"
    HYBRID = "hybrid"


class QuestionType(str, Enum):
    """Enumeration of question types."""
    SCHEMA = "schema"
    INSTANCE = "instance"
    RELATIONSHIP = "relationship"
    AGGREGATION = "aggregation"
    INFERENCE = "inference"
    TEMPORAL = "temporal"
    

class DifficultyLevel(int, Enum):
    """
    Standardized difficulty levels for benchmark questions.
    
    This enumeration defines a 5-level scale for question difficulty with clear criteria:
    
    1. BEGINNER: Simple lookups, single entity, direct fact retrieval
       Example: "What is the primary key of the CUSTOMERS table?"
    
    2. BASIC: Simple aggregations, basic relationships, straightforward conditions
       Example: "How many customers are there in the database?"
    
    3. INTERMEDIATE: Multiple entities, basic analytics, simple joins
       Example: "Which product category has the most orders in 2023?"
    
    4. ADVANCED: Complex joins, advanced analytics, temporal reasoning
       Example: "What is the month-over-month growth rate of sales by product category?"
    
    5. EXPERT: Complex inferences, specialized knowledge, multiple steps
       Example: "Based on customer purchase patterns, which product bundles would likely increase sales?"
    """
    BEGINNER = 1
    BASIC = 2
    INTERMEDIATE = 3
    ADVANCED = 4
    EXPERT = 5
    
    # Pre-compile regular expression patterns for performance optimization
    # These patterns are used to detect complexity indicators in question text
    __entity_pattern = re.compile(r'\b(table|entity|column|field|tables|entities|columns|fields)\b')
    __condition_pattern = re.compile(r'\b(where|having|filter|if|when)\b|(\band\b|\bor\b|\bnot\b|\bexcept\b)')
    __function_pattern = re.compile(r'\b(count|sum|average|avg|max|min|total|mean)\b|\bnumber\s+of\b')
    __join_pattern = re.compile(r'\b(join|related|relationship|between|connect|link|association)\b')
    __subquery_pattern = re.compile(r'\b(subquery|sub-query|nested)\b|\bfor\s+each\b|\b(within|inside)\b')
    __domain_pattern = re.compile(r'\b(specific|special|domain|expert|complex)\b|\bbusiness\s+rule\b|\blogic\b')
    
    # Weights for different complexity indicators, used to adjust the base difficulty
    __complexity_weights = {
        "entities": {"threshold": 3, "adjustment": 1, "zero_penalty": -1},
        "conditions": {"threshold": 3, "adjustment": 1},
        "functions": {"threshold": 2, "adjustment": 1},
        "joins": {"threshold": 2, "adjustment": 1},
        "subqueries": {"threshold": 1, "adjustment": 2},
        "domain": {"threshold": 2, "adjustment": 1},
    }
    
    # Base difficulty map by question type
    __base_difficulty_map = {
        QuestionType.SCHEMA: 1,      # BEGINNER
        QuestionType.INSTANCE: 2,    # BASIC
        QuestionType.RELATIONSHIP: 3, # INTERMEDIATE
        QuestionType.AGGREGATION: 3,  # INTERMEDIATE
        QuestionType.TEMPORAL: 4,     # ADVANCED
        QuestionType.INFERENCE: 5,    # EXPERT
    }
    
    @classmethod
    def assess_difficulty(cls, question_text: str, question_type: QuestionType) -> "DifficultyLevel":
        """
        Assess difficulty of a question based on standardized criteria.
        
        This method implements a multi-factor algorithm that:
        1. Starts with a base difficulty determined by question type
        2. Analyzes text for complexity indicators using optimized regex patterns
        3. Adjusts the difficulty score based on detected indicators and their weights
        4. Ensures the final score stays within the valid range (1-5)
        
        The algorithm aims to provide consistent, objective assessments by counting
        specific terms related to database operations and analytical complexity.
        
        Args:
            question_text: The question text to analyze
            question_type: Type of question (SCHEMA, INSTANCE, etc.)
            
        Returns:
            Assessed difficulty level (1-5) as a DifficultyLevel enum value
            
        Examples:
            >>> DifficultyLevel.assess_difficulty(
            ...     "What is the primary key of the CUSTOMERS table?", 
            ...     QuestionType.SCHEMA
            ... )
            DifficultyLevel.BEGINNER  # Level 1
            
            >>> DifficultyLevel.assess_difficulty(
            ...     "How many orders were placed by each customer in Q3 2023?", 
            ...     QuestionType.AGGREGATION
            ... )
            DifficultyLevel.ADVANCED  # Level 4
        """
        # Performance optimization: return immediately for obvious cases
        if len(question_text) < 10:
            return DifficultyLevel(1)  # BEGINNER for very short questions
            
        # Convert to lowercase for case-insensitive analysis
        text = question_text.lower()
        
        # Count complexity indicators using optimized regex patterns
        indicators = {
            "entities": len(cls.__entity_pattern.findall(text)),
            "conditions": len(cls.__condition_pattern.findall(text)),
            "functions": len(cls.__function_pattern.findall(text)),
            "joins": len(cls.__join_pattern.findall(text)),
            "subqueries": len(cls.__subquery_pattern.findall(text)),
            "domain": len(cls.__domain_pattern.findall(text))
        }
        
        # Calculate base difficulty score from question type
        base_difficulty = cls.__base_difficulty_map.get(question_type, 3)  # Default to INTERMEDIATE
        difficulty_score = base_difficulty
        
        # Apply adjustments based on complexity indicators
        for indicator, count in indicators.items():
            weight_info = cls.__complexity_weights.get(indicator, {})
            threshold = weight_info.get("threshold", 0)
            adjustment = weight_info.get("adjustment", 0)
            
            if count >= threshold and threshold > 0:
                difficulty_score += adjustment
            elif count == 0 and weight_info.get("zero_penalty") is not None:
                difficulty_score += weight_info.get("zero_penalty", 0)
        
        # Ensure difficulty is within valid range (1-5)
        difficulty_score = max(1, min(5, difficulty_score))
        
        # Convert to enum value
        return DifficultyLevel(difficulty_score)
    
    @classmethod
    def get_difficulty_analysis(cls, question_text: str, question_type: QuestionType) -> Dict[str, Any]:
        """
        Perform detailed difficulty analysis and return explanation of the assessment.
        
        This method extends assess_difficulty by providing a detailed breakdown of:
        - Base difficulty from question type
        - Detected complexity indicators
        - Adjustments applied to the score
        - Final difficulty level with explanation
        
        This is useful for understanding how the algorithm arrived at a particular 
        difficulty rating and for auditing or debugging the assessment process.
        
        Args:
            question_text: The question text to analyze
            question_type: Type of question (SCHEMA, INSTANCE, etc.)
            
        Returns:
            Dictionary with detailed analysis results
        """
        # Convert to lowercase for analysis
        text = question_text.lower()
        
        # Count complexity indicators
        indicators = {
            "entities": len(cls.__entity_pattern.findall(text)),
            "conditions": len(cls.__condition_pattern.findall(text)),
            "functions": len(cls.__function_pattern.findall(text)),
            "joins": len(cls.__join_pattern.findall(text)),
            "subqueries": len(cls.__subquery_pattern.findall(text)),
            "domain": len(cls.__domain_pattern.findall(text))
        }
        
        # Get base difficulty from question type
        base_difficulty = cls.__base_difficulty_map.get(question_type, 3)
        
        # Initialize adjustments tracking
        adjustments = []
        difficulty_score = base_difficulty
        
        # Calculate adjustments
        for indicator, count in indicators.items():
            weight_info = cls.__complexity_weights.get(indicator, {})
            threshold = weight_info.get("threshold", 0)
            adjustment = weight_info.get("adjustment", 0)
            
            if count >= threshold and threshold > 0:
                difficulty_score += adjustment
                adjustments.append({
                    "indicator": indicator,
                    "count": count,
                    "threshold": threshold,
                    "adjustment": f"+{adjustment}",
                    "reason": f"{indicator.capitalize()} count ({count}) meets/exceeds threshold ({threshold})"
                })
            elif count == 0 and weight_info.get("zero_penalty") is not None:
                difficulty_score += weight_info.get("zero_penalty", 0)
                adjustments.append({
                    "indicator": indicator,
                    "count": count,
                    "threshold": "N/A",
                    "adjustment": f"{weight_info.get('zero_penalty')}",
                    "reason": f"No {indicator} found (zero penalty applied)"
                })
        
        # Apply bounds
        original_score = difficulty_score
        difficulty_score = max(1, min(5, difficulty_score))
        
        if original_score != difficulty_score:
            adjustments.append({
                "indicator": "bounds",
                "adjustment": f"→ {difficulty_score}",
                "reason": f"Applied min/max bounds (1-5) to raw score {original_score}"
            })
        
        # Prepare the final difficulty level with name
        final_difficulty = DifficultyLevel(difficulty_score)
        difficulty_name = final_difficulty.name
        
        # Build the explanation
        level_explanations = {
            1: "Simple direct lookups, single entity, factual recall",
            2: "Simple aggregations, basic relationships, straightforward questions",
            3: "Multiple entities, basic analytics, simple joins required",
            4: "Complex joins, advanced analytics, temporal reasoning",
            5: "Complex inferences, specialized knowledge, multiple analytical steps"
        }
        
        explanation = level_explanations.get(difficulty_score, "")
        
        return {
            "question_text": question_text,
            "question_type": question_type.value,
            "base_difficulty": {
                "score": base_difficulty,
                "reason": f"Base difficulty for {question_type.value} question type"
            },
            "complexity_indicators": indicators,
            "adjustments": adjustments,
            "final_difficulty": {
                "score": difficulty_score,
                "name": difficulty_name,
                "explanation": explanation
            },
            "assessment_version": "2.0",
            "assessment_timestamp": time.time()
        }


class AnswerSource(str, Enum):
    """Enumeration of answer sources."""
    DIRECT_QUERY = "direct_query"
    VECTOR_SEARCH = "vector_search"
    SPARQL = "sparql"
    LLM = "llm"
    HYBRID = "hybrid"


@dataclass
class QuestionProvenance:
    """Detailed provenance information for benchmark questions."""
    source_type: str  # "model", "human", "template", etc.
    source_id: str  # Model ID, user ID, template ID, etc.
    generation_timestamp: float
    generation_method: str  # How the question was generated
    generation_parameters: Dict[str, Any]  # Parameters used for generation
    input_data_sources: List[str]  # Data sources used to generate the question
    revision_history: List[Dict[str, Any]] = field(default_factory=list)  # History of changes
    
    def add_revision(self, editor: str, timestamp: float, reason: str, changes: Dict[str, Any]) -> None:
        """
        Add a revision to the history.
        
        Args:
            editor: Who made the change
            timestamp: When the change was made
            reason: Why the change was made
            changes: What was changed (old_value -> new_value)
        """
        self.revision_history.append({
            "editor": editor,
            "timestamp": timestamp,
            "reason": reason,
            "changes": changes
        })

@dataclass
class BenchmarkQuestion:
    """A question for benchmarking data representations."""
    question_id: str
    question_text: str
    reference_answer: str
    question_type: QuestionType
    entity_reference: str  # Table/entity the question references
    difficulty: int  # 1-5 scale
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    provenance: Optional[QuestionProvenance] = None


@dataclass
class BenchmarkAnswer:
    """An answer to a benchmark question from a specific representation type."""
    answer_id: str
    question_id: str
    representation_type: RepresentationType
    answer_source: AnswerSource
    answer_text: str
    grade: Optional[FactualityGrade] = None
    confidence: Optional[float] = None
    response_time_ms: Optional[int] = None
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataRepresentationMetrics:
    """Metrics for a data representation type."""
    representation_type: RepresentationType
    correct_count: int = 0
    incorrect_count: int = 0
    not_attempted_count: int = 0
    ambiguous_count: int = 0
    total_count: int = 0
    avg_response_time_ms: Optional[float] = None
    metrics_by_question_type: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metrics_by_difficulty: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    
    @property
    def accuracy(self) -> float:
        """Calculate the accuracy rate."""
        if self.total_count == 0:
            return 0.0
        return self.correct_count / self.total_count
    
    @property
    def attempted_accuracy(self) -> float:
        """Calculate the accuracy rate for attempted questions."""
        attempted = self.correct_count + self.incorrect_count + self.ambiguous_count
        if attempted == 0:
            return 0.0
        return self.correct_count / attempted
    
    @property
    def f_score(self) -> float:
        """Calculate the F-score (harmonic mean of accuracy and attempted accuracy)."""
        if self.accuracy == 0 or self.attempted_accuracy == 0:
            return 0.0
        return 2 * (self.accuracy * self.attempted_accuracy) / (self.accuracy + self.attempted_accuracy)


class HanaStorageManager:
    """
    Manager for storing and retrieving benchmark data in SAP HANA.
    
    This class provides methods for CRUD operations on benchmark questions,
    answers, and metrics in a SAP HANA database.
    """
    
    def __init__(
        self,
        connection: dbapi.Connection,
        schema_name: str,
        questions_table: str = "BENCHMARK_QUESTIONS",
        answers_table: str = "BENCHMARK_ANSWERS",
        results_table: str = "BENCHMARK_RESULTS",
    ):
        """
        Initialize the HANA storage manager.
        
        Args:
            connection: SAP HANA database connection
            schema_name: Database schema name
            questions_table: Name of the questions table
            answers_table: Name of the answers table
            results_table: Name of the results table
        """
        self.connection = connection
        self.schema_name = schema_name
        self.questions_table = questions_table
        self.answers_table = answers_table
        self.results_table = results_table
        
        # Initialize tables
        self._initialize_tables()
    
    def _initialize_tables(self) -> None:
        """Initialize the database tables."""
        try:
            cursor = self.connection.cursor()
            
            # Create schema if it doesn't exist
            try:
                cursor.execute(f"CREATE SCHEMA {self.schema_name}")
                logger.info(f"Created schema {self.schema_name}")
            except Exception as e:
                # Ignore error if schema already exists
                if "exists" not in str(e).lower():
                    logger.warning(f"Error creating schema: {str(e)}")
            
            # Create questions table
            questions_table = f"{self.schema_name}.{self.questions_table}"
            cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {questions_table} (
                QUESTION_ID VARCHAR(100) PRIMARY KEY,
                QUESTION_TEXT NVARCHAR(2000) NOT NULL,
                REFERENCE_ANSWER NVARCHAR(2000) NOT NULL,
                QUESTION_TYPE VARCHAR(50) NOT NULL,
                ENTITY_REFERENCE VARCHAR(200) NOT NULL,
                DIFFICULTY INTEGER NOT NULL,
                CREATED_AT TIMESTAMP NOT NULL,
                METADATA NCLOB,
                PROVENANCE NCLOB
            )
            """)
            
            # Create provenance tracking table for questions
            provenance_table = f"{self.schema_name}.QUESTION_PROVENANCE"
            cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {provenance_table} (
                PROVENANCE_ID VARCHAR(100) PRIMARY KEY,
                QUESTION_ID VARCHAR(100) NOT NULL,
                SOURCE_TYPE VARCHAR(50) NOT NULL,
                SOURCE_ID VARCHAR(200) NOT NULL,
                GENERATION_TIMESTAMP TIMESTAMP NOT NULL,
                GENERATION_METHOD VARCHAR(200) NOT NULL,
                GENERATION_PARAMETERS NCLOB,
                INPUT_DATA_SOURCES NCLOB,
                REVISION_HISTORY NCLOB,
                CREATED_AT TIMESTAMP NOT NULL,
                FOREIGN KEY (QUESTION_ID) REFERENCES {questions_table}(QUESTION_ID)
            )
            """)
            
            # Create answers table
            answers_table = f"{self.schema_name}.{self.answers_table}"
            cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {answers_table} (
                ANSWER_ID VARCHAR(100) PRIMARY KEY,
                QUESTION_ID VARCHAR(100) NOT NULL,
                REPRESENTATION_TYPE VARCHAR(50) NOT NULL,
                ANSWER_SOURCE VARCHAR(50) NOT NULL,
                ANSWER_TEXT NVARCHAR(2000) NOT NULL,
                GRADE VARCHAR(20),
                CONFIDENCE FLOAT,
                RESPONSE_TIME_MS INTEGER,
                CREATED_AT TIMESTAMP NOT NULL,
                METADATA NCLOB,
                PROVENANCE NCLOB,
                FOREIGN KEY (QUESTION_ID) REFERENCES {questions_table}(QUESTION_ID)
            )
            """)
            
            # Create results table
            results_table = f"{self.schema_name}.{self.results_table}"
            cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {results_table} (
                RESULT_ID VARCHAR(100) PRIMARY KEY,
                BENCHMARK_ID VARCHAR(100) NOT NULL,
                BENCHMARK_NAME VARCHAR(200) NOT NULL,
                REPRESENTATION_TYPE VARCHAR(50) NOT NULL,
                RUN_DATE TIMESTAMP NOT NULL,
                TOTAL_QUESTIONS INTEGER NOT NULL,
                CORRECT_COUNT INTEGER NOT NULL,
                INCORRECT_COUNT INTEGER NOT NULL,
                NOT_ATTEMPTED_COUNT INTEGER NOT NULL,
                AMBIGUOUS_COUNT INTEGER NOT NULL,
                ACCURACY FLOAT NOT NULL,
                F_SCORE FLOAT NOT NULL,
                AVG_RESPONSE_TIME_MS FLOAT,
                DETAILED_METRICS NCLOB,
                METADATA NCLOB,
                PROVENANCE NCLOB
            )
            """)
            
            # Create indices
            try:
                cursor.execute(f"CREATE INDEX IDX_{self.questions_table}_TYPE ON {questions_table}(QUESTION_TYPE)")
                cursor.execute(f"CREATE INDEX IDX_{self.questions_table}_ENTITY ON {questions_table}(ENTITY_REFERENCE)")
                cursor.execute(f"CREATE INDEX IDX_{self.questions_table}_DIFFICULTY ON {questions_table}(DIFFICULTY)")
                
                cursor.execute(f"CREATE INDEX IDX_QUESTION_PROVENANCE_QID ON {provenance_table}(QUESTION_ID)")
                cursor.execute(f"CREATE INDEX IDX_QUESTION_PROVENANCE_SRC ON {provenance_table}(SOURCE_TYPE, SOURCE_ID)")
                cursor.execute(f"CREATE INDEX IDX_QUESTION_PROVENANCE_METHOD ON {provenance_table}(GENERATION_METHOD)")
                
                cursor.execute(f"CREATE INDEX IDX_{self.answers_table}_QUESTION ON {answers_table}(QUESTION_ID)")
                cursor.execute(f"CREATE INDEX IDX_{self.answers_table}_TYPE ON {answers_table}(REPRESENTATION_TYPE)")
                cursor.execute(f"CREATE INDEX IDX_{self.answers_table}_GRADE ON {answers_table}(GRADE)")
                
                cursor.execute(f"CREATE INDEX IDX_{self.results_table}_BENCHMARK ON {results_table}(BENCHMARK_ID)")
                cursor.execute(f"CREATE INDEX IDX_{self.results_table}_TYPE ON {results_table}(REPRESENTATION_TYPE)")
            except Exception as e:
                # Ignore error if indices already exist
                if "exists" not in str(e).lower():
                    logger.warning(f"Error creating indices: {str(e)}")
            
            logger.info(f"Initialized database tables in schema {self.schema_name}")
        
        except Exception as e:
            logger.error(f"Error initializing database tables: {str(e)}")
            raise
        finally:
            cursor.close()
    
    def create_question(self, question: BenchmarkQuestion) -> None:
        """
        Create a new benchmark question with provenance tracking.
        
        Args:
            question: The question to create with provenance information
            
        Raises:
            ValueError: If validation fails
        """
        # Validate question before storage
        self._validate_question_for_storage(question)
        
        try:
            cursor = self.connection.cursor()
            
            questions_table = f"{self.schema_name}.{self.questions_table}"
            provenance_table = f"{self.schema_name}.QUESTION_PROVENANCE"
            
            # Check for existing question with same ID
            cursor.execute(f"""
            SELECT COUNT(*) FROM {questions_table}
            WHERE QUESTION_ID = ?
            """, (question.question_id,))
            
            count = cursor.fetchone()[0]
            if count > 0:
                raise ValueError(f"Question with ID {question.question_id} already exists")
            
            # Serialize metadata to JSON
            # Add validation timestamp
            metadata = dict(question.metadata)
            metadata["storage_validation"] = {
                "validated_at": time.time(),
                "validation_version": "1.0",
            }
            metadata_json = json.dumps(metadata)
            
            # Sanitize inputs (beyond the validation already done)
            question_text = self._sanitize_for_storage(question.question_text)
            reference_answer = self._sanitize_for_storage(question.reference_answer)
            entity_reference = self._sanitize_for_storage(question.entity_reference)
            
            # Serialize provenance to JSON if exists
            provenance_json = None
            if question.provenance:
                provenance_json = json.dumps(asdict(question.provenance))
            
            # Insert the question
            cursor.execute(f"""
            INSERT INTO {questions_table} (
                QUESTION_ID, QUESTION_TEXT, REFERENCE_ANSWER, QUESTION_TYPE,
                ENTITY_REFERENCE, DIFFICULTY, CREATED_AT, METADATA, PROVENANCE
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                question.question_id,
                question_text,
                reference_answer,
                question.question_type.value,
                entity_reference,
                question.difficulty,
                datetime.datetime.fromtimestamp(question.created_at),
                metadata_json,
                provenance_json,
            ))
            
            # Store detailed provenance information in dedicated table if provided
            if question.provenance:
                provenance_id = str(uuid.uuid4())
                
                # Serialize complex fields to JSON
                generation_parameters_json = json.dumps(question.provenance.generation_parameters)
                input_data_sources_json = json.dumps(question.provenance.input_data_sources)
                revision_history_json = json.dumps(question.provenance.revision_history)
                
                cursor.execute(f"""
                INSERT INTO {provenance_table} (
                    PROVENANCE_ID, QUESTION_ID, SOURCE_TYPE, SOURCE_ID,
                    GENERATION_TIMESTAMP, GENERATION_METHOD, GENERATION_PARAMETERS,
                    INPUT_DATA_SOURCES, REVISION_HISTORY, CREATED_AT
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    provenance_id,
                    question.question_id,
                    question.provenance.source_type,
                    question.provenance.source_id,
                    datetime.datetime.fromtimestamp(question.provenance.generation_timestamp),
                    question.provenance.generation_method,
                    generation_parameters_json,
                    input_data_sources_json,
                    revision_history_json,
                    datetime.datetime.now(),
                ))
                
                logger.info(f"Created provenance record {provenance_id} for question {question.question_id}")
            
            self.connection.commit()
            logger.info(f"Created question {question.question_id}")
        
        except Exception as e:
            logger.error(f"Error creating question: {str(e)}")
            self.connection.rollback()
            raise
        finally:
            cursor.close()
            
    def _validate_question_for_storage(self, question: BenchmarkQuestion) -> None:
        """
        Validate a question before storage.
        
        Args:
            question: The question to validate
            
        Raises:
            ValueError: If validation fails
        """
        # Check required fields
        if not question.question_id:
            raise ValueError("Question ID is required")
            
        if not question.question_text or len(question.question_text.strip()) < 10:
            raise ValueError("Question text is required and must be at least 10 characters")
            
        if not question.reference_answer or len(question.reference_answer.strip()) < 2:
            raise ValueError("Reference answer is required and must be at least 2 characters")
            
        if not question.entity_reference:
            raise ValueError("Entity reference is required")
            
        # Validate UUID format for question_id
        try:
            uuid_obj = uuid.UUID(question.question_id)
            if str(uuid_obj) != question.question_id:
                raise ValueError("Invalid UUID format for question_id")
        except ValueError:
            raise ValueError("Question ID must be a valid UUID")
            
        # Validate question type
        valid_question_types = [qt.value for qt in QuestionType]
        if question.question_type.value not in valid_question_types:
            raise ValueError(f"Invalid question type: {question.question_type}")
            
        # Validate difficulty level
        if not isinstance(question.difficulty, int) or question.difficulty < 1 or question.difficulty > 5:
            raise ValueError(f"Difficulty must be an integer between 1 and 5, got: {question.difficulty}")
            
        # Validate created_at timestamp
        if not isinstance(question.created_at, (int, float)):
            raise ValueError("Created at timestamp must be a number")
            
        if question.created_at > time.time() + 86400:  # No future dates (allow 1 day for clock skew)
            raise ValueError("Created at timestamp cannot be in the future")
            
        if question.created_at < 946684800:  # No dates before 2000-01-01
            raise ValueError("Created at timestamp is too old")
            
        # Validate metadata is serializable
        try:
            json.dumps(question.metadata)
        except (TypeError, OverflowError):
            raise ValueError("Metadata must be JSON serializable")
        
    def _sanitize_for_storage(self, text: str) -> str:
        """
        Sanitize text for storage.
        
        Args:
            text: The text to sanitize
            
        Returns:
            Sanitized text
        """
        if not text:
            return ""
            
        # Remove null bytes and other control characters (except newlines and tabs)
        text = ''.join(c for c in text if c == '\n' or c == '\t' or (ord(c) >= 32 and ord(c) <= 126))
        
        # Limit length to prevent excessive storage
        max_length = 1990  # Just under the 2000 NVARCHAR limit
        if len(text) > max_length:
            text = text[:max_length] + "..."
            
        return text
    
    def get_question(self, question_id: str) -> Optional[BenchmarkQuestion]:
        """
        Get a benchmark question by ID with provenance information.
        
        Args:
            question_id: ID of the question
            
        Returns:
            The question or None if not found
        """
        try:
            cursor = self.connection.cursor()
            
            questions_table = f"{self.schema_name}.{self.questions_table}"
            provenance_table = f"{self.schema_name}.QUESTION_PROVENANCE"
            
            # Query the question
            cursor.execute(f"""
            SELECT QUESTION_ID, QUESTION_TEXT, REFERENCE_ANSWER, QUESTION_TYPE,
                   ENTITY_REFERENCE, DIFFICULTY, CREATED_AT, METADATA, PROVENANCE
            FROM {questions_table}
            WHERE QUESTION_ID = ?
            """, (question_id,))
            
            row = cursor.fetchone()
            
            if not row:
                return None
            
            # Parse metadata from JSON
            metadata = json.loads(row[7]) if row[7] else {}
            
            # Parse provenance from JSON
            provenance = None
            if row[8]:
                try:
                    provenance_dict = json.loads(row[8])
                    
                    # Check for revision_history field format
                    if isinstance(provenance_dict.get("revision_history"), list):
                        # Create QuestionProvenance object
                        provenance = QuestionProvenance(
                            source_type=provenance_dict["source_type"],
                            source_id=provenance_dict["source_id"],
                            generation_timestamp=provenance_dict["generation_timestamp"],
                            generation_method=provenance_dict["generation_method"],
                            generation_parameters=provenance_dict["generation_parameters"],
                            input_data_sources=provenance_dict["input_data_sources"],
                            revision_history=provenance_dict["revision_history"],
                        )
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Error parsing provenance for question {question_id}: {str(e)}")
            
            # If no provenance in main table, check the dedicated provenance table
            if not provenance:
                try:
                    cursor.execute(f"""
                    SELECT SOURCE_TYPE, SOURCE_ID, GENERATION_TIMESTAMP, GENERATION_METHOD,
                           GENERATION_PARAMETERS, INPUT_DATA_SOURCES, REVISION_HISTORY
                    FROM {provenance_table}
                    WHERE QUESTION_ID = ?
                    ORDER BY CREATED_AT DESC
                    LIMIT 1
                    """, (question_id,))
                    
                    prov_row = cursor.fetchone()
                    
                    if prov_row:
                        # Parse JSON fields
                        generation_parameters = json.loads(prov_row[4]) if prov_row[4] else {}
                        input_data_sources = json.loads(prov_row[5]) if prov_row[5] else []
                        revision_history = json.loads(prov_row[6]) if prov_row[6] else []
                        
                        # Create QuestionProvenance object
                        provenance = QuestionProvenance(
                            source_type=prov_row[0],
                            source_id=prov_row[1],
                            generation_timestamp=prov_row[2].timestamp(),
                            generation_method=prov_row[3],
                            generation_parameters=generation_parameters,
                            input_data_sources=input_data_sources,
                            revision_history=revision_history,
                        )
                except Exception as e:
                    logger.warning(f"Error retrieving provenance from dedicated table for question {question_id}: {str(e)}")
            
            # Create and return the question
            return BenchmarkQuestion(
                question_id=row[0],
                question_text=row[1],
                reference_answer=row[2],
                question_type=QuestionType(row[3]),
                entity_reference=row[4],
                difficulty=row[5],
                created_at=row[6].timestamp(),
                metadata=metadata,
                provenance=provenance,
            )
        
        except Exception as e:
            logger.error(f"Error getting question: {str(e)}")
            raise
        finally:
            cursor.close()
    
    def update_question(self, question: BenchmarkQuestion) -> None:
        """
        Update a benchmark question with provenance tracking.
        
        Args:
            question: The question to update
            
        Notes:
            This method automatically tracks the changes in the provenance history
            if provenance tracking is enabled for this question.
        """
        try:
            cursor = self.connection.cursor()
            
            questions_table = f"{self.schema_name}.{self.questions_table}"
            provenance_table = f"{self.schema_name}.QUESTION_PROVENANCE"
            
            # Get the current state of the question to detect changes
            cursor.execute(f"""
            SELECT QUESTION_TEXT, REFERENCE_ANSWER, QUESTION_TYPE, ENTITY_REFERENCE, 
                   DIFFICULTY, METADATA, PROVENANCE
            FROM {questions_table}
            WHERE QUESTION_ID = ?
            """, (question.question_id,))
            
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Question {question.question_id} not found")
            
            # Detect changes for provenance tracking
            changes = {}
            if row[0] != question.question_text:
                changes["question_text"] = {"old": row[0], "new": question.question_text}
            if row[1] != question.reference_answer:
                changes["reference_answer"] = {"old": row[1], "new": question.reference_answer}
            if row[2] != question.question_type.value:
                changes["question_type"] = {"old": row[2], "new": question.question_type.value}
            if row[3] != question.entity_reference:
                changes["entity_reference"] = {"old": row[3], "new": question.entity_reference}
            if row[4] != question.difficulty:
                changes["difficulty"] = {"old": row[4], "new": question.difficulty}
            
            # Update the question's provenance if it exists
            existing_provenance = None
            if row[6]:
                try:
                    existing_provenance = json.loads(row[6])
                except json.JSONDecodeError:
                    pass
            
            # If the question has provenance tracking and there are changes, update the revision history
            if question.provenance and changes:
                if not hasattr(question.provenance, "revision_history"):
                    question.provenance.revision_history = []
                
                # Add the new revision
                question.provenance.add_revision(
                    editor="storage_manager",
                    timestamp=time.time(),
                    reason="update_operation",
                    changes=changes
                )
                
                # Serialize the updated provenance
                provenance_json = json.dumps(asdict(question.provenance))
                
                # Store the detailed provenance in the dedicated table
                provenance_id = str(uuid.uuid4())
                
                # Serialize complex fields to JSON
                generation_parameters_json = json.dumps(question.provenance.generation_parameters)
                input_data_sources_json = json.dumps(question.provenance.input_data_sources)
                revision_history_json = json.dumps(question.provenance.revision_history)
                
                cursor.execute(f"""
                INSERT INTO {provenance_table} (
                    PROVENANCE_ID, QUESTION_ID, SOURCE_TYPE, SOURCE_ID,
                    GENERATION_TIMESTAMP, GENERATION_METHOD, GENERATION_PARAMETERS,
                    INPUT_DATA_SOURCES, REVISION_HISTORY, CREATED_AT
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    provenance_id,
                    question.question_id,
                    question.provenance.source_type,
                    question.provenance.source_id,
                    datetime.datetime.fromtimestamp(question.provenance.generation_timestamp),
                    question.provenance.generation_method,
                    generation_parameters_json,
                    input_data_sources_json,
                    revision_history_json,
                    datetime.datetime.now(),
                ))
                
                logger.info(f"Created updated provenance record {provenance_id} for question {question.question_id}")
            else:
                # No provenance or no changes to track
                provenance_json = row[6]  # Keep existing provenance
            
            # Serialize metadata to JSON
            metadata_json = json.dumps(question.metadata)
            
            # Update the question
            cursor.execute(f"""
            UPDATE {questions_table}
            SET QUESTION_TEXT = ?,
                REFERENCE_ANSWER = ?,
                QUESTION_TYPE = ?,
                ENTITY_REFERENCE = ?,
                DIFFICULTY = ?,
                METADATA = ?,
                PROVENANCE = ?
            WHERE QUESTION_ID = ?
            """, (
                question.question_text,
                question.reference_answer,
                question.question_type.value,
                question.entity_reference,
                question.difficulty,
                metadata_json,
                provenance_json,
                question.question_id,
            ))
            
            self.connection.commit()
            logger.info(f"Updated question {question.question_id}")
        
        except Exception as e:
            logger.error(f"Error updating question: {str(e)}")
            self.connection.rollback()
            raise
        finally:
            cursor.close()
    
    def delete_question(self, question_id: str) -> None:
        """
        Delete a benchmark question.
        
        Args:
            question_id: ID of the question to delete
        """
        try:
            cursor = self.connection.cursor()
            
            # First delete related answers
            answers_table = f"{self.schema_name}.{self.answers_table}"
            cursor.execute(f"""
            DELETE FROM {answers_table}
            WHERE QUESTION_ID = ?
            """, (question_id,))
            
            # Then delete the question
            questions_table = f"{self.schema_name}.{self.questions_table}"
            cursor.execute(f"""
            DELETE FROM {questions_table}
            WHERE QUESTION_ID = ?
            """, (question_id,))
            
            if cursor.rowcount == 0:
                raise ValueError(f"Question {question_id} not found")
            
            self.connection.commit()
            logger.info(f"Deleted question {question_id}")
        
        except Exception as e:
            logger.error(f"Error deleting question: {str(e)}")
            self.connection.rollback()
            raise
        finally:
            cursor.close()
    
    def get_question_provenance_history(self, question_id: str) -> List[Dict[str, Any]]:
        """
        Get the complete provenance history for a question.
        
        Args:
            question_id: ID of the question
            
        Returns:
            List of provenance records in chronological order
        """
        try:
            cursor = self.connection.cursor()
            
            provenance_table = f"{self.schema_name}.QUESTION_PROVENANCE"
            
            # Query all provenance records for this question
            cursor.execute(f"""
            SELECT PROVENANCE_ID, SOURCE_TYPE, SOURCE_ID, GENERATION_TIMESTAMP,
                   GENERATION_METHOD, GENERATION_PARAMETERS, INPUT_DATA_SOURCES,
                   REVISION_HISTORY, CREATED_AT
            FROM {provenance_table}
            WHERE QUESTION_ID = ?
            ORDER BY CREATED_AT ASC
            """, (question_id,))
            
            # Process results
            provenance_history = []
            for row in cursor.fetchall():
                # Parse JSON fields
                generation_parameters = json.loads(row[5]) if row[5] else {}
                input_data_sources = json.loads(row[6]) if row[6] else []
                revision_history = json.loads(row[7]) if row[7] else []
                
                # Create and add the provenance record
                provenance_record = {
                    "provenance_id": row[0],
                    "source_type": row[1],
                    "source_id": row[2],
                    "generation_timestamp": row[3].isoformat() if row[3] else None,
                    "generation_method": row[4],
                    "generation_parameters": generation_parameters,
                    "input_data_sources": input_data_sources,
                    "revision_history": revision_history,
                    "created_at": row[8].isoformat() if row[8] else None,
                }
                provenance_history.append(provenance_record)
            
            return provenance_history
        
        except Exception as e:
            logger.error(f"Error getting provenance history: {str(e)}")
            raise
        finally:
            cursor.close()
    
    def search_questions(
        self,
        question_type: Optional[QuestionType] = None,
        entity_reference: Optional[str] = None,
        difficulty: Optional[int] = None,
        source_type: Optional[str] = None,
        source_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[BenchmarkQuestion]:
        """
        Search for benchmark questions with optional provenance filters.
        
        Args:
            question_type: Optional question type filter
            entity_reference: Optional entity reference filter
            difficulty: Optional difficulty filter
            source_type: Optional source type filter (for provenance)
            source_id: Optional source ID filter (for provenance)
            limit: Maximum number of questions to return
            
        Returns:
            List of matching questions with provenance information
        """
        try:
            cursor = self.connection.cursor()
            
            questions_table = f"{self.schema_name}.{self.questions_table}"
            provenance_table = f"{self.schema_name}.QUESTION_PROVENANCE"
            
            # Determine if we need to join with provenance table
            need_provenance_join = source_type is not None or source_id is not None
            
            # Base query
            if need_provenance_join:
                # Join with provenance table for filtering by source
                query = f"""
                SELECT q.QUESTION_ID, q.QUESTION_TEXT, q.REFERENCE_ANSWER, q.QUESTION_TYPE,
                       q.ENTITY_REFERENCE, q.DIFFICULTY, q.CREATED_AT, q.METADATA, q.PROVENANCE
                FROM {questions_table} q
                JOIN {provenance_table} p ON q.QUESTION_ID = p.QUESTION_ID
                WHERE 1=1
                """
            else:
                # No need to join if not filtering by source
                query = f"""
                SELECT QUESTION_ID, QUESTION_TEXT, REFERENCE_ANSWER, QUESTION_TYPE,
                       ENTITY_REFERENCE, DIFFICULTY, CREATED_AT, METADATA, PROVENANCE
                FROM {questions_table}
                WHERE 1=1
                """
            
            params = []
            
            # Add filters
            if question_type:
                if need_provenance_join:
                    query += " AND q.QUESTION_TYPE = ?"
                else:
                    query += " AND QUESTION_TYPE = ?"
                params.append(question_type.value)
            
            if entity_reference:
                if need_provenance_join:
                    query += " AND q.ENTITY_REFERENCE = ?"
                else:
                    query += " AND ENTITY_REFERENCE = ?"
                params.append(entity_reference)
            
            if difficulty:
                if need_provenance_join:
                    query += " AND q.DIFFICULTY = ?"
                else:
                    query += " AND DIFFICULTY = ?"
                params.append(difficulty)
            
            # Add provenance filters
            if source_type:
                query += " AND p.SOURCE_TYPE = ?"
                params.append(source_type)
            
            if source_id:
                query += " AND p.SOURCE_ID = ?"
                params.append(source_id)
            
            # Add ordering and limit
            if need_provenance_join:
                query += " ORDER BY q.CREATED_AT DESC LIMIT ?"
            else:
                query += " ORDER BY CREATED_AT DESC LIMIT ?"
            params.append(limit)
            
            # Execute query
            cursor.execute(query, tuple(params))
            
            # Process results
            questions = []
            for row in cursor.fetchall():
                # Parse metadata from JSON
                metadata = json.loads(row[7]) if row[7] else {}
                
                # Parse provenance from JSON
                provenance = None
                if row[8]:
                    try:
                        provenance_dict = json.loads(row[8])
                        
                        # Check for revision_history field format
                        if isinstance(provenance_dict.get("revision_history"), list):
                            # Create QuestionProvenance object
                            provenance = QuestionProvenance(
                                source_type=provenance_dict["source_type"],
                                source_id=provenance_dict["source_id"],
                                generation_timestamp=provenance_dict["generation_timestamp"],
                                generation_method=provenance_dict["generation_method"],
                                generation_parameters=provenance_dict["generation_parameters"],
                                input_data_sources=provenance_dict["input_data_sources"],
                                revision_history=provenance_dict["revision_history"],
                            )
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Error parsing provenance for question {row[0]}: {str(e)}")
                
                # Create and add the question
                question = BenchmarkQuestion(
                    question_id=row[0],
                    question_text=row[1],
                    reference_answer=row[2],
                    question_type=QuestionType(row[3]),
                    entity_reference=row[4],
                    difficulty=row[5],
                    created_at=row[6].timestamp(),
                    metadata=metadata,
                    provenance=provenance,
                )
                questions.append(question)
            
            return questions
        
        except Exception as e:
            logger.error(f"Error searching questions: {str(e)}")
            raise
        finally:
            cursor.close()
    
    def create_answer(self, answer: BenchmarkAnswer) -> None:
        """
        Create a new benchmark answer.
        
        Args:
            answer: The answer to create
        """
        try:
            cursor = self.connection.cursor()
            
            answers_table = f"{self.schema_name}.{self.answers_table}"
            
            # Serialize metadata to JSON
            metadata_json = json.dumps(answer.metadata)
            
            # Insert the answer
            cursor.execute(f"""
            INSERT INTO {answers_table} (
                ANSWER_ID, QUESTION_ID, REPRESENTATION_TYPE, ANSWER_SOURCE,
                ANSWER_TEXT, GRADE, CONFIDENCE, RESPONSE_TIME_MS,
                CREATED_AT, METADATA
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                answer.answer_id,
                answer.question_id,
                answer.representation_type.value,
                answer.answer_source.value,
                answer.answer_text,
                answer.grade.value if answer.grade else None,
                answer.confidence,
                answer.response_time_ms,
                datetime.datetime.fromtimestamp(answer.created_at),
                metadata_json,
            ))
            
            self.connection.commit()
            logger.info(f"Created answer {answer.answer_id}")
        
        except Exception as e:
            logger.error(f"Error creating answer: {str(e)}")
            self.connection.rollback()
            raise
        finally:
            cursor.close()
    
    def get_answer(self, answer_id: str) -> Optional[BenchmarkAnswer]:
        """
        Get a benchmark answer by ID.
        
        Args:
            answer_id: ID of the answer
            
        Returns:
            The answer or None if not found
        """
        try:
            cursor = self.connection.cursor()
            
            answers_table = f"{self.schema_name}.{self.answers_table}"
            
            # Query the answer
            cursor.execute(f"""
            SELECT ANSWER_ID, QUESTION_ID, REPRESENTATION_TYPE, ANSWER_SOURCE,
                   ANSWER_TEXT, GRADE, CONFIDENCE, RESPONSE_TIME_MS,
                   CREATED_AT, METADATA
            FROM {answers_table}
            WHERE ANSWER_ID = ?
            """, (answer_id,))
            
            row = cursor.fetchone()
            
            if not row:
                return None
            
            # Parse metadata from JSON
            metadata = json.loads(row[9]) if row[9] else {}
            
            # Create and return the answer
            return BenchmarkAnswer(
                answer_id=row[0],
                question_id=row[1],
                representation_type=RepresentationType(row[2]),
                answer_source=AnswerSource(row[3]),
                answer_text=row[4],
                grade=FactualityGrade(row[5]) if row[5] else None,
                confidence=row[6],
                response_time_ms=row[7],
                created_at=row[8].timestamp(),
                metadata=metadata,
            )
        
        except Exception as e:
            logger.error(f"Error getting answer: {str(e)}")
            raise
        finally:
            cursor.close()
    
    def update_answer(self, answer: BenchmarkAnswer) -> None:
        """
        Update a benchmark answer.
        
        Args:
            answer: The answer to update
        """
        try:
            cursor = self.connection.cursor()
            
            answers_table = f"{self.schema_name}.{self.answers_table}"
            
            # Serialize metadata to JSON
            metadata_json = json.dumps(answer.metadata)
            
            # Update the answer
            cursor.execute(f"""
            UPDATE {answers_table}
            SET ANSWER_TEXT = ?,
                GRADE = ?,
                CONFIDENCE = ?,
                RESPONSE_TIME_MS = ?,
                METADATA = ?
            WHERE ANSWER_ID = ?
            """, (
                answer.answer_text,
                answer.grade.value if answer.grade else None,
                answer.confidence,
                answer.response_time_ms,
                metadata_json,
                answer.answer_id,
            ))
            
            if cursor.rowcount == 0:
                raise ValueError(f"Answer {answer.answer_id} not found")
            
            self.connection.commit()
            logger.info(f"Updated answer {answer.answer_id}")
        
        except Exception as e:
            logger.error(f"Error updating answer: {str(e)}")
            self.connection.rollback()
            raise
        finally:
            cursor.close()
    
    def delete_answer(self, answer_id: str) -> None:
        """
        Delete a benchmark answer.
        
        Args:
            answer_id: ID of the answer to delete
        """
        try:
            cursor = self.connection.cursor()
            
            answers_table = f"{self.schema_name}.{self.answers_table}"
            
            # Delete the answer
            cursor.execute(f"""
            DELETE FROM {answers_table}
            WHERE ANSWER_ID = ?
            """, (answer_id,))
            
            if cursor.rowcount == 0:
                raise ValueError(f"Answer {answer_id} not found")
            
            self.connection.commit()
            logger.info(f"Deleted answer {answer_id}")
        
        except Exception as e:
            logger.error(f"Error deleting answer: {str(e)}")
            self.connection.rollback()
            raise
        finally:
            cursor.close()
    
    def get_answers_for_question(
        self,
        question_id: str,
        representation_type: Optional[RepresentationType] = None,
    ) -> List[BenchmarkAnswer]:
        """
        Get answers for a specific question.
        
        Args:
            question_id: ID of the question
            representation_type: Optional representation type filter
            
        Returns:
            List of answers for the question
        """
        try:
            cursor = self.connection.cursor()
            
            answers_table = f"{self.schema_name}.{self.answers_table}"
            
            # Build query
            query = f"""
            SELECT ANSWER_ID, QUESTION_ID, REPRESENTATION_TYPE, ANSWER_SOURCE,
                   ANSWER_TEXT, GRADE, CONFIDENCE, RESPONSE_TIME_MS,
                   CREATED_AT, METADATA
            FROM {answers_table}
            WHERE QUESTION_ID = ?
            """
            
            params = [question_id]
            
            if representation_type:
                query += " AND REPRESENTATION_TYPE = ?"
                params.append(representation_type.value)
            
            query += " ORDER BY CREATED_AT DESC"
            
            # Execute query
            cursor.execute(query, tuple(params))
            
            # Process results
            answers = []
            for row in cursor.fetchall():
                # Parse metadata from JSON
                metadata = json.loads(row[9]) if row[9] else {}
                
                # Create and add the answer
                answer = BenchmarkAnswer(
                    answer_id=row[0],
                    question_id=row[1],
                    representation_type=RepresentationType(row[2]),
                    answer_source=AnswerSource(row[3]),
                    answer_text=row[4],
                    grade=FactualityGrade(row[5]) if row[5] else None,
                    confidence=row[6],
                    response_time_ms=row[7],
                    created_at=row[8].timestamp(),
                    metadata=metadata,
                )
                answers.append(answer)
            
            return answers
        
        except Exception as e:
            logger.error(f"Error getting answers for question: {str(e)}")
            raise
        finally:
            cursor.close()
    
    def store_benchmark_result(
        self,
        benchmark_id: str,
        benchmark_name: str,
        representation_type: RepresentationType,
        metrics: DataRepresentationMetrics,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Store benchmark results.
        
        Args:
            benchmark_id: ID of the benchmark
            benchmark_name: Name of the benchmark
            representation_type: Type of data representation
            metrics: Metrics for the benchmark
            metadata: Optional metadata
            
        Returns:
            ID of the stored result
        """
        try:
            cursor = self.connection.cursor()
            
            results_table = f"{self.schema_name}.{self.results_table}"
            
            # Generate result ID
            result_id = str(uuid.uuid4())
            
            # Serialize detailed metrics and metadata to JSON
            detailed_metrics = {
                "by_question_type": metrics.metrics_by_question_type,
                "by_difficulty": metrics.metrics_by_difficulty,
            }
            detailed_metrics_json = json.dumps(detailed_metrics)
            metadata_json = json.dumps(metadata or {})
            
            # Insert the result
            cursor.execute(f"""
            INSERT INTO {results_table} (
                RESULT_ID, BENCHMARK_ID, BENCHMARK_NAME, REPRESENTATION_TYPE,
                RUN_DATE, TOTAL_QUESTIONS, CORRECT_COUNT, INCORRECT_COUNT,
                NOT_ATTEMPTED_COUNT, AMBIGUOUS_COUNT, ACCURACY, F_SCORE,
                AVG_RESPONSE_TIME_MS, DETAILED_METRICS, METADATA
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result_id,
                benchmark_id,
                benchmark_name,
                representation_type.value,
                datetime.datetime.now(),
                metrics.total_count,
                metrics.correct_count,
                metrics.incorrect_count,
                metrics.not_attempted_count,
                metrics.ambiguous_count,
                metrics.accuracy,
                metrics.f_score,
                metrics.avg_response_time_ms,
                detailed_metrics_json,
                metadata_json,
            ))
            
            self.connection.commit()
            logger.info(f"Stored benchmark result {result_id}")
            
            return result_id
        
        except Exception as e:
            logger.error(f"Error storing benchmark result: {str(e)}")
            self.connection.rollback()
            raise
        finally:
            cursor.close()
    
    def get_benchmark_results(
        self,
        benchmark_id: Optional[str] = None,
        representation_type: Optional[RepresentationType] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get benchmark results.
        
        Args:
            benchmark_id: Optional benchmark ID filter
            representation_type: Optional representation type filter
            limit: Maximum number of results to return
            
        Returns:
            List of benchmark results
        """
        try:
            cursor = self.connection.cursor()
            
            results_table = f"{self.schema_name}.{self.results_table}"
            
            # Build query
            query = f"""
            SELECT RESULT_ID, BENCHMARK_ID, BENCHMARK_NAME, REPRESENTATION_TYPE,
                   RUN_DATE, TOTAL_QUESTIONS, CORRECT_COUNT, INCORRECT_COUNT,
                   NOT_ATTEMPTED_COUNT, AMBIGUOUS_COUNT, ACCURACY, F_SCORE,
                   AVG_RESPONSE_TIME_MS, DETAILED_METRICS, METADATA
            FROM {results_table}
            WHERE 1=1
            """
            
            params = []
            
            if benchmark_id:
                query += " AND BENCHMARK_ID = ?"
                params.append(benchmark_id)
            
            if representation_type:
                query += " AND REPRESENTATION_TYPE = ?"
                params.append(representation_type.value)
            
            query += " ORDER BY RUN_DATE DESC LIMIT ?"
            params.append(limit)
            
            # Execute query
            cursor.execute(query, tuple(params))
            
            # Process results
            results = []
            for row in cursor.fetchall():
                # Parse JSON fields
                detailed_metrics = json.loads(row[13]) if row[13] else {}
                metadata = json.loads(row[14]) if row[14] else {}
                
                # Create and add the result
                result = {
                    "result_id": row[0],
                    "benchmark_id": row[1],
                    "benchmark_name": row[2],
                    "representation_type": row[3],
                    "run_date": row[4],
                    "total_questions": row[5],
                    "correct_count": row[6],
                    "incorrect_count": row[7],
                    "not_attempted_count": row[8],
                    "ambiguous_count": row[9],
                    "accuracy": row[10],
                    "f_score": row[11],
                    "avg_response_time_ms": row[12],
                    "detailed_metrics": detailed_metrics,
                    "metadata": metadata,
                }
                results.append(result)
            
            return results
        
        except Exception as e:
            logger.error(f"Error getting benchmark results: {str(e)}")
            raise
        finally:
            cursor.close()


class QuestionGenerator:
    """
    Generator for benchmark questions from SAP HANA schemas and data.
    
    This class provides methods for generating diverse question types
    for benchmarking different data representation methods.
    """
    
    def __init__(
        self,
        generation_model: BaseLanguageModel,
        model_id: str = "question-generator",
    ):
        """
        Initialize the question generator.
        
        Args:
            generation_model: Language model for generating questions
            model_id: ID of the generation model
        """
        self.generation_model = generation_model
        self.model_id = model_id
        
        # Define question generation prompt templates for different question types
        self.schema_prompt = PromptTemplate.from_template(
            """You are an expert in database design and SAP HANA. 
Generate {num_questions} precise, fact-seeking questions about the following database schema.
Each question should have a single, indisputable answer based on the schema information provided.

Schema Information:
{schema_info}

For each question, provide:
1. A clear, specific question about the schema
2. The definitive, factual answer
3. The specific entity (table, column, relationship) the question refers to
4. Difficulty level (1-5, where 1 is easiest and 5 is hardest)

Example format:
Question: What is the primary key of the CUSTOMER table?
Answer: CUSTOMER_ID
Entity: CUSTOMER.CUSTOMER_ID
Difficulty: 1

Your questions should test understanding of:
- Table structures and relationships
- Primary and foreign keys
- Column data types and constraints
- Schema organization and naming conventions

Generate {num_questions} diverse questions:
"""
        )
        
        self.instance_prompt = PromptTemplate.from_template(
            """You are an expert in database analysis and SAP HANA. 
Generate {num_questions} precise, fact-seeking questions about specific data instances in the following database.
Each question should have a single, indisputable answer based on the data provided.

Data Information:
{data_info}

For each question, provide:
1. A clear, specific question about particular data values or instances
2. The definitive, factual answer
3. The specific entity (table.column) the question refers to
4. Difficulty level (1-5, where 1 is easiest and 5 is hardest)

Example format:
Question: What is the email address of customer with ID 1?
Answer: john.doe@example.com
Entity: CUSTOMERS.EMAIL
Difficulty: 1

Your questions should focus on:
- Specific data values for particular records
- Finding exact matches
- Simple data lookups
- Concrete instances, not aggregations or relationships

Generate {num_questions} diverse questions:
"""
        )
        
        self.relationship_prompt = PromptTemplate.from_template(
            """You are an expert in database relationships and SAP HANA. 
Generate {num_questions} precise, fact-seeking questions about relationships between entities in the following database.
Each question should have a single, indisputable answer based on the schema and data provided.

Schema and Data Information:
{schema_data_info}

For each question, provide:
1. A clear, specific question about relationships between entities
2. The definitive, factual answer
3. The specific relationship the question refers to (e.g., ORDERS_CUSTOMERS)
4. Difficulty level (1-5, where 1 is easiest and 5 is hardest)

Example format:
Question: Which customer placed order ID 1001?
Answer: John Doe (Customer ID 42)
Entity: ORDERS_CUSTOMERS
Difficulty: 2

Your questions should focus on:
- Foreign key relationships
- Parent-child relationships
- Many-to-many relationships
- Join conditions and cardinality

Generate {num_questions} diverse questions:
"""
        )
        
        self.aggregation_prompt = PromptTemplate.from_template(
            """You are an expert in data analysis and SAP HANA. 
Generate {num_questions} precise, fact-seeking questions about data aggregations in the following database.
Each question should have a single, indisputable answer based on the data provided.

Data Information:
{data_info}

For each question, provide:
1. A clear, specific question about data aggregations
2. The definitive, factual answer
3. The table(s) the question refers to
4. Difficulty level (1-5, where 1 is easiest and 5 is hardest)

Example format:
Question: What is the total number of orders placed by customer ID 42?
Answer: 7
Entity: ORDERS
Difficulty: 2

Your questions should focus on:
- Counts, sums, averages, minimums, maximums
- Group-by scenarios
- Distribution analysis
- Time-based aggregations

Generate {num_questions} diverse questions:
"""
        )
        
        self.inference_prompt = PromptTemplate.from_template(
            """You are an expert in data analysis and logical inference. 
Generate {num_questions} questions that require logical inference from the following database information.
Each question should have a single, correct answer that can be logically deduced from the data.

Schema and Data Information:
{schema_data_info}

For each question, provide:
1. A clear question that requires inference (not just direct lookup)
2. The correct answer that can be logically deduced
3. The tables/entities involved
4. Difficulty level (1-5, where 1 is easiest and 5 is hardest)

Example format:
Question: Based on order history, which product category appears to be most popular among customers aged 25-35?
Answer: Electronics
Entity: ORDERS,CUSTOMERS,PRODUCTS
Difficulty: 4

Your questions should require:
- Logical deduction
- Pattern recognition
- Cross-referencing multiple tables
- Understanding implications of the data
- Drawing valid conclusions from incomplete information

Avoid questions that can be answered with simple aggregation or direct lookup.
Focus on questions that would test a system's ability to make valid inferences.

Generate {num_questions} diverse inference questions:
"""
        )
        
        self.temporal_prompt = PromptTemplate.from_template(
            """You are an expert in temporal data analysis and SAP HANA. 
Generate {num_questions} precise, fact-seeking questions about time-based patterns in the following database.
Each question should have a single, indisputable answer based on the temporal data provided.

Temporal Data Information:
{temporal_data_info}

For each question, provide:
1. A clear, specific question about temporal patterns or time-based data
2. The definitive, factual answer
3. The table(s) and timestamp/date column(s) the question refers to
4. Difficulty level (1-5, where 1 is easiest and 5 is hardest)

Example format:
Question: In which month of 2022 did the store have the highest number of orders?
Answer: December 2022
Entity: ORDERS.ORDER_DATE
Difficulty: 3

Your questions should focus on:
- Trends over time
- Seasonal patterns
- Time intervals and durations
- Sequential events
- Historical comparisons

Generate {num_questions} diverse temporal questions:
"""
        )
    
    def parse_generated_questions(self, text: str, question_type: QuestionType) -> List[Dict[str, Any]]:
        """
        Parse generated questions from model output.
        
        Args:
            text: The model's generated text
            question_type: Type of questions to parse
            
        Returns:
            List of parsed questions
        """
        questions = []
        current_question = {}
        
        lines = text.strip().split("\n")
        for line in lines:
            line = line.strip()
            
            if line.startswith("Question:"):
                # Start a new question
                if current_question and "question" in current_question:
                    current_question["type"] = question_type.value
                    questions.append(current_question)
                current_question = {"question": line[len("Question:"):].strip()}
            elif line.startswith("Answer:") and current_question:
                current_question["answer"] = line[len("Answer:"):].strip()
            elif line.startswith("Entity:") and current_question:
                current_question["entity"] = line[len("Entity:"):].strip()
            elif line.startswith("Difficulty:") and current_question:
                # Check if there's a specified difficulty in the generated output
                try:
                    specified_difficulty = int(line[len("Difficulty:"):].strip())
                    # Ensure difficulty is between 1 and 5
                    specified_difficulty = max(1, min(5, specified_difficulty))
                    
                    # Store the model-specified difficulty as metadata
                    if "metadata" not in current_question:
                        current_question["metadata"] = {}
                    current_question["metadata"]["model_specified_difficulty"] = specified_difficulty
                    
                    # We'll still calculate our standardized difficulty later
                except ValueError:
                    # No valid difficulty specified, we'll just use our standardized assessment
                    pass
        
        # Add the last question
        if current_question and "question" in current_question:
            current_question["type"] = question_type.value
            questions.append(current_question)
        
        # Validate and filter questions
        validated_questions = []
        for q in questions:
            # Check for required fields
            if not all(k in q for k in ["question", "answer", "entity"]):
                logger.warning(f"Skipping incomplete question: {q}")
                continue
            
            # Validate question text
            if not self._validate_question_text(q["question"]):
                logger.warning(f"Skipping invalid question text: {q['question']}")
                continue
                
            # Validate answer text
            if not self._validate_answer_text(q["answer"]):
                logger.warning(f"Skipping invalid answer: {q['answer']}")
                continue
                
            # Validate entity reference
            if not self._validate_entity_reference(q["entity"]):
                logger.warning(f"Skipping invalid entity reference: {q['entity']}")
                continue
            
            # Apply standardized difficulty assessment
            question_type_obj = QuestionType(q["type"])
            standardized_difficulty = DifficultyLevel.assess_difficulty(q["question"], question_type_obj)
            q["difficulty"] = int(standardized_difficulty)
            
            # Store difficulty assessment details in metadata
            if "metadata" not in q:
                q["metadata"] = {}
                
            q["metadata"]["difficulty_assessment"] = {
                "standardized_difficulty": int(standardized_difficulty),
                "assessment_method": "DifficultyLevel.assess_difficulty",
                "assessment_version": "1.0",
                "indicators": self._get_difficulty_indicators(q["question"])
            }
                
            # Add validation metadata
            q["validation"] = {
                "validated_at": time.time(),
                "validation_version": "1.0",
            }
            
            validated_questions.append(q)
        
        logger.info(f"Generated {len(questions)} questions, {len(validated_questions)} passed validation")
        return validated_questions
        
    def _get_difficulty_assessment(self, question_text: str, question_type: QuestionType) -> Dict[str, Any]:
        """
        Get comprehensive difficulty assessment for a question.
        
        This method uses the enhanced DifficultyLevel.get_difficulty_analysis to provide
        a detailed assessment of question difficulty, including:
        - Complexity indicators detected in the question text
        - Base difficulty level based on question type
        - Adjustments made to difficulty based on complexity indicators
        - Final difficulty score with explanation
        
        Args:
            question_text: The question text to analyze
            question_type: Type of question (SCHEMA, INSTANCE, etc.)
            
        Returns:
            Dictionary with comprehensive difficulty assessment data
        """
        # Use the enhanced analysis method for detailed assessment
        return DifficultyLevel.get_difficulty_analysis(question_text, question_type)
        
    def _validate_question_text(self, question_text: str) -> bool:
        """
        Validate question text for quality and edge cases.
        
        Args:
            question_text: The question text to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Skip empty or very short questions
        if not question_text or len(question_text) < 10:
            return False
            
        # Skip questions that are too long
        if len(question_text) > 500:
            return False
            
        # Skip questions without question marks
        if "?" not in question_text:
            return False
            
        # Skip questions with SQL or code injection attempts
        dangerous_patterns = ["--", ";", "DROP", "DELETE", "INSERT", "UPDATE", "EXEC", "EXECUTE"]
        if any(pattern in question_text.upper() for pattern in dangerous_patterns):
            return False
            
        return True
        
    def _validate_answer_text(self, answer_text: str) -> bool:
        """
        Validate answer text for quality and edge cases.
        
        Args:
            answer_text: The answer text to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Skip empty or very short answers
        if not answer_text or len(answer_text) < 2:
            return False
            
        # Skip answers that are too long
        if len(answer_text) > 1000:
            return False
            
        # Skip answers with SQL or code injection attempts
        dangerous_patterns = ["--", ";", "DROP", "DELETE", "INSERT", "UPDATE", "EXEC", "EXECUTE"]
        if any(pattern in answer_text.upper() for pattern in dangerous_patterns):
            return False
            
        return True
        
    def _validate_entity_reference(self, entity_reference: str) -> bool:
        """
        Validate entity reference format and quality.
        
        Args:
            entity_reference: The entity reference to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Skip empty references
        if not entity_reference:
            return False
            
        # Skip references that are too long
        if len(entity_reference) > 200:
            return False
            
        # Check for basic table.column format for many entity types
        if "." in entity_reference:
            # Validate each part contains valid identifier characters
            parts = entity_reference.split(".")
            for part in parts:
                if not part or not all(c.isalnum() or c == '_' for c in part):
                    return False
        
        return True
    
    def generate_schema_questions(
        self,
        schema_info: str,
        num_questions: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Generate questions about database schema.
        
        Args:
            schema_info: Information about the schema
            num_questions: Number of questions to generate
            
        Returns:
            List of generated questions
        """
        # Format the prompt
        prompt = self.schema_prompt.format(
            schema_info=schema_info,
            num_questions=num_questions,
        )
        
        # Get the model's response
        response = self.generation_model.invoke(prompt)
        generated_text = response.content.strip()
        
        # Parse the generated questions
        return self.parse_generated_questions(generated_text, QuestionType.SCHEMA)
    
    def generate_instance_questions(
        self,
        data_info: str,
        num_questions: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Generate questions about specific data instances.
        
        Args:
            data_info: Information about the data
            num_questions: Number of questions to generate
            
        Returns:
            List of generated questions
        """
        # Format the prompt
        prompt = self.instance_prompt.format(
            data_info=data_info,
            num_questions=num_questions,
        )
        
        # Get the model's response
        response = self.generation_model.invoke(prompt)
        generated_text = response.content.strip()
        
        # Parse the generated questions
        return self.parse_generated_questions(generated_text, QuestionType.INSTANCE)
    
    def generate_relationship_questions(
        self,
        schema_data_info: str,
        num_questions: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Generate questions about entity relationships.
        
        Args:
            schema_data_info: Information about schema and data
            num_questions: Number of questions to generate
            
        Returns:
            List of generated questions
        """
        # Format the prompt
        prompt = self.relationship_prompt.format(
            schema_data_info=schema_data_info,
            num_questions=num_questions,
        )
        
        # Get the model's response
        response = self.generation_model.invoke(prompt)
        generated_text = response.content.strip()
        
        # Parse the generated questions
        return self.parse_generated_questions(generated_text, QuestionType.RELATIONSHIP)
    
    def generate_aggregation_questions(
        self,
        data_info: str,
        num_questions: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Generate questions about data aggregations.
        
        Args:
            data_info: Information about the data
            num_questions: Number of questions to generate
            
        Returns:
            List of generated questions
        """
        # Format the prompt
        prompt = self.aggregation_prompt.format(
            data_info=data_info,
            num_questions=num_questions,
        )
        
        # Get the model's response
        response = self.generation_model.invoke(prompt)
        generated_text = response.content.strip()
        
        # Parse the generated questions
        return self.parse_generated_questions(generated_text, QuestionType.AGGREGATION)
    
    def generate_inference_questions(
        self,
        schema_data_info: str,
        num_questions: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Generate questions requiring logical inference.
        
        Args:
            schema_data_info: Information about schema and data
            num_questions: Number of questions to generate
            
        Returns:
            List of generated questions
        """
        # Format the prompt
        prompt = self.inference_prompt.format(
            schema_data_info=schema_data_info,
            num_questions=num_questions,
        )
        
        # Get the model's response
        response = self.generation_model.invoke(prompt)
        generated_text = response.content.strip()
        
        # Parse the generated questions
        return self.parse_generated_questions(generated_text, QuestionType.INFERENCE)
    
    def generate_temporal_questions(
        self,
        temporal_data_info: str,
        num_questions: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Generate questions about temporal patterns.
        
        Args:
            temporal_data_info: Information about temporal data
            num_questions: Number of questions to generate
            
        Returns:
            List of generated questions
        """
        # Format the prompt
        prompt = self.temporal_prompt.format(
            temporal_data_info=temporal_data_info,
            num_questions=num_questions,
        )
        
        # Get the model's response
        response = self.generation_model.invoke(prompt)
        generated_text = response.content.strip()
        
        # Parse the generated questions
        return self.parse_generated_questions(generated_text, QuestionType.TEMPORAL)
    
    def create_benchmark_questions(
        self,
        schema_info: str,
        data_info: str,
        num_schema_questions: int = 10,
        num_instance_questions: int = 10,
        num_relationship_questions: int = 10,
        num_aggregation_questions: int = 10,
        num_inference_questions: int = 5,
        num_temporal_questions: int = 5,
    ) -> List[BenchmarkQuestion]:
        """
        Create a comprehensive set of benchmark questions.
        
        Args:
            schema_info: Information about the schema
            data_info: Information about the data
            num_schema_questions: Number of schema questions
            num_instance_questions: Number of instance questions
            num_relationship_questions: Number of relationship questions
            num_aggregation_questions: Number of aggregation questions
            num_inference_questions: Number of inference questions
            num_temporal_questions: Number of temporal questions
            
        Returns:
            List of benchmark questions with full provenance information
        """
        benchmark_questions = []
        generation_start_time = time.time()
        
        # Track data sources
        data_sources = []
        if schema_info:
            # Extract schema name from the schema info
            schema_match = re.search(r"Schema:\s+([A-Za-z0-9_]+)", schema_info)
            if schema_match:
                schema_name = schema_match.group(1)
                data_sources.append(f"schema:{schema_name}")
            else:
                data_sources.append("schema:unknown")
        
        if data_info:
            # Extract schema name from the data info
            schema_match = re.search(r"Schema:\s+([A-Za-z0-9_]+)", data_info)
            if schema_match:
                schema_name = schema_match.group(1)
                data_sources.append(f"data:{schema_name}")
            else:
                data_sources.append("data:unknown")
                
        # Capture generation parameters
        generation_parameters = {
            "num_schema_questions": num_schema_questions,
            "num_instance_questions": num_instance_questions,
            "num_relationship_questions": num_relationship_questions,
            "num_aggregation_questions": num_aggregation_questions, 
            "num_inference_questions": num_inference_questions,
            "num_temporal_questions": num_temporal_questions,
            "schema_info_length": len(schema_info) if schema_info else 0,
            "data_info_length": len(data_info) if data_info else 0,
            "generation_model": self.model_id,
        }
        
        # Generate schema questions
        if num_schema_questions > 0:
            logger.info(f"Generating {num_schema_questions} schema questions")
            schema_questions = self.generate_schema_questions(
                schema_info=schema_info,
                num_questions=num_schema_questions,
            )
            
            for q in schema_questions:
                if all(k in q for k in ["question", "answer", "entity", "difficulty"]):
                    # Create provenance information
                    provenance = QuestionProvenance(
                        source_type="model",
                        source_id=self.model_id,
                        generation_timestamp=time.time(),
                        generation_method="schema_prompt_template",
                        generation_parameters={
                            "schema_prompt_template": self.schema_prompt.template[:50] + "...",
                            "num_questions": num_schema_questions,
                            **generation_parameters
                        },
                        input_data_sources=data_sources.copy(),
                    )
                    
                    # Add model-specific validation metadata if available
                    if "validation" in q:
                        provenance.add_revision(
                            editor="validation_pipeline",
                            timestamp=q["validation"].get("validated_at", time.time()),
                            reason="initial_validation",
                            changes={
                                "validation_version": q["validation"].get("validation_version", "1.0"),
                            }
                        )
                    
                    benchmark_questions.append(BenchmarkQuestion(
                        question_id=str(uuid.uuid4()),
                        question_text=q["question"],
                        reference_answer=q["answer"],
                        question_type=QuestionType.SCHEMA,
                        entity_reference=q["entity"],
                        difficulty=q["difficulty"],
                        metadata=q.get("metadata", {}),
                        provenance=provenance,
                    ))
        
        # Generate instance questions
        if num_instance_questions > 0:
            logger.info(f"Generating {num_instance_questions} instance questions")
            instance_questions = self.generate_instance_questions(
                data_info=data_info,
                num_questions=num_instance_questions,
            )
            
            for q in instance_questions:
                if all(k in q for k in ["question", "answer", "entity", "difficulty"]):
                    # Create provenance information
                    provenance = QuestionProvenance(
                        source_type="model",
                        source_id=self.model_id,
                        generation_timestamp=time.time(),
                        generation_method="instance_prompt_template",
                        generation_parameters={
                            "instance_prompt_template": self.instance_prompt.template[:50] + "...",
                            "num_questions": num_instance_questions,
                            **generation_parameters
                        },
                        input_data_sources=data_sources.copy(),
                    )
                    
                    # Add model-specific validation metadata if available
                    if "validation" in q:
                        provenance.add_revision(
                            editor="validation_pipeline",
                            timestamp=q["validation"].get("validated_at", time.time()),
                            reason="initial_validation",
                            changes={
                                "validation_version": q["validation"].get("validation_version", "1.0"),
                            }
                        )
                    
                    benchmark_questions.append(BenchmarkQuestion(
                        question_id=str(uuid.uuid4()),
                        question_text=q["question"],
                        reference_answer=q["answer"],
                        question_type=QuestionType.INSTANCE,
                        entity_reference=q["entity"],
                        difficulty=q["difficulty"],
                        metadata=q.get("metadata", {}),
                        provenance=provenance,
                    ))
        
        # Generate relationship questions
        if num_relationship_questions > 0:
            logger.info(f"Generating {num_relationship_questions} relationship questions")
            relationship_questions = self.generate_relationship_questions(
                schema_data_info=f"{schema_info}\n\n{data_info}",
                num_questions=num_relationship_questions,
            )
            
            for q in relationship_questions:
                if all(k in q for k in ["question", "answer", "entity", "difficulty"]):
                    # Create provenance information
                    provenance = QuestionProvenance(
                        source_type="model",
                        source_id=self.model_id,
                        generation_timestamp=time.time(),
                        generation_method="relationship_prompt_template",
                        generation_parameters={
                            "relationship_prompt_template": self.relationship_prompt.template[:50] + "...",
                            "num_questions": num_relationship_questions,
                            **generation_parameters
                        },
                        input_data_sources=data_sources.copy(),
                    )
                    
                    # Add model-specific validation metadata if available
                    if "validation" in q:
                        provenance.add_revision(
                            editor="validation_pipeline",
                            timestamp=q["validation"].get("validated_at", time.time()),
                            reason="initial_validation",
                            changes={
                                "validation_version": q["validation"].get("validation_version", "1.0"),
                            }
                        )
                    
                    benchmark_questions.append(BenchmarkQuestion(
                        question_id=str(uuid.uuid4()),
                        question_text=q["question"],
                        reference_answer=q["answer"],
                        question_type=QuestionType.RELATIONSHIP,
                        entity_reference=q["entity"],
                        difficulty=q["difficulty"],
                        metadata=q.get("metadata", {}),
                        provenance=provenance,
                    ))
        
        # Generate aggregation questions
        if num_aggregation_questions > 0:
            logger.info(f"Generating {num_aggregation_questions} aggregation questions")
            aggregation_questions = self.generate_aggregation_questions(
                data_info=data_info,
                num_questions=num_aggregation_questions,
            )
            
            for q in aggregation_questions:
                if all(k in q for k in ["question", "answer", "entity", "difficulty"]):
                    # Create provenance information
                    provenance = QuestionProvenance(
                        source_type="model",
                        source_id=self.model_id,
                        generation_timestamp=time.time(),
                        generation_method="aggregation_prompt_template",
                        generation_parameters={
                            "aggregation_prompt_template": self.aggregation_prompt.template[:50] + "...",
                            "num_questions": num_aggregation_questions,
                            **generation_parameters
                        },
                        input_data_sources=data_sources.copy(),
                    )
                    
                    # Add model-specific validation metadata if available
                    if "validation" in q:
                        provenance.add_revision(
                            editor="validation_pipeline",
                            timestamp=q["validation"].get("validated_at", time.time()),
                            reason="initial_validation",
                            changes={
                                "validation_version": q["validation"].get("validation_version", "1.0"),
                            }
                        )
                    
                    benchmark_questions.append(BenchmarkQuestion(
                        question_id=str(uuid.uuid4()),
                        question_text=q["question"],
                        reference_answer=q["answer"],
                        question_type=QuestionType.AGGREGATION,
                        entity_reference=q["entity"],
                        difficulty=q["difficulty"],
                        metadata=q.get("metadata", {}),
                        provenance=provenance,
                    ))
        
        # Generate inference questions
        if num_inference_questions > 0:
            logger.info(f"Generating {num_inference_questions} inference questions")
            inference_questions = self.generate_inference_questions(
                schema_data_info=f"{schema_info}\n\n{data_info}",
                num_questions=num_inference_questions,
            )
            
            for q in inference_questions:
                if all(k in q for k in ["question", "answer", "entity", "difficulty"]):
                    # Create provenance information
                    provenance = QuestionProvenance(
                        source_type="model",
                        source_id=self.model_id,
                        generation_timestamp=time.time(),
                        generation_method="inference_prompt_template",
                        generation_parameters={
                            "inference_prompt_template": self.inference_prompt.template[:50] + "...",
                            "num_questions": num_inference_questions,
                            **generation_parameters
                        },
                        input_data_sources=data_sources.copy(),
                    )
                    
                    # Add model-specific validation metadata if available
                    if "validation" in q:
                        provenance.add_revision(
                            editor="validation_pipeline",
                            timestamp=q["validation"].get("validated_at", time.time()),
                            reason="initial_validation",
                            changes={
                                "validation_version": q["validation"].get("validation_version", "1.0"),
                            }
                        )
                    
                    benchmark_questions.append(BenchmarkQuestion(
                        question_id=str(uuid.uuid4()),
                        question_text=q["question"],
                        reference_answer=q["answer"],
                        question_type=QuestionType.INFERENCE,
                        entity_reference=q["entity"],
                        difficulty=q["difficulty"],
                        metadata=q.get("metadata", {}),
                        provenance=provenance,
                    ))
        
        # Generate temporal questions
        if num_temporal_questions > 0:
            logger.info(f"Generating {num_temporal_questions} temporal questions")
            temporal_questions = self.generate_temporal_questions(
                temporal_data_info=data_info,
                num_questions=num_temporal_questions,
            )
            
            for q in temporal_questions:
                if all(k in q for k in ["question", "answer", "entity", "difficulty"]):
                    # Create provenance information
                    provenance = QuestionProvenance(
                        source_type="model",
                        source_id=self.model_id,
                        generation_timestamp=time.time(),
                        generation_method="temporal_prompt_template",
                        generation_parameters={
                            "temporal_prompt_template": self.temporal_prompt.template[:50] + "...",
                            "num_questions": num_temporal_questions,
                            **generation_parameters
                        },
                        input_data_sources=data_sources.copy(),
                    )
                    
                    # Add model-specific validation metadata if available
                    if "validation" in q:
                        provenance.add_revision(
                            editor="validation_pipeline",
                            timestamp=q["validation"].get("validated_at", time.time()),
                            reason="initial_validation",
                            changes={
                                "validation_version": q["validation"].get("validation_version", "1.0"),
                            }
                        )
                    
                    benchmark_questions.append(BenchmarkQuestion(
                        question_id=str(uuid.uuid4()),
                        question_text=q["question"],
                        reference_answer=q["answer"],
                        question_type=QuestionType.TEMPORAL,
                        entity_reference=q["entity"],
                        difficulty=q["difficulty"],
                        metadata=q.get("metadata", {}),
                        provenance=provenance,
                    ))
        
        # Add overall generation metadata to each question
        generation_end_time = time.time()
        generation_duration = generation_end_time - generation_start_time
        generation_timestamp = datetime.datetime.fromtimestamp(generation_start_time).isoformat()
        
        for question in benchmark_questions:
            if question.metadata is None:
                question.metadata = {}
                
            question.metadata.update({
                "batch_generation_id": str(uuid.uuid4()),  # Same for all questions in this batch
                "generation_timestamp": generation_timestamp,
                "generation_duration_seconds": generation_duration,
                "generator_model_id": self.model_id,
                "question_count_in_batch": len(benchmark_questions),
            })
        
        return benchmark_questions