"""
Unit tests for benchmark system data classes.

This module contains unit tests for the BenchmarkQuestion, BenchmarkAnswer,
and DataRepresentationMetrics classes.
"""

import unittest
import time
import uuid
import json
from dataclasses import asdict

import pytest

from langchain_hana.reasoning.benchmark_system import (
    BenchmarkQuestion,
    BenchmarkAnswer,
    DataRepresentationMetrics,
    QuestionType,
    RepresentationType,
    AnswerSource,
    QuestionProvenance,
    FactualityGrade
)


class TestBenchmarkQuestion(unittest.TestCase):
    """Tests for the BenchmarkQuestion class."""
    
    def test_initialization(self):
        """Test initialization with required parameters."""
        # Create a question
        question_id = str(uuid.uuid4())
        question = BenchmarkQuestion(
            question_id=question_id,
            question_text="What is the primary key of the CUSTOMERS table?",
            reference_answer="CUSTOMER_ID",
            question_type=QuestionType.SCHEMA,
            entity_reference="CUSTOMERS",
            difficulty=1
        )
        
        # Verify field values
        self.assertEqual(question.question_id, question_id)
        self.assertEqual(question.question_text, "What is the primary key of the CUSTOMERS table?")
        self.assertEqual(question.reference_answer, "CUSTOMER_ID")
        self.assertEqual(question.question_type, QuestionType.SCHEMA)
        self.assertEqual(question.entity_reference, "CUSTOMERS")
        self.assertEqual(question.difficulty, 1)
        self.assertIsNotNone(question.created_at)
        self.assertEqual(question.metadata, {})
        self.assertIsNone(question.provenance)
    
    def test_with_provenance(self):
        """Test initialization with provenance information."""
        # Create provenance
        provenance = QuestionProvenance(
            source_type="model",
            source_id="gpt-4",
            generation_timestamp=time.time(),
            generation_method="template",
            generation_parameters={"template_id": "schema-questions"},
            input_data_sources=["schema:SALES"]
        )
        
        # Create a question with provenance
        question = BenchmarkQuestion(
            question_id=str(uuid.uuid4()),
            question_text="What is the primary key of the CUSTOMERS table?",
            reference_answer="CUSTOMER_ID",
            question_type=QuestionType.SCHEMA,
            entity_reference="CUSTOMERS",
            difficulty=1,
            metadata={"test": "value"},
            provenance=provenance
        )
        
        # Verify provenance and metadata
        self.assertIsNotNone(question.provenance)
        self.assertEqual(question.provenance.source_type, "model")
        self.assertEqual(question.provenance.source_id, "gpt-4")
        self.assertEqual(question.metadata, {"test": "value"})
    
    def test_serialization(self):
        """Test serialization to dictionary."""
        # Create a question with metadata
        question_id = str(uuid.uuid4())
        created_at = time.time()
        question = BenchmarkQuestion(
            question_id=question_id,
            question_text="What is the primary key of the CUSTOMERS table?",
            reference_answer="CUSTOMER_ID",
            question_type=QuestionType.SCHEMA,
            entity_reference="CUSTOMERS",
            difficulty=1,
            created_at=created_at,
            metadata={"test": "value"}
        )
        
        # Convert to dictionary
        question_dict = asdict(question)
        
        # Verify dictionary fields
        self.assertEqual(question_dict["question_id"], question_id)
        self.assertEqual(question_dict["question_text"], "What is the primary key of the CUSTOMERS table?")
        self.assertEqual(question_dict["reference_answer"], "CUSTOMER_ID")
        self.assertEqual(question_dict["question_type"], QuestionType.SCHEMA)
        self.assertEqual(question_dict["entity_reference"], "CUSTOMERS")
        self.assertEqual(question_dict["difficulty"], 1)
        self.assertEqual(question_dict["created_at"], created_at)
        self.assertEqual(question_dict["metadata"], {"test": "value"})
        self.assertIsNone(question_dict["provenance"])
        
        # Test JSON serialization
        json_str = json.dumps(question_dict, default=str)  # Handle enum serialization
        self.assertGreater(len(json_str), 10)  # Basic check that JSON is non-empty


class TestBenchmarkAnswer(unittest.TestCase):
    """Tests for the BenchmarkAnswer class."""
    
    def test_initialization(self):
        """Test initialization with required parameters."""
        # Create an answer
        answer_id = str(uuid.uuid4())
        question_id = str(uuid.uuid4())
        answer = BenchmarkAnswer(
            answer_id=answer_id,
            question_id=question_id,
            representation_type=RepresentationType.RELATIONAL,
            answer_source=AnswerSource.DIRECT_QUERY,
            answer_text="CUSTOMER_ID"
        )
        
        # Verify field values
        self.assertEqual(answer.answer_id, answer_id)
        self.assertEqual(answer.question_id, question_id)
        self.assertEqual(answer.representation_type, RepresentationType.RELATIONAL)
        self.assertEqual(answer.answer_source, AnswerSource.DIRECT_QUERY)
        self.assertEqual(answer.answer_text, "CUSTOMER_ID")
        self.assertIsNone(answer.grade)
        self.assertIsNone(answer.confidence)
        self.assertIsNone(answer.response_time_ms)
        self.assertIsNotNone(answer.created_at)
        self.assertEqual(answer.metadata, {})
    
    def test_with_optional_fields(self):
        """Test initialization with all optional fields."""
        # Create an answer with all fields
        answer = BenchmarkAnswer(
            answer_id=str(uuid.uuid4()),
            question_id=str(uuid.uuid4()),
            representation_type=RepresentationType.VECTOR,
            answer_source=AnswerSource.VECTOR_SEARCH,
            answer_text="The primary key is CUSTOMER_ID",
            grade=FactualityGrade.CORRECT,
            confidence=0.95,
            response_time_ms=150,
            created_at=time.time(),
            metadata={"test": "value"}
        )
        
        # Verify optional fields
        self.assertEqual(answer.grade, FactualityGrade.CORRECT)
        self.assertEqual(answer.confidence, 0.95)
        self.assertEqual(answer.response_time_ms, 150)
        self.assertEqual(answer.metadata, {"test": "value"})
    
    def test_serialization(self):
        """Test serialization to dictionary."""
        # Create an answer with all fields
        answer_id = str(uuid.uuid4())
        question_id = str(uuid.uuid4())
        created_at = time.time()
        answer = BenchmarkAnswer(
            answer_id=answer_id,
            question_id=question_id,
            representation_type=RepresentationType.VECTOR,
            answer_source=AnswerSource.VECTOR_SEARCH,
            answer_text="The primary key is CUSTOMER_ID",
            grade=FactualityGrade.CORRECT,
            confidence=0.95,
            response_time_ms=150,
            created_at=created_at,
            metadata={"test": "value"}
        )
        
        # Convert to dictionary
        answer_dict = asdict(answer)
        
        # Verify dictionary fields
        self.assertEqual(answer_dict["answer_id"], answer_id)
        self.assertEqual(answer_dict["question_id"], question_id)
        self.assertEqual(answer_dict["representation_type"], RepresentationType.VECTOR)
        self.assertEqual(answer_dict["answer_source"], AnswerSource.VECTOR_SEARCH)
        self.assertEqual(answer_dict["answer_text"], "The primary key is CUSTOMER_ID")
        self.assertEqual(answer_dict["grade"], FactualityGrade.CORRECT)
        self.assertEqual(answer_dict["confidence"], 0.95)
        self.assertEqual(answer_dict["response_time_ms"], 150)
        self.assertEqual(answer_dict["created_at"], created_at)
        self.assertEqual(answer_dict["metadata"], {"test": "value"})
        
        # Test JSON serialization
        json_str = json.dumps(answer_dict, default=str)  # Handle enum serialization
        self.assertGreater(len(json_str), 10)  # Basic check that JSON is non-empty


class TestDataRepresentationMetrics(unittest.TestCase):
    """Tests for the DataRepresentationMetrics class."""
    
    def test_initialization(self):
        """Test initialization with required parameters."""
        # Create metrics
        metrics = DataRepresentationMetrics(
            representation_type=RepresentationType.RELATIONAL
        )
        
        # Verify field values
        self.assertEqual(metrics.representation_type, RepresentationType.RELATIONAL)
        self.assertEqual(metrics.correct_count, 0)
        self.assertEqual(metrics.incorrect_count, 0)
        self.assertEqual(metrics.not_attempted_count, 0)
        self.assertEqual(metrics.ambiguous_count, 0)
        self.assertEqual(metrics.total_count, 0)
        self.assertIsNone(metrics.avg_response_time_ms)
        self.assertEqual(metrics.metrics_by_question_type, {})
        self.assertEqual(metrics.metrics_by_difficulty, {})
    
    def test_accuracy_calculation(self):
        """Test accuracy calculation."""
        # Create metrics with some data
        metrics = DataRepresentationMetrics(
            representation_type=RepresentationType.RELATIONAL,
            correct_count=75,
            incorrect_count=15,
            not_attempted_count=5,
            ambiguous_count=5,
            total_count=100
        )
        
        # Verify accuracy calculations
        self.assertEqual(metrics.accuracy, 0.75)  # 75 correct out of 100 total
        self.assertEqual(metrics.attempted_accuracy, 75/(75+15+5))  # 75 correct out of 95 attempted
        
        # F-score calculation
        expected_f_score = 2 * (metrics.accuracy * metrics.attempted_accuracy) / (metrics.accuracy + metrics.attempted_accuracy)
        self.assertEqual(metrics.f_score, expected_f_score)
    
    def test_zero_division_handling(self):
        """Test handling of zero division in calculations."""
        # Create empty metrics
        metrics = DataRepresentationMetrics(
            representation_type=RepresentationType.RELATIONAL
        )
        
        # Verify zero handling
        self.assertEqual(metrics.accuracy, 0.0)  # Should not raise ZeroDivisionError
        self.assertEqual(metrics.attempted_accuracy, 0.0)  # Should not raise ZeroDivisionError
        self.assertEqual(metrics.f_score, 0.0)  # Should not raise ZeroDivisionError
        
        # Create metrics with only not_attempted
        metrics = DataRepresentationMetrics(
            representation_type=RepresentationType.RELATIONAL,
            not_attempted_count=10,
            total_count=10
        )
        
        # Verify zero handling
        self.assertEqual(metrics.accuracy, 0.0)
        self.assertEqual(metrics.attempted_accuracy, 0.0)  # No attempted questions
        self.assertEqual(metrics.f_score, 0.0)
    
    def test_metrics_by_question_type(self):
        """Test metrics by question type."""
        # Create metrics with question type breakdown
        metrics_by_question_type = {
            QuestionType.SCHEMA.value: {
                "total": 20,
                "correct": 15,
                "accuracy": 0.75
            },
            QuestionType.INSTANCE.value: {
                "total": 30,
                "correct": 25,
                "accuracy": 0.833
            }
        }
        
        metrics = DataRepresentationMetrics(
            representation_type=RepresentationType.RELATIONAL,
            correct_count=40,  # 15 + 25
            incorrect_count=10,  # (20-15) + (30-25)
            total_count=50,  # 20 + 30
            metrics_by_question_type=metrics_by_question_type
        )
        
        # Verify metrics by question type
        self.assertEqual(metrics.metrics_by_question_type[QuestionType.SCHEMA.value]["total"], 20)
        self.assertEqual(metrics.metrics_by_question_type[QuestionType.SCHEMA.value]["correct"], 15)
        self.assertAlmostEqual(metrics.metrics_by_question_type[QuestionType.SCHEMA.value]["accuracy"], 0.75)
        
        self.assertEqual(metrics.metrics_by_question_type[QuestionType.INSTANCE.value]["total"], 30)
        self.assertEqual(metrics.metrics_by_question_type[QuestionType.INSTANCE.value]["correct"], 25)
        self.assertAlmostEqual(metrics.metrics_by_question_type[QuestionType.INSTANCE.value]["accuracy"], 0.833, places=3)
    
    def test_metrics_by_difficulty(self):
        """Test metrics by difficulty level."""
        # Create metrics with difficulty breakdown
        metrics_by_difficulty = {
            1: {
                "total": 20,
                "correct": 18,
                "accuracy": 0.9
            },
            3: {
                "total": 30,
                "correct": 20,
                "accuracy": 0.667
            }
        }
        
        metrics = DataRepresentationMetrics(
            representation_type=RepresentationType.RELATIONAL,
            correct_count=38,  # 18 + 20
            incorrect_count=12,  # (20-18) + (30-20)
            total_count=50,  # 20 + 30
            metrics_by_difficulty=metrics_by_difficulty
        )
        
        # Verify metrics by difficulty
        self.assertEqual(metrics.metrics_by_difficulty[1]["total"], 20)
        self.assertEqual(metrics.metrics_by_difficulty[1]["correct"], 18)
        self.assertAlmostEqual(metrics.metrics_by_difficulty[1]["accuracy"], 0.9)
        
        self.assertEqual(metrics.metrics_by_difficulty[3]["total"], 30)
        self.assertEqual(metrics.metrics_by_difficulty[3]["correct"], 20)
        self.assertAlmostEqual(metrics.metrics_by_difficulty[3]["accuracy"], 0.667, places=3)
    
    def test_serialization(self):
        """Test serialization to dictionary."""
        # Create metrics with some data
        metrics = DataRepresentationMetrics(
            representation_type=RepresentationType.RELATIONAL,
            correct_count=75,
            incorrect_count=15,
            not_attempted_count=5,
            ambiguous_count=5,
            total_count=100,
            avg_response_time_ms=250.5,
            metrics_by_question_type={
                "schema": {"total": 50, "correct": 40, "accuracy": 0.8}
            },
            metrics_by_difficulty={
                1: {"total": 30, "correct": 25, "accuracy": 0.833},
                3: {"total": 70, "correct": 50, "accuracy": 0.714}
            }
        )
        
        # Convert to dictionary
        metrics_dict = asdict(metrics)
        
        # Verify dictionary fields
        self.assertEqual(metrics_dict["representation_type"], RepresentationType.RELATIONAL)
        self.assertEqual(metrics_dict["correct_count"], 75)
        self.assertEqual(metrics_dict["incorrect_count"], 15)
        self.assertEqual(metrics_dict["not_attempted_count"], 5)
        self.assertEqual(metrics_dict["ambiguous_count"], 5)
        self.assertEqual(metrics_dict["total_count"], 100)
        self.assertEqual(metrics_dict["avg_response_time_ms"], 250.5)
        self.assertEqual(metrics_dict["metrics_by_question_type"]["schema"]["total"], 50)
        self.assertEqual(metrics_dict["metrics_by_difficulty"][1]["correct"], 25)
        
        # Test JSON serialization
        json_str = json.dumps(metrics_dict, default=str)  # Handle enum serialization
        self.assertGreater(len(json_str), 10)  # Basic check that JSON is non-empty


if __name__ == "__main__":
    unittest.main()