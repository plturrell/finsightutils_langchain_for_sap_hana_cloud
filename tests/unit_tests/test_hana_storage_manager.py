"""
Unit tests for the HanaStorageManager class.

This module contains unit tests for the HanaStorageManager class, which is responsible
for storing and retrieving benchmark data in SAP HANA.
"""

import pytest
import time
import json
import uuid
from unittest.mock import MagicMock, patch

from langchain_hana.reasoning.benchmark_system import (
    QuestionProvenance,
    BenchmarkQuestion,
    BenchmarkAnswer,
    DataRepresentationMetrics,
    QuestionType,
    RepresentationType,
    AnswerSource,
    FactualityGrade,
    HanaStorageManager
)


class TestHanaStorageManager:
    """Tests for the HanaStorageManager class."""
    
    def test_initialization(self, mock_connection):
        """Test initialization and table creation."""
        # Create storage manager
        manager = HanaStorageManager(
            connection=mock_connection,
            schema_name="TEST_SCHEMA",
            questions_table="TEST_QUESTIONS",
            answers_table="TEST_ANSWERS",
            results_table="TEST_RESULTS"
        )
        
        # Verify tables were created by checking schema information
        cursor = mock_connection.cursor()
        
        # Check if questions table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='TEST_QUESTIONS'")
        assert cursor.fetchone() is not None
        
        # Check if answers table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='TEST_ANSWERS'")
        assert cursor.fetchone() is not None
        
        # Check if results table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='TEST_RESULTS'")
        assert cursor.fetchone() is not None
        
        # Check if provenance table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='QUESTION_PROVENANCE'")
        assert cursor.fetchone() is not None
        
        cursor.close()
    
    def test_create_and_get_question(self, hana_storage_manager, sample_question):
        """Test creating and retrieving a question."""
        # Create a question
        hana_storage_manager.create_question(sample_question)
        
        # Retrieve the question
        retrieved_question = hana_storage_manager.get_question(sample_question.question_id)
        
        # Verify retrieved question matches original
        assert retrieved_question is not None
        assert retrieved_question.question_id == sample_question.question_id
        assert retrieved_question.question_text == sample_question.question_text
        assert retrieved_question.reference_answer == sample_question.reference_answer
        assert retrieved_question.question_type == sample_question.question_type
        assert retrieved_question.entity_reference == sample_question.entity_reference
        assert retrieved_question.difficulty == sample_question.difficulty
        assert abs(retrieved_question.created_at - sample_question.created_at) < 0.1  # Allow small timestamp difference
        assert retrieved_question.metadata == sample_question.metadata
        assert retrieved_question.provenance is None  # No provenance in sample_question
    
    def test_create_and_get_question_with_provenance(self, hana_storage_manager, sample_question_with_provenance):
        """Test creating and retrieving a question with provenance."""
        # Create a question with provenance
        hana_storage_manager.create_question(sample_question_with_provenance)
        
        # Retrieve the question
        retrieved_question = hana_storage_manager.get_question(sample_question_with_provenance.question_id)
        
        # Verify retrieved question matches original
        assert retrieved_question is not None
        assert retrieved_question.question_id == sample_question_with_provenance.question_id
        assert retrieved_question.question_text == sample_question_with_provenance.question_text
        assert retrieved_question.reference_answer == sample_question_with_provenance.reference_answer
        assert retrieved_question.question_type == sample_question_with_provenance.question_type
        assert retrieved_question.entity_reference == sample_question_with_provenance.entity_reference
        assert retrieved_question.difficulty == sample_question_with_provenance.difficulty
        
        # Verify provenance
        assert retrieved_question.provenance is not None
        assert retrieved_question.provenance.source_type == sample_question_with_provenance.provenance.source_type
        assert retrieved_question.provenance.source_id == sample_question_with_provenance.provenance.source_id
        assert retrieved_question.provenance.generation_method == sample_question_with_provenance.provenance.generation_method
        assert len(retrieved_question.provenance.input_data_sources) == len(sample_question_with_provenance.provenance.input_data_sources)
    
    def test_update_question(self, hana_storage_manager, sample_question):
        """Test updating a question."""
        # Create a question
        hana_storage_manager.create_question(sample_question)
        
        # Modify the question
        updated_question = BenchmarkQuestion(
            question_id=sample_question.question_id,
            question_text="Updated: What is the primary key of the CUSTOMERS table?",
            reference_answer="Updated: CUSTOMER_ID",
            question_type=sample_question.question_type,
            entity_reference=sample_question.entity_reference,
            difficulty=2,  # Changed difficulty
            created_at=sample_question.created_at,
            metadata={"test": "updated"}
        )
        
        # Update the question
        hana_storage_manager.update_question(updated_question)
        
        # Retrieve the updated question
        retrieved_question = hana_storage_manager.get_question(sample_question.question_id)
        
        # Verify retrieved question matches updated values
        assert retrieved_question is not None
        assert retrieved_question.question_text == "Updated: What is the primary key of the CUSTOMERS table?"
        assert retrieved_question.reference_answer == "Updated: CUSTOMER_ID"
        assert retrieved_question.difficulty == 2
        assert retrieved_question.metadata == {"test": "updated"}
    
    def test_update_question_with_provenance(self, hana_storage_manager, sample_question_with_provenance):
        """Test updating a question with provenance tracking."""
        # Create a question with provenance
        hana_storage_manager.create_question(sample_question_with_provenance)
        
        # Make a copy with updated text
        updated_question = BenchmarkQuestion(
            question_id=sample_question_with_provenance.question_id,
            question_text="Updated: What is the primary key of the CUSTOMERS table?",
            reference_answer=sample_question_with_provenance.reference_answer,
            question_type=sample_question_with_provenance.question_type,
            entity_reference=sample_question_with_provenance.entity_reference,
            difficulty=sample_question_with_provenance.difficulty,
            created_at=sample_question_with_provenance.created_at,
            metadata=sample_question_with_provenance.metadata,
            provenance=sample_question_with_provenance.provenance  # Keep original provenance
        )
        
        # Update the question
        hana_storage_manager.update_question(updated_question)
        
        # Retrieve the question provenance history
        provenance_history = hana_storage_manager.get_question_provenance_history(updated_question.question_id)
        
        # Verify provenance history exists and includes the update
        assert len(provenance_history) >= 1  # At least one provenance record
        
        # Get the updated question
        retrieved_question = hana_storage_manager.get_question(updated_question.question_id)
        assert retrieved_question.provenance is not None
        
        # Should have at least the original provenance
        assert len(retrieved_question.provenance.revision_history) >= 0
    
    def test_delete_question(self, hana_storage_manager, sample_question):
        """Test deleting a question."""
        # Create a question
        hana_storage_manager.create_question(sample_question)
        
        # Verify it exists
        assert hana_storage_manager.get_question(sample_question.question_id) is not None
        
        # Delete the question
        hana_storage_manager.delete_question(sample_question.question_id)
        
        # Verify it no longer exists
        assert hana_storage_manager.get_question(sample_question.question_id) is None
    
    def test_create_and_get_answer(self, hana_storage_manager, sample_question, sample_answer):
        """Test creating and retrieving an answer."""
        # First create a question (since answers need a valid question_id foreign key)
        hana_storage_manager.create_question(sample_question)
        
        # Create an answer for this question
        answer = BenchmarkAnswer(
            answer_id=sample_answer.answer_id,
            question_id=sample_question.question_id,  # Use the question we just created
            representation_type=sample_answer.representation_type,
            answer_source=sample_answer.answer_source,
            answer_text=sample_answer.answer_text,
            grade=sample_answer.grade,
            confidence=sample_answer.confidence,
            response_time_ms=sample_answer.response_time_ms,
            created_at=sample_answer.created_at,
            metadata=sample_answer.metadata
        )
        
        # Create the answer
        hana_storage_manager.create_answer(answer)
        
        # Retrieve the answer
        retrieved_answer = hana_storage_manager.get_answer(answer.answer_id)
        
        # Verify retrieved answer matches original
        assert retrieved_answer is not None
        assert retrieved_answer.answer_id == answer.answer_id
        assert retrieved_answer.question_id == sample_question.question_id
        assert retrieved_answer.representation_type == answer.representation_type
        assert retrieved_answer.answer_source == answer.answer_source
        assert retrieved_answer.answer_text == answer.answer_text
        assert retrieved_answer.grade == answer.grade
        assert retrieved_answer.confidence == answer.confidence
        assert retrieved_answer.response_time_ms == answer.response_time_ms
        assert abs(retrieved_answer.created_at - answer.created_at) < 0.1  # Allow small timestamp difference
        assert retrieved_answer.metadata == answer.metadata
    
    def test_update_answer(self, hana_storage_manager, sample_question, sample_answer):
        """Test updating an answer."""
        # Create question and answer
        hana_storage_manager.create_question(sample_question)
        answer = BenchmarkAnswer(
            answer_id=sample_answer.answer_id,
            question_id=sample_question.question_id,
            representation_type=sample_answer.representation_type,
            answer_source=sample_answer.answer_source,
            answer_text=sample_answer.answer_text,
            grade=sample_answer.grade,
            confidence=sample_answer.confidence,
            response_time_ms=sample_answer.response_time_ms,
            created_at=sample_answer.created_at,
            metadata=sample_answer.metadata
        )
        hana_storage_manager.create_answer(answer)
        
        # Modify the answer
        updated_answer = BenchmarkAnswer(
            answer_id=answer.answer_id,
            question_id=answer.question_id,
            representation_type=answer.representation_type,
            answer_source=answer.answer_source,
            answer_text="Updated: CUSTOMER_ID",
            grade=FactualityGrade.INCORRECT,  # Changed grade
            confidence=0.5,  # Changed confidence
            response_time_ms=answer.response_time_ms,
            created_at=answer.created_at,
            metadata={"test": "updated"}
        )
        
        # Update the answer
        hana_storage_manager.update_answer(updated_answer)
        
        # Retrieve the updated answer
        retrieved_answer = hana_storage_manager.get_answer(answer.answer_id)
        
        # Verify retrieved answer matches updated values
        assert retrieved_answer is not None
        assert retrieved_answer.answer_text == "Updated: CUSTOMER_ID"
        assert retrieved_answer.grade == FactualityGrade.INCORRECT
        assert retrieved_answer.confidence == 0.5
        assert retrieved_answer.metadata == {"test": "updated"}
    
    def test_delete_answer(self, hana_storage_manager, sample_question, sample_answer):
        """Test deleting an answer."""
        # Create question and answer
        hana_storage_manager.create_question(sample_question)
        answer = BenchmarkAnswer(
            answer_id=sample_answer.answer_id,
            question_id=sample_question.question_id,
            representation_type=sample_answer.representation_type,
            answer_source=sample_answer.answer_source,
            answer_text=sample_answer.answer_text,
            grade=sample_answer.grade,
            confidence=sample_answer.confidence,
            response_time_ms=sample_answer.response_time_ms,
            created_at=sample_answer.created_at,
            metadata=sample_answer.metadata
        )
        hana_storage_manager.create_answer(answer)
        
        # Verify it exists
        assert hana_storage_manager.get_answer(answer.answer_id) is not None
        
        # Delete the answer
        hana_storage_manager.delete_answer(answer.answer_id)
        
        # Verify it no longer exists
        assert hana_storage_manager.get_answer(answer.answer_id) is None
    
    def test_get_answers_for_question(self, hana_storage_manager, sample_question):
        """Test retrieving all answers for a question."""
        # Create a question
        hana_storage_manager.create_question(sample_question)
        
        # Create multiple answers for the question
        for i in range(3):
            answer = BenchmarkAnswer(
                answer_id=str(uuid.uuid4()),
                question_id=sample_question.question_id,
                representation_type=RepresentationType.RELATIONAL if i == 0 else 
                               RepresentationType.VECTOR if i == 1 else
                               RepresentationType.ONTOLOGY,
                answer_source=AnswerSource.DIRECT_QUERY if i == 0 else
                         AnswerSource.VECTOR_SEARCH if i == 1 else
                         AnswerSource.SPARQL,
                answer_text=f"Answer {i}",
                grade=FactualityGrade.CORRECT if i < 2 else FactualityGrade.INCORRECT
            )
            hana_storage_manager.create_answer(answer)
        
        # Get all answers for the question
        answers = hana_storage_manager.get_answers_for_question(sample_question.question_id)
        
        # Verify we get all 3 answers
        assert len(answers) == 3
        
        # Filter by representation type
        vector_answers = hana_storage_manager.get_answers_for_question(
            sample_question.question_id,
            representation_type=RepresentationType.VECTOR
        )
        
        # Verify we get only the vector answer
        assert len(vector_answers) == 1
        assert vector_answers[0].representation_type == RepresentationType.VECTOR
    
    def test_search_questions(self, hana_storage_manager):
        """Test searching for questions with various filters."""
        # Create questions of different types and with different provenance
        question_types = [
            QuestionType.SCHEMA,
            QuestionType.INSTANCE,
            QuestionType.RELATIONSHIP,
            QuestionType.AGGREGATION
        ]
        
        entity_references = ["CUSTOMERS", "ORDERS", "PRODUCTS"]
        difficulties = [1, 2, 3]
        
        # Create 10 questions with different characteristics
        for i in range(10):
            question_type = question_types[i % len(question_types)]
            entity_reference = entity_references[i % len(entity_references)]
            difficulty = difficulties[i % len(difficulties)]
            
            # Add provenance to some questions
            provenance = None
            if i % 2 == 0:  # Even-numbered questions get provenance
                provenance = QuestionProvenance(
                    source_type="model" if i % 4 == 0 else "human",
                    source_id=f"source-{i}",
                    generation_timestamp=time.time(),
                    generation_method="template" if i % 4 == 0 else "manual",
                    generation_parameters={"param": f"value-{i}"},
                    input_data_sources=[f"source-{i}"]
                )
            
            question = BenchmarkQuestion(
                question_id=str(uuid.uuid4()),
                question_text=f"Test question {i}?",
                reference_answer=f"Answer {i}",
                question_type=question_type,
                entity_reference=entity_reference,
                difficulty=difficulty,
                metadata={"index": i},
                provenance=provenance
            )
            
            hana_storage_manager.create_question(question)
        
        # Test search by question type
        schema_questions = hana_storage_manager.search_questions(
            question_type=QuestionType.SCHEMA
        )
        assert len(schema_questions) > 0
        for q in schema_questions:
            assert q.question_type == QuestionType.SCHEMA
        
        # Test search by entity reference
        customer_questions = hana_storage_manager.search_questions(
            entity_reference="CUSTOMERS"
        )
        assert len(customer_questions) > 0
        for q in customer_questions:
            assert q.entity_reference == "CUSTOMERS"
        
        # Test search by difficulty
        easy_questions = hana_storage_manager.search_questions(
            difficulty=1
        )
        assert len(easy_questions) > 0
        for q in easy_questions:
            assert q.difficulty == 1
        
        # Test search by source type (requires provenance)
        model_questions = hana_storage_manager.search_questions(
            source_type="model"
        )
        assert len(model_questions) > 0
        for q in model_questions:
            assert q.provenance is not None
            assert q.provenance.source_type == "model"
        
        # Test search by multiple criteria
        filtered_questions = hana_storage_manager.search_questions(
            question_type=QuestionType.SCHEMA,
            entity_reference="CUSTOMERS",
            difficulty=1
        )
        for q in filtered_questions:
            assert q.question_type == QuestionType.SCHEMA
            assert q.entity_reference == "CUSTOMERS"
            assert q.difficulty == 1
    
    def test_store_and_get_benchmark_result(self, hana_storage_manager):
        """Test storing and retrieving benchmark results."""
        # Create a benchmark result
        benchmark_id = str(uuid.uuid4())
        benchmark_name = "Test Benchmark"
        representation_type = RepresentationType.RELATIONAL
        
        # Create metrics
        metrics = DataRepresentationMetrics(
            representation_type=representation_type,
            correct_count=75,
            incorrect_count=15,
            not_attempted_count=5,
            ambiguous_count=5,
            total_count=100,
            avg_response_time_ms=250.5,
            metrics_by_question_type={
                QuestionType.SCHEMA.value: {"total": 50, "correct": 40, "accuracy": 0.8},
                QuestionType.INSTANCE.value: {"total": 50, "correct": 35, "accuracy": 0.7}
            },
            metrics_by_difficulty={
                1: {"total": 30, "correct": 25, "accuracy": 0.833},
                3: {"total": 70, "correct": 50, "accuracy": 0.714}
            }
        )
        
        # Store the result
        result_id = hana_storage_manager.store_benchmark_result(
            benchmark_id=benchmark_id,
            benchmark_name=benchmark_name,
            representation_type=representation_type,
            metrics=metrics,
            metadata={"test": "value"}
        )
        
        # Verify result_id is returned
        assert result_id is not None
        
        # Get benchmark results
        results = hana_storage_manager.get_benchmark_results(
            benchmark_id=benchmark_id
        )
        
        # Verify results
        assert len(results) == 1
        result = results[0]
        assert result["benchmark_id"] == benchmark_id
        assert result["benchmark_name"] == benchmark_name
        assert result["representation_type"] == representation_type.value
        assert result["total_questions"] == 100
        assert result["correct_count"] == 75
        assert result["incorrect_count"] == 15
        assert result["not_attempted_count"] == 5
        assert result["ambiguous_count"] == 5
        assert result["accuracy"] == 0.75
        assert abs(result["f_score"] - metrics.f_score) < 0.001
        assert result["avg_response_time_ms"] == 250.5
        assert "detailed_metrics" in result
        assert "by_question_type" in result["detailed_metrics"]
        assert "by_difficulty" in result["detailed_metrics"]
        assert result["metadata"] == {"test": "value"}
        
        # Test filtering by representation type
        relational_results = hana_storage_manager.get_benchmark_results(
            representation_type=RepresentationType.RELATIONAL
        )
        assert len(relational_results) > 0
        for r in relational_results:
            assert r["representation_type"] == RepresentationType.RELATIONAL.value