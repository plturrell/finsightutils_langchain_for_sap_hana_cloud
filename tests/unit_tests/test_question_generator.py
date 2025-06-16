"""
Unit tests for the QuestionGenerator class.

This module contains unit tests for the QuestionGenerator class, which is responsible
for generating benchmark questions from SAP HANA schemas and data.
"""

import unittest
import pytest
from unittest.mock import MagicMock, patch

from langchain_hana.reasoning.benchmark_system import (
    QuestionGenerator,
    QuestionType,
    DifficultyLevel
)


class TestQuestionGenerator:
    """Tests for the QuestionGenerator class."""
    
    def test_initialization(self, mock_llm):
        """Test initialization with required parameters."""
        # Create question generator
        generator = QuestionGenerator(
            generation_model=mock_llm,
            model_id="test-model"
        )
        
        # Verify initialization
        assert generator.generation_model == mock_llm
        assert generator.model_id == "test-model"
        
        # Verify prompt templates are initialized
        assert generator.schema_prompt is not None
        assert generator.instance_prompt is not None
        assert generator.relationship_prompt is not None
        assert generator.aggregation_prompt is not None
        assert generator.inference_prompt is not None
        assert generator.temporal_prompt is not None
    
    def test_parse_generated_questions(self, mock_llm):
        """Test parsing of generated questions from model output."""
        # Create question generator
        generator = QuestionGenerator(
            generation_model=mock_llm,
            model_id="test-model"
        )
        
        # Sample model output
        model_output = """
        Question: What is the primary key of the CUSTOMERS table?
        Answer: CUSTOMER_ID
        Entity: CUSTOMERS
        Difficulty: 1
        
        Question: How many columns are in the ORDERS table?
        Answer: 5
        Entity: ORDERS
        Difficulty: 2
        
        Question: Which field contains customer email addresses?
        Answer: CUSTOMERS.EMAIL
        Entity: CUSTOMERS.EMAIL
        Difficulty: 1
        """
        
        # Parse questions
        parsed_questions = generator.parse_generated_questions(model_output, QuestionType.SCHEMA)
        
        # Verify parsing
        assert len(parsed_questions) == 3
        
        # Check first question
        q1 = parsed_questions[0]
        assert q1["question"] == "What is the primary key of the CUSTOMERS table?"
        assert q1["answer"] == "CUSTOMER_ID"
        assert q1["entity"] == "CUSTOMERS"
        assert q1["type"] == "schema"
        assert "difficulty" in q1
        assert "metadata" in q1
        assert "difficulty_assessment" in q1["metadata"]
        
        # Check second question
        q2 = parsed_questions[1]
        assert q2["question"] == "How many columns are in the ORDERS table?"
        assert q2["answer"] == "5"
        assert q2["entity"] == "ORDERS"
        assert q2["type"] == "schema"
        
        # Verify model-specified difficulty is stored in metadata
        assert "metadata" in q2
        assert "model_specified_difficulty" in q2["metadata"]
        assert q2["metadata"]["model_specified_difficulty"] == 2
        
        # Verify standardized difficulty assessment was applied
        assert "difficulty_assessment" in q2["metadata"]
        assert "standardized_difficulty" in q2["metadata"]["difficulty_assessment"]
    
    def test_validation_filters(self, mock_llm):
        """Test that validation filters out invalid questions."""
        # Create question generator
        generator = QuestionGenerator(
            generation_model=mock_llm,
            model_id="test-model"
        )
        
        # Sample model output with some invalid questions
        model_output = """
        Question: What is the primary key of the CUSTOMERS table?
        Answer: CUSTOMER_ID
        Entity: CUSTOMERS
        Difficulty: 1
        
        Question: 
        Answer: Missing question
        Entity: UNKNOWN
        Difficulty: 1
        
        Question: Short?
        Answer: Too short
        Entity: ORDERS
        Difficulty: 1
        
        Question: What is the SQL command to drop the CUSTOMERS table?
        Answer: DROP TABLE CUSTOMERS;
        Entity: CUSTOMERS
        Difficulty: 2
        """
        
        # Parse questions
        parsed_questions = generator.parse_generated_questions(model_output, QuestionType.SCHEMA)
        
        # Verify only valid questions are included
        assert len(parsed_questions) == 1  # Only the first question should pass validation
        assert parsed_questions[0]["question"] == "What is the primary key of the CUSTOMERS table?"
    
    def test_generate_schema_questions(self, mock_llm, sample_schema_info):
        """Test generation of schema questions."""
        # Create question generator
        generator = QuestionGenerator(
            generation_model=mock_llm,
            model_id="test-model"
        )
        
        # Generate questions
        questions = generator.generate_schema_questions(
            schema_info=sample_schema_info,
            num_questions=2
        )
        
        # Verify questions were generated
        assert len(questions) > 0
        for q in questions:
            assert "question" in q
            assert "answer" in q
            assert "entity" in q
            assert "type" in q
            assert q["type"] == "schema"
            assert "difficulty" in q
    
    def test_generate_instance_questions(self, mock_llm, sample_data_info):
        """Test generation of instance questions."""
        # Create question generator
        generator = QuestionGenerator(
            generation_model=mock_llm,
            model_id="test-model"
        )
        
        # Generate questions
        questions = generator.generate_instance_questions(
            data_info=sample_data_info,
            num_questions=2
        )
        
        # Verify questions were generated
        assert len(questions) > 0
        for q in questions:
            assert "question" in q
            assert "answer" in q
            assert "entity" in q
            assert "type" in q
            assert q["type"] == "instance"
            assert "difficulty" in q
    
    def test_create_benchmark_questions(self, mock_llm, sample_schema_info, sample_data_info):
        """Test creation of benchmark questions with provenance tracking."""
        # Create question generator
        generator = QuestionGenerator(
            generation_model=mock_llm,
            model_id="test-model"
        )
        
        # Create benchmark questions
        benchmark_questions = generator.create_benchmark_questions(
            schema_info=sample_schema_info,
            data_info=sample_data_info,
            num_schema_questions=2,
            num_instance_questions=2,
            num_relationship_questions=0,
            num_aggregation_questions=0,
            num_inference_questions=0,
            num_temporal_questions=0
        )
        
        # Verify benchmark questions were created
        assert len(benchmark_questions) > 0
        
        # Check question fields and provenance
        for question in benchmark_questions:
            assert question.question_id is not None
            assert question.question_text is not None
            assert question.reference_answer is not None
            assert question.question_type is not None
            assert question.entity_reference is not None
            assert 1 <= question.difficulty <= 5
            assert question.created_at is not None
            assert question.metadata is not None
            
            # Verify provenance tracking
            assert question.provenance is not None
            assert question.provenance.source_type == "model"
            assert question.provenance.source_id == "test-model"
            assert question.provenance.generation_timestamp is not None
            assert question.provenance.generation_method is not None
            assert question.provenance.generation_parameters is not None
            assert question.provenance.input_data_sources is not None
    
    def test_difficulty_assessment(self, mock_llm):
        """Test the difficulty assessment during question generation."""
        # Create question generator
        generator = QuestionGenerator(
            generation_model=mock_llm,
            model_id="test-model"
        )
        
        # Sample questions of different complexity
        questions_by_type = {
            QuestionType.SCHEMA: [
                "What is the primary key of the CUSTOMERS table?",  # Simple - BEGINNER
                "What are the foreign key relationships between ORDERS and CUSTOMERS tables?"  # More complex - BASIC/INTERMEDIATE
            ],
            QuestionType.INSTANCE: [
                "What is the email address of customer with ID 1?",  # Simple - BASIC
                "Find all customers who signed up in January 2023 and have placed at least 3 orders"  # Complex - ADVANCED
            ],
            QuestionType.AGGREGATION: [
                "How many orders are there in total?",  # Simple - INTERMEDIATE
                "Calculate the average order value by month for premium customers who have purchased at least 5 times"  # Complex - EXPERT
            ]
        }
        
        # Test each question type
        for q_type, questions in questions_by_type.items():
            # Test simple question
            simple_q = questions[0]
            simple_difficulty = DifficultyLevel.assess_difficulty(simple_q, q_type)
            
            # Test complex question
            complex_q = questions[1]
            complex_difficulty = DifficultyLevel.assess_difficulty(complex_q, q_type)
            
            # Complex question should have higher difficulty than simple one
            assert complex_difficulty > simple_difficulty, f"Complex question should have higher difficulty for {q_type}"
            
            # Get detailed assessment for complex question
            assessment = generator._get_difficulty_assessment(complex_q, q_type)
            
            # Verify assessment details
            assert assessment["question_text"] == complex_q
            assert assessment["question_type"] == q_type.value
            assert "base_difficulty" in assessment
            assert "complexity_indicators" in assessment
            assert "final_difficulty" in assessment
            
            # The final difficulty should match what assess_difficulty returns
            assert assessment["final_difficulty"]["score"] == int(complex_difficulty)
    
    def test_custom_prompt_templates(self):
        """Test using custom prompt templates."""
        # Create mock LLM
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = """
        Question: Custom question?
        Answer: Custom answer
        Entity: CUSTOM
        Difficulty: 2
        """
        mock_llm.invoke.return_value = mock_response
        
        # Create question generator with custom prompt
        generator = QuestionGenerator(
            generation_model=mock_llm,
            model_id="test-model"
        )
        
        # Override schema prompt with a custom template
        custom_template = "Generate {num_questions} custom questions about {schema_info}"
        generator.schema_prompt.template = custom_template
        
        # Generate questions
        generator.generate_schema_questions(
            schema_info="test schema",
            num_questions=1
        )
        
        # Verify the custom prompt was used
        call_args = mock_llm.invoke.call_args[0][0]
        assert "Generate 1 custom questions about test schema" in call_args
    
    def test_various_question_types(self, mock_llm, sample_schema_info, sample_data_info):
        """Test generation of all question types."""
        # Create question generator
        generator = QuestionGenerator(
            generation_model=mock_llm,
            model_id="test-model"
        )
        
        # Test all question generation methods
        schema_questions = generator.generate_schema_questions(
            schema_info=sample_schema_info,
            num_questions=1
        )
        assert len(schema_questions) > 0
        
        instance_questions = generator.generate_instance_questions(
            data_info=sample_data_info,
            num_questions=1
        )
        assert len(instance_questions) > 0
        
        relationship_questions = generator.generate_relationship_questions(
            schema_data_info=sample_schema_info + "\n\n" + sample_data_info,
            num_questions=1
        )
        assert len(relationship_questions) > 0
        
        aggregation_questions = generator.generate_aggregation_questions(
            data_info=sample_data_info,
            num_questions=1
        )
        assert len(aggregation_questions) > 0
        
        inference_questions = generator.generate_inference_questions(
            schema_data_info=sample_schema_info + "\n\n" + sample_data_info,
            num_questions=1
        )
        assert len(inference_questions) > 0
        
        temporal_questions = generator.generate_temporal_questions(
            temporal_data_info=sample_data_info,
            num_questions=1
        )
        assert len(temporal_questions) > 0