"""
Pytest fixtures for benchmark system unit tests.

This module provides common fixtures used across benchmark system unit tests.
"""

import pytest
import sqlite3
import time
import uuid
from unittest.mock import MagicMock

from langchain_core.language_models import BaseLanguageModel
from langchain_hana.reasoning.benchmark_system import (
    BenchmarkQuestion,
    BenchmarkAnswer,
    QuestionType,
    RepresentationType,
    AnswerSource,
    FactualityGrade,
    QuestionProvenance,
    HanaStorageManager
)


@pytest.fixture
def mock_connection():
    """Create an in-memory SQLite database for testing."""
    conn = sqlite3.connect(":memory:")
    # Make SQLite more HANA-like by returning column names in results
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


@pytest.fixture
def hana_storage_manager(mock_connection):
    """Create a HanaStorageManager with in-memory SQLite database."""
    manager = HanaStorageManager(
        connection=mock_connection,
        schema_name="BENCHMARK_TEST",
        questions_table="TEST_QUESTIONS",
        answers_table="TEST_ANSWERS",
        results_table="TEST_RESULTS"
    )
    yield manager


@pytest.fixture
def sample_question():
    """Create a sample benchmark question for testing."""
    question_id = str(uuid.uuid4())
    return BenchmarkQuestion(
        question_id=question_id,
        question_text="What is the primary key of the CUSTOMERS table?",
        reference_answer="CUSTOMER_ID",
        question_type=QuestionType.SCHEMA,
        entity_reference="CUSTOMERS",
        difficulty=1,
        created_at=time.time(),
        metadata={"test": "value"}
    )


@pytest.fixture
def sample_question_with_provenance():
    """Create a sample benchmark question with provenance for testing."""
    question_id = str(uuid.uuid4())
    
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
    return BenchmarkQuestion(
        question_id=question_id,
        question_text="What is the primary key of the CUSTOMERS table?",
        reference_answer="CUSTOMER_ID",
        question_type=QuestionType.SCHEMA,
        entity_reference="CUSTOMERS",
        difficulty=1,
        created_at=time.time(),
        metadata={"test": "value"},
        provenance=provenance
    )


@pytest.fixture
def sample_answer():
    """Create a sample benchmark answer for testing."""
    answer_id = str(uuid.uuid4())
    question_id = str(uuid.uuid4())
    return BenchmarkAnswer(
        answer_id=answer_id,
        question_id=question_id,
        representation_type=RepresentationType.RELATIONAL,
        answer_source=AnswerSource.DIRECT_QUERY,
        answer_text="CUSTOMER_ID",
        grade=FactualityGrade.CORRECT,
        confidence=0.95,
        response_time_ms=150,
        created_at=time.time(),
        metadata={"test": "value"}
    )


@pytest.fixture
def mock_llm():
    """Create a mock language model for testing."""
    mock = MagicMock(spec=BaseLanguageModel)
    
    # Set up the invoke method to return a mock response
    mock_response = MagicMock()
    mock_response.content = """
    Question: What is the primary key of the CUSTOMERS table?
    Answer: CUSTOMER_ID
    Entity: CUSTOMERS
    Difficulty: 1
    
    Question: How many columns are in the ORDERS table?
    Answer: 5
    Entity: ORDERS
    Difficulty: 1
    """
    mock.invoke.return_value = mock_response
    
    return mock


@pytest.fixture
def sample_schema_info():
    """Sample schema information for testing."""
    return """
    Schema: SALES

    Tables:
    - CUSTOMERS
      Columns:
      - CUSTOMER_ID: INTEGER NOT NULL (Primary key for customer)
      - NAME: NVARCHAR(100) NOT NULL (Customer's full name)
      - EMAIL: VARCHAR(255) NOT NULL (Customer's email address)
      - PHONE: VARCHAR(20) NULL (Customer's phone number)
      - ADDRESS: NVARCHAR(200) NULL (Customer's physical address)
      - SIGNUP_DATE: DATE NOT NULL (Date when customer signed up)
      Primary Key: CUSTOMER_ID

    - ORDERS
      Columns:
      - ORDER_ID: INTEGER NOT NULL (Primary key for order)
      - CUSTOMER_ID: INTEGER NOT NULL (Customer who placed the order)
      - ORDER_DATE: TIMESTAMP NOT NULL (Date and time when order was placed)
      - STATUS: VARCHAR(20) NOT NULL (Order status)
      - TOTAL_AMOUNT: DECIMAL(12,2) NOT NULL (Total order amount)
      Primary Key: ORDER_ID
      Foreign Keys:
        - CUSTOMER_ID -> SALES.CUSTOMERS.CUSTOMER_ID
    """


@pytest.fixture
def sample_data_info():
    """Sample data information for testing."""
    return """
    Schema: SALES

    Table Data:

    Table: CUSTOMERS
    - Row count: 1000
    - Column: CUSTOMER_ID (INTEGER)
      - Min: 1
      - Max: 1000
      - Avg: 500.5
      - Distinct values: 1000
      - NULL count: 0
      - Sample values: 1, 2, 3, 4, 5
    - Column: NAME (NVARCHAR)
      - Distinct values: 997
      - NULL count: 0
      - Sample values: Aaron Smith, Alice Johnson, Bob Williams, Carol Davis, David Miller
    - Sample rows (5):
      - CUSTOMER_ID=1, NAME=Aaron Smith, EMAIL=aaron.smith@example.com
      - CUSTOMER_ID=2, NAME=Alice Johnson, EMAIL=alice.j@example.com
      - CUSTOMER_ID=3, NAME=Bob Williams, EMAIL=bob.williams@example.com
    
    Table: ORDERS
    - Row count: 2500
    - Column: ORDER_ID (INTEGER)
      - Min: 1
      - Max: 2500
      - Avg: 1250.5
      - Distinct values: 2500
      - NULL count: 0
      - Sample values: 1, 2, 3, 4, 5
    - Sample rows (5):
      - ORDER_ID=1, CUSTOMER_ID=42, ORDER_DATE=2020-01-01, STATUS=Completed
      - ORDER_ID=2, CUSTOMER_ID=87, ORDER_DATE=2020-01-01, STATUS=Completed
      - ORDER_ID=3, CUSTOMER_ID=15, ORDER_DATE=2020-01-01, STATUS=Completed
    """