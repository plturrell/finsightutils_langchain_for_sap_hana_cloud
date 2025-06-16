"""
Representation handlers and benchmark runner for the SAP HANA benchmarking system.

This module extends the benchmark_system.py file with classes for handling
different data representation methods and running comprehensive benchmarks.
"""

import uuid
import time
import json
import logging
import datetime
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable, Type

import numpy as np
from hdbcli import dbapi
from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from langchain_hana.reasoning.factuality import FactualityEvaluator, FactualityGrade
from langchain_hana.reasoning.benchmark_system import (
    BenchmarkQuestion,
    BenchmarkAnswer,
    RepresentationType,
    QuestionType,
    AnswerSource,
    DataRepresentationMetrics,
    HanaStorageManager,
)

logger = logging.getLogger(__name__)


class BaseRepresentationHandler:
    """
    Base class for data representation handlers.
    
    This abstract class defines the interface for handlers that generate
    answers using different data representation methods.
    """
    
    def __init__(
        self,
        representation_type: RepresentationType,
        connection: dbapi.Connection,
        schema_name: str,
    ):
        """
        Initialize the representation handler.
        
        Args:
            representation_type: Type of data representation
            connection: SAP HANA database connection
            schema_name: Database schema name
        """
        self.representation_type = representation_type
        self.connection = connection
        self.schema_name = schema_name
    
    def answer_question(
        self,
        question: BenchmarkQuestion,
    ) -> BenchmarkAnswer:
        """
        Generate an answer for a benchmark question.
        
        Args:
            question: The question to answer
            
        Returns:
            The generated answer
            
        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement answer_question")
    
    def _create_answer(
        self,
        question_id: str,
        answer_text: str,
        answer_source: AnswerSource,
        confidence: Optional[float] = None,
        response_time_ms: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BenchmarkAnswer:
        """
        Create a benchmark answer.
        
        Args:
            question_id: ID of the question
            answer_text: The answer text
            answer_source: Source of the answer
            confidence: Optional confidence score
            response_time_ms: Optional response time in milliseconds
            metadata: Optional metadata
            
        Returns:
            The created answer
        """
        return BenchmarkAnswer(
            answer_id=str(uuid.uuid4()),
            question_id=question_id,
            representation_type=self.representation_type,
            answer_source=answer_source,
            answer_text=answer_text,
            confidence=confidence,
            response_time_ms=response_time_ms,
            metadata=metadata or {},
        )
    
    def supports_question_type(self, question_type: QuestionType) -> bool:
        """
        Check if this handler supports a question type.
        
        Args:
            question_type: Type of question
            
        Returns:
            True if supported, False otherwise
        """
        return True  # Default to supporting all question types


class RelationalRepresentationHandler(BaseRepresentationHandler):
    """
    Handler for relational data representation.
    
    Generates answers using SQL queries against relational tables.
    """
    
    def __init__(
        self,
        connection: dbapi.Connection,
        schema_name: str,
        llm: Optional[BaseLanguageModel] = None,
    ):
        """
        Initialize the relational representation handler.
        
        Args:
            connection: SAP HANA database connection
            schema_name: Database schema name
            llm: Optional language model for generating SQL queries
        """
        super().__init__(RepresentationType.RELATIONAL, connection, schema_name)
        self.llm = llm
        
        # SQL generation prompts
        self.sql_prompt_template = """
You are an expert SQL developer working with SAP HANA.
Generate a single SQL query to answer the following question about a database.

Database schema:
{schema_info}

Question: {question}

Rules:
1. Your answer must be a single, correct SQL query that will run on SAP HANA.
2. Do not include any explanations or comments, only the SQL query.
3. Use qualified names (schema.table) in your query.
4. The query should retrieve exactly the information needed to answer the question.

SQL query:
"""
    
    def answer_question(
        self,
        question: BenchmarkQuestion,
    ) -> BenchmarkAnswer:
        """
        Generate an answer using SQL queries.
        
        Args:
            question: The question to answer
            
        Returns:
            The generated answer
        """
        start_time = time.time()
        
        try:
            # Generate SQL query (either using LLM or predefined templates)
            sql_query = self._generate_sql_query(question)
            
            # Execute the query
            cursor = self.connection.cursor()
            cursor.execute(sql_query)
            result = cursor.fetchall()
            
            # Format the result as an answer
            if not result:
                answer_text = "No results found"
            elif len(result) == 1 and len(result[0]) == 1:
                # Single value result
                answer_text = str(result[0][0])
            else:
                # Multiple rows or columns
                answer_text = self._format_sql_result(result, cursor.description)
            
            response_time_ms = int((time.time() - start_time) * 1000)
            
            # Create and return the answer
            return self._create_answer(
                question_id=question.question_id,
                answer_text=answer_text,
                answer_source=AnswerSource.DIRECT_QUERY,
                response_time_ms=response_time_ms,
                metadata={
                    "sql_query": sql_query,
                    "raw_result": str(result),
                },
            )
        
        except Exception as e:
            logger.error(f"Error generating relational answer: {str(e)}")
            
            response_time_ms = int((time.time() - start_time) * 1000)
            
            # Return an error answer
            return self._create_answer(
                question_id=question.question_id,
                answer_text="I couldn't determine the answer due to a technical error.",
                answer_source=AnswerSource.DIRECT_QUERY,
                response_time_ms=response_time_ms,
                metadata={
                    "error": str(e),
                },
            )
    
    def _generate_sql_query(self, question: BenchmarkQuestion) -> str:
        """
        Generate an SQL query for a question.
        
        Args:
            question: The question to generate a query for
            
        Returns:
            The generated SQL query
        """
        if self.llm:
            # Use LLM to generate SQL
            schema_info = self._get_schema_info()
            prompt = self.sql_prompt_template.format(
                schema_info=schema_info,
                question=question.question_text,
            )
            
            response = self.llm.invoke(prompt)
            sql_query = response.content.strip()
            
            # Basic validation and cleanup
            if not sql_query.upper().startswith("SELECT"):
                raise ValueError(f"Generated query is not a SELECT statement: {sql_query}")
            
            return sql_query
        else:
            # Use predefined SQL templates based on question type and entity
            return self._get_template_sql_query(question)
    
    def _get_schema_info(self) -> str:
        """
        Get schema information for SQL generation.
        
        Returns:
            Formatted schema information
        """
        try:
            cursor = self.connection.cursor()
            
            # Get tables
            cursor.execute(f"""
            SELECT TABLE_NAME, COMMENTS
            FROM SYS.TABLES
            WHERE SCHEMA_NAME = '{self.schema_name}'
            ORDER BY TABLE_NAME
            """)
            
            tables = cursor.fetchall()
            
            schema_info = f"Schema: {self.schema_name}\n\n"
            schema_info += "Tables:\n"
            
            for table_name, comments in tables:
                schema_info += f"- {table_name}"
                if comments:
                    schema_info += f" ({comments})"
                schema_info += "\n"
                
                # Get columns for this table
                cursor.execute(f"""
                SELECT COLUMN_NAME, DATA_TYPE_NAME, LENGTH, SCALE, IS_NULLABLE, COMMENTS, POSITION
                FROM SYS.TABLE_COLUMNS
                WHERE SCHEMA_NAME = '{self.schema_name}' AND TABLE_NAME = '{table_name}'
                ORDER BY POSITION
                """)
                
                columns = cursor.fetchall()
                
                schema_info += "  Columns:\n"
                for col_name, data_type, length, scale, is_nullable, col_comments, position in columns:
                    nullable = "NULL" if is_nullable == "TRUE" else "NOT NULL"
                    schema_info += f"  - {col_name}: {data_type}"
                    
                    if data_type in ["VARCHAR", "NVARCHAR", "CHAR", "NCHAR"]:
                        schema_info += f"({length})"
                    elif data_type in ["DECIMAL"]:
                        schema_info += f"({length},{scale})"
                    
                    schema_info += f" {nullable}"
                    
                    if col_comments:
                        schema_info += f" ({col_comments})"
                    
                    schema_info += "\n"
                
                # Get primary key
                cursor.execute(f"""
                SELECT COLUMN_NAME
                FROM SYS.CONSTRAINTS AS C
                JOIN SYS.CONSTRAINT_COLUMNS AS CC ON C.SCHEMA_NAME = CC.SCHEMA_NAME AND C.TABLE_NAME = CC.TABLE_NAME AND C.CONSTRAINT_NAME = CC.CONSTRAINT_NAME
                WHERE C.SCHEMA_NAME = '{self.schema_name}' AND C.TABLE_NAME = '{table_name}' AND C.IS_PRIMARY_KEY = 'TRUE'
                ORDER BY CC.POSITION
                """)
                
                pk_columns = cursor.fetchall()
                
                if pk_columns:
                    pk_cols = [col[0] for col in pk_columns]
                    schema_info += f"  Primary Key: {', '.join(pk_cols)}\n"
                
                # Get foreign keys
                cursor.execute(f"""
                SELECT C.CONSTRAINT_NAME, CC.COLUMN_NAME, C.REFERENCED_SCHEMA_NAME, C.REFERENCED_TABLE_NAME, C.REFERENCED_COLUMN_NAME
                FROM SYS.CONSTRAINTS AS C
                JOIN SYS.CONSTRAINT_COLUMNS AS CC ON C.SCHEMA_NAME = CC.SCHEMA_NAME AND C.TABLE_NAME = CC.TABLE_NAME AND C.CONSTRAINT_NAME = CC.CONSTRAINT_NAME
                WHERE C.SCHEMA_NAME = '{self.schema_name}' AND C.TABLE_NAME = '{table_name}' AND C.IS_FOREIGN_KEY = 'TRUE'
                ORDER BY C.CONSTRAINT_NAME, CC.POSITION
                """)
                
                fk_constraints = cursor.fetchall()
                
                if fk_constraints:
                    schema_info += "  Foreign Keys:\n"
                    current_constraint = None
                    fk_columns = []
                    
                    for constraint_name, column_name, ref_schema, ref_table, ref_column in fk_constraints:
                        if constraint_name != current_constraint:
                            if current_constraint:
                                schema_info += f"    - {', '.join(fk_columns)} -> {ref_schema}.{ref_table}.{ref_column}\n"
                                fk_columns = []
                            current_constraint = constraint_name
                        
                        fk_columns.append(column_name)
                    
                    if fk_columns:
                        schema_info += f"    - {', '.join(fk_columns)} -> {ref_schema}.{ref_table}.{ref_column}\n"
                
                schema_info += "\n"
            
            return schema_info
        
        except Exception as e:
            logger.error(f"Error getting schema info: {str(e)}")
            return f"Schema: {self.schema_name}"
    
    def _get_template_sql_query(self, question: BenchmarkQuestion) -> str:
        """
        Get a template SQL query based on question type and entity.
        
        Args:
            question: The question to generate a query for
            
        Returns:
            SQL query template
        """
        # Extract key information for query generation
        question_text = question.question_text.lower()
        entity_reference = question.entity_reference
        question_type = question.question_type
        
        # Extract table and column names safely
        table_name = None
        column_name = None
        
        if "." in entity_reference:
            parts = entity_reference.split(".")
            if len(parts) >= 1:
                table_name = parts[0]
            if len(parts) >= 2:
                column_name = parts[1]
        else:
            # For entity references without a column specification
            table_name = entity_reference
        
        # Safety check to prevent SQL injection
        if table_name:
            if not self._is_valid_identifier(table_name):
                logger.warning(f"Invalid table name: {table_name}")
                return "SELECT 'Invalid table name' FROM DUMMY"
                
        if column_name:
            if not self._is_valid_identifier(column_name):
                logger.warning(f"Invalid column name: {column_name}")
                return "SELECT 'Invalid column name' FROM DUMMY"
        
        # Generate query based on question type
        if question_type == QuestionType.SCHEMA:
            return self._generate_schema_query(question_text, table_name, column_name)
        elif question_type == QuestionType.INSTANCE:
            return self._generate_instance_query(question_text, table_name, column_name)
        elif question_type == QuestionType.RELATIONSHIP:
            return self._generate_relationship_query(question_text, entity_reference)
        elif question_type == QuestionType.AGGREGATION:
            return self._generate_aggregation_query(question_text, table_name, column_name)
        elif question_type == QuestionType.INFERENCE:
            return self._generate_inference_query(question_text, entity_reference)
        elif question_type == QuestionType.TEMPORAL:
            return self._generate_temporal_query(question_text, table_name, column_name)
        
        # Default generic query
        return "SELECT 'Query not implemented for this question type' FROM DUMMY"
    
    def _is_valid_identifier(self, identifier: str) -> bool:
        """
        Check if a string is a valid SQL identifier.
        
        Args:
            identifier: The identifier to check
            
        Returns:
            True if valid, False otherwise
        """
        # Alphanumeric and underscore only, cannot start with a number
        if not identifier:
            return False
        
        if identifier[0].isdigit():
            return False
            
        return all(c.isalnum() or c == '_' for c in identifier)
        
    def _generate_schema_query(self, question_text: str, table_name: str, column_name: str = None) -> str:
        """
        Generate a query for schema questions.
        
        Args:
            question_text: The question text
            table_name: The table name
            column_name: Optional column name
            
        Returns:
            SQL query
        """
        if not table_name:
            return "SELECT 'Unable to determine table name' FROM DUMMY"
            
        # Primary key questions
        if any(term in question_text for term in ["primary key", "pk", "primary keys"]):
            return f"""
            SELECT COLUMN_NAME
            FROM SYS.CONSTRAINTS AS C
            JOIN SYS.CONSTRAINT_COLUMNS AS CC ON C.SCHEMA_NAME = CC.SCHEMA_NAME 
                AND C.TABLE_NAME = CC.TABLE_NAME 
                AND C.CONSTRAINT_NAME = CC.CONSTRAINT_NAME
            WHERE C.SCHEMA_NAME = '{self.schema_name}' 
                AND C.TABLE_NAME = '{table_name}' 
                AND C.IS_PRIMARY_KEY = 'TRUE'
            ORDER BY CC.POSITION
            """
            
        # Foreign key questions
        elif any(term in question_text for term in ["foreign key", "fk", "foreign keys", "references"]):
            return f"""
            SELECT CC.COLUMN_NAME, C.REFERENCED_SCHEMA_NAME, C.REFERENCED_TABLE_NAME, C.REFERENCED_COLUMN_NAME
            FROM SYS.CONSTRAINTS AS C
            JOIN SYS.CONSTRAINT_COLUMNS AS CC ON C.SCHEMA_NAME = CC.SCHEMA_NAME 
                AND C.TABLE_NAME = CC.TABLE_NAME 
                AND C.CONSTRAINT_NAME = CC.CONSTRAINT_NAME
            WHERE C.SCHEMA_NAME = '{self.schema_name}' 
                AND C.TABLE_NAME = '{table_name}' 
                AND C.IS_FOREIGN_KEY = 'TRUE'
            ORDER BY C.CONSTRAINT_NAME, CC.POSITION
            """
            
        # Data type questions
        elif any(term in question_text for term in ["data type", "column type", "datatype"]):
            if column_name:
                return f"""
                SELECT DATA_TYPE_NAME, LENGTH, SCALE, IS_NULLABLE, COMMENTS
                FROM SYS.TABLE_COLUMNS
                WHERE SCHEMA_NAME = '{self.schema_name}' 
                    AND TABLE_NAME = '{table_name}' 
                    AND COLUMN_NAME = '{column_name}'
                """
            else:
                # List all column types if no specific column is mentioned
                return f"""
                SELECT COLUMN_NAME, DATA_TYPE_NAME, LENGTH, SCALE, IS_NULLABLE
                FROM SYS.TABLE_COLUMNS
                WHERE SCHEMA_NAME = '{self.schema_name}' 
                    AND TABLE_NAME = '{table_name}'
                ORDER BY POSITION
                """
                
        # Column count questions
        elif any(term in question_text for term in ["how many columns", "number of columns", "column count"]):
            return f"""
            SELECT COUNT(*)
            FROM SYS.TABLE_COLUMNS
            WHERE SCHEMA_NAME = '{self.schema_name}' 
                AND TABLE_NAME = '{table_name}'
            """
            
        # Nullability questions
        elif any(term in question_text for term in ["nullable", "null", "not null"]):
            if column_name:
                return f"""
                SELECT IS_NULLABLE
                FROM SYS.TABLE_COLUMNS
                WHERE SCHEMA_NAME = '{self.schema_name}' 
                    AND TABLE_NAME = '{table_name}' 
                    AND COLUMN_NAME = '{column_name}'
                """
            else:
                return f"""
                SELECT COLUMN_NAME, IS_NULLABLE
                FROM SYS.TABLE_COLUMNS
                WHERE SCHEMA_NAME = '{self.schema_name}' 
                    AND TABLE_NAME = '{table_name}'
                ORDER BY POSITION
                """
                
        # Table existence questions
        elif any(term in question_text for term in ["exist", "exists", "present", "available"]):
            return f"""
            SELECT COUNT(*)
            FROM SYS.TABLES
            WHERE SCHEMA_NAME = '{self.schema_name}' 
                AND TABLE_NAME = '{table_name}'
            """
            
        # Default schema query
        return f"""
        SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE_NAME, IS_NULLABLE
        FROM SYS.TABLE_COLUMNS
        WHERE SCHEMA_NAME = '{self.schema_name}' 
            AND TABLE_NAME = '{table_name}'
        ORDER BY POSITION
        """
    
    def _generate_instance_query(self, question_text: str, table_name: str, column_name: str = None) -> str:
        """
        Generate a query for instance questions.
        
        Args:
            question_text: The question text
            table_name: The table name
            column_name: Optional column name
            
        Returns:
            SQL query
        """
        if not table_name:
            return "SELECT 'Unable to determine table name' FROM DUMMY"
            
        # Extract filter conditions from the question
        filter_conditions = []
        
        # Look for specific ID patterns (ID = X)
        id_match = re.search(r'id\s*[=:]\s*(\d+)', question_text)
        if id_match:
            id_value = id_match.group(1)
            # Find ID column (assuming it ends with _ID or is named ID)
            filter_conditions.append(f"(COLUMN_NAME LIKE '%ID' AND COLUMN_NAME = {id_value})")
            
        # Look for specific name patterns ("name is X")
        name_match = re.search(r'name\s*(?:is|=)\s*[\'"]?([a-zA-Z0-9\s]+)[\'"]?', question_text)
        if name_match:
            name_value = name_match.group(1).strip()
            filter_conditions.append(f"NAME = '{name_value}'")
            
        # Build the query
        if column_name:
            # Query for a specific column
            query = f"SELECT {column_name} FROM {self.schema_name}.{table_name}"
        else:
            # Query for all columns
            query = f"SELECT * FROM {self.schema_name}.{table_name}"
            
        # Add filter conditions if present
        if filter_conditions:
            query += " WHERE " + " AND ".join(filter_conditions)
            
        # Limit results
        query += " LIMIT 10"
        
        return query
    
    def _generate_relationship_query(self, question_text: str, entity_reference: str) -> str:
        """
        Generate a query for relationship questions.
        
        Args:
            question_text: The question text
            entity_reference: The entity reference
            
        Returns:
            SQL query
        """
        # Parse entity reference - expect format like "TABLE1_TABLE2" or "TABLE1,TABLE2"
        tables = re.split(r'[_,]', entity_reference)
        
        if len(tables) < 2:
            return "SELECT 'Unable to determine relationship tables' FROM DUMMY"
            
        # Ensure valid identifiers
        tables = [t for t in tables if self._is_valid_identifier(t)]
        
        if len(tables) < 2:
            return "SELECT 'Invalid table names in relationship' FROM DUMMY"
            
        # Get the join condition from foreign key constraints
        query = f"""
        WITH FK_INFO AS (
            SELECT 
                C.TABLE_NAME AS SOURCE_TABLE,
                CC.COLUMN_NAME AS SOURCE_COLUMN,
                C.REFERENCED_TABLE_NAME AS TARGET_TABLE,
                C.REFERENCED_COLUMN_NAME AS TARGET_COLUMN
            FROM SYS.CONSTRAINTS AS C
            JOIN SYS.CONSTRAINT_COLUMNS AS CC 
                ON C.SCHEMA_NAME = CC.SCHEMA_NAME 
                AND C.TABLE_NAME = CC.TABLE_NAME 
                AND C.CONSTRAINT_NAME = CC.CONSTRAINT_NAME
            WHERE C.SCHEMA_NAME = '{self.schema_name}' 
                AND C.IS_FOREIGN_KEY = 'TRUE'
                AND (
                    (C.TABLE_NAME = '{tables[0]}' AND C.REFERENCED_TABLE_NAME = '{tables[1]}')
                    OR
                    (C.TABLE_NAME = '{tables[1]}' AND C.REFERENCED_TABLE_NAME = '{tables[0]}')
                )
        )
        SELECT 
            FK_INFO.SOURCE_TABLE, 
            FK_INFO.SOURCE_COLUMN, 
            FK_INFO.TARGET_TABLE, 
            FK_INFO.TARGET_COLUMN
        FROM FK_INFO
        """
        
        return query
    
    def _generate_aggregation_query(self, question_text: str, table_name: str, column_name: str = None) -> str:
        """
        Generate a query for aggregation questions.
        
        Args:
            question_text: The question text
            table_name: The table name
            column_name: Optional column name
            
        Returns:
            SQL query
        """
        if not table_name:
            return "SELECT 'Unable to determine table name' FROM DUMMY"
            
        # Determine aggregation function
        agg_function = "COUNT(*)"
        
        if not column_name:
            # Try to extract column name from question text for aggregations other than COUNT
            # This is a simplification - in a real system, use NLP to extract this
            for col_match in re.finditer(r'(?:sum|average|avg|max|min|count)(?:\s+of)?\s+([a-zA-Z_]+)', question_text):
                potential_column = col_match.group(1)
                if self._is_valid_identifier(potential_column):
                    column_name = potential_column
                    break
        
        # Set aggregation function based on question
        if any(term in question_text for term in ["average", "avg", "mean"]):
            agg_function = f"AVG({column_name or '*'})"
        elif any(term in question_text for term in ["sum", "total"]):
            agg_function = f"SUM({column_name or '*'})"
        elif any(term in question_text for term in ["maximum", "max", "highest", "largest"]):
            agg_function = f"MAX({column_name or '*'})"
        elif any(term in question_text for term in ["minimum", "min", "lowest", "smallest"]):
            agg_function = f"MIN({column_name or '*'})"
        elif any(term in question_text for term in ["count", "how many", "number of"]):
            if column_name:
                agg_function = f"COUNT({column_name})"
            else:
                agg_function = "COUNT(*)"
        
        # Extract potential GROUP BY column
        group_by_clause = ""
        group_by_match = re.search(r'(?:group(?:ed)? by|per|by|for each)\s+([a-zA-Z_]+)', question_text)
        if group_by_match:
            group_by_col = group_by_match.group(1)
            if self._is_valid_identifier(group_by_col):
                group_by_clause = f"GROUP BY {group_by_col}"
                return f"""
                SELECT {group_by_col}, {agg_function} AS RESULT
                FROM {self.schema_name}.{table_name}
                {group_by_clause}
                ORDER BY {group_by_col}
                """
        
        # Basic aggregation query
        return f"""
        SELECT {agg_function} AS RESULT
        FROM {self.schema_name}.{table_name}
        """
    
    def _generate_inference_query(self, question_text: str, entity_reference: str) -> str:
        """
        Generate a query for inference questions.
        
        Args:
            question_text: The question text
            entity_reference: The entity reference
            
        Returns:
            SQL query
        """
        # Parse entity reference - expect comma-separated tables
        tables = entity_reference.split(",")
        tables = [t.strip() for t in tables if t.strip() and self._is_valid_identifier(t.strip())]
        
        if not tables:
            return "SELECT 'Unable to determine tables for inference' FROM DUMMY"
            
        # For inference questions, we need to join tables and perform analytics
        # This is a sophisticated query that tries to handle common inference patterns
        
        if len(tables) == 1:
            # Single table inference (typically involves grouping and filtering)
            table = tables[0]
            
            # Look for patterns like "most common", "most popular", etc.
            if any(term in question_text for term in ["most common", "most popular", "highest frequency"]):
                # Extract the target column from the question
                target_col = None
                for potential_col in ["category", "type", "status", "name", "product"]:
                    if potential_col in question_text:
                        target_col = potential_col.upper()
                        break
                
                if target_col:
                    return f"""
                    SELECT {target_col}, COUNT(*) AS FREQUENCY
                    FROM {self.schema_name}.{table}
                    GROUP BY {target_col}
                    ORDER BY FREQUENCY DESC
                    LIMIT 1
                    """
                else:
                    # If we can't identify a specific column, get frequencies for all columns
                    return f"""
                    SELECT 'Most common values analysis' AS ANALYSIS_TYPE,
                           COLUMN_NAME,
                           'Run specific query for details' AS RECOMMENDATION
                    FROM SYS.TABLE_COLUMNS
                    WHERE SCHEMA_NAME = '{self.schema_name}' 
                        AND TABLE_NAME = '{table}'
                        AND DATA_TYPE_NAME IN ('VARCHAR', 'NVARCHAR', 'CHAR', 'NCHAR')
                    ORDER BY POSITION
                    """
        
        elif len(tables) == 2:
            # Two-table inference (typically involves joins and aggregations)
            table1, table2 = tables[0], tables[1]
            
            # Try to find the join condition
            join_query = f"""
            SELECT 
                CC.COLUMN_NAME AS SOURCE_COLUMN,
                C.REFERENCED_COLUMN_NAME AS TARGET_COLUMN
            FROM SYS.CONSTRAINTS AS C
            JOIN SYS.CONSTRAINT_COLUMNS AS CC 
                ON C.SCHEMA_NAME = CC.SCHEMA_NAME 
                AND C.TABLE_NAME = CC.TABLE_NAME 
                AND C.CONSTRAINT_NAME = CC.CONSTRAINT_NAME
            WHERE C.SCHEMA_NAME = '{self.schema_name}' 
                AND C.IS_FOREIGN_KEY = 'TRUE'
                AND (
                    (C.TABLE_NAME = '{table1}' AND C.REFERENCED_TABLE_NAME = '{table2}')
                    OR
                    (C.TABLE_NAME = '{table2}' AND C.REFERENCED_TABLE_NAME = '{table1}')
                )
            LIMIT 1
            """
            
            # Return the analysis query
            return f"""
            WITH JOIN_INFO AS ({join_query})
            SELECT 
                'The system has identified a relationship between {table1} and {table2}.' AS INFERENCE_SETUP,
                'To answer this specific question, additional context about the business domain is needed.' AS RECOMMENDATION,
                'The following information may help structure a more specific query:' AS ADDITIONAL_INFO,
                SOURCE_COLUMN, TARGET_COLUMN
            FROM JOIN_INFO
            """
        
        else:
            # Multi-table inference (complex analytics)
            tables_list = ", ".join(tables)
            return f"""
            SELECT 'Complex inference across multiple tables requested' AS INFERENCE_TYPE,
                   '{tables_list}' AS TABLES_INVOLVED,
                   'This requires custom analytics based on business rules' AS RECOMMENDATION
            FROM DUMMY
            """
    
    def _generate_temporal_query(self, question_text: str, table_name: str, column_name: str = None) -> str:
        """
        Generate a query for temporal questions.
        
        Args:
            question_text: The question text
            table_name: The table name
            column_name: Optional column name
            
        Returns:
            SQL query
        """
        if not table_name:
            return "SELECT 'Unable to determine table name' FROM DUMMY"
            
        # Find date/time columns if not specified
        date_column_query = ""
        if not column_name:
            date_column_query = f"""
            SELECT COLUMN_NAME
            FROM SYS.TABLE_COLUMNS
            WHERE SCHEMA_NAME = '{self.schema_name}' 
                AND TABLE_NAME = '{table_name}'
                AND DATA_TYPE_NAME IN ('DATE', 'TIME', 'TIMESTAMP')
            ORDER BY POSITION
            LIMIT 1
            """
            # Use a placeholder for the column that will be filled in by the actual query execution
            column_name = "DATE_COLUMN_PLACEHOLDER"
        
        # Check for time period patterns
        time_period = None
        if any(term in question_text for term in ["year", "yearly", "annual"]):
            time_period = "YEAR"
        elif any(term in question_text for term in ["month", "monthly"]):
            time_period = "MONTH"
        elif any(term in question_text for term in ["quarter", "quarterly"]):
            time_period = "QUARTER"
        elif any(term in question_text for term in ["week", "weekly"]):
            time_period = "WEEK"
        elif any(term in question_text for term in ["day", "daily"]):
            time_period = "DAY"
        
        # Look for specific time frame
        year_match = re.search(r'\b(20\d{2})\b', question_text)
        year = year_match.group(1) if year_match else None
        
        month_names = ["january", "february", "march", "april", "may", "june", 
                      "july", "august", "september", "october", "november", "december"]
        month = None
        for i, name in enumerate(month_names, 1):
            if name in question_text.lower():
                month = i
                break
        
        # Generate query based on question patterns
        if "trend" in question_text or "pattern" in question_text:
            # Trend analysis query
            if time_period:
                return f"""
                SELECT 
                    {time_period}({column_name}) AS TIME_PERIOD,
                    COUNT(*) AS COUNT
                FROM {self.schema_name}.{table_name}
                GROUP BY {time_period}({column_name})
                ORDER BY TIME_PERIOD
                """
        
        elif any(term in question_text for term in ["busiest", "highest", "most", "peak"]):
            # Peak period query
            if time_period:
                return f"""
                SELECT 
                    {time_period}({column_name}) AS TIME_PERIOD,
                    COUNT(*) AS COUNT
                FROM {self.schema_name}.{table_name}
                GROUP BY {time_period}({column_name})
                ORDER BY COUNT DESC
                LIMIT 1
                """
        
        elif year and month:
            # Specific year and month query
            return f"""
            SELECT *
            FROM {self.schema_name}.{table_name}
            WHERE YEAR({column_name}) = {year}
            AND MONTH({column_name}) = {month}
            """
        
        elif year:
            # Specific year query
            return f"""
            SELECT *
            FROM {self.schema_name}.{table_name}
            WHERE YEAR({column_name}) = {year}
            """
        
        # Default temporal query
        return f"""
        SELECT 
            {column_name},
            COUNT(*) AS COUNT
        FROM {self.schema_name}.{table_name}
        GROUP BY {column_name}
        ORDER BY {column_name}
        """
    
    def _format_sql_result(self, result: List[Tuple], description: List[Tuple]) -> str:
        """
        Format SQL query result as a readable answer.
        
        Args:
            result: Query result rows
            description: Column descriptions
            
        Returns:
            Formatted answer text
        """
        if not result:
            return "No results found"
        
        # Get column names
        column_names = [col[0] for col in description]
        
        # For a single row result, format as a simple list of values
        if len(result) == 1:
            row = result[0]
            if len(row) == 1:
                return str(row[0])
            else:
                return ", ".join(f"{name}: {value}" for name, value in zip(column_names, row))
        
        # For multiple rows, summarize
        if len(result) <= 5:
            # Show all rows for small result sets
            rows_text = []
            for row in result:
                row_text = ", ".join(f"{name}: {value}" for name, value in zip(column_names, row))
                rows_text.append(row_text)
            
            return "\n".join(rows_text)
        else:
            # Summarize for larger result sets
            return f"{len(result)} rows returned. First row: {', '.join(f'{name}: {value}' for name, value in zip(column_names, result[0]))}"


class VectorRepresentationHandler(BaseRepresentationHandler):
    """
    Handler for vector data representation.
    
    Generates answers using vector similarity search.
    """
    
    def __init__(
        self,
        connection: dbapi.Connection,
        schema_name: str,
        embeddings: Embeddings,
        vector_store: VectorStore,
        llm: Optional[BaseLanguageModel] = None,
    ):
        """
        Initialize the vector representation handler.
        
        Args:
            connection: SAP HANA database connection
            schema_name: Database schema name
            embeddings: Embeddings model
            vector_store: Vector store
            llm: Optional language model for generating answers
        """
        super().__init__(RepresentationType.VECTOR, connection, schema_name)
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.llm = llm
    
    def answer_question(
        self,
        question: BenchmarkQuestion,
    ) -> BenchmarkAnswer:
        """
        Generate an answer using vector similarity search.
        
        Args:
            question: The question to answer
            
        Returns:
            The generated answer
        """
        start_time = time.time()
        
        try:
            # Generate query embedding
            query = question.question_text
            
            # Search for similar documents
            docs = self.vector_store.similarity_search(query, k=5)
            
            if not docs:
                answer_text = "No relevant information found"
            elif self.llm:
                # Use LLM to generate answer from retrieved documents
                context = "\n\n".join(doc.page_content for doc in docs)
                prompt = f"""
Answer the following question based only on the provided context:

Context:
{context}

Question: {query}

Answer the question directly and concisely. If the context doesn't contain the information needed to answer the question, say "I don't have enough information to answer this question."
"""
                response = self.llm.invoke(prompt)
                answer_text = response.content.strip()
            else:
                # Simple answer based on most similar document
                answer_text = docs[0].page_content
            
            response_time_ms = int((time.time() - start_time) * 1000)
            
            # Create and return the answer
            return self._create_answer(
                question_id=question.question_id,
                answer_text=answer_text,
                answer_source=AnswerSource.VECTOR_SEARCH,
                response_time_ms=response_time_ms,
                metadata={
                    "num_docs_retrieved": len(docs),
                    "top_doc_score": getattr(docs[0].metadata, "score", None) if docs else None,
                },
            )
        
        except Exception as e:
            logger.error(f"Error generating vector answer: {str(e)}")
            
            response_time_ms = int((time.time() - start_time) * 1000)
            
            # Return an error answer
            return self._create_answer(
                question_id=question.question_id,
                answer_text="I couldn't determine the answer due to a technical error.",
                answer_source=AnswerSource.VECTOR_SEARCH,
                response_time_ms=response_time_ms,
                metadata={
                    "error": str(e),
                },
            )
    
    def supports_question_type(self, question_type: QuestionType) -> bool:
        """
        Check if this handler supports a question type.
        
        Args:
            question_type: Type of question
            
        Returns:
            True if supported, False otherwise
        """
        # Vector search works well for these question types
        return question_type in [
            QuestionType.INSTANCE,
            QuestionType.RELATIONSHIP,
            QuestionType.INFERENCE,
        ]


class OntologyRepresentationHandler(BaseRepresentationHandler):
    """
    Handler for ontology/OWL data representation.
    
    Generates answers using SPARQL queries against an ontology.
    """
    
    def __init__(
        self,
        connection: dbapi.Connection,
        schema_name: str,
        sparql_endpoint: str,
        llm: Optional[BaseLanguageModel] = None,
    ):
        """
        Initialize the ontology representation handler.
        
        Args:
            connection: SAP HANA database connection
            schema_name: Database schema name
            sparql_endpoint: SPARQL endpoint URL
            llm: Optional language model for generating SPARQL queries
        """
        super().__init__(RepresentationType.ONTOLOGY, connection, schema_name)
        self.sparql_endpoint = sparql_endpoint
        self.llm = llm
        
        # SPARQL generation prompts
        self.sparql_prompt_template = """
You are an expert in semantic web technologies and SPARQL.
Generate a single SPARQL query to answer the following question about an ontology.

Ontology information:
{ontology_info}

Question: {question}

Rules:
1. Your answer must be a single, correct SPARQL query.
2. Do not include any explanations or comments, only the SPARQL query.
3. The query should retrieve exactly the information needed to answer the question.

SPARQL query:
"""
    
    def answer_question(
        self,
        question: BenchmarkQuestion,
    ) -> BenchmarkAnswer:
        """
        Generate an answer using SPARQL queries.
        
        Args:
            question: The question to answer
            
        Returns:
            The generated answer
        """
        start_time = time.time()
        
        try:
            # Generate SPARQL query
            sparql_query = self._generate_sparql_query(question)
            
            # Execute the query (simplified implementation)
            result = self._execute_sparql_query(sparql_query)
            
            # Format the result as an answer
            if not result:
                answer_text = "No results found"
            elif len(result) == 1 and len(result[0]) == 1:
                # Single value result
                answer_text = str(result[0][0])
            else:
                # Multiple rows or columns
                answer_text = self._format_sparql_result(result)
            
            response_time_ms = int((time.time() - start_time) * 1000)
            
            # Create and return the answer
            return self._create_answer(
                question_id=question.question_id,
                answer_text=answer_text,
                answer_source=AnswerSource.SPARQL,
                response_time_ms=response_time_ms,
                metadata={
                    "sparql_query": sparql_query,
                    "raw_result": str(result),
                },
            )
        
        except Exception as e:
            logger.error(f"Error generating ontology answer: {str(e)}")
            
            response_time_ms = int((time.time() - start_time) * 1000)
            
            # Return an error answer
            return self._create_answer(
                question_id=question.question_id,
                answer_text="I couldn't determine the answer due to a technical error.",
                answer_source=AnswerSource.SPARQL,
                response_time_ms=response_time_ms,
                metadata={
                    "error": str(e),
                },
            )
    
    def _generate_sparql_query(self, question: BenchmarkQuestion) -> str:
        """
        Generate a SPARQL query for a question.
        
        Args:
            question: The question to generate a query for
            
        Returns:
            The generated SPARQL query
        """
        if self.llm:
            # Use LLM to generate SPARQL
            ontology_info = self._get_ontology_info()
            prompt = self.sparql_prompt_template.format(
                ontology_info=ontology_info,
                question=question.question_text,
            )
            
            response = self.llm.invoke(prompt)
            sparql_query = response.content.strip()
            
            # Basic validation and cleanup
            if not sparql_query.upper().startswith("SELECT") and not sparql_query.upper().startswith("ASK"):
                raise ValueError(f"Generated query is not a valid SPARQL query: {sparql_query}")
            
            return sparql_query
        else:
            # Use predefined SPARQL templates based on question type and entity
            return self._get_template_sparql_query(question)
    
    def _get_ontology_info(self) -> str:
        """
        Get ontology information for SPARQL generation.
        
        Returns:
            Formatted ontology information
        """
        # In a real implementation, this would query the ontology to get class and property information
        return """
        Classes:
        - Customer
        - Order
        - Product
        - OrderItem
        
        Properties:
        - hasOrder (Customer -> Order)
        - containsItem (Order -> OrderItem)
        - refersToProduct (OrderItem -> Product)
        - hasName (Customer -> xsd:string)
        - hasEmail (Customer -> xsd:string)
        - hasPrice (Product -> xsd:decimal)
        - hasQuantity (OrderItem -> xsd:integer)
        """
    
    def _get_template_sparql_query(self, question: BenchmarkQuestion) -> str:
        """
        Get a template SPARQL query based on question type and entity.
        
        Args:
            question: The question to generate a query for
            
        Returns:
            SPARQL query template
        """
        # This is a simplified implementation - in a real system,
        # you would have more sophisticated templates for different question types
        
        if question.question_type == QuestionType.INSTANCE:
            # Instance questions look up specific values
            return """
            SELECT ?value
            WHERE {
                ?entity rdf:type ont:Entity ;
                       ont:hasProperty ?value .
            }
            LIMIT 1
            """
        
        elif question.question_type == QuestionType.RELATIONSHIP:
            # Relationship questions query relationships between entities
            return """
            SELECT ?related
            WHERE {
                ?entity1 rdf:type ont:Entity1 ;
                         ont:relatesTo ?related .
                ?related rdf:type ont:Entity2 .
            }
            """
        
        elif question.question_type == QuestionType.INFERENCE:
            # Inference questions use more complex patterns
            return """
            SELECT ?inferred
            WHERE {
                ?entity1 rdf:type ont:Entity1 ;
                         ont:property1 ?value1 .
                ?entity2 rdf:type ont:Entity2 ;
                         ont:property2 ?value2 .
                FILTER (?value1 > ?value2)
                ?entity1 ont:relatesTo ?inferred .
            }
            """
        
        # Default generic query
        return """
        SELECT ?subject ?predicate ?object
        WHERE {
            ?subject ?predicate ?object .
        }
        LIMIT 10
        """
    
    def _execute_sparql_query(self, sparql_query: str) -> List[Tuple]:
        """
        Execute a SPARQL query.
        
        Args:
            sparql_query: SPARQL query to execute
            
        Returns:
            Query results
        """
        # This is a mock implementation - in a real system,
        # you would use a SPARQL client to execute the query against your endpoint
        
        logger.info(f"Would execute SPARQL query: {sparql_query}")
        
        # Return mock results based on query type
        if "COUNT" in sparql_query.upper():
            return [("42",)]
        elif "ASK" in sparql_query.upper():
            return [("true",)]
        elif "LIMIT 1" in sparql_query:
            return [("Sample Result",)]
        else:
            return [("Result 1",), ("Result 2",)]
    
    def _format_sparql_result(self, result: List[Tuple]) -> str:
        """
        Format SPARQL query result as a readable answer.
        
        Args:
            result: Query result rows
            
        Returns:
            Formatted answer text
        """
        if not result:
            return "No results found"
        
        # For a single row result, format as a simple string
        if len(result) == 1:
            row = result[0]
            if len(row) == 1:
                return str(row[0])
            else:
                return ", ".join(str(value) for value in row)
        
        # For multiple rows, format as a list
        return ", ".join(str(row[0]) for row in result)
    
    def supports_question_type(self, question_type: QuestionType) -> bool:
        """
        Check if this handler supports a question type.
        
        Args:
            question_type: Type of question
            
        Returns:
            True if supported, False otherwise
        """
        # Ontology representations work well for these question types
        return question_type in [
            QuestionType.RELATIONSHIP,
            QuestionType.INFERENCE,
            QuestionType.TEMPORAL,
        ]


class HybridRepresentationHandler(BaseRepresentationHandler):
    """
    Handler for hybrid data representation.
    
    Combines multiple representation methods to generate answers.
    """
    
    def __init__(
        self,
        connection: dbapi.Connection,
        schema_name: str,
        handlers: List[BaseRepresentationHandler],
        llm: BaseLanguageModel,
    ):
        """
        Initialize the hybrid representation handler.
        
        Args:
            connection: SAP HANA database connection
            schema_name: Database schema name
            handlers: List of representation handlers
            llm: Language model for combining answers
        """
        super().__init__(RepresentationType.HYBRID, connection, schema_name)
        self.handlers = handlers
        self.llm = llm
    
    def answer_question(
        self,
        question: BenchmarkQuestion,
    ) -> BenchmarkAnswer:
        """
        Generate an answer using multiple representation methods.
        
        Args:
            question: The question to answer
            
        Returns:
            The generated answer
        """
        start_time = time.time()
        
        try:
            # Get answers from all handlers that support this question type
            handler_answers = []
            
            for handler in self.handlers:
                if handler.supports_question_type(question.question_type):
                    answer = handler.answer_question(question)
                    handler_answers.append({
                        "representation_type": handler.representation_type.value,
                        "answer_text": answer.answer_text,
                        "answer_source": answer.answer_source.value,
                        "response_time_ms": answer.response_time_ms,
                    })
            
            if not handler_answers:
                answer_text = "No representation handler could answer this question."
            else:
                # Combine answers using LLM
                prompt = f"""
You are an expert knowledge system that combines answers from different data representation methods.
Select or synthesize the best answer to the following question based on multiple answers from different systems.

Question: {question.question_text}

Answers from different systems:
"""
                
                for i, ha in enumerate(handler_answers, 1):
                    prompt += f"\n{i}. {ha['representation_type']} ({ha['answer_source']}): {ha['answer_text']}"
                
                prompt += """

Based on these answers, provide the single most accurate answer to the question.
If the answers conflict, explain briefly which one you believe is correct and why.
If no system provided a good answer, say "I don't have enough information to answer this question."

Answer:
"""
                
                response = self.llm.invoke(prompt)
                answer_text = response.content.strip()
            
            response_time_ms = int((time.time() - start_time) * 1000)
            
            # Create and return the answer
            return self._create_answer(
                question_id=question.question_id,
                answer_text=answer_text,
                answer_source=AnswerSource.HYBRID,
                response_time_ms=response_time_ms,
                metadata={
                    "handler_answers": handler_answers,
                },
            )
        
        except Exception as e:
            logger.error(f"Error generating hybrid answer: {str(e)}")
            
            response_time_ms = int((time.time() - start_time) * 1000)
            
            # Return an error answer
            return self._create_answer(
                question_id=question.question_id,
                answer_text="I couldn't determine the answer due to a technical error.",
                answer_source=AnswerSource.HYBRID,
                response_time_ms=response_time_ms,
                metadata={
                    "error": str(e),
                },
            )
    
    def supports_question_type(self, question_type: QuestionType) -> bool:
        """
        Check if this handler supports a question type.
        
        Args:
            question_type: Type of question
            
        Returns:
            True if supported, False otherwise
        """
        # Hybrid approach supports all question types if at least one handler does
        return any(h.supports_question_type(question_type) for h in self.handlers)


class BenchmarkRunner:
    """
    Runner for executing comprehensive benchmarks.
    
    This class orchestrates the execution of benchmarks across different
    data representation methods, manages grading, and calculates metrics.
    """
    
    def __init__(
        self,
        storage_manager: HanaStorageManager,
        evaluator: FactualityEvaluator,
        handlers: Dict[RepresentationType, BaseRepresentationHandler],
        benchmark_id: Optional[str] = None,
        benchmark_name: Optional[str] = None,
    ):
        """
        Initialize the benchmark runner.
        
        Args:
            storage_manager: Storage manager for benchmark data
            evaluator: Evaluator for grading answers
            handlers: Dictionary of representation handlers
            benchmark_id: Optional benchmark ID
            benchmark_name: Optional benchmark name
        """
        self.storage_manager = storage_manager
        self.evaluator = evaluator
        self.handlers = handlers
        self.benchmark_id = benchmark_id or str(uuid.uuid4())
        self.benchmark_name = benchmark_name or f"Benchmark-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    def run_benchmark(
        self,
        questions: List[BenchmarkQuestion],
        representation_types: Optional[List[RepresentationType]] = None,
    ) -> Dict[RepresentationType, DataRepresentationMetrics]:
        """
        Run a benchmark on a set of questions.
        
        Args:
            questions: List of benchmark questions
            representation_types: Optional list of representation types to benchmark
            
        Returns:
            Dictionary of metrics by representation type
        """
        if not representation_types:
            representation_types = list(self.handlers.keys())
        
        # Store questions in the database
        logger.info(f"Storing {len(questions)} questions in the database")
        for question in questions:
            try:
                self.storage_manager.create_question(question)
            except Exception as e:
                logger.warning(f"Error storing question {question.question_id}: {str(e)}")
        
        # Initialize metrics for each representation type
        metrics = {rt: DataRepresentationMetrics(representation_type=rt) for rt in representation_types}
        
        # Initialize metrics by question type
        for rt in representation_types:
            question_types = set(q.question_type for q in questions)
            for qt in question_types:
                metrics[rt].metrics_by_question_type[qt.value] = {
                    "total": 0,
                    "correct": 0,
                    "incorrect": 0,
                    "not_attempted": 0,
                    "ambiguous": 0,
                    "accuracy": 0.0,
                }
            
            # Initialize metrics by difficulty
            difficulties = set(q.difficulty for q in questions)
            for difficulty in difficulties:
                metrics[rt].metrics_by_difficulty[difficulty] = {
                    "total": 0,
                    "correct": 0,
                    "incorrect": 0,
                    "not_attempted": 0,
                    "ambiguous": 0,
                    "accuracy": 0.0,
                }
        
        # Process each question for each representation type
        for question in questions:
            logger.info(f"Processing question: {question.question_text}")
            
            for rt in representation_types:
                if rt not in self.handlers:
                    logger.warning(f"No handler for representation type {rt}")
                    continue
                
                handler = self.handlers[rt]
                
                # Skip if the handler doesn't support this question type
                if not handler.supports_question_type(question.question_type):
                    logger.info(f"Handler {rt} doesn't support question type {question.question_type}")
                    continue
                
                try:
                    # Generate answer
                    logger.info(f"Generating answer using {rt} representation")
                    answer = handler.answer_question(question)
                    
                    # Grade the answer
                    logger.info("Grading answer")
                    grade, grading_notes = self.evaluator.grade_answer(
                        question=question.question_text,
                        reference_answer=question.reference_answer,
                        model_answer=answer.answer_text,
                    )
                    
                    answer.grade = grade
                    answer.metadata["grading_notes"] = grading_notes
                    
                    # Store the answer
                    try:
                        self.storage_manager.create_answer(answer)
                    except Exception as e:
                        logger.warning(f"Error storing answer {answer.answer_id}: {str(e)}")
                    
                    # Update metrics
                    metrics[rt].total_count += 1
                    
                    if grade == FactualityGrade.CORRECT:
                        metrics[rt].correct_count += 1
                    elif grade == FactualityGrade.INCORRECT:
                        metrics[rt].incorrect_count += 1
                    elif grade == FactualityGrade.NOT_ATTEMPTED:
                        metrics[rt].not_attempted_count += 1
                    elif grade == FactualityGrade.AMBIGUOUS:
                        metrics[rt].ambiguous_count += 1
                    
                    # Update metrics by question type
                    qt_metrics = metrics[rt].metrics_by_question_type[question.question_type.value]
                    qt_metrics["total"] += 1
                    
                    if grade == FactualityGrade.CORRECT:
                        qt_metrics["correct"] += 1
                    elif grade == FactualityGrade.INCORRECT:
                        qt_metrics["incorrect"] += 1
                    elif grade == FactualityGrade.NOT_ATTEMPTED:
                        qt_metrics["not_attempted"] += 1
                    elif grade == FactualityGrade.AMBIGUOUS:
                        qt_metrics["ambiguous"] += 1
                    
                    # Update metrics by difficulty
                    diff_metrics = metrics[rt].metrics_by_difficulty[question.difficulty]
                    diff_metrics["total"] += 1
                    
                    if grade == FactualityGrade.CORRECT:
                        diff_metrics["correct"] += 1
                    elif grade == FactualityGrade.INCORRECT:
                        diff_metrics["incorrect"] += 1
                    elif grade == FactualityGrade.NOT_ATTEMPTED:
                        diff_metrics["not_attempted"] += 1
                    elif grade == FactualityGrade.AMBIGUOUS:
                        diff_metrics["ambiguous"] += 1
                    
                    # Update response time metrics
                    if answer.response_time_ms is not None:
                        if metrics[rt].avg_response_time_ms is None:
                            metrics[rt].avg_response_time_ms = answer.response_time_ms
                        else:
                            # Running average
                            count = metrics[rt].total_count
                            metrics[rt].avg_response_time_ms = (
                                (metrics[rt].avg_response_time_ms * (count - 1) + answer.response_time_ms) / count
                            )
                
                except Exception as e:
                    logger.error(f"Error processing question with {rt} handler: {str(e)}")
        
        # Calculate final metrics
        for rt in representation_types:
            # Update accuracy for question types
            for qt, qt_metrics in metrics[rt].metrics_by_question_type.items():
                if qt_metrics["total"] > 0:
                    qt_metrics["accuracy"] = qt_metrics["correct"] / qt_metrics["total"]
            
            # Update accuracy for difficulties
            for diff, diff_metrics in metrics[rt].metrics_by_difficulty.items():
                if diff_metrics["total"] > 0:
                    diff_metrics["accuracy"] = diff_metrics["correct"] / diff_metrics["total"]
            
            # Store benchmark results
            try:
                self.storage_manager.store_benchmark_result(
                    benchmark_id=self.benchmark_id,
                    benchmark_name=self.benchmark_name,
                    representation_type=rt,
                    metrics=metrics[rt],
                    metadata={
                        "timestamp": time.time(),
                        "num_questions": len(questions),
                    },
                )
            except Exception as e:
                logger.warning(f"Error storing benchmark results for {rt}: {str(e)}")
        
        return metrics
    
    def get_recommendations(
        self,
        metrics: Dict[RepresentationType, DataRepresentationMetrics],
    ) -> Dict[str, Any]:
        """
        Generate recommendations based on benchmark results.
        
        Args:
            metrics: Dictionary of metrics by representation type
            
        Returns:
            Recommendations for improving accuracy
        """
        recommendations = {
            "overall": [],
            "by_representation": {},
            "by_question_type": {},
            "performance": [],
        }
        
        # Find best and worst performing representation types
        if metrics:
            rt_by_accuracy = sorted(
                metrics.keys(),
                key=lambda rt: metrics[rt].accuracy,
                reverse=True,
            )
            
            best_rt = rt_by_accuracy[0]
            worst_rt = rt_by_accuracy[-1]
            
            # Overall recommendations
            recommendations["overall"].append({
                "recommendation": f"Use {best_rt.value} representation for best overall accuracy ({metrics[best_rt].accuracy:.2%})",
                "priority": "high",
                "evidence": f"Benchmark results show {best_rt.value} achieves {metrics[best_rt].accuracy:.2%} accuracy across all questions",
            })
            
            # Recommendations by representation type
            for rt, rt_metrics in metrics.items():
                rt_recommendations = []
                
                # Check accuracy threshold
                if rt_metrics.accuracy < 0.7:
                    rt_recommendations.append({
                        "recommendation": f"Improve {rt.value} representation accuracy",
                        "priority": "high" if rt_metrics.accuracy < 0.5 else "medium",
                        "evidence": f"Current accuracy is {rt_metrics.accuracy:.2%}, which is below the target of 70%",
                    })
                
                # Check attempted questions
                attempted_ratio = (rt_metrics.total_count - rt_metrics.not_attempted_count) / rt_metrics.total_count if rt_metrics.total_count > 0 else 0
                if attempted_ratio < 0.9:
                    rt_recommendations.append({
                        "recommendation": f"Increase {rt.value} representation's question coverage",
                        "priority": "medium",
                        "evidence": f"{rt_metrics.not_attempted_count} out of {rt_metrics.total_count} questions were not attempted",
                    })
                
                # Question type specific recommendations
                for qt, qt_metrics in rt_metrics.metrics_by_question_type.items():
                    if qt_metrics["total"] > 0 and qt_metrics["accuracy"] < 0.6:
                        rt_recommendations.append({
                            "recommendation": f"Improve {rt.value} representation for {qt} questions",
                            "priority": "medium",
                            "evidence": f"Accuracy for {qt} questions is {qt_metrics['accuracy']:.2%}",
                        })
                
                recommendations["by_representation"][rt.value] = rt_recommendations
            
            # Recommendations by question type
            for qt in QuestionType:
                qt_value = qt.value
                qt_recommendations = []
                
                # Find best representation for this question type
                best_rt_for_qt = None
                best_accuracy = 0
                
                for rt, rt_metrics in metrics.items():
                    if qt_value in rt_metrics.metrics_by_question_type:
                        qt_metrics = rt_metrics.metrics_by_question_type[qt_value]
                        if qt_metrics["total"] > 0 and qt_metrics["accuracy"] > best_accuracy:
                            best_accuracy = qt_metrics["accuracy"]
                            best_rt_for_qt = rt
                
                if best_rt_for_qt:
                    qt_recommendations.append({
                        "recommendation": f"Use {best_rt_for_qt.value} representation for {qt_value} questions",
                        "priority": "medium",
                        "evidence": f"Achieves {best_accuracy:.2%} accuracy for {qt_value} questions",
                    })
                
                recommendations["by_question_type"][qt_value] = qt_recommendations
            
            # Performance recommendations
            response_times = [(rt, rt_metrics.avg_response_time_ms) for rt, rt_metrics in metrics.items() if rt_metrics.avg_response_time_ms is not None]
            if response_times:
                slowest_rt, slowest_time = max(response_times, key=lambda x: x[1])
                
                if slowest_time > 1000:  # More than 1 second
                    recommendations["performance"].append({
                        "recommendation": f"Optimize {slowest_rt.value} representation for better performance",
                        "priority": "medium",
                        "evidence": f"Average response time is {slowest_time:.2f}ms",
                    })
        
        return recommendations