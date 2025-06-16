"""
Unit tests for the DifficultyLevel class.

This module contains unit tests for the DifficultyLevel class, which provides
standardized difficulty assessment for benchmark questions.
"""

import unittest
import re

import pytest

from langchain_hana.reasoning.benchmark_system import DifficultyLevel, QuestionType


class TestDifficultyLevel(unittest.TestCase):
    """Tests for the DifficultyLevel class."""
    
    def test_enum_values(self):
        """Test the enum values for difficulty levels."""
        # Verify the enum values are correct and in ascending order
        self.assertEqual(DifficultyLevel.BEGINNER, 1)
        self.assertEqual(DifficultyLevel.BASIC, 2)
        self.assertEqual(DifficultyLevel.INTERMEDIATE, 3)
        self.assertEqual(DifficultyLevel.ADVANCED, 4)
        self.assertEqual(DifficultyLevel.EXPERT, 5)
    
    def test_assess_difficulty_basic(self):
        """Test the basic difficulty assessment functionality."""
        # Test a simple schema question (should be level 1 - BEGINNER)
        schema_q = "What is the primary key of the CUSTOMERS table?"
        schema_difficulty = DifficultyLevel.assess_difficulty(schema_q, QuestionType.SCHEMA)
        self.assertEqual(schema_difficulty, DifficultyLevel.BEGINNER)
        
        # Test a simple instance question (should be level 2 - BASIC)
        instance_q = "What is the email address of the customer with ID 1?"
        instance_difficulty = DifficultyLevel.assess_difficulty(instance_q, QuestionType.INSTANCE)
        self.assertEqual(instance_difficulty, DifficultyLevel.BASIC)
        
        # Test a relationship question (should be level 3 - INTERMEDIATE)
        relationship_q = "Which orders were placed by customer John Doe?"
        relationship_difficulty = DifficultyLevel.assess_difficulty(relationship_q, QuestionType.RELATIONSHIP)
        self.assertEqual(relationship_difficulty, DifficultyLevel.INTERMEDIATE)
        
        # Test an aggregation question (should be level 3 - INTERMEDIATE)
        aggregation_q = "What is the total number of orders placed in January 2023?"
        aggregation_difficulty = DifficultyLevel.assess_difficulty(aggregation_q, QuestionType.AGGREGATION)
        self.assertEqual(aggregation_difficulty, DifficultyLevel.INTERMEDIATE)
        
        # Test a temporal question (should be level 4 - ADVANCED)
        temporal_q = "What is the month-over-month growth rate of sales by product category?"
        temporal_difficulty = DifficultyLevel.assess_difficulty(temporal_q, QuestionType.TEMPORAL)
        self.assertEqual(temporal_difficulty, DifficultyLevel.ADVANCED)
        
        # Test an inference question (should be level 5 - EXPERT)
        inference_q = "Based on customer purchase patterns, which product bundles would likely increase sales?"
        inference_difficulty = DifficultyLevel.assess_difficulty(inference_q, QuestionType.INFERENCE)
        self.assertEqual(inference_difficulty, DifficultyLevel.EXPERT)
    
    def test_assess_difficulty_edge_cases(self):
        """Test edge cases for difficulty assessment."""
        # Very short question should get minimum difficulty (BEGINNER)
        short_q = "PK?"
        short_difficulty = DifficultyLevel.assess_difficulty(short_q, QuestionType.SCHEMA)
        self.assertEqual(short_difficulty, DifficultyLevel.BEGINNER)
        
        # Empty question should get minimum difficulty (BEGINNER)
        empty_q = ""
        empty_difficulty = DifficultyLevel.assess_difficulty(empty_q, QuestionType.SCHEMA)
        self.assertEqual(empty_difficulty, DifficultyLevel.BEGINNER)
        
        # Unicode and special characters should be handled correctly
        unicode_q = "What is the primary key of the CUSTOMERS table? 🔑"
        unicode_difficulty = DifficultyLevel.assess_difficulty(unicode_q, QuestionType.SCHEMA)
        self.assertEqual(unicode_difficulty, DifficultyLevel.BEGINNER)
    
    def test_complexity_indicators(self):
        """Test detection of complexity indicators that influence difficulty."""
        # Question with multiple entities (tables/columns)
        multiple_entities_q = "What are the column names of the CUSTOMERS, ORDERS, and PRODUCTS tables?"
        self.assertEqual(
            DifficultyLevel.assess_difficulty(multiple_entities_q, QuestionType.SCHEMA),
            DifficultyLevel.INTERMEDIATE  # Base(1) + Entity adjustment(2) = 3
        )
        
        # Question with multiple conditions
        conditions_q = "Find customers who have placed more than 5 orders and have not made any returns"
        self.assertEqual(
            DifficultyLevel.assess_difficulty(conditions_q, QuestionType.INSTANCE),
            DifficultyLevel.ADVANCED  # Base(2) + Condition adjustment(2) = 4
        )
        
        # Question with aggregate functions
        functions_q = "What is the average order value, total revenue, and count of orders by month?"
        self.assertEqual(
            DifficultyLevel.assess_difficulty(functions_q, QuestionType.AGGREGATION),
            DifficultyLevel.ADVANCED  # Base(3) + Function adjustment(1) = 4
        )
        
        # Question with joins
        joins_q = "Show customers who have purchased products from all categories, joined with their order history"
        self.assertEqual(
            DifficultyLevel.assess_difficulty(joins_q, QuestionType.RELATIONSHIP),
            DifficultyLevel.ADVANCED  # Base(3) + Join adjustment(1) = 4
        )
        
        # Question with subqueries
        subqueries_q = "Find products that are ordered more frequently than the average product within each category"
        self.assertEqual(
            DifficultyLevel.assess_difficulty(subqueries_q, QuestionType.AGGREGATION),
            DifficultyLevel.EXPERT  # Base(3) + Subquery adjustment(2) = 5
        )
        
        # Question with domain knowledge
        domain_q = "Calculate the specific inventory turnover ratio using COGS and average inventory for each product"
        self.assertEqual(
            DifficultyLevel.assess_difficulty(domain_q, QuestionType.AGGREGATION),
            DifficultyLevel.EXPERT  # Base(3) + Domain adjustment(1) + Function(1) = 5
        )
    
    def test_difficulty_bounds(self):
        """Test that difficulty is always bounded between 1 and 5."""
        # Very complex question should not exceed 5
        complex_q = "Calculate the specific complex business logic using nested subqueries for each related table, " + \
                    "joining multiple entities with complex conditions and domain-specific aggregation functions " + \
                    "while filtering based on temporal patterns and statistical inference"
        complex_difficulty = DifficultyLevel.assess_difficulty(complex_q, QuestionType.INFERENCE)
        self.assertEqual(complex_difficulty, DifficultyLevel.EXPERT)  # Should be capped at 5
        
        # Even with a simpler question type, complex indicators should not push above 5
        complex_schema_q = complex_q
        complex_schema_difficulty = DifficultyLevel.assess_difficulty(complex_schema_q, QuestionType.SCHEMA)
        self.assertLessEqual(complex_schema_difficulty, DifficultyLevel.EXPERT)  # Should not exceed 5
        
        # Very simple question with no complexity indicators should not go below 1
        simple_q = "A"
        simple_difficulty = DifficultyLevel.assess_difficulty(simple_q, QuestionType.SCHEMA)
        self.assertEqual(simple_difficulty, DifficultyLevel.BEGINNER)  # Should be minimum 1
    
    def test_get_difficulty_analysis(self):
        """Test the detailed difficulty analysis method."""
        # Test a question with multiple complexity indicators
        test_q = "What is the average order value for customers who have placed more than 5 orders, " + \
                "grouped by customer segment and joined with their demographic information?"
        analysis = DifficultyLevel.get_difficulty_analysis(test_q, QuestionType.AGGREGATION)
        
        # Verify the analysis structure
        self.assertIn("question_text", analysis)
        self.assertIn("question_type", analysis)
        self.assertIn("base_difficulty", analysis)
        self.assertIn("complexity_indicators", analysis)
        self.assertIn("adjustments", analysis)
        self.assertIn("final_difficulty", analysis)
        self.assertIn("assessment_version", analysis)
        self.assertIn("assessment_timestamp", analysis)
        
        # Verify the base difficulty
        self.assertEqual(analysis["base_difficulty"]["score"], 3)  # AGGREGATION base is 3
        
        # Verify complexity indicators were detected
        indicators = analysis["complexity_indicators"]
        self.assertGreater(indicators["functions"], 0)  # "average" should be detected
        self.assertGreater(indicators["conditions"], 0)  # "more than 5" should be detected
        self.assertGreater(indicators["joins"], 0)  # "joined with" should be detected
        
        # Verify adjustments were applied
        self.assertTrue(len(analysis["adjustments"]) > 0)
        
        # Verify final difficulty
        self.assertIn("score", analysis["final_difficulty"])
        self.assertIn("name", analysis["final_difficulty"])
        self.assertIn("explanation", analysis["final_difficulty"])
        
        # Final difficulty should be at least INTERMEDIATE (3)
        self.assertGreaterEqual(analysis["final_difficulty"]["score"], 3)
    
    def test_regex_patterns(self):
        """Test the regex patterns used for complexity detection."""
        # Create test cases for each pattern type
        test_cases = [
            # Format: (pattern name, text to test, expected match count)
            ("__entity_pattern", "table column field entity tables", 5),
            ("__condition_pattern", "where having filter if when and or not except", 8),
            ("__function_pattern", "count sum average avg max min total mean number of", 9),
            ("__join_pattern", "join related relationship between connect link association", 7),
            ("__subquery_pattern", "subquery sub-query nested for each within inside", 6),
            ("__domain_pattern", "specific special domain expert complex business rule logic", 8)
        ]
        
        # Access the private regex patterns for testing
        patterns = {
            "__entity_pattern": DifficultyLevel._DifficultyLevel__entity_pattern,
            "__condition_pattern": DifficultyLevel._DifficultyLevel__condition_pattern,
            "__function_pattern": DifficultyLevel._DifficultyLevel__function_pattern,
            "__join_pattern": DifficultyLevel._DifficultyLevel__join_pattern,
            "__subquery_pattern": DifficultyLevel._DifficultyLevel__subquery_pattern,
            "__domain_pattern": DifficultyLevel._DifficultyLevel__domain_pattern,
        }
        
        # Test each pattern
        for pattern_name, test_text, expected_count in test_cases:
            pattern = patterns.get(pattern_name)
            self.assertIsNotNone(pattern, f"Pattern {pattern_name} not found")
            
            # Count matches
            matches = pattern.findall(test_text)
            self.assertEqual(
                len(matches), 
                expected_count, 
                f"Pattern {pattern_name} matched {len(matches)} items in '{test_text}', expected {expected_count}"
            )
    
    def test_consistency(self):
        """Test that difficulty assessment is consistent for similar questions."""
        # Create pairs of similar questions
        question_pairs = [
            # Format: (question1, question2, question_type)
            (
                "What is the primary key of the CUSTOMERS table?",
                "What is the primary key of the ORDERS table?",
                QuestionType.SCHEMA
            ),
            (
                "How many customers are in the database?",
                "How many orders are in the database?",
                QuestionType.AGGREGATION
            ),
            (
                "Which customer placed the most orders?",
                "Which product was ordered the most?",
                QuestionType.AGGREGATION
            )
        ]
        
        # Test each pair
        for q1, q2, q_type in question_pairs:
            d1 = DifficultyLevel.assess_difficulty(q1, q_type)
            d2 = DifficultyLevel.assess_difficulty(q2, q_type)
            self.assertEqual(
                d1, 
                d2, 
                f"Questions '{q1}' and '{q2}' received different difficulty ratings: {d1} vs {d2}"
            )


if __name__ == "__main__":
    unittest.main()