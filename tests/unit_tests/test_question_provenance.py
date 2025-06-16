"""
Unit tests for the QuestionProvenance class.

This module contains unit tests for the QuestionProvenance class, which is responsible
for tracking the provenance information of benchmark questions.
"""

import unittest
import time
import json
from dataclasses import asdict
from datetime import datetime

import pytest

from langchain_hana.reasoning.benchmark_system import QuestionProvenance


class TestQuestionProvenance(unittest.TestCase):
    """Tests for the QuestionProvenance class."""
    
    def test_initialization(self):
        """Test initialization with required parameters."""
        # Create provenance with minimal required fields
        provenance = QuestionProvenance(
            source_type="model",
            source_id="gpt-4",
            generation_timestamp=time.time(),
            generation_method="template",
            generation_parameters={"template_id": "schema-questions"},
            input_data_sources=["schema:SALES"]
        )
        
        # Verify field values
        self.assertEqual(provenance.source_type, "model")
        self.assertEqual(provenance.source_id, "gpt-4")
        self.assertTrue(isinstance(provenance.generation_timestamp, float))
        self.assertEqual(provenance.generation_method, "template")
        self.assertEqual(provenance.generation_parameters, {"template_id": "schema-questions"})
        self.assertEqual(provenance.input_data_sources, ["schema:SALES"])
        self.assertEqual(provenance.revision_history, [])  # Should be empty by default
    
    def test_add_revision(self):
        """Test adding revisions to the history."""
        # Create provenance
        provenance = QuestionProvenance(
            source_type="model",
            source_id="gpt-4",
            generation_timestamp=time.time(),
            generation_method="template",
            generation_parameters={"template_id": "schema-questions"},
            input_data_sources=["schema:SALES"]
        )
        
        # Add a revision
        current_time = time.time()
        provenance.add_revision(
            editor="human",
            timestamp=current_time,
            reason="improve_clarity",
            changes={
                "question_text": {
                    "old": "What is the PK of CUSTOMERS?",
                    "new": "What is the primary key of the CUSTOMERS table?"
                }
            }
        )
        
        # Verify revision was added
        self.assertEqual(len(provenance.revision_history), 1)
        revision = provenance.revision_history[0]
        self.assertEqual(revision["editor"], "human")
        self.assertEqual(revision["timestamp"], current_time)
        self.assertEqual(revision["reason"], "improve_clarity")
        self.assertTrue("changes" in revision)
        
        # Add another revision
        second_time = time.time()
        provenance.add_revision(
            editor="validator",
            timestamp=second_time,
            reason="fix_answer",
            changes={
                "reference_answer": {
                    "old": "CUSTOMER_ID",
                    "new": "The primary key is CUSTOMER_ID."
                }
            }
        )
        
        # Verify both revisions are present in correct order
        self.assertEqual(len(provenance.revision_history), 2)
        self.assertEqual(provenance.revision_history[0]["editor"], "human")
        self.assertEqual(provenance.revision_history[1]["editor"], "validator")
    
    def test_serialization(self):
        """Test serialization to dictionary and JSON."""
        # Create provenance with revisions
        timestamp = time.time()
        provenance = QuestionProvenance(
            source_type="model",
            source_id="gpt-4",
            generation_timestamp=timestamp,
            generation_method="template",
            generation_parameters={"template_id": "schema-questions"},
            input_data_sources=["schema:SALES"]
        )
        
        revision_time = time.time()
        provenance.add_revision(
            editor="human",
            timestamp=revision_time,
            reason="improve_clarity",
            changes={
                "question_text": {
                    "old": "What is the PK of CUSTOMERS?",
                    "new": "What is the primary key of the CUSTOMERS table?"
                }
            }
        )
        
        # Convert to dictionary
        provenance_dict = asdict(provenance)
        
        # Verify dictionary fields
        self.assertEqual(provenance_dict["source_type"], "model")
        self.assertEqual(provenance_dict["source_id"], "gpt-4")
        self.assertEqual(provenance_dict["generation_timestamp"], timestamp)
        self.assertEqual(provenance_dict["generation_method"], "template")
        self.assertEqual(provenance_dict["generation_parameters"], {"template_id": "schema-questions"})
        self.assertEqual(provenance_dict["input_data_sources"], ["schema:SALES"])
        self.assertEqual(len(provenance_dict["revision_history"]), 1)
        
        # Test JSON serialization
        json_str = json.dumps(provenance_dict)
        deserialized = json.loads(json_str)
        
        # Verify deserialized fields
        self.assertEqual(deserialized["source_type"], "model")
        self.assertEqual(deserialized["source_id"], "gpt-4")
        self.assertEqual(deserialized["generation_timestamp"], timestamp)
        self.assertEqual(len(deserialized["revision_history"]), 1)
        self.assertEqual(deserialized["revision_history"][0]["editor"], "human")
    
    def test_complex_provenance(self):
        """Test a more complex provenance scenario with multiple data sources and revisions."""
        # Create provenance with multiple data sources
        provenance = QuestionProvenance(
            source_type="model",
            source_id="gpt-4",
            generation_timestamp=time.time(),
            generation_method="hybrid",
            generation_parameters={
                "temperature": 0.2,
                "max_tokens": 100,
                "question_types": ["schema", "instance"],
                "batch_size": 5
            },
            input_data_sources=[
                "schema:SALES",
                "data:SALES.CUSTOMERS",
                "data:SALES.ORDERS"
            ]
        )
        
        # Add multiple revisions
        for i in range(3):
            provenance.add_revision(
                editor=f"editor-{i}",
                timestamp=time.time() + i,
                reason=f"revision-{i}",
                changes={f"field-{i}": {"old": f"old-{i}", "new": f"new-{i}"}}
            )
        
        # Verify the complexity is preserved
        self.assertEqual(len(provenance.input_data_sources), 3)
        self.assertEqual(len(provenance.generation_parameters), 4)
        self.assertEqual(len(provenance.revision_history), 3)
        
        # Convert to dictionary and back to ensure full roundtrip works
        provenance_dict = asdict(provenance)
        json_str = json.dumps(provenance_dict)
        deserialized = json.loads(json_str)
        
        # Verify complex structures are preserved
        self.assertEqual(len(deserialized["input_data_sources"]), 3)
        self.assertEqual(len(deserialized["generation_parameters"]), 4)
        self.assertEqual(len(deserialized["revision_history"]), 3)
        self.assertEqual(deserialized["revision_history"][0]["reason"], "revision-0")
        self.assertEqual(deserialized["revision_history"][2]["reason"], "revision-2")
    
    def test_iso_format_dates(self):
        """Test handling of ISO formatted dates in the provenance."""
        # Create provenance with ISO date strings
        iso_timestamp = datetime.utcnow().isoformat()
        
        # Use string timestamps for generation and revision
        provenance = QuestionProvenance(
            source_type="model",
            source_id="gpt-4",
            generation_timestamp=iso_timestamp,  # Using ISO string instead of float
            generation_method="template",
            generation_parameters={"timestamp": iso_timestamp},  # Nested ISO string
            input_data_sources=["schema:SALES"]
        )
        
        # Add revision with ISO string
        provenance.add_revision(
            editor="system",
            timestamp=iso_timestamp,  # Using ISO string instead of float
            reason="standardize",
            changes={"timestamp": {"old": "unix_time", "new": "iso_format"}}
        )
        
        # Verify serialization works properly with ISO date strings
        provenance_dict = asdict(provenance)
        json_str = json.dumps(provenance_dict)
        deserialized = json.loads(json_str)
        
        # The ISO strings should be preserved as strings
        self.assertEqual(deserialized["generation_timestamp"], iso_timestamp)
        self.assertEqual(deserialized["generation_parameters"]["timestamp"], iso_timestamp)
        self.assertEqual(deserialized["revision_history"][0]["timestamp"], iso_timestamp)


if __name__ == "__main__":
    unittest.main()