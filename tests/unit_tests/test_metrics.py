"""Unit tests for the metrics module."""

import os
import json
import tempfile
import unittest
import threading
import time
import numpy as np
from unittest.mock import patch, MagicMock

# Import the module to test
from langchain_hana.financial.metrics import (
    MetricsCollector,
    FinancialModelEvaluator,
    create_metrics_collector,
    create_model_evaluator,
)


class TestMetricsCollector(unittest.TestCase):
    """Tests for the MetricsCollector class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.metrics_file = os.path.join(self.temp_dir, "test_metrics.json")
        self.collector = MetricsCollector(
            metrics_file=self.metrics_file,
            metrics_prefix="test_metrics",
            auto_save=True,
            max_items_per_metric=100,
            chunk_size=10,
            buffer_flush_threshold=5,
        )

    def tearDown(self):
        """Tear down test fixtures."""
        # Clean up temp files
        if os.path.exists(self.metrics_file):
            os.unlink(self.metrics_file)
        os.rmdir(self.temp_dir)

    def test_update_metric(self):
        """Test updating a single metric."""
        # Update a metric
        self.collector.update_metric("test_metric", 1.0)
        
        # Check that the metric was updated
        metric = self.collector.get_metric("test_metric")
        self.assertEqual(metric, [1.0])
        
        # Update the metric again
        self.collector.update_metric("test_metric", 2.0)
        
        # Check that the metric was updated
        metric = self.collector.get_metric("test_metric")
        self.assertEqual(metric, [1.0, 2.0])

    def test_update_metrics(self):
        """Test updating multiple metrics at once."""
        # Update multiple metrics
        self.collector.update_metrics({
            "metric1": 1.0,
            "metric2": 2.0,
            "metric3": 3.0,
        })
        
        # Check that the metrics were updated
        self.assertEqual(self.collector.get_metric("metric1"), [1.0])
        self.assertEqual(self.collector.get_metric("metric2"), [2.0])
        self.assertEqual(self.collector.get_metric("metric3"), [3.0])

    def test_get_latest_metric(self):
        """Test getting the latest value for a metric."""
        # Update a metric multiple times
        self.collector.update_metric("test_metric", 1.0)
        self.collector.update_metric("test_metric", 2.0)
        self.collector.update_metric("test_metric", 3.0)
        
        # Check that the latest value is returned
        latest = self.collector.get_latest_metric("test_metric")
        self.assertEqual(latest, 3.0)

    def test_save_and_load_metrics(self):
        """Test saving and loading metrics to/from a file."""
        # Update some metrics
        self.collector.update_metrics({
            "metric1": 1.0,
            "metric2": 2.0,
            "metric3": 3.0,
        })
        
        # Save metrics
        self.collector.save_metrics()
        
        # Create a new collector that loads the metrics
        new_collector = MetricsCollector(metrics_file=self.metrics_file, auto_save=False)
        new_collector.load_metrics()
        
        # Check that the metrics were loaded
        self.assertEqual(new_collector.get_metric("metric1"), [1.0])
        self.assertEqual(new_collector.get_metric("metric2"), [2.0])
        self.assertEqual(new_collector.get_metric("metric3"), [3.0])

    def test_clear_metrics(self):
        """Test clearing all metrics."""
        # Update some metrics
        self.collector.update_metrics({
            "metric1": 1.0,
            "metric2": 2.0,
            "metric3": 3.0,
        })
        
        # Clear metrics
        self.collector.clear_metrics()
        
        # Check that the metrics were cleared
        self.assertEqual(self.collector.get_metric("metric1"), [])
        self.assertEqual(self.collector.get_metric("metric2"), [])
        self.assertEqual(self.collector.get_metric("metric3"), [])

    def test_calculate_summary(self):
        """Test calculating summary statistics for metrics."""
        # Update a metric with multiple values
        self.collector.update_metric("test_metric", [1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Calculate summary
        summary = self.collector.calculate_summary()
        
        # Check summary statistics
        self.assertIn("test_metric", summary)
        self.assertAlmostEqual(summary["test_metric"]["mean"], 3.0)
        self.assertAlmostEqual(summary["test_metric"]["std"], np.std([1.0, 2.0, 3.0, 4.0, 5.0]))
        self.assertAlmostEqual(summary["test_metric"]["min"], 1.0)
        self.assertAlmostEqual(summary["test_metric"]["max"], 5.0)
        self.assertEqual(summary["test_metric"]["count"], 5)

    def test_custom_metrics(self):
        """Test adding and retrieving custom metrics."""
        # Add a custom metric
        self.collector.update_metric("custom_metric", 1.0)
        
        # Check that the custom metric was added
        self.assertEqual(self.collector.get_metric("custom_metric"), [1.0])
        
        # Update the custom metric
        self.collector.update_metric("custom_metric", 2.0)
        
        # Check that the custom metric was updated
        self.assertEqual(self.collector.get_metric("custom_metric"), [1.0, 2.0])

    def test_buffer_flush(self):
        """Test that the buffer is flushed when the threshold is reached."""
        # Update a metric multiple times, but less than the threshold
        for i in range(4):
            self.collector.update_metric("test_metric", i)
        
        # Check that the metric was updated in the buffer but not flushed
        self.assertEqual(len(self.collector.update_buffer.get("test_metric", [])), 4)
        
        # Update one more time to trigger a flush
        self.collector.update_metric("test_metric", 4)
        
        # Check that the buffer was flushed
        self.assertEqual(len(self.collector.update_buffer.get("test_metric", [])), 0)
        
        # Check that the metric was updated
        self.assertEqual(self.collector.get_metric("test_metric"), [0, 1, 2, 3, 4])

    def test_thread_safety(self):
        """Test thread safety of the metrics collector."""
        # Create a collector with a high buffer threshold to test threading
        collector = MetricsCollector(
            metrics_file=self.metrics_file,
            auto_save=False,
            buffer_flush_threshold=1000,
        )
        
        # Function to update metrics in a thread
        def update_metrics():
            for i in range(100):
                collector.update_metric("thread_metric", i)
        
        # Create and start threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=update_metrics)
            thread.start()
            threads.append(thread)
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Flush buffer and check results
        collector._flush_buffer()
        
        # Check that all updates were applied
        metric = collector.get_metric("thread_metric")
        self.assertEqual(len(metric), 1000)

    def test_max_items_per_metric(self):
        """Test that the number of items per metric is limited."""
        # Create a collector with a small max_items_per_metric
        collector = MetricsCollector(
            metrics_file=self.metrics_file,
            max_items_per_metric=10,
            auto_save=False,
        )
        
        # Update a metric with more values than the limit
        collector.update_metric("test_metric", list(range(20)))
        
        # Check that only the most recent values are kept
        metric = collector.get_metric("test_metric")
        self.assertEqual(len(metric), 10)
        self.assertEqual(metric, list(range(10, 20)))
        
        # Check that statistics were updated
        self.assertIn("test_metric", collector.statistics)
        self.assertEqual(collector.statistics["test_metric"]["count"], 20)
        self.assertAlmostEqual(collector.statistics["test_metric"]["mean"], 9.5)

    def test_factory_function(self):
        """Test the factory function for creating a metrics collector."""
        # Create a collector using the factory function
        collector = create_metrics_collector(
            metrics_file=self.metrics_file,
            metrics_prefix="factory_test",
        )
        
        # Check that the collector was created correctly
        self.assertIsInstance(collector, MetricsCollector)
        self.assertEqual(collector.metrics_file, self.metrics_file)


class TestFinancialModelEvaluator(unittest.TestCase):
    """Tests for the FinancialModelEvaluator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.metrics_file = os.path.join(self.temp_dir, "test_metrics.json")
        self.collector = MetricsCollector(metrics_file=self.metrics_file, auto_save=False)
        self.evaluator = FinancialModelEvaluator(metrics_collector=self.collector)

    def tearDown(self):
        """Tear down test fixtures."""
        # Clean up temp files
        if os.path.exists(self.metrics_file):
            os.unlink(self.metrics_file)
        os.rmdir(self.temp_dir)

    def test_calculate_semantic_similarity(self):
        """Test calculating semantic similarity between embeddings."""
        # Create test embeddings
        embedding1 = np.array([1.0, 0.0, 0.0])
        embedding2 = np.array([0.0, 1.0, 0.0])
        embedding3 = np.array([1.0, 1.0, 0.0])
        
        # Calculate similarities
        sim12 = self.evaluator.calculate_semantic_similarity(embedding1, embedding2)
        sim13 = self.evaluator.calculate_semantic_similarity(embedding1, embedding3)
        sim23 = self.evaluator.calculate_semantic_similarity(embedding2, embedding3)
        
        # Check similarities
        self.assertAlmostEqual(sim12, 0.0)
        self.assertAlmostEqual(sim13, 1.0 / np.sqrt(2))
        self.assertAlmostEqual(sim23, 1.0 / np.sqrt(2))

    def test_calculate_retrieval_metrics(self):
        """Test calculating retrieval metrics."""
        # Define relevant and retrieved documents
        relevant_docs = ["doc1", "doc2", "doc3"]
        retrieved_docs = ["doc1", "doc4", "doc2", "doc5"]
        
        # Calculate metrics
        metrics = self.evaluator.calculate_retrieval_metrics(relevant_docs, retrieved_docs, k=3)
        
        # Check metrics
        self.assertAlmostEqual(metrics["precision"], 2/3)
        self.assertAlmostEqual(metrics["recall"], 2/3)
        self.assertAlmostEqual(metrics["f1_score"], 2/3)
        self.assertEqual(metrics["relevant_count"], 3)
        self.assertEqual(metrics["retrieved_count"], 4)
        self.assertEqual(metrics["true_positives"], 2)

    def test_evaluate_model_improvement(self):
        """Test calculating improvement metrics between models."""
        # Define base and tuned model results
        base_results = {
            "precision": 0.7,
            "recall": 0.6,
            "f1_score": 0.65,
            "execution_time": 0.2,
            "similarity": 0.8,
        }
        
        tuned_results = {
            "precision": 0.8,
            "recall": 0.7,
            "f1_score": 0.75,
            "execution_time": 0.15,
            "similarity": 0.9,
        }
        
        # Calculate improvements
        improvements = self.evaluator.evaluate_model_improvement(base_results, tuned_results)
        
        # Check improvements
        self.assertAlmostEqual(improvements["precision_improvement"], 100 * (0.8 - 0.7) / 0.7)
        self.assertAlmostEqual(improvements["recall_improvement"], 100 * (0.7 - 0.6) / 0.6)
        self.assertAlmostEqual(improvements["f1_score_improvement"], 100 * (0.75 - 0.65) / 0.65)
        self.assertAlmostEqual(improvements["execution_time_improvement"], 100 * (0.2 - 0.15) / 0.2)
        self.assertAlmostEqual(improvements["similarity_improvement"], 100 * (0.9 - 0.8) / 0.8)

    def test_evaluate_model_on_queries(self):
        """Test evaluating a model on a set of queries."""
        # Create mock model
        mock_model = MagicMock()
        mock_model.encode.side_effect = lambda text, **kwargs: np.array([1.0, 0.0, 0.0])
        
        # Define test data
        queries = ["query1", "query2"]
        relevant_docs = {"query1": ["doc1", "doc2"], "query2": ["doc3", "doc4"]}
        doc_embeddings = {
            "doc1": np.array([1.0, 0.0, 0.0]),  # Same as query embedding, perfect similarity
            "doc2": np.array([0.5, 0.5, 0.0]),  # Some similarity
            "doc3": np.array([0.0, 1.0, 0.0]),  # No similarity
            "doc4": np.array([0.0, 0.0, 1.0]),  # No similarity
            "doc5": np.array([0.8, 0.2, 0.0]),  # High similarity, but not relevant
        }
        
        # Evaluate model
        results = self.evaluator.evaluate_model_on_queries(
            mock_model, queries, relevant_docs, doc_embeddings, k=2
        )
        
        # Check results
        self.assertIn("query_results", results)
        self.assertIn("query1", results["query_results"])
        self.assertIn("query2", results["query_results"])
        self.assertIn("retrieved_docs", results["query_results"]["query1"])
        self.assertIn("execution_time", results["query_results"]["query1"])
        self.assertIn("avg_precision", results)
        self.assertIn("avg_recall", results)
        self.assertIn("avg_f1_score", results)
        self.assertIn("avg_execution_time", results)

    def test_factory_function(self):
        """Test the factory function for creating a model evaluator."""
        # Create an evaluator using the factory function
        evaluator = create_model_evaluator(
            metrics_collector=self.collector,
            evaluation_data={"test": "data"},
        )
        
        # Check that the evaluator was created correctly
        self.assertIsInstance(evaluator, FinancialModelEvaluator)
        self.assertEqual(evaluator.metrics_collector, self.collector)
        self.assertEqual(evaluator.evaluation_data, {"test": "data"})


if __name__ == "__main__":
    unittest.main()