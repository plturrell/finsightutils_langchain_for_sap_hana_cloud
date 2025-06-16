"""Unit tests for the visualization module."""

import os
import json
import tempfile
import unittest
import threading
import time
from unittest.mock import patch, MagicMock, call

# Import the module to test
from langchain_hana.financial.visualization import (
    TrainingVisualizer,
    MetricsVisualizer,
    ModelComparisonVisualizer,
    create_training_visualizer,
    create_model_comparison_visualizer,
)


class TestTrainingVisualizer(unittest.TestCase):
    """Tests for the TrainingVisualizer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.progress_file = os.path.join(self.temp_dir, "test_progress.json")
        self.metrics_file = os.path.join(self.temp_dir, "test_metrics.json")
        self.output_file = os.path.join(self.temp_dir, "test_output.txt")
        
        # Create progress and metrics files with test data
        self.progress_data = {
            "status": "running",
            "progress": 0.5,
            "step": 50,
            "total_steps": 100,
            "stage": "Learning financial relationships",
            "started_at": time.time() - 60,  # Started 1 minute ago
            "estimated_completion": time.time() + 60,  # Will finish in 1 minute
            "messages": ["Starting", "Progress update"]
        }
        
        self.metrics_data = {
            "loss": [0.5, 0.4, 0.3, 0.2, 0.1],
            "batch_times": [0.1, 0.1, 0.09, 0.1, 0.09],
            "custom": {
                "financial_accuracy": [0.6, 0.7, 0.8, 0.85, 0.9]
            }
        }
        
        with open(self.progress_file, "w") as f:
            json.dump(self.progress_data, f)
        
        with open(self.metrics_file, "w") as f:
            json.dump(self.metrics_data, f)
        
        # Create visualizer
        self.visualizer = TrainingVisualizer(
            progress_file=self.progress_file,
            metrics_file=self.metrics_file,
            refresh_interval=0.1,  # Fast refresh for testing
            output_file=self.output_file,
        )

    def tearDown(self):
        """Tear down test fixtures."""
        # Stop monitoring if running
        if hasattr(self, "visualizer") and self.visualizer.running:
            self.visualizer.stop_monitoring()
        
        # Clean up temp files
        for file in [self.progress_file, self.metrics_file, self.output_file]:
            if os.path.exists(file):
                os.unlink(file)
        
        os.rmdir(self.temp_dir)

    def test_load_progress_and_metrics(self):
        """Test loading progress and metrics from files."""
        # Call private method directly for testing
        self.visualizer._load_progress_and_metrics()
        
        # Check that progress and metrics were loaded
        self.assertEqual(self.visualizer.last_progress, self.progress_data)
        self.assertEqual(self.visualizer.last_metrics, self.metrics_data)

    @patch("sys.stdout")
    def test_visualize_progress(self, mock_stdout):
        """Test visualizing progress."""
        # Set output file to None to use standard output
        self.visualizer.output_file = None
        
        # Visualize progress
        self.visualizer._visualize_progress()
        
        # Check that output was generated
        mock_stdout.write.assert_called()

    def test_visualize_to_file(self):
        """Test visualizing progress to a file."""
        # Visualize progress
        self.visualizer._visualize_progress()
        
        # Check that output file was created
        self.assertTrue(os.path.exists(self.output_file))
        
        # Check file contents
        with open(self.output_file, "r") as f:
            content = f.read()
        
        # Check that key information is in the output
        self.assertIn("Financial Model Enlightenment", content)
        self.assertIn("Status: running", content)
        self.assertIn("Stage: Learning financial relationships", content)
        self.assertIn("Progress: 50.0%", content)
        self.assertIn("Loss:", content)

    def test_start_and_stop_monitoring(self):
        """Test starting and stopping monitoring."""
        # Start monitoring
        thread = self.visualizer.start_monitoring()
        
        # Check that monitoring is running
        self.assertTrue(self.visualizer.running)
        self.assertTrue(thread.is_alive())
        
        # Stop monitoring
        self.visualizer.stop_monitoring()
        
        # Check that monitoring has stopped
        self.assertFalse(self.visualizer.running)
        
        # Wait for thread to exit
        thread.join(timeout=1.0)
        self.assertFalse(thread.is_alive())

    def test_callback(self):
        """Test callback function."""
        # Create a mock callback
        callback = MagicMock()
        
        # Create visualizer with callback
        visualizer = TrainingVisualizer(
            progress_file=self.progress_file,
            metrics_file=self.metrics_file,
            callback=callback,
        )
        
        # Call private method to trigger callback
        visualizer._load_progress_and_metrics()
        visualizer._visualize_progress()
        
        # Check that callback was called
        callback.assert_called_once_with(self.progress_data, self.metrics_data)

    def test_queues(self):
        """Test using queues for IPC."""
        # Create mock queues
        progress_queue = MagicMock()
        metrics_queue = MagicMock()
        
        # Create visualizer with queues
        visualizer = TrainingVisualizer(
            progress_file=self.progress_file,
            metrics_file=self.metrics_file,
        )
        
        # Start monitoring with queues
        visualizer.start_monitoring(
            progress_queue=progress_queue,
            metrics_queue=metrics_queue,
            daemon=True,
        )
        
        # Give monitoring thread time to start
        time.sleep(0.2)
        
        # Stop monitoring
        visualizer.stop_monitoring()
        
        # Check that queues were used
        self.assertEqual(visualizer.progress_queue, progress_queue)
        self.assertEqual(visualizer.metrics_queue, metrics_queue)

    def test_factory_function(self):
        """Test the factory function for creating a training visualizer."""
        # Create visualizer using factory function
        visualizer = create_training_visualizer(
            progress_file=self.progress_file,
            metrics_file=self.metrics_file,
            refresh_interval=0.5,
        )
        
        # Check that visualizer was created correctly
        self.assertIsInstance(visualizer, TrainingVisualizer)
        self.assertEqual(visualizer.progress_file, self.progress_file)
        self.assertEqual(visualizer.metrics_file, self.metrics_file)
        self.assertEqual(visualizer.refresh_interval, 0.5)


class TestMetricsVisualizer(unittest.TestCase):
    """Tests for the MetricsVisualizer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.metrics_file = os.path.join(self.temp_dir, "test_metrics.json")
        self.output_file = os.path.join(self.temp_dir, "test_output.txt")
        
        # Create metrics file with test data
        self.metrics_data = {
            "loss": [0.5, 0.4, 0.3, 0.2, 0.1],
            "batch_times": [0.1, 0.1, 0.09, 0.1, 0.09],
            "custom": {
                "financial_accuracy": [0.6, 0.7, 0.8, 0.85, 0.9]
            }
        }
        
        with open(self.metrics_file, "w") as f:
            json.dump(self.metrics_data, f)
        
        # Create visualizer
        self.visualizer = MetricsVisualizer(
            metrics_file=self.metrics_file,
            output_file=self.output_file,
        )

    def tearDown(self):
        """Tear down test fixtures."""
        # Clean up temp files
        for file in [self.metrics_file, self.output_file]:
            if os.path.exists(file):
                os.unlink(file)
        
        os.rmdir(self.temp_dir)

    @patch("sys.stdout")
    def test_visualize_metrics(self, mock_stdout):
        """Test visualizing metrics."""
        # Set output file to None to use standard output
        self.visualizer.output_file = None
        
        # Visualize metrics
        self.visualizer.visualize_metrics()
        
        # Check that output was generated
        mock_stdout.write.assert_called()

    def test_visualize_to_file(self):
        """Test visualizing metrics to a file."""
        # Visualize metrics
        self.visualizer.visualize_metrics()
        
        # Check that output file was created
        self.assertTrue(os.path.exists(self.output_file))
        
        # Check file contents
        with open(self.output_file, "r") as f:
            content = f.read()
        
        # Check that key information is in the output
        self.assertIn("Financial Model Metrics", content)
        self.assertIn("Loss:", content)
        self.assertIn("Performance:", content)
        self.assertIn("Custom Metrics:", content)
        self.assertIn("financial_accuracy", content)


class TestModelComparisonVisualizer(unittest.TestCase):
    """Tests for the ModelComparisonVisualizer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.base_metrics_file = os.path.join(self.temp_dir, "test_base_metrics.json")
        self.tuned_metrics_file = os.path.join(self.temp_dir, "test_tuned_metrics.json")
        self.output_file = os.path.join(self.temp_dir, "test_output.txt")
        
        # Create sample data
        self.base_results = {
            "name": "base_model",
            "metrics": {
                "precision": 0.7,
                "recall": 0.6,
                "f1_score": 0.65,
                "execution_time": 0.2,
            },
            "query_results": {
                "query1": {
                    "execution_time": 0.2,
                    "retrieved_docs": ["doc1", "doc3"]
                },
                "query2": {
                    "execution_time": 0.3,
                    "retrieved_docs": ["doc2", "doc4"]
                }
            }
        }
        
        self.tuned_results = {
            "name": "tuned_model",
            "metrics": {
                "precision": 0.8,
                "recall": 0.7,
                "f1_score": 0.75,
                "execution_time": 0.15,
            },
            "query_results": {
                "query1": {
                    "execution_time": 0.15,
                    "retrieved_docs": ["doc1", "doc2"]
                },
                "query2": {
                    "execution_time": 0.25,
                    "retrieved_docs": ["doc2", "doc5"]
                }
            }
        }
        
        # Create visualizer
        self.visualizer = ModelComparisonVisualizer(
            base_metrics_file=self.base_metrics_file,
            tuned_metrics_file=self.tuned_metrics_file,
            output_file=self.output_file,
        )

    def tearDown(self):
        """Tear down test fixtures."""
        # Clean up temp files
        for file in [self.base_metrics_file, self.tuned_metrics_file, self.output_file]:
            if os.path.exists(file):
                os.unlink(file)
        
        os.rmdir(self.temp_dir)

    @patch("sys.stdout")
    def test_visualize_comparison(self, mock_stdout):
        """Test visualizing comparison."""
        # Set output file to None to use standard output
        self.visualizer.output_file = None
        
        # Visualize comparison
        self.visualizer.visualize_comparison(
            base_results=self.base_results,
            tuned_results=self.tuned_results,
        )
        
        # Check that output was generated
        mock_stdout.write.assert_called()

    def test_visualize_to_file(self):
        """Test visualizing comparison to a file."""
        # Visualize comparison
        self.visualizer.visualize_comparison(
            base_results=self.base_results,
            tuned_results=self.tuned_results,
        )
        
        # Check that output file was created
        self.assertTrue(os.path.exists(self.output_file))
        
        # Check file contents
        with open(self.output_file, "r") as f:
            content = f.read()
        
        # Check that key information is in the output
        self.assertIn("Financial Model Transformation", content)
        self.assertIn("Metric", content)
        self.assertIn("Base Model", content)
        self.assertIn("Tuned Model", content)
        self.assertIn("Improvement", content)
        self.assertIn("Semantic Understanding", content)

    def test_extract_metric_value(self):
        """Test extracting metric values from results."""
        # Call method directly
        precision = self.visualizer._extract_metric_value(self.base_results, "precision")
        recall = self.visualizer._extract_metric_value(self.base_results, "recall")
        execution_time = self.visualizer._extract_metric_value(self.base_results, "execution_time")
        
        # Check extracted values
        self.assertEqual(precision, 0.7)
        self.assertEqual(recall, 0.6)
        self.assertEqual(execution_time, 0.2)
        
        # Test with missing metric
        missing = self.visualizer._extract_metric_value(self.base_results, "missing_metric")
        self.assertIsNone(missing)

    def test_calculate_similarity_improvement(self):
        """Test calculating similarity improvement."""
        # Call method directly
        improvement = self.visualizer._calculate_similarity_improvement(
            self.base_results,
            self.tuned_results,
        )
        
        # Check improvement calculation
        self.assertGreaterEqual(improvement, 0.0)

    def test_factory_function(self):
        """Test the factory function for creating a model comparison visualizer."""
        # Create visualizer using factory function
        visualizer = create_model_comparison_visualizer(
            base_metrics_file=self.base_metrics_file,
            tuned_metrics_file=self.tuned_metrics_file,
            output_file=self.output_file,
        )
        
        # Check that visualizer was created correctly
        self.assertIsInstance(visualizer, ModelComparisonVisualizer)
        self.assertEqual(visualizer.base_metrics_file, self.base_metrics_file)
        self.assertEqual(visualizer.tuned_metrics_file, self.tuned_metrics_file)
        self.assertEqual(visualizer.output_file, self.output_file)


if __name__ == "__main__":
    unittest.main()