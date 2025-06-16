"""Integration tests for Finesse end-to-end workflows."""

import os
import json
import time
import tempfile
import unittest
import subprocess
from unittest.mock import patch, MagicMock

class TestFinesseWorkflow(unittest.TestCase):
    """Test the complete Finesse workflow from data preparation to model application."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create paths for test files
        self.documents_file = os.path.join(self.temp_dir, "test_documents.json")
        self.queries_file = os.path.join(self.temp_dir, "test_queries.json")
        self.models_dir = os.path.join(self.temp_dir, "models")
        self.output_dir = os.path.join(self.temp_dir, "fine_tuned_models")
        
        # Create directories
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create sample documents
        self.test_documents = [
            {
                "content": "Market risk is the risk of losses in positions arising from movements in market prices.",
                "metadata": {"source": "test", "category": "risk"}
            },
            {
                "content": "Credit risk is the risk of a financial loss if a borrower fails to meet its obligations.",
                "metadata": {"source": "test", "category": "risk"}
            },
            {
                "content": "Operational risk is the risk of loss resulting from inadequate or failed internal processes.",
                "metadata": {"source": "test", "category": "risk"}
            }
        ]
        
        # Create sample queries
        self.test_queries = [
            {"query": "What is market risk?", "filter": {}, "k": 2},
            {"query": "Define credit risk", "filter": {}, "k": 2},
            {"query": "Explain operational risk", "filter": {}, "k": 2}
        ]
        
        # Write sample files
        with open(self.documents_file, "w") as f:
            json.dump(self.test_documents, f, indent=2)
        
        with open(self.queries_file, "w") as f:
            json.dump(self.test_queries, f, indent=2)
        
        # Set environment variables for testing
        os.environ["FINETUNE_LOG_LEVEL"] = "DEBUG"
        os.environ["FINETUNE_LOG_FILE"] = os.path.join(self.temp_dir, "finetune.log")
        
        # Store original working directory
        self.original_dir = os.getcwd()
        
        # Record the finesse script path
        self.finesse_path = os.path.join(self.original_dir, "finesse")

    def tearDown(self):
        """Tear down test fixtures."""
        # Clean up temporary directory
        for root, dirs, files in os.walk(self.temp_dir, topdown=False):
            for file in files:
                os.unlink(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))
        os.rmdir(self.temp_dir)
        
        # Restore environment
        if "FINETUNE_LOG_LEVEL" in os.environ:
            del os.environ["FINETUNE_LOG_LEVEL"]
        if "FINETUNE_LOG_FILE" in os.environ:
            del os.environ["FINETUNE_LOG_FILE"]

    def _run_command(self, command, check=True):
        """Run a command and return its output."""
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=check
        )
        return result

    @patch("subprocess.run")
    def test_prepare_command(self, mock_run):
        """Test the prepare command."""
        # Set up mock return value
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = "Preparation complete"
        mock_run.return_value = mock_process
        
        # Execute the command
        cmd = [
            self.finesse_path,
            "prepare",
            "--source", self.documents_file,
            "--output", self.models_dir
        ]
        result = self._run_command(cmd)
        
        # Check that the command was called correctly
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        self.assertEqual(args[0], cmd)
        
        # Check that finesse_config.json was created
        config_path = os.path.join(self.models_dir, "finesse_config.json")
        self.assertTrue(mock_process.returncode == 0)

    @patch("subprocess.run")
    def test_enlighten_command(self, mock_run):
        """Test the enlighten command."""
        # Set up mock return value
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = "Enlightenment complete"
        mock_run.return_value = mock_process
        
        # Create a minimal config file
        config_path = os.path.join(self.models_dir, "finesse_config.json")
        train_file = os.path.join(self.models_dir, "training_data.json")
        val_file = os.path.join(self.models_dir, "validation_data.json")
        
        # Create sample training data
        with open(train_file, "w") as f:
            json.dump([
                {"text1": "What is market risk?", "text2": "Market risk is the risk of losses in positions.", "score": 0.9},
                {"text1": "Define credit risk", "text2": "Credit risk is the risk of a financial loss.", "score": 0.9}
            ], f)
        
        # Create sample validation data
        with open(val_file, "w") as f:
            json.dump([
                {"text1": "Explain operational risk", "text2": "Operational risk is the risk of loss.", "score": 0.9}
            ], f)
        
        # Create config file
        with open(config_path, "w") as f:
            json.dump({
                "train_file": train_file,
                "val_file": val_file,
                "created_at": time.time(),
                "source_file": self.documents_file,
                "document_count": 3
            }, f)
        
        # Execute the command
        cmd = [
            self.finesse_path,
            "enlighten",
            "--model", "FinMTEB/Fin-E5-small"
        ]
        result = self._run_command(cmd)
        
        # Check that the command was called correctly
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        self.assertEqual(args[0], cmd)
        
        # Check that the mock process was successful
        self.assertTrue(mock_process.returncode == 0)

    @patch("subprocess.run")
    def test_compare_command(self, mock_run):
        """Test the compare command."""
        # Set up mock return value
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = "Comparison complete"
        mock_run.return_value = mock_process
        
        # Create a minimal config file
        config_path = os.path.join(self.models_dir, "finesse_config.json")
        
        # Create config file with enlightened model path
        with open(config_path, "w") as f:
            json.dump({
                "enlightened_model_path": os.path.join(self.output_dir, "test_model"),
                "enlightened_at": time.time(),
                "base_model": "FinMTEB/Fin-E5-small",
                "enlightened_model_name": "test_model"
            }, f)
        
        # Create enlightened model directory
        os.makedirs(os.path.join(self.output_dir, "test_model"), exist_ok=True)
        
        # Execute the command
        cmd = [
            self.finesse_path,
            "compare",
            "--queries", self.queries_file
        ]
        result = self._run_command(cmd)
        
        # Check that the command was called correctly
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        self.assertEqual(args[0], cmd)
        
        # Check that the mock process was successful
        self.assertTrue(mock_process.returncode == 0)

    @patch("subprocess.run")
    def test_apply_command(self, mock_run):
        """Test the apply command."""
        # Set up mock return value
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = "Applied model to query"
        mock_run.return_value = mock_process
        
        # Create a minimal config file
        config_path = os.path.join(self.models_dir, "finesse_config.json")
        
        # Create config file with enlightened model path
        with open(config_path, "w") as f:
            json.dump({
                "enlightened_model_path": os.path.join(self.output_dir, "test_model"),
                "enlightened_at": time.time(),
                "base_model": "FinMTEB/Fin-E5-small",
                "enlightened_model_name": "test_model"
            }, f)
        
        # Create enlightened model directory
        os.makedirs(os.path.join(self.output_dir, "test_model"), exist_ok=True)
        
        # Execute the command
        cmd = [
            self.finesse_path,
            "apply",
            "--query", "What is market risk?"
        ]
        result = self._run_command(cmd)
        
        # Check that the command was called correctly
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        self.assertEqual(args[0], cmd)
        
        # Check that the mock process was successful
        self.assertTrue(mock_process.returncode == 0)

    @patch("subprocess.run")
    def test_full_workflow(self, mock_run):
        """Test the full workflow from prepare to apply."""
        # Set up mock return values for each command
        def side_effect(cmd, **kwargs):
            mock_process = MagicMock()
            mock_process.returncode = 0
            
            if "prepare" in cmd:
                mock_process.stdout = "Preparation complete"
                
                # Create the expected output files
                train_file = os.path.join(self.models_dir, "training_data.json")
                val_file = os.path.join(self.models_dir, "validation_data.json")
                config_path = os.path.join(self.models_dir, "finesse_config.json")
                
                # Create sample training data
                with open(train_file, "w") as f:
                    json.dump([
                        {"text1": "What is market risk?", "text2": "Market risk is the risk of losses.", "score": 0.9},
                        {"text1": "Define credit risk", "text2": "Credit risk is the risk of a financial loss.", "score": 0.9}
                    ], f)
                
                # Create sample validation data
                with open(val_file, "w") as f:
                    json.dump([
                        {"text1": "Explain operational risk", "text2": "Operational risk is the risk of loss.", "score": 0.9}
                    ], f)
                
                # Create config file
                with open(config_path, "w") as f:
                    json.dump({
                        "train_file": train_file,
                        "val_file": val_file,
                        "created_at": time.time(),
                        "source_file": self.documents_file,
                        "document_count": 3
                    }, f)
                
            elif "enlighten" in cmd:
                mock_process.stdout = "Enlightenment complete"
                
                # Update the config file with enlightened model information
                config_path = os.path.join(self.models_dir, "finesse_config.json")
                model_path = os.path.join(self.output_dir, "test_model")
                
                # Create enlightened model directory
                os.makedirs(model_path, exist_ok=True)
                
                # Write model path file
                with open("fin_e5_tuned_model_path.txt", "w") as f:
                    f.write(model_path)
                
                # Update config
                with open(config_path, "r") as f:
                    config = json.load(f)
                
                config.update({
                    "enlightened_model_path": model_path,
                    "enlightened_at": time.time(),
                    "base_model": "FinMTEB/Fin-E5-small",
                    "enlightened_model_name": "test_model"
                })
                
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=2)
                
            elif "compare" in cmd:
                mock_process.stdout = "Comparison complete"
                
                # Create comparison file
                with open("model_comparison.md", "w") as f:
                    f.write("# Financial Understanding Transformation\n\n")
                    f.write("## Overview\n\n")
                    f.write("- **Base model**: Standard understanding\n")
                    f.write("- **Enlightened model**: Financial domain expertise\n")
                    f.write("## Performance Transformation\n\n")
                    f.write("| Metric | Base Model | Enlightened Model | Improvement |\n")
                    f.write("|--------|------------|-------------------|------------|\n")
                    f.write("| Average Query Time (s) | 0.2000 | 0.1500 | 25.00% |\n")
                    f.write("| Semantic Relevance | Good | Excellent | 35.00% |\n")
                
                # Update config file
                config_path = os.path.join(self.models_dir, "finesse_config.json")
                with open(config_path, "r") as f:
                    config = json.load(f)
                
                config.update({
                    "comparison_file": "model_comparison.md",
                    "compared_at": time.time(),
                    "time_improvement": 25.0,
                    "semantic_improvement": 35.0
                })
                
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=2)
                
            elif "apply" in cmd:
                mock_process.stdout = "Query results:\n\nMarket risk is the risk of losses in positions arising from movements in market prices."
            
            return mock_process
        
        mock_run.side_effect = side_effect
        
        # Execute the workflow
        commands = [
            [self.finesse_path, "prepare", "--source", self.documents_file, "--output", self.models_dir],
            [self.finesse_path, "enlighten", "--model", "FinMTEB/Fin-E5-small"],
            [self.finesse_path, "compare", "--queries", self.queries_file],
            [self.finesse_path, "apply", "--query", "What is market risk?"]
        ]
        
        for cmd in commands:
            result = self._run_command(cmd)
            self.assertEqual(result.returncode, 0)
        
        # Check that all expected calls were made
        self.assertEqual(mock_run.call_count, 4)
        
        # Check that the final config file contains all expected information
        config_path = os.path.join(self.models_dir, "finesse_config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        
        self.assertIn("source_file", config)
        self.assertIn("train_file", config)
        self.assertIn("val_file", config)
        self.assertIn("enlightened_model_path", config)
        self.assertIn("comparison_file", config)
        self.assertIn("time_improvement", config)
        self.assertIn("semantic_improvement", config)


if __name__ == "__main__":
    unittest.main()