"""
Tests for profiling and telemetry functionality.

This module contains unit tests for the profiling and telemetry tools provided
by the langchain_hana.monitoring.profiler module.
"""

import unittest
from unittest.mock import MagicMock, patch
import json
import time
from dataclasses import asdict

from langchain_hana.monitoring.profiler import (
    GPUEvent,
    BatchProfile,
    BatchSizeProfileResult,
    GPUProfiler
)


class TestGPUEvent(unittest.TestCase):
    """Tests for the GPUEvent class."""
    
    def test_gpu_event_creation(self):
        """Test creation of a GPUEvent."""
        event = GPUEvent(
            name="test_event",
            start_time=100.0,
            end_time=101.0,
            duration_ms=1000.0,
            cuda_time_ms=800.0,
            memory_allocated_mb=200.0,
            memory_reserved_mb=300.0,
            max_memory_allocated_mb=250.0,
            max_memory_reserved_mb=350.0,
            cuda_sync_time_ms=50.0,
            kernel_launch_count=10,
            device_id=0
        )
        
        # Check that all fields are set correctly
        self.assertEqual(event.name, "test_event")
        self.assertEqual(event.start_time, 100.0)
        self.assertEqual(event.end_time, 101.0)
        self.assertEqual(event.duration_ms, 1000.0)
        self.assertEqual(event.cuda_time_ms, 800.0)
        self.assertEqual(event.memory_allocated_mb, 200.0)
        self.assertEqual(event.memory_reserved_mb, 300.0)
        self.assertEqual(event.max_memory_allocated_mb, 250.0)
        self.assertEqual(event.max_memory_reserved_mb, 350.0)
        self.assertEqual(event.cuda_sync_time_ms, 50.0)
        self.assertEqual(event.kernel_launch_count, 10)
        self.assertEqual(event.device_id, 0)
    
    def test_to_dict(self):
        """Test conversion of GPUEvent to dictionary."""
        event = GPUEvent(
            name="test_event",
            start_time=100.0,
            end_time=101.0,
            duration_ms=1000.0,
            cuda_time_ms=800.0,
            memory_allocated_mb=200.0,
            memory_reserved_mb=300.0,
            max_memory_allocated_mb=250.0,
            max_memory_reserved_mb=350.0,
            cuda_sync_time_ms=50.0,
            kernel_launch_count=10,
            device_id=0
        )
        
        # Convert to dictionary
        event_dict = event.to_dict()
        
        # Check that the dictionary contains all fields
        self.assertEqual(event_dict["name"], "test_event")
        self.assertEqual(event_dict["start_time"], 100.0)
        self.assertEqual(event_dict["end_time"], 101.0)
        self.assertEqual(event_dict["duration_ms"], 1000.0)
        self.assertEqual(event_dict["cuda_time_ms"], 800.0)
        self.assertEqual(event_dict["memory_allocated_mb"], 200.0)
        self.assertEqual(event_dict["memory_reserved_mb"], 300.0)
        self.assertEqual(event_dict["max_memory_allocated_mb"], 250.0)
        self.assertEqual(event_dict["max_memory_reserved_mb"], 350.0)
        self.assertEqual(event_dict["cuda_sync_time_ms"], 50.0)
        self.assertEqual(event_dict["kernel_launch_count"], 10)
        self.assertEqual(event_dict["device_id"], 0)


class TestBatchProfile(unittest.TestCase):
    """Tests for the BatchProfile class."""
    
    def test_batch_profile_creation(self):
        """Test creation of a BatchProfile."""
        # Create a GPU event for the batch profile
        event = GPUEvent(
            name="test_event",
            start_time=100.0,
            end_time=101.0,
            duration_ms=1000.0,
            cuda_time_ms=800.0
        )
        
        # Create the batch profile
        profile = BatchProfile(
            batch_size=10,
            start_time=100.0,
            end_time=101.0,
            duration_ms=1000.0,
            items_per_second=10.0,
            events=[event],
            memory_peak_mb=200.0,
            cuda_time_ms=800.0,
            host_time_ms=200.0,
            kernel_launches=10,
            host_to_device_time_ms=50.0,
            device_to_host_time_ms=50.0,
            compute_time_ms=700.0,
            tokenization_time_ms=100.0,
            forward_pass_time_ms=600.0
        )
        
        # Check that all fields are set correctly
        self.assertEqual(profile.batch_size, 10)
        self.assertEqual(profile.start_time, 100.0)
        self.assertEqual(profile.end_time, 101.0)
        self.assertEqual(profile.duration_ms, 1000.0)
        self.assertEqual(profile.items_per_second, 10.0)
        self.assertEqual(len(profile.events), 1)
        self.assertEqual(profile.events[0], event)
        self.assertEqual(profile.memory_peak_mb, 200.0)
        self.assertEqual(profile.cuda_time_ms, 800.0)
        self.assertEqual(profile.host_time_ms, 200.0)
        self.assertEqual(profile.kernel_launches, 10)
        self.assertEqual(profile.host_to_device_time_ms, 50.0)
        self.assertEqual(profile.device_to_host_time_ms, 50.0)
        self.assertEqual(profile.compute_time_ms, 700.0)
        self.assertEqual(profile.tokenization_time_ms, 100.0)
        self.assertEqual(profile.forward_pass_time_ms, 600.0)
    
    def test_efficiency_ratio(self):
        """Test calculation of efficiency ratio."""
        profile = BatchProfile(
            batch_size=10,
            start_time=100.0,
            end_time=101.0,
            duration_ms=1000.0,
            items_per_second=10.0,
            cuda_time_ms=800.0
        )
        
        # Calculate efficiency ratio: cuda_time_ms / duration_ms
        self.assertEqual(profile.efficiency_ratio(), 800.0 / 1000.0)
        
        # Test with None cuda_time_ms
        profile.cuda_time_ms = None
        self.assertEqual(profile.efficiency_ratio(), 0.0)
    
    def test_time_per_item(self):
        """Test calculation of time per item."""
        profile = BatchProfile(
            batch_size=10,
            start_time=100.0,
            end_time=101.0,
            duration_ms=1000.0,
            items_per_second=10.0
        )
        
        # Calculate time per item: duration_ms / batch_size
        self.assertEqual(profile.time_per_item_ms(), 1000.0 / 10.0)
        
        # Test with zero batch size
        profile.batch_size = 0
        self.assertEqual(profile.time_per_item_ms(), 0.0)
    
    def test_cuda_time_per_item(self):
        """Test calculation of CUDA time per item."""
        profile = BatchProfile(
            batch_size=10,
            start_time=100.0,
            end_time=101.0,
            duration_ms=1000.0,
            items_per_second=10.0,
            cuda_time_ms=800.0
        )
        
        # Calculate CUDA time per item: cuda_time_ms / batch_size
        self.assertEqual(profile.cuda_time_per_item_ms(), 800.0 / 10.0)
        
        # Test with None cuda_time_ms
        profile.cuda_time_ms = None
        self.assertIsNone(profile.cuda_time_per_item_ms())
        
        # Test with zero batch size
        profile.cuda_time_ms = 800.0
        profile.batch_size = 0
        self.assertIsNone(profile.cuda_time_per_item_ms())
    
    def test_memory_per_item(self):
        """Test calculation of memory per item."""
        profile = BatchProfile(
            batch_size=10,
            start_time=100.0,
            end_time=101.0,
            duration_ms=1000.0,
            items_per_second=10.0,
            memory_peak_mb=200.0
        )
        
        # Calculate memory per item: memory_peak_mb / batch_size
        self.assertEqual(profile.memory_per_item_mb(), 200.0 / 10.0)
        
        # Test with None memory_peak_mb
        profile.memory_peak_mb = None
        self.assertIsNone(profile.memory_per_item_mb())
        
        # Test with zero batch size
        profile.memory_peak_mb = 200.0
        profile.batch_size = 0
        self.assertIsNone(profile.memory_per_item_mb())
    
    def test_to_dict(self):
        """Test conversion of BatchProfile to dictionary."""
        # Create a GPU event for the batch profile
        event = GPUEvent(
            name="test_event",
            start_time=100.0,
            end_time=101.0,
            duration_ms=1000.0
        )
        
        # Create the batch profile
        profile = BatchProfile(
            batch_size=10,
            start_time=100.0,
            end_time=101.0,
            duration_ms=1000.0,
            items_per_second=10.0,
            events=[event],
            memory_peak_mb=200.0
        )
        
        # Convert to dictionary
        profile_dict = profile.to_dict()
        
        # Check that the dictionary contains all fields
        self.assertEqual(profile_dict["batch_size"], 10)
        self.assertEqual(profile_dict["start_time"], 100.0)
        self.assertEqual(profile_dict["end_time"], 101.0)
        self.assertEqual(profile_dict["duration_ms"], 1000.0)
        self.assertEqual(profile_dict["items_per_second"], 10.0)
        self.assertEqual(len(profile_dict["events"]), 1)
        self.assertEqual(profile_dict["events"][0]["name"], "test_event")
        self.assertEqual(profile_dict["memory_peak_mb"], 200.0)


class TestBatchSizeProfileResult(unittest.TestCase):
    """Tests for the BatchSizeProfileResult class."""
    
    def setUp(self):
        """Set up test batch profiles."""
        # Create batch profiles with different batch sizes
        self.profile1 = BatchProfile(
            batch_size=1,
            start_time=100.0,
            end_time=101.0,
            duration_ms=1000.0,
            items_per_second=1.0,
            cuda_time_ms=800.0,
            memory_peak_mb=100.0
        )
        
        self.profile2 = BatchProfile(
            batch_size=2,
            start_time=100.0,
            end_time=101.0,
            duration_ms=1500.0,
            items_per_second=1.33,
            cuda_time_ms=1300.0,
            memory_peak_mb=150.0
        )
        
        self.profile4 = BatchProfile(
            batch_size=4,
            start_time=100.0,
            end_time=101.0,
            duration_ms=2000.0,
            items_per_second=2.0,
            cuda_time_ms=1800.0,
            memory_peak_mb=250.0
        )
        
        self.profile8 = BatchProfile(
            batch_size=8,
            start_time=100.0,
            end_time=101.0,
            duration_ms=3000.0,
            items_per_second=2.67,
            cuda_time_ms=2700.0,
            memory_peak_mb=450.0
        )
    
    def test_batch_size_profile_result_creation(self):
        """Test creation of a BatchSizeProfileResult."""
        # Create the batch size profile result
        result = BatchSizeProfileResult(
            model_name="test-model",
            device_name="test-device",
            batch_sizes=[1, 2, 4, 8],
            profiles=[self.profile1, self.profile2, self.profile4, self.profile8],
            system_info={"gpu": "test-gpu"},
            cuda_version="11.0"
        )
        
        # Check that all fields are set correctly
        self.assertEqual(result.model_name, "test-model")
        self.assertEqual(result.device_name, "test-device")
        self.assertEqual(result.batch_sizes, [1, 2, 4, 8])
        self.assertEqual(len(result.profiles), 4)
        self.assertEqual(result.profiles[0], self.profile1)
        self.assertEqual(result.profiles[1], self.profile2)
        self.assertEqual(result.profiles[2], self.profile4)
        self.assertEqual(result.profiles[3], self.profile8)
        self.assertEqual(result.system_info, {"gpu": "test-gpu"})
        self.assertEqual(result.cuda_version, "11.0")
    
    def test_to_dict(self):
        """Test conversion of BatchSizeProfileResult to dictionary."""
        # Create the batch size profile result
        result = BatchSizeProfileResult(
            model_name="test-model",
            device_name="test-device",
            batch_sizes=[1, 2, 4, 8],
            profiles=[self.profile1, self.profile2, self.profile4, self.profile8],
            system_info={"gpu": "test-gpu"},
            cuda_version="11.0"
        )
        
        # Convert to dictionary
        result_dict = result.to_dict()
        
        # Check that the dictionary contains all fields
        self.assertEqual(result_dict["model_name"], "test-model")
        self.assertEqual(result_dict["device_name"], "test-device")
        self.assertEqual(result_dict["batch_sizes"], [1, 2, 4, 8])
        self.assertEqual(len(result_dict["profiles"]), 4)
        self.assertEqual(result_dict["profiles"][0]["batch_size"], 1)
        self.assertEqual(result_dict["system_info"], {"gpu": "test-gpu"})
        self.assertEqual(result_dict["cuda_version"], "11.0")
    
    def test_to_json(self):
        """Test conversion of BatchSizeProfileResult to JSON."""
        # Create the batch size profile result
        result = BatchSizeProfileResult(
            model_name="test-model",
            device_name="test-device",
            batch_sizes=[1, 2, 4, 8],
            profiles=[self.profile1, self.profile2, self.profile4, self.profile8],
            system_info={"gpu": "test-gpu"},
            cuda_version="11.0"
        )
        
        # Convert to JSON
        result_json = result.to_json()
        
        # Parse JSON back to dictionary
        result_dict = json.loads(result_json)
        
        # Check that the dictionary contains all fields
        self.assertEqual(result_dict["model_name"], "test-model")
        self.assertEqual(result_dict["device_name"], "test-device")
        self.assertEqual(result_dict["batch_sizes"], [1, 2, 4, 8])
        self.assertEqual(len(result_dict["profiles"]), 4)
        self.assertEqual(result_dict["profiles"][0]["batch_size"], 1)
        self.assertEqual(result_dict["system_info"], {"gpu": "test-gpu"})
        self.assertEqual(result_dict["cuda_version"], "11.0")
    
    def test_get_optimal_batch_size(self):
        """Test finding the optimal batch size based on throughput."""
        # Create the batch size profile result
        result = BatchSizeProfileResult(
            model_name="test-model",
            device_name="test-device",
            batch_sizes=[1, 2, 4, 8],
            profiles=[self.profile1, self.profile2, self.profile4, self.profile8],
            system_info={"gpu": "test-gpu"},
            cuda_version="11.0"
        )
        
        # Find optimal batch size (highest items_per_second)
        optimal_batch_size = result.get_optimal_batch_size()
        
        # Batch size 8 has the highest items_per_second (2.67)
        self.assertEqual(optimal_batch_size, 8)
        
        # Test with empty profiles
        result.profiles = []
        self.assertEqual(result.get_optimal_batch_size(), 1)
    
    def test_get_optimal_batch_size_efficiency(self):
        """Test finding the optimal batch size based on GPU efficiency."""
        # Create the batch size profile result
        result = BatchSizeProfileResult(
            model_name="test-model",
            device_name="test-device",
            batch_sizes=[1, 2, 4, 8],
            profiles=[self.profile1, self.profile2, self.profile4, self.profile8],
            system_info={"gpu": "test-gpu"},
            cuda_version="11.0"
        )
        
        # Find optimal batch size based on efficiency (cuda_time_ms / duration_ms)
        optimal_batch_size = result.get_optimal_batch_size_efficiency()
        
        # Batch size 8 has the highest efficiency (2700.0 / 3000.0 = 0.9)
        self.assertEqual(optimal_batch_size, 8)
        
        # Test with empty profiles
        result.profiles = []
        self.assertEqual(result.get_optimal_batch_size_efficiency(), 1)
    
    def test_generate_analysis_report(self):
        """Test generation of analysis report."""
        # Create the batch size profile result
        result = BatchSizeProfileResult(
            model_name="test-model",
            device_name="test-device",
            batch_sizes=[1, 2, 4, 8],
            profiles=[self.profile1, self.profile2, self.profile4, self.profile8],
            system_info={"gpu": "test-gpu"},
            cuda_version="11.0"
        )
        
        # Generate analysis report
        report = result.generate_analysis_report()
        
        # Check that the report contains the expected sections
        self.assertIn("optimal_batch_size", report)
        self.assertIn("throughput", report["optimal_batch_size"])
        self.assertIn("efficiency", report["optimal_batch_size"])
        self.assertIn("relative_performances", report)
        self.assertIn("memory_scaling", report)
        self.assertIn("bottlenecks", report)
        self.assertIn("recommendations", report)
        
        # Check that the optimal batch sizes are correct
        self.assertEqual(report["optimal_batch_size"]["throughput"], 8)
        self.assertEqual(report["optimal_batch_size"]["efficiency"], 8)
        
        # Test with empty profiles
        result.profiles = []
        self.assertEqual(result.generate_analysis_report(), {"error": "No profiling data available"})


class TestGPUProfiler(unittest.TestCase):
    """Tests for the GPUProfiler class."""
    
    @patch('langchain_hana.monitoring.profiler.HAS_TORCH', False)
    @patch('langchain_hana.monitoring.profiler.HAS_NVTX', False)
    @patch('langchain_hana.monitoring.profiler.HAS_PYINSTRUMENT', False)
    def test_initialization_without_gpu(self):
        """Test initialization without GPU libraries."""
        profiler = GPUProfiler(device_id=0)
        
        # Check that GPU-specific features are disabled
        self.assertFalse(profiler.nvtx_ranges)
        self.assertFalse(profiler.enable_pyinstrument)
        self.assertFalse(profiler.memory_stats)
        self.assertFalse(profiler.collect_cuda_events)
    
    @patch('langchain_hana.monitoring.profiler.time')
    def test_start_end_event(self, mock_time):
        """Test starting and ending an event."""
        # Set up mock time
        mock_time.time.side_effect = [100.0, 101.0]
        
        # Create profiler
        profiler = GPUProfiler(device_id=0)
        
        # Start and end an event
        profiler.start_event("test_event")
        event = profiler.end_event()
        
        # Check the event properties
        self.assertEqual(event.name, "test_event")
        self.assertEqual(event.start_time, 100.0)
        self.assertEqual(event.end_time, 101.0)
        self.assertEqual(event.duration_ms, 1000.0)  # 1.0 seconds = 1000 ms
    
    @patch('langchain_hana.monitoring.profiler.time')
    def test_profile_function(self, mock_time):
        """Test profiling a function."""
        # Set up mock time
        mock_time.time.side_effect = [100.0, 101.0]
        
        # Create profiler
        profiler = GPUProfiler(device_id=0)
        
        # Define a test function
        def test_function(x, y):
            return x + y
        
        # Profile the function
        result, event = profiler.profile_function(test_function, 1, 2)
        
        # Check the result and event
        self.assertEqual(result, 3)
        self.assertEqual(event.name, "test_function")
        self.assertEqual(event.duration_ms, 1000.0)
    
    @patch('langchain_hana.monitoring.profiler.time')
    def test_profile_batch_sizes(self, mock_time):
        """Test profiling different batch sizes."""
        # Set up mock time for consistent timing
        start_time = 100.0
        mock_time.time.return_value = start_time
        
        # Create profiler
        profiler = GPUProfiler(device_id=0, warmup_iterations=0)
        
        # Mock out _get_gpu_memory_info to return None
        profiler._get_gpu_memory_info = MagicMock(return_value=None)
        
        # Create test operation function and data generator
        def operation_fn(batch):
            return [i * 2 for i in batch]
        
        def data_generator_fn(batch_size):
            return list(range(batch_size))
        
        # Mock time.time to increment by 0.1 seconds for each call
        call_count = [0]
        def side_effect():
            call_count[0] += 1
            return start_time + (call_count[0] * 0.1)
        mock_time.time.side_effect = side_effect
        
        # Profile batch sizes
        batch_sizes = [1, 2, 4]
        result = profiler.profile_batch_sizes(
            operation_fn=operation_fn,
            batch_sizes=batch_sizes,
            data_generator_fn=data_generator_fn,
            model_name="test-model",
            iterations=1
        )
        
        # Check the result
        self.assertEqual(result.model_name, "test-model")
        self.assertEqual(result.batch_sizes, batch_sizes)
        self.assertEqual(len(result.profiles), 3)
        
        # Check that profiles contain expected batch sizes
        actual_batch_sizes = [p.batch_size for p in result.profiles]
        self.assertEqual(actual_batch_sizes, batch_sizes)


if __name__ == "__main__":
    unittest.main()