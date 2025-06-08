"""Unit tests for the multi-GPU manager."""

import os
import time
import threading
import unittest
from unittest.mock import patch, MagicMock

import pytest

from langchain_hana.gpu.multi_gpu_manager import (
    EnhancedMultiGPUManager, 
    get_multi_gpu_manager,
    Task,
    TaskResult,
    GPUInfo
)


class TestGPUInfo(unittest.TestCase):
    """Test the GPUInfo class."""
    
    def test_initialization(self):
        """Test that GPUInfo initializes correctly."""
        gpu_info = GPUInfo(device_id=0)
        
        # Basic properties should be set
        self.assertEqual(gpu_info.device_id, 0)
        self.assertEqual(gpu_info.device_name, "cuda:0")
        self.assertEqual(gpu_info.active_tasks, 0)
        self.assertEqual(gpu_info.completed_tasks, 0)
        self.assertTrue(gpu_info.is_available)

    def test_to_dict(self):
        """Test the to_dict method."""
        gpu_info = GPUInfo(device_id=0)
        
        # Mock properties
        gpu_info.properties = MagicMock()
        gpu_info.properties.name = "Test GPU"
        gpu_info.properties.total_memory = 1024 * 1024 * 1024  # 1 GB
        gpu_info.compute_capability = (7, 5)
        gpu_info.supports_tensor_cores = True
        gpu_info.memory_allocated = 512 * 1024 * 1024  # 512 MB
        gpu_info.memory_reserved = 768 * 1024 * 1024  # 768 MB
        gpu_info.utilization = 0.5
        gpu_info.temperature = 65
        gpu_info.power_draw = 150
        gpu_info.active_tasks = 2
        gpu_info.completed_tasks = 10
        
        # Convert to dict and verify fields
        info_dict = gpu_info.to_dict()
        self.assertEqual(info_dict["device_id"], 0)
        self.assertEqual(info_dict["device_name"], "cuda:0")
        self.assertEqual(info_dict["name"], "Test GPU")
        self.assertEqual(info_dict["compute_capability"], "7.5")
        self.assertTrue(info_dict["supports_tensor_cores"])
        self.assertEqual(info_dict["total_memory_mb"], 1024)
        self.assertEqual(info_dict["memory_allocated_mb"], 512)
        self.assertEqual(info_dict["memory_reserved_mb"], 768)
        self.assertEqual(info_dict["utilization"], 0.5)
        self.assertEqual(info_dict["temperature_c"], 65)
        self.assertEqual(info_dict["power_draw_watts"], 150)
        self.assertEqual(info_dict["active_tasks"], 2)
        self.assertEqual(info_dict["completed_tasks"], 10)


class TestTask(unittest.TestCase):
    """Test the Task class."""
    
    def test_task_creation(self):
        """Test task creation with various parameters."""
        # Test with minimal parameters
        task = Task(lambda: 42)
        self.assertIsNotNone(task.task_id)
        self.assertEqual(task.priority, 0)
        self.assertIsNone(task.device_preference)
        self.assertIsNone(task.timeout)
        
        # Test with all parameters
        task_id = "test-task"
        task = Task(
            func=lambda x: x * 2,
            args=(21,),
            kwargs={"extra": True},
            task_id=task_id,
            priority=10,
            device_preference=0,
            timeout=60.0,
        )
        self.assertEqual(task.task_id, task_id)
        self.assertEqual(task.priority, 10)
        self.assertEqual(task.device_preference, 0)
        self.assertEqual(task.timeout, 60.0)
        self.assertEqual(task.args, (21,))
        self.assertEqual(task.kwargs, {"extra": True})

    def test_task_execution_success(self):
        """Test successful task execution."""
        def test_func(x, y=1):
            return x + y
            
        task = Task(func=test_func, args=(1,), kwargs={"y": 2})
        
        # Execute task
        result = task.execute(device_id=0)
        
        # Verify result
        self.assertEqual(result.task_id, task.task_id)
        self.assertTrue(result.success)
        self.assertIsNone(result.error)
        self.assertEqual(result.result, 3)  # 1 + 2
        self.assertEqual(result.device_id, 0)
        self.assertGreaterEqual(result.execution_time, 0)

    def test_task_execution_failure(self):
        """Test task execution with an exception."""
        def failing_func():
            raise ValueError("Test error")
            
        task = Task(func=failing_func)
        
        # Execute task
        result = task.execute(device_id=0)
        
        # Verify result
        self.assertEqual(result.task_id, task.task_id)
        self.assertFalse(result.success)
        self.assertIsInstance(result.error, ValueError)
        self.assertEqual(str(result.error), "Test error")
        self.assertEqual(result.device_id, 0)


@pytest.mark.skipif(not os.environ.get("TEST_GPU"), reason="GPU tests disabled")
class TestEnhancedMultiGPUManager:
    """Test the EnhancedMultiGPUManager class."""
    
    @patch("torch.cuda.device_count", return_value=2)
    @patch("torch.cuda.set_device")
    def test_initialization(self, mock_set_device, mock_device_count):
        """Test manager initialization."""
        with patch("langchain_hana.gpu.multi_gpu_manager.TORCH_AVAILABLE", True):
            manager = EnhancedMultiGPUManager(enabled=True)
            
            # Verify manager state
            assert manager.enabled
            assert manager.initialized
            assert len(manager.devices) == 2
            
            # Verify worker threads created
            assert len(manager.worker_threads) == 2
            assert 0 in manager.worker_threads
            assert 1 in manager.worker_threads

    @patch("torch.cuda.is_available", return_value=False)
    def test_cpu_fallback(self, mock_cuda):
        """Test that manager falls back to CPU when CUDA is not available."""
        with patch("langchain_hana.gpu.multi_gpu_manager.TORCH_AVAILABLE", False):
            manager = EnhancedMultiGPUManager(enabled=True)
            
            # Verify manager is disabled
            assert not manager.enabled
            assert not manager.initialized

    def test_submit_task(self):
        """Test task submission."""
        # Create a mock manager
        manager = EnhancedMultiGPUManager(enabled=False)  # Disabled for testing
        
        # Mock function that returns its input
        def echo(x):
            return x
        
        # Submit task directly (will execute on CPU since GPU is disabled)
        task_id = manager.submit_task(func=echo, args=(42,))
        
        # For a disabled manager, the task executes immediately and returns a result
        assert task_id is not None

    @patch("torch.cuda.device_count", return_value=2)
    @patch("torch.cuda.set_device")
    def test_batch_process(self, mock_set_device, mock_device_count):
        """Test batch processing."""
        # Create a mock manager with mock device information
        with patch("langchain_hana.gpu.multi_gpu_manager.TORCH_AVAILABLE", True):
            manager = EnhancedMultiGPUManager(enabled=True)
            
            # Replace the submit_task method
            original_submit = manager.submit_task
            
            def mock_submit_task(*args, **kwargs):
                task_id = f"task-{len(manager.pending_tasks)}"
                # Immediately add a "result" to completed_tasks
                manager.completed_tasks[task_id] = TaskResult(
                    task_id=task_id,
                    result=kwargs["args"][0],  # Return the batch
                    device_id=0,
                    execution_time=0.01,
                )
                return task_id
                
            manager.submit_task = mock_submit_task
            
            # Test batch_process with a simple identity function
            batches = [["item1", "item2"], ["item3", "item4"]]
            
            results = manager.batch_process(
                func=lambda x: x,  # Identity function
                items=["item1", "item2", "item3", "item4"],
                batch_size=2,
                wait=True,
            )
            
            # Verify results
            assert len(results) == 4
            assert results == ["item1", "item2", "item3", "item4"]
            
            # Restore original method
            manager.submit_task = original_submit

    def test_get_multi_gpu_manager(self):
        """Test the global manager accessor."""
        # Test with environment variable
        with patch.dict(os.environ, {"MULTI_GPU_ENABLED": "true"}):
            with patch("langchain_hana.gpu.multi_gpu_manager.EnhancedMultiGPUManager") as mock_manager:
                # Mock instance returned by constructor
                mock_instance = MagicMock()
                mock_manager.return_value = mock_instance
                
                # Get the manager
                manager = get_multi_gpu_manager()
                
                # Verify manager was created with expected settings
                mock_manager.assert_called_once()
                assert mock_manager.call_args[1]["enabled"] is True
                
                # Subsequent calls should return the same instance
                manager2 = get_multi_gpu_manager()
                assert manager2 is mock_instance
                assert mock_manager.call_count == 1  # Constructor called only once