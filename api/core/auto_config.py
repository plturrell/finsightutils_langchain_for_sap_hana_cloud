"""
Auto-configuration module for SAP HANA Cloud LangChain Integration.

This module provides:
1. Dynamic loading of configuration from auto-tuning
2. Runtime parameter adjustment based on system conditions
3. Continuous configuration optimization
4. Integration with monitoring systems
"""

import json
import logging
import os
import time
from typing import Dict, List, Optional, Union, Any

# Configure logging
logger = logging.getLogger(__name__)

# Default configuration values
DEFAULT_CONFIG = {
    'batch_size': {
        'default': 32,
        'embedding_generation': 64,
        'vector_search': 16,
        'max_batch_size': 128,
    },
    'precision': 'fp16',
    'worker_counts': {
        'api_workers': 4,
        'gpu_workers': 1,
        'db_pool_size': 8,
        'thread_count': 4,
    },
    'memory_allocation': {
        'memory_fraction': 0.8,
        'cache_size_mb': 2048,
        'max_workspace_size_mb': 1024,
    },
    'hnsw_parameters': {
        'm': 16,
        'ef_construction': 100,
        'ef_search': 50,
    },
}


class AutoConfig:
    """
    Auto-configuration manager.
    
    Handles dynamic configuration loading and adjustment.
    """
    
    def __init__(
        self,
        config_dir: str = '/app/config',
        auto_tuned_file: str = 'auto_tuned_config.json',
        learned_dir: str = 'learned',
    ):
        """
        Initialize the auto-configuration manager.
        
        Args:
            config_dir: Path to configuration directory
            auto_tuned_file: Name of auto-tuned configuration file
            learned_dir: Name of learned configuration directory
        """
        self.config_dir = config_dir
        self.auto_tuned_file = os.path.join(config_dir, auto_tuned_file)
        self.learned_dir = os.path.join(config_dir, learned_dir)
        self.config = DEFAULT_CONFIG.copy()
        self.last_update = 0
        self.update_interval = 60  # seconds
        
        # Load initial configuration
        self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from file."""
        # Check for auto-tuned configuration
        if os.path.exists(self.auto_tuned_file):
            try:
                with open(self.auto_tuned_file, 'r') as f:
                    auto_tuned_config = json.load(f)
                
                # Update configuration with auto-tuned values
                self.update_from_auto_tuned(auto_tuned_config)
                logger.info(f"Auto-tuned configuration loaded from {self.auto_tuned_file}")
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Error loading auto-tuned configuration: {e}")
        else:
            logger.info(f"Auto-tuned configuration not found at {self.auto_tuned_file}")
        
        # Check for learned configuration
        self.check_learned_config()
        
        # Apply environment variable overrides
        self.apply_env_overrides()
        
        # Update last update timestamp
        self.last_update = time.time()
    
    def update_from_auto_tuned(self, auto_tuned_config: Dict[str, Any]) -> None:
        """
        Update configuration from auto-tuned values.
        
        Args:
            auto_tuned_config: Auto-tuned configuration dict
        """
        # Update batch sizes
        if 'batch_sizes' in auto_tuned_config:
            self.config['batch_size'] = auto_tuned_config['batch_sizes']
        
        # Update precision
        if 'precision' in auto_tuned_config:
            self.config['precision'] = auto_tuned_config['precision']
        
        # Update worker counts
        if 'worker_counts' in auto_tuned_config:
            self.config['worker_counts'] = auto_tuned_config['worker_counts']
        
        # Update memory allocation
        if 'memory_allocation' in auto_tuned_config:
            self.config['memory_allocation'] = auto_tuned_config['memory_allocation']
        
        # Update HNSW parameters
        if 'hnsw_parameters' in auto_tuned_config:
            self.config['hnsw_parameters'] = auto_tuned_config['hnsw_parameters']
    
    def check_learned_config(self) -> None:
        """Check for and load the latest learned configuration."""
        if not os.path.exists(self.learned_dir):
            return
        
        # Find the latest learned configuration
        learned_files = [
            f for f in os.listdir(self.learned_dir)
            if f.startswith('learned_config_') and f.endswith('.json')
        ]
        
        if not learned_files:
            return
        
        # Sort by iteration number
        learned_files.sort(key=lambda f: int(f.split('_')[-1].split('.')[0]), reverse=True)
        latest_file = os.path.join(self.learned_dir, learned_files[0])
        
        try:
            with open(latest_file, 'r') as f:
                learned_config = json.load(f)
            
            # Update configuration with learned parameters
            if 'parameters' in learned_config:
                # Apply learned parameters
                for param_name, param_value in learned_config['parameters'].items():
                    if param_name == 'batch_size':
                        self.config['batch_size']['default'] = param_value
                    elif param_name == 'gpu_memory_fraction':
                        self.config['memory_allocation']['memory_fraction'] = param_value
                    elif param_name == 'worker_count':
                        self.config['worker_counts']['api_workers'] = param_value
            
            logger.info(f"Learned configuration loaded from {latest_file}")
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Error loading learned configuration: {e}")
    
    def apply_env_overrides(self) -> None:
        """Apply environment variable overrides to configuration."""
        # Override batch size
        if 'BATCH_SIZE' in os.environ:
            try:
                batch_size = int(os.environ['BATCH_SIZE'])
                self.config['batch_size']['default'] = batch_size
                logger.info(f"Batch size overridden from environment: {batch_size}")
            except ValueError:
                pass
        
        # Override maximum batch size
        if 'MAX_BATCH_SIZE' in os.environ:
            try:
                max_batch_size = int(os.environ['MAX_BATCH_SIZE'])
                self.config['batch_size']['max_batch_size'] = max_batch_size
                logger.info(f"Maximum batch size overridden from environment: {max_batch_size}")
            except ValueError:
                pass
        
        # Override precision
        if 'TENSORRT_PRECISION' in os.environ:
            precision = os.environ['TENSORRT_PRECISION'].lower()
            if precision in ['fp32', 'fp16', 'int8']:
                self.config['precision'] = precision
                logger.info(f"Precision overridden from environment: {precision}")
        
        # Override GPU memory fraction
        if 'GPU_MEMORY_FRACTION' in os.environ:
            try:
                memory_fraction = float(os.environ['GPU_MEMORY_FRACTION'])
                if 0.0 < memory_fraction <= 1.0:
                    self.config['memory_allocation']['memory_fraction'] = memory_fraction
                    logger.info(f"GPU memory fraction overridden from environment: {memory_fraction}")
            except ValueError:
                pass
        
        # Override worker count
        if 'API_WORKERS' in os.environ:
            try:
                api_workers = int(os.environ['API_WORKERS'])
                if api_workers > 0:
                    self.config['worker_counts']['api_workers'] = api_workers
                    logger.info(f"API workers overridden from environment: {api_workers}")
            except ValueError:
                pass
    
    def check_for_updates(self) -> None:
        """Check for configuration updates."""
        current_time = time.time()
        if current_time - self.last_update > self.update_interval:
            self.load_config()
    
    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key (can use dot notation for nested keys)
            default: Default value if key is not found
            
        Returns:
            Configuration value or default
        """
        self.check_for_updates()
        
        # Handle nested keys
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_batch_size(self, operation: str = 'default') -> int:
        """
        Get batch size for specific operation.
        
        Dynamically adjusts batch size based on current system conditions.
        
        Args:
            operation: Operation type (default, embedding_generation, vector_search)
            
        Returns:
            Batch size for the operation
        """
        self.check_for_updates()
        
        # Get batch size from configuration
        batch_sizes = self.config['batch_size']
        
        if operation in batch_sizes:
            batch_size = batch_sizes[operation]
        else:
            batch_size = batch_sizes['default']
        
        # Dynamically adjust based on system load
        try:
            import psutil
            system_load = psutil.cpu_percent(interval=0.1)
            
            # Reduce batch size if system is under heavy load
            if system_load > 90:
                batch_size = max(1, batch_size // 2)
            elif system_load > 75:
                batch_size = max(1, int(batch_size * 0.75))
        except ImportError:
            pass
        
        # Check for GPU memory pressure
        try:
            import torch
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
                memory_allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)  # GB
                memory_total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)  # GB
                
                # Calculate memory usage percentage
                memory_usage = memory_allocated / memory_total
                
                # Reduce batch size if GPU memory is under pressure
                if memory_usage > 0.9:
                    batch_size = max(1, batch_size // 2)
                elif memory_usage > 0.8:
                    batch_size = max(1, int(batch_size * 0.75))
        except (ImportError, RuntimeError):
            pass
        
        return batch_size
    
    def get_precision(self) -> str:
        """
        Get precision mode.
        
        Dynamically adjusts precision based on GPU capabilities.
        
        Returns:
            Precision mode (fp32, fp16, int8)
        """
        self.check_for_updates()
        
        precision = self.config['precision']
        
        # Check for GPU capabilities
        try:
            import torch
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
                gpu_compute_capability = torch.cuda.get_device_capability(device)
                
                # Tensor Cores available on Volta (7.0), Turing (7.5), Ampere (8.0+)
                has_tensor_cores = gpu_compute_capability[0] >= 7
                
                # INT8 precision available on Turing (7.5) and later
                supports_int8 = gpu_compute_capability[0] > 7 or (
                    gpu_compute_capability[0] == 7 and gpu_compute_capability[1] >= 5
                )
                
                # FP16 precision available on Pascal (6.0) and later
                supports_fp16 = gpu_compute_capability[0] >= 6
                
                # Adjust precision based on GPU capabilities
                if precision == 'int8' and not supports_int8:
                    precision = 'fp16'
                
                if precision == 'fp16' and not supports_fp16:
                    precision = 'fp32'
        except (ImportError, RuntimeError):
            pass
        
        return precision
    
    def get_hnsw_parameters(self) -> Dict[str, int]:
        """
        Get HNSW parameters for vector search.
        
        Returns:
            Dict with HNSW parameters
        """
        self.check_for_updates()
        
        return self.config['hnsw_parameters']
    
    def get_worker_count(self, worker_type: str = 'api_workers') -> int:
        """
        Get worker count for specific service.
        
        Dynamically adjusts worker count based on system resources.
        
        Args:
            worker_type: Worker type (api_workers, gpu_workers, thread_count)
            
        Returns:
            Worker count
        """
        self.check_for_updates()
        
        worker_counts = self.config['worker_counts']
        
        if worker_type in worker_counts:
            worker_count = worker_counts[worker_type]
        else:
            worker_count = worker_counts.get('api_workers', 4)
        
        # Dynamically adjust based on system resources
        try:
            import psutil
            cpu_count = psutil.cpu_count()
            
            if worker_type == 'api_workers':
                # Use at most CPU count - 1
                worker_count = min(worker_count, max(1, cpu_count - 1))
            elif worker_type == 'thread_count':
                # Use at most CPU count / 2
                worker_count = min(worker_count, max(1, cpu_count // 2))
        except ImportError:
            pass
        
        # Adjust GPU workers based on available GPUs
        if worker_type == 'gpu_workers':
            try:
                import torch
                if torch.cuda.is_available():
                    # Use at most the number of available GPUs
                    gpu_count = torch.cuda.device_count()
                    worker_count = min(worker_count, gpu_count)
            except ImportError:
                pass
        
        return worker_count
    
    def get_memory_allocation(self) -> Dict[str, Union[float, int]]:
        """
        Get memory allocation parameters.
        
        Dynamically adjusts memory allocation based on system memory.
        
        Returns:
            Dict with memory allocation parameters
        """
        self.check_for_updates()
        
        memory_allocation = self.config['memory_allocation'].copy()
        
        # Dynamically adjust based on system memory
        try:
            import psutil
            total_memory = psutil.virtual_memory().total / (1024 ** 3)  # GB
            
            # Adjust cache size based on system memory
            if total_memory > 64:
                memory_allocation['cache_size_mb'] = max(memory_allocation['cache_size_mb'], 8192)
            elif total_memory > 32:
                memory_allocation['cache_size_mb'] = max(memory_allocation['cache_size_mb'], 4096)
            elif total_memory > 16:
                memory_allocation['cache_size_mb'] = max(memory_allocation['cache_size_mb'], 2048)
            else:
                memory_allocation['cache_size_mb'] = max(memory_allocation['cache_size_mb'], 1024)
        except ImportError:
            pass
        
        return memory_allocation


# Create global configuration instance
config = AutoConfig()