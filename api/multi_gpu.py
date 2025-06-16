"""Multi-GPU management and allocation module.

This module provides utilities for managing multiple GPUs and allocating
workloads across them. It includes automatic CPU fallback for environments
without NVIDIA GPUs.
"""

import logging
import os
from typing import List, Dict, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)

# Environment-based configuration
CPU_ONLY_MODE = os.environ.get('CPU_ONLY_MODE', '').lower() == 'true'
DISABLE_GPU_CHECK = os.environ.get('DISABLE_GPU_CHECK', '').lower() == 'true'
TEST_MODE = os.environ.get('TEST_MODE', '').lower() == 'true'


class GPUManager:
    """Manages GPU resources and allocations with automatic CPU fallback.

    This class provides a production-standard implementation that supports:
    1. Automatic detection of GPU devices
    2. Environment-based CPU-only mode configuration
    3. Graceful fallbacks for CPU-only environments
    4. Round-robin GPU allocation for load balancing
    """

    def __init__(self):
        """Initialize the GPU manager with automatic environment detection."""
        self.cpu_only = CPU_ONLY_MODE
        self.devices = []
        self.device_count = 0
        self.current_idx = 0

        if self.cpu_only:
            logger.info(
                "Running in CPU-only mode (set by environment variable). "
                "Multi-GPU functionality disabled."
            )
            return

        # Try to detect GPUs if not explicitly in CPU-only mode
        try:
            if not DISABLE_GPU_CHECK:
                # Only import GPU modules if not in CPU-only mode
                try:
                    # First try to import from local relative path
                    from api.gpu import gpu_utils
                except (ImportError, ModuleNotFoundError):
                    try:
                        # Then try absolute import
                        import gpu_utils
                    except (ImportError, ModuleNotFoundError):
                        # If both fail, we're in CPU-only environment
                        logger.warning(
                            "gpu_utils module not found. "
                            "Falling back to CPU mode."
                        )
                        self.cpu_only = True
                        return

                # If we got here, gpu_utils is available, check for GPUs
                gpu_info = gpu_utils.get_gpu_info()
                self.device_count = gpu_info.get('gpu_count', 0)
                self.devices = list(range(self.device_count))

                if self.device_count > 0:
                    logger.info(
                        "Found %s GPUs: %s",
                        self.device_count,
                        gpu_info.get('gpu_names', [])
                    )
                else:
                    logger.warning("No GPUs detected, falling back to CPU mode.")
                    self.cpu_only = True
            else:
                logger.info("GPU checks disabled. Operating in CPU-only mode.")
                self.cpu_only = True

        except Exception as e:  # pylint: disable=broad-except
            logger.warning(
                "Error detecting GPUs, falling back to CPU mode: %s",
                str(e)
            )
            self.cpu_only = True

    def get_next_available_gpu(self) -> Optional[int]:
        """Get the next available GPU index using round-robin allocation.

        Returns:
            GPU index or None if in CPU-only mode.
        """
        if self.cpu_only or self.device_count == 0:
            logger.debug("No GPUs available, using CPU instead.")
            return None

        # Simple round-robin allocation
        gpu_id = self.devices[self.current_idx]
        self.current_idx = (self.current_idx + 1) % self.device_count
        return gpu_id

    def get_gpu_utilization(self) -> List[Dict[str, Any]]:
        """Get GPU utilization information.

        Returns:
            List of dictionaries with GPU utilization information or empty list
            in CPU-only mode.
        """
        if self.cpu_only or self.device_count == 0:
            return []

        try:
            # Only import gpu-specific modules if we have GPUs
            try:
                from api.gpu import gpu_utils
            except (ImportError, ModuleNotFoundError):
                try:
                    import gpu_utils
                except (ImportError, ModuleNotFoundError):
                    return []

            return gpu_utils.get_gpu_utilization()
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("Error getting GPU utilization: %s", str(e))
            return []


# Singleton instance of the GPU manager
_gpu_manager = None


def get_gpu_manager() -> GPUManager:
    """Get a GPU manager instance (singleton pattern).

    Returns:
        GPUManager instance with automatic CPU fallback.
    """
    global _gpu_manager  # pylint: disable=global-statement
    if _gpu_manager is None:
        _gpu_manager = GPUManager()
    return _gpu_manager
