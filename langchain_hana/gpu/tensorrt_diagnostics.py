"""
TensorRT diagnostics utilities for financial embeddings.

This module provides diagnostic tools for TensorRT initialization, error handling,
and fallback mechanisms to ensure robust operation in production environments.
"""

import os
import sys
import logging
import inspect
import traceback
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
import json

logger = logging.getLogger(__name__)

class TensorRTDiagnosticResult:
    """Container for TensorRT diagnostic results."""
    
    def __init__(
        self,
        success: bool,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
        diagnostics: Optional[Dict[str, Any]] = None,
        recommendations: Optional[List[str]] = None,
        cuda_version: Optional[str] = None,
        tensorrt_version: Optional[str] = None,
        device_info: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize TensorRT diagnostic result.
        
        Parameters
        ----------
        success : bool
            Whether TensorRT initialization succeeded
        error_type : str, optional
            Type of error if initialization failed
        error_message : str, optional
            Detailed error message if initialization failed
        diagnostics : Dict[str, Any], optional
            Diagnostic information about the environment
        recommendations : List[str], optional
            Recommendations for fixing the error
        cuda_version : str, optional
            CUDA version detected
        tensorrt_version : str, optional
            TensorRT version detected
        device_info : Dict[str, Any], optional
            Information about the GPU device
        """
        self.success = success
        self.error_type = error_type
        self.error_message = error_message
        self.diagnostics = diagnostics or {}
        self.recommendations = recommendations or []
        self.cuda_version = cuda_version
        self.tensorrt_version = tensorrt_version
        self.device_info = device_info or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert diagnostic result to dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary representation of the diagnostic result
        """
        return {
            "success": self.success,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "diagnostics": self.diagnostics,
            "recommendations": self.recommendations,
            "cuda_version": self.cuda_version,
            "tensorrt_version": self.tensorrt_version,
            "device_info": self.device_info,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """
        Convert diagnostic result to JSON string.
        
        Parameters
        ----------
        indent : int, default=2
            Indentation level for JSON formatting
            
        Returns
        -------
        str
            JSON string representation of the diagnostic result
        """
        return json.dumps(self.to_dict(), indent=indent)
    
    def get_summary(self) -> str:
        """
        Get a human-readable summary of the diagnostic result.
        
        Returns
        -------
        str
            Summary of the diagnostic result
        """
        if self.success:
            summary = "TensorRT initialization successful."
            if self.cuda_version:
                summary += f" CUDA version: {self.cuda_version}."
            if self.tensorrt_version:
                summary += f" TensorRT version: {self.tensorrt_version}."
            if self.device_info and "name" in self.device_info:
                summary += f" Device: {self.device_info['name']}."
            return summary
        
        summary = f"TensorRT initialization failed: {self.error_type}"
        if self.error_message:
            summary += f" - {self.error_message}"
        
        if self.recommendations:
            summary += "\n\nRecommendations:"
            for i, recommendation in enumerate(self.recommendations, 1):
                summary += f"\n{i}. {recommendation}"
        
        return summary


class TensorRTDiagnostics:
    """
    Diagnostics utilities for TensorRT initialization and error handling.
    
    This class provides tools for diagnosing TensorRT initialization issues,
    collecting environment information, and providing actionable recommendations.
    """
    
    @staticmethod
    def check_cuda_availability() -> Tuple[bool, Dict[str, Any]]:
        """
        Check if CUDA is available and collect version information.
        
        Returns
        -------
        Tuple[bool, Dict[str, Any]]
            Tuple of (success, diagnostics)
        """
        diagnostics = {}
        
        # Check if PyTorch is available
        try:
            import torch
            diagnostics["torch_available"] = True
            diagnostics["torch_version"] = torch.__version__
            
            # Check CUDA availability
            cuda_available = torch.cuda.is_available()
            diagnostics["cuda_available"] = cuda_available
            
            if cuda_available:
                diagnostics["cuda_version"] = torch.version.cuda
                diagnostics["cuda_device_count"] = torch.cuda.device_count()
                
                # Get device information
                device_info = []
                for i in range(torch.cuda.device_count()):
                    device = {
                        "index": i,
                        "name": torch.cuda.get_device_name(i),
                        "memory_total": torch.cuda.get_device_properties(i).total_memory,
                    }
                    device_info.append(device)
                diagnostics["devices"] = device_info
                
                return True, diagnostics
            
            return False, diagnostics
            
        except ImportError:
            diagnostics["torch_available"] = False
            return False, diagnostics
    
    @staticmethod
    def check_tensorrt_availability() -> Tuple[bool, Dict[str, Any]]:
        """
        Check if TensorRT is available and collect version information.
        
        Returns
        -------
        Tuple[bool, Dict[str, Any]]
            Tuple of (success, diagnostics)
        """
        diagnostics = {}
        
        # Try to import TensorRT
        try:
            import tensorrt
            diagnostics["tensorrt_available"] = True
            diagnostics["tensorrt_version"] = tensorrt.__version__
            return True, diagnostics
        except ImportError:
            diagnostics["tensorrt_available"] = False
            return False, diagnostics
    
    @staticmethod
    def analyze_error(error: Exception) -> Tuple[str, List[str]]:
        """
        Analyze a TensorRT initialization error and provide recommendations.
        
        Parameters
        ----------
        error : Exception
            The exception raised during TensorRT initialization
            
        Returns
        -------
        Tuple[str, List[str]]
            Tuple of (error_type, recommendations)
        """
        error_type = type(error).__name__
        error_message = str(error)
        recommendations = []
        
        # Check for common error types and provide specific recommendations
        if error_type == "ImportError":
            if "tensorrt" in error_message.lower():
                error_type = "TensorRTNotInstalled"
                recommendations.extend([
                    "Install TensorRT following the NVIDIA installation guide",
                    "Ensure compatible versions of CUDA, cuDNN, and TensorRT",
                    "Verify system PATH includes TensorRT libraries"
                ])
            elif "cuda" in error_message.lower():
                error_type = "CUDANotAvailable"
                recommendations.extend([
                    "Install CUDA toolkit matching your GPU driver version",
                    "Ensure CUDA_HOME environment variable is set correctly",
                    "Verify GPU drivers are installed and up to date"
                ])
                
        elif error_type == "RuntimeError":
            if "cuda out of memory" in error_message.lower():
                error_type = "CUDAOutOfMemory"
                recommendations.extend([
                    "Reduce batch size or model size",
                    "Use a smaller precision (FP16 instead of FP32)",
                    "Increase GPU memory or use a GPU with more memory",
                    "Enable memory optimization techniques like gradient checkpointing"
                ])
            elif "driver version" in error_message.lower():
                error_type = "CUDADriverMismatch"
                recommendations.extend([
                    "Update GPU drivers to match CUDA toolkit version",
                    "Reinstall CUDA toolkit to match GPU driver version"
                ])
            elif "tensorrt" in error_message.lower() and "engine" in error_message.lower():
                error_type = "TensorRTEngineBuildFailure"
                recommendations.extend([
                    "Check model compatibility with TensorRT",
                    "Ensure calibration data is appropriate for the model",
                    "Try different precision settings (FP32, FP16, or INT8)",
                    "Rebuild TensorRT engine with force_engine_rebuild=True"
                ])
                
        elif error_type == "ValueError":
            if "device" in error_message.lower():
                error_type = "InvalidDeviceConfiguration"
                recommendations.extend([
                    "Verify CUDA device is available and visible",
                    "Check device index is valid for your system",
                    "Ensure no other processes are using the GPU exclusively"
                ])
        
        # If no specific recommendations, add general ones
        if not recommendations:
            recommendations.extend([
                "Check TensorRT and CUDA compatibility with your hardware",
                "Verify environment variables are set correctly",
                "Consider using standard embeddings without TensorRT acceleration"
            ])
            
        return error_type, recommendations
    
    @staticmethod
    def run_diagnostics() -> TensorRTDiagnosticResult:
        """
        Run comprehensive TensorRT diagnostics.
        
        Returns
        -------
        TensorRTDiagnosticResult
            Diagnostic result object
        """
        # Check CUDA availability
        cuda_success, cuda_diagnostics = TensorRTDiagnostics.check_cuda_availability()
        
        # Check TensorRT availability
        tensorrt_success, tensorrt_diagnostics = TensorRTDiagnostics.check_tensorrt_availability()
        
        # Combine diagnostics
        diagnostics = {**cuda_diagnostics, **tensorrt_diagnostics}
        
        # Determine overall success
        success = cuda_success and tensorrt_success
        
        # Create device info
        device_info = None
        if cuda_success and "devices" in cuda_diagnostics and cuda_diagnostics["devices"]:
            device_info = cuda_diagnostics["devices"][0]
        
        # Get versions
        cuda_version = cuda_diagnostics.get("cuda_version") if cuda_success else None
        tensorrt_version = tensorrt_diagnostics.get("tensorrt_version") if tensorrt_success else None
        
        # Generate recommendations if needed
        recommendations = []
        if not success:
            if not cuda_success:
                recommendations.extend([
                    "Install or update CUDA toolkit",
                    "Verify GPU drivers are installed and up to date",
                    "Ensure CUDA_HOME environment variable is set correctly"
                ])
            if not tensorrt_success:
                recommendations.extend([
                    "Install TensorRT following the NVIDIA installation guide",
                    "Ensure compatible versions of CUDA, cuDNN, and TensorRT",
                    "Verify system PATH includes TensorRT libraries"
                ])
        
        return TensorRTDiagnosticResult(
            success=success,
            diagnostics=diagnostics,
            recommendations=recommendations,
            cuda_version=cuda_version,
            tensorrt_version=tensorrt_version,
            device_info=device_info
        )
    
    @staticmethod
    def diagnose_error(error: Exception) -> TensorRTDiagnosticResult:
        """
        Diagnose a TensorRT initialization error.
        
        Parameters
        ----------
        error : Exception
            The exception raised during TensorRT initialization
            
        Returns
        -------
        TensorRTDiagnosticResult
            Diagnostic result object
        """
        # Run basic diagnostics
        diagnostics = TensorRTDiagnostics.run_diagnostics()
        
        # Analyze specific error
        error_type, recommendations = TensorRTDiagnostics.analyze_error(error)
        
        # Combine with diagnostics
        diagnostics.success = False
        diagnostics.error_type = error_type
        diagnostics.error_message = str(error)
        diagnostics.recommendations.extend(recommendations)
        
        return diagnostics


def with_tensorrt_diagnostics(func: Callable) -> Callable:
    """
    Decorator to add TensorRT diagnostics to a function.
    
    This decorator catches TensorRT-related exceptions, runs diagnostics,
    and provides detailed error information and recommendations.
    
    Parameters
    ----------
    func : Callable
        Function to decorate
        
    Returns
    -------
    Callable
        Decorated function with TensorRT diagnostics
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Run diagnostics
            diagnostics = TensorRTDiagnostics.diagnose_error(e)
            
            # Log detailed error information
            logger.error(f"TensorRT error: {diagnostics.error_type} - {diagnostics.error_message}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            logger.error(f"Diagnostics: {json.dumps(diagnostics.diagnostics, indent=2)}")
            
            if diagnostics.recommendations:
                logger.error("Recommendations:")
                for i, recommendation in enumerate(diagnostics.recommendations, 1):
                    logger.error(f"{i}. {recommendation}")
            
            # Re-raise with enhanced information
            raise RuntimeError(
                f"TensorRT initialization failed: {diagnostics.error_type} - {diagnostics.error_message}\n"
                f"Run TensorRTDiagnostics.run_diagnostics() for detailed diagnostics.\n"
                f"Recommendations: {', '.join(diagnostics.recommendations)}"
            ) from e
    
    return wrapper


def try_import_tensorrt() -> Tuple[bool, Optional[Any], Optional[TensorRTDiagnosticResult]]:
    """
    Try to import TensorRT with diagnostics.
    
    Returns
    -------
    Tuple[bool, Optional[Any], Optional[TensorRTDiagnosticResult]]
        Tuple of (success, tensorrt_module, diagnostic_result)
    """
    try:
        import tensorrt
        return True, tensorrt, None
    except Exception as e:
        diagnostics = TensorRTDiagnostics.diagnose_error(e)
        logger.warning(f"TensorRT import failed: {diagnostics.get_summary()}")
        return False, None, diagnostics