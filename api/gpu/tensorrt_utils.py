"""
TensorRT utilities for optimizing embedding operations.
"""
import os
import logging
from typing import List, Optional, Dict, Any, Union, Callable, Iterator
import time
import torch
from pathlib import Path
import numpy as np

# Check if TensorRT is available
try:
    import tensorrt as trt
    import torch_tensorrt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    # Create placeholder for TensorRT when not available
    trt = None

# Enhanced import strategy that works in all contexts
import os
import sys
from pathlib import Path

# Add project root to sys.path if needed for absolute imports
project_root = str(Path(__file__).parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import gpu_utils with fallback strategy
try:
    # Try relative import first (when importing from within the package)
    from .gpu_utils import get_available_gpu_memory, is_gpu_available
except ImportError:
    try:
        # Fall back to absolute import (when running as a script)
        from api.gpu.gpu_utils import get_available_gpu_memory, is_gpu_available
    except ImportError:
        # Final fallback - if we're in the same directory
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "gpu_utils", os.path.join(os.path.dirname(__file__), "gpu_utils.py")
        )
        if spec and spec.loader:
            gpu_utils = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(gpu_utils)
            get_available_gpu_memory = gpu_utils.get_available_gpu_memory
            is_gpu_available = gpu_utils.is_gpu_available
        else:
            # Create dummy functions if all imports fail
            # Define logger at top of file, this reference is safe
            logging.warning("Could not import gpu_utils. Using dummy implementations.")
            def get_available_gpu_memory():
                return {"available": 0, "total": 0}
            def is_gpu_available():
                return False
                
# Import multi_gpu with fallback strategy
try:
    # Try relative import first (when importing from within the package)
    from .multi_gpu import setup_multi_gpu, distribute_workload
except ImportError:
    try:
        # Fall back to absolute import (when running as a script)
        from api.gpu.multi_gpu import setup_multi_gpu, distribute_workload
    except ImportError:
        # Final fallback - if we're in the same directory
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "multi_gpu", os.path.join(os.path.dirname(__file__), "multi_gpu.py")
        )
        if spec and spec.loader:
            multi_gpu = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(multi_gpu)
            setup_multi_gpu = multi_gpu.setup_multi_gpu
            distribute_workload = multi_gpu.distribute_workload
        else:
            # Create dummy functions if all imports fail
            logging.warning("Could not import multi_gpu. Using dummy implementations.")
            def setup_multi_gpu(enabled=True, device_ids=None, memory_fraction=0.9, force_reinit=False):
                return False
            def distribute_workload(items, process_fn, batch_size=None, use_gpu=True, *args, **kwargs):
                # Process on CPU if GPUs not available
                if batch_size is None:
                    # Use a reasonable default batch size for CPU
                    batch_size = 32
                    
                # Split into batches
                batches = [items[i:i+batch_size] for i in range(0, len(items), batch_size)]
                
                # Process batches sequentially
                results = []
                for batch in batches:
                    batch_result = process_fn(batch, *args, **kwargs)
                    if isinstance(batch_result, list):
                        results.extend(batch_result)
                    else:
                        results.append(batch_result)
                        
                return results

logger = logging.getLogger(__name__)


# Declare for type hinting
tensorrt_optimizer = None

# Only define TensorRT-dependent classes if TensorRT is available
if TENSORRT_AVAILABLE:
    class INT8Calibrator(trt.IInt8EntropyCalibrator2):
        """
    INT8 calibrator for TensorRT quantization.
    
    This class implements the TensorRT IInt8EntropyCalibrator2 interface
    to provide calibration for INT8 precision inference.
    """
    def __init__(
        self,
        calibration_dataset,
        cache_file: str,
        batch_size: int = 1,
        input_names: List[str] = ["input_ids", "attention_mask"],
    ):
        """
        Initialize the calibrator.
        
        Args:
            calibration_dataset: Dataset providing calibration data
            cache_file: File to cache calibration data
            batch_size: Batch size for calibration
            input_names: Names of model inputs
        """
        super().__init__()
        self.dataset = calibration_dataset
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.input_names = input_names
        self.current_batch = 0
        self.device_buffers = {}
        
        # Create device buffers for each input
        for batch in self.dataset.get_batch(batch_size=1):
            for name in input_names:
                if name in batch:
                    tensor = batch[name].cuda()
                    self.device_buffers[name] = cuda.mem_alloc(tensor.numel() * 4)  # float32 = 4 bytes
            break  # Just need one sample to get shapes
    
    def get_batch_size(self):
        """Return the batch size for calibration."""
        return self.batch_size
    
    def get_batch(self, names):
        """
        Get the next batch of calibration data.
        
        Args:
            names: List of input names
            
        Returns:
            List of device pointers to calibration data, or None if no more batches
        """
        batch = self.dataset.get_batch(self.batch_size)
        if not batch:
            return None
        
        # Copy data to device buffers
        bindings = []
        for name in names:
            if name in batch[0]:
                # Stack tensors in batch
                tensors = [sample[name] for sample in batch]
                tensor = torch.cat(tensors, dim=0).cuda()
                # Copy to device buffer
                cuda.memcpy_htod(self.device_buffers[name], tensor.data_ptr())
                bindings.append(int(self.device_buffers[name]))
            else:
                bindings.append(0)  # Not used
        
        self.current_batch += 1
        return bindings
    
    def read_calibration_cache(self):
        """
        Read calibration cache from file if it exists.
        
        Returns:
            Calibration cache data or None if not available
        """
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                logger.info(f"Reading INT8 calibration cache from {self.cache_file}")
                return f.read()
        return None
    
    def write_calibration_cache(self, cache):
        """
        Write calibration cache to file.
        
        Args:
            cache: Calibration cache data
        """
        with open(self.cache_file, "wb") as f:
            logger.info(f"Writing INT8 calibration cache to {self.cache_file}")
            f.write(cache)


if TENSORRT_AVAILABLE:
    class INT8CalibrationDataset:
        """
    Dataset for INT8 calibration.
    
    This class provides calibration data for INT8 quantization in TensorRT.
    It accepts a list of input tensors or a data generator function.
    """
    def __init__(
        self,
        calibration_data: Optional[List[torch.Tensor]] = None,
        data_generator: Optional[Callable[[], Iterator[torch.Tensor]]] = None,
        max_samples: int = 100,
    ):
        """
        Initialize the calibration dataset.
        
        Args:
            calibration_data: List of input tensors for calibration
            data_generator: Function that yields input tensors
            max_samples: Maximum number of samples to use for calibration
        """
        self.calibration_data = calibration_data
        self.data_generator = data_generator
        self.max_samples = max_samples
        self.current_index = 0
        
        if calibration_data is None and data_generator is None:
            raise ValueError("Either calibration_data or data_generator must be provided")
    
    def get_batch(self, batch_size: int = 1) -> List[torch.Tensor]:
        """
        Get a batch of calibration data.
        
        Args:
            batch_size: Batch size to return
            
        Returns:
            List of input tensors for calibration
        """
        if self.calibration_data is not None:
            # Return slices from the provided calibration data
            if self.current_index >= len(self.calibration_data):
                self.current_index = 0
                
            end_idx = min(self.current_index + batch_size, len(self.calibration_data))
            batch = self.calibration_data[self.current_index:end_idx]
            self.current_index = end_idx
            return batch
        else:
            # Use the data generator
            batch = []
            for _ in range(batch_size):
                try:
                    sample = next(self.data_generator())
                    batch.append(sample)
                except StopIteration:
                    break
            return batch
    
    @staticmethod
    def create_text_calibration_dataset(
        texts: List[str],
        tokenizer,
        max_length: int = 128,
        max_samples: int = 100,
    ) -> 'INT8CalibrationDataset':
        """
        Create a calibration dataset from text samples.
        
        Args:
            texts: List of text samples for calibration
            tokenizer: Tokenizer to use for text encoding
            max_length: Maximum sequence length
            max_samples: Maximum number of samples to use
            
        Returns:
            INT8CalibrationDataset for the provided texts
        """
        # Limit the number of samples
        texts = texts[:max_samples]
        
        # Tokenize the texts
        encodings = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        
        # Extract the required tensors
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]
        
        # Create batches of inputs
        calibration_data = []
        for i in range(len(texts)):
            sample = {
                "input_ids": input_ids[i:i+1],
                "attention_mask": attention_mask[i:i+1],
            }
            calibration_data.append(sample)
        
        return INT8CalibrationDataset(calibration_data=calibration_data)
    
    @staticmethod
    def get_default_calibration_texts() -> List[str]:
        """
        Get a default set of text samples for calibration.
        
        Returns:
            List of text samples suitable for calibration
        """
        # Default calibration texts covering different domains and language patterns
        return [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "TensorRT provides high-performance inference for deep learning models.",
            "The Eiffel Tower is located in Paris, France.",
            "Quantum computing leverages quantum mechanics to process information.",
            "Natural language processing enables computers to understand human language.",
            "The human genome contains approximately 3 billion base pairs.",
            "Climate change is affecting ecosystems worldwide.",
            "The speed of light in vacuum is approximately 299,792,458 meters per second.",
            "Neural networks consist of layers of interconnected nodes.",
            "The Internet of Things connects everyday devices to the internet.",
            "Cloud computing provides on-demand access to computing resources.",
            "Cybersecurity is essential for protecting digital systems from attacks.",
            "Blockchain technology enables secure and transparent transactions.",
            "Renewable energy sources include solar, wind, and hydroelectric power.",
            "Virtual reality creates immersive digital environments.",
            "The theory of relativity was developed by Albert Einstein.",
            "Data science combines statistics, mathematics, and programming.",
            "Embedded systems are specialized computing systems with dedicated functions.",
            "Autonomous vehicles use sensors and AI to navigate without human input.",
        ]

if TENSORRT_AVAILABLE:
    class TensorRTOptimizer:
        """
    Handles TensorRT optimization for embedding models.
    """
    def __init__(
        self,
        cache_dir: str = "/tmp/tensorrt_engines",
        precision: str = "fp16",
        enable_caching: bool = True,
        calibration_cache_dir: str = "/tmp/tensorrt_calibration",
    ):
        """
        Initialize TensorRT optimizer.
        
        Args:
            cache_dir: Directory to cache compiled TensorRT engines
            precision: Precision to use for TensorRT ('fp32', 'fp16', or 'int8')
            enable_caching: Whether to cache and reuse compiled engines
            calibration_cache_dir: Directory to cache INT8 calibration data
        """
        self.cache_dir = cache_dir
        self.precision = precision
        self.enable_caching = enable_caching
        self.calibration_cache_dir = calibration_cache_dir
        self.engines = {}
        
        if not TENSORRT_AVAILABLE:
            logger.warning("TensorRT not available. Using PyTorch directly.")
            return
            
        if not is_gpu_available():
            logger.warning("GPU not available. TensorRT optimization disabled.")
            return
            
        # Create cache directories if they don't exist
        if self.enable_caching:
            os.makedirs(self.cache_dir, exist_ok=True)
            os.makedirs(self.calibration_cache_dir, exist_ok=True)
            
        logger.info(f"TensorRT optimizer initialized with precision {precision}")
        
    def _get_engine_path(self, model_name: str) -> str:
        """Get path for cached TensorRT engine."""
        sanitized_name = model_name.replace('/', '_').replace('\\', '_')
        return os.path.join(self.cache_dir, f"{sanitized_name}_{self.precision}.engine")
    
    def _get_calibration_cache_path(self, model_name: str) -> str:
        """Get path for INT8 calibration cache."""
        sanitized_name = model_name.replace('/', '_').replace('\\', '_')
        return os.path.join(self.calibration_cache_dir, f"{sanitized_name}_int8_calibration.cache")
        
    def optimize_model(
        self, 
        model: torch.nn.Module,
        model_name: str,
        input_shape: List[int] = [1, 512],
        max_batch_size: int = 128,
        dynamic_shapes: bool = True,
        calibration_data: Optional[List[str]] = None,
        force_rebuild: bool = False,
    ) -> Optional[torch.nn.Module]:
        """
        Optimize PyTorch model with TensorRT.
        
        Args:
            model: PyTorch model to optimize
            model_name: Name of the model (used for caching)
            input_shape: Input shape for compilation
            max_batch_size: Maximum batch size
            dynamic_shapes: Whether to use dynamic shapes
            calibration_data: Text samples for INT8 calibration (if using INT8 precision)
            force_rebuild: Force rebuilding the engine even if a cached version exists
            
        Returns:
            Optimized TensorRT model or original model if optimization fails
        """
        if not TENSORRT_AVAILABLE or not is_gpu_available():
            return model
            
        engine_path = self._get_engine_path(model_name)
        
        # Check if engine already exists in cache and we're not forcing a rebuild
        if self.enable_caching and os.path.exists(engine_path) and not force_rebuild:
            try:
                logger.info(f"Loading cached TensorRT engine for {model_name}")
                return self._load_engine(engine_path, model)
            except Exception as e:
                logger.warning(f"Failed to load cached engine: {e}. Recompiling...")
        
        logger.info(f"Optimizing model {model_name} with TensorRT (precision: {self.precision})")
        start_time = time.time()
        
        try:
            # Move model to GPU for compilation
            model = model.cuda().eval()
            
            if self.precision == "int8":
                # INT8 quantization requires a special approach with calibration
                return self._optimize_model_int8(
                    model=model,
                    model_name=model_name,
                    input_shape=input_shape,
                    max_batch_size=max_batch_size,
                    dynamic_shapes=dynamic_shapes,
                    calibration_data=calibration_data,
                )
            else:
                # Standard FP16/FP32 optimization with Torch-TensorRT
                return self._optimize_model_fp(
                    model=model,
                    model_name=model_name,
                    input_shape=input_shape,
                    max_batch_size=max_batch_size,
                    dynamic_shapes=dynamic_shapes,
                )
                
        except Exception as e:
            logger.error(f"TensorRT optimization failed: {e}")
            logger.info("Falling back to original PyTorch model")
            return model
            
    def _optimize_model_fp(
        self,
        model: torch.nn.Module,
        model_name: str,
        input_shape: List[int],
        max_batch_size: int,
        dynamic_shapes: bool,
    ) -> torch.nn.Module:
        """
        Optimize model with FP16 or FP32 precision.
        
        This uses the standard Torch-TensorRT compilation pipeline.
        """
        engine_path = self._get_engine_path(model_name)
        
        # Configure precision
        enabled_precisions = {torch.float32}
        if self.precision == "fp16":
            enabled_precisions.add(torch.float16)
            
        # Create dynamic shapes if needed
        if dynamic_shapes:
            input_shapes = [
                (1, input_shape[1]),                   # min shape
                (max_batch_size//2, input_shape[1]),   # opt shape
                (max_batch_size, input_shape[1])       # max shape
            ]
            dynamic_batch = True
        else:
            input_shapes = [(input_shape[0], input_shape[1])]
            dynamic_batch = False
        
        start_time = time.time()
        
        # Compile with Torch-TensorRT
        optimized_model = torch_tensorrt.compile(
            model,
            inputs=[
                torch_tensorrt.Input(
                    input_shapes,
                    dtype=torch.float32,
                    dynamic_batch=dynamic_batch,
                )
            ],
            enabled_precisions=enabled_precisions,
            workspace_size=1 << 30,  # 1GB workspace
            require_full_compilation=False,
        )
        
        logger.info(f"TensorRT optimization completed in {time.time() - start_time:.2f}s")
        
        # Cache the engine if enabled
        if self.enable_caching:
            self._save_engine(optimized_model, engine_path)
            
        return optimized_model
        
    def _optimize_model_int8(
        self,
        model: torch.nn.Module,
        model_name: str,
        input_shape: List[int],
        max_batch_size: int,
        dynamic_shapes: bool,
        calibration_data: Optional[List[str]] = None,
    ) -> torch.nn.Module:
        """
        Optimize model with INT8 precision.
        
        This requires calibration data for proper quantization.
        """
        engine_path = self._get_engine_path(model_name)
        calibration_cache_path = self._get_calibration_cache_path(model_name)
        
        # Check if CUDA supports INT8
        if not torch.cuda.get_device_capability()[0] >= 7:
            logger.warning("CUDA device does not support efficient INT8. Falling back to FP16.")
            self.precision = "fp16"
            return self._optimize_model_fp(
                model=model,
                model_name=model_name,
                input_shape=input_shape,
                max_batch_size=max_batch_size,
                dynamic_shapes=dynamic_shapes,
            )
        
        try:
            from transformers import AutoTokenizer
            
            # Get tokenizer for the model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Prepare calibration dataset
            if calibration_data is None:
                calibration_data = INT8CalibrationDataset.get_default_calibration_texts()
                
            logger.info(f"Creating INT8 calibration dataset with {len(calibration_data)} samples")
            calibration_dataset = INT8CalibrationDataset.create_text_calibration_dataset(
                texts=calibration_data,
                tokenizer=tokenizer,
                max_length=input_shape[1] if len(input_shape) > 1 else 128,
                max_samples=100,  # Limit to 100 samples for speed
            )
            
            # Create INT8 calibrator
            calibrator = INT8Calibrator(
                calibration_dataset=calibration_dataset,
                cache_file=calibration_cache_path,
                batch_size=8,  # Small batch size for calibration
            )
            
            # Set up TensorRT builder and config
            logger.info("Setting up TensorRT with INT8 calibrator")
            trt_logger = trt.Logger(trt.Logger.INFO)
            builder = trt.Builder(trt_logger)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            config = builder.create_builder_config()
            config.max_workspace_size = 1 << 30  # 1GB
            
            # Enable INT8 precision
            config.set_flag(trt.BuilderFlag.INT8)
            config.int8_calibrator = calibrator
            
            # Add FP16 as fallback precision
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                
            # Export model to ONNX and create TensorRT engine
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".onnx") as tmp:
                # Create dummy input
                dummy_input = {
                    "input_ids": torch.ones((1, input_shape[1]), dtype=torch.long, device="cuda"),
                    "attention_mask": torch.ones((1, input_shape[1]), dtype=torch.long, device="cuda"),
                }
                
                # Export to ONNX
                logger.info("Exporting model to ONNX for TensorRT conversion")
                torch.onnx.export(
                    model,
                    tuple(dummy_input.values()),
                    tmp.name,
                    input_names=list(dummy_input.keys()),
                    output_names=["output"],
                    dynamic_axes={
                        "input_ids": {0: "batch_size"},
                        "attention_mask": {0: "batch_size"},
                        "output": {0: "batch_size"},
                    },
                    opset_version=12,
                )
                
                # Parse ONNX model
                parser = trt.OnnxParser(network, trt_logger)
                with open(tmp.name, "rb") as f:
                    if not parser.parse(f.read()):
                        for error in range(parser.num_errors):
                            logger.error(f"ONNX parse error: {parser.get_error(error)}")
                        raise RuntimeError("Failed to parse ONNX model")
            
            # Set up optimization profiles for dynamic shapes
            if dynamic_shapes:
                profile = builder.create_optimization_profile()
                profile.set_shape(
                    "input_ids",
                    min=(1, input_shape[1]),
                    opt=(max_batch_size//2, input_shape[1]),
                    max=(max_batch_size, input_shape[1]),
                )
                profile.set_shape(
                    "attention_mask",
                    min=(1, input_shape[1]),
                    opt=(max_batch_size//2, input_shape[1]),
                    max=(max_batch_size, input_shape[1]),
                )
                config.add_optimization_profile(profile)
            
            # Build engine
            logger.info("Building TensorRT engine with INT8 precision")
            start_time = time.time()
            serialized_engine = builder.build_serialized_network(network, config)
            
            if serialized_engine is None:
                raise RuntimeError("Failed to build TensorRT engine")
                
            # Save engine to file
            with open(engine_path, "wb") as f:
                f.write(serialized_engine)
                
            logger.info(f"INT8 TensorRT engine built and saved in {time.time() - start_time:.2f}s")
            
            # Load the optimized engine
            runtime = trt.Runtime(trt_logger)
            engine = runtime.deserialize_cuda_engine(serialized_engine)
            
            # Wrap TensorRT engine in a PyTorch module
            class TRTModule(torch.nn.Module):
                def __init__(self, engine):
                    super().__init__()
                    self.engine = engine
                    self.context = engine.create_execution_context()
                    
                def forward(self, input_ids, attention_mask):
                    batch_size = input_ids.shape[0]
                    
                    # Set input shapes for dynamic batch size
                    self.context.set_binding_shape(0, (batch_size, input_ids.shape[1]))
                    self.context.set_binding_shape(1, (batch_size, attention_mask.shape[1]))
                    
                    # Prepare output buffer
                    output_shape = self.context.get_binding_shape(2)
                    output = torch.zeros(output_shape, device="cuda")
                    
                    # Run inference
                    bindings = [
                        input_ids.data_ptr(),
                        attention_mask.data_ptr(),
                        output.data_ptr(),
                    ]
                    self.context.execute_v2(bindings)
                    
                    return output
            
            # Create TRT module with the engine
            trt_model = TRTModule(engine)
            
            # Cache the model state dict if possible
            if self.enable_caching:
                try:
                    torch.save({"engine_path": engine_path}, engine_path + ".state_dict")
                except Exception as e:
                    logger.warning(f"Failed to save TRT model state dict: {e}")
            
            logger.info(f"INT8 TensorRT optimization completed successfully")
            return trt_model
            
        except Exception as e:
            logger.error(f"INT8 optimization failed: {e}")
            logger.warning("Falling back to FP16 precision")
            self.precision = "fp16"
            return self._optimize_model_fp(
                model=model,
                model_name=model_name,
                input_shape=input_shape,
                max_batch_size=max_batch_size,
                dynamic_shapes=dynamic_shapes,
            )
            
    def _save_engine(self, optimized_model: torch.nn.Module, path: str) -> None:
        """Save TensorRT engine to disk."""
        try:
            if hasattr(optimized_model, 'engine'):
                # For INT8 engines, we've already saved the serialized engine
                logger.info(f"TensorRT engine already saved to {path}")
            else:
                # For FP16/FP32 engines with torch_tensorrt
                torch.save(optimized_model.state_dict(), path)
                logger.info(f"Saved TensorRT engine to {path}")
        except Exception as e:
            logger.warning(f"Failed to save TensorRT engine: {e}")
            
    def _load_engine(self, path: str, original_model: torch.nn.Module) -> torch.nn.Module:
        """Load TensorRT engine from disk."""
        if path in self.engines:
            return self.engines[path]
        
        # Check if this is an INT8 engine with special state dict
        state_dict_path = path + ".state_dict"
        if os.path.exists(state_dict_path):
            try:
                # This is an INT8 engine that we need to load differently
                logger.info(f"Loading INT8 TensorRT engine from {path}")
                
                # Load the engine using TensorRT APIs
                trt_logger = trt.Logger(trt.Logger.INFO)
                runtime = trt.Runtime(trt_logger)
                
                with open(path, "rb") as f:
                    engine_data = f.read()
                
                engine = runtime.deserialize_cuda_engine(engine_data)
                
                # Create wrapper module
                class TRTModule(torch.nn.Module):
                    def __init__(self, engine):
                        super().__init__()
                        self.engine = engine
                        self.context = engine.create_execution_context()
                        
                    def forward(self, input_ids, attention_mask):
                        batch_size = input_ids.shape[0]
                        
                        # Set input shapes for dynamic batch size
                        self.context.set_binding_shape(0, (batch_size, input_ids.shape[1]))
                        self.context.set_binding_shape(1, (batch_size, attention_mask.shape[1]))
                        
                        # Prepare output buffer
                        output_shape = self.context.get_binding_shape(2)
                        output = torch.zeros(output_shape, device="cuda")
                        
                        # Run inference
                        bindings = [
                            input_ids.data_ptr(),
                            attention_mask.data_ptr(),
                            output.data_ptr(),
                        ]
                        self.context.execute_v2(bindings)
                        
                        return output
                
                trt_model = TRTModule(engine)
                
                # Cache for future use
                self.engines[path] = trt_model
                return trt_model
                
            except Exception as e:
                logger.warning(f"Failed to load INT8 engine: {e}. Falling back to standard loading.")
        
        try:
            # Standard loading for FP16/FP32 engines
            state_dict = torch.load(path)
            original_model.load_state_dict(state_dict)
            
            # Cache for future use
            self.engines[path] = original_model
            return original_model
        except Exception as e:
            logger.error(f"Failed to load engine: {e}")
            raise
        
    def get_optimal_precision(self) -> str:
        """Determine optimal precision based on GPU capabilities."""
        if not TENSORRT_AVAILABLE or not is_gpu_available():
            return "fp32"
            
        # Check if Tensor Cores are available (Volta, Turing, Ampere or newer)
        cuda_capability = torch.cuda.get_device_capability(0)
        major, minor = cuda_capability
        
        # Check for INT8 support (Turing+ or Volta+ with additional capabilities)
        if major >= 7 and minor >= 5:  # Turing or newer
            return "int8"  # Use INT8 for Turing or newer for best throughput
        elif major >= 7:  # Volta or newer architecture
            return "fp16"  # Use FP16 for Volta for best performance
        else:
            return "fp32"  # Use FP32 for older GPUs
            
    def benchmark_inference(
        self, 
        model: torch.nn.Module,
        input_shape: List[int] = [1, 512],
        iterations: int = 100,
        warmup: int = 10,
        batch_sizes: List[int] = [1, 8, 32, 64, 128],
    ) -> Dict[str, Any]:
        """
        Benchmark inference performance.
        
        Args:
            model: Model to benchmark
            input_shape: Input shape for benchmark
            iterations: Number of iterations
            warmup: Number of warmup iterations
            batch_sizes: List of batch sizes to benchmark
            
        Returns:
            Dictionary with benchmark results
        """
        if not is_gpu_available():
            return {"error": "GPU not available"}
            
        # Move model to GPU and set to eval mode
        model = model.cuda().eval()
        
        results = {
            "precision": self.precision,
            "model_type": type(model).__name__,
            "input_shape": input_shape,
            "iterations": iterations,
            "batch_sizes": {},
        }
        
        # Check if we're benchmarking an INT8 model
        is_int8_model = hasattr(model, 'engine') and self.precision == "int8"
        
        # Test with different batch sizes
        for batch_size in batch_sizes:
            # Skip larger batch sizes if they exceed max batch size for INT8 models
            if is_int8_model and batch_size > model.context.get_binding_shape(0)[0]:
                continue
                
            # Create appropriate input tensors based on model type
            if is_int8_model:
                # INT8 TRTModule expects input_ids and attention_mask
                input_ids = torch.ones((batch_size, input_shape[1]), dtype=torch.long, device="cuda")
                attention_mask = torch.ones((batch_size, input_shape[1]), dtype=torch.long, device="cuda")
                input_tensors = (input_ids, attention_mask)
                
                # Warmup
                with torch.no_grad():
                    for _ in range(warmup):
                        _ = model(*input_tensors)
                
                # Benchmark
                latencies = []
                with torch.no_grad():
                    for _ in range(iterations):
                        start_time = time.time()
                        _ = model(*input_tensors)
                        torch.cuda.synchronize()  # Wait for GPU operations to complete
                        latencies.append((time.time() - start_time) * 1000)  # ms
            else:
                # Standard model expects a single input tensor
                input_tensor = torch.randn(batch_size, *input_shape[1:], device="cuda")
                
                # Warmup
                with torch.no_grad():
                    for _ in range(warmup):
                        _ = model(input_tensor)
                
                # Benchmark
                latencies = []
                with torch.no_grad():
                    for _ in range(iterations):
                        start_time = time.time()
                        _ = model(input_tensor)
                        torch.cuda.synchronize()  # Wait for GPU operations to complete
                        latencies.append((time.time() - start_time) * 1000)  # ms
            
            # Calculate statistics
            latencies = np.array(latencies)
            batch_result = {
                "mean_latency_ms": float(np.mean(latencies)),
                "median_latency_ms": float(np.median(latencies)),
                "min_latency_ms": float(np.min(latencies)),
                "max_latency_ms": float(np.max(latencies)),
                "p95_latency_ms": float(np.percentile(latencies, 95)),
                "p99_latency_ms": float(np.percentile(latencies, 99)),
                "throughput_samples_per_second": float(batch_size * 1000 / np.mean(latencies)),
                "tokens_per_second": float(batch_size * input_shape[1] * 1000 / np.mean(latencies)),
            }
            
            results["batch_sizes"][str(batch_size)] = batch_result
        
        # Add GPU memory usage info
        try:
            torch.cuda.synchronize()
            results["gpu_memory_usage_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
            results["gpu_memory_reserved_mb"] = torch.cuda.max_memory_reserved() / (1024 * 1024)
        except Exception as e:
            results["gpu_memory_error"] = str(e)
            
        return results
    
    def benchmark_precision_comparison(
        self,
        model: torch.nn.Module,
        model_name: str,
        input_shape: List[int] = [1, 512],
        max_batch_size: int = 128,
        dynamic_shapes: bool = True,
        calibration_data: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Benchmark comparison of different precision modes.
        
        Args:
            model: PyTorch model to optimize and benchmark
            model_name: Name of the model (used for caching)
            input_shape: Input shape for compilation
            max_batch_size: Maximum batch size
            dynamic_shapes: Whether to use dynamic shapes
            calibration_data: Text samples for INT8 calibration
            
        Returns:
            Dictionary with benchmark results for different precision modes
        """
        if not TENSORRT_AVAILABLE or not is_gpu_available():
            return {"error": "TensorRT or GPU not available"}
            
        results = {
            "model_name": model_name,
            "input_shape": input_shape,
            "max_batch_size": max_batch_size,
            "dynamic_shapes": dynamic_shapes,
            "cuda_capability": f"{torch.cuda.get_device_capability()[0]}.{torch.cuda.get_device_capability()[1]}",
            "device_name": torch.cuda.get_device_name(0),
            "precision_modes": {},
        }
        
        # Save original precision
        original_precision = self.precision
        
        # Test with each precision mode
        for precision in ["fp32", "fp16", "int8"]:
            # Skip INT8 if not supported by hardware
            if precision == "int8" and torch.cuda.get_device_capability()[0] < 7:
                results["precision_modes"]["int8"] = {"error": "INT8 not supported on this GPU"}
                continue
                
            try:
                # Set precision mode
                self.precision = precision
                logger.info(f"Benchmarking with {precision} precision")
                
                # Optimize model with current precision
                optimized_model = self.optimize_model(
                    model=model.cpu(),  # Start from CPU to avoid CUDA memory issues
                    model_name=model_name,
                    input_shape=input_shape,
                    max_batch_size=max_batch_size,
                    dynamic_shapes=dynamic_shapes,
                    calibration_data=calibration_data if precision == "int8" else None,
                    force_rebuild=True,  # Force rebuild to ensure fair comparison
                )
                
                # Benchmark the optimized model
                benchmark_result = self.benchmark_inference(
                    model=optimized_model,
                    input_shape=input_shape,
                    iterations=50,  # Fewer iterations for comparison
                    warmup=10,
                    batch_sizes=[1, 8, 32] if precision == "int8" else [1, 8, 32, 64, 128],
                )
                
                # Add to results
                results["precision_modes"][precision] = benchmark_result
                
                # Free up CUDA memory
                del optimized_model
                torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"Failed to benchmark {precision} precision: {e}")
                results["precision_modes"][precision] = {"error": str(e)}
        
        # Restore original precision
        self.precision = original_precision
        
        # Calculate speedup factors
        if "fp32" in results["precision_modes"] and "error" not in results["precision_modes"]["fp32"]:
            fp32_baseline = results["precision_modes"]["fp32"]["batch_sizes"]["32"]["throughput_samples_per_second"]
            
            # Calculate FP16 speedup
            if "fp16" in results["precision_modes"] and "error" not in results["precision_modes"]["fp16"]:
                fp16_throughput = results["precision_modes"]["fp16"]["batch_sizes"]["32"]["throughput_samples_per_second"]
                results["fp16_speedup_over_fp32"] = fp16_throughput / fp32_baseline
            
            # Calculate INT8 speedup
            if "int8" in results["precision_modes"] and "error" not in results["precision_modes"]["int8"]:
                int8_throughput = results["precision_modes"]["int8"]["batch_sizes"]["32"]["throughput_samples_per_second"]
                results["int8_speedup_over_fp32"] = int8_throughput / fp32_baseline
                
                # INT8 vs FP16 comparison if available
                if "fp16" in results["precision_modes"] and "error" not in results["precision_modes"]["fp16"]:
                    fp16_throughput = results["precision_modes"]["fp16"]["batch_sizes"]["32"]["throughput_samples_per_second"]
                    results["int8_speedup_over_fp16"] = int8_throughput / fp16_throughput
        
        return results
    
    # Global optimizer instance
    def _create_optimized_tensorrt_instance():
        """Create global TensorRT optimizer instance with optimal settings."""
        if not TENSORRT_AVAILABLE:
            logger.warning("TensorRT not available, cannot create optimizer instance")
            return None
            
        # Determine cache directory
        cache_dir = os.environ.get("TENSORRT_CACHE_DIR", "/tmp/tensorrt_engines")
        calibration_cache_dir = os.environ.get(
            "TENSORRT_CALIBRATION_CACHE_DIR", "/tmp/tensorrt_calibration"
        )
        
        # Check if caching is enabled
        enable_caching = os.environ.get("TENSORRT_ENABLE_CACHING", "1").lower() in ["1", "true", "yes"]
        
        # Determine precision
        precision = os.environ.get("TENSORRT_PRECISION", "auto").lower()
        
        # Create optimizer instance
        optimizer = TensorRTOptimizer(
            cache_dir=cache_dir,
            precision=precision,
            enable_caching=enable_caching,
            calibration_cache_dir=calibration_cache_dir,
        )
        
        return optimizer
    
# Define the DummyOptimizer at the module level
class DummyOptimizer:
    def __init__(self):
        self.precision = "none"
        
    def optimize_model(self, *args, **kwargs):
        logger.warning("TensorRT not available. Using original model.")
        return args[0] if args else None
        
    def benchmark_inference(self, *args, **kwargs):
        return {"error": "TensorRT not available"}
        
    def benchmark_precision_comparison(self, *args, **kwargs):
        return {"error": "TensorRT not available"}

# Add create_tensorrt_engine function for backward compatibility
def create_tensorrt_engine(model, model_name, input_shape=None, max_batch_size=128, 
                       dynamic_shapes=True, precision=None, calibration_data=None, force_rebuild=False):
    """
    Create a TensorRT engine from a PyTorch model.
    
    This is a wrapper function for the TensorRTOptimizer.optimize_model method to provide
    backward compatibility with existing code.
    
    Args:
        model: PyTorch model to optimize
        model_name: Name of the model (used for caching)
        input_shape: Input shape for compilation (default depends on model)
        max_batch_size: Maximum batch size for the engine
        dynamic_shapes: Whether to use dynamic shapes for inputs
        precision: Precision to use (fp32, fp16, int8) - if None, uses optimizer default
        calibration_data: Text samples for INT8 calibration (if using INT8)
        force_rebuild: Force rebuilding the engine even if a cached version exists
        
    Returns:
        Optimized TensorRT model or original model if optimization fails
    """
    if input_shape is None:
        input_shape = [1, 512]  # Default input shape
    
    # Use the global optimizer with custom precision if provided
    global tensorrt_optimizer
    old_precision = None
    
    if precision is not None and TENSORRT_AVAILABLE:
        old_precision = tensorrt_optimizer.precision
        tensorrt_optimizer.precision = precision
    
    try:
        result = tensorrt_optimizer.optimize_model(
            model=model,
            model_name=model_name,
            input_shape=input_shape,
            max_batch_size=max_batch_size,
            dynamic_shapes=dynamic_shapes,
            calibration_data=calibration_data,
            force_rebuild=force_rebuild
        )
    finally:
        # Restore original precision setting
        if old_precision is not None and TENSORRT_AVAILABLE:
            tensorrt_optimizer.precision = old_precision
    
    return result

# Add optimize_with_tensorrt as an alias for create_tensorrt_engine for backward compatibility
def optimize_with_tensorrt(model, model_name, input_shape=None, max_batch_size=128,
                         dynamic_shapes=True, precision=None, calibration_data=None, force_rebuild=False):
    """
    Alias for create_tensorrt_engine function.
    
    This function exists solely for backward compatibility with code that imports optimize_with_tensorrt.
    It calls create_tensorrt_engine with the same parameters.
    
    Args:
        Same as create_tensorrt_engine
        
    Returns:
        Same as create_tensorrt_engine
    """
    return create_tensorrt_engine(
        model=model,
        model_name=model_name,
        input_shape=input_shape,
        max_batch_size=max_batch_size,
        dynamic_shapes=dynamic_shapes,
        precision=precision,
        calibration_data=calibration_data,
        force_rebuild=force_rebuild
    )

# Create the global optimizer instance based on TensorRT availability
if TENSORRT_AVAILABLE:
    tensorrt_optimizer = _create_optimized_tensorrt_instance()
else:
    tensorrt_optimizer = DummyOptimizer()

# Make sure tensorrt_optimizer is defined as a module-level variable