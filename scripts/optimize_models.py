#!/usr/bin/env python
"""
TensorRT model optimization script for SAP HANA Cloud LangChain integration.

This script optimizes embedding models using TensorRT for improved performance:
1. Converts PyTorch/Hugging Face models to ONNX format
2. Optimizes ONNX models with TensorRT
3. Generates calibration data for INT8 quantization (if precision is set to INT8)
4. Exports models in the desired formats
5. Creates Triton Server model repository configurations

Usage:
    python -m scripts.optimize_models \
        --model-name="all-MiniLM-L6-v2" \
        --precision=fp16 \
        --batch-sizes=1,2,4,8,16,32,64,128 \
        --calibration-cache=/app/calibration_cache \
        --export-format=tensorrt,onnx \
        --output-dir=/app/models \
        --cache-dir=/app/trt_engines
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Set, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

# Add parent directory to path so we can import from api.gpu
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # Import our GPU utilities
    from api.gpu import tensorrt_utils, calibration_datasets
    HAS_GPU_UTILS = True
except ImportError:
    HAS_GPU_UTILS = False
    print("Warning: Unable to import GPU utilities. Optimization will be limited.")

try:
    # Import for Triton Server model config
    import tritonclient.grpc as triton_grpc
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("Warning: Triton client not found. Triton model repository will not be created.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("model_optimizer")

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_PRECISION = "fp16"
DEFAULT_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Optimize embedding models with TensorRT"
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"Model name or path (default: {DEFAULT_MODEL_NAME})",
    )
    
    parser.add_argument(
        "--precision",
        type=str,
        default=DEFAULT_PRECISION,
        choices=["fp32", "fp16", "int8"],
        help=f"TensorRT precision (default: {DEFAULT_PRECISION})",
    )
    
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default=",".join(map(str, DEFAULT_BATCH_SIZES)),
        help=f"Comma-separated list of batch sizes (default: {','.join(map(str, DEFAULT_BATCH_SIZES))})",
    )
    
    parser.add_argument(
        "--calibration-cache",
        type=str,
        default="/app/calibration_cache",
        help="Path to calibration cache directory (for INT8 precision)",
    )
    
    parser.add_argument(
        "--export-format",
        type=str,
        default="tensorrt,onnx",
        help="Comma-separated list of export formats (tensorrt, onnx)",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/app/models",
        help="Output directory for optimized models",
    )
    
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/app/trt_engines",
        help="Cache directory for TensorRT engines",
    )
    
    parser.add_argument(
        "--triton-model-repository",
        type=str,
        default="/models",
        help="Triton model repository path",
    )
    
    return parser.parse_args()


def load_model(model_name: str) -> Tuple[AutoModel, AutoTokenizer]:
    """Load model and tokenizer from Hugging Face."""
    logger.info(f"Loading model: {model_name}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Set model to evaluation mode
    model.eval()
    
    return model, tokenizer


def prepare_calibration_data(
    tokenizer: AutoTokenizer,
    calibration_dir: str,
    num_samples: int = 100,
) -> np.ndarray:
    """
    Prepare calibration data for INT8 quantization.
    
    Args:
        tokenizer: The tokenizer for the model
        calibration_dir: Directory to save calibration data
        num_samples: Number of calibration samples to generate
        
    Returns:
        Numpy array with calibration data
    """
    logger.info(f"Preparing {num_samples} calibration samples")
    
    os.makedirs(calibration_dir, exist_ok=True)
    
    if HAS_GPU_UTILS:
        # Use our existing calibration dataset generator
        return calibration_datasets.generate_embedding_calibration_data(
            tokenizer=tokenizer,
            num_samples=num_samples,
            output_dir=calibration_dir,
        )
    else:
        # Basic implementation if our utilities aren't available
        sentences = [
            "This is a sample sentence for calibration.",
            "Another example sentence with different words.",
            "The quick brown fox jumps over the lazy dog.",
            # Add more diverse sentences...
        ]
        
        # Repeat sentences to get desired number of samples
        sentences = sentences * (num_samples // len(sentences) + 1)
        sentences = sentences[:num_samples]
        
        # Tokenize sentences
        encodings = tokenizer(
            sentences,
            padding=True,
            truncation=True,
            return_tensors="np",
        )
        
        # Save calibration data
        calibration_file = os.path.join(calibration_dir, "calibration_data.npz")
        np.savez(calibration_file, **encodings)
        
        return encodings


def export_to_onnx(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    output_dir: str,
    model_name: str,
    batch_size: int = 1,
) -> str:
    """
    Export PyTorch model to ONNX format.
    
    Args:
        model: The PyTorch model
        tokenizer: The tokenizer
        output_dir: Output directory
        model_name: Name of the model
        batch_size: Batch size for the model
        
    Returns:
        Path to the exported ONNX model
    """
    logger.info(f"Exporting model to ONNX (batch_size={batch_size})")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dummy input
    text = ["This is a test sentence"] * batch_size
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    # Get input names
    input_names = list(inputs.keys())
    
    # Define output path
    safe_model_name = model_name.replace("/", "_")
    onnx_path = os.path.join(
        output_dir, f"{safe_model_name}_bs{batch_size}.onnx"
    )
    
    # Export to ONNX
    with torch.no_grad():
        torch.onnx.export(
            model,
            tuple(inputs.values()),
            onnx_path,
            input_names=input_names,
            output_names=["last_hidden_state", "pooler_output"],
            dynamic_axes={
                name: {0: "batch_size", 1: "sequence_length"}
                for name in input_names
            },
            opset_version=13,
            do_constant_folding=True,
        )
    
    logger.info(f"ONNX model saved to: {onnx_path}")
    return onnx_path


def optimize_with_tensorrt(
    onnx_path: str,
    precision: str,
    output_dir: str,
    cache_dir: str,
    calibration_data: Optional[np.ndarray] = None,
    calibration_cache_dir: Optional[str] = None,
) -> str:
    """
    Optimize ONNX model with TensorRT.
    
    Args:
        onnx_path: Path to ONNX model
        precision: Precision mode (fp32, fp16, int8)
        output_dir: Output directory
        cache_dir: Cache directory for TensorRT engines
        calibration_data: Calibration data for INT8 quantization
        calibration_cache_dir: Directory for calibration cache
        
    Returns:
        Path to the TensorRT engine file
    """
    logger.info(f"Optimizing model with TensorRT (precision={precision})")
    
    if not HAS_GPU_UTILS:
        logger.error("TensorRT utilities not available. Skipping optimization.")
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    
    # Get model name and batch size from ONNX path
    onnx_filename = os.path.basename(onnx_path)
    model_bs = onnx_filename.split(".onnx")[0]
    
    # Define engine path
    engine_path = os.path.join(
        output_dir, f"{model_bs}_{precision}.engine"
    )
    
    # Define precision mode
    if precision == "fp16":
        precision_mode = tensorrt_utils.PrecisionMode.FP16
    elif precision == "int8":
        precision_mode = tensorrt_utils.PrecisionMode.INT8
    else:
        precision_mode = tensorrt_utils.PrecisionMode.FP32
    
    # Prepare INT8 calibration if needed
    int8_calibrator = None
    if precision == "int8" and calibration_data is not None:
        if calibration_cache_dir:
            os.makedirs(calibration_cache_dir, exist_ok=True)
        
        calib_cache_file = os.path.join(
            calibration_cache_dir or ".", f"{model_bs}_calibration.cache"
        )
        
        int8_calibrator = tensorrt_utils.prepare_int8_calibrator(
            calibration_data=calibration_data,
            cache_file=calib_cache_file,
        )
    
    # Build TensorRT engine
    tensorrt_utils.build_engine(
        onnx_path=onnx_path,
        engine_path=engine_path,
        precision=precision_mode,
        calibrator=int8_calibrator,
        cache_dir=cache_dir,
    )
    
    logger.info(f"TensorRT engine saved to: {engine_path}")
    return engine_path


def create_triton_model_repository(
    model_name: str,
    output_dir: str,
    precision: str,
    batch_sizes: List[int],
    triton_repo_path: str,
) -> None:
    """
    Create Triton Server model repository structure.
    
    Args:
        model_name: Name of the model
        output_dir: Directory with optimized models
        precision: Precision mode
        batch_sizes: List of batch sizes
        triton_repo_path: Path to Triton model repository
    """
    if not HAS_TRITON:
        logger.warning("Triton client not found. Skipping model repository creation.")
        return
    
    logger.info(f"Creating Triton model repository at: {triton_repo_path}")
    
    safe_model_name = model_name.replace("/", "_")
    model_dir = os.path.join(triton_repo_path, safe_model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Create model configuration
    config = {
        "name": safe_model_name,
        "backend": "tensorrt",
        "max_batch_size": max(batch_sizes),
        "input": [
            {
                "name": "input_ids",
                "data_type": "TYPE_INT64",
                "dims": [-1],
            },
            {
                "name": "attention_mask",
                "data_type": "TYPE_INT64",
                "dims": [-1],
            },
            {
                "name": "token_type_ids",
                "data_type": "TYPE_INT64",
                "dims": [-1],
            },
        ],
        "output": [
            {
                "name": "pooler_output",
                "data_type": "TYPE_FP32",
                "dims": [768],  # Adjust based on model embedding dimension
            },
        ],
        "instance_group": [
            {
                "count": 1,
                "kind": "KIND_GPU",
                "gpus": [0],
            }
        ],
        "dynamic_batching": {
            "preferred_batch_size": batch_sizes,
        },
    }
    
    # Write config.pbtxt
    config_path = os.path.join(model_dir, "config.pbtxt")
    with open(config_path, "w") as f:
        f.write(str(config))
    
    # Create version directory
    version_dir = os.path.join(model_dir, "1")
    os.makedirs(version_dir, exist_ok=True)
    
    # Copy TensorRT engine to version directory
    for batch_size in batch_sizes:
        source_path = os.path.join(
            output_dir, 
            f"{safe_model_name}_bs{batch_size}_{precision}.engine"
        )
        if os.path.exists(source_path):
            dest_path = os.path.join(version_dir, f"model_bs{batch_size}.plan")
            os.system(f"cp {source_path} {dest_path}")
    
    logger.info(f"Triton model repository created at: {model_dir}")


def main():
    """Main optimization function."""
    args = parse_args()
    
    # Parse batch sizes
    batch_sizes = [int(bs) for bs in args.batch_sizes.split(",")]
    
    # Parse export formats
    export_formats = set(args.export_format.split(","))
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    
    if args.precision == "int8":
        os.makedirs(args.calibration_cache, exist_ok=True)
    
    # Load model and tokenizer
    try:
        model, tokenizer = load_model(args.model_name)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)
    
    # Prepare calibration data for INT8 quantization
    calibration_data = None
    if args.precision == "int8":
        try:
            calibration_data = prepare_calibration_data(
                tokenizer=tokenizer,
                calibration_dir=args.calibration_cache,
            )
        except Exception as e:
            logger.error(f"Failed to prepare calibration data: {e}")
            if args.precision == "int8":
                logger.warning("Falling back to FP16 precision")
                args.precision = "fp16"
    
    # Export model to ONNX for each batch size
    onnx_paths = {}
    for batch_size in batch_sizes:
        try:
            onnx_path = export_to_onnx(
                model=model,
                tokenizer=tokenizer,
                output_dir=args.output_dir,
                model_name=args.model_name,
                batch_size=batch_size,
            )
            onnx_paths[batch_size] = onnx_path
        except Exception as e:
            logger.error(f"Failed to export model to ONNX (batch_size={batch_size}): {e}")
    
    # Optimize with TensorRT
    trt_paths = {}
    if "tensorrt" in export_formats and HAS_GPU_UTILS:
        for batch_size, onnx_path in onnx_paths.items():
            try:
                trt_path = optimize_with_tensorrt(
                    onnx_path=onnx_path,
                    precision=args.precision,
                    output_dir=args.output_dir,
                    cache_dir=args.cache_dir,
                    calibration_data=calibration_data if args.precision == "int8" else None,
                    calibration_cache_dir=args.calibration_cache if args.precision == "int8" else None,
                )
                trt_paths[batch_size] = trt_path
            except Exception as e:
                logger.error(f"Failed to optimize model with TensorRT (batch_size={batch_size}): {e}")
    
    # Create Triton model repository
    if trt_paths and HAS_TRITON:
        try:
            create_triton_model_repository(
                model_name=args.model_name,
                output_dir=args.output_dir,
                precision=args.precision,
                batch_sizes=batch_sizes,
                triton_repo_path=args.triton_model_repository,
            )
        except Exception as e:
            logger.error(f"Failed to create Triton model repository: {e}")
    
    logger.info("Optimization completed successfully")


if __name__ == "__main__":
    main()