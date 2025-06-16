# Finesse Configuration Guide

## Overview

This guide covers the configuration options for the Finesse system, including fine-tuning parameters, visualization settings, and performance optimizations.

## Table of Contents

1. [Command Line Configuration](#command-line-configuration)
2. [Environment Variables](#environment-variables)
3. [Configuration Files](#configuration-files)
4. [Performance Tuning](#performance-tuning)
5. [Logging Configuration](#logging-configuration)
6. [Visualization Options](#visualization-options)
7. [Advanced Configuration](#advanced-configuration)

## Command Line Configuration

The Finesse system can be configured through command line arguments to the `finesse` script or directly to the `finetune_fin_e5.py` script.

### Finesse Script

```bash
# Basic usage
./finesse prepare --source documents.json
./finesse enlighten --model FinMTEB/Fin-E5
./finesse compare --queries queries.json
./finesse apply --query "What market risks are mentioned?"

# With additional options
./finesse prepare --source documents.json --output custom_models
./finesse enlighten --model FinMTEB/Fin-E5 --quiet
./finesse compare --queries custom_queries.json
./finesse apply --model custom_model --query "What market risks are mentioned?"
```

### Fine-tuning Script

```bash
python finetune_fin_e5.py \
  --train-file financial_training_data.json \
  --val-file financial_validation_data.json \
  --base-model FinMTEB/Fin-E5 \
  --output-dir ./custom_models \
  --output-model-name financial-model-v1 \
  --epochs 5 \
  --batch-size 16 \
  --learning-rate 1e-5 \
  --max-seq-length 512 \
  --evaluation-steps 100 \
  --enable-ipc \
  --progress-file ./progress.json \
  --metrics-file ./metrics.json \
  --visualization-mode full \
  --log-level INFO \
  --log-file finetune.log \
  --num-workers 4 \
  --chunk-size 1000 \
  --use-fp16
```

## Environment Variables

The Finesse system respects the following environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `FINETUNE_LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) | INFO |
| `FINETUNE_LOG_FILE` | Path to log file | finetune.log |

Example:

```bash
# Set environment variables
export FINETUNE_LOG_LEVEL=DEBUG
export FINETUNE_LOG_FILE=./logs/finetune_debug.log

# Run Finesse
./finesse enlighten
```

## Configuration Files

The Finesse system uses several configuration files:

### finesse_config.json

Main configuration file for the Finesse system, located at `./models/finesse_config.json`. Created and updated automatically by the Finesse script.

Example:

```json
{
  "train_file": "./models/training_data.json",
  "val_file": "./models/validation_data.json",
  "created_at": 1623456789.0,
  "source_file": "documents.json",
  "document_count": 100,
  "enlightened_model_path": "./fine_tuned_models/FinMTEB-Fin-E5-financial-1623456789",
  "enlightened_at": 1623456790.0,
  "base_model": "FinMTEB/Fin-E5",
  "enlightened_model_name": "FinMTEB-Fin-E5-financial-1623456789",
  "training_progress_file": "/tmp/finesse_progress_1623456789.json",
  "training_metrics_file": "/tmp/finesse_metrics_1623456789.json",
  "comparison_file": "model_comparison.md",
  "compared_at": 1623456791.0,
  "time_improvement": 25.5,
  "semantic_improvement": 35.0
}
```

### Training Metadata

Training metadata is stored in `training_metadata.json` within the output directory. Contains information about the training process.

Example:

```json
{
  "base_model": "FinMTEB/Fin-E5",
  "train_file": "./models/training_data.json",
  "val_file": "./models/validation_data.json",
  "epochs": 3,
  "batch_size": 8,
  "learning_rate": 2e-5,
  "completed_at": 1623456790.0,
  "output_model_path": "./fine_tuned_models/FinMTEB-Fin-E5-financial-1623456789",
  "progress_file": "/tmp/finesse_progress_1623456789.json",
  "metrics_file": "/tmp/finesse_metrics_1623456789.json",
  "visualization_mode": "full"
}
```

### Progress File

Training progress is stored in a JSON file, typically in the temporary directory. Contains information about the current state of the training process.

Example:

```json
{
  "status": "running",
  "progress": 0.5,
  "step": 50,
  "total_steps": 100,
  "stage": "Learning financial relationships and context",
  "started_at": 1623456789.0,
  "estimated_completion": 1623456889.0,
  "messages": [
    "Starting fine-tuning process",
    "Epoch 1/3 completed",
    "Batch 50: loss=0.1234"
  ]
}
```

### Metrics File

Training metrics are stored in a JSON file, typically in the temporary directory. Contains metrics collected during training.

Example:

```json
{
  "loss": [0.5, 0.4, 0.3, 0.2, 0.1],
  "similarity": [0.7, 0.75, 0.8, 0.85, 0.9],
  "batch_times": [0.1, 0.1, 0.09, 0.1, 0.09],
  "training_steps": [10, 20, 30, 40, 50],
  "custom": {
    "financial_accuracy": [0.6, 0.7, 0.8, 0.85, 0.9]
  },
  "statistics": {
    "loss": {
      "count": 100,
      "mean": 0.25,
      "sum": 25.0,
      "sum_squares": 10.0,
      "min": 0.1,
      "max": 0.5
    }
  }
}
```

## Performance Tuning

The Finesse system provides several options for performance tuning:

### Batch Size

The batch size controls how many examples are processed at once during training. A larger batch size can improve training speed but requires more memory.

```bash
python finetune_fin_e5.py --batch-size 32
```

### Number of Workers

The number of worker processes for data loading. More workers can improve data loading speed but consume more CPU resources.

```bash
python finetune_fin_e5.py --num-workers 8
```

### Chunk Size

The chunk size for processing large datasets. A larger chunk size can improve processing speed but requires more memory.

```bash
python finetune_fin_e5.py --chunk-size 2000
```

### Mixed Precision Training

Mixed precision training uses 16-bit floating point (FP16) operations where possible, which can significantly improve training speed on compatible hardware (e.g., NVIDIA GPUs with Tensor Cores).

```bash
python finetune_fin_e5.py --use-fp16
```

### Metrics Collection

The metrics collection system includes several optimizations for handling large datasets:

- **Buffer Flush Threshold**: Number of updates before flushing the buffer to disk (default: 50)
- **Maximum Items per Metric**: Maximum number of values to store for each metric (default: 10000)

These can be configured when creating a metrics collector:

```python
from langchain_hana.financial.metrics import create_metrics_collector

metrics = create_metrics_collector(
    buffer_flush_threshold=100,
    max_items_per_metric=5000
)
```

## Logging Configuration

The Finesse system uses Python's built-in logging module for logging. Logging can be configured through command line arguments or environment variables.

### Log Level

The log level controls which messages are logged. Available levels are:

- `DEBUG`: Detailed information, typically of interest only when diagnosing problems
- `INFO`: Confirmation that things are working as expected
- `WARNING`: Indication that something unexpected happened, but the program is still working
- `ERROR`: Due to a more serious problem, the program may not perform some function
- `CRITICAL`: A serious error, indicating that the program itself may be unable to continue running

```bash
python finetune_fin_e5.py --log-level DEBUG
```

Or:

```bash
export FINETUNE_LOG_LEVEL=DEBUG
```

### Log File

The log file is the path where log messages are written:

```bash
python finetune_fin_e5.py --log-file ./logs/finetune.log
```

Or:

```bash
export FINETUNE_LOG_FILE=./logs/finetune.log
```

## Visualization Options

The Finesse system provides several options for visualizing training progress and metrics.

### Visualization Mode

The visualization mode controls the level of detail in the visualization:

- `full`: Detailed visualization with progress bar, metrics, and messages
- `minimal`: Simplified visualization with just progress bar and basic information
- `none`: No visualization (useful for headless environments)

```bash
python finetune_fin_e5.py --visualization-mode minimal
```

### Progress File

The progress file is the path where training progress is stored:

```bash
python finetune_fin_e5.py --progress-file ./progress.json
```

### Metrics File

The metrics file is the path where training metrics are stored:

```bash
python finetune_fin_e5.py --metrics-file ./metrics.json
```

### Refresh Interval

The refresh interval controls how often the visualization is updated (in seconds):

```python
from langchain_hana.financial.visualization import create_training_visualizer

visualizer = create_training_visualizer(
    refresh_interval=1.0  # Update every second
)
```

### Output File

The output file is the path where the visualization is written (if not using terminal output):

```python
from langchain_hana.financial.visualization import create_training_visualizer

visualizer = create_training_visualizer(
    output_file="./visualization.txt"
)
```

## Advanced Configuration

### Custom Metrics

The metrics collection system supports custom metrics, which can be added during training:

```python
from langchain_hana.financial.metrics import create_metrics_collector

metrics = create_metrics_collector()

# Add custom metric
metrics.update_metric("financial_accuracy", 0.85)

# Add multiple custom metrics
metrics.update_metrics({
    "financial_accuracy": 0.85,
    "financial_f1": 0.9,
    "domain_precision": 0.95
})
```

### Callback Functions

The visualization system supports callback functions, which are called when progress is updated:

```python
from langchain_hana.financial.visualization import create_training_visualizer

def progress_callback(progress_data, metrics_data):
    # Custom processing of progress and metrics
    print(f"Progress: {progress_data['progress']:.2f}")
    if "loss" in metrics_data:
        print(f"Loss: {metrics_data['loss'][-1]:.4f}")

visualizer = create_training_visualizer(
    callback=progress_callback
)
```

### IPC Configuration

The Inter-Process Communication (IPC) system can be configured for real-time communication between the training process and visualization:

```python
import multiprocessing as mp
from langchain_hana.financial.visualization import create_training_visualizer

# Create queues
progress_queue = mp.Queue()
metrics_queue = mp.Queue()

# Create visualizer with queues
visualizer = create_training_visualizer()
visualizer.start_monitoring(
    progress_queue=progress_queue,
    metrics_queue=metrics_queue
)

# In training process, use the queues
progress_queue.put({"progress": 0.5, "stage": "Training"})
metrics_queue.put({"loss": 0.1, "accuracy": 0.9})
```

### Custom Comparisons

The model comparison system supports custom comparisons between models:

```python
from langchain_hana.financial.comparison import create_model_comparison

# Create comparison
comparison = create_model_comparison(
    base_model_name="FinMTEB/Fin-E5",
    tuned_model_name="FinMTEB-Fin-E5-custom"
)

# Define custom financial terms and concepts
financial_terms = [
    "market risk",
    "liquidity risk",
    "credit risk",
    "operational risk",
    "regulatory risk"
]

financial_concepts = [
    "Risk management involves identifying and mitigating financial risks.",
    "Market risk arises from movements in market prices.",
    "Liquidity risk is the risk that a company cannot meet its financial obligations.",
    "Credit risk is the risk of default on a debt.",
    "Operational risk arises from internal processes, people, and systems."
]

financial_relationships = [
    ("market risk", "volatility"),
    ("liquidity risk", "cash flow"),
    ("credit risk", "default"),
    ("operational risk", "processes"),
    ("regulatory risk", "compliance")
]

# Analyze semantic understanding
comparison.analyze_semantic_understanding(
    base_model=base_model,
    tuned_model=tuned_model,
    financial_terms=financial_terms,
    financial_concepts=financial_concepts,
    financial_relationships=financial_relationships
)
```

---

For more detailed information, refer to the API documentation and the source code.