# Finesse API Documentation

## Overview

The Finesse system provides a comprehensive set of APIs for fine-tuning, visualizing, and evaluating financial language models. This document outlines the key components, their configurations, and usage patterns.

## Table of Contents

1. [Command Line Interface](#command-line-interface)
2. [Fine-tuning Module](#fine-tuning-module)
3. [Metrics Collection](#metrics-collection)
4. [Visualization](#visualization)
5. [Model Comparison](#model-comparison)
6. [Configuration Options](#configuration-options)
7. [Performance Considerations](#performance-considerations)
8. [Common Usage Patterns](#common-usage-patterns)

## Command Line Interface

The primary interface for the Finesse system is the `finesse` command line tool, which provides an elegant interface for fine-tuning and evaluating financial language models.

### Subcommands

| Command | Description | Options |
|---------|-------------|---------|
| `prepare` | Prepares training data | `--source FILE`, `--output DIRECTORY` |
| `enlighten` | Fine-tunes the model | `--model MODEL`, `--quiet` |
| `compare` | Compares models | `--queries FILE` |
| `apply` | Applies model to a query | `--model MODEL`, `--query TEXT` |

### Examples

```bash
# Prepare training data
./finesse prepare --source documents.json

# Fine-tune the model
./finesse enlighten --model FinMTEB/Fin-E5

# Compare models
./finesse compare --queries queries.json

# Apply model to a query
./finesse apply --query "What market risks are mentioned in the quarterly report?"
```

## Fine-tuning Module

The fine-tuning module (`finetune_fin_e5.py`) provides the core functionality for fine-tuning financial language models.

### Key Functions

| Function | Description | Parameters |
|----------|-------------|------------|
| `main` | Main entry point | `args`: Command line arguments |
| `validate_args` | Validates command line arguments | `args`: Command line arguments |
| `prepare_training_data` | Prepares training data | `args`: Command line arguments |
| `download_base_model` | Downloads base model | `args`: Command line arguments |
| `fine_tune_model` | Fine-tunes model | `args`, `model_path`, `train_texts`, `train_labels`, `val_texts`, `val_labels` |
| `test_model` | Tests fine-tuned model | `args`, `model_path` |

### Command Line Options

The fine-tuning module supports a wide range of command line options, organized into groups:

#### Data Parameters

| Option | Description | Default |
|--------|-------------|---------|
| `--train-file` | Training data file (JSON) | Required |
| `--val-file` | Validation data file (JSON) | None |
| `--training-format` | Training data format (`pairs`, `documents`, `simple`) | `pairs` |

#### Model Parameters

| Option | Description | Default |
|--------|-------------|---------|
| `--base-model` | Base model to fine-tune | FinMTEB/Fin-E5 |
| `--models-dir` | Models directory | ./financial_models |
| `--output-dir` | Output directory for fine-tuned models | ./fine_tuned_models |
| `--output-model-name` | Name for fine-tuned model | FinMTEB-Fin-E5-custom |
| `--force-download` | Force download of base model | False |

#### Training Parameters

| Option | Description | Default |
|--------|-------------|---------|
| `--epochs` | Number of training epochs | 3 |
| `--batch-size` | Training batch size | 16 |
| `--learning-rate` | Learning rate | 2e-5 |
| `--max-seq-length` | Maximum sequence length | 512 |
| `--evaluation-steps` | Evaluation steps | 100 |

#### Visualization Parameters

| Option | Description | Default |
|--------|-------------|---------|
| `--enable-ipc` | Enable IPC mechanisms for visualization | False |
| `--progress-file` | Path to progress file | Auto-generated in temp directory |
| `--metrics-file` | Path to metrics file | Auto-generated in temp directory |
| `--visualization-mode` | Visualization detail level (`full`, `minimal`, `none`) | full |

#### Logging Parameters

| Option | Description | Default |
|--------|-------------|---------|
| `--log-level` | Logging level | INFO (from env var or default) |
| `--log-file` | Path to log file | finetune.log (from env var or default) |

#### Performance Parameters

| Option | Description | Default |
|--------|-------------|---------|
| `--num-workers` | Number of worker processes for data loading | 4 |
| `--chunk-size` | Chunk size for processing large datasets | 1000 |
| `--use-fp16` | Use mixed precision training (fp16) | False |

### Example Usage

```bash
python finetune_fin_e5.py \
  --train-file financial_training_data.json \
  --val-file financial_validation_data.json \
  --base-model FinMTEB/Fin-E5 \
  --epochs 5 \
  --batch-size 16 \
  --learning-rate 1e-5 \
  --enable-ipc \
  --visualization-mode full \
  --log-level INFO
```

## Metrics Collection

The metrics collection module (`langchain_hana/financial/metrics.py`) provides tools for collecting, analyzing, and storing metrics during model training and evaluation.

### Classes

#### MetricsCollector

Collects and stores metrics for financial models with optimizations for large datasets.

##### Constructor Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `metrics_file` | Path to metrics file | Auto-generated temp file |
| `metrics_prefix` | Prefix for auto-generated metrics files | fin_metrics |
| `auto_save` | Whether to automatically save metrics on update | True |
| `max_items_per_metric` | Maximum number of items to keep per metric | 10000 |
| `chunk_size` | Size of chunks for processing large datasets | 1000 |
| `buffer_flush_threshold` | Number of updates before flushing buffer to disk | 50 |

##### Methods

| Method | Description | Parameters |
|--------|-------------|------------|
| `update_metric` | Update a specific metric | `metric_name`, `value`, `step` (optional) |
| `update_metrics` | Update multiple metrics at once | `metrics_dict`, `step` (optional) |
| `get_metric` | Get values for a specific metric | `metric_name`, `max_items` (optional) |
| `get_latest_metric` | Get the most recent value for a specific metric | `metric_name` |
| `save_metrics` | Save metrics to file | None |
| `load_metrics` | Load metrics from file | None |
| `clear_metrics` | Clear all metrics | None |
| `calculate_summary` | Calculate summary statistics for metrics | None |

#### FinancialModelEvaluator

Evaluates financial embedding models using various metrics.

##### Constructor Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `metrics_collector` | Metrics collector | New instance if None |
| `evaluation_data` | Evaluation data | {} |

##### Methods

| Method | Description | Parameters |
|--------|-------------|------------|
| `set_reference_model` | Set reference embeddings from a model | `model_name`, `embeddings` |
| `calculate_semantic_similarity` | Calculate cosine similarity between embeddings | `embeddings1`, `embeddings2` |
| `calculate_retrieval_metrics` | Calculate precision, recall, and F1 score | `relevant_docs`, `retrieved_docs`, `k` (optional) |
| `evaluate_model_improvement` | Calculate improvement metrics | `base_model_results`, `tuned_model_results` |
| `evaluate_model_on_queries` | Evaluate a model on queries | `model`, `queries`, `relevant_docs`, `doc_embeddings`, `k` (optional) |

### Factory Functions

| Function | Description | Parameters |
|----------|-------------|------------|
| `create_metrics_collector` | Create a metrics collector | `metrics_file` (optional), `metrics_prefix` (optional) |
| `create_model_evaluator` | Create a model evaluator | `metrics_collector` (optional), `evaluation_data` (optional) |

### Example Usage

```python
from langchain_hana.financial.metrics import create_metrics_collector, create_model_evaluator

# Create metrics collector
metrics_collector = create_metrics_collector(
    metrics_file="training_metrics.json",
    metrics_prefix="financial_model"
)

# Update metrics
metrics_collector.update_metrics({
    "loss": 0.1,
    "accuracy": 0.95,
    "batch_time": 0.05
})

# Get metrics
loss_values = metrics_collector.get_metric("loss")
latest_accuracy = metrics_collector.get_latest_metric("accuracy")

# Calculate summary
summary = metrics_collector.calculate_summary()
```

## Visualization

The visualization module (`langchain_hana/financial/visualization.py`) provides tools for visualizing training progress and metrics.

### Classes

#### TrainingVisualizer

Visualizes financial model training progress.

##### Constructor Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `progress_file` | Path to progress file | Auto-generated temp file |
| `metrics_file` | Path to metrics file | Auto-generated temp file |
| `refresh_interval` | Refresh interval in seconds | 0.5 |
| `output_file` | Path to output visualization file | None (terminal output) |
| `callback` | Callback function for progress updates | None |

##### Methods

| Method | Description | Parameters |
|--------|-------------|------------|
| `start_monitoring` | Start monitoring training progress | `progress_queue` (optional), `metrics_queue` (optional), `daemon` (optional) |
| `stop_monitoring` | Stop monitoring training progress | None |

#### MetricsVisualizer

Visualizes financial model metrics.

##### Constructor Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `metrics_file` | Path to metrics file | Auto-generated temp file |
| `output_file` | Path to output visualization file | None (terminal output) |

##### Methods

| Method | Description | Parameters |
|--------|-------------|------------|
| `visualize_metrics` | Visualize metrics from file | None |

#### ModelComparisonVisualizer

Visualizes comparisons between financial models.

##### Constructor Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `base_metrics_file` | Path to base model metrics file | None |
| `tuned_metrics_file` | Path to tuned model metrics file | None |
| `output_file` | Path to output visualization file | None (terminal output) |

##### Methods

| Method | Description | Parameters |
|--------|-------------|------------|
| `visualize_comparison` | Visualize comparison between models | `base_results` (optional), `tuned_results` (optional) |

### Factory Functions

| Function | Description | Parameters |
|----------|-------------|------------|
| `create_training_visualizer` | Create a training visualizer | `progress_file` (optional), `metrics_file` (optional), `refresh_interval` (optional) |
| `create_model_comparison_visualizer` | Create a model comparison visualizer | `base_metrics_file` (optional), `tuned_metrics_file` (optional), `output_file` (optional) |

### Example Usage

```python
from langchain_hana.financial.visualization import create_training_visualizer

# Create training visualizer
visualizer = create_training_visualizer(
    progress_file="training_progress.json",
    metrics_file="training_metrics.json",
    refresh_interval=1.0
)

# Start monitoring (in a separate thread)
visualizer.start_monitoring()

# ... Run training ...

# Stop monitoring
visualizer.stop_monitoring()
```

## Model Comparison

The model comparison module (`langchain_hana/financial/comparison.py`) provides tools for comparing and evaluating financial embedding models.

### Classes

#### ModelComparison

Compares financial embedding models across multiple dimensions.

##### Constructor Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `base_model_name` | Name of base model | Required |
| `tuned_model_name` | Name of tuned model | Required |
| `metrics_collector` | Metrics collector | New instance if None |
| `output_dir` | Output directory for comparison results | Auto-generated temp directory |

##### Methods

| Method | Description | Parameters |
|--------|-------------|------------|
| `compare_models_on_queries` | Compare models on queries | `queries`, `relevant_docs`, `base_model`, `tuned_model`, `doc_texts`, `k` (optional) |
| `analyze_semantic_understanding` | Analyze semantic understanding | `base_model`, `tuned_model`, `financial_terms`, `financial_concepts`, `financial_relationships` |
| `save_results` | Save comparison results to file | `file_path` (optional) |
| `load_results` | Load comparison results from file | `file_path` (optional) |
| `generate_comparison_report` | Generate a detailed comparison report | `file_path` (optional) |

### Factory Functions

| Function | Description | Parameters |
|----------|-------------|------------|
| `create_model_comparison` | Create a model comparison | `base_model_name`, `tuned_model_name`, `output_dir` (optional) |

### Example Usage

```python
from langchain_hana.financial.comparison import create_model_comparison
from sentence_transformers import SentenceTransformer

# Load models
base_model = SentenceTransformer("FinMTEB/Fin-E5")
tuned_model = SentenceTransformer("./fine_tuned_models/FinMTEB-Fin-E5-custom")

# Create comparison
comparison = create_model_comparison(
    base_model_name="FinMTEB/Fin-E5",
    tuned_model_name="FinMTEB-Fin-E5-custom",
    output_dir="./comparison_results"
)

# Compare models on queries
results = comparison.compare_models_on_queries(
    queries=["What market risks are mentioned?", "How has profitability changed?"],
    relevant_docs={"What market risks are mentioned?": ["doc1", "doc2"]},
    base_model=base_model,
    tuned_model=tuned_model,
    doc_texts={"doc1": "Market risks include...", "doc2": "The report mentions..."}
)

# Generate report
report_path = comparison.generate_comparison_report()
```

## Configuration Options

The Finesse system can be configured through various mechanisms:

1. **Command Line Arguments**: For fine-tuning and visualization options
2. **Environment Variables**: For logging and file paths
3. **Configuration Files**: For persistent settings

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `FINETUNE_LOG_LEVEL` | Logging level | INFO |
| `FINETUNE_LOG_FILE` | Path to log file | finetune.log |

### Configuration Files

The system uses JSON configuration files for storing training and comparison results:

- `./models/finesse_config.json`: Main configuration for the Finesse system
- `fin_e5_tuned_model_path.txt`: Path to the fine-tuned model
- `training_metadata.json`: Metadata about the training process

## Performance Considerations

The Finesse system includes several optimizations for handling large datasets and long training runs:

### Metrics Collection

- **Buffered Updates**: Metrics are buffered in memory and flushed to disk periodically to reduce I/O
- **Running Statistics**: For metrics with many values, running statistics are maintained instead of storing all values
- **Chunked Processing**: Large datasets are processed in chunks to minimize memory usage

### Visualization

- **Incremental Updates**: Visualizations are updated incrementally to minimize resources
- **Optional Output Files**: Visualizations can be directed to files instead of the terminal
- **Configurable Refresh Rate**: The refresh rate can be adjusted to balance responsiveness and resource usage

### File Handling

- **Chunked File I/O**: Large files are read and written in chunks to minimize memory usage
- **Temporary Files**: Temporary files are used for intermediate results to minimize disk usage
- **Thread Safety**: File operations are protected by locks to ensure thread safety

## Common Usage Patterns

### Fine-tuning a Model

```bash
# Prepare data
./finesse prepare --source financial_documents.json

# Fine-tune model
./finesse enlighten --model FinMTEB/Fin-E5

# Compare with base model
./finesse compare --queries test_queries.json

# Apply to specific query
./finesse apply --query "What financial risks are mentioned in the report?"
```

### Evaluating Model Improvements

```python
from langchain_hana.financial.comparison import create_model_comparison
from sentence_transformers import SentenceTransformer

# Load models
base_model = SentenceTransformer("FinMTEB/Fin-E5")
tuned_model = SentenceTransformer("./fine_tuned_models/FinMTEB-Fin-E5-custom")

# Create comparison
comparison = create_model_comparison(
    base_model_name="FinMTEB/Fin-E5",
    tuned_model_name="FinMTEB-Fin-E5-custom"
)

# Compare models
comparison.compare_models_on_queries(...)

# Generate report
comparison.generate_comparison_report()
```

### Custom Visualizations

```python
from langchain_hana.financial.metrics import create_metrics_collector
from langchain_hana.financial.visualization import create_training_visualizer

# Create metrics collector
metrics = create_metrics_collector(metrics_file="custom_metrics.json")

# Create visualizer
visualizer = create_training_visualizer(
    progress_file="custom_progress.json",
    metrics_file="custom_metrics.json",
    output_file="training_visualization.txt"
)

# Start monitoring
visualizer.start_monitoring()

# Update metrics during training
for epoch in range(epochs):
    # ... Run training ...
    metrics.update_metrics({
        "loss": current_loss,
        "accuracy": current_accuracy,
        "batch_time": batch_time
    })

# Stop monitoring
visualizer.stop_monitoring()
```

---

For more detailed information, refer to the documentation for each module and the source code.