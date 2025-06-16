# SAP HANA Reasoning Module

This module provides advanced reasoning capabilities for SAP HANA databases, including factuality measurement, benchmarking, data lineage tracking, and transparency features.

## Factuality Measurement and Benchmarking

The factuality benchmarking system allows you to measure the factual accuracy of language models on SAP HANA database content. It supports:

- Generating questions from SAP HANA schemas and data
- Evaluating answers for factual accuracy
- Benchmarking different data representation methods (relational, vector, ontology, hybrid)
- Storing benchmark results in SAP HANA tables

### Core Components

#### Factuality Measurement (`factuality.py`)

- `SchemaFactualityBenchmark`: Core benchmark class for schema-based factuality testing
- `FactualityEvaluator`: Evaluator for grading model answers
- `HanaSchemaQuestionGenerator`: Generator for creating questions about SAP HANA schemas
- `AuditStyleEvaluator`: Dual-perspective evaluator (KPMG audit and Jony Ive design)

#### Comprehensive Benchmarking (`benchmark_system.py` and `benchmark_system_part2.py`)

- `QuestionGenerator`: Generator for diverse question types (schema, instance, relationship, etc.)
- `HanaStorageManager`: Manager for CRUD operations on benchmark data in SAP HANA
- Representation Handlers:
  - `RelationalRepresentationHandler`: For SQL-based answers
  - `VectorRepresentationHandler`: For vector similarity search-based answers
  - `OntologyRepresentationHandler`: For SPARQL-based answers 
  - `HybridRepresentationHandler`: Combines multiple methods
- `BenchmarkRunner`: Orchestrates benchmarks and calculates metrics

### Usage Examples

Two example scripts demonstrate the factuality benchmarking system:

1. **Basic Factuality Measurement** (`examples/factuality_benchmark_example.py`):
   - Generates questions from SAP HANA schemas
   - Evaluates model answers for factual accuracy
   - Assesses question-answer quality from both KPMG audit and Jony Ive design perspectives

2. **Comprehensive Benchmarking** (`examples/comprehensive_benchmark_example.py`):
   - Compares different data representation methods (relational, vector, ontology, hybrid)
   - Supports diverse question types (schema, instance, relationship, aggregation, etc.)
   - Provides metrics by representation type and question type
   - Generates recommendations for improvement

### Example Command

```bash
python examples/comprehensive_benchmark_example.py \
  --openai-api-key your_openai_api_key \
  --use-hana \
  --host your_hana_host \
  --port your_hana_port \
  --user your_hana_user \
  --password your_hana_password \
  --schema SALES \
  --storage-schema BENCHMARK \
  --num-schema-questions 5 \
  --num-instance-questions 5 \
  --output benchmark_results.json
```

For a mock example without connecting to SAP HANA:

```bash
python examples/comprehensive_benchmark_example.py \
  --openai-api-key your_openai_api_key \
  --output benchmark_results.json
```

## Other Reasoning Capabilities

- **Data Lineage** (`data_lineage.py`): Track data provenance and transformation history
- **Transparency** (`transparency.py`): Explain model reasoning and decision processes
- **Metrics** (`metrics.py`): Measure and evaluate model performance
- **Validation** (`validation.py`): Validate model outputs against defined criteria
- **Feedback** (`feedback.py`): Collect and incorporate user feedback

## Integration

The reasoning module integrates with other LangChain for SAP HANA components, including:

- Vector stores for semantic search
- Embeddings for vector representation
- RDF graph capabilities for ontological reasoning