# SAP HANA Reasoning Benchmark System

![Benchmark System Dashboard](images/benchmark_dashboard.png)

## Overview

The SAP HANA Reasoning Benchmark System is a comprehensive framework for evaluating and comparing different data representation methods for answering questions about SAP HANA database schemas. The system enables you to measure the factual accuracy of different approaches - relational, vector, ontology, and hybrid - providing insights into which methods work best for different types of questions.

## Features

- **Simple one-line setup** - Get started with minimal configuration
- **Beautiful visualizations** - Understand your results at a glance
- **Comprehensive metrics** - Detailed breakdown by question type and difficulty
- **Provenance tracking** - Complete lineage for all questions and answers
- **Standardized evaluation** - Consistent criteria for fair comparisons

## Quick Start

```python
from langchain_openai import ChatOpenAI
from langchain_hana.reasoning.simplified_interface import SimpleBenchmark, BenchmarkConfig

# Initialize the language model
llm = ChatOpenAI(temperature=0)

# Create a simple configuration
config = BenchmarkConfig(
    name="My First Benchmark",
    schema="SALES"
)

# Create and run the benchmark with one line
benchmark = SimpleBenchmark(llm=llm, config=config)
results = benchmark.run()

# Visualize the results
benchmark.visualize_results()
```

## Understanding the Interface

The benchmark system is designed with simplicity in mind. The main components are:

### BenchmarkConfig

A simple configuration class with sensible defaults:

![Configuration Options](images/benchmark_config.png)

| Parameter | Description | Default |
|-----------|-------------|---------|
| name | Benchmark name | "Benchmark-{timestamp}" |
| description | Optional description | None |
| host | SAP HANA host | None (optional) |
| port | SAP HANA port | None (optional) |
| user | SAP HANA user | None (optional) |
| password | SAP HANA password | None (optional) |
| schema | Schema to benchmark | "SALES" |
| question_counts | Number of questions by type | Balanced defaults |
| representation_types | Methods to benchmark | ["relational", "vector", "hybrid"] |
| storage_schema | Schema for storing results | "BENCHMARK" |
| save_results | Whether to save results | True |
| output_file | File to save results | "benchmark_results_{timestamp}.json" |

### SimpleBenchmark

The main interface class that handles everything:

```python
benchmark = SimpleBenchmark(
    llm=my_language_model,  # Required
    connection=my_connection,  # Optional
    config=my_config  # Optional, uses defaults if not provided
)
```

## Visual Results

The benchmark system automatically generates beautiful visualizations:

### Overall Accuracy

![Accuracy Chart](images/accuracy_chart.png)

Compare the overall accuracy of different representation methods at a glance.

### Performance by Question Type

![Question Type Performance](images/question_type_chart.png)

See which methods excel at different types of questions, from simple schema questions to complex inference.

### Performance by Difficulty

![Difficulty Performance](images/difficulty_chart.png)

Understand how each method performs as questions increase in complexity.

### Metrics Comparison

![Metrics Spider Chart](images/metrics_spider.png)

Compare multiple metrics across representation methods with an elegant spider chart.

## Understanding Question Types

The benchmark system evaluates six types of questions:

1. **Schema Questions** - Basic questions about database structure
   ```
   What is the primary key of the CUSTOMERS table?
   ```

2. **Instance Questions** - Questions about specific data values
   ```
   What is the email address of customer with ID 1?
   ```

3. **Relationship Questions** - Questions about relationships between entities
   ```
   Which orders were placed by customer John Doe?
   ```

4. **Aggregation Questions** - Questions requiring calculations across data
   ```
   What is the total revenue from orders in January 2023?
   ```

5. **Inference Questions** - Questions requiring logical reasoning
   ```
   Based on ordering patterns, which products are frequently purchased together?
   ```

6. **Temporal Questions** - Questions about time-based patterns
   ```
   What is the month-over-month growth rate in customer acquisition?
   ```

## Difficulty Levels

The system automatically assesses question difficulty on a 5-level scale:

![Difficulty Scale](images/difficulty_scale.png)

1. **Beginner (Level 1)** - Simple lookups, single entity, direct fact retrieval
2. **Basic (Level 2)** - Simple aggregations, basic relationships
3. **Intermediate (Level 3)** - Multiple entities, basic analytics, simple joins
4. **Advanced (Level 4)** - Complex joins, advanced analytics, temporal reasoning
5. **Expert (Level 5)** - Complex inferences, specialized knowledge, multiple steps

## Seamless Integration

The benchmark system is designed to integrate seamlessly with your existing workflows:

### With Vector Databases

```python
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Create your vector store
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_texts(my_texts, embeddings)

# Run benchmark with your vector store
benchmark = SimpleBenchmark(llm=llm, config=config)
results = benchmark.run(vector_store=vector_store)
```

### With Existing Questions

```python
# Use your existing benchmark questions
benchmark = SimpleBenchmark(llm=llm, config=config)
results = benchmark.run(existing_questions=my_questions)
```

### With Existing Database Connection

```python
from hdbcli import dbapi

# Use your existing connection
connection = dbapi.connect(
    address="your-hana-host",
    port=30015,
    user="your-user",
    password="your-password"
)

benchmark = SimpleBenchmark(llm=llm, connection=connection, config=config)
results = benchmark.run()
```

## Provenance Tracking

The system maintains complete provenance information for all questions:

![Provenance Tracking](images/provenance_tracking.png)

Each question includes:
- Source (model, human, template)
- Generation parameters
- Input data sources
- Complete revision history

## Best Practices

For optimal results:

1. **Balance question types** - Include a mix of question types for comprehensive evaluation
2. **Test all representation methods** - Compare at least relational, vector, and hybrid approaches
3. **Visualize results** - Use the built-in visualization tools to understand patterns
4. **Store results** - Save benchmark results for historical comparison
5. **Review recommendations** - The system provides actionable recommendations based on results

## Next Steps

To learn more:

- Explore the [API Reference](api/benchmark_reference.md)
- Try the [Advanced Configuration Guide](configuration/advanced_benchmarking.md)
- View [Example Notebooks](examples/benchmark_examples.md)

---

<div align="center">
<img src="images/sap_logo.png" width="120" alt="SAP Logo">
<p>© 2023 SAP SE. All rights reserved.</p>
</div>