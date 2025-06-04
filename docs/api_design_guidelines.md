# API Design Guidelines for LangChain SAP HANA Cloud Integration

This document outlines design principles and standards for maintaining a cohesive API across the LangChain SAP HANA Cloud integration.

## Core Principles

1. **Consistency**: Methods and parameters should follow consistent patterns
2. **Clarity**: Method names and parameters should clearly indicate their purpose
3. **Simplicity**: APIs should be intuitive and minimize cognitive load
4. **Documentation**: All public methods should have complete docstrings
5. **Error Handling**: Errors should be clear, actionable, and user-focused

## Naming Conventions

### Class Names

- Use PascalCase for class names
- Prefix SAP HANA-specific classes with `Hana` (not `HANA`)
- Examples:
  - `HanaDB` (vector store)
  - `HanaRdfGraph` (knowledge graph)
  - `HanaInternalEmbeddings` (embeddings)

### Method Names

- Use snake_case for method names
- Use action verbs to start method names
- Follow patterns:
  - `get_*` for accessors
  - `set_*` for modifiers
  - `create_*` for constructors/factories
  - `add_*` for appending/insertion
  - `delete_*` or `remove_*` for deletion
  - `validate_*` for validation

### Parameter Names

- Use snake_case for parameter names
- Use consistent parameter names across similar methods
- Always use the same name for the same concept throughout the API
- Examples:
  - Always use `connection` (not `conn` or `db_connection`)
  - Always use `embedding` (not `embeddings` or `embedding_model`)
  - Always use `table_name` (not `tableName` or `vector_table`)

## Parameter Ordering

Standard parameter ordering for methods:

1. Required parameters first
2. Optional parameters next
3. Configuration/behavior parameters last
4. Use keyword-only parameters (`*,`) for clarity when appropriate

Example pattern:
```python
def method(self, 
           required_param1, 
           required_param2, 
           optional_param1=default1, 
           optional_param2=default2, 
           *, 
           config_param1=default3, 
           config_param2=default4):
    """Method docstring."""
    pass
```

## Return Types

- Methods should have consistent return signatures
- Similar methods should return similar types
- Example patterns:
  - `similarity_search`: Returns `List[Document]`
  - `similarity_search_with_score`: Returns `List[Tuple[Document, float]]`

## Error Messages

Follow these patterns for error messages:

1. **Problem Statement**: What went wrong
2. **Context**: Relevant details about the error
3. **Solution**: What the user can do to fix it

Example template:
```python
raise ValueError(
    f"[Problem] {what_went_wrong}. "
    f"[Context] {additional_details}. "
    f"[Solution] {how_to_fix_it}"
)
```

## Documentation

### Class Docstrings

```python
class ExampleClass:
    """Short one-line summary.
    
    More detailed description that explains the purpose and usage.
    
    Args:
        param1: Description of first parameter
        param2: Description of second parameter
        
    Examples:
        ```python
        example = ExampleClass(param1="value", param2="value")
        result = example.method()
        ```
    """
```

### Method Docstrings

```python
def example_method(self, param1: str, param2: Optional[int] = None) -> List[str]:
    """Short one-line summary.
    
    More detailed description that explains what the method does.
    
    Args:
        param1: Description of first parameter
        param2: Description of second parameter, defaults to None
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: Description of when this error is raised
    """
```

## Type Annotations

- Use type annotations consistently
- Use types from the `typing` module for complex types
- Examples:
  - `List[str]` instead of `list`
  - `Dict[str, Any]` instead of `dict`
  - `Optional[int]` instead of `int = None`

## Default Values

- Use module-level constants for default values
- Name constants with `DEFAULT_` prefix in UPPER_SNAKE_CASE
- Example:
```python
DEFAULT_TABLE_NAME = "EMBEDDINGS"
DEFAULT_DISTANCE_STRATEGY = DistanceStrategy.COSINE

def __init__(self, 
             connection: dbapi.Connection,
             embedding: Embeddings,
             table_name: str = DEFAULT_TABLE_NAME,
             distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY):
    # ...
```

## Static vs Instance Methods

- Use instance methods for operations that:
  - Need access to instance state
  - Represent actions on the instance
  - Are specific to an instance

- Use static methods for operations that:
  - Don't require instance state
  - Are utility functions
  - Could logically belong to the class but not any specific instance

## Resource Management

- Always use try-finally blocks for cursor management
- Close resources explicitly
- Example:
```python
cursor = self.connection.cursor()
try:
    cursor.execute(sql_str, params)
    # Process results
finally:
    cursor.close()
```

## Public vs Private API

- Prefix private methods with underscore `_`
- Limit public API surface to methods users need
- Document which methods are part of the public API
- Don't expose implementation details

## Consistent Imports

- Import modules and classes in a consistent order:
  1. Standard library imports
  2. Third-party imports
  3. Local imports
- Use absolute imports instead of relative imports

```python
# Standard library
import json
import re
from typing import Any, Dict, List, Optional

# Third-party
from hdbcli import dbapi
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# Local
from langchain_hana.utils import DistanceStrategy
```

## API Evolution Guidelines

When updating the API:

1. **Backwards Compatibility**: Maintain compatibility when possible
2. **Deprecation Warnings**: Use deprecation warnings before removing features
3. **Version Numbering**: Follow semantic versioning (MAJOR.MINOR.PATCH)
4. **Documentation**: Document changes in release notes

```python
import warnings

def deprecated_method(self):
    """This method is deprecated and will be removed in version 1.0.0.
    
    Use new_method() instead.
    """
    warnings.warn(
        "deprecated_method is deprecated and will be removed in version 1.0.0. "
        "Use new_method instead.",
        DeprecationWarning,
        stacklevel=2
    )
    # Implementation
```

By following these guidelines, we ensure a cohesive, user-friendly API that provides a consistent experience across all components of the LangChain SAP HANA Cloud integration.