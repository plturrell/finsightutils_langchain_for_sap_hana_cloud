"""
Query generation utilities for SAP HANA Cloud.

This module provides functions for generating optimized SQL queries
for vector operations in SAP HANA Cloud.
"""

import re
from typing import Dict, List, Any, Tuple, Optional, Union

from langchain_hana_integration.utils.distance import DistanceStrategy, get_hana_function_name, get_sql_order_direction


# Define operators for filter conditions
COMPARISON_OPERATORS = {
    "$eq": "=",
    "$ne": "<>",
    "$lt": "<",
    "$lte": "<=",
    "$gt": ">",
    "$gte": ">=",
    "$contains": "LIKE",
    "$startswith": "LIKE",
    "$endswith": "LIKE"
}

# Define logical operators
LOGICAL_OPERATORS = {
    "$and": "AND",
    "$or": "OR",
    "$not": "NOT"
}

# Regex for SQL injection prevention
SAFE_COLUMN_PATTERN = re.compile(r'^[a-zA-Z0-9_]+$')


def build_vector_search_query(
    table_name: str,
    vector_column: str,
    content_column: str,
    metadata_column: str,
    embedding_expr: str,
    distance_strategy: DistanceStrategy,
    limit: int,
    filter_clause: Optional[str] = None,
    filter_params: Optional[List[Any]] = None,
    specific_metadata_columns: Optional[List[str]] = None,
    embedding_params: Optional[List[Any]] = None,
) -> Tuple[str, List[Any]]:
    """
    Build an optimized SQL query for vector similarity search.
    
    Args:
        table_name: Name of the table
        vector_column: Name of the vector column
        content_column: Name of the content column
        metadata_column: Name of the metadata column
        embedding_expr: SQL expression for the query embedding
        distance_strategy: Distance strategy to use
        limit: Maximum number of results to return
        filter_clause: Optional WHERE clause for filtering
        filter_params: Optional parameters for the filter clause
        specific_metadata_columns: Optional list of specific metadata columns to include
        embedding_params: Optional parameters for the embedding expression
        
    Returns:
        Tuple of (SQL query, parameters)
    """
    # Validate inputs to prevent SQL injection
    _validate_sql_identifier(table_name)
    _validate_sql_identifier(vector_column)
    _validate_sql_identifier(content_column)
    _validate_sql_identifier(metadata_column)
    
    if specific_metadata_columns:
        for col in specific_metadata_columns:
            _validate_sql_identifier(col)
    
    # Get the appropriate distance function and sort direction
    distance_func = get_hana_function_name(distance_strategy)
    sort_direction = get_sql_order_direction(distance_strategy)
    
    # Build column list
    columns = [
        f'"{content_column}"',
        f'"{metadata_column}"',
        f'"{vector_column}"',
        f'{distance_func}("{vector_column}", {embedding_expr}) AS similarity_score'
    ]
    
    if specific_metadata_columns:
        for col in specific_metadata_columns:
            columns.append(f'"{col}"')
    
    # Build the query
    query = f'SELECT {", ".join(columns)} FROM "{table_name}"'
    
    # Add filter if provided
    params = []
    if embedding_params:
        params.extend(embedding_params)
    
    if filter_clause:
        query += f" WHERE {filter_clause}"
        if filter_params:
            params.extend(filter_params)
    
    # Add order by and limit
    query += f" ORDER BY similarity_score {sort_direction} LIMIT {limit}"
    
    return query, params


def build_filter_clause(
    filter_dict: Dict[str, Any],
    metadata_column: str,
    specific_metadata_columns: List[str] = None
) -> Tuple[str, List[Any]]:
    """
    Build a SQL WHERE clause from a filter dictionary.
    
    Args:
        filter_dict: Dictionary of filter criteria
        metadata_column: Name of the metadata column
        specific_metadata_columns: Optional list of specific metadata columns
        
    Returns:
        Tuple of (WHERE clause, parameters)
    """
    if not filter_dict:
        return "", []
    
    # Initialize parameters list
    params = []
    
    # Set of specific metadata columns for lookup
    specific_cols = set(specific_metadata_columns or [])
    
    def _process_condition(key, value):
        # Handle logical operators
        if key in LOGICAL_OPERATORS:
            if key == "$and" or key == "$or":
                if not isinstance(value, list) or not value:
                    raise ValueError(f"Value for {key} must be a non-empty list")
                
                subclauses = []
                for subcond in value:
                    subclause, subparams = _build_subclause(subcond)
                    subclauses.append(subclause)
                    params.extend(subparams)
                
                op = LOGICAL_OPERATORS[key]
                return f"({' ' + op + ' '.join(subclauses)})"
            
            elif key == "$not":
                if not isinstance(value, dict):
                    raise ValueError("Value for $not must be a dictionary")
                
                subclause, subparams = _build_subclause(value)
                params.extend(subparams)
                
                return f"(NOT {subclause})"
            
        # Handle field conditions
        elif isinstance(value, dict) and any(op in value for op in COMPARISON_OPERATORS):
            # Field with operators
            return _process_field_condition(key, value)
        
        else:
            # Simple equality condition
            return _process_field_condition(key, {"$eq": value})
    
    def _process_field_condition(field, condition):
        _validate_sql_identifier(field)
        
        # Determine if this is a specific metadata column or part of the JSON
        if field in specific_cols:
            field_expr = f'"{field}"'
        else:
            field_expr = f'JSON_VALUE("{metadata_column}", \'$.{field}\')'
        
        clauses = []
        
        for op, val in condition.items():
            if op not in COMPARISON_OPERATORS:
                raise ValueError(f"Unsupported operator: {op}")
            
            sql_op = COMPARISON_OPERATORS[op]
            
            # Handle special string operators
            if op == "$contains":
                clauses.append(f"{field_expr} LIKE ?")
                params.append(f"%{val}%")
            
            elif op == "$startswith":
                clauses.append(f"{field_expr} LIKE ?")
                params.append(f"{val}%")
            
            elif op == "$endswith":
                clauses.append(f"{field_expr} LIKE ?")
                params.append(f"%{val}")
            
            # Handle standard operators
            else:
                clauses.append(f"{field_expr} {sql_op} ?")
                params.append(val)
        
        # Join multiple conditions for the same field with AND
        return f"({' AND '.join(clauses)})"
    
    def _build_subclause(filter_item):
        subparams = []
        subclauses = []
        
        for k, v in filter_item.items():
            subclauses.append(_process_condition(k, v))
        
        return f"({' AND '.join(subclauses)})", subparams
    
    # Process the top-level filter
    clause = _process_condition("$and", [filter_dict])
    
    return clause, params


def build_mmr_query(
    table_name: str,
    vector_column: str,
    content_column: str,
    metadata_column: str,
    embedding_expr: str,
    distance_strategy: DistanceStrategy,
    fetch_k: int,
    filter_clause: Optional[str] = None,
    filter_params: Optional[List[Any]] = None,
    specific_metadata_columns: Optional[List[str]] = None,
    embedding_params: Optional[List[Any]] = None,
) -> Tuple[str, List[Any]]:
    """
    Build a SQL query for the first stage of MMR search.
    
    This query retrieves candidates for further MMR processing.
    
    Args:
        table_name: Name of the table
        vector_column: Name of the vector column
        content_column: Name of the content column
        metadata_column: Name of the metadata column
        embedding_expr: SQL expression for the query embedding
        distance_strategy: Distance strategy to use
        fetch_k: Number of candidates to fetch
        filter_clause: Optional WHERE clause for filtering
        filter_params: Optional parameters for the filter clause
        specific_metadata_columns: Optional list of specific metadata columns to include
        embedding_params: Optional parameters for the embedding expression
        
    Returns:
        Tuple of (SQL query, parameters)
    """
    # This is similar to build_vector_search_query but optimized for MMR
    return build_vector_search_query(
        table_name=table_name,
        vector_column=vector_column,
        content_column=content_column,
        metadata_column=metadata_column,
        embedding_expr=embedding_expr,
        distance_strategy=distance_strategy,
        limit=fetch_k,
        filter_clause=filter_clause,
        filter_params=filter_params,
        specific_metadata_columns=specific_metadata_columns,
        embedding_params=embedding_params,
    )


def _validate_sql_identifier(identifier: str) -> None:
    """
    Validate that an SQL identifier is safe to use in a query.
    
    Args:
        identifier: SQL identifier to validate
        
    Raises:
        ValueError: If the identifier is potentially unsafe
    """
    if not identifier or not SAFE_COLUMN_PATTERN.match(identifier):
        raise ValueError(f"Invalid SQL identifier: {identifier}")


def build_metadata_projection(
    metadata_column: str,
    projected_columns: List[str]
) -> str:
    """
    Generate a SQL WITH clause to project metadata columns for filtering.
    
    Args:
        metadata_column: Name of the JSON metadata column
        projected_columns: List of metadata fields to project
        
    Returns:
        SQL WITH clause for metadata projection
    """
    _validate_sql_identifier(metadata_column)
    
    projections = []
    for col in projected_columns:
        _validate_sql_identifier(col)
        projections.append(
            f"JSON_VALUE(\"{metadata_column}\", '$.{col}') AS \"{col}\""
        )
    
    if not projections:
        return ""
    
    return f"WITH projected_metadata AS (SELECT *, {', '.join(projections)})"