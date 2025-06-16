"""
SAP HANA Cloud Query Utilities for LangChain

This module provides utilities for constructing SQL queries and filters for use with
SAP HANA Cloud's vector capabilities, including:

1. Filter builders for metadata filtering
2. SQL query builders for vector similarity search
3. Utilities for handling arrays in SAP HANA

These utilities are used by the HanaVectorStore class for optimal integration with 
SAP HANA Cloud.
"""

import logging
import json
import re
from typing import Dict, List, Any, Optional, Tuple, Union, Callable

logger = logging.getLogger(__name__)

# Constants for filter operators
OPERATORS = {
    "eq": "=",              # Equal to
    "ne": "!=",             # Not equal to
    "gt": ">",              # Greater than
    "gte": ">=",            # Greater than or equal to
    "lt": "<",              # Less than
    "lte": "<=",            # Less than or equal to
    "contains": "LIKE",     # String contains (case-sensitive)
    "icontains": "LIKE",    # String contains (case-insensitive)
    "startswith": "LIKE",   # String starts with
    "endswith": "LIKE",     # String ends with
    "in": "IN",             # In a list of values
    "nin": "NOT IN",        # Not in a list of values
    "exists": "IS NOT NULL", # Field exists
    "nexists": "IS NULL",   # Field does not exist
}

# Constants for logical operators
LOGICAL_OPERATORS = {
    "and": "AND",
    "or": "OR",
    "not": "NOT",
}

def build_filter_clause(
    filter_dict: Optional[Dict[str, Any]], 
    metadata_column: str = "METADATA"
) -> Tuple[str, List[Any]]:
    """
    Build a WHERE clause from a filter dictionary.
    
    Args:
        filter_dict: Dictionary containing filter criteria
        metadata_column: Name of the column containing metadata as JSON
        
    Returns:
        Tuple of (WHERE clause string, parameters list)
        
    Examples:
        Filter: {"category": "finance", "year": {"$gt": 2020}}
        Result: "JSON_VALUE(METADATA, '$.category') = ? AND JSON_VALUE(METADATA, '$.year') > ?", ["finance", 2020]
        
        Filter: {"$or": [{"category": "finance"}, {"category": "business"}]}
        Result: "(JSON_VALUE(METADATA, '$.category') = ? OR JSON_VALUE(METADATA, '$.category') = ?)", ["finance", "business"]
    """
    if not filter_dict:
        return "", []
    
    # Create query builder
    query_builder = FilterBuilder(metadata_column)
    
    # Build WHERE clause
    where_clause, params = query_builder.build(filter_dict)
    
    return where_clause, params


class FilterBuilder:
    """
    Helper class for building SQL filter clauses from nested filter dictionaries.
    """
    
    def __init__(self, metadata_column: str = "METADATA"):
        """
        Initialize the filter builder.
        
        Args:
            metadata_column: Name of the column containing metadata as JSON
        """
        self.metadata_column = metadata_column
    
    def build(self, filter_dict: Dict[str, Any]) -> Tuple[str, List[Any]]:
        """
        Build a WHERE clause from a filter dictionary.
        
        Args:
            filter_dict: Dictionary containing filter criteria
            
        Returns:
            Tuple of (WHERE clause string, parameters list)
        """
        clauses = []
        params = []
        
        for key, value in filter_dict.items():
            # Check if this is a logical operator
            if key.startswith("$") and key[1:].lower() in LOGICAL_OPERATORS:
                operator = key[1:].lower()
                logical_clauses = []
                
                # Handle logical operators
                if isinstance(value, list):
                    for sub_filter in value:
                        sub_clause, sub_params = self.build(sub_filter)
                        if sub_clause:
                            logical_clauses.append(f"({sub_clause})")
                            params.extend(sub_params)
                
                # Join sub-clauses with the appropriate logical operator
                if logical_clauses:
                    join_str = f" {LOGICAL_OPERATORS[operator]} "
                    if operator == "not" and len(logical_clauses) == 1:
                        clauses.append(f"{LOGICAL_OPERATORS[operator]} {logical_clauses[0]}")
                    else:
                        clauses.append(f"({join_str.join(logical_clauses)})")
            else:
                # Handle field comparison
                if key.startswith("$"):
                    # Error: invalid operator at top level
                    logger.warning(f"Invalid top-level operator: {key}")
                    continue
                
                # Check if value is a comparison operator dict
                if isinstance(value, dict) and all(k.startswith("$") for k in value.keys()):
                    for op_key, op_value in value.items():
                        if op_key.startswith("$") and op_key[1:].lower() in OPERATORS:
                            op = op_key[1:].lower()
                            sub_clause, sub_params = self._build_comparison(key, op, op_value)
                            if sub_clause:
                                clauses.append(sub_clause)
                                params.extend(sub_params)
                else:
                    # Simple equality
                    sub_clause, sub_params = self._build_comparison(key, "eq", value)
                    if sub_clause:
                        clauses.append(sub_clause)
                        params.extend(sub_params)
        
        # Combine all clauses with AND
        if clauses:
            return " AND ".join(clauses), params
        
        return "", []
    
    def _build_comparison(
        self, field: str, operator: str, value: Any
    ) -> Tuple[str, List[Any]]:
        """
        Build a comparison clause for a field.
        
        Args:
            field: Field name to compare
            operator: Comparison operator
            value: Value to compare against
            
        Returns:
            Tuple of (comparison clause, parameters)
        """
        # Get the SQL operator
        sql_op = OPERATORS.get(operator.lower())
        if not sql_op:
            logger.warning(f"Unsupported operator: {operator}")
            return "", []
        
        # Get JSON path expression for the field
        json_path = self._sanitize_json_path(field)
        
        # Special handling for various operators
        if operator.lower() in ["contains", "icontains", "startswith", "endswith"]:
            # String pattern matching
            pattern = value
            if operator.lower() == "contains" or operator.lower() == "icontains":
                pattern = f"%{value}%"
            elif operator.lower() == "startswith":
                pattern = f"{value}%"
            elif operator.lower() == "endswith":
                pattern = f"%{value}"
            
            # Case sensitivity
            func = "JSON_VALUE"
            if operator.lower() == "icontains":
                func = f"LOWER(JSON_VALUE"
                pattern = pattern.lower() if isinstance(pattern, str) else pattern
                return f"{func}({self.metadata_column}, '{json_path}')) {sql_op} ?", [pattern]
            
            return f"{func}({self.metadata_column}, '{json_path}') {sql_op} ?", [pattern]
        
        elif operator.lower() in ["in", "nin"]:
            # IN and NOT IN operators
            if not isinstance(value, list):
                value = [value]
            
            placeholders = ", ".join(["?"] * len(value))
            return f"JSON_VALUE({self.metadata_column}, '{json_path}') {sql_op} ({placeholders})", value
        
        elif operator.lower() in ["exists", "nexists"]:
            # EXISTS and NOT EXISTS (IS NOT NULL and IS NULL)
            return f"JSON_VALUE({self.metadata_column}, '{json_path}') {sql_op}", []
        
        else:
            # Standard comparison operators
            return f"JSON_VALUE({self.metadata_column}, '{json_path}') {sql_op} ?", [value]
    
    def _sanitize_json_path(self, field: str) -> str:
        """
        Convert a field name to a JSON path expression, sanitizing as needed.
        
        Args:
            field: Field name to convert
            
        Returns:
            JSON path expression for the field
        """
        # Basic sanitization to prevent SQL injection in JSON path
        sanitized = re.sub(r"['\";]", "", field)
        
        # Convert dots to JSON path notation
        if "." in sanitized:
            parts = sanitized.split(".")
            sanitized = f"$.{'.'.join(parts)}"
        else:
            sanitized = f"$.{sanitized}"
        
        return sanitized


def build_array_string(vector: List[float]) -> str:
    """
    Convert a list of floats to a HANA ARRAY string representation.
    
    Args:
        vector: List of float values
        
    Returns:
        String in HANA ARRAY format: "ARRAY(1.0, 2.0, 3.0, ...)"
    """
    values = ", ".join(str(x) for x in vector)
    return f"ARRAY({values})"


def parse_array_string(array_string: str) -> List[float]:
    """
    Parse a HANA ARRAY string representation into a list of floats.
    
    Args:
        array_string: String in HANA ARRAY format: "ARRAY(1.0, 2.0, 3.0, ...)"
        
    Returns:
        List of float values
    """
    if not array_string.startswith("ARRAY(") or not array_string.endswith(")"):
        raise ValueError(f"Invalid ARRAY string format: {array_string}")
    
    # Extract values between parentheses
    content = array_string[6:-1].strip()
    
    # Handle empty array
    if not content:
        return []
    
    # Parse values
    values = [float(x.strip()) for x in content.split(",")]
    return values


def build_similarity_query(
    embedding_expr: str,
    table_name: str,
    content_column: str,
    metadata_column: str,
    vector_column: str,
    distance_function: str,
    sort_order: str,
    k: int = 4,
    filter_clause: Optional[str] = None,
    offset: int = 0,
) -> str:
    """
    Build a SQL query for similarity search.
    
    Args:
        embedding_expr: Expression for the query embedding
        table_name: Name of the table to query
        content_column: Name of the column containing document content
        metadata_column: Name of the column containing metadata
        vector_column: Name of the column containing embeddings
        distance_function: Distance function to use (e.g., "COSINE_SIMILARITY")
        sort_order: Sort order (ASC or DESC)
        k: Number of results to return
        filter_clause: Optional WHERE clause for filtering
        offset: Optional offset for pagination
        
    Returns:
        SQL query string for similarity search
    """
    # Start building the query
    query = f"""
    SELECT TOP {k} 
        ID, {content_column}, {metadata_column},
        {distance_function}({vector_column}, {embedding_expr}) AS similarity_score
    FROM {table_name}
    """
    
    # Add filter if provided
    if filter_clause:
        query += f" WHERE {filter_clause}"
    
    # Add order by clause
    query += f" ORDER BY similarity_score {sort_order}"
    
    # Add offset if provided
    if offset > 0:
        query += f" OFFSET {offset}"
    
    return query