#!/usr/bin/env python3
"""Script to extract vector data for financial visualization from SAP HANA Cloud."""

import sys
import logging
import json
import numpy as np
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    # Import the required modules
    from hdbcli import dbapi
    import json
    
    # Connect to SAP HANA
    conn = dbapi.connect(
        address="d93a8739-44a8-4845-bef3-8ec724dea2ce.hana.prod-us10.hanacloud.ondemand.com",
        port=443,
        user="DBADMIN",
        password="Initial@1",
        encrypt=True,
        sslValidateCertificate=False
    )
    
    logger.info("Successfully connected to SAP HANA Cloud")
    
    # Extract financial vector data
    cursor = conn.cursor()
    table_name = "GENAI_EARNINGS_Q125_EMBEDDED_NON_TRANSCRIPTS"
    vector_column = "VEC_VECTOR"
    
    # Check how many vectors we have
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    total_vectors = cursor.fetchone()[0]
    logger.info(f"Total vectors in {table_name}: {total_vectors}")
    
    # Get sample data (up to 100 vectors)
    max_vectors = min(100, total_vectors)
    
    # Get all columns except the vector column first
    cursor.execute(f"""
        SELECT COLUMN_NAME, DATA_TYPE_NAME
        FROM SYS.TABLE_COLUMNS 
        WHERE TABLE_NAME = '{table_name}'
        ORDER BY POSITION
    """)
    
    columns_info = cursor.fetchall()
    column_names = [col[0] for col in columns_info if col[0] != vector_column]
    column_types = {col[0]: col[1] for col in columns_info}
    
    # Build query to get data
    query_columns = ", ".join(column_names + [vector_column])
    query = f"""
        SELECT {query_columns}
        FROM {table_name}
        LIMIT {max_vectors}
    """
    
    cursor.execute(query)
    rows = cursor.fetchall()
    
    # Prepare data for visualization
    visualization_data = []
    
    # First, we'll reduce the dimensionality of the vectors for visualization
    # Since we're doing 3D visualization, we'll use a simple approach to extract 3D coordinates
    
    # Get all vectors
    full_vectors = []
    for row in rows:
        # Get vector from the last column
        vector = row[-1]
        
        if hasattr(vector, 'tolist'):
            # Convert to Python list if possible
            vector_list = vector.tolist()
            full_vectors.append(vector_list)
        else:
            # Skip if vector cannot be converted
            logger.warning(f"Skipping vector that cannot be converted to list")
            continue
    
    # Simple dimensionality reduction to 3D
    # For a real implementation, you'd use UMAP, t-SNE, or PCA
    # Here we'll use a simple approach for demonstration
    reduced_vectors = []
    vector_dimension = len(full_vectors[0]) if full_vectors else 0
    
    if vector_dimension > 0:
        logger.info(f"Vector dimension: {vector_dimension}")
        
        # Simple approach: Take first 3 dimensions, or do a basic transformation
        if vector_dimension >= 3:
            # Method 1: Take first 3 dimensions
            # reduced_vectors = [vec[:3] for vec in full_vectors]
            
            # Method 2: Use sum of groups of dimensions
            # Group dimensions and take sums to create 3D vector
            group_size = vector_dimension // 3
            
            for vec in full_vectors:
                reduced = [
                    sum(vec[:group_size]) / group_size,
                    sum(vec[group_size:2*group_size]) / group_size,
                    sum(vec[2*group_size:3*group_size]) / group_size
                ]
                reduced_vectors.append(reduced)
        else:
            # Pad with zeros if less than 3 dimensions
            reduced_vectors = [vec + [0] * (3 - len(vec)) for vec in full_vectors]
    
    # Create normalized similarities (for demonstration)
    # In a real implementation, you'd calculate these based on a query
    similarities = []
    for _ in range(len(reduced_vectors)):
        # Generate random similarity scores between 0.5 and 1.0
        similarities.append(0.5 + 0.5 * np.random.random())
    
    # Create metadata for each vector
    metadata = []
    for i, row in enumerate(rows):
        # Skip if we don't have a corresponding reduced vector
        if i >= len(reduced_vectors):
            continue
        
        # Extract content and metadata from row
        meta = {}
        for j, col_name in enumerate(column_names):
            # Convert value to the appropriate Python type
            value = row[j]
            
            # Handle JSON fields
            if column_types[col_name] in ('VARCHAR', 'NVARCHAR', 'TEXT') and value:
                try:
                    if value.startswith('{') and value.endswith('}'):
                        json_value = json.loads(value)
                        meta[col_name] = json_value
                        
                        # Extract title from JSON if possible
                        if 'source_file' in json_value:
                            meta['title'] = json_value['source_file']
                        continue
                except:
                    pass
            
            meta[col_name] = value
        
        # Add defaults if certain fields are missing
        if 'title' not in meta:
            meta['title'] = f"Financial Document {i+1}"
        
        metadata.append(meta)
    
    # Create clusters (for demonstration)
    # In a real implementation, you'd use k-means or another clustering algorithm
    clusters = []
    for _ in range(len(reduced_vectors)):
        # Assign to one of 7 clusters
        clusters.append(np.random.randint(0, 7))
    
    # Create final data structure for visualization
    visualization_data = {
        "points": reduced_vectors,
        "metadata": metadata,
        "similarities": similarities,
        "clusters": clusters,
        "total_vectors": len(reduced_vectors)
    }
    
    # Save to file
    with open('financial_visualization_data.json', 'w') as f:
        json.dump(visualization_data, f, indent=2)
    
    logger.info(f"Saved visualization data for {len(reduced_vectors)} vectors to financial_visualization_data.json")
    
    # Close connection
    cursor.close()
    conn.close()
    logger.info("Connection closed")

except ImportError as e:
    logger.error(f"Import error: {e}")
    sys.exit(1)
except Exception as e:
    logger.error(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)