#!/usr/bin/env python3
"""
Custom Fine-Tuning Example for FinMTEB/Fin-E5

This example demonstrates how to create custom training data from your own financial
documents and fine-tune the FinMTEB/Fin-E5 model for your specific domain.
"""

import os
import json
import logging
import argparse
from typing import List, Dict, Any, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def load_documents(file_path: str) -> List[Dict[str, Any]]:
    """Load documents from a JSON file."""
    logger.info(f"Loading documents from {file_path}")
    with open(file_path, 'r') as f:
        return json.load(f)


def generate_pairs(documents: List[Dict[str, Any]], num_pairs: int = 20) -> List[Dict[str, Any]]:
    """
    Generate training pairs from documents.
    
    This function creates training pairs by:
    1. Creating pairs from documents with similar metadata
    2. Creating pairs from documents with related content
    3. Creating pairs from representative queries and relevant documents
    """
    import random
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    
    logger.info(f"Generating {num_pairs} training pairs from {len(documents)} documents")
    
    # Extract content and metadata
    contents = [doc["content"] for doc in documents]
    metadata_list = [doc.get("metadata", {}) for doc in documents]
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    content_vectors = vectorizer.fit_transform(contents)
    
    # Calculate content similarities
    content_similarities = cosine_similarity(content_vectors)
    
    # Create pairs
    training_pairs = []
    
    # 1. Create pairs from documents with similar metadata
    for i, meta1 in enumerate(metadata_list):
        for j, meta2 in enumerate(metadata_list):
            if i != j:
                # Calculate metadata similarity
                shared_keys = set(meta1.keys()) & set(meta2.keys())
                if not shared_keys:
                    continue
                
                matching_values = sum(meta1[k] == meta2[k] for k in shared_keys)
                meta_similarity = matching_values / len(shared_keys)
                
                # Only use documents with similar metadata
                if meta_similarity > 0.5:
                    training_pairs.append({
                        "text1": contents[i],
                        "text2": contents[j],
                        "score": float(0.7 + 0.3 * meta_similarity)  # Scale to 0.7-1.0
                    })
    
    # 2. Create pairs from documents with related content
    for i in range(len(contents)):
        # Find most similar documents by content
        similarities = content_similarities[i]
        
        # Get top 2 most similar documents (excluding self)
        most_similar = np.argsort(similarities)[::-1][1:3]
        
        for j in most_similar:
            if similarities[j] > 0.3:  # Only use if similarity is reasonable
                training_pairs.append({
                    "text1": contents[i],
                    "text2": contents[j],
                    "score": float(similarities[j])
                })
    
    # 3. Create pairs from representative queries and relevant documents
    # These are manually created examples of queries and matching documents
    sample_queries = [
        "What are the latest earnings results?",
        "Explain market risks for this quarter",
        "Tell me about recent acquisitions",
        "What regulatory changes should we know about?",
        "Which investments have strong growth potential?",
    ]
    
    # Match queries to documents using TF-IDF similarity
    query_vectors = vectorizer.transform(sample_queries)
    query_doc_similarities = cosine_similarity(query_vectors, content_vectors)
    
    for q_idx, query in enumerate(sample_queries):
        # Get top 2 most relevant documents for this query
        most_relevant = np.argsort(query_doc_similarities[q_idx])[::-1][:2]
        
        for doc_idx in most_relevant:
            sim_score = query_doc_similarities[q_idx][doc_idx]
            if sim_score > 0.2:  # Only use if similarity is reasonable
                training_pairs.append({
                    "text1": query,
                    "text2": contents[doc_idx],
                    "score": float(sim_score)
                })
    
    # Deduplicate and limit pairs
    unique_pairs = []
    seen = set()
    
    for pair in training_pairs:
        pair_key = f"{pair['text1'][:50]}|{pair['text2'][:50]}"
        if pair_key not in seen:
            seen.add(pair_key)
            unique_pairs.append(pair)
    
    # Limit to requested number, prioritizing higher scores
    unique_pairs.sort(key=lambda x: x["score"], reverse=True)
    return unique_pairs[:num_pairs]


def split_data(pairs: List[Dict[str, Any]], val_ratio: float = 0.2) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split data into training and validation sets."""
    import random
    
    # Shuffle pairs
    random.shuffle(pairs)
    
    # Calculate split point
    split_idx = int(len(pairs) * (1 - val_ratio))
    
    # Split data
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]
    
    logger.info(f"Split data into {len(train_pairs)} training pairs and {len(val_pairs)} validation pairs")
    
    return train_pairs, val_pairs


def main(args):
    """Main function."""
    # Load documents
    documents = load_documents(args.documents_file)
    
    # Generate training pairs
    pairs = generate_pairs(documents, args.num_pairs)
    
    # Split into training and validation sets
    train_pairs, val_pairs = split_data(pairs, args.val_ratio)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.train_output), exist_ok=True)
    
    # Save training and validation data
    with open(args.train_output, 'w') as f:
        json.dump(train_pairs, f, indent=2)
    
    with open(args.val_output, 'w') as f:
        json.dump(val_pairs, f, indent=2)
    
    logger.info(f"Training data saved to {args.train_output}")
    logger.info(f"Validation data saved to {args.val_output}")
    
    # Print next steps
    print("\nNext steps to fine-tune the model:")
    print(f"1. Run: python finetune_fin_e5.py --train-file {args.train_output} --val-file {args.val_output} --training-format pairs")
    print("2. Or use the convenience script: ./run_finetune_fin_e5.sh")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate custom training data for fine-tuning")
    
    parser.add_argument("--documents-file", required=True, help="JSON file containing financial documents")
    parser.add_argument("--train-output", default="custom_training_data.json", help="Output file for training data")
    parser.add_argument("--val-output", default="custom_validation_data.json", help="Output file for validation data")
    parser.add_argument("--num-pairs", type=int, default=30, help="Number of training pairs to generate")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Ratio of validation data")
    
    args = parser.parse_args()
    main(args)