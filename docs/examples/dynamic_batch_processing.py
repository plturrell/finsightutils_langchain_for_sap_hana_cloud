"""
Dynamic Batch Processing Example

This example demonstrates how to use the dynamic batch processor
for efficient embedding generation with automatic memory management.
"""

import time
import random
from typing import List

import numpy as np

# Import the dynamic batch processor
from langchain_hana.gpu.batch_processor import (
    EmbeddingBatchProcessor,
    ModelMemoryProfile,
    BatchProcessingStats
)

# Import the TensorRT embeddings class
from langchain_hana.gpu.tensorrt_embeddings import (
    TensorRTEmbeddings,
    get_available_gpus
)

def main():
    """Main function demonstrating dynamic batch processing."""
    
    # Check for available GPUs
    gpus = get_available_gpus()
    if not gpus:
        print("No GPUs available. This example requires a GPU.")
        return
    
    print(f"Found {len(gpus)} GPU(s):")
    for gpu in gpus:
        print(f"  - {gpu}")
    
    # Choose the GPU with the most available memory
    gpu = max(gpus, key=lambda g: g.memory_free)
    print(f"\nUsing GPU: {gpu}")
    
    # Create a TensorRT embeddings model
    print("\nInitializing TensorRT embeddings model...")
    embeddings_model = TensorRTEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        max_batch_size=32,  # This will be dynamically adjusted
        precision="fp16"  # Use FP16 for better performance
    )
    
    # Generate sample texts
    print("\nGenerating sample texts...")
    num_texts = 1000
    texts = [
        f"This is a sample text number {i} for testing dynamic batch processing"
        for i in range(num_texts)
    ]
    
    # Add some longer texts to test variable length handling
    for i in range(50):
        # Create texts of varying lengths
        words = random.randint(50, 200)
        long_text = " ".join([
            random.choice(["dynamic", "batch", "processing", "gpu", "memory", 
                          "optimization", "embedding", "vector", "language", 
                          "model", "efficiency", "performance", "scaling"]) 
            for _ in range(words)
        ])
        texts.append(long_text)
    
    print(f"Generated {len(texts)} texts")
    
    # Create manual embedding function for demonstrating manual batching
    def embed_batch_manual(batch: List[str]) -> List[List[float]]:
        return embeddings_model._get_embeddings_gpu_batch(batch)
    
    # Create batch processor
    batch_processor = EmbeddingBatchProcessor(
        embedding_fn=embed_batch_manual,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim=384,  # Dimension of the embedding vectors
        device_id=gpu.index,
        initial_batch_size=32,  # Starting batch size
        min_batch_size=1,
        max_batch_size=128,
        safety_factor=0.8,
        oom_recovery_factor=0.5,
        dtype="float16",
        enable_caching=True
    )
    
    # Benchmark dynamic batch processing
    print("\nBenchmarking dynamic batch processing...")
    start_time = time.time()
    embeddings, stats = batch_processor.embed_documents(texts)
    end_time = time.time()
    
    # Print results
    print("\nDynamic batch processing results:")
    print(f"  Total items: {stats.total_items}")
    print(f"  Total batches: {stats.total_batches}")
    print(f"  Total time: {stats.total_time:.2f} seconds")
    print(f"  Items per second: {stats.items_per_second:.2f}")
    print(f"  Batch size adjustment: {stats.initial_batch_size} → {stats.final_batch_size}")
    print(f"  Min/max batch size used: {stats.min_batch_size}/{stats.max_batch_size}")
    print(f"  OOM events: {stats.oom_events}")
    print(f"  Batch size adjustments: {stats.batch_size_adjustments}")
    print(f"  Peak memory used: {stats.peak_memory_used_mb:.2f} MB")
    
    # Compare with standard batching
    print("\nBenchmarking standard fixed-batch processing...")
    batch_size = 32  # Fixed batch size
    start_time = time.time()
    
    std_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_embeddings = embeddings_model._get_embeddings_gpu_batch(batch)
        std_embeddings.extend(batch_embeddings)
    
    std_end_time = time.time()
    std_total_time = std_end_time - start_time
    std_items_per_second = len(texts) / std_total_time
    
    print("\nStandard fixed-batch processing results:")
    print(f"  Total items: {len(texts)}")
    print(f"  Total batches: {(len(texts) + batch_size - 1) // batch_size}")
    print(f"  Total time: {std_total_time:.2f} seconds")
    print(f"  Items per second: {std_items_per_second:.2f}")
    print(f"  Batch size: {batch_size} (fixed)")
    
    # Compare results
    print("\nPerformance comparison:")
    speedup = std_total_time / stats.total_time if stats.total_time > 0 else 0
    print(f"  Dynamic batching is {speedup:.2f}x faster than fixed batching")
    
    # Verify that embeddings are identical
    embeddings_match = np.allclose(
        np.array(embeddings[:num_texts]), 
        np.array(std_embeddings[:num_texts])
    )
    print(f"  Embeddings are identical: {embeddings_match}")
    
    print("\nDone!")

if __name__ == "__main__":
    main()