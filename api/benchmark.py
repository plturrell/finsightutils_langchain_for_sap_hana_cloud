"""Benchmarking tools for CPU vs GPU performance."""

import gc
import json
import logging
import os
import random
import string
import time
from typing import Dict, List, Optional, Union, Any, Callable, Tuple

import numpy as np

import gpu_utils
from multi_gpu import get_gpu_manager
from dynamic_batching import get_batch_sizer
from memory_optimization import get_memory_optimizer
from embeddings import GPUAcceleratedEmbeddings
from embeddings_multi_gpu import MultiGPUEmbeddings

logger = logging.getLogger(__name__)


class Benchmark:
    """
    Benchmarking tool for CPU vs GPU performance.
    
    This class provides utilities to benchmark and compare the performance
    of CPU and GPU implementations for various operations, including:
    
    1. Embedding generation
    2. Similarity search
    3. Maximal marginal relevance
    """
    
    def __init__(
        self,
        results_dir: str = "benchmark_results",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize the benchmark tool.
        
        Args:
            results_dir: Directory to store benchmark results.
            embedding_model: Name of the embedding model to use.
        """
        self.results_dir = results_dir
        self.embedding_model = embedding_model
        
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize components
        self.gpu_manager = get_gpu_manager()
        self.memory_optimizer = get_memory_optimizer()
        
        # Initialize embeddings models
        self.cpu_embeddings = None
        self.gpu_embeddings = None
        self.multi_gpu_embeddings = None
        
        # Performance metrics
        self.metrics = {}
    
    def _initialize_embeddings(self) -> None:
        """Initialize embedding models for benchmarking."""
        try:
            # CPU embeddings
            logger.info(f"Initializing CPU embeddings model: {self.embedding_model}")
            from sentence_transformers import SentenceTransformer
            self.cpu_embeddings = SentenceTransformer(self.embedding_model, device="cpu")
            
            # Single GPU embeddings
            if gpu_utils.is_torch_available():
                logger.info(f"Initializing GPU embeddings model: {self.embedding_model}")
                self.gpu_embeddings = GPUAcceleratedEmbeddings(
                    model_name=self.embedding_model,
                    device="cuda" if gpu_utils.is_torch_available() else "cpu",
                )
                
                # Multi-GPU embeddings
                if len(self.gpu_manager.get_available_devices()) > 1:
                    logger.info(f"Initializing Multi-GPU embeddings model: {self.embedding_model}")
                    self.multi_gpu_embeddings = MultiGPUEmbeddings(
                        model_name=self.embedding_model,
                    )
        
        except Exception as e:
            logger.error(f"Error initializing embeddings models: {str(e)}")
    
    def _generate_random_text(self, length: int) -> str:
        """
        Generate random text for benchmarking.
        
        Args:
            length: Length of the text to generate.
            
        Returns:
            Random text string.
        """
        words = []
        word_length = random.randint(3, 10)
        
        while len(" ".join(words)) < length:
            word = "".join(
                random.choice(string.ascii_lowercase)
                for _ in range(word_length)
            )
            words.append(word)
            word_length = random.randint(3, 10)
        
        return " ".join(words)
    
    def _generate_test_data(
        self,
        num_samples: int,
        text_length: int,
    ) -> List[str]:
        """
        Generate test data for benchmarking.
        
        Args:
            num_samples: Number of samples to generate.
            text_length: Average length of each text sample.
            
        Returns:
            List of text samples.
        """
        return [
            self._generate_random_text(
                length=random.randint(
                    int(text_length * 0.8),
                    int(text_length * 1.2),
                )
            )
            for _ in range(num_samples)
        ]
    
    def benchmark_embedding(
        self,
        num_samples: int = 1000,
        text_length: int = 100,
        batch_sizes: List[int] = [1, 8, 32, 128, 512],
        runs_per_batch: int = 3,
    ) -> Dict:
        """
        Benchmark embedding generation performance.
        
        Args:
            num_samples: Number of samples to use for benchmarking.
            text_length: Average length of each text sample.
            batch_sizes: List of batch sizes to benchmark.
            runs_per_batch: Number of runs for each batch size.
            
        Returns:
            Dictionary of benchmark results.
        """
        # Initialize embeddings if not already initialized
        if self.cpu_embeddings is None:
            self._initialize_embeddings()
        
        # Generate test data
        logger.info(f"Generating {num_samples} test samples for embedding benchmark")
        texts = self._generate_test_data(num_samples, text_length)
        
        # Prepare results
        results = {
            "cpu": {},
            "gpu": {},
            "multi_gpu": {},
            "parameters": {
                "num_samples": num_samples,
                "text_length": text_length,
                "batch_sizes": batch_sizes,
                "runs_per_batch": runs_per_batch,
                "embedding_model": self.embedding_model,
            },
        }
        
        # Benchmark CPU
        if self.cpu_embeddings is not None:
            logger.info("Benchmarking CPU embeddings")
            cpu_results = self._benchmark_embedding_model(
                model="cpu",
                texts=texts,
                batch_sizes=batch_sizes,
                runs_per_batch=runs_per_batch,
            )
            results["cpu"] = cpu_results
        
        # Benchmark GPU
        if self.gpu_embeddings is not None:
            logger.info("Benchmarking GPU embeddings")
            gpu_results = self._benchmark_embedding_model(
                model="gpu",
                texts=texts,
                batch_sizes=batch_sizes,
                runs_per_batch=runs_per_batch,
            )
            results["gpu"] = gpu_results
        
        # Benchmark Multi-GPU
        if self.multi_gpu_embeddings is not None:
            logger.info("Benchmarking Multi-GPU embeddings")
            multi_gpu_results = self._benchmark_embedding_model(
                model="multi_gpu",
                texts=texts,
                batch_sizes=batch_sizes,
                runs_per_batch=runs_per_batch,
            )
            results["multi_gpu"] = multi_gpu_results
        
        # Save results
        self._save_benchmark_results("embedding", results)
        
        return results
    
    def _benchmark_embedding_model(
        self,
        model: str,
        texts: List[str],
        batch_sizes: List[int],
        runs_per_batch: int,
    ) -> Dict:
        """
        Benchmark a specific embedding model.
        
        Args:
            model: Model to benchmark ('cpu', 'gpu', or 'multi_gpu').
            texts: List of text samples.
            batch_sizes: List of batch sizes to benchmark.
            runs_per_batch: Number of runs for each batch size.
            
        Returns:
            Dictionary of benchmark results.
        """
        # Get the model
        if model == "cpu":
            embedding_model = self.cpu_embeddings
        elif model == "gpu":
            embedding_model = self.gpu_embeddings
        elif model == "multi_gpu":
            embedding_model = self.multi_gpu_embeddings
        else:
            raise ValueError(f"Unknown model: {model}")
        
        # Initialize results
        results = {}
        
        # Benchmark each batch size
        for batch_size in batch_sizes:
            # Skip if batch size is larger than the number of samples
            if batch_size > len(texts):
                continue
            
            batch_results = []
            
            for run in range(runs_per_batch):
                # Clear memory before each run
                gc.collect()
                if model in ["gpu", "multi_gpu"]:
                    self.memory_optimizer.clear_cache()
                
                # Select random texts for this run
                batch_texts = random.sample(texts, batch_size)
                
                # Benchmark embedding generation
                start_time = time.time()
                
                if model in ["cpu", "gpu"]:
                    # Process all texts at once
                    _ = embedding_model.embed_documents(batch_texts)
                elif model == "multi_gpu":
                    # Process texts using multi-GPU
                    _ = embedding_model.embed_documents(batch_texts)
                
                end_time = time.time()
                
                # Record results
                duration = end_time - start_time
                items_per_second = batch_size / duration
                
                batch_results.append({
                    "duration": duration,
                    "items_per_second": items_per_second,
                    "batch_size": batch_size,
                })
                
                logger.debug(
                    f"{model} embedding: batch_size={batch_size}, "
                    f"run={run + 1}/{runs_per_batch}, "
                    f"duration={duration:.4f}s, "
                    f"items_per_second={items_per_second:.2f}"
                )
            
            # Calculate statistics
            durations = [result["duration"] for result in batch_results]
            items_per_second = [result["items_per_second"] for result in batch_results]
            
            results[str(batch_size)] = {
                "min_duration": min(durations),
                "max_duration": max(durations),
                "avg_duration": sum(durations) / len(durations),
                "min_items_per_second": min(items_per_second),
                "max_items_per_second": max(items_per_second),
                "avg_items_per_second": sum(items_per_second) / len(items_per_second),
                "runs": batch_results,
            }
        
        return results
    
    def benchmark_mmr(
        self,
        num_vectors: int = 1000,
        vector_dim: int = 768,
        k_values: List[int] = [4, 8, 16, 32],
        fetch_k_values: List[int] = [20, 50, 100, 200],
        runs_per_config: int = 3,
    ) -> Dict:
        """
        Benchmark Maximal Marginal Relevance (MMR) performance.
        
        Args:
            num_vectors: Number of vectors to use for benchmarking.
            vector_dim: Dimension of each vector.
            k_values: List of k values to benchmark.
            fetch_k_values: List of fetch_k values to benchmark.
            runs_per_config: Number of runs for each configuration.
            
        Returns:
            Dictionary of benchmark results.
        """
        # Generate random vectors
        logger.info(f"Generating {num_vectors} random vectors for MMR benchmark")
        query_vector = np.random.randn(vector_dim).astype(np.float32)
        vectors = np.random.randn(num_vectors, vector_dim).astype(np.float32)
        
        # Normalize vectors
        query_vector = query_vector / np.linalg.norm(query_vector)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        
        # Prepare results
        results = {
            "cpu": {},
            "gpu": {},
            "parameters": {
                "num_vectors": num_vectors,
                "vector_dim": vector_dim,
                "k_values": k_values,
                "fetch_k_values": fetch_k_values,
                "runs_per_config": runs_per_config,
            },
        }
        
        # Benchmark CPU MMR
        logger.info("Benchmarking CPU MMR")
        cpu_results = {}
        
        for k in k_values:
            cpu_results[str(k)] = {}
            
            for fetch_k in fetch_k_values:
                # Skip if fetch_k < k
                if fetch_k < k:
                    continue
                
                # Skip if fetch_k > num_vectors
                if fetch_k > num_vectors:
                    continue
                
                run_results = []
                
                for run in range(runs_per_config):
                    # Clear memory before each run
                    gc.collect()
                    
                    # Benchmark CPU MMR
                    start_time = time.time()
                    
                    from langchain_core.vectorstores.utils import maximal_marginal_relevance
                    _ = maximal_marginal_relevance(
                        query_vector,
                        vectors[:fetch_k],
                        k=k,
                        lambda_mult=0.5,
                    )
                    
                    end_time = time.time()
                    
                    # Record results
                    duration = end_time - start_time
                    
                    run_results.append({
                        "duration": duration,
                        "k": k,
                        "fetch_k": fetch_k,
                    })
                    
                    logger.debug(
                        f"CPU MMR: k={k}, fetch_k={fetch_k}, "
                        f"run={run + 1}/{runs_per_config}, "
                        f"duration={duration:.4f}s"
                    )
                
                # Calculate statistics
                durations = [result["duration"] for result in run_results]
                
                cpu_results[str(k)][str(fetch_k)] = {
                    "min_duration": min(durations),
                    "max_duration": max(durations),
                    "avg_duration": sum(durations) / len(durations),
                    "runs": run_results,
                }
        
        results["cpu"] = cpu_results
        
        # Benchmark GPU MMR
        if gpu_utils.is_cupy_available():
            logger.info("Benchmarking GPU MMR")
            gpu_results = {}
            
            for k in k_values:
                gpu_results[str(k)] = {}
                
                for fetch_k in fetch_k_values:
                    # Skip if fetch_k < k
                    if fetch_k < k:
                        continue
                    
                    # Skip if fetch_k > num_vectors
                    if fetch_k > num_vectors:
                        continue
                    
                    run_results = []
                    
                    for run in range(runs_per_config):
                        # Clear memory before each run
                        gc.collect()
                        self.memory_optimizer.clear_cache()
                        
                        # Benchmark GPU MMR
                        start_time = time.time()
                        
                        _ = gpu_utils.gpu_maximal_marginal_relevance(
                            query_embedding=query_vector,
                            embedding_list=vectors[:fetch_k],
                            k=k,
                            lambda_mult=0.5,
                        )
                        
                        end_time = time.time()
                        
                        # Record results
                        duration = end_time - start_time
                        
                        run_results.append({
                            "duration": duration,
                            "k": k,
                            "fetch_k": fetch_k,
                        })
                        
                        logger.debug(
                            f"GPU MMR: k={k}, fetch_k={fetch_k}, "
                            f"run={run + 1}/{runs_per_config}, "
                            f"duration={duration:.4f}s"
                        )
                    
                    # Calculate statistics
                    durations = [result["duration"] for result in run_results]
                    
                    gpu_results[str(k)][str(fetch_k)] = {
                        "min_duration": min(durations),
                        "max_duration": max(durations),
                        "avg_duration": sum(durations) / len(durations),
                        "runs": run_results,
                    }
            
            results["gpu"] = gpu_results
        
        # Save results
        self._save_benchmark_results("mmr", results)
        
        return results
    
    def _save_benchmark_results(self, benchmark_name: str, results: Dict) -> None:
        """
        Save benchmark results to a file.
        
        Args:
            benchmark_name: Name of the benchmark.
            results: Dictionary of benchmark results.
        """
        # Create a timestamp for the filename
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # Create the filename
        filename = f"{benchmark_name}_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        # Save the results
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved benchmark results to {filepath}")
    
    def generate_summary(self, benchmark_type: Optional[str] = None) -> Dict:
        """
        Generate a summary of benchmark results.
        
        Args:
            benchmark_type: Type of benchmark to summarize. If None, summarizes all.
            
        Returns:
            Dictionary of benchmark summaries.
        """
        # Find all benchmark result files
        benchmark_files = []
        
        for file in os.listdir(self.results_dir):
            if file.endswith(".json"):
                if benchmark_type is None or file.startswith(f"{benchmark_type}_"):
                    benchmark_files.append(os.path.join(self.results_dir, file))
        
        # Group files by benchmark type
        benchmark_groups = {}
        
        for file in benchmark_files:
            basename = os.path.basename(file)
            benchmark_name = basename.split("_")[0]
            
            if benchmark_name not in benchmark_groups:
                benchmark_groups[benchmark_name] = []
            
            benchmark_groups[benchmark_name].append(file)
        
        # Generate summary for each benchmark type
        summaries = {}
        
        for benchmark_name, files in benchmark_groups.items():
            # Sort files by timestamp (newest first)
            files.sort(reverse=True)
            
            # Load the most recent benchmark result
            with open(files[0], "r") as f:
                benchmark_results = json.load(f)
            
            # Generate summary
            summary = self._generate_benchmark_summary(benchmark_name, benchmark_results)
            summaries[benchmark_name] = summary
        
        return summaries
    
    def _generate_benchmark_summary(
        self, benchmark_name: str, results: Dict
    ) -> Dict:
        """
        Generate a summary for a specific benchmark.
        
        Args:
            benchmark_name: Name of the benchmark.
            results: Dictionary of benchmark results.
            
        Returns:
            Dictionary of benchmark summary.
        """
        summary = {
            "name": benchmark_name,
            "parameters": results.get("parameters", {}),
            "summary": {},
        }
        
        if benchmark_name == "embedding":
            # Summarize embedding benchmark
            cpu_results = results.get("cpu", {})
            gpu_results = results.get("gpu", {})
            multi_gpu_results = results.get("multi_gpu", {})
            
            # Find common batch sizes
            batch_sizes = set()
            for results_dict in [cpu_results, gpu_results, multi_gpu_results]:
                batch_sizes.update(results_dict.keys())
            
            batch_sizes = sorted([int(bs) for bs in batch_sizes])
            
            # Compare performance for each batch size
            batch_comparisons = {}
            
            for batch_size in batch_sizes:
                bs_str = str(batch_size)
                
                comparison = {}
                
                # CPU performance
                if bs_str in cpu_results:
                    comparison["cpu"] = {
                        "avg_duration": cpu_results[bs_str]["avg_duration"],
                        "avg_items_per_second": cpu_results[bs_str]["avg_items_per_second"],
                    }
                
                # GPU performance
                if bs_str in gpu_results:
                    comparison["gpu"] = {
                        "avg_duration": gpu_results[bs_str]["avg_duration"],
                        "avg_items_per_second": gpu_results[bs_str]["avg_items_per_second"],
                    }
                    
                    # Calculate speedup over CPU
                    if "cpu" in comparison:
                        speedup = (
                            comparison["gpu"]["avg_items_per_second"] /
                            comparison["cpu"]["avg_items_per_second"]
                        )
                        comparison["gpu"]["speedup_vs_cpu"] = speedup
                
                # Multi-GPU performance
                if bs_str in multi_gpu_results:
                    comparison["multi_gpu"] = {
                        "avg_duration": multi_gpu_results[bs_str]["avg_duration"],
                        "avg_items_per_second": multi_gpu_results[bs_str]["avg_items_per_second"],
                    }
                    
                    # Calculate speedup over CPU
                    if "cpu" in comparison:
                        speedup = (
                            comparison["multi_gpu"]["avg_items_per_second"] /
                            comparison["cpu"]["avg_items_per_second"]
                        )
                        comparison["multi_gpu"]["speedup_vs_cpu"] = speedup
                    
                    # Calculate speedup over GPU
                    if "gpu" in comparison:
                        speedup = (
                            comparison["multi_gpu"]["avg_items_per_second"] /
                            comparison["gpu"]["avg_items_per_second"]
                        )
                        comparison["multi_gpu"]["speedup_vs_gpu"] = speedup
                
                batch_comparisons[bs_str] = comparison
            
            summary["summary"]["batch_comparisons"] = batch_comparisons
            
            # Find optimal batch size for each implementation
            optimal_batch_sizes = {}
            
            for impl, results_dict in [
                ("cpu", cpu_results),
                ("gpu", gpu_results),
                ("multi_gpu", multi_gpu_results),
            ]:
                if not results_dict:
                    continue
                
                max_throughput = 0
                optimal_batch_size = None
                
                for bs_str, result in results_dict.items():
                    throughput = result["avg_items_per_second"]
                    
                    if throughput > max_throughput:
                        max_throughput = throughput
                        optimal_batch_size = bs_str
                
                if optimal_batch_size is not None:
                    optimal_batch_sizes[impl] = {
                        "batch_size": optimal_batch_size,
                        "throughput": max_throughput,
                    }
            
            summary["summary"]["optimal_batch_sizes"] = optimal_batch_sizes
            
            # Overall speedup
            if (
                "cpu" in optimal_batch_sizes and
                "gpu" in optimal_batch_sizes
            ):
                cpu_throughput = optimal_batch_sizes["cpu"]["throughput"]
                gpu_throughput = optimal_batch_sizes["gpu"]["throughput"]
                
                summary["summary"]["overall_gpu_speedup"] = gpu_throughput / cpu_throughput
            
            if (
                "cpu" in optimal_batch_sizes and
                "multi_gpu" in optimal_batch_sizes
            ):
                cpu_throughput = optimal_batch_sizes["cpu"]["throughput"]
                multi_gpu_throughput = optimal_batch_sizes["multi_gpu"]["throughput"]
                
                summary["summary"]["overall_multi_gpu_speedup"] = multi_gpu_throughput / cpu_throughput
        
        elif benchmark_name == "mmr":
            # Summarize MMR benchmark
            cpu_results = results.get("cpu", {})
            gpu_results = results.get("gpu", {})
            
            # Compare performance for each configuration
            config_comparisons = {}
            
            for k_str, fetch_k_dict in cpu_results.items():
                if k_str not in config_comparisons:
                    config_comparisons[k_str] = {}
                
                for fetch_k_str, result in fetch_k_dict.items():
                    key = f"{k_str}_{fetch_k_str}"
                    
                    comparison = {
                        "cpu": {
                            "avg_duration": result["avg_duration"],
                        }
                    }
                    
                    # GPU performance
                    if (
                        k_str in gpu_results and
                        fetch_k_str in gpu_results[k_str]
                    ):
                        gpu_result = gpu_results[k_str][fetch_k_str]
                        
                        comparison["gpu"] = {
                            "avg_duration": gpu_result["avg_duration"],
                        }
                        
                        # Calculate speedup
                        speedup = result["avg_duration"] / gpu_result["avg_duration"]
                        comparison["gpu"]["speedup_vs_cpu"] = speedup
                    
                    config_comparisons[key] = comparison
            
            summary["summary"]["config_comparisons"] = config_comparisons
            
            # Calculate overall speedup
            if gpu_results:
                cpu_durations = []
                gpu_durations = []
                
                for k_str, fetch_k_dict in cpu_results.items():
                    for fetch_k_str, result in fetch_k_dict.items():
                        if (
                            k_str in gpu_results and
                            fetch_k_str in gpu_results[k_str]
                        ):
                            cpu_durations.append(result["avg_duration"])
                            gpu_durations.append(gpu_results[k_str][fetch_k_str]["avg_duration"])
                
                if cpu_durations and gpu_durations:
                    avg_cpu_duration = sum(cpu_durations) / len(cpu_durations)
                    avg_gpu_duration = sum(gpu_durations) / len(gpu_durations)
                    
                    overall_speedup = avg_cpu_duration / avg_gpu_duration
                    summary["summary"]["overall_gpu_speedup"] = overall_speedup
        
        return summary


# Command-line interface
if __name__ == "__main__":
    import argparse
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Benchmark CPU vs GPU performance")
    parser.add_argument(
        "--embedding",
        action="store_true",
        help="Benchmark embedding generation performance",
    )
    parser.add_argument(
        "--mmr",
        action="store_true",
        help="Benchmark Maximal Marginal Relevance (MMR) performance",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of samples to use for benchmarking",
    )
    parser.add_argument(
        "--text-length",
        type=int,
        default=100,
        help="Average length of each text sample",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Name of the embedding model to use",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="benchmark_results",
        help="Directory to store benchmark results",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Generate a summary of benchmark results",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,8,32,128,512",
        help="Comma-separated list of batch sizes to benchmark",
    )
    
    args = parser.parse_args()
    
    # Create benchmark instance
    benchmark = Benchmark(
        results_dir=args.results_dir,
        embedding_model=args.model,
    )
    
    # Parse batch sizes
    batch_sizes = [int(bs) for bs in args.batch_sizes.split(",")]
    
    # Run benchmarks
    if args.embedding:
        logger.info("Running embedding benchmark")
        benchmark.benchmark_embedding(
            num_samples=args.num_samples,
            text_length=args.text_length,
            batch_sizes=batch_sizes,
        )
    
    if args.mmr:
        logger.info("Running MMR benchmark")
        benchmark.benchmark_mmr()
    
    # Generate summary
    if args.summary:
        logger.info("Generating benchmark summary")
        summaries = benchmark.generate_summary()
        
        # Print summary
        for benchmark_name, summary in summaries.items():
            logger.info(f"Summary for {benchmark_name} benchmark:")
            
            if "overall_gpu_speedup" in summary["summary"]:
                logger.info(
                    f"Overall GPU speedup: {summary['summary']['overall_gpu_speedup']:.2f}x"
                )
            
            if "overall_multi_gpu_speedup" in summary["summary"]:
                logger.info(
                    f"Overall Multi-GPU speedup: {summary['summary']['overall_multi_gpu_speedup']:.2f}x"
                )
    
    # If no benchmarks specified, print help
    if not (args.embedding or args.mmr or args.summary):
        parser.print_help()