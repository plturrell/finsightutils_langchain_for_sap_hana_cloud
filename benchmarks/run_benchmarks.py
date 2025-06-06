#!/usr/bin/env python
"""
Automated benchmarking tool for LangChain SAP HANA integration.

This script provides comprehensive benchmarking of various aspects of the system:
- Embedding generation performance
- Vector search performance
- MMR search performance
- Database operations performance
- GPU acceleration benefits
- TensorRT optimization

It can be run manually or as part of the CI/CD pipeline.
"""

import os
import sys
import time
import json
import logging
import argparse
import datetime
import subprocess
from typing import Dict, List, Any, Optional, Tuple

import requests
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("benchmark")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run automated performance benchmarks for LangChain SAP HANA integration"
    )
    
    parser.add_argument(
        "--api-url",
        type=str,
        default=os.environ.get("BENCHMARK_API_URL", "http://localhost:8000"),
        help="API URL for benchmarking",
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("BENCHMARK_API_KEY", "test-api-key"),
        help="API key for authentication",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.environ.get("BENCHMARK_OUTPUT_DIR", "benchmark_results"),
        help="Directory for benchmark results",
    )
    
    parser.add_argument(
        "--skip-embedding",
        action="store_true",
        help="Skip embedding benchmarks",
    )
    
    parser.add_argument(
        "--skip-search",
        action="store_true",
        help="Skip search benchmarks",
    )
    
    parser.add_argument(
        "--skip-mmr",
        action="store_true",
        help="Skip MMR benchmarks",
    )
    
    parser.add_argument(
        "--skip-tensorrt",
        action="store_true",
        help="Skip TensorRT benchmarks",
    )
    
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip generating plots",
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.environ.get("BENCHMARK_TIMEOUT", "300")),
        help="Timeout for API requests in seconds",
    )
    
    parser.add_argument(
        "--iterations",
        type=int,
        default=int(os.environ.get("BENCHMARK_ITERATIONS", "3")),
        help="Number of iterations for each benchmark",
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    
    parser.add_argument(
        "--compare",
        type=str,
        help="Path to previous benchmark results for comparison",
    )
    
    return parser.parse_args()


class BenchmarkRunner:
    """
    Automated benchmark runner for LangChain SAP HANA integration.
    
    This class provides methods for running various benchmarks and
    generating comprehensive reports.
    """
    
    def __init__(
        self,
        api_url: str,
        api_key: str,
        output_dir: str,
        timeout: int = 300,
        iterations: int = 3,
        skip_plots: bool = False,
    ):
        """Initialize the benchmark runner."""
        self.api_url = api_url
        self.api_key = api_key
        self.output_dir = output_dir
        self.timeout = timeout
        self.iterations = iterations
        self.skip_plots = skip_plots
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create subdirectories for different types of results
        os.makedirs(os.path.join(output_dir, "data"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "reports"), exist_ok=True)
        
        # Create session
        self.session = requests.Session()
        self.session.headers.update({
            "X-API-Key": api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        })
        
        # Store benchmark results
        self.results = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "config": {
                "api_url": api_url,
                "timeout": timeout,
                "iterations": iterations,
            },
            "system_info": {},
            "benchmarks": {},
        }
        
        # Get system info
        self._get_system_info()
    
    def _get_system_info(self):
        """Get system information from the API."""
        try:
            # Get GPU info
            response = self.session.get(f"{self.api_url}/gpu/info", timeout=self.timeout)
            response.raise_for_status()
            self.results["system_info"]["gpu"] = response.json()
            
            # Get API info
            response = self.session.get(f"{self.api_url}/", timeout=self.timeout)
            response.raise_for_status()
            self.results["system_info"]["api"] = response.json()
            
            # Get detailed health info
            response = self.session.get(f"{self.api_url}/health/complete", timeout=self.timeout)
            response.raise_for_status()
            self.results["system_info"]["health"] = response.json()
            
        except Exception as e:
            logger.warning(f"Failed to get system info: {str(e)}")
            self.results["system_info"]["error"] = str(e)
    
    def run_embedding_benchmark(self):
        """Run embedding generation benchmark."""
        logger.info("Running embedding generation benchmark")
        
        # Define benchmark parameters
        params = {
            "texts": [
                "This is a short text for benchmarking embedding performance.",
                "This is another short text with slightly different content.",
                "SAP HANA Cloud provides powerful database capabilities for enterprise applications.",
                "Vector embeddings are useful for semantic search and similarity comparisons.",
                "GPU acceleration can significantly improve the performance of embedding generation.",
            ],
            "count": self.iterations * 10,  # Run multiple iterations for better statistics
            "batch_size": 32,  # Use a standard batch size
        }
        
        try:
            # Run benchmark
            response = self.session.post(
                f"{self.api_url}/benchmark/embedding",
                json=params,
                timeout=self.timeout,
            )
            response.raise_for_status()
            
            # Store results
            self.results["benchmarks"]["embedding"] = response.json()
            
            # Save raw data
            self._save_benchmark_data("embedding", self.results["benchmarks"]["embedding"])
            
            # Generate plots if enabled
            if not self.skip_plots:
                self._generate_embedding_plots(self.results["benchmarks"]["embedding"])
            
            logger.info("Embedding benchmark completed successfully")
            return True
        
        except Exception as e:
            logger.error(f"Embedding benchmark failed: {str(e)}")
            self.results["benchmarks"]["embedding"] = {"error": str(e)}
            return False
    
    def run_search_benchmark(self):
        """Run vector search benchmark."""
        logger.info("Running vector search benchmark")
        
        # Define benchmark parameters
        params = {
            "query": "SAP HANA Cloud with vector search capabilities",
            "k": 10,
            "iterations": self.iterations * 5,  # Run multiple iterations for better statistics
        }
        
        try:
            # Run benchmark
            response = self.session.post(
                f"{self.api_url}/benchmark/search",
                json=params,
                timeout=self.timeout,
            )
            response.raise_for_status()
            
            # Store results
            self.results["benchmarks"]["search"] = response.json()
            
            # Save raw data
            self._save_benchmark_data("search", self.results["benchmarks"]["search"])
            
            # Generate plots if enabled
            if not self.skip_plots:
                self._generate_search_plots(self.results["benchmarks"]["search"])
            
            logger.info("Search benchmark completed successfully")
            return True
        
        except Exception as e:
            logger.error(f"Search benchmark failed: {str(e)}")
            self.results["benchmarks"]["search"] = {"error": str(e)}
            return False
    
    def run_mmr_benchmark(self):
        """Run MMR search benchmark."""
        logger.info("Running MMR search benchmark")
        
        # Check if GPU is available (MMR benchmark requires GPU)
        if not self.results["system_info"].get("gpu", {}).get("gpu_available", False):
            logger.warning("Skipping MMR benchmark: GPU not available")
            self.results["benchmarks"]["mmr"] = {"error": "GPU not available"}
            return False
        
        try:
            # Get the benchmark via API
            response = self.session.post(
                f"{self.api_url}/benchmark/mmr",
                json={},
                timeout=self.timeout,
            )
            response.raise_for_status()
            
            # Store results
            self.results["benchmarks"]["mmr"] = response.json()
            
            # Save raw data
            self._save_benchmark_data("mmr", self.results["benchmarks"]["mmr"])
            
            # Generate plots if enabled
            if not self.skip_plots:
                self._generate_mmr_plots(self.results["benchmarks"]["mmr"])
            
            logger.info("MMR benchmark completed successfully")
            return True
        
        except Exception as e:
            logger.error(f"MMR benchmark failed: {str(e)}")
            self.results["benchmarks"]["mmr"] = {"error": str(e)}
            return False
    
    def run_tensorrt_benchmark(self):
        """Run TensorRT optimization benchmark."""
        logger.info("Running TensorRT optimization benchmark")
        
        # Check if TensorRT is available
        if not self.results["system_info"].get("gpu", {}).get("tensorrt_available", False):
            logger.warning("Skipping TensorRT benchmark: TensorRT not available")
            self.results["benchmarks"]["tensorrt"] = {"error": "TensorRT not available"}
            return False
        
        # Define benchmark parameters
        params = {
            "model_name": "all-MiniLM-L6-v2",
            "precision": "fp16",
            "batch_sizes": [1, 8, 32, 64, 128],
            "input_length": 128,
            "iterations": self.iterations,
        }
        
        try:
            # Run benchmark
            response = self.session.post(
                f"{self.api_url}/benchmark/tensorrt",
                json=params,
                timeout=self.timeout * 2,  # TensorRT benchmark can take longer
            )
            response.raise_for_status()
            
            # Store results
            self.results["benchmarks"]["tensorrt"] = response.json()
            
            # Save raw data
            self._save_benchmark_data("tensorrt", self.results["benchmarks"]["tensorrt"])
            
            # Generate plots if enabled
            if not self.skip_plots:
                self._generate_tensorrt_plots(self.results["benchmarks"]["tensorrt"])
            
            logger.info("TensorRT benchmark completed successfully")
            return True
        
        except Exception as e:
            logger.error(f"TensorRT benchmark failed: {str(e)}")
            self.results["benchmarks"]["tensorrt"] = {"error": str(e)}
            return False
    
    def _save_benchmark_data(self, benchmark_name: str, data: Dict):
        """Save benchmark data to a JSON file."""
        filename = f"{benchmark_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.output_dir, "data", filename)
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {benchmark_name} benchmark data to {filepath}")
    
    def _generate_embedding_plots(self, data: Dict):
        """Generate plots for embedding benchmark results."""
        try:
            # Extract data for plotting
            batch_sizes = []
            cpu_throughput = []
            gpu_throughput = []
            multi_gpu_throughput = []
            
            for batch_size, results in data.items():
                if batch_size == "parameters":
                    continue
                
                batch_sizes.append(int(batch_size))
                
                if "cpu" in results:
                    cpu_throughput.append(results["cpu"]["avg_items_per_second"])
                else:
                    cpu_throughput.append(None)
                
                if "gpu" in results:
                    gpu_throughput.append(results["gpu"]["avg_items_per_second"])
                else:
                    gpu_throughput.append(None)
                
                if "multi_gpu" in results:
                    multi_gpu_throughput.append(results["multi_gpu"]["avg_items_per_second"])
                else:
                    multi_gpu_throughput.append(None)
            
            # Sort by batch size
            sorted_indices = np.argsort(batch_sizes)
            batch_sizes = [batch_sizes[i] for i in sorted_indices]
            cpu_throughput = [cpu_throughput[i] for i in sorted_indices]
            gpu_throughput = [gpu_throughput[i] for i in sorted_indices]
            multi_gpu_throughput = [multi_gpu_throughput[i] for i in sorted_indices]
            
            # Create throughput plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if any(x is not None for x in cpu_throughput):
                ax.plot(batch_sizes, cpu_throughput, marker='o', label='CPU')
            
            if any(x is not None for x in gpu_throughput):
                ax.plot(batch_sizes, gpu_throughput, marker='s', label='GPU')
            
            if any(x is not None for x in multi_gpu_throughput):
                ax.plot(batch_sizes, multi_gpu_throughput, marker='^', label='Multi-GPU')
            
            ax.set_xlabel('Batch Size')
            ax.set_ylabel('Throughput (items/second)')
            ax.set_title('Embedding Generation Throughput')
            ax.set_xscale('log', base=2)
            ax.set_yscale('log')
            ax.grid(True)
            ax.legend()
            
            # Save plot
            plot_path = os.path.join(self.output_dir, "plots", "embedding_throughput.png")
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            # Create speedup plot
            if any(x is not None for x in gpu_throughput) and any(x is not None for x in cpu_throughput):
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Calculate speedup
                gpu_speedup = []
                multi_gpu_speedup = []
                valid_batch_sizes = []
                
                for i, bs in enumerate(batch_sizes):
                    if cpu_throughput[i] is not None and cpu_throughput[i] > 0:
                        valid_batch_sizes.append(bs)
                        
                        if gpu_throughput[i] is not None:
                            gpu_speedup.append(gpu_throughput[i] / cpu_throughput[i])
                        else:
                            gpu_speedup.append(None)
                        
                        if multi_gpu_throughput[i] is not None:
                            multi_gpu_speedup.append(multi_gpu_throughput[i] / cpu_throughput[i])
                        else:
                            multi_gpu_speedup.append(None)
                
                if valid_batch_sizes:
                    if any(x is not None for x in gpu_speedup):
                        ax.plot(valid_batch_sizes, gpu_speedup, marker='s', label='GPU vs CPU')
                    
                    if any(x is not None for x in multi_gpu_speedup):
                        ax.plot(valid_batch_sizes, multi_gpu_speedup, marker='^', label='Multi-GPU vs CPU')
                    
                    ax.set_xlabel('Batch Size')
                    ax.set_ylabel('Speedup Factor')
                    ax.set_title('GPU Acceleration Speedup')
                    ax.set_xscale('log', base=2)
                    ax.grid(True)
                    ax.legend()
                    
                    # Save plot
                    plot_path = os.path.join(self.output_dir, "plots", "embedding_speedup.png")
                    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                
                plt.close(fig)
                
            logger.info("Generated embedding benchmark plots")
        
        except Exception as e:
            logger.error(f"Failed to generate embedding plots: {str(e)}")
    
    def _generate_search_plots(self, data: Dict):
        """Generate plots for search benchmark results."""
        try:
            # Check if we have valid data
            if "iterations" not in data or not data["iterations"]:
                logger.warning("No valid data for search benchmark plots")
                return
            
            # Extract data
            iterations = data["iterations"]
            durations = [iter_data["duration"] for iter_data in iterations]
            
            # Create histogram of search durations
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.hist(durations, bins=20, alpha=0.7)
            ax.axvline(x=np.mean(durations), color='r', linestyle='--', 
                      label=f'Mean: {np.mean(durations):.3f}s')
            ax.axvline(x=np.median(durations), color='g', linestyle='-', 
                      label=f'Median: {np.median(durations):.3f}s')
            
            ax.set_xlabel('Duration (seconds)')
            ax.set_ylabel('Frequency')
            ax.set_title('Vector Search Duration Distribution')
            ax.grid(True)
            ax.legend()
            
            # Save plot
            plot_path = os.path.join(self.output_dir, "plots", "search_duration_hist.png")
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            logger.info("Generated search benchmark plots")
        
        except Exception as e:
            logger.error(f"Failed to generate search plots: {str(e)}")
    
    def _generate_mmr_plots(self, data: Dict):
        """Generate plots for MMR benchmark results."""
        try:
            # Check if we have valid data
            if "cpu" not in data or "gpu" not in data:
                logger.warning("No valid data for MMR benchmark plots")
                return
            
            cpu_results = data["cpu"]
            gpu_results = data["gpu"]
            
            # Extract data for various configurations
            configs = []
            cpu_times = []
            gpu_times = []
            speedups = []
            
            for k_str, fetch_k_dict in cpu_results.items():
                for fetch_k_str, cpu_result in fetch_k_dict.items():
                    if k_str in gpu_results and fetch_k_str in gpu_results[k_str]:
                        gpu_result = gpu_results[k_str][fetch_k_str]
                        
                        configs.append(f"k={k_str}, fetch_k={fetch_k_str}")
                        cpu_times.append(cpu_result["avg_duration"])
                        gpu_times.append(gpu_result["avg_duration"])
                        speedups.append(cpu_result["avg_duration"] / gpu_result["avg_duration"])
            
            if not configs:
                logger.warning("No matching configurations for MMR benchmark plots")
                return
            
            # Create bar chart comparing CPU and GPU times
            fig, ax = plt.subplots(figsize=(12, 6))
            
            x = np.arange(len(configs))
            width = 0.35
            
            ax.bar(x - width/2, cpu_times, width, label='CPU')
            ax.bar(x + width/2, gpu_times, width, label='GPU')
            
            ax.set_xlabel('Configuration')
            ax.set_ylabel('Duration (seconds)')
            ax.set_title('MMR Search Duration Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(configs, rotation=45, ha='right')
            ax.legend()
            
            fig.tight_layout()
            
            # Save plot
            plot_path = os.path.join(self.output_dir, "plots", "mmr_duration_comparison.png")
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            # Create speedup plot
            fig, ax = plt.subplots(figsize=(12, 6))
            
            ax.bar(x, speedups, width)
            
            ax.set_xlabel('Configuration')
            ax.set_ylabel('Speedup Factor (CPU time / GPU time)')
            ax.set_title('GPU Acceleration Speedup for MMR Search')
            ax.set_xticks(x)
            ax.set_xticklabels(configs, rotation=45, ha='right')
            ax.grid(True, axis='y')
            
            fig.tight_layout()
            
            # Save plot
            plot_path = os.path.join(self.output_dir, "plots", "mmr_speedup.png")
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            logger.info("Generated MMR benchmark plots")
        
        except Exception as e:
            logger.error(f"Failed to generate MMR plots: {str(e)}")
    
    def _generate_tensorrt_plots(self, data: Dict):
        """Generate plots for TensorRT benchmark results."""
        try:
            # Check if we have valid data
            if "batch_results" not in data or not data["batch_results"]:
                logger.warning("No valid data for TensorRT benchmark plots")
                return
            
            # Extract data
            batch_sizes = []
            pytorch_times = []
            tensorrt_times = []
            speedups = []
            
            for result in data["batch_results"]:
                batch_sizes.append(result["batch_size"])
                pytorch_times.append(result["pytorch_time_ms"])
                tensorrt_times.append(result["tensorrt_time_ms"])
                speedups.append(result["speedup_factor"])
            
            # Create comparison plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.plot(batch_sizes, pytorch_times, marker='o', label='PyTorch')
            ax.plot(batch_sizes, tensorrt_times, marker='s', label='TensorRT')
            
            ax.set_xlabel('Batch Size')
            ax.set_ylabel('Inference Time (ms)')
            ax.set_title('PyTorch vs TensorRT Inference Time')
            ax.set_xscale('log', base=2)
            ax.set_yscale('log')
            ax.grid(True)
            ax.legend()
            
            # Save plot
            plot_path = os.path.join(self.output_dir, "plots", "tensorrt_inference_time.png")
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            # Create speedup plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.plot(batch_sizes, speedups, marker='o')
            
            ax.set_xlabel('Batch Size')
            ax.set_ylabel('Speedup Factor (PyTorch time / TensorRT time)')
            ax.set_title('TensorRT Optimization Speedup')
            ax.set_xscale('log', base=2)
            ax.grid(True)
            
            # Save plot
            plot_path = os.path.join(self.output_dir, "plots", "tensorrt_speedup.png")
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            # Create throughput plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            pytorch_throughput = [batch_sizes[i] * 1000 / pytorch_times[i] for i in range(len(batch_sizes))]
            tensorrt_throughput = [batch_sizes[i] * 1000 / tensorrt_times[i] for i in range(len(batch_sizes))]
            
            ax.plot(batch_sizes, pytorch_throughput, marker='o', label='PyTorch')
            ax.plot(batch_sizes, tensorrt_throughput, marker='s', label='TensorRT')
            
            ax.set_xlabel('Batch Size')
            ax.set_ylabel('Throughput (items/second)')
            ax.set_title('PyTorch vs TensorRT Throughput')
            ax.set_xscale('log', base=2)
            ax.grid(True)
            ax.legend()
            
            # Save plot
            plot_path = os.path.join(self.output_dir, "plots", "tensorrt_throughput.png")
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            logger.info("Generated TensorRT benchmark plots")
        
        except Exception as e:
            logger.error(f"Failed to generate TensorRT plots: {str(e)}")
    
    def generate_report(self, comparison_data: Optional[Dict] = None):
        """
        Generate a comprehensive benchmark report.
        
        Args:
            comparison_data: Optional previous benchmark results for comparison
        """
        logger.info("Generating benchmark report")
        
        try:
            # Basic report data
            report = {
                "timestamp": self.results["timestamp"],
                "api_url": self.api_url,
                "system_info": self._extract_system_info(),
                "benchmarks": {},
                "comparison": None,
            }
            
            # Extract benchmark summaries
            if "embedding" in self.results["benchmarks"]:
                report["benchmarks"]["embedding"] = self._extract_embedding_summary()
            
            if "search" in self.results["benchmarks"]:
                report["benchmarks"]["search"] = self._extract_search_summary()
            
            if "mmr" in self.results["benchmarks"]:
                report["benchmarks"]["mmr"] = self._extract_mmr_summary()
            
            if "tensorrt" in self.results["benchmarks"]:
                report["benchmarks"]["tensorrt"] = self._extract_tensorrt_summary()
            
            # Add comparison if available
            if comparison_data:
                report["comparison"] = self._generate_comparison(comparison_data)
            
            # Save report to JSON
            report_path = os.path.join(self.output_dir, "reports", f"benchmark_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
            
            # Generate HTML report
            html_report = self._generate_html_report(report)
            html_path = os.path.join(self.output_dir, "reports", f"benchmark_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
            with open(html_path, "w") as f:
                f.write(html_report)
            
            # Generate Markdown report
            md_report = self._generate_markdown_report(report)
            md_path = os.path.join(self.output_dir, "reports", f"benchmark_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
            with open(md_path, "w") as f:
                f.write(md_report)
            
            logger.info(f"Benchmark reports saved to {os.path.join(self.output_dir, 'reports')}")
            
            return report_path, html_path, md_path
        
        except Exception as e:
            logger.error(f"Failed to generate benchmark report: {str(e)}")
            return None, None, None
    
    def _extract_system_info(self) -> Dict:
        """Extract relevant system information for the report."""
        system_info = {}
        
        # Extract GPU info
        gpu_info = self.results["system_info"].get("gpu", {})
        
        if gpu_info:
            system_info["gpu"] = {
                "available": gpu_info.get("gpu_available", False),
                "count": gpu_info.get("device_count", 0),
                "devices": [],
                "cuda_version": gpu_info.get("cuda_version", "unknown"),
                "tensorrt_available": gpu_info.get("tensorrt_available", False),
            }
            
            for device in gpu_info.get("devices", []):
                system_info["gpu"]["devices"].append({
                    "name": device.get("name", "unknown"),
                    "memory_total_mb": device.get("memory_total", 0) / (1024 * 1024),
                    "compute_capability": device.get("compute_capability", "unknown"),
                })
        
        # Extract API info
        api_info = self.results["system_info"].get("api", {})
        
        if api_info:
            system_info["api"] = {
                "version": api_info.get("version", "unknown"),
                "platform": api_info.get("platform", "unknown"),
            }
        
        return system_info
    
    def _extract_embedding_summary(self) -> Dict:
        """Extract embedding benchmark summary."""
        embedding_data = self.results["benchmarks"].get("embedding", {})
        
        if "error" in embedding_data:
            return {"error": embedding_data["error"]}
        
        summary = {
            "batch_sizes": [],
            "throughput": {},
            "speedup": {},
        }
        
        # Process each batch size
        for batch_size, results in embedding_data.items():
            if batch_size == "parameters":
                continue
            
            summary["batch_sizes"].append(int(batch_size))
            
            # Extract throughput
            if "cpu" in results:
                if "throughput" not in summary["throughput"]:
                    summary["throughput"]["cpu"] = []
                summary["throughput"]["cpu"].append(results["cpu"]["avg_items_per_second"])
            
            if "gpu" in results:
                if "throughput" not in summary["throughput"]:
                    summary["throughput"]["gpu"] = []
                summary["throughput"]["gpu"].append(results["gpu"]["avg_items_per_second"])
            
            if "multi_gpu" in results:
                if "throughput" not in summary["throughput"]:
                    summary["throughput"]["multi_gpu"] = []
                summary["throughput"]["multi_gpu"].append(results["multi_gpu"]["avg_items_per_second"])
            
            # Calculate speedup
            if "cpu" in results:
                cpu_throughput = results["cpu"]["avg_items_per_second"]
                
                if "gpu" in results:
                    if "gpu_vs_cpu" not in summary["speedup"]:
                        summary["speedup"]["gpu_vs_cpu"] = []
                    
                    gpu_throughput = results["gpu"]["avg_items_per_second"]
                    speedup = gpu_throughput / cpu_throughput if cpu_throughput > 0 else 0
                    summary["speedup"]["gpu_vs_cpu"].append(speedup)
                
                if "multi_gpu" in results:
                    if "multi_gpu_vs_cpu" not in summary["speedup"]:
                        summary["speedup"]["multi_gpu_vs_cpu"] = []
                    
                    multi_gpu_throughput = results["multi_gpu"]["avg_items_per_second"]
                    speedup = multi_gpu_throughput / cpu_throughput if cpu_throughput > 0 else 0
                    summary["speedup"]["multi_gpu_vs_cpu"].append(speedup)
        
        return summary
    
    def _extract_search_summary(self) -> Dict:
        """Extract search benchmark summary."""
        search_data = self.results["benchmarks"].get("search", {})
        
        if "error" in search_data:
            return {"error": search_data["error"]}
        
        summary = {
            "iterations": len(search_data.get("iterations", [])),
            "duration": {
                "mean": np.mean([i["duration"] for i in search_data.get("iterations", [])]),
                "median": np.median([i["duration"] for i in search_data.get("iterations", [])]),
                "min": np.min([i["duration"] for i in search_data.get("iterations", [])]),
                "max": np.max([i["duration"] for i in search_data.get("iterations", [])]),
                "std": np.std([i["duration"] for i in search_data.get("iterations", [])]),
            },
            "parameters": search_data.get("parameters", {}),
        }
        
        return summary
    
    def _extract_mmr_summary(self) -> Dict:
        """Extract MMR benchmark summary."""
        mmr_data = self.results["benchmarks"].get("mmr", {})
        
        if "error" in mmr_data:
            return {"error": mmr_data["error"]}
        
        summary = {
            "configurations": [],
            "cpu_times": [],
            "gpu_times": [],
            "speedups": [],
        }
        
        # Process each configuration
        cpu_results = mmr_data.get("cpu", {})
        gpu_results = mmr_data.get("gpu", {})
        
        for k_str, fetch_k_dict in cpu_results.items():
            for fetch_k_str, cpu_result in fetch_k_dict.items():
                if k_str in gpu_results and fetch_k_str in gpu_results[k_str]:
                    gpu_result = gpu_results[k_str][fetch_k_str]
                    
                    summary["configurations"].append(f"k={k_str}, fetch_k={fetch_k_str}")
                    summary["cpu_times"].append(cpu_result["avg_duration"])
                    summary["gpu_times"].append(gpu_result["avg_duration"])
                    summary["speedups"].append(cpu_result["avg_duration"] / gpu_result["avg_duration"])
        
        # Calculate overall statistics
        if summary["speedups"]:
            summary["overall"] = {
                "mean_speedup": np.mean(summary["speedups"]),
                "median_speedup": np.median(summary["speedups"]),
                "min_speedup": np.min(summary["speedups"]),
                "max_speedup": np.max(summary["speedups"]),
            }
        
        return summary
    
    def _extract_tensorrt_summary(self) -> Dict:
        """Extract TensorRT benchmark summary."""
        tensorrt_data = self.results["benchmarks"].get("tensorrt", {})
        
        if "error" in tensorrt_data:
            return {"error": tensorrt_data["error"]}
        
        summary = {
            "model": tensorrt_data.get("model", "unknown"),
            "precision": tensorrt_data.get("precision", "unknown"),
            "batch_sizes": [],
            "pytorch_times": [],
            "tensorrt_times": [],
            "speedups": [],
            "pytorch_throughput": [],
            "tensorrt_throughput": [],
        }
        
        # Process batch results
        for result in tensorrt_data.get("batch_results", []):
            summary["batch_sizes"].append(result["batch_size"])
            summary["pytorch_times"].append(result["pytorch_time_ms"])
            summary["tensorrt_times"].append(result["tensorrt_time_ms"])
            summary["speedups"].append(result["speedup_factor"])
            summary["pytorch_throughput"].append(result["pytorch_throughput"])
            summary["tensorrt_throughput"].append(result["tensorrt_throughput"])
        
        # Calculate overall statistics
        if summary["speedups"]:
            summary["overall"] = {
                "mean_speedup": np.mean(summary["speedups"]),
                "median_speedup": np.median(summary["speedups"]),
                "min_speedup": np.min(summary["speedups"]),
                "max_speedup": np.max(summary["speedups"]),
            }
        
        return summary
    
    def _generate_comparison(self, comparison_data: Dict) -> Dict:
        """Generate comparison with previous benchmark results."""
        comparison = {
            "timestamp_current": self.results["timestamp"],
            "timestamp_previous": comparison_data.get("timestamp", "unknown"),
            "benchmarks": {},
        }
        
        # Compare embedding benchmarks
        if ("embedding" in self.results["benchmarks"] and 
            "embedding" in comparison_data.get("benchmarks", {})):
            
            current = self.results["benchmarks"]["embedding"]
            previous = comparison_data["benchmarks"]["embedding"]
            
            # Skip if either has an error
            if "error" in current or "error" in previous:
                comparison["benchmarks"]["embedding"] = {
                    "status": "error",
                    "message": "Cannot compare due to errors in benchmarks",
                }
                
            else:
                # Compare optimal batch sizes and throughput
                embedding_comparison = {
                    "optimal_batch_size": {},
                    "throughput_change": {},
                    "speedup_change": {},
                }
                
                # Process each implementation
                for impl in ["cpu", "gpu", "multi_gpu"]:
                    if (impl in current.get("summary", {}).get("optimal_batch_sizes", {}) and
                        impl in previous.get("summary", {}).get("optimal_batch_sizes", {})):
                        
                        current_opt = current["summary"]["optimal_batch_sizes"][impl]
                        previous_opt = previous["summary"]["optimal_batch_sizes"][impl]
                        
                        embedding_comparison["optimal_batch_size"][impl] = {
                            "current": current_opt["batch_size"],
                            "previous": previous_opt["batch_size"],
                        }
                        
                        embedding_comparison["throughput_change"][impl] = {
                            "current": current_opt["throughput"],
                            "previous": previous_opt["throughput"],
                            "percent_change": ((current_opt["throughput"] - previous_opt["throughput"]) / 
                                              previous_opt["throughput"]) * 100,
                        }
                
                # Compare speedups
                if "overall_gpu_speedup" in current.get("summary", {}) and "overall_gpu_speedup" in previous.get("summary", {}):
                    embedding_comparison["speedup_change"]["gpu_vs_cpu"] = {
                        "current": current["summary"]["overall_gpu_speedup"],
                        "previous": previous["summary"]["overall_gpu_speedup"],
                        "percent_change": ((current["summary"]["overall_gpu_speedup"] - previous["summary"]["overall_gpu_speedup"]) /
                                          previous["summary"]["overall_gpu_speedup"]) * 100,
                    }
                
                if "overall_multi_gpu_speedup" in current.get("summary", {}) and "overall_multi_gpu_speedup" in previous.get("summary", {}):
                    embedding_comparison["speedup_change"]["multi_gpu_vs_cpu"] = {
                        "current": current["summary"]["overall_multi_gpu_speedup"],
                        "previous": previous["summary"]["overall_multi_gpu_speedup"],
                        "percent_change": ((current["summary"]["overall_multi_gpu_speedup"] - previous["summary"]["overall_multi_gpu_speedup"]) /
                                          previous["summary"]["overall_multi_gpu_speedup"]) * 100,
                    }
                
                comparison["benchmarks"]["embedding"] = embedding_comparison
        
        # Compare TensorRT benchmarks
        if ("tensorrt" in self.results["benchmarks"] and 
            "tensorrt" in comparison_data.get("benchmarks", {})):
            
            current = self.results["benchmarks"]["tensorrt"]
            previous = comparison_data["benchmarks"]["tensorrt"]
            
            # Skip if either has an error
            if "error" in current or "error" in previous:
                comparison["benchmarks"]["tensorrt"] = {
                    "status": "error",
                    "message": "Cannot compare due to errors in benchmarks",
                }
                
            else:
                # Extract overall speedups
                current_summary = self._extract_tensorrt_summary()
                
                # Previous summary might be in a different format
                if "overall" in current_summary and "overall" in previous:
                    tensorrt_comparison = {
                        "speedup_change": {
                            "current": current_summary["overall"]["mean_speedup"],
                            "previous": previous["overall"]["mean_speedup"],
                            "percent_change": ((current_summary["overall"]["mean_speedup"] - previous["overall"]["mean_speedup"]) /
                                              previous["overall"]["mean_speedup"]) * 100,
                        }
                    }
                    
                    comparison["benchmarks"]["tensorrt"] = tensorrt_comparison
        
        return comparison
    
    def _generate_html_report(self, report: Dict) -> str:
        """Generate an HTML report from benchmark data."""
        # This is a simplified HTML report generator
        # In a real implementation, you might use a template engine
        
        html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LangChain SAP HANA Benchmark Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
        h1, h2, h3 { color: #333; }
        .container { max-width: 1200px; margin: 0 auto; }
        .section { margin-bottom: 30px; }
        .card { border: 1px solid #ddd; border-radius: 4px; padding: 15px; margin-bottom: 20px; }
        .header { display: flex; justify-content: space-between; align-items: center; }
        table { width: 100%; border-collapse: collapse; }
        th, td { text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }
        th { background-color: #f2f2f2; }
        .plot { margin: 20px 0; text-align: center; }
        .plot img { max-width: 100%; height: auto; }
        .comparison { background-color: #f9f9f9; padding: 10px; border-radius: 4px; }
        .positive { color: green; }
        .negative { color: red; }
    </style>
</head>
<body>
    <div class="container">
        <h1>LangChain SAP HANA Benchmark Report</h1>
        
        <div class="section">
            <div class="header">
                <h2>System Information</h2>
                <p>Generated: """ + report["timestamp"] + """</p>
            </div>
"""
        
        # Add system info
        system_info = report["system_info"]
        
        html += """
            <div class="card">
                <h3>API Information</h3>
                <table>
                    <tr>
                        <th>API URL</th>
                        <td>""" + report["api_url"] + """</td>
                    </tr>
                    <tr>
                        <th>Version</th>
                        <td>""" + system_info.get("api", {}).get("version", "Unknown") + """</td>
                    </tr>
                </table>
            </div>
"""
        
        # Add GPU info
        if "gpu" in system_info:
            gpu_info = system_info["gpu"]
            html += """
            <div class="card">
                <h3>GPU Information</h3>
                <table>
                    <tr>
                        <th>GPU Available</th>
                        <td>""" + str(gpu_info.get("available", False)) + """</td>
                    </tr>
                    <tr>
                        <th>GPU Count</th>
                        <td>""" + str(gpu_info.get("count", 0)) + """</td>
                    </tr>
                    <tr>
                        <th>CUDA Version</th>
                        <td>""" + gpu_info.get("cuda_version", "Unknown") + """</td>
                    </tr>
                    <tr>
                        <th>TensorRT Available</th>
                        <td>""" + str(gpu_info.get("tensorrt_available", False)) + """</td>
                    </tr>
                </table>
            </div>
"""
        
            # Add GPU device info
            if gpu_info.get("devices"):
                html += """
            <div class="card">
                <h3>GPU Devices</h3>
                <table>
                    <tr>
                        <th>Device</th>
                        <th>Model</th>
                        <th>Memory (MB)</th>
                        <th>Compute Capability</th>
                    </tr>
"""
                
                for i, device in enumerate(gpu_info["devices"]):
                    html += """
                    <tr>
                        <td>""" + str(i) + """</td>
                        <td>""" + device.get("name", "Unknown") + """</td>
                        <td>""" + str(device.get("memory_total_mb", 0)) + """</td>
                        <td>""" + device.get("compute_capability", "Unknown") + """</td>
                    </tr>
"""
                
                html += """
                </table>
            </div>
"""
        
        # Add benchmark results
        html += """
        </div>
        
        <div class="section">
            <h2>Benchmark Results</h2>
"""
        
        # Add embedding benchmark results
        if "embedding" in report["benchmarks"] and "error" not in report["benchmarks"]["embedding"]:
            embedding = report["benchmarks"]["embedding"]
            
            html += """
            <div class="card">
                <h3>Embedding Generation Performance</h3>
"""
            
            # Add plots if available
            plot_path = os.path.join(self.output_dir, "plots", "embedding_throughput.png")
            if os.path.exists(plot_path):
                html += """
                <div class="plot">
                    <img src="../plots/embedding_throughput.png" alt="Embedding Throughput">
                    <p>Embedding generation throughput comparison</p>
                </div>
"""
            
            plot_path = os.path.join(self.output_dir, "plots", "embedding_speedup.png")
            if os.path.exists(plot_path):
                html += """
                <div class="plot">
                    <img src="../plots/embedding_speedup.png" alt="Embedding Speedup">
                    <p>GPU acceleration speedup for embedding generation</p>
                </div>
"""
            
            # Add comparison if available
            if (report.get("comparison") and 
                "embedding" in report["comparison"].get("benchmarks", {})):
                
                comparison = report["comparison"]["benchmarks"]["embedding"]
                
                html += """
                <div class="comparison">
                    <h4>Comparison with Previous Benchmark</h4>
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Current</th>
                            <th>Previous</th>
                            <th>Change</th>
                        </tr>
"""
                
                for impl, data in comparison.get("throughput_change", {}).items():
                    percent_change = data.get("percent_change", 0)
                    change_class = "positive" if percent_change >= 0 else "negative"
                    
                    html += f"""
                        <tr>
                            <td>{impl.upper()} Throughput</td>
                            <td>{data.get('current', 0):.2f} items/s</td>
                            <td>{data.get('previous', 0):.2f} items/s</td>
                            <td class="{change_class}">{percent_change:.2f}%</td>
                        </tr>
"""
                
                for impl, data in comparison.get("speedup_change", {}).items():
                    percent_change = data.get("percent_change", 0)
                    change_class = "positive" if percent_change >= 0 else "negative"
                    
                    html += f"""
                        <tr>
                            <td>{impl.replace('_', ' ').upper()} Speedup</td>
                            <td>{data.get('current', 0):.2f}x</td>
                            <td>{data.get('previous', 0):.2f}x</td>
                            <td class="{change_class}">{percent_change:.2f}%</td>
                        </tr>
"""
                
                html += """
                    </table>
                </div>
"""
            
            html += """
            </div>
"""
        
        # Add TensorRT benchmark results
        if "tensorrt" in report["benchmarks"] and "error" not in report["benchmarks"]["tensorrt"]:
            tensorrt = report["benchmarks"]["tensorrt"]
            
            html += """
            <div class="card">
                <h3>TensorRT Optimization Performance</h3>
"""
            
            # Add plots if available
            plot_path = os.path.join(self.output_dir, "plots", "tensorrt_inference_time.png")
            if os.path.exists(plot_path):
                html += """
                <div class="plot">
                    <img src="../plots/tensorrt_inference_time.png" alt="TensorRT Inference Time">
                    <p>PyTorch vs TensorRT inference time comparison</p>
                </div>
"""
            
            plot_path = os.path.join(self.output_dir, "plots", "tensorrt_speedup.png")
            if os.path.exists(plot_path):
                html += """
                <div class="plot">
                    <img src="../plots/tensorrt_speedup.png" alt="TensorRT Speedup">
                    <p>TensorRT optimization speedup over PyTorch</p>
                </div>
"""
            
            plot_path = os.path.join(self.output_dir, "plots", "tensorrt_throughput.png")
            if os.path.exists(plot_path):
                html += """
                <div class="plot">
                    <img src="../plots/tensorrt_throughput.png" alt="TensorRT Throughput">
                    <p>PyTorch vs TensorRT throughput comparison</p>
                </div>
"""
            
            # Add overall statistics
            if "overall" in tensorrt:
                html += """
                <div class="card">
                    <h4>TensorRT Overall Performance</h4>
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                        <tr>
                            <td>Mean Speedup</td>
                            <td>""" + f"{tensorrt['overall']['mean_speedup']:.2f}x" + """</td>
                        </tr>
                        <tr>
                            <td>Median Speedup</td>
                            <td>""" + f"{tensorrt['overall']['median_speedup']:.2f}x" + """</td>
                        </tr>
                        <tr>
                            <td>Min Speedup</td>
                            <td>""" + f"{tensorrt['overall']['min_speedup']:.2f}x" + """</td>
                        </tr>
                        <tr>
                            <td>Max Speedup</td>
                            <td>""" + f"{tensorrt['overall']['max_speedup']:.2f}x" + """</td>
                        </tr>
                    </table>
                </div>
"""
            
            # Add comparison if available
            if (report.get("comparison") and 
                "tensorrt" in report["comparison"].get("benchmarks", {})):
                
                comparison = report["comparison"]["benchmarks"]["tensorrt"]
                
                if "speedup_change" in comparison:
                    data = comparison["speedup_change"]
                    percent_change = data.get("percent_change", 0)
                    change_class = "positive" if percent_change >= 0 else "negative"
                    
                    html += """
                <div class="comparison">
                    <h4>Comparison with Previous Benchmark</h4>
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Current</th>
                            <th>Previous</th>
                            <th>Change</th>
                        </tr>
                        <tr>
                            <td>TensorRT Speedup</td>
                            <td>""" + f"{data.get('current', 0):.2f}x" + """</td>
                            <td>""" + f"{data.get('previous', 0):.2f}x" + """</td>
                            <td class=\"""" + change_class + """\">""" + f"{percent_change:.2f}%" + """</td>
                        </tr>
                    </table>
                </div>
"""
            
            html += """
            </div>
"""
        
        # Close HTML tags
        html += """
        </div>
    </div>
</body>
</html>
"""
        
        return html
    
    def _generate_markdown_report(self, report: Dict) -> str:
        """Generate a Markdown report from benchmark data."""
        md = f"""# LangChain SAP HANA Benchmark Report

Generated: {report["timestamp"]}

## System Information

- **API URL**: {report["api_url"]}
- **API Version**: {report["system_info"].get("api", {}).get("version", "Unknown")}

"""
        
        # Add GPU info
        if "gpu" in report["system_info"]:
            gpu_info = report["system_info"]["gpu"]
            md += f"""### GPU Information

- **GPU Available**: {gpu_info.get("available", False)}
- **GPU Count**: {gpu_info.get("count", 0)}
- **CUDA Version**: {gpu_info.get("cuda_version", "Unknown")}
- **TensorRT Available**: {gpu_info.get("tensorrt_available", False)}

"""
            
            # Add GPU device info
            if gpu_info.get("devices"):
                md += "### GPU Devices\n\n"
                md += "| Device | Model | Memory (MB) | Compute Capability |\n"
                md += "|--------|-------|-------------|-------------------|\n"
                
                for i, device in enumerate(gpu_info["devices"]):
                    md += f"| {i} | {device.get('name', 'Unknown')} | {device.get('memory_total_mb', 0)} | {device.get('compute_capability', 'Unknown')} |\n"
                
                md += "\n"
        
        # Add benchmark results
        md += "## Benchmark Results\n\n"
        
        # Add embedding benchmark results
        if "embedding" in report["benchmarks"] and "error" not in report["benchmarks"]["embedding"]:
            embedding = report["benchmarks"]["embedding"]
            
            md += "### Embedding Generation Performance\n\n"
            
            # Add plots if available
            plot_path = os.path.join("plots", "embedding_throughput.png")
            rel_path = os.path.join("..", plot_path)
            if os.path.exists(os.path.join(self.output_dir, plot_path)):
                md += f"![Embedding Throughput]({rel_path})\n\n"
            
            plot_path = os.path.join("plots", "embedding_speedup.png")
            rel_path = os.path.join("..", plot_path)
            if os.path.exists(os.path.join(self.output_dir, plot_path)):
                md += f"![Embedding Speedup]({rel_path})\n\n"
            
            # Add comparison if available
            if (report.get("comparison") and 
                "embedding" in report["comparison"].get("benchmarks", {})):
                
                comparison = report["comparison"]["benchmarks"]["embedding"]
                
                md += "#### Comparison with Previous Benchmark\n\n"
                
                if comparison.get("throughput_change"):
                    md += "**Throughput Change:**\n\n"
                    md += "| Implementation | Current (items/s) | Previous (items/s) | Change (%) |\n"
                    md += "|---------------|------------------|-------------------|------------|\n"
                    
                    for impl, data in comparison["throughput_change"].items():
                        percent_change = data.get("percent_change", 0)
                        change_str = f"{percent_change:.2f}%"
                        if percent_change >= 0:
                            change_str = f"✅ {change_str}"
                        else:
                            change_str = f"❌ {change_str}"
                        
                        md += f"| {impl.upper()} | {data.get('current', 0):.2f} | {data.get('previous', 0):.2f} | {change_str} |\n"
                    
                    md += "\n"
                
                if comparison.get("speedup_change"):
                    md += "**Speedup Change:**\n\n"
                    md += "| Comparison | Current (x) | Previous (x) | Change (%) |\n"
                    md += "|------------|-------------|--------------|------------|\n"
                    
                    for impl, data in comparison["speedup_change"].items():
                        percent_change = data.get("percent_change", 0)
                        change_str = f"{percent_change:.2f}%"
                        if percent_change >= 0:
                            change_str = f"✅ {change_str}"
                        else:
                            change_str = f"❌ {change_str}"
                        
                        md += f"| {impl.replace('_', ' ').upper()} | {data.get('current', 0):.2f}x | {data.get('previous', 0):.2f}x | {change_str} |\n"
                    
                    md += "\n"
        
        # Add TensorRT benchmark results
        if "tensorrt" in report["benchmarks"] and "error" not in report["benchmarks"]["tensorrt"]:
            tensorrt = report["benchmarks"]["tensorrt"]
            
            md += "### TensorRT Optimization Performance\n\n"
            
            # Add plots if available
            plot_path = os.path.join("plots", "tensorrt_inference_time.png")
            rel_path = os.path.join("..", plot_path)
            if os.path.exists(os.path.join(self.output_dir, plot_path)):
                md += f"![TensorRT Inference Time]({rel_path})\n\n"
            
            plot_path = os.path.join("plots", "tensorrt_speedup.png")
            rel_path = os.path.join("..", plot_path)
            if os.path.exists(os.path.join(self.output_dir, plot_path)):
                md += f"![TensorRT Speedup]({rel_path})\n\n"
            
            plot_path = os.path.join("plots", "tensorrt_throughput.png")
            rel_path = os.path.join("..", plot_path)
            if os.path.exists(os.path.join(self.output_dir, plot_path)):
                md += f"![TensorRT Throughput]({rel_path})\n\n"
            
            # Add overall statistics
            if "overall" in tensorrt:
                md += "#### TensorRT Overall Performance\n\n"
                md += f"- **Mean Speedup**: {tensorrt['overall']['mean_speedup']:.2f}x\n"
                md += f"- **Median Speedup**: {tensorrt['overall']['median_speedup']:.2f}x\n"
                md += f"- **Min Speedup**: {tensorrt['overall']['min_speedup']:.2f}x\n"
                md += f"- **Max Speedup**: {tensorrt['overall']['max_speedup']:.2f}x\n\n"
            
            # Add comparison if available
            if (report.get("comparison") and 
                "tensorrt" in report["comparison"].get("benchmarks", {})):
                
                comparison = report["comparison"]["benchmarks"]["tensorrt"]
                
                if "speedup_change" in comparison:
                    data = comparison["speedup_change"]
                    percent_change = data.get("percent_change", 0)
                    change_str = f"{percent_change:.2f}%"
                    if percent_change >= 0:
                        change_str = f"✅ {change_str}"
                    else:
                        change_str = f"❌ {change_str}"
                    
                    md += "#### Comparison with Previous Benchmark\n\n"
                    md += "| Metric | Current | Previous | Change |\n"
                    md += "|--------|---------|----------|--------|\n"
                    md += f"| TensorRT Speedup | {data.get('current', 0):.2f}x | {data.get('previous', 0):.2f}x | {change_str} |\n\n"
        
        return md


def load_comparison_data(filepath: str) -> Dict:
    """Load comparison data from a previous benchmark report."""
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load comparison data: {str(e)}")
        return {}


def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info(f"Starting benchmarks against {args.api_url}")
    
    # Create benchmark runner
    runner = BenchmarkRunner(
        api_url=args.api_url,
        api_key=args.api_key,
        output_dir=args.output_dir,
        timeout=args.timeout,
        iterations=args.iterations,
        skip_plots=args.skip_plots,
    )
    
    # Load comparison data if specified
    comparison_data = None
    if args.compare:
        comparison_data = load_comparison_data(args.compare)
    
    # Run benchmarks
    if not args.skip_embedding:
        runner.run_embedding_benchmark()
    
    if not args.skip_search:
        runner.run_search_benchmark()
    
    if not args.skip_mmr:
        runner.run_mmr_benchmark()
    
    if not args.skip_tensorrt:
        runner.run_tensorrt_benchmark()
    
    # Generate report
    json_path, html_path, md_path = runner.generate_report(comparison_data)
    
    if html_path:
        logger.info(f"HTML report generated: {html_path}")
    
    if md_path:
        logger.info(f"Markdown report generated: {md_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())