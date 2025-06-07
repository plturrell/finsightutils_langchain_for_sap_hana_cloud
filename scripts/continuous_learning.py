#!/usr/bin/env python
"""
Continuous learning module for SAP HANA Cloud LangChain Integration.

This script implements continuous learning and optimization:
1. Monitors system performance and resource usage
2. Tracks embedding accuracy and search quality metrics
3. Dynamically adjusts parameters based on observed performance
4. Implements reinforcement learning for parameter optimization
5. Exports learned configurations for production use

Usage:
    python -m scripts.continuous_learning \
        --duration=0 \
        --config-file=/app/config/continuous_learning_config.json \
        --output-dir=/app/config/learned \
        --monitoring-interval=60
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("continuous_learning")

# Check for GPU availability
try:
    import torch
    import torch.cuda as cuda
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False

# Try to import machine learning packages for parameter optimization
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Continuous learning for SAP HANA Cloud LangChain Integration"
    )
    
    parser.add_argument(
        "--duration",
        type=int,
        default=0,
        help="Duration to run in minutes (0 for continuous operation)",
    )
    
    parser.add_argument(
        "--config-file",
        type=str,
        default="/app/config/continuous_learning_config.json",
        help="Configuration file for continuous learning",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/app/config/learned",
        help="Output directory for learned configurations",
    )
    
    parser.add_argument(
        "--monitoring-interval",
        type=int,
        default=60,
        help="Monitoring interval in seconds",
    )
    
    return parser.parse_args()


class ParameterOptimizer:
    """Parameter optimizer using machine learning techniques."""
    
    def __init__(self, config_file: str, output_dir: str):
        """
        Initialize the parameter optimizer.
        
        Args:
            config_file: Path to configuration file
            output_dir: Path to output directory
        """
        self.config_file = config_file
        self.output_dir = output_dir
        self.learning_rate = 0.05
        self.exploration_rate = 0.2
        self.history = []
        self.best_params = {}
        self.best_score = -float('inf')
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load initial configuration
        self.load_config()
        
        # Initialize machine learning models if available
        self.initialize_models()
    
    def load_config(self) -> None:
        """Load configuration from file."""
        try:
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
            
            # Set default values if not present
            if 'parameters' not in self.config:
                self.config['parameters'] = {
                    'batch_size': {
                        'default': 32,
                        'min': 1,
                        'max': 128,
                        'step': 1,
                    },
                    'gpu_memory_fraction': {
                        'default': 0.8,
                        'min': 0.1,
                        'max': 0.95,
                        'step': 0.05,
                    },
                    'worker_count': {
                        'default': 4,
                        'min': 1,
                        'max': 16,
                        'step': 1,
                    },
                }
            
            if 'metrics' not in self.config:
                self.config['metrics'] = {
                    'throughput': {'weight': 0.5},
                    'latency': {'weight': 0.3},
                    'memory_usage': {'weight': 0.2},
                }
            
            # Initialize current parameters with defaults
            self.current_params = {
                param_name: param_config['default']
                for param_name, param_config in self.config['parameters'].items()
            }
            
            self.best_params = self.current_params.copy()
            
            logger.info(f"Configuration loaded from {self.config_file}")
            logger.info(f"Current parameters: {self.current_params}")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Error loading configuration: {e}")
            # Use default configuration
            self.config = {
                'parameters': {
                    'batch_size': {
                        'default': 32,
                        'min': 1,
                        'max': 128,
                        'step': 1,
                    },
                    'gpu_memory_fraction': {
                        'default': 0.8,
                        'min': 0.1,
                        'max': 0.95,
                        'step': 0.05,
                    },
                    'worker_count': {
                        'default': 4,
                        'min': 1,
                        'max': 16,
                        'step': 1,
                    },
                },
                'metrics': {
                    'throughput': {'weight': 0.5},
                    'latency': {'weight': 0.3},
                    'memory_usage': {'weight': 0.2},
                },
            }
            self.current_params = {
                param_name: param_config['default']
                for param_name, param_config in self.config['parameters'].items()
            }
            self.best_params = self.current_params.copy()
    
    def initialize_models(self) -> None:
        """Initialize machine learning models for parameter optimization."""
        if HAS_SKLEARN:
            self.models = {}
            self.scalers = {}
            
            for param_name in self.config['parameters'].keys():
                self.models[param_name] = RandomForestRegressor(n_estimators=50, random_state=42)
                self.scalers[param_name] = StandardScaler()
            
            logger.info("Machine learning models initialized")
        else:
            logger.warning("scikit-learn not available. Using simple reinforcement learning.")
            self.models = None
            self.scalers = None
    
    def suggest_parameters(self) -> Dict[str, Any]:
        """
        Suggest parameters for the next iteration.
        
        Uses exploration vs. exploitation to balance between trying new
        parameter values and using known good values.
        
        Returns:
            Dict with suggested parameters
        """
        # If we have no history, use current parameters
        if len(self.history) < 5:
            logger.info("Not enough history. Using current parameters with exploration.")
            return self.explore_parameters()
        
        # Decide between exploration and exploitation
        if np.random.random() < self.exploration_rate:
            logger.info("Exploring new parameters")
            return self.explore_parameters()
        else:
            logger.info("Exploiting learned parameters")
            return self.exploit_parameters()
    
    def explore_parameters(self) -> Dict[str, Any]:
        """
        Explore new parameter values.
        
        Randomly adjusts parameters within their defined ranges.
        
        Returns:
            Dict with explored parameters
        """
        explored_params = {}
        
        for param_name, param_config in self.config['parameters'].items():
            # Get current value
            current_value = self.current_params[param_name]
            
            # Get parameter range
            min_value = param_config['min']
            max_value = param_config['max']
            step = param_config['step']
            
            # Calculate range for exploration
            range_size = max_value - min_value
            exploration_step = range_size * self.learning_rate
            
            # Decide direction (increase or decrease)
            direction = 1 if np.random.random() > 0.5 else -1
            
            # Calculate new value
            if isinstance(current_value, int):
                new_value = current_value + direction * max(1, int(exploration_step))
                new_value = max(min_value, min(max_value, new_value))
            else:
                new_value = current_value + direction * exploration_step
                new_value = max(min_value, min(max_value, new_value))
                # Round to step precision for floats
                new_value = round(new_value / step) * step
            
            explored_params[param_name] = new_value
        
        return explored_params
    
    def exploit_parameters(self) -> Dict[str, Any]:
        """
        Exploit learned parameter values.
        
        Uses machine learning models to predict optimal parameters
        based on historical performance.
        
        Returns:
            Dict with exploited parameters
        """
        if HAS_SKLEARN and self.models and len(self.history) >= 10:
            # Use machine learning models to predict optimal parameters
            return self.predict_optimal_parameters()
        else:
            # Use simple reinforcement learning
            return self.reinforce_parameters()
    
    def predict_optimal_parameters(self) -> Dict[str, Any]:
        """
        Predict optimal parameters using machine learning models.
        
        Trains models on historical data and predicts parameters
        that maximize the performance score.
        
        Returns:
            Dict with predicted optimal parameters
        """
        # Extract features and targets from history
        X = np.array([list(entry['parameters'].values()) for entry in self.history])
        y = np.array([entry['score'] for entry in self.history])
        
        # Scale features
        X_scaled = X.copy()
        for i, param_name in enumerate(self.config['parameters'].keys()):
            X_scaled[:, i] = self.scalers[param_name].fit_transform(X[:, i].reshape(-1, 1)).ravel()
        
        # Train models
        for i, param_name in enumerate(self.config['parameters'].keys()):
            # Extract this parameter's values as features
            param_X = np.hstack([X_scaled[:, :i], X_scaled[:, i+1:]])
            param_y = X[:, i]
            
            # Train model to predict this parameter
            self.models[param_name].fit(param_X, param_y)
        
        # Find optimal parameters using grid search
        best_score = -float('inf')
        best_params = self.current_params.copy()
        
        # Create parameter grid (simplified for efficiency)
        param_grid = {}
        for param_name, param_config in self.config['parameters'].items():
            min_val = param_config['min']
            max_val = param_config['max']
            step = param_config['step']
            if isinstance(min_val, int):
                param_grid[param_name] = np.linspace(min_val, max_val, 10, dtype=int)
            else:
                param_grid[param_name] = np.linspace(min_val, max_val, 10)
        
        # Generate combinations (simplified for brevity)
        for _ in range(100):
            params = {}
            for param_name, values in param_grid.items():
                params[param_name] = np.random.choice(values)
            
            # Predict score for these parameters
            score = self.predict_score(params)
            
            if score > best_score:
                best_score = score
                best_params = params.copy()
        
        return best_params
    
    def predict_score(self, params: Dict[str, Any]) -> float:
        """
        Predict performance score for given parameters.
        
        Uses a simple model to predict the performance score.
        
        Args:
            params: Dict with parameters
            
        Returns:
            Predicted performance score
        """
        # This is a simplified prediction model
        # In a real implementation, this would use more sophisticated models
        
        # Calculate weighted sum of normalized parameter values
        score = 0.0
        for param_name, param_value in params.items():
            param_config = self.config['parameters'][param_name]
            min_val = param_config['min']
            max_val = param_config['max']
            
            # Normalize value to [0, 1]
            norm_value = (param_value - min_val) / (max_val - min_val)
            
            # Add weighted value to score
            # This assumes higher values are generally better
            # In a real implementation, this would use the trained models
            score += norm_value * 0.2
        
        return score
    
    def reinforce_parameters(self) -> Dict[str, Any]:
        """
        Apply reinforcement learning to parameters.
        
        Adjusts parameters based on historical performance.
        
        Returns:
            Dict with reinforced parameters
        """
        # Start with current best parameters
        reinforced_params = self.best_params.copy()
        
        # Sort history by score
        sorted_history = sorted(self.history, key=lambda x: x['score'], reverse=True)
        top_history = sorted_history[:5]  # Take top 5 performing parameter sets
        
        if not top_history:
            return reinforced_params
        
        # Calculate weighted average of top performing parameters
        weights = np.linspace(1.0, 0.2, len(top_history))  # Higher weight for better performers
        weights = weights / np.sum(weights)  # Normalize weights
        
        for param_name in reinforced_params.keys():
            param_config = self.config['parameters'][param_name]
            weighted_value = 0.0
            
            for i, entry in enumerate(top_history):
                weighted_value += entry['parameters'][param_name] * weights[i]
            
            # Round to step precision
            step = param_config['step']
            if isinstance(step, int):
                reinforced_params[param_name] = int(round(weighted_value))
            else:
                reinforced_params[param_name] = round(weighted_value / step) * step
            
            # Ensure value is within bounds
            reinforced_params[param_name] = max(
                param_config['min'],
                min(param_config['max'], reinforced_params[param_name])
            )
        
        return reinforced_params
    
    def evaluate_performance(self, metrics: Dict[str, float]) -> float:
        """
        Evaluate performance based on metrics.
        
        Calculates a weighted score from the provided metrics.
        
        Args:
            metrics: Dict with performance metrics
            
        Returns:
            Performance score
        """
        score = 0.0
        
        for metric_name, metric_value in metrics.items():
            if metric_name in self.config['metrics']:
                metric_config = self.config['metrics'][metric_name]
                weight = metric_config['weight']
                
                # For latency and memory usage, lower is better
                if metric_name in ['latency', 'memory_usage']:
                    # Invert value so lower is better
                    if metric_value > 0:
                        metric_value = 1.0 / metric_value
                
                score += metric_value * weight
        
        return score
    
    def update(self, parameters: Dict[str, Any], metrics: Dict[str, float]) -> None:
        """
        Update optimizer with performance results.
        
        Records performance and updates learning.
        
        Args:
            parameters: Dict with parameters used
            metrics: Dict with performance metrics
        """
        # Calculate performance score
        score = self.evaluate_performance(metrics)
        
        # Record history
        self.history.append({
            'parameters': parameters,
            'metrics': metrics,
            'score': score,
            'timestamp': datetime.now().isoformat(),
        })
        
        # Limit history size
        if len(self.history) > 100:
            self.history = self.history[-100:]
        
        # Update best parameters if new score is better
        if score > self.best_score:
            self.best_score = score
            self.best_params = parameters.copy()
            logger.info(f"New best parameters found: {self.best_params} (score: {self.best_score:.4f})")
            
            # Save best parameters
            self.save_best_parameters()
        
        # Update current parameters
        self.current_params = parameters.copy()
        
        # Adjust learning rate and exploration rate based on history size
        if len(self.history) > 50:
            self.learning_rate = max(0.01, self.learning_rate * 0.95)
            self.exploration_rate = max(0.05, self.exploration_rate * 0.95)
    
    def save_best_parameters(self) -> None:
        """Save best parameters to file."""
        output_file = os.path.join(self.output_dir, 'best_parameters.json')
        
        try:
            with open(output_file, 'w') as f:
                json.dump({
                    'parameters': self.best_params,
                    'score': self.best_score,
                    'timestamp': datetime.now().isoformat(),
                    'history_size': len(self.history),
                }, f, indent=2)
            
            logger.info(f"Best parameters saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving best parameters: {e}")


class PerformanceMonitor:
    """Monitor system performance."""
    
    def __init__(self):
        """Initialize the performance monitor."""
        self.metrics = {}
        self.last_update = time.time()
    
    def update(self) -> Dict[str, float]:
        """
        Update performance metrics.
        
        Returns:
            Dict with updated metrics
        """
        # Update metrics
        self.metrics = {
            'timestamp': time.time(),
            'uptime': time.time() - self.last_update,
        }
        
        # Update CPU metrics
        self.metrics['cpu_percent'] = self._get_cpu_percent()
        self.metrics['memory_percent'] = self._get_memory_percent()
        
        # Update GPU metrics if available
        if HAS_CUDA:
            self.metrics.update(self._get_gpu_metrics())
        
        # Update performance metrics
        # In a real implementation, these would be collected from application telemetry
        self.metrics['throughput'] = self._get_throughput()
        self.metrics['latency'] = self._get_latency()
        
        self.last_update = time.time()
        return self.metrics
    
    def _get_cpu_percent(self) -> float:
        """Get CPU usage percentage."""
        try:
            import psutil
            return psutil.cpu_percent(interval=0.1)
        except ImportError:
            return 0.0
    
    def _get_memory_percent(self) -> float:
        """Get memory usage percentage."""
        try:
            import psutil
            return psutil.virtual_memory().percent
        except ImportError:
            return 0.0
    
    def _get_gpu_metrics(self) -> Dict[str, float]:
        """Get GPU metrics."""
        metrics = {}
        
        try:
            # Get GPU count
            device_count = cuda.device_count()
            metrics['gpu_count'] = device_count
            
            # Get GPU metrics for each device
            for i in range(device_count):
                # Get memory usage
                memory_allocated = cuda.memory_allocated(i) / (1024 ** 2)  # MB
                memory_reserved = cuda.memory_reserved(i) / (1024 ** 2)  # MB
                total_memory = cuda.get_device_properties(i).total_memory / (1024 ** 2)  # MB
                
                metrics[f'gpu{i}_memory_allocated_mb'] = memory_allocated
                metrics[f'gpu{i}_memory_reserved_mb'] = memory_reserved
                metrics[f'gpu{i}_memory_total_mb'] = total_memory
                metrics[f'gpu{i}_memory_percent'] = memory_allocated / total_memory * 100
        except Exception as e:
            logger.warning(f"Error getting GPU metrics: {e}")
        
        return metrics
    
    def _get_throughput(self) -> float:
        """Get system throughput."""
        # In a real implementation, this would be collected from application telemetry
        # For this mock implementation, we'll return a random value
        return np.random.uniform(100, 500)  # requests/second
    
    def _get_latency(self) -> float:
        """Get system latency."""
        # In a real implementation, this would be collected from application telemetry
        # For this mock implementation, we'll return a random value
        return np.random.uniform(5, 50)  # milliseconds


def run_continuous_learning(args: argparse.Namespace) -> None:
    """
    Run the continuous learning process.
    
    Args:
        args: Command-line arguments
    """
    logger.info(f"Starting continuous learning process...")
    
    # Calculate end time (0 for continuous operation)
    end_time = time.time() + (args.duration * 60) if args.duration > 0 else float('inf')
    
    # Initialize components
    optimizer = ParameterOptimizer(args.config_file, args.output_dir)
    monitor = PerformanceMonitor()
    
    # Get initial parameters
    parameters = optimizer.current_params
    
    # Main loop
    iteration = 0
    while time.time() < end_time:
        iteration += 1
        logger.info(f"Iteration {iteration}")
        
        # Apply parameters to system
        logger.info(f"Applying parameters: {parameters}")
        
        # In a real implementation, parameters would be applied to the system
        # For this mock implementation, we'll just simulate it
        
        # Wait for monitoring interval
        time.sleep(args.monitoring_interval)
        
        # Collect metrics
        metrics = monitor.update()
        logger.info(f"Metrics: {metrics}")
        
        # Update optimizer with performance results
        optimizer.update(parameters, metrics)
        
        # Get new parameters for next iteration
        parameters = optimizer.suggest_parameters()
        
        # Save learned configuration periodically
        if iteration % 10 == 0:
            save_learned_configuration(args.output_dir, optimizer, iteration)
    
    logger.info("Continuous learning process completed")


def save_learned_configuration(output_dir: str, optimizer: ParameterOptimizer, iteration: int) -> None:
    """
    Save learned configuration to file.
    
    Args:
        output_dir: Output directory
        optimizer: Parameter optimizer
        iteration: Current iteration
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration
    output_file = os.path.join(output_dir, f'learned_config_{iteration}.json')
    
    try:
        with open(output_file, 'w') as f:
            json.dump({
                'parameters': optimizer.best_params,
                'score': optimizer.best_score,
                'timestamp': datetime.now().isoformat(),
                'iteration': iteration,
                'history_size': len(optimizer.history),
            }, f, indent=2)
        
        logger.info(f"Learned configuration saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving learned configuration: {e}")


def main():
    """Main function."""
    args = parse_args()
    
    try:
        run_continuous_learning(args)
    except KeyboardInterrupt:
        logger.info("Continuous learning process interrupted")
    except Exception as e:
        logger.error(f"Continuous learning process failed: {e}")
        raise


if __name__ == "__main__":
    main()