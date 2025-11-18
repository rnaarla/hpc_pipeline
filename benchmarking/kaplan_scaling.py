#!/usr/bin/env python3
"""
Kaplan Scaling Law Analysis for LLM Training
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from typing import Dict, List, Tuple
from dataclasses import dataclass
import json
from pathlib import Path
from prometheus_client import Gauge

# Metrics
kaplan_prediction_error = Gauge("kaplan_prediction_error_percent", 
                              "Kaplan scaling prediction error")

@dataclass
class ModelConfig:
    """Model configuration for scaling analysis."""
    num_parameters: int
    batch_size: int
    learning_rate: float
    train_tokens: int
    compute_flops: float
    final_loss: float

def kaplan_compute_optimal(n_params: float) -> float:
    """Compute optimal training compute given model size (Kaplan eq. 6)."""
    return 6e4 * (n_params ** 0.8)

def kaplan_loss_prediction(n_params: float, compute: float, 
                         a: float = 0.76, b: float = -0.13, c: float = -0.09) -> float:
    """Predict loss using Kaplan scaling laws."""
    return a + b * np.log(n_params) + c * np.log(compute)

class KaplanScaling:
    """Kaplan scaling law analysis for model training."""
    
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.results_dir.mkdir(exist_ok=True)
        self.training_data: List[ModelConfig] = []
    
    def add_training_run(self, config: ModelConfig):
        """Add training run data."""
        self.training_data.append(config)
    
    def fit_scaling_laws(self) -> Tuple[float, float, float]:
        """Fit Kaplan scaling law parameters."""
        if len(self.training_data) < 3:
            raise ValueError("Need at least 3 data points to fit scaling laws")
        
        # Extract data
        n_params = np.array([d.num_parameters for d in self.training_data])
        compute = np.array([d.compute_flops for d in self.training_data])
        loss = np.array([d.final_loss for d in self.training_data])
        
        # Fit using log-transformed data
        log_n_params = np.log(n_params)
        log_compute = np.log(compute)
        
        def fit_func(X, a, b, c):
            log_n, log_c = X
            return a + b * log_n + c * log_c
        
        # Perform fit
        popt, _ = curve_fit(fit_func, (log_n_params, log_compute), loss)
        a, b, c = popt
        
        # Save parameters
        params = {
            'a': float(a),
            'b': float(b),
            'c': float(c),
            'n_datapoints': len(self.training_data)
        }
        
        with open(self.results_dir / 'kaplan_params.json', 'w') as f:
            json.dump(params, f, indent=2)
        
        return a, b, c
    
    def predict_loss(self, n_params: int, compute_flops: float) -> float:
        """Predict loss for given model size and compute."""
        a, b, c = self.fit_scaling_laws()
        return kaplan_loss_prediction(n_params, compute_flops, a, b, c)
    
    def validate_predictions(self, test_data: List[ModelConfig]) -> bool:
        """Validate predictions against test data."""
        errors = []
        
        for config in test_data:
            predicted_loss = self.predict_loss(
                config.num_parameters,
                config.compute_flops
            )
            
            error_percent = abs(predicted_loss - config.final_loss) / config.final_loss * 100
            errors.append(error_percent)
            
            kaplan_prediction_error.set(error_percent)
        
        avg_error = np.mean(errors)
        print(f"Average prediction error: {avg_error:.2f}%")
        
        return avg_error <= 10.0  # Target: ≤10% error
    
    def plot_scaling_curves(self):
        """Generate scaling curves plot."""
        a, b, c = self.fit_scaling_laws()
        
        # Create parameter and compute ranges
        n_params = np.logspace(6, 12, 100)  # 1M to 1T parameters
        compute = np.array([kaplan_compute_optimal(n) for n in n_params])
        
        # Calculate predicted loss
        loss = kaplan_loss_prediction(n_params, compute, a, b, c)
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.loglog(n_params, loss, 'b-', label='Predicted Loss')
        
        # Plot actual data points
        actual_params = [d.num_parameters for d in self.training_data]
        actual_loss = [d.final_loss for d in self.training_data]
        plt.scatter(actual_params, actual_loss, c='r', label='Actual Data')
        
        plt.xlabel('Number of Parameters')
        plt.ylabel('Loss')
        plt.title('Kaplan Scaling Law Fit')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plt.savefig(self.results_dir / 'kaplan_scaling.png')
        plt.close()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default="results/kaplan")
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = KaplanScaling(args.results_dir)
    
    # Add example training runs
    training_runs = [
        ModelConfig(1e6, 1024, 1e-4, 1e9, 1e18, 3.2),   # 1M params
        ModelConfig(10e6, 2048, 1e-4, 2e9, 2e18, 2.8),  # 10M params
        ModelConfig(100e6, 4096, 1e-4, 4e9, 4e18, 2.4), # 100M params
    ]
    
    for run in training_runs:
        analyzer.add_training_run(run)
    
    # Fit and validate
    analyzer.fit_scaling_laws()
    analyzer.plot_scaling_curves()
    
    # Validate on test data
    test_runs = [
        ModelConfig(50e6, 3072, 1e-4, 3e9, 3e18, 2.6),  # 50M params
    ]
    
    success = analyzer.validate_predictions(test_runs)
    
    if success:
        print("✅ Kaplan scaling validation passed")
    else:
        print("❌ Kaplan scaling validation failed")
        exit(1)

if __name__ == "__main__":
    main()
