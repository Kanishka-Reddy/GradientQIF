# run_experiments.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from experiment_runner import run_mnist_experiment, run_validation_experiment


def main():
    """Run all experiments and analyze results."""
    print("Starting QIF Neural Network Experiments")
    print("======================================")

    # Create results directory
    os.makedirs('results', exist_ok=True)

    # First, validate theoretical predictions
    print("\n1. Validating Theoretical Predictions")
    print("------------------------------------")
    validation_results = run_validation_experiment()

    # Run experiments on different datasets
    print("\n2. MNIST Classification Experiment")
    print("--------------------------------")
    mnist_results = run_mnist_experiment()

    # Summarize results
    print("\nResults Summary:")
    print("===============")

    # Theoretical validation
    print(f"\nTheoretical Validation:")
    print(f"- Predicted critical point: μ ≈ {-0.125:.4f}")
    print(f"- Numerically found critical point: μ ≈ {validation_results['critical_mu']:.4f}")
    print(f"- Mean absolute error between analytical and numerical firing rates: "
          f"{np.mean(np.abs(np.array(validation_results['analytical_rates']) - np.array(validation_results['numerical_rates']))):.6f}")

    # MNIST results
    derived_history, standard_history = mnist_results[0], mnist_results[1]

    print(f"\nMNIST Classification:")
    print(f"- Final test accuracy with derived gradient: {derived_history['test_acc'][-1]:.2f}%")
    print(f"- Final test accuracy with standard gradient: {standard_history['test_acc'][-1]:.2f}%")
    print(f"- Improvement: {derived_history['test_acc'][-1] - standard_history['test_acc'][-1]:.2f}%")

    print("\nExperiments completed. Results saved to 'results/' directory.")


if __name__ == "__main__":
    main()