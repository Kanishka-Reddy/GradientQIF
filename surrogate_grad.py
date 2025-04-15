# surrogate_grad.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DerivedQIFSurrogate(torch.autograd.Function):
    """
    Surrogate gradient function derived from mean-field theory of QIF neurons.

    This implements the derived form:
    g(V) ∝ ((V-μ_V)/σ_V²)·window(V-V_th)

    Which accounts for population statistics in the gradient.
    """

    @staticmethod
    def forward(ctx, voltage, threshold, v_mean, v_var, window_width=2.0):
        """
        Forward pass: Heaviside step function.

        Args:
            ctx: Context object
            voltage: Membrane potential
            threshold: Threshold voltage
            v_mean: Population mean voltage
            v_var: Population voltage variance
            window_width: Width of the surrogate window

        Returns:
            Spike output (0 or 1)
        """
        # Save voltage and other parameters for backward
        ctx.save_for_backward(voltage, threshold, v_mean, v_var,
                              torch.tensor(window_width))

        # Heaviside function: 1 if v >= threshold, 0 otherwise
        return (voltage >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: Apply derived surrogate gradient.

        Args:
            ctx: Context object
            grad_output: Gradient from downstream layers

        Returns:
            Gradient for each input
        """
        voltage, threshold, v_mean, v_var, window_width = ctx.saved_tensors

        # Centered voltage
        v_centered = voltage - v_mean

        # Create a Gaussian window centered at threshold
        window = torch.exp(-(voltage - threshold) ** 2 / (2 * window_width ** 2))

        # Combine into our derived surrogate
        # g(V) ∝ ((V-μ_V)/σ_V²)·window(V-V_th)
        grad = (v_centered / (v_var + 1e-6)) * window

        # Scale to have a reasonable magnitude (similar to fast sigmoid)
        scale_factor = 1.0 / (torch.max(torch.abs(grad)) + 1e-6)
        grad = grad * scale_factor

        return grad * grad_output, None, None, None, None


class PopulationTracker(nn.Module):
    """
    Track population statistics needed for the derived surrogate gradient.
    """

    def __init__(self, alpha=0.99):
        """
        Initialize population tracker.

        Args:
            alpha (float): Exponential moving average factor
        """
        super().__init__()
        self.alpha = alpha

        # Population statistics (as buffers to save with model)
        self.register_buffer('v_mean', torch.tensor(0.0))
        self.register_buffer('v_var', torch.tensor(1.0))
        self.register_buffer('n_updates', torch.tensor(0.0))

    def update(self, voltage_batch):
        """
        Update population statistics based on new voltage samples.

        Args:
            voltage_batch (torch.Tensor): Batch of voltage values

        Returns:
            tuple: Updated (v_mean, v_var)
        """
        if not self.training:
            return self.v_mean, self.v_var

        # Compute batch statistics
        batch_mean = torch.mean(voltage_batch)
        batch_var = torch.var(voltage_batch, unbiased=True)

        if self.n_updates < 1:
            # Initialize with first batch
            self.v_mean = batch_mean
            self.v_var = batch_var
        else:
            # Update using exponential moving average
            self.v_mean = self.alpha * self.v_mean + (1 - self.alpha) * batch_mean
            self.v_var = self.alpha * self.v_var + (1 - self.alpha) * batch_var

        self.n_updates += 1
        return self.v_mean, self.v_var