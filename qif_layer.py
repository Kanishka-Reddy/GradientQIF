# qif_layer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from surrogate_grad import DerivedQIFSurrogate, PopulationTracker


class QIFLayer(nn.Module):
    """
    Layer of QIF neurons with our derived surrogate gradient.

    Implements the QIF dynamics with population-tracking for
    surrogate gradient calculation.
    """

    def __init__(self, input_size, output_size, tau_m=10.0, dt=1.0,
                 v_th=20.0, v_reset=-20.0, surrogate_width=2.0,
                 use_derived_grad=True):
        """
        Initialize a layer of QIF neurons.

        Args:
            input_size (int): Number of input features
            output_size (int): Number of output neurons
            tau_m (float): Membrane time constant in ms
            dt (float): Simulation time step in ms
            v_th (float): Spike threshold
            v_reset (float): Reset voltage
            surrogate_width (float): Width of surrogate window
            use_derived_grad (bool): Whether to use our derived gradient
        """
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.tau_m = tau_m
        self.dt = dt
        self.v_th = v_th
        self.v_reset = v_reset
        self.surrogate_width = surrogate_width
        self.use_derived_grad = use_derived_grad

        # Initialize weights and biases
        self.weight = nn.Parameter(torch.Tensor(output_size, input_size))
        self.bias = nn.Parameter(torch.Tensor(output_size))
        self.reset_parameters()

        # Initialize state variables
        self.register_buffer('v', torch.zeros(output_size))

        # Population statistics tracker
        self.pop_tracker = PopulationTracker()

    def reset_parameters(self):
        """Initialize parameters near the critical point."""
        # We'll use slightly tuned Kaiming initialization
        # with a scale that puts neurons near the optimal regime
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

        # Initialize biases to place neurons near the "critical point"
        # For QIF, this is around μ ≈ -σ/4 where gain is maximal
        fan_in = self.input_size
        sigma_estimate = 1.0 / np.sqrt(fan_in)  # Rough estimate of input variance
        optimal_mu = -sigma_estimate / 4  # Critical point for QIF

        # Convert to bias
        bound = 1 / np.sqrt(fan_in)
        self.bias.data.uniform_(-bound, bound)
        self.bias.data += optimal_mu

    def forward(self, input_spikes, states=None):
        """
        Forward pass for a single time step.

        Args:
            input_spikes (torch.Tensor): Input spike tensor [batch_size, input_size]
            states (tuple, optional): Previous state (v,)

        Returns:
            tuple: (output_spikes, new_states)
                - output_spikes: Output spike tensor [batch_size, output_size]
                - new_states: New voltage states (v,)
        """
        batch_size = input_spikes.size(0)

        # Use provided state or current state
        if states is not None:
            v = states
        else:
            v = self.v.repeat(batch_size, 1)

        # Compute input current
        I = F.linear(input_spikes, self.weight, self.bias)

        # Compute voltage update
        dv = (v * v + I) * (self.dt / self.tau_m)
        v_new = v + dv

        # Track population statistics
        v_mean, v_var = self.pop_tracker.update(v_new)

        # Spike generation with appropriate surrogate gradient
        if self.use_derived_grad and self.training:
            # Use our derived surrogate gradient
            spike_out = DerivedQIFSurrogate.apply(
                v_new, self.v_th, v_mean, v_var, self.surrogate_width
            )
        else:
            # Fall back to standard fast sigmoid surrogate
            spike = (v_new >= self.v_th).float()
            if self.training:
                alpha = 10.0
                grad = alpha / (1.0 + alpha * torch.abs(v_new - self.v_th)) ** 2
                spike_out = SpikeFunction.apply(v_new, self.v_th, grad)
            else:
                spike_out = spike

        # Reset after spike
        v_new = torch.where(spike_out > 0, torch.tensor(self.v_reset, device=v_new.device), v_new)

        # Save state if not provided externally
        if states is None:
            self.v = v_new.detach().mean(0)

        return spike_out, v_new

    def reset_state(self):
        """Reset the layer state."""
        self.v.zero_()


class SpikeFunction(torch.autograd.Function):
    """Standard surrogate gradient function for spiking neurons."""

    @staticmethod
    def forward(ctx, voltage, threshold, grad):
        ctx.save_for_backward(grad)
        return (voltage >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        grad, = ctx.saved_tensors
        return grad * grad_output, None, None