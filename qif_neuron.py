# qif_neuron.py
import numpy as np
import torch
import torch.nn as nn


class QIFNeuron(nn.Module):
    """
    Quadratic Integrate-and-Fire neuron model.

    This implements the dynamics:
    τ_m * dV/dt = V² + I(t)

    with spike generation when V → ∞ and reset to -∞

    In practice, we use numerical thresholds for spiking and reset.
    """

    def __init__(self, tau_m=10.0, v_th=20.0, v_reset=-20.0, dt=1.0, spike_grad='derived'):
        """
        Initialize a QIF neuron.

        Args:
            tau_m (float): Membrane time constant in ms
            v_th (float): Spike threshold
            v_reset (float): Reset voltage
            dt (float): Simulation time step in ms
            spike_grad (str): Surrogate gradient type ('derived', 'fast_sigmoid', 'rect', etc.)
        """
        super().__init__()
        self.tau_m = tau_m
        self.v_th = v_th
        self.v_reset = v_reset
        self.dt = dt
        self.spike_grad = spike_grad

        # State variables
        self.v = None  # Membrane potential
        self.reset()

    def reset(self):
        """Reset the neuron state."""
        self.v = torch.tensor(0.0)  # Initialize at rest

    def forward(self, I_in, states=None):
        """
        Simulate the neuron for one time step.

        Args:
            I_in (torch.Tensor): Input current
            states (tuple, optional): Previous state (v,)

        Returns:
            tuple: (spike, new_state)
                - spike: Binary spike output
                - new_state: New voltage state (v,)
        """
        # Use provided state or current state
        if states is not None:
            v = states
        else:
            v = self.v

        # Compute voltage update
        dv = (v * v + I_in) * (self.dt / self.tau_m)
        v_new = v + dv

        # Spike generation (with surrogate gradient)
        spike = self.spike_function(v_new)

        # Reset after spike
        v_new = torch.where(spike > 0, self.v_reset, v_new)

        # Update state
        self.v = v_new.detach()

        return spike, v_new

    def spike_function(self, v):
        """
        Spike generation function with surrogate gradient.

        Args:
            v (torch.Tensor): Membrane potential

        Returns:
            torch.Tensor: Spike output with appropriate gradient
        """
        # Forward: Heaviside function (1 if v >= v_th, 0 otherwise)
        spike = (v >= self.v_th).float()

        if not self.training:
            return spike

        # Use different surrogate gradients based on configuration
        if self.spike_grad == 'fast_sigmoid':
            alpha = 10.0  # Steepness
            grad = alpha / (1.0 + alpha * abs(v - self.v_th).detach()) ** 2

        elif self.spike_grad == 'rect':
            width = 1.0  # Window width
            grad = ((v - self.v_th).abs() < width / 2).float() / width

        elif self.spike_grad == 'derived':
            # Our theoretically derived surrogate - will implement later
            # Placeholder using fast sigmoid for now
            alpha = 10.0
            grad = alpha / (1.0 + alpha * abs(v - self.v_th).detach()) ** 2

        else:
            raise ValueError(f"Unknown spike gradient: {self.spike_grad}")

        # Use straight-through estimator with custom gradient
        return SpikeFunction.apply(v, self.v_th, grad, spike)


class SpikeFunction(torch.autograd.Function):
    """Custom autograd function for spike generation with surrogate gradient."""

    @staticmethod
    def forward(ctx, v, v_th, grad, spike_out=None):
        """
        Forward pass: Heaviside step function.

        Args:
            ctx: Context object
            v: Membrane potential
            v_th: Threshold voltage
            grad: Surrogate gradient at this point
            spike_out: Optional pre-computed spike output

        Returns:
            Spike output (0 or 1)
        """
        ctx.save_for_backward(grad)

        if spike_out is None:
            return (v >= v_th).float()
        return spike_out

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: Apply surrogate gradient.

        Args:
            ctx: Context object
            grad_output: Gradient from downstream layers

        Returns:
            Gradient for each input
        """
        grad, = ctx.saved_tensors
        return grad * grad_output, None, None, None