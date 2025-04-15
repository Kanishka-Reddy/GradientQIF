
# critical_init.py
import torch
import torch.nn as nn
import numpy as np


def qif_critical_init(module, mean_input_scale=-0.25, noise_scale=1.0):
    """
    Initialize QIF networks near the critical point for optimal learning.

    This places neurons near μ ≈ -σ/4 where the gain is maximal.

    Args:
        module (nn.Module): Module to initialize
        mean_input_scale (float): Scale factor for mean input (-0.25 is optimal)
        noise_scale (float): Scale factor for input variance

    Returns:
        nn.Module: Initialized module
    """
    for name, param in module.named_parameters():
        if 'weight' in name:
            n = param.size(1)  # fan-in
            # Standard initialization
            param.data.normal_(0, np.sqrt(2.0 / n))

            # Store fan-in for bias initialization
            if hasattr(module, name.replace('weight', 'bias')):
                module._qif_fan_in = n

        elif 'bias' in name:
            # Initialize to place neurons near critical point
            if hasattr(module, '_qif_fan_in'):
                fan_in = module._qif_fan_in
                # Estimate input variance based on fan-in
                sigma_est = np.sqrt(2.0 / fan_in) * noise_scale
                # Set bias to μ ≈ -σ/4
                optimal_mu = mean_input_scale * sigma_est
                param.data.fill_(optimal_mu)
            else:
                # Default initialization if fan-in not available
                param.data.fill_(mean_input_scale)

    return module