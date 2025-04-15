# custom_optimizer.py
import torch
import torch.optim as optim
import numpy as np


class QIFAwareOptimizer(optim.Optimizer):
    """
    Optimizer that incorporates QIF population dynamics awareness.

    This optimizer modifies gradients based on population statistics
    and the predicted gain of QIF neurons.
    """

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, gain_scaling=True):
        """
        Initialize QIF-aware optimizer.

        Args:
            params: Model parameters
            lr (float): Learning rate
            betas (tuple): Adam beta parameters
            eps (float): Small constant for numerical stability
            weight_decay (float): Weight decay coefficient
            amsgrad (bool): Use AMSGrad variant
            gain_scaling (bool): Apply QIF gain scaling
        """
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad,
                        gain_scaling=gain_scaling)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """
        Perform a single optimization step.

        Args:
            closure (callable, optional): Reevaluates model and returns loss

        Returns:
            loss: Loss value if closure is provided
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data

                # Apply QIF gain scaling if enabled
                if group['gain_scaling'] and hasattr(p, '_qif_stats'):
                    mu = p._qif_stats['mean_input']
                    sigma = p._qif_stats['input_std']

                    # Calculate QIF gain scaling factor
                    # Gain is proportional to 1/√(√(μ²+σ²/2)+μ)
                    gain = 1.0 / np.sqrt(np.sqrt(mu ** 2 + sigma ** 2 / 2) + mu + 1e-6)

                    # Normalize gain to avoid exploding gradients
                    gain = min(gain, 5.0)  # Cap maximum gain

                    # Apply gain modulation to gradient
                    grad = grad * gain

                # Apply standard Adam updates
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if group['amsgrad']:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if group['amsgrad']:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Apply weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                # Adam update
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if group['amsgrad']:
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * np.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

