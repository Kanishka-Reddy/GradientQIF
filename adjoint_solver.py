# adjoint_solver.py
import numpy as np
import torch
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


class AdjointFokkerPlanckQIF:
    """
    Solver for the adjoint Fokker-Planck equation of QIF neurons.

    Implements the backward equation:
    -∂ψ/∂t = (V² + I₀ + μ(t))·∂ψ/∂V + σ²(t)/2·∂²ψ/∂V² + δC/δP(V,t)

    This propagates loss gradients backward in time for QIF populations.
    """

    def __init__(self, v_min=-10.0, v_max=10.0, n_points=1000, dt=0.1):
        """
        Initialize the adjoint Fokker-Planck solver.

        Args:
            v_min (float): Minimum voltage for discretization
            v_max (float): Maximum voltage for discretization
            n_points (int): Number of voltage discretization points
            dt (float): Time step for integration
        """
        self.v_min = v_min
        self.v_max = v_max
        self.n_points = n_points
        self.dt = dt

        # Discretize voltage space
        self.v_grid = np.linspace(v_min, v_max, n_points)
        self.dv = (v_max - v_min) / (n_points - 1)

        # Initialize operators
        self._init_operators()

    def _init_operators(self):
        """Initialize finite difference operators for PDE solving."""
        n = self.n_points

        # First derivative operator (central difference)
        self.D1 = np.zeros((n, n))
        for i in range(1, n - 1):
            self.D1[i, i - 1] = -0.5 / self.dv
            self.D1[i, i + 1] = 0.5 / self.dv
        # One-sided differences at boundaries
        self.D1[0, 0] = -1.0 / self.dv
        self.D1[0, 1] = 1.0 / self.dv
        self.D1[n - 1, n - 2] = -1.0 / self.dv
        self.D1[n - 1, n - 1] = 1.0 / self.dv

        # Second derivative operator
        self.D2 = np.zeros((n, n))
        for i in range(1, n - 1):
            self.D2[i, i - 1] = 1.0 / self.dv ** 2
            self.D2[i, i] = -2.0 / self.dv ** 2
            self.D2[i, i + 1] = 1.0 / self.dv ** 2
        # One-sided differences at boundaries
        self.D2[0, 0] = 2.0 / self.dv ** 2
        self.D2[0, 1] = -2.0 / self.dv ** 2
        self.D2[n - 1, n - 2] = 1.0 / self.dv ** 2
        self.D2[n - 1, n - 1] = -1.0 / self.dv ** 2

    def rhs(self, t, psi, mu, sigma, loss_grad, p_history, t_history):
        """
        Right-hand side of the adjoint Fokker-Planck equation.

        Args:
            t (float): Current time
            psi (ndarray): Adjoint variable
            mu (float or callable): Mean input current
            sigma (float or callable): Noise standard deviation
            loss_grad (callable): ∂L/∂ν at time t
            p_history (list): History of probability densities
            t_history (list): Corresponding time points

        Returns:
            ndarray: Time derivative of adjoint variable (negative for backward integration)
        """
        # Get current values of mu and sigma
        mu_t = mu(t) if callable(mu) else mu
        sigma_t = sigma(t) if callable(sigma) else sigma

        # Interpolate probability density at current time
        p_t = np.zeros_like(psi)
        if len(t_history) > 0:
            # Find closest time point in history
            idx = np.argmin(np.abs(np.array(t_history) - t))
            p_t = p_history[idx]

        # Calculate the loss gradient with respect to firing rate
        dl_dnu = loss_grad(t)

        # Calculate the source term: δL/δP(V,t)
        # For QIF, the firing rate depends on the flux at v_max
        source = np.zeros_like(psi)

        # The loss gradient injects at the boundary where firing rate is calculated
        # This is a simplified version - a more accurate implementation would use
        # the full δL/δP expression based on the flux definition
        source[-1] = dl_dnu * (self.v_grid[-1] ** 2 + mu_t)
        source[-2] = -dl_dnu * (0.5 * sigma_t ** 2 / self.dv)

        # Compute the advection term: (V² + μ)·∂ψ/∂V
        advection = np.zeros_like(psi)
        for i in range(self.n_points):
            advection_coef = self.v_grid[i] ** 2 + mu_t
            advection += advection_coef * self.D1[i, :] @ psi

        # Compute the diffusion term: σ²/2·∂²ψ/∂V²
        diffusion = 0.5 * sigma_t ** 2 * self.D2 @ psi

        # Combine terms (negative for backward integration)
        dpsi_dt = -(advection + diffusion + source)

        # Apply boundary conditions
        dpsi_dt[0] = 0
        dpsi_dt[-1] = 0

        return dpsi_dt

    def solve_backward(self, t_span, mu, sigma, loss_grad, p_history, t_history):
        """
        Solve the adjoint equation backward in time.

        Args:
            t_span (tuple): (t_start, t_end) time span
            mu (float or callable): Mean input current
            sigma (float or callable): Noise standard deviation
            loss_grad (callable): Function returning ∂L/∂ν at given time
            p_history (list): History of probability densities
            t_history (list): Corresponding time points

        Returns:
            dict: Results containing:
                - t: Time points (reversed)
                - psi: Adjoint variable over time
                - gradient: ∫ψ(V,t)·(-∂P(V,t)/∂V)dV over time
        """
        # Initial condition for psi (zero at final time)
        psi0 = np.zeros(self.n_points)

        # We need to solve backward in time, so reverse the time span
        t_span_reversed = (t_span[1], t_span[0])

        # Solve the adjoint PDE backward in time
        sol = solve_ivp(
            lambda t, psi: self.rhs(t, psi, mu, sigma, loss_grad, p_history, t_history),
            t_span_reversed,
            psi0,
            method='RK45',
            t_eval=np.arange(t_span[1], t_span[0], -self.dt)
        )

        # Calculate the gradient term at each time step
        # ∫ψ(V,t)·(-∂P(V,t)/∂V)dV
        gradient = np.zeros(len(sol.t))
        for i, t in enumerate(sol.t):
            psi = sol.y[:, i]

            # Find closest probability density in history
            idx = np.argmin(np.abs(np.array(t_history) - t))
            p = p_history[idx]

            # Calculate -∂P/∂V
            dp_dv = -np.dot(self.D1, p)

            # Calculate the gradient integral
            gradient[i] = np.sum(psi * dp_dv) * self.dv

        # Reverse the time order back to forward for consistency
        return {
            't': np.flip(sol.t),
            'psi': np.flip(sol.y, axis=1),
            'gradient': np.flip(gradient)
        }

    def local_approximation(self, v_mean, v_var, v_th=5.0, window_width=2.0):
        """
        Generate a local approximation of the adjoint solution.

        This implements the approximation:
        ψ(V) ≈ ((V-μ_V)/σ_V²)·window(V-V_th)

        Args:
            v_mean (float): Mean voltage
            v_var (float): Voltage variance
            v_th (float): Threshold voltage
            window_width (float): Width of the window function

        Returns:
            ndarray: Approximated adjoint function
        """
        # Create a window function centered at threshold
        window = np.exp(-(self.v_grid - v_th) ** 2 / (2 * window_width ** 2))

        # Create the voltage-dependent factor
        v_factor = (self.v_grid - v_mean) / v_var

        # Combined approximation
        psi_approx = v_factor * window

        # Normalize
        psi_approx = psi_approx / np.max(np.abs(psi_approx))

        return psi_approx

    def plot_comparison(self, psi_exact, psi_approx, title="Adjoint Comparison"):
        """
        Plot comparison between exact and approximated adjoint solutions.

        Args:
            psi_exact (ndarray): Exact adjoint solution
            psi_approx (ndarray): Approximated adjoint solution
            title (str): Plot title
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.v_grid, psi_exact, 'b-', label='Exact Solution')
        plt.plot(self.v_grid, psi_approx, 'r--', label='Approximation')
        plt.xlabel('Membrane Potential (V)')
        plt.ylabel('Adjoint Variable (ψ)')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()