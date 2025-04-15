# fokker_planck.py
import numpy as np
import torch
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


class FokkerPlanckQIF:
    """
    Solver for the Fokker-Planck equation of QIF neurons.

    Implements numerical solutions to:
    ∂P/∂t + ∂/∂V[(V² + I₀ + μ(t))P] = σ²(t)/2·∂²P/∂V²

    with appropriate boundary conditions for the QIF model.
    """

    def __init__(self, v_min=-10.0, v_max=10.0, n_points=1000, dt=0.1):
        """
        Initialize the Fokker-Planck solver.

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

    def rhs(self, t, p, mu, sigma):
        """
        Right-hand side of the Fokker-Planck equation.

        Args:
            t (float): Current time
            p (ndarray): Probability density vector
            mu (float or callable): Mean input current (can be time-dependent)
            sigma (float or callable): Noise standard deviation (can be time-dependent)

        Returns:
            ndarray: Time derivative of probability density
        """
        # Get current values of mu and sigma if they're callables
        mu_t = mu(t) if callable(mu) else mu
        sigma_t = sigma(t) if callable(sigma) else sigma

        # Compute drift term: ∂/∂V[(V² + I₀ + μ(t))P]
        drift = np.zeros_like(p)
        for i in range(self.n_points):
            v = self.v_grid[i]
            drift_coef = v ** 2 + mu_t
            drift += -self.D1[i, :] * drift_coef * p

        # Compute diffusion term: σ²(t)/2·∂²P/∂V²
        diffusion = 0.5 * sigma_t ** 2 * np.dot(self.D2, p)

        # Combine terms
        dp_dt = drift + diffusion

        # Implement boundary conditions: probability conservation
        # (Ensure no probability leaks at boundaries)
        dp_dt[0] = 0
        dp_dt[-1] = 0

        # Normalize to conserve total probability
        p_sum = np.sum(p) * self.dv
        if abs(p_sum - 1.0) > 1e-6:
            dp_dt += (1.0 - p_sum) / self.dt

        return dp_dt

    def solve(self, p0, t_span, mu, sigma):
        """
        Solve the Fokker-Planck equation over a time span.

        Args:
            p0 (ndarray): Initial probability density
            t_span (tuple): (t_start, t_end) time span
            mu (float or callable): Mean input current
            sigma (float or callable): Noise standard deviation

        Returns:
            dict: Results containing:
                - t: Time points
                - p: Probability density over time
                - firing_rate: Population firing rate over time
        """
        # Normalize initial condition
        p0 = p0 / (np.sum(p0) * self.dv)

        # Solve the PDE
        sol = solve_ivp(
            lambda t, p: self.rhs(t, p, mu, sigma),
            t_span,
            p0,
            method='RK45',
            t_eval=np.arange(t_span[0], t_span[1], self.dt)
        )

        # Calculate firing rate at each time step
        # For QIF, firing rate is the probability flux at v_max
        firing_rate = np.zeros(len(sol.t))
        for i, t in enumerate(sol.t):
            p = sol.y[:, i]
            mu_t = mu(t) if callable(mu) else mu
            sigma_t = sigma(t) if callable(sigma) else sigma

            # Calculate the probability flux at v_max (approximating v→∞)
            v_max_flux = (self.v_grid[-1] ** 2 + mu_t) * p[-1] - 0.5 * sigma_t ** 2 * (p[-1] - p[-2]) / self.dv
            firing_rate[i] = v_max_flux

        return {
            't': sol.t,
            'p': sol.y,
            'firing_rate': firing_rate
        }

    def steady_state(self, mu, sigma, tol=1e-6, max_iter=1000):
        """
        Compute the steady-state solution of the Fokker-Planck equation.

        Args:
            mu (float): Mean input current
            sigma (float): Noise standard deviation
            tol (float): Convergence tolerance
            max_iter (int): Maximum number of iterations

        Returns:
            dict: Results containing:
                - p: Steady-state probability density
                - firing_rate: Steady-state firing rate
        """
        # Initialize with uniform distribution
        p = np.ones(self.n_points) / (self.n_points * self.dv)

        # Iterate until convergence
        for i in range(max_iter):
            dp = self.rhs(0, p, mu, sigma) * self.dt
            p_new = p + dp

            # Ensure non-negativity
            p_new = np.maximum(p_new, 0)

            # Normalize
            p_new = p_new / (np.sum(p_new) * self.dv)

            # Check convergence
            if np.max(np.abs(p_new - p)) < tol:
                p = p_new
                break

            p = p_new

        # Calculate steady-state firing rate
        v_max_flux = (self.v_grid[-1] ** 2 + mu) * p[-1] - 0.5 * sigma ** 2 * (p[-1] - p[-2]) / self.dv
        firing_rate = v_max_flux

        return {
            'p': p,
            'firing_rate': firing_rate
        }

    def analytical_firing_rate(self, mu, sigma):
        """
        Compute the analytical steady-state firing rate for QIF neurons.

        This implements the formula:
        ν(μ,σ) = (1/π)·√(√(μ² + σ²/2) + μ)

        Args:
            mu (float): Mean input current
            sigma (float): Noise standard deviation

        Returns:
            float: Analytical firing rate
        """
        if isinstance(mu, (float, int)) and isinstance(sigma, (float, int)):
            return 1.0 / (np.pi * np.sqrt(2)) * np.sqrt(np.sqrt(mu ** 2 + sigma ** 2 / 2) + mu)
        else:
            # Vectorized version
            return 1.0 / (np.pi * np.sqrt(2)) * np.sqrt(np.sqrt(mu ** 2 + sigma ** 2 / 2) + mu)

    def compute_statistics(self, p):
        """
        Compute statistics of the voltage distribution.

        Args:
            p (ndarray): Probability density function

        Returns:
            dict: Statistics containing:
                - mean: Mean voltage
                - var: Voltage variance
                - skewness: Distribution skewness
        """
        # Ensure proper normalization
        p_norm = p / (np.sum(p) * self.dv)

        # Calculate moments
        mean = np.sum(p_norm * self.v_grid) * self.dv

        # Center distribution for variance calculation
        v_centered = self.v_grid - mean
        var = np.sum(p_norm * v_centered ** 2) * self.dv

        # Calculate skewness
        skewness = np.sum(p_norm * v_centered ** 3) * self.dv / var ** (3 / 2) if var > 0 else 0

        return {
            'mean': mean,
            'var': var,
            'skewness': skewness
        }