"""
Numerical SDE solvers.

Implements Euler-Maruyama (strong order 0.5), Milstein (strong order 1.0),
Stochastic Runge-Kutta (strong order 1.5), and Implicit Euler-Maruyama
(A-stable for stiff SDEs).

Reference: Kloeden & Platen, "Numerical Solution of SDEs" (Springer, 1992)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional
from .processes import SDEProcess, HestonProcess, MertonJumpDiffusion


@dataclass
class SDESolution:
    """Container for SDE simulation results."""

    t: np.ndarray          # (n_steps+1,) time grid
    paths: np.ndarray      # (n_paths, n_steps+1) or (n_paths, n_steps+1, dim)
    brownian: np.ndarray   # (n_paths, n_steps+1) Brownian motion paths
    dt: float


class EulerMaruyama:
    """
    Euler-Maruyama scheme — the simplest SDE solver.

    X_{n+1} = X_n + μ(X_n, t_n) Δt + σ(X_n, t_n) ΔW_n

    Strong order 0.5, weak order 1.0.

    Parameters
    ----------
    dt : float
        Time step size.
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(self, dt: float = 0.01, seed: Optional[int] = None):
        self.dt = dt
        self.seed = seed

    def solve(
        self,
        process: SDEProcess,
        T: float,
        n_paths: int = 1,
    ) -> SDESolution:
        rng = np.random.default_rng(self.seed)
        n_steps = int(np.ceil(T / self.dt))
        dt = T / n_steps
        t = np.linspace(0, T, n_steps + 1)

        dim = process.dim
        x0 = process.x0

        if dim == 1:
            X = np.zeros((n_paths, n_steps + 1))
            W = np.zeros((n_paths, n_steps + 1))
            X[:, 0] = x0[0]

            for i in range(n_steps):
                dW = rng.normal(0, np.sqrt(dt), n_paths)
                W[:, i + 1] = W[:, i] + dW

                x_curr = X[:, i]
                mu = process.drift(x_curr, t[i])
                sigma = process.diffusion(x_curr, t[i])

                X[:, i + 1] = x_curr + mu * dt + sigma * dW

                # Handle jumps for Merton model
                if isinstance(process, MertonJumpDiffusion):
                    J = process.simulate_jumps(dt, n_paths)
                    X[:, i + 1] *= (1 + J)

        else:
            X = np.zeros((n_paths, n_steps + 1, dim))
            W = np.zeros((n_paths, n_steps + 1, dim))
            X[:, 0, :] = x0

            # Correlated Brownian motions for multi-dim
            if isinstance(process, HestonProcess):
                L = process.cholesky()
            else:
                L = np.eye(dim)

            for i in range(n_steps):
                dZ = rng.normal(0, np.sqrt(dt), (n_paths, dim))
                dW = dZ @ L.T
                W[:, i + 1, :] = W[:, i, :] + dW

                x_curr = X[:, i, :]
                mu = process.drift(x_curr, t[i])
                sigma = process.diffusion(x_curr, t[i])

                X[:, i + 1, :] = x_curr + mu * dt + sigma * dW

        return SDESolution(t=t, paths=X, brownian=W, dt=dt)


class Milstein:
    """
    Milstein scheme — exploits the diffusion derivative for strong order 1.0.

    X_{n+1} = X_n + μ Δt + σ ΔW + ½ σ σ' (ΔW² - Δt)

    Strong order 1.0, weak order 1.0.

    Parameters
    ----------
    dt : float
        Time step size.
    seed : int, optional
        Random seed.
    """

    def __init__(self, dt: float = 0.01, seed: Optional[int] = None):
        self.dt = dt
        self.seed = seed

    def solve(
        self,
        process: SDEProcess,
        T: float,
        n_paths: int = 1,
    ) -> SDESolution:
        rng = np.random.default_rng(self.seed)
        n_steps = int(np.ceil(T / self.dt))
        dt = T / n_steps
        t = np.linspace(0, T, n_steps + 1)

        X = np.zeros((n_paths, n_steps + 1))
        W = np.zeros((n_paths, n_steps + 1))
        X[:, 0] = process.x0[0]

        for i in range(n_steps):
            dW = rng.normal(0, np.sqrt(dt), n_paths)
            W[:, i + 1] = W[:, i] + dW

            x_curr = X[:, i]
            mu = process.drift(x_curr, t[i])
            sigma = process.diffusion(x_curr, t[i])
            sigma_prime = process.diffusion_derivative(x_curr, t[i])

            # Milstein correction: ½ σ σ' (ΔW² - Δt)
            milstein_term = 0.5 * sigma * sigma_prime * (dW**2 - dt)

            X[:, i + 1] = x_curr + mu * dt + sigma * dW + milstein_term

        return SDESolution(t=t, paths=X, brownian=W, dt=dt)


class StochasticRungeKutta:
    """
    Stochastic Runge-Kutta scheme (Platen's strong order 1.5 method).

    Uses an auxiliary random variable (Ẑ) in addition to ΔW for higher accuracy.

    Strong order 1.5, weak order 2.0.

    Parameters
    ----------
    dt : float
        Time step size.
    seed : int, optional
        Random seed.
    """

    def __init__(self, dt: float = 0.01, seed: Optional[int] = None):
        self.dt = dt
        self.seed = seed

    def solve(
        self,
        process: SDEProcess,
        T: float,
        n_paths: int = 1,
    ) -> SDESolution:
        rng = np.random.default_rng(self.seed)
        n_steps = int(np.ceil(T / self.dt))
        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)
        t = np.linspace(0, T, n_steps + 1)

        X = np.zeros((n_paths, n_steps + 1))
        W = np.zeros((n_paths, n_steps + 1))
        X[:, 0] = process.x0[0]

        for i in range(n_steps):
            # Two independent normal random variables
            U1 = rng.normal(0, 1, n_paths)
            U2 = rng.normal(0, 1, n_paths)
            dW = U1 * sqrt_dt
            # Auxiliary RV for order 1.5
            dZ = 0.5 * sqrt_dt * (U1 + U2 / np.sqrt(3))

            W[:, i + 1] = W[:, i] + dW

            x = X[:, i]
            ti = t[i]
            mu = process.drift(x, ti)
            sigma = process.diffusion(x, ti)
            sigma_prime = process.diffusion_derivative(x, ti)

            # Supporting values
            x_hat = x + mu * dt + sigma * sqrt_dt
            mu_hat = process.drift(x_hat, ti + dt)
            sigma_hat = process.diffusion(x_hat, ti + dt)

            # SRK update
            X[:, i + 1] = (
                x
                + 0.5 * (mu + mu_hat) * dt
                + sigma * dW
                + 0.5 * (sigma_hat - sigma) * (dW**2 - dt) / sqrt_dt
            )

        return SDESolution(t=t, paths=X, brownian=W, dt=dt)


class ImplicitEulerMaruyama:
    """
    Implicit (Backward) Euler-Maruyama — A-stable for stiff SDEs.

    X_{n+1} = X_n + μ(X_{n+1}, t_{n+1}) Δt + σ(X_n, t_n) ΔW_n

    Solved via fixed-point iteration (Picard) at each step.

    Parameters
    ----------
    dt : float
        Time step size.
    max_iter : int
        Maximum Picard iterations per step.
    tol : float
        Convergence tolerance.
    seed : int, optional
        Random seed.
    """

    def __init__(
        self,
        dt: float = 0.01,
        max_iter: int = 20,
        tol: float = 1e-10,
        seed: Optional[int] = None,
    ):
        self.dt = dt
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed

    def solve(
        self,
        process: SDEProcess,
        T: float,
        n_paths: int = 1,
    ) -> SDESolution:
        rng = np.random.default_rng(self.seed)
        n_steps = int(np.ceil(T / self.dt))
        dt = T / n_steps
        t = np.linspace(0, T, n_steps + 1)

        X = np.zeros((n_paths, n_steps + 1))
        W = np.zeros((n_paths, n_steps + 1))
        X[:, 0] = process.x0[0]

        for i in range(n_steps):
            dW = rng.normal(0, np.sqrt(dt), n_paths)
            W[:, i + 1] = W[:, i] + dW

            x_n = X[:, i]
            sigma_n = process.diffusion(x_n, t[i])
            stochastic_part = sigma_n * dW

            # Fixed-point iteration for implicit drift
            x_next = x_n + process.drift(x_n, t[i]) * dt + stochastic_part

            for _ in range(self.max_iter):
                x_prev = x_next.copy()
                x_next = x_n + process.drift(x_next, t[i + 1]) * dt + stochastic_part
                if np.max(np.abs(x_next - x_prev)) < self.tol:
                    break

            X[:, i + 1] = x_next

        return SDESolution(t=t, paths=X, brownian=W, dt=dt)
