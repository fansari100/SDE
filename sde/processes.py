"""
Stochastic process definitions.

Each process defines drift μ(X,t) and diffusion σ(X,t) for the Itô SDE:
    dX_t = μ(X_t, t) dt + σ(X_t, t) dW_t

Where analytical solutions exist, they are provided for benchmarking.
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Callable


class SDEProcess(ABC):
    """Base class for stochastic processes."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimensionality of the state vector."""
        ...

    @abstractmethod
    def drift(self, x: np.ndarray, t: float) -> np.ndarray:
        """Drift coefficient μ(x, t)."""
        ...

    @abstractmethod
    def diffusion(self, x: np.ndarray, t: float) -> np.ndarray:
        """Diffusion coefficient σ(x, t)."""
        ...

    def diffusion_derivative(self, x: np.ndarray, t: float) -> np.ndarray:
        """∂σ/∂x — required for Milstein scheme. Defaults to finite differences."""
        eps = 1e-6 * (1.0 + np.abs(x))
        return (self.diffusion(x + eps, t) - self.diffusion(x - eps, t)) / (2.0 * eps)

    @property
    @abstractmethod
    def x0(self) -> np.ndarray:
        """Initial condition."""
        ...

    def exact_solution(self, t: float, W_t: float) -> Optional[np.ndarray]:
        """Analytical solution (if available) for benchmarking."""
        return None


class GeometricBrownianMotion(SDEProcess):
    """
    Geometric Brownian Motion:  dS = μ S dt + σ S dW

    Analytical solution: S_t = S_0 exp((μ - σ²/2)t + σ W_t)
    """

    def __init__(self, mu: float = 0.05, sigma: float = 0.2, S0: float = 100.0):
        self.mu = mu
        self.sigma = sigma
        self._S0 = S0

    @property
    def dim(self) -> int:
        return 1

    @property
    def x0(self) -> np.ndarray:
        return np.array([self._S0])

    def drift(self, x: np.ndarray, t: float) -> np.ndarray:
        return self.mu * x

    def diffusion(self, x: np.ndarray, t: float) -> np.ndarray:
        return self.sigma * x

    def diffusion_derivative(self, x: np.ndarray, t: float) -> np.ndarray:
        return np.full_like(x, self.sigma)

    def exact_solution(self, t: float, W_t: float) -> np.ndarray:
        return self._S0 * np.exp((self.mu - 0.5 * self.sigma**2) * t + self.sigma * W_t)


class OrnsteinUhlenbeck(SDEProcess):
    """
    Ornstein-Uhlenbeck (mean-reverting):  dX = θ(μ - X) dt + σ dW

    Analytical solution available via variation of constants.
    """

    def __init__(self, theta: float = 1.0, mu: float = 0.0, sigma: float = 0.3, X0: float = 0.0):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self._X0 = X0

    @property
    def dim(self) -> int:
        return 1

    @property
    def x0(self) -> np.ndarray:
        return np.array([self._X0])

    def drift(self, x: np.ndarray, t: float) -> np.ndarray:
        return self.theta * (self.mu - x)

    def diffusion(self, x: np.ndarray, t: float) -> np.ndarray:
        return np.full_like(x, self.sigma)

    def diffusion_derivative(self, x: np.ndarray, t: float) -> np.ndarray:
        return np.zeros_like(x)

    def exact_mean(self, t: float) -> float:
        return self.mu + (self._X0 - self.mu) * np.exp(-self.theta * t)

    def exact_variance(self, t: float) -> float:
        return (self.sigma**2 / (2 * self.theta)) * (1 - np.exp(-2 * self.theta * t))


class CoxIngersollRoss(SDEProcess):
    """
    Cox-Ingersoll-Ross:  dX = κ(θ - X) dt + σ √X dW

    Ensures non-negativity if 2κθ ≥ σ² (Feller condition).
    """

    def __init__(self, kappa: float = 2.0, theta: float = 0.04, sigma: float = 0.3, X0: float = 0.04):
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self._X0 = X0

    @property
    def feller_satisfied(self) -> bool:
        return 2 * self.kappa * self.theta >= self.sigma**2

    @property
    def dim(self) -> int:
        return 1

    @property
    def x0(self) -> np.ndarray:
        return np.array([self._X0])

    def drift(self, x: np.ndarray, t: float) -> np.ndarray:
        return self.kappa * (self.theta - x)

    def diffusion(self, x: np.ndarray, t: float) -> np.ndarray:
        return self.sigma * np.sqrt(np.maximum(x, 0.0))

    def diffusion_derivative(self, x: np.ndarray, t: float) -> np.ndarray:
        safe_x = np.maximum(x, 1e-12)
        return 0.5 * self.sigma / np.sqrt(safe_x)


class VasicekProcess(SDEProcess):
    """
    Vasicek short-rate model:  dr = a(b - r) dt + σ dW

    Additive noise variant of OU — analytically tractable for bond pricing.
    """

    def __init__(self, a: float = 0.5, b: float = 0.05, sigma: float = 0.01, r0: float = 0.03):
        self.a = a
        self.b = b
        self.sigma = sigma
        self._r0 = r0

    @property
    def dim(self) -> int:
        return 1

    @property
    def x0(self) -> np.ndarray:
        return np.array([self._r0])

    def drift(self, x: np.ndarray, t: float) -> np.ndarray:
        return self.a * (self.b - x)

    def diffusion(self, x: np.ndarray, t: float) -> np.ndarray:
        return np.full_like(x, self.sigma)

    def diffusion_derivative(self, x: np.ndarray, t: float) -> np.ndarray:
        return np.zeros_like(x)

    def zero_coupon_bond(self, t: float, T: float, r: float) -> float:
        """Analytical ZCB price P(t,T) under Vasicek."""
        tau = T - t
        B = (1 - np.exp(-self.a * tau)) / self.a
        A = np.exp(
            (B - tau) * (self.a**2 * self.b - 0.5 * self.sigma**2) / self.a**2
            - self.sigma**2 * B**2 / (4 * self.a)
        )
        return A * np.exp(-B * r)


class HestonProcess(SDEProcess):
    """
    Heston stochastic volatility model (2D correlated SDE):

        dS = μ S dt + √v S dW_1
        dv = κ(θ - v) dt + ξ √v dW_2
        dW_1 dW_2 = ρ dt

    State vector: [S, v]
    """

    def __init__(
        self,
        mu: float = 0.05,
        kappa: float = 2.0,
        theta: float = 0.04,
        xi: float = 0.3,
        rho: float = -0.7,
        S0: float = 100.0,
        v0: float = 0.04,
    ):
        self.mu = mu
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho
        self._S0 = S0
        self._v0 = v0

    @property
    def dim(self) -> int:
        return 2

    @property
    def x0(self) -> np.ndarray:
        return np.array([self._S0, self._v0])

    def drift(self, x: np.ndarray, t: float) -> np.ndarray:
        S, v = x[..., 0], x[..., 1]
        v_safe = np.maximum(v, 0.0)
        dS = self.mu * S
        dv = self.kappa * (self.theta - v_safe)
        return np.stack([dS, dv], axis=-1)

    def diffusion(self, x: np.ndarray, t: float) -> np.ndarray:
        S, v = x[..., 0], x[..., 1]
        sqrt_v = np.sqrt(np.maximum(v, 0.0))
        sig_S = sqrt_v * S
        sig_v = self.xi * sqrt_v
        return np.stack([sig_S, sig_v], axis=-1)

    @property
    def correlation_matrix(self) -> np.ndarray:
        return np.array([[1.0, self.rho], [self.rho, 1.0]])

    def cholesky(self) -> np.ndarray:
        return np.linalg.cholesky(self.correlation_matrix)


class SABRProcess(SDEProcess):
    """
    SABR stochastic volatility model:

        dF = σ F^β dW_1
        dσ = α σ dW_2
        dW_1 dW_2 = ρ dt

    State vector: [F, σ]
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.5,
        rho: float = -0.25,
        F0: float = 100.0,
        sigma0: float = 0.2,
    ):
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self._F0 = F0
        self._sigma0 = sigma0

    @property
    def dim(self) -> int:
        return 2

    @property
    def x0(self) -> np.ndarray:
        return np.array([self._F0, self._sigma0])

    def drift(self, x: np.ndarray, t: float) -> np.ndarray:
        return np.zeros_like(x)

    def diffusion(self, x: np.ndarray, t: float) -> np.ndarray:
        F, sig = x[..., 0], x[..., 1]
        F_safe = np.maximum(F, 1e-10)
        sig_safe = np.maximum(sig, 1e-10)
        dF = sig_safe * F_safe**self.beta
        dsig = self.alpha * sig_safe
        return np.stack([dF, dsig], axis=-1)

    @property
    def correlation_matrix(self) -> np.ndarray:
        return np.array([[1.0, self.rho], [self.rho, 1.0]])


class MertonJumpDiffusion(SDEProcess):
    """
    Merton Jump-Diffusion:  dS/S = (μ - λk) dt + σ dW + J dN

    Where N is a Poisson process with intensity λ, and
    log(1+J) ~ N(μ_J, σ_J²) so k = E[J] = exp(μ_J + σ_J²/2) - 1.
    """

    def __init__(
        self,
        mu: float = 0.1,
        sigma: float = 0.2,
        lam: float = 1.0,
        mu_J: float = -0.05,
        sigma_J: float = 0.1,
        S0: float = 100.0,
    ):
        self.mu = mu
        self.sigma = sigma
        self.lam = lam
        self.mu_J = mu_J
        self.sigma_J = sigma_J
        self._S0 = S0
        self.k = np.exp(mu_J + 0.5 * sigma_J**2) - 1

    @property
    def dim(self) -> int:
        return 1

    @property
    def x0(self) -> np.ndarray:
        return np.array([self._S0])

    def drift(self, x: np.ndarray, t: float) -> np.ndarray:
        return (self.mu - self.lam * self.k) * x

    def diffusion(self, x: np.ndarray, t: float) -> np.ndarray:
        return self.sigma * x

    def simulate_jumps(self, dt: float, n_paths: int) -> np.ndarray:
        """Generate jump component for one time step."""
        N = np.random.poisson(self.lam * dt, n_paths)
        J = np.zeros(n_paths)
        for i in range(n_paths):
            if N[i] > 0:
                log_jumps = np.random.normal(self.mu_J, self.sigma_J, N[i])
                J[i] = np.prod(1 + np.exp(log_jumps) - 1) - 1
        return J


class CustomSDE(SDEProcess):
    """
    User-defined SDE with arbitrary drift and diffusion functions.

    ```python
    sde = CustomSDE(
        drift_fn=lambda x, t: -0.5 * x,
        diffusion_fn=lambda x, t: 0.3 * np.ones_like(x),
        x0=np.array([1.0]),
    )
    ```
    """

    def __init__(
        self,
        drift_fn: Callable[[np.ndarray, float], np.ndarray],
        diffusion_fn: Callable[[np.ndarray, float], np.ndarray],
        x0: np.ndarray,
        diffusion_deriv_fn: Optional[Callable] = None,
    ):
        self._drift_fn = drift_fn
        self._diffusion_fn = diffusion_fn
        self._x0 = np.asarray(x0)
        self._diffusion_deriv_fn = diffusion_deriv_fn

    @property
    def dim(self) -> int:
        return len(self._x0)

    @property
    def x0(self) -> np.ndarray:
        return self._x0

    def drift(self, x: np.ndarray, t: float) -> np.ndarray:
        return self._drift_fn(x, t)

    def diffusion(self, x: np.ndarray, t: float) -> np.ndarray:
        return self._diffusion_fn(x, t)

    def diffusion_derivative(self, x: np.ndarray, t: float) -> np.ndarray:
        if self._diffusion_deriv_fn is not None:
            return self._diffusion_deriv_fn(x, t)
        return super().diffusion_derivative(x, t)
