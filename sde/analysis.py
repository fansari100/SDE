"""
Convergence and error analysis for SDE solvers.

Estimates strong and weak convergence rates by running the solver at
multiple resolutions and computing the empirical rate via log-log regression.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional

from .processes import SDEProcess, GeometricBrownianMotion
from .solvers import EulerMaruyama, Milstein, StochasticRungeKutta


@dataclass
class ConvergenceResult:
    """Results of a convergence analysis."""

    dt_values: np.ndarray
    errors: np.ndarray
    order: float
    intercept: float
    method: str
    convergence_type: str  # "strong" or "weak"


class ConvergenceAnalyzer:
    """
    Analyze convergence rates of SDE numerical schemes.

    Uses a fine-grid reference solution (or analytical solution if available)
    to compute errors at various step sizes.

    Parameters
    ----------
    process : SDEProcess
        The stochastic process to analyze.
    T : float
        Time horizon.
    n_paths : int
        Number of Monte Carlo paths.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        process: SDEProcess,
        T: float = 1.0,
        n_paths: int = 5000,
        seed: int = 42,
    ):
        self.process = process
        self.T = T
        self.n_paths = n_paths
        self.seed = seed

    def strong_convergence(
        self,
        dt_values: Optional[list[float]] = None,
        method: str = "euler_maruyama",
    ) -> ConvergenceResult:
        """
        Estimate strong convergence rate: E[|X_T - X̂_T|].

        Strong convergence measures pathwise accuracy.

        Parameters
        ----------
        dt_values : list of step sizes to test
        method : "euler_maruyama", "milstein", or "srk"
        """
        if dt_values is None:
            dt_values = [0.1, 0.05, 0.02, 0.01, 0.005]

        dt_arr = np.array(dt_values)
        errors = np.zeros(len(dt_values))

        # Reference solution at very fine dt
        dt_ref = min(dt_values) / 10.0
        solver_ref = self._make_solver(method, dt_ref)
        rng = np.random.default_rng(self.seed)

        # Generate shared Brownian motion at finest grid
        n_steps_ref = int(np.ceil(self.T / dt_ref))
        dt_actual_ref = self.T / n_steps_ref
        dW_fine = rng.normal(0, np.sqrt(dt_actual_ref), (self.n_paths, n_steps_ref))

        ref_sol = solver_ref.solve(self.process, self.T, self.n_paths)
        X_ref = ref_sol.paths[:, -1] if ref_sol.paths.ndim == 2 else ref_sol.paths[:, -1, 0]

        for idx, dt in enumerate(dt_values):
            solver = self._make_solver(method, dt)
            sol = solver.solve(self.process, self.T, self.n_paths)
            X_approx = sol.paths[:, -1] if sol.paths.ndim == 2 else sol.paths[:, -1, 0]
            errors[idx] = np.mean(np.abs(X_approx - X_ref))

        order, intercept = self._fit_rate(dt_arr, errors)

        return ConvergenceResult(
            dt_values=dt_arr,
            errors=errors,
            order=order,
            intercept=intercept,
            method=method,
            convergence_type="strong",
        )

    def weak_convergence(
        self,
        dt_values: Optional[list[float]] = None,
        method: str = "euler_maruyama",
        test_fn: Optional[callable] = None,
    ) -> ConvergenceResult:
        """
        Estimate weak convergence rate: |E[g(X_T)] - E[g(X̂_T)]|.

        Weak convergence measures accuracy of distributional properties.

        Parameters
        ----------
        dt_values : list of step sizes
        method : solver method name
        test_fn : function g(x) to test (default: identity)
        """
        if dt_values is None:
            dt_values = [0.1, 0.05, 0.02, 0.01, 0.005]
        if test_fn is None:
            test_fn = lambda x: x

        dt_arr = np.array(dt_values)
        errors = np.zeros(len(dt_values))

        # Reference at very fine grid
        dt_ref = min(dt_values) / 10.0
        solver_ref = self._make_solver(method, dt_ref)
        ref_sol = solver_ref.solve(self.process, self.T, self.n_paths * 5)
        X_ref = ref_sol.paths[:, -1] if ref_sol.paths.ndim == 2 else ref_sol.paths[:, -1, 0]
        E_ref = np.mean(test_fn(X_ref))

        for idx, dt in enumerate(dt_values):
            solver = self._make_solver(method, dt)
            sol = solver.solve(self.process, self.T, self.n_paths * 5)
            X_approx = sol.paths[:, -1] if sol.paths.ndim == 2 else sol.paths[:, -1, 0]
            E_approx = np.mean(test_fn(X_approx))
            errors[idx] = np.abs(E_approx - E_ref)

        order, intercept = self._fit_rate(dt_arr, errors)

        return ConvergenceResult(
            dt_values=dt_arr,
            errors=errors,
            order=order,
            intercept=intercept,
            method=method,
            convergence_type="weak",
        )

    @staticmethod
    def _fit_rate(dt_values: np.ndarray, errors: np.ndarray) -> tuple[float, float]:
        """Log-log linear regression to estimate convergence order."""
        mask = errors > 1e-15
        if mask.sum() < 2:
            return 0.0, 0.0
        log_dt = np.log(dt_values[mask])
        log_err = np.log(errors[mask])
        A = np.column_stack([log_dt, np.ones_like(log_dt)])
        coeffs, _, _, _ = np.linalg.lstsq(A, log_err, rcond=None)
        return float(coeffs[0]), float(coeffs[1])

    @staticmethod
    def _make_solver(method: str, dt: float):
        solvers = {
            "euler_maruyama": EulerMaruyama,
            "milstein": Milstein,
            "srk": StochasticRungeKutta,
        }
        cls = solvers.get(method, EulerMaruyama)
        return cls(dt=dt)

    def monte_carlo_error(
        self,
        dt: float = 0.01,
        n_paths_list: Optional[list[int]] = None,
    ) -> dict:
        """
        Estimate Monte Carlo standard error vs number of paths.

        Returns dict with 'n_paths', 'means', 'std_errors'.
        """
        if n_paths_list is None:
            n_paths_list = [100, 500, 1000, 5000, 10000, 50000]

        means = []
        std_errors = []

        for n in n_paths_list:
            solver = EulerMaruyama(dt=dt)
            sol = solver.solve(self.process, self.T, n)
            terminal = sol.paths[:, -1] if sol.paths.ndim == 2 else sol.paths[:, -1, 0]
            means.append(np.mean(terminal))
            std_errors.append(np.std(terminal) / np.sqrt(n))

        return {
            "n_paths": np.array(n_paths_list),
            "means": np.array(means),
            "std_errors": np.array(std_errors),
        }
