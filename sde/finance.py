"""
Financial applications of SDE simulation.

European/exotic option pricing via Monte Carlo, Greeks computation
(finite-difference and pathwise), and implied volatility surface generation.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional

from .processes import SDEProcess, GeometricBrownianMotion
from .solvers import EulerMaruyama, Milstein, SDESolution


@dataclass
class OptionPrice:
    """Option pricing result with confidence interval."""

    price: float
    std_error: float
    ci_lower: float
    ci_upper: float
    n_paths: int


@dataclass
class Greeks:
    """Option Greeks (sensitivities)."""

    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


class OptionPricer:
    """
    Monte Carlo option pricer using SDE simulation.

    Parameters
    ----------
    process : SDEProcess
        The underlying asset dynamics.
    solver : EulerMaruyama or Milstein
        Numerical solver to use.
    """

    def __init__(
        self,
        process: Optional[SDEProcess] = None,
        solver: Optional[EulerMaruyama] = None,
    ):
        self.process = process or GeometricBrownianMotion()
        self.solver = solver or EulerMaruyama(dt=1 / 252)

    def european_call(
        self,
        K: float,
        T: float,
        r: float,
        n_paths: int = 100_000,
    ) -> OptionPrice:
        """Price a European call option via Monte Carlo."""
        sol = self.solver.solve(self.process, T, n_paths)
        S_T = sol.paths[:, -1] if sol.paths.ndim == 2 else sol.paths[:, -1, 0]

        payoffs = np.maximum(S_T - K, 0)
        discounted = np.exp(-r * T) * payoffs

        price = float(np.mean(discounted))
        se = float(np.std(discounted) / np.sqrt(n_paths))

        return OptionPrice(
            price=price,
            std_error=se,
            ci_lower=price - 1.96 * se,
            ci_upper=price + 1.96 * se,
            n_paths=n_paths,
        )

    def european_put(
        self,
        K: float,
        T: float,
        r: float,
        n_paths: int = 100_000,
    ) -> OptionPrice:
        """Price a European put option via Monte Carlo."""
        sol = self.solver.solve(self.process, T, n_paths)
        S_T = sol.paths[:, -1] if sol.paths.ndim == 2 else sol.paths[:, -1, 0]

        payoffs = np.maximum(K - S_T, 0)
        discounted = np.exp(-r * T) * payoffs

        price = float(np.mean(discounted))
        se = float(np.std(discounted) / np.sqrt(n_paths))

        return OptionPrice(
            price=price, std_error=se,
            ci_lower=price - 1.96 * se, ci_upper=price + 1.96 * se,
            n_paths=n_paths,
        )

    def asian_call(
        self,
        K: float,
        T: float,
        r: float,
        n_paths: int = 100_000,
    ) -> OptionPrice:
        """Arithmetic Asian call — payoff on average price."""
        sol = self.solver.solve(self.process, T, n_paths)
        paths = sol.paths if sol.paths.ndim == 2 else sol.paths[:, :, 0]
        avg_price = np.mean(paths, axis=1)

        payoffs = np.maximum(avg_price - K, 0)
        discounted = np.exp(-r * T) * payoffs

        price = float(np.mean(discounted))
        se = float(np.std(discounted) / np.sqrt(n_paths))

        return OptionPrice(
            price=price, std_error=se,
            ci_lower=price - 1.96 * se, ci_upper=price + 1.96 * se,
            n_paths=n_paths,
        )

    def barrier_call(
        self,
        K: float,
        T: float,
        r: float,
        barrier: float,
        barrier_type: str = "up-and-out",
        n_paths: int = 100_000,
    ) -> OptionPrice:
        """Barrier option pricing."""
        sol = self.solver.solve(self.process, T, n_paths)
        paths = sol.paths if sol.paths.ndim == 2 else sol.paths[:, :, 0]
        S_T = paths[:, -1]

        if barrier_type == "up-and-out":
            knocked = np.any(paths > barrier, axis=1)
            payoffs = np.where(knocked, 0, np.maximum(S_T - K, 0))
        elif barrier_type == "down-and-out":
            knocked = np.any(paths < barrier, axis=1)
            payoffs = np.where(knocked, 0, np.maximum(S_T - K, 0))
        elif barrier_type == "up-and-in":
            triggered = np.any(paths > barrier, axis=1)
            payoffs = np.where(triggered, np.maximum(S_T - K, 0), 0)
        elif barrier_type == "down-and-in":
            triggered = np.any(paths < barrier, axis=1)
            payoffs = np.where(triggered, np.maximum(S_T - K, 0), 0)
        else:
            raise ValueError(f"Unknown barrier type: {barrier_type}")

        discounted = np.exp(-r * T) * payoffs
        price = float(np.mean(discounted))
        se = float(np.std(discounted) / np.sqrt(n_paths))

        return OptionPrice(
            price=price, std_error=se,
            ci_lower=price - 1.96 * se, ci_upper=price + 1.96 * se,
            n_paths=n_paths,
        )

    def lookback_call(
        self,
        T: float,
        r: float,
        n_paths: int = 100_000,
    ) -> OptionPrice:
        """Floating-strike lookback call: payoff = S_T - min(S)."""
        sol = self.solver.solve(self.process, T, n_paths)
        paths = sol.paths if sol.paths.ndim == 2 else sol.paths[:, :, 0]
        S_T = paths[:, -1]
        S_min = np.min(paths, axis=1)

        payoffs = S_T - S_min
        discounted = np.exp(-r * T) * payoffs

        price = float(np.mean(discounted))
        se = float(np.std(discounted) / np.sqrt(n_paths))

        return OptionPrice(
            price=price, std_error=se,
            ci_lower=price - 1.96 * se, ci_upper=price + 1.96 * se,
            n_paths=n_paths,
        )

    def greeks(
        self,
        K: float,
        T: float,
        r: float,
        n_paths: int = 100_000,
        bump: float = 0.01,
    ) -> Greeks:
        """
        Compute Greeks via finite differences (bump-and-revalue).

        Parameters
        ----------
        K, T, r : option parameters
        n_paths : Monte Carlo paths per valuation
        bump : relative bump size for finite differences
        """
        S0 = self.process.x0[0]
        base_price = self.european_call(K, T, r, n_paths).price

        # Delta: ∂C/∂S
        dS = S0 * bump
        proc_up = self._bump_S0(S0 + dS)
        proc_dn = self._bump_S0(S0 - dS)
        pricer_up = OptionPricer(proc_up, self.solver)
        pricer_dn = OptionPricer(proc_dn, self.solver)
        C_up = pricer_up.european_call(K, T, r, n_paths).price
        C_dn = pricer_dn.european_call(K, T, r, n_paths).price
        delta = (C_up - C_dn) / (2 * dS)
        gamma = (C_up - 2 * base_price + C_dn) / (dS**2)

        # Vega: ∂C/∂σ
        if hasattr(self.process, "sigma"):
            sig = self.process.sigma
            dsig = sig * bump
            proc_v_up = self._bump_sigma(sig + dsig)
            proc_v_dn = self._bump_sigma(sig - dsig)
            C_v_up = OptionPricer(proc_v_up, self.solver).european_call(K, T, r, n_paths).price
            C_v_dn = OptionPricer(proc_v_dn, self.solver).european_call(K, T, r, n_paths).price
            vega = (C_v_up - C_v_dn) / (2 * dsig)
        else:
            vega = 0.0

        # Theta: -∂C/∂T
        dT = T * bump
        C_T_up = self.european_call(K, T + dT, r, n_paths).price
        theta = -(C_T_up - base_price) / dT

        # Rho: ∂C/∂r
        dr = 0.001
        C_r_up = self.european_call(K, T, r + dr, n_paths).price
        rho = (C_r_up - base_price) / dr

        return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)

    def _bump_S0(self, new_S0: float) -> SDEProcess:
        """Create a copy of the process with bumped initial value."""
        if isinstance(self.process, GeometricBrownianMotion):
            return GeometricBrownianMotion(
                mu=self.process.mu, sigma=self.process.sigma, S0=new_S0
            )
        raise NotImplementedError("S0 bumping for this process type")

    def _bump_sigma(self, new_sigma: float) -> SDEProcess:
        """Create a copy of the process with bumped volatility."""
        if isinstance(self.process, GeometricBrownianMotion):
            return GeometricBrownianMotion(
                mu=self.process.mu, sigma=new_sigma, S0=self.process.x0[0]
            )
        raise NotImplementedError("Sigma bumping for this process type")


def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Analytical Black-Scholes European call price for benchmarking."""
    from scipy.stats import norm

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def black_scholes_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Analytical Black-Scholes European put price."""
    from scipy.stats import norm

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
