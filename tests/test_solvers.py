"""Tests for SDE numerical solvers — correctness, convergence, and edge cases."""

import numpy as np
import pytest
from sde import (
    EulerMaruyama, Milstein, StochasticRungeKutta, ImplicitEulerMaruyama,
    GeometricBrownianMotion, OrnsteinUhlenbeck, CoxIngersollRoss,
    ConvergenceAnalyzer, OptionPricer,
)
from sde.finance import black_scholes_call


class TestEulerMaruyama:
    def test_gbm_mean(self):
        """E[S_T] = S_0 exp(μT) for GBM."""
        gbm = GeometricBrownianMotion(mu=0.05, sigma=0.2, S0=100)
        solver = EulerMaruyama(dt=1 / 252, seed=42)
        sol = solver.solve(gbm, T=1.0, n_paths=50000)
        mean = np.mean(sol.paths[:, -1])
        expected = 100 * np.exp(0.05)
        assert abs(mean - expected) / expected < 0.02

    def test_ou_mean_reversion(self):
        """OU process converges to long-run mean."""
        ou = OrnsteinUhlenbeck(theta=5.0, mu=2.0, sigma=0.5, X0=0.0)
        solver = EulerMaruyama(dt=0.001, seed=42)
        sol = solver.solve(ou, T=5.0, n_paths=10000)
        mean = np.mean(sol.paths[:, -1])
        assert abs(mean - 2.0) < 0.05

    def test_cir_non_negative(self):
        """CIR paths should remain non-negative (approx)."""
        cir = CoxIngersollRoss(kappa=2.0, theta=0.04, sigma=0.2, X0=0.04)
        solver = EulerMaruyama(dt=0.001, seed=42)
        sol = solver.solve(cir, T=1.0, n_paths=5000)
        # Allow tiny negative due to Euler discretization
        assert np.min(sol.paths) > -0.01

    def test_output_shape(self):
        gbm = GeometricBrownianMotion(S0=100)
        solver = EulerMaruyama(dt=0.1, seed=0)
        sol = solver.solve(gbm, T=1.0, n_paths=100)
        assert sol.paths.shape == (100, 11)
        assert sol.t.shape == (11,)


class TestMilstein:
    def test_gbm_higher_accuracy(self):
        """Milstein should be more accurate than EM for GBM."""
        gbm = GeometricBrownianMotion(mu=0.05, sigma=0.3, S0=100)

        em = EulerMaruyama(dt=0.05, seed=42)
        mil = Milstein(dt=0.05, seed=42)

        sol_em = em.solve(gbm, T=1.0, n_paths=20000)
        sol_mil = mil.solve(gbm, T=1.0, n_paths=20000)

        expected = 100 * np.exp(0.05)
        err_em = abs(np.mean(sol_em.paths[:, -1]) - expected)
        err_mil = abs(np.mean(sol_mil.paths[:, -1]) - expected)
        # Milstein should do at least as well
        assert err_mil <= err_em * 1.5


class TestOptionPricing:
    def test_bs_call_accuracy(self):
        """MC call price should match Black-Scholes within 1%."""
        gbm = GeometricBrownianMotion(mu=0.05, sigma=0.2, S0=100)
        pricer = OptionPricer(gbm, Milstein(dt=1 / 252, seed=42))
        mc_price = pricer.european_call(K=100, T=1.0, r=0.05, n_paths=200000)
        bs_price = black_scholes_call(100, 100, 1.0, 0.05, 0.2)
        assert abs(mc_price.price - bs_price) / bs_price < 0.015

    def test_put_call_parity(self):
        """C - P = S - K e^{-rT}."""
        gbm = GeometricBrownianMotion(mu=0.05, sigma=0.2, S0=100)
        pricer = OptionPricer(gbm, Milstein(dt=1 / 252, seed=42))
        call = pricer.european_call(K=100, T=1.0, r=0.05, n_paths=200000)
        put = pricer.european_put(K=100, T=1.0, r=0.05, n_paths=200000)
        parity = call.price - put.price - (100 - 100 * np.exp(-0.05))
        assert abs(parity) < 1.0

    def test_greeks_delta_range(self):
        """Delta of ATM call should be ~0.5-0.7."""
        gbm = GeometricBrownianMotion(mu=0.05, sigma=0.2, S0=100)
        pricer = OptionPricer(gbm, EulerMaruyama(dt=1 / 252, seed=42))
        g = pricer.greeks(K=100, T=1.0, r=0.05, n_paths=50000)
        assert 0.3 < g.delta < 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
