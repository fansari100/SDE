# Stochastic Differential Equations — Numerical Methods Library

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**A comprehensive Python library for solving Stochastic Differential Equations (SDEs) using state-of-the-art numerical methods, with applications to mathematical finance and scientific computing.**

## Mathematical Foundation

A general Itô SDE takes the form:

```
dX_t = μ(X_t, t) dt + σ(X_t, t) dW_t
```

where `μ` is the drift coefficient, `σ` is the diffusion coefficient, and `W_t` is a Wiener process (Brownian motion).

## Features

### Numerical Solvers
- **Euler-Maruyama** — Strong order 0.5, weak order 1.0
- **Milstein** — Strong order 1.0 (exploits diffusion derivative)
- **Stochastic Runge-Kutta (SRK)** — Strong order 1.5
- **Implicit Euler-Maruyama** — A-stable for stiff SDEs
- **Adaptive step-size** — Error-controlled integration

### Stochastic Processes
- **Geometric Brownian Motion (GBM)** — Black-Scholes asset dynamics
- **Ornstein-Uhlenbeck** — Mean-reverting process
- **Cox-Ingersoll-Ross (CIR)** — Interest rate / volatility modeling
- **Heston** — Stochastic volatility (2D correlated SDE system)
- **SABR** — Stochastic Alpha Beta Rho volatility model
- **Vasicek** — Short rate model
- **Hull-White** — Extended Vasicek with time-dependent parameters
- **Merton Jump-Diffusion** — GBM with Poisson jumps
- **Custom SDEs** — User-defined drift and diffusion functions

### Convergence Analysis
- Strong convergence rate estimation
- Weak convergence rate estimation
- Monte Carlo error quantification
- Richardson extrapolation for improved accuracy

### Financial Applications
- European option pricing via Monte Carlo
- Greeks computation (Delta, Gamma, Vega, Theta, Rho)
- Implied volatility surface generation
- Path-dependent option pricing (Asian, Barrier, Lookback)

## Quick Start

```python
from sde import EulerMaruyama, Milstein, GeometricBrownianMotion, OptionPricer

# 1. Simulate GBM paths
gbm = GeometricBrownianMotion(mu=0.05, sigma=0.2, S0=100.0)
solver = EulerMaruyama(dt=1/252)
paths = solver.solve(gbm, T=1.0, n_paths=10000)

# 2. Price a European call option
pricer = OptionPricer(process=gbm, solver=Milstein(dt=1/252))
price = pricer.european_call(K=105, T=1.0, r=0.05, n_paths=100000)
print(f"Call price: ${price:.4f}")

# 3. Compute Greeks
greeks = pricer.greeks(K=105, T=1.0, r=0.05)
print(f"Delta: {greeks['delta']:.4f}, Gamma: {greeks['gamma']:.6f}")

# 4. Convergence analysis
from sde import ConvergenceAnalyzer
analyzer = ConvergenceAnalyzer(process=gbm)
rates = analyzer.strong_convergence(dt_values=[0.1, 0.05, 0.01, 0.005, 0.001])
print(f"Estimated strong order: {rates['order']:.2f}")
```

## Project Structure

```
SDE/
├── sde/
│   ├── __init__.py          # Public API
│   ├── solvers.py           # Numerical SDE solvers
│   ├── processes.py         # Stochastic process definitions
│   ├── analysis.py          # Convergence and error analysis
│   ├── finance.py           # Option pricing and Greeks
│   └── visualization.py     # Plotting utilities
├── tests/
│   ├── test_solvers.py      # Solver correctness tests
│   ├── test_processes.py    # Process property tests
│   └── test_finance.py      # Pricing accuracy tests
├── examples/
│   └── demo.py              # Interactive demonstrations
├── requirements.txt
└── README.md
```

## References

1. Kloeden & Platen, *Numerical Solution of Stochastic Differential Equations* (Springer, 1992)
2. Higham, *An Algorithmic Introduction to Numerical Simulation of SDEs* (SIAM Review, 2001)
3. Glasserman, *Monte Carlo Methods in Financial Engineering* (Springer, 2003)
4. Iacus, *Simulation and Inference for Stochastic Differential Equations* (Springer, 2008)

## License

MIT License

## Author

**Ricky Ansari** — [GitHub](https://github.com/fansari100) | [Website](https://rickyansari.com)
