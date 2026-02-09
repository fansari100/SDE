"""
SDE — Stochastic Differential Equations Numerical Methods Library.

Comprehensive library for solving SDEs using Euler-Maruyama, Milstein,
and higher-order schemes, with applications to mathematical finance.
"""

from .solvers import EulerMaruyama, Milstein, StochasticRungeKutta, ImplicitEulerMaruyama
from .processes import (
    GeometricBrownianMotion,
    OrnsteinUhlenbeck,
    CoxIngersollRoss,
    HestonProcess,
    SABRProcess,
    VasicekProcess,
    MertonJumpDiffusion,
    CustomSDE,
)
from .analysis import ConvergenceAnalyzer
from .finance import OptionPricer

__all__ = [
    "EulerMaruyama",
    "Milstein",
    "StochasticRungeKutta",
    "ImplicitEulerMaruyama",
    "GeometricBrownianMotion",
    "OrnsteinUhlenbeck",
    "CoxIngersollRoss",
    "HestonProcess",
    "SABRProcess",
    "VasicekProcess",
    "MertonJumpDiffusion",
    "CustomSDE",
    "ConvergenceAnalyzer",
    "OptionPricer",
]

__version__ = "1.0.0"
