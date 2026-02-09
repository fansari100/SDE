"""Plotting utilities for SDE paths, convergence analysis, and option surfaces."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

from .solvers import SDESolution
from .analysis import ConvergenceResult


def plot_paths(
    solution: SDESolution,
    n_show: int = 20,
    title: str = "SDE Sample Paths",
    ylabel: str = "X(t)",
    figsize: tuple = (12, 6),
    alpha: float = 0.4,
) -> plt.Figure:
    """Plot sample paths from an SDE simulation."""
    fig, ax = plt.subplots(figsize=figsize)

    paths = solution.paths
    if paths.ndim == 3:
        paths = paths[:, :, 0]

    n_show = min(n_show, paths.shape[0])
    for i in range(n_show):
        ax.plot(solution.t, paths[i], alpha=alpha, linewidth=0.8)

    ax.plot(solution.t, np.mean(paths, axis=0), color="black", linewidth=2, label="Mean")
    q05 = np.percentile(paths, 5, axis=0)
    q95 = np.percentile(paths, 95, axis=0)
    ax.fill_between(solution.t, q05, q95, alpha=0.15, color="blue", label="90% CI")

    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_convergence(
    result: ConvergenceResult,
    figsize: tuple = (10, 7),
) -> plt.Figure:
    """Plot convergence rate analysis (log-log scale)."""
    fig, ax = plt.subplots(figsize=figsize)

    ax.loglog(result.dt_values, result.errors, "o-", markersize=8, linewidth=2, label="Measured error")

    # Reference lines
    dt_ref = np.array([result.dt_values[0], result.dt_values[-1]])
    for order, ls, label in [(0.5, "--", "Order 0.5"), (1.0, "-.", "Order 1.0"), (1.5, ":", "Order 1.5")]:
        ref = result.errors[0] * (dt_ref / dt_ref[0]) ** order
        ax.loglog(dt_ref, ref, ls, color="gray", alpha=0.6, label=label)

    ax.set_xlabel("Step size Δt", fontsize=12)
    ax.set_ylabel(f"{result.convergence_type.capitalize()} error", fontsize=12)
    ax.set_title(
        f"{result.convergence_type.capitalize()} Convergence — {result.method}\n"
        f"Estimated order: {result.order:.2f}",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    return fig


def plot_terminal_distribution(
    solution: SDESolution,
    bins: int = 100,
    title: str = "Terminal Distribution",
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """Histogram of terminal values with statistics."""
    fig, ax = plt.subplots(figsize=figsize)

    terminal = solution.paths[:, -1] if solution.paths.ndim == 2 else solution.paths[:, -1, 0]

    ax.hist(terminal, bins=bins, density=True, alpha=0.7, color="steelblue", edgecolor="white")

    mean_val = np.mean(terminal)
    std_val = np.std(terminal)
    ax.axvline(mean_val, color="red", linewidth=2, label=f"Mean = {mean_val:.4f}")
    ax.axvline(mean_val - 2 * std_val, color="orange", linewidth=1.5, linestyle="--", label=f"±2σ")
    ax.axvline(mean_val + 2 * std_val, color="orange", linewidth=1.5, linestyle="--")

    ax.set_xlabel("X(T)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
