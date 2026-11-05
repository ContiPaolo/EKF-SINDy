"""Plotting helpers for EKF-SINDy results."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np

from .ekf import EKFResult


def plot_estimate(
    t: np.ndarray,
    result: EKFResult,
    index: int,
    truth: np.ndarray | float | None = None,
    measured: np.ndarray | None = None,
    label: str = "",
    color: str = "red",
    n_sigma: float = 1.96,
    ax: plt.Axes | None = None,
    window: slice = slice(None),
) -> plt.Axes:
    """Plot ``result.mean[:, index]`` with its ``n_sigma`` band against truth/measurements."""
    ax = ax or plt.gca()
    t = np.asarray(t)[window]
    lower, upper = result.bounds(n_sigma)
    if measured is not None:
        ax.plot(t, np.asarray(measured)[window], color="orange", lw=0.8, label=f"{label} measured")
    if truth is not None:
        truth = np.broadcast_to(truth, np.shape(result.mean[:, index]))[window]
        ax.plot(t, truth, "k", label=f"{label} true")
    ax.plot(t, result.mean[window, index], "--", color=color, label=f"{label} estimated")
    ax.fill_between(t, lower[window, index], upper[window, index], color=color, alpha=0.2,
                    label=rf"{label} $\pm {n_sigma}\sigma$")
    ax.set_xlabel("t")
    ax.legend()
    return ax


def plot_estimates(
    t: np.ndarray,
    result: EKFResult,
    indices: Sequence[int],
    labels: Sequence[str],
    truths: Sequence[np.ndarray | float | None] | None = None,
    measured: Sequence[np.ndarray | None] | None = None,
    ncols: int = 2,
    figsize_per_axis: tuple[float, float] = (8, 3.5),
    window: slice = slice(None),
):
    """Grid of ``plot_estimate`` panels, one per entry of ``indices``."""
    n = len(indices)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                             figsize=(figsize_per_axis[0] * ncols, figsize_per_axis[1] * nrows))
    truths = truths if truths is not None else [None] * n
    measured = measured if measured is not None else [None] * n
    for ax, i, lab, tr, me in zip(axes.flat, indices, labels, truths, measured):
        plot_estimate(t, result, i, truth=tr, measured=me, label=lab, ax=ax, window=window)
        ax.set_title(lab)
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig, axes


def plot_phase_space(true_states: np.ndarray, result: EKFResult, labels=("$u_1$", "$u_2$"),
                     window: slice = slice(None), ax: plt.Axes | None = None) -> plt.Axes:
    """Trajectory of the first two states: reference vs. EKF estimate."""
    ax = ax or plt.gca()
    ax.plot(true_states[window, 0], true_states[window, 1], "k", label="reference")
    ax.plot(result.mean[window, 0], result.mean[window, 1], "r--", label="estimate")
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.legend()
    return ax
