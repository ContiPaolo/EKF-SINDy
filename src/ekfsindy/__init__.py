"""EKF-SINDy: extended Kalman filter with sparse identification of nonlinear dynamics."""

import jax

# Kalman filtering needs double precision (covariances span many orders of magnitude).
jax.config.update("jax_enable_x64", True)

from . import data, observation, plotting, systems
from .ekf import AugmentedEKF, EKFResult
from .library import SymbolicLibrary
from .model import SparseModel

__all__ = [
    "AugmentedEKF",
    "EKFResult",
    "SparseModel",
    "SymbolicLibrary",
    "data",
    "observation",
    "plotting",
    "systems",
]
