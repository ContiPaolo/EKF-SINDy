"""Observation maps ``y = h(z, u)`` for the augmented state ``z = [x, p, xi]``."""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp

from .model import SparseModel


def select_states(indices: Sequence[int]):
    """Observe a subset of the augmented state, e.g. ``select_states([0, 1])``."""
    idx = jnp.asarray(indices)

    def h(z, u=None):
        return z[idx]

    return h


def linear(C):
    """``y = C @ x`` with ``x`` the first ``C.shape[1]`` entries of ``z`` (the states)."""
    C = jnp.atleast_2d(jnp.asarray(C, dtype=float))
    n = C.shape[1]

    def h(z, u=None):
        return C @ z[:n]

    return h


def states_and_rates(model: SparseModel, state_indices: Sequence[int], rate_indices: Sequence[int]):
    """Observe some states directly and some state derivatives through the model.

    Example: displacement and velocity measured, acceleration from the identified dynamics.
    """
    s_idx, r_idx = jnp.asarray(state_indices), jnp.asarray(rate_indices)

    def h(z, u=None):
        x, p, xi = model.split(z)
        return jnp.concatenate([x[s_idx], model.rhs(x, p, u, xi)[r_idx]])

    return h
