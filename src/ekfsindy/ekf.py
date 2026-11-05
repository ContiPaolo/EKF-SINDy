"""Extended Kalman filter on the augmented state ``z = [x, p, xi]`` of a ``SparseModel``."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from .model import SparseModel

Observation = Callable[[jax.Array, jax.Array], jax.Array]


@dataclass(frozen=True)
class EKFResult:
    """Filter history, time along the first axis.

    ``mean[k]`` and ``cov[k]`` are the corrected estimates at step ``k`` (``k = 0`` is the
    initial condition) and ``obs[k]`` the observation predicted from the prior at step ``k``.
    """

    mean: np.ndarray  # (n_steps, n_z)
    cov: np.ndarray  # (n_steps, n_z, n_z)
    obs: np.ndarray  # (n_steps, n_obs)
    n_states: int
    n_params: int
    n_estimated: int

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(np.diagonal(self.cov, axis1=1, axis2=2))

    def bounds(self, n_sigma: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
        """Lower and upper credibility bounds ``mean -/+ n_sigma * std``."""
        return self.mean - n_sigma * self.std, self.mean + n_sigma * self.std

    @property
    def states(self) -> np.ndarray:
        return self.mean[:, :self.n_states]

    @property
    def params(self) -> np.ndarray:
        return self.mean[:, self.n_states:self.n_states + self.n_params]

    @property
    def coefficients(self) -> np.ndarray:
        return self.mean[:, self.n_states + self.n_params:]


class AugmentedEKF:
    """Continuous-discrete EKF for ``dz/dt = f(z, u)`` observed through ``y = h(z, u) + v``.

    The transition Jacobian ``F = df/dz`` and observation Jacobian ``H = dh/dz`` are obtained
    with ``jax.jacfwd`` through the symbolic SINDy library, so any library works unchanged.

    Prediction from step ``k-1`` to ``k`` uses the input ``u[k-1]``; the correction at step ``k``
    uses ``u[k]``. The covariance is propagated with the same scheme as the mean:

    - ``"euler"``: ``P + dt (F P + P F^T + Q)`` with ``F`` at the current estimate.
    - ``"rk4"``: classical Runge-Kutta on ``dP/dt = F P + P F^T + Q`` with ``F`` evaluated
      at the Runge-Kutta stages of the mean.
    """

    def __init__(
        self,
        model: SparseModel,
        observation: Observation,
        Q,
        R,
        dt: float,
        integrator: Literal["euler", "rk4"] = "euler",
    ):
        if integrator not in ("euler", "rk4"):
            raise ValueError("integrator must be 'euler' or 'rk4'")
        self.model = model
        self.observation = observation
        self.Q = jnp.asarray(Q, dtype=float)
        self.R = jnp.atleast_2d(jnp.asarray(R, dtype=float))
        self.dt = float(dt)
        self.integrator = integrator
        n_z = model.n_augmented
        if self.Q.shape != (n_z, n_z):
            raise ValueError(f"Q must have shape {(n_z, n_z)} (augmented state), got {self.Q.shape}")

        self._f = model.augmented_rhs
        self._F = jax.jacfwd(model.augmented_rhs)
        self._H = jax.jacfwd(observation)
        self.predict = jax.jit(self._predict)
        self.update = jax.jit(self._update)

    # ---------------------------------------------------------------- filter steps
    def _riccati(self, F, P):
        return F @ P + P @ F.T + self.Q

    def _predict(self, z, P, u):
        dt, f, J = self.dt, self._f, self._F
        if self.integrator == "euler":
            return z + dt * f(z, u), P + dt * self._riccati(J(z, u), P)

        k1 = f(z, u)
        k2 = f(z + dt / 2 * k1, u)
        k3 = f(z + dt / 2 * k2, u)
        k4 = f(z + dt * k3, u)
        z_pred = z + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        P1 = self._riccati(J(z, u), P)
        P2 = self._riccati(J(z + dt / 2 * k1, u), P + dt / 2 * P1)
        P3 = self._riccati(J(z + dt / 2 * k2, u), P + dt / 2 * P2)
        P4 = self._riccati(J(z + dt * k3, u), P + dt * P3)
        P_pred = P + dt / 6 * (P1 + 2 * P2 + 2 * P3 + P4)
        return z_pred, P_pred

    def _update(self, z_pred, P_pred, y, u):
        y_pred = jnp.atleast_1d(self.observation(z_pred, u))
        H = jnp.atleast_2d(self._H(z_pred, u))
        S = H @ P_pred @ H.T + self.R
        G = jnp.linalg.solve(S, H @ P_pred).T  # Kalman gain P H^T S^-1 (S, P symmetric)
        z = z_pred + G @ (y - y_pred)
        IGH = jnp.eye(z.shape[0]) - G @ H
        P = IGH @ P_pred @ IGH.T + G @ self.R @ G.T  # Joseph form
        return z, P, y_pred

    # ---------------------------------------------------------------- full run
    def run(self, z0, P0, y, u=None) -> EKFResult:
        """Filter the observations ``y`` (``(n_steps, n_obs)``) starting from ``(z0, P0)``."""
        y = jnp.asarray(y, dtype=float)
        y = y[:, None] if y.ndim == 1 else y
        n_steps = y.shape[0]
        u = jnp.zeros((n_steps, 0)) if u is None else jnp.asarray(u, dtype=float)
        u = u[:, None] if u.ndim == 1 else u
        z0 = jnp.asarray(z0, dtype=float)
        P0 = jnp.asarray(P0, dtype=float)

        def step(carry, inputs):
            z, P = carry
            y_k, u_prev, u_k = inputs
            z_pred, P_pred = self._predict(z, P, u_prev)
            z, P, y_pred = self._update(z_pred, P_pred, y_k, u_k)
            return (z, P), (z, P, y_pred)

        @jax.jit
        def scan(z0, P0, y, u):
            _, (zs, Ps, ys) = jax.lax.scan(step, (z0, P0), (y[1:], u[:-1], u[1:]))
            y0 = jnp.atleast_1d(self.observation(z0, u[0]))
            return (jnp.concatenate([z0[None], zs]), jnp.concatenate([P0[None], Ps]),
                    jnp.concatenate([y0[None], ys]))

        zs, Ps, ys = scan(z0, P0, y, u)
        m = self.model
        return EKFResult(np.asarray(zs), np.asarray(Ps), np.asarray(ys),
                         m.n_states, m.n_params, m.n_estimated)
