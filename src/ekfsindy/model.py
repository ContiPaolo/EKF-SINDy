"""Sparse dynamics ``dx/dt = Xi @ Theta(x, p, u)`` with optionally estimated coefficients."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import jax.numpy as jnp
import numpy as np
import sympy as sp

from .library import SymbolicLibrary


class SparseModel:
    """SINDy model whose library inputs are ``z = [x, p, u]``.

    - ``x`` (``n_states``): dynamic states, one equation each.
    - ``p`` (``n_params``): physical parameters that enter the library as variables.
    - ``u`` (``n_inputs``): known external inputs (e.g. forcing).

    ``estimated`` lists the ``(equation, feature)`` entries of ``coefficients`` that are treated
    as unknowns ``xi`` (e.g. tracked online by the EKF); all other entries stay frozen.
    """

    def __init__(
        self,
        library: SymbolicLibrary,
        coefficients: np.ndarray,
        n_states: int,
        n_params: int = 0,
        n_inputs: int = 0,
        estimated: Iterable[tuple[int, int]] = (),
        state_names: Sequence[str] | None = None,
    ):
        coefficients = np.asarray(coefficients, dtype=float)
        if coefficients.shape != (n_states, library.n_features):
            raise ValueError(f"coefficients must have shape {(n_states, library.n_features)}, "
                             f"got {coefficients.shape}")
        if library.n_variables != n_states + n_params + n_inputs:
            raise ValueError("library variables must be the states, parameters and inputs")
        self.library = library
        self.coefficients = coefficients
        self.n_states, self.n_params, self.n_inputs = n_states, n_params, n_inputs
        self.estimated = tuple((int(i), int(j)) for i, j in estimated)
        names = state_names or [str(v) for v in library.variables[:n_states]]
        self.state_names = tuple(names)

    @classmethod
    def from_pysindy(
        cls,
        model,
        library: SymbolicLibrary,
        n_params: int = 0,
        n_inputs: int = 0,
        estimated: str | Iterable[tuple[int, int]] = (),
        coefficients: np.ndarray | None = None,
    ) -> SparseModel:
        """Wrap a fitted ``pysindy.SINDy`` model.

        ``estimated="nonzero"`` tracks every nonzero coefficient, ordered row by row
        (the order of ``np.nonzero``).
        """
        coefficients = model.coefficients() if coefficients is None else coefficients
        if isinstance(estimated, str):
            if estimated != "nonzero":
                raise ValueError("estimated must be 'nonzero' or a list of (eq, feature) pairs")
            estimated = zip(*np.nonzero(coefficients))
        n_states = coefficients.shape[0]
        return cls(library, coefficients, n_states, n_params, n_inputs, estimated)

    # ---------------------------------------------------------------- sizes and layout
    @property
    def n_estimated(self) -> int:
        return len(self.estimated)

    @property
    def n_augmented(self) -> int:
        """Size of the augmented state ``[x, p, xi]``."""
        return self.n_states + self.n_params + self.n_estimated

    def split(self, z):
        """Split an augmented state into ``(x, p, xi)``."""
        nx, npar = self.n_states, self.n_params
        return z[:nx], z[nx:nx + npar], z[nx + npar:]

    @property
    def estimated_values(self) -> np.ndarray:
        """Current values of the estimated coefficients (the initial ``xi``)."""
        if not self.estimated:
            return np.zeros(0)
        rows, cols = zip(*self.estimated)
        return self.coefficients[list(rows), list(cols)]

    @property
    def estimated_names(self) -> list[str]:
        return [f"{self.state_names[i]}: {self.library.names[j]}" for i, j in self.estimated]

    # ---------------------------------------------------------------- evaluation
    def coefficient_matrix(self, xi=None):
        """Coefficients with the estimated entries replaced by ``xi``."""
        Xi = jnp.asarray(self.coefficients)
        if xi is None or not self.estimated:
            return Xi
        rows, cols = (np.array(a) for a in zip(*self.estimated))
        return Xi.at[rows, cols].set(xi)

    def rhs(self, x, p=None, u=None, xi=None):
        """``dx/dt = Xi(xi) @ Theta(x, p, u)``; jax-traceable in every argument."""
        parts = [jnp.atleast_1d(x)]
        if self.n_params:
            parts.append(jnp.atleast_1d(p))
        if self.n_inputs:
            parts.append(jnp.atleast_1d(u))
        theta = self.library.evaluate(jnp.concatenate(parts))
        return self.coefficient_matrix(xi) @ theta

    def augmented_rhs(self, z, u=None):
        """Dynamics of ``z = [x, p, xi]``: parameters and coefficients are constant."""
        x, p, xi = self.split(z)
        dx = self.rhs(x, p, u, xi)
        return jnp.concatenate([dx, jnp.zeros(self.n_params + self.n_estimated, dtype=dx.dtype)])

    def augmented_state(self, x0, p0=None, xi0=None) -> np.ndarray:
        """Initial ``z0 = [x0, p0, xi0]``; ``xi0`` defaults to the current coefficients."""
        xi0 = self.estimated_values if xi0 is None else xi0
        p0 = np.zeros(0) if p0 is None else p0
        return np.concatenate([np.atleast_1d(x0), np.atleast_1d(p0), np.atleast_1d(xi0)]).astype(float)

    # ---------------------------------------------------------------- display
    def equations(self, precision: int = 3, xi=None) -> list[sp.Expr]:
        """Symbolic right-hand sides with coefficients rounded to ``precision`` digits."""
        Xi = np.round(np.asarray(self.coefficient_matrix(xi)), precision)
        return [sp.Add(*(float(c) * t for c, t in zip(row, self.library.expressions) if c != 0))
                for row in Xi]

    def print(self, precision: int = 3, xi=None) -> None:
        for name, eq in zip(self.state_names, self.equations(precision, xi)):
            print(f"({name})' = {eq}")
