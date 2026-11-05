import sys
from pathlib import Path

import jax
import numpy as np
import pysindy as ps
import pytest
import sympy as sp

from ekfsindy import SparseModel, SymbolicLibrary

sys.path.insert(0, str(Path(__file__).parent / "legacy"))

rng = np.random.default_rng(0)


@pytest.mark.parametrize("n_vars, include_bias", [(2, True), (5, False), (6, False)])
def test_polynomial_matches_pysindy(n_vars, include_bias):
    names = [f"x{i}" for i in range(n_vars)]
    X = rng.normal(size=(50, n_vars))
    reference = ps.PolynomialLibrary(degree=3, include_bias=include_bias).fit(X)
    library = SymbolicLibrary.polynomial(names, degree=3, include_bias=include_bias)
    np.testing.assert_allclose(np.asarray(library(X)), reference.transform(X), rtol=1e-12)
    parsed = SymbolicLibrary.from_feature_names(reference.get_feature_names(names), names)
    assert parsed.expressions == library.expressions


def test_from_feature_names_rejects_unknown_symbols():
    with pytest.raises(ValueError):
        SymbolicLibrary.from_feature_names(["x0 y"], ["x0"])
    with pytest.raises(ValueError):
        SymbolicLibrary.from_feature_names(["__import__('os')"], ["x0"])


def test_jacobian_matches_symbolic_and_finite_differences():
    library = SymbolicLibrary.polynomial(["a", "b", "c"], degree=3)
    z = rng.normal(size=3)
    J = np.asarray(library.jacobian(z))
    J_sym = np.array(library.symbolic_jacobian().subs(dict(zip(library.variables, z))),
                     dtype=float)
    np.testing.assert_allclose(J, J_sym, rtol=1e-12, atol=1e-14)
    eps = 1e-6
    J_fd = np.column_stack([(np.asarray(library(z + eps * e)) - np.asarray(library(z - eps * e)))
                            / (2 * eps) for e in np.eye(3)])
    np.testing.assert_allclose(J, J_fd, rtol=1e-6, atol=1e-8)


def test_to_pysindy_matches_compiled_library():
    variables = ["x1", "x2", "w0", "F1"]
    library = SymbolicLibrary(variables, [1, "x1", "x2*w0", "F1*x1**2", "sin(x2)"])
    X = rng.normal(size=(20, 4))
    custom = library.to_pysindy().fit(X)
    np.testing.assert_allclose(custom.transform(X), np.asarray(library(X)), rtol=1e-12)
    assert custom.get_feature_names() == list(library.names)


@pytest.mark.parametrize("n_params", [1, 2])
def test_augmented_jacobian_vs_legacy(n_params):
    """AD Jacobian equals the hand-coded legacy one (up to its known bug in the k columns)."""
    import utils_EKFSINDy as legacy

    names = ["x1", "x2", "x3", "x4"] + ["p1", "p2"][:n_params]
    library = SymbolicLibrary.polynomial(names, degree=3, include_bias=False)
    coefficients = rng.normal(size=(4, library.n_features))
    model = SparseModel(library, coefficients, n_states=4, n_params=n_params)
    z = rng.normal(size=4 + n_params)

    F_ad = np.asarray(jax.jacfwd(model.augmented_rhs)(z))
    F_legacy = legacy.jacobian_A(coefficients, z)
    if n_params == 2:
        np.testing.assert_allclose(F_ad, F_legacy, rtol=1e-12, atol=1e-12)
    else:
        # Legacy d(x_i k^2)/dk is x_i k instead of 2 x_i k: compare after fixing those entries.
        x, k = sp.symbols("x1:5"), sp.Symbol("p1")
        buggy = [library.expressions.index(xi * k**2) for xi in x]
        J_fixed = np.array(library.symbolic_jacobian().subs(dict(zip(library.variables, z))),
                           dtype=float)
        J_fixed[buggy, -1] = z[:4] * z[-1]
        np.testing.assert_allclose(coefficients @ J_fixed, F_legacy[:4], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(F_ad[:4, :4], F_legacy[:4, :4], rtol=1e-12, atol=1e-12)


def test_estimated_coefficients_scatter():
    library = SymbolicLibrary.polynomial(["u1", "u2"], degree=2)
    coefficients = np.zeros((2, library.n_features))
    coefficients[0, 0], coefficients[1, 2] = 0.5, -1.0
    model = SparseModel(library, coefficients, n_states=2, estimated=[(0, 0), (1, 2)])
    np.testing.assert_allclose(model.estimated_values, [0.5, -1.0])
    x = np.array([0.3, -0.7])
    np.testing.assert_allclose(model.rhs(x, xi=np.array([2.0, 3.0])), [2.0, 3.0 * x[1]])
    z = model.augmented_state(x)
    F = np.asarray(jax.jacfwd(model.augmented_rhs)(z))
    np.testing.assert_allclose(F[:2, 2:], [[1.0, 0.0], [0.0, x[1]]])
    np.testing.assert_allclose(F[2:], 0.0)
