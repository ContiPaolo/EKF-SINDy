"""Compare the refactored pipelines with the legacy implementation.

References are produced by ``tests/legacy/make_reference.py``; missing ones are skipped.
"""

from pathlib import Path

import cases
import numpy as np
import pytest

REFERENCE_DIR = Path(__file__).parent / "reference"


def load_reference(name):
    path = REFERENCE_DIR / f"{name}.npz"
    if not path.exists():
        pytest.skip(f"no reference for {name}; run tests/legacy/make_reference.py {name}")
    return np.load(path)


def assert_matches(result, ref, rtol, atol):
    np.testing.assert_allclose(result.mean, ref["mean"].T, rtol=rtol, atol=atol)
    np.testing.assert_allclose(result.cov, np.moveaxis(ref["cov"], -1, 0), rtol=rtol,
                               atol=atol * 1e-2)


@pytest.mark.parametrize("name, poor", [("selkov", False), ("selkov_poor", True)])
def test_selkov(name, poor):
    ref = load_reference(name)
    out = cases.selkov(poor=poor)
    np.testing.assert_array_equal(out["y"], ref["y"])
    np.testing.assert_allclose(out["z0"], ref["z0"])
    assert_matches(out["result"], ref, rtol=1e-6, atol=1e-9)


def test_coupling():
    ref = load_reference("coupling")
    out = cases.coupled("coupling")
    np.testing.assert_allclose(out["y"], ref["y"][: len(out["y"])])
    np.testing.assert_allclose(out["z0"], ref["z0"])
    assert_matches(out["result"], ref, rtol=1e-6, atol=1e-9)


def test_stiffness():
    """The legacy Jacobian had a bug in d/dk, so only near-equality of the estimates holds."""
    ref = load_reference("stiffness")
    out = cases.coupled("stiffness")
    np.testing.assert_allclose(out["z0"], ref["z0"])
    k_new, k_old = out["result"].params[:, 0], ref["mean"][-1]
    true_k = float(out["true_param"][0])
    late = slice(len(k_new) // 2, None)
    assert np.max(np.abs(k_new - k_old)) < 0.05 * true_k
    assert abs(np.mean(k_new[late]) - true_k) <= abs(np.mean(k_old[late]) - true_k) + 0.01 * true_k


def test_shear():
    """The shear SINDy regression is ill-conditioned: rounding-level differences in the scaled
    training data move large, nearly collinear coefficients without changing the dynamics, so
    the filter output is compared relative to the scale of each state."""
    ref = load_reference("shear")
    out = cases.shear()
    np.testing.assert_allclose(out["z0"], ref["z0"])
    mean, ref_mean = out["result"].mean, ref["mean"].T
    scale = np.abs(ref_mean).max(axis=0)
    assert np.all(np.abs(mean - ref_mean).max(axis=0) <= 1e-6 * scale)
    std, ref_std = out["result"].std, np.sqrt(np.diagonal(ref["cov"], axis1=0, axis2=1))
    np.testing.assert_allclose(std, ref_std, rtol=1e-5)
