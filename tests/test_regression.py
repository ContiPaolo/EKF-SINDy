"""Compare the refactored pipelines with the legacy implementation.

References are produced by ``tests/legacy/make_reference.py`` (which needs the original
environment: pysindy 1.7.5, numpy 1.26); missing ones are skipped. The current environment
(pysindy 2, numpy 2, newer scipy) differs at rounding level in the data generation and SINDy
fits, so estimates are compared relative to the scale of each augmented-state entry.
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


def assert_matches(result, ref, rtol):
    """Mean within ``rtol`` of each entry's range; standard deviations within ``rtol``."""
    mean, ref_mean = result.mean, ref["mean"].T
    scale = np.abs(ref_mean).max(axis=0)
    error = np.abs(mean - ref_mean).max(axis=0)
    assert np.all(error <= rtol * scale), f"relative errors {error / scale}"
    ref_std = np.sqrt(np.diagonal(ref["cov"], axis1=0, axis2=1))
    np.testing.assert_allclose(result.std, ref_std, rtol=10 * rtol)


@pytest.mark.parametrize("name, poor", [("selkov", False), ("selkov_poor", True)])
def test_selkov(name, poor):
    ref = load_reference(name)
    out = cases.selkov(poor=poor)
    np.testing.assert_allclose(out["y"], ref["y"], rtol=1e-12)
    np.testing.assert_allclose(out["z0"], ref["z0"])
    assert_matches(out["result"], ref, rtol=1e-6)


def test_coupling():
    ref = load_reference("coupling")
    out = cases.coupled("coupling")
    np.testing.assert_allclose(out["y"], ref["y"][: len(out["y"])], rtol=1e-10)
    np.testing.assert_allclose(out["z0"], ref["z0"])
    assert_matches(out["result"], ref, rtol=1e-6)


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
    """The shear SINDy regression is ill-conditioned (the two forcing features are collinear):
    rounding-level differences move large coefficients without changing the dynamics."""
    ref = load_reference("shear")
    out = cases.shear()
    np.testing.assert_allclose(out["z0"], ref["z0"])
    assert_matches(out["result"], ref, rtol=1e-6)
