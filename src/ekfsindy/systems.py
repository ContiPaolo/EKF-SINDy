"""Ground-truth dynamical systems used to generate data in the examples."""

from __future__ import annotations

import numpy as np


def rho_schedule(t, system: dict):
    """Selkov forcing ``rho(t)``: constant, then a linear sweep over ``[T1, T1 + T_sweep]``."""
    rho_start, rho_end = system["rho_start"], system["rho_end"]
    T1, T_sweep = system["T1"], system["T_sweep"]
    ramp = rho_start + (rho_end - rho_start) * (np.asarray(t) - T1) / T_sweep
    return np.where(t < T1, rho_start, np.where(t > T1 + T_sweep, rho_end, ramp))


def selkov(y, t, system: dict, varying_rho: bool = False):
    """Selkov glycolysis model; ``rho`` follows ``rho_schedule`` when ``varying_rho``."""
    u1, u2 = y
    rho = float(rho_schedule(t, system)) if varying_rho else system["rho"]
    du1 = rho - system["sigma1"] * u1 - system["b122_1"] * u1 * u2**2
    du2 = system["sigma2"] * u1 - system["k2"] * u2 + system["b122_2"] * u1 * u2**2
    return [du1, du2]


def coupled_oscillators(y, t, system: dict):
    """Two oscillators with linear (``alpha``) and quadratic (``beta``) coupling."""
    u1, v1, u2, v2 = y
    du1 = v1
    dv1 = -system["k1"] * u1 - system["c1"] * v1 - system["alpha"] * u2
    du2 = v2
    dv2 = (-system["k2"] * u2 - system["c2"] * v2 - system["beta"] * u1**2
           - system["alpha"] * u1 - system["gamma"] * u2**3)
    return np.array([du1, dv1, du2, dv2])
