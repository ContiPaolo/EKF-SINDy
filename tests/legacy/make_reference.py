"""Run the original (pre-refactor) notebook pipelines and store their outputs.

The code below mirrors the notebook cells as they were before the refactor, using the
legacy utilities in this folder. Outputs are written to ``tests/reference/<case>.npz`` and
used by ``tests/test_regression.py``.

It needs the original environment (Python <= 3.12, numpy 1.26, pysindy 1.7.5, pyDOE 0.3.8,
setuptools < 81), not the one in ``uv.lock``::

    uv run --isolated --python 3.12 --with "numpy<2" --with "pysindy==1.7.5" \
        --with "pyDOE==0.3.8" --with "setuptools<81" --with scikit-learn --with sympy \
        python tests/legacy/make_reference.py

Usage::

    uv run python tests/legacy/make_reference.py [selkov selkov_poor stiffness coupling shear]
"""

import copy
import sys
from pathlib import Path

import numpy as np
import pysindy as ps
import sympy as sp
from scipy.integrate import odeint
from sklearn.model_selection import train_test_split
from sklearn.utils.extmath import randomized_svd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

REFERENCE_DIR = HERE.parent / "reference"
DATA_DIR = HERE.parents[1] / "data"
SEED = 1


# --------------------------------------------------------------------------------------
# Selkov model (SelkovModel_OnlineLearning.ipynb)
# --------------------------------------------------------------------------------------
def _selkov(poor):
    import utils_EKFSINDy_online as legacy

    system = {"rho": 0.92, "sigma1": 0.10, "sigma2": 0.10, "k2": 1.0, "b122_1": 1.0,
              "b122_2": 1.0, "t0": 0.00, "T_train": 100.0, "dt": 0.10}

    def f(y, t, system=system, train=True):
        u1, u2 = y
        sigma1 = system["sigma1"]
        k2 = system["k2"]
        b112_1 = system["b122_1"]
        b122_2 = system["b122_2"]
        if train:
            rho = system["rho"]
        else:
            rho_start, rho_end, T1, T_sweep = (system["rho_start"], system["rho_end"],
                                               system["T1"], system["T_sweep"])
            if t < T1:
                rho = rho_start
            elif t > T1 + T_sweep:
                rho = rho_end
            else:
                rho = rho_start + (rho_end - rho_start) * (t - T1) / T_sweep
        du1 = rho - sigma1 * u1 - b112_1 * u1 * u2**2
        du2 = sigma1 * u1 - k2 * u2 + b122_2 * u1 * u2**2
        return [du1, du2]

    t_train = np.arange(system["t0"], system["T_train"], system["dt"])
    np.random.seed(seed=SEED)
    u1_0 = np.random.normal(0.2, 2, 20)
    u2_0 = np.random.normal(0.2, 2, 20)
    Xtrain = [odeint(f, [u1_0[i], u2_0[i]], t_train, args=(system, True)) for i in range(16)]

    threshold = 0.2 if poor else 5e-2
    model = ps.SINDy(feature_names=["u1", "u2"], feature_library=ps.PolynomialLibrary(degree=3),
                     optimizer=ps.STLSQ(threshold=threshold))
    model.fit(Xtrain, t=system["dt"], multiple_trajectories=True)
    coeffs = model.coefficients()

    system = {"rho_start": 0.92, "rho_end": 0.70, "sigma1": 0.10, "sigma2": 0.10, "k2": 1.0,
              "b122_1": 1.0, "b122_2": 1.0, "t0": 0.0, "T1": 50.0, "T_sweep": 50.0,
              "T": 300.0, "dt": 0.10}
    t0, T, dt = system["t0"], system["T"], system["dt"]
    t = np.arange(t0, T, dt)
    Nt = len(t)
    np.random.seed(seed=SEED)
    u1_0 = np.random.normal(0.2, 2, 1)
    u2_0 = np.random.normal(0.2, 2, 1)
    X = odeint(f, np.array([u1_0[0], u2_0[0]]), t, args=(system, False))
    X_clean = X.copy()
    X[:, 0] = legacy.add_noise_with_snr(X[:, 0], 25)
    X[:, 1] = legacy.add_noise_with_snr(X[:, 1], 25)

    sindy_library_names = model.get_feature_names()
    u1, u2 = sp.symbols("u1 u2")
    terms = [sp.parse_expr(n.replace("^", "**").replace(" ", "*"), local_dict={"u1": u1, "u2": u2})
             for n in sindy_library_names]
    sindy_library = [sp.lambdify((u1, u2), e) for e in terms]
    sindy_library_du1 = [sp.lambdify((u1, u2), sp.diff(e, u1)) for e in terms]
    sindy_library_du2 = [sp.lambdify((u1, u2), sp.diff(e, u2)) for e in terms]

    n_eqs = N_x = 2
    N_obs = 2
    N_xu = 2

    if poor:
        coeffs[0, 1] = 1e-9
        coeffs[1, 1] = 1e-9
        p0_param = [5e-4, 1e-3, 1e-3, 1e-4, 5e-4, 1e-3]
        q_param = [5e-6, 1e-9, 1e-10, 1e-11, 1e-9, 1e-12]
    else:
        p0_param = [5e-4, 1e-3, 5e-4, 1e-3, 1e-4, 5e-4, 1e-3]
        q_param = [5e-6, 1e-9, 1e-10, 1e-10, 1e-11, 1e-9, 1e-12]

    non_zero = np.nonzero(coeffs)
    eqOfInterest = non_zero[0]
    keepEntries = np.arange(len(non_zero[0]))
    N_param = len(keepEntries)

    (_, unique_terms, sindy_terms, _, A, d_u1, d_u2) = legacy.extract_coefficients(
        coeffs, keepEntries, eqOfInterest, sindy_library, sindy_library_names,
        sindy_library_du1, sindy_library_du2, n_eqs)
    (_, _, sindy_terms_out, _, A_out, d_u1_out, d_u2_out) = legacy.extract_coefficients(
        coeffs, [], [], sindy_library, sindy_library_names, sindy_library_du1,
        sindy_library_du2, n_eqs)

    h_coeffs = np.zeros(shape=(N_obs, A.shape[1]))
    h_coeffs[0, 1] = 1.0
    h_coeffs[1, 1] = 1.0
    h_coeffs_out = np.zeros(shape=(N_obs, len(unique_terms)))

    p0_u = 1e-8
    q_u_1, q_u_2 = 8e-7, 8e-7
    r_u_1, r_u_2 = 5e-4, 5e-4

    P = np.identity(N_x + N_param)
    P[:N_xu, :N_xu] *= p0_u
    for i in range(N_param):
        P[N_xu + i, N_xu + i] *= p0_param[i]
    Q = np.identity(N_x + N_param)
    Q[0, 0] *= q_u_1
    Q[1, 1] *= q_u_2
    for i in range(N_param):
        Q[N_xu + i, N_xu + i] *= q_param[i]
    R = np.identity(N_obs)
    R[0, 0] *= r_u_1
    R[1, 1] *= r_u_2
    I = np.identity(N_x + N_param)
    P0 = P.copy()

    Aupd = copy.copy(A)
    xhat_taxis = np.zeros((N_x + N_param, Nt))
    obs_taxis = np.zeros((N_obs, Nt))
    P_taxis = np.zeros((N_x + N_param, N_x + N_param, Nt))
    xhat = np.zeros(N_x + N_param)
    xhat[:-N_param] = X[0]
    i1 = 0
    for i0 in range(N_param):
        xhat[N_x + i0] = Aupd[eqOfInterest[i0], i0 - i1]
        if i0 < N_param - 1 and eqOfInterest[i0 + 1] > eqOfInterest[i0]:
            i1 += i0 + 1
    z0 = xhat.copy()
    xhat_taxis[:, 0], obs_taxis[:, 0], P_taxis[:, :, 0] = xhat, xhat[:-N_param], P

    t_plus_1 = t0
    for i0 in range(1, Nt):
        t_ = t_plus_1
        t_plus_1 = t_ + dt
        out = X[i0]
        param = xhat[-N_param:]
        xhat_pred, P_pred = legacy.numInt(xhat[:-N_param], param, dt, t_, sindy_terms,
                                          sindy_terms_out, Aupd, A_out, P, Q, d_u1, d_u2,
                                          d_u1_out, d_u2_out, n_eqs, method="RK4")
        xhat_pred = np.append(xhat_pred, param)
        H = legacy.jacobian_H_Hout(h_coeffs, h_coeffs_out, xhat_pred[:-N_param], param, t_plus_1,
                                   d_u1, d_u2, d_u1_out, d_u2_out, sindy_terms, sindy_terms_out)
        G = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)
        obs = (legacy.model_Asindy(h_coeffs, xhat_pred[:-N_param], t_plus_1, sindy_terms)
               + legacy.model_Asindy(h_coeffs_out, xhat_pred[:-N_param], t_plus_1, sindy_terms_out))
        xhat = xhat_pred + G @ (out - obs)
        P = (I - G @ H) @ P_pred @ (I - G @ H).T + G @ R @ G.T
        xhat_taxis[:, i0], obs_taxis[:, i0], P_taxis[:, :, i0] = xhat, obs, P
        Aupd = legacy.updateA(Aupd, xhat[-N_param:], eqOfInterest, n_eqs)

    return dict(t=t, y=X, y_clean=X_clean, coefficients=coeffs, z0=z0, P0=P0, Q=Q, R=R,
                mean=xhat_taxis, cov=P_taxis, obs=obs_taxis)


# --------------------------------------------------------------------------------------
# Coupled oscillators (CoupledOscillators_*.ipynb)
# --------------------------------------------------------------------------------------
def _create_hankel(vector, length, shift=1):
    p = len(vector) - (length * shift) + 1
    H = vector[0:p]
    for i in range(1, length):
        H = np.vstack([H, vector[(i * shift):(i * shift) + p]])
    return H


def _coupled(case):
    import utils_EKFSINDy as legacy

    system = {"k1": 1.0**2, "c1": 2e-2, "c2": 1.95e-2, "gamma": 1e-3, "t0": 0.0,
              "T_end": 200.0, "dt": 1e-2}
    if case == "stiffness":
        system.update(beta=2e-3, alpha=-1e-1)
    else:
        system.update(k2=1.95**2)

    def f(y, t, system=system):
        u1, v1, u2, v2 = y
        du1 = v1
        dv1 = -system["k1"] * u1 - system["c1"] * v1 - system["alpha"] * u2
        du2 = v2
        dv2 = (-system["k2"] * u2 - system["c2"] * v2 - system["beta"] * (u1**2)
               - system["alpha"] * u1 - system["gamma"] * (u2**3))
        return np.array([du1, dv1, du2, dv2])

    t0, T_end, dt = system["t0"], system["T_end"], system["dt"]
    t = np.arange(t0, T_end, dt)
    T = len(t)

    np.random.seed(seed=SEED)
    if case == "stiffness":
        n_ics = 16
        u1_0 = np.random.normal(-2, 1, n_ics)
        u2_0 = np.random.normal(3, 1, n_ics)
        params = np.linspace(1.0, 2.0, n_ics, endpoint=True)
    else:
        from pyDOE import lhs

        n_ics = 20
        u1_0 = np.random.normal(-2, 1, n_ics)
        u2_0 = np.random.normal(3, 1, n_ics)
        samples = lhs(2, samples=n_ics)
        alpha = np.log(0.005) + (np.log(0.5) - np.log(0.005)) * samples[:, 0]
        beta = np.log(0.005) + (np.log(0.5) - np.log(0.005)) * samples[:, 1]
        params = np.concatenate((-np.exp(alpha).reshape(-1, 1), np.exp(beta).reshape(-1, 1)), axis=1)

    X = []
    for i in range(n_ics):
        y0 = np.array([u1_0[i], 0.0, u2_0[i], 0.0])
        if case == "stiffness":
            system["k2"] = params[i]
        else:
            system["alpha"], system["beta"] = params[i]
        X.append(odeint(f, y0, t))
    X = np.array(X)
    X_train, X_test, param_train, param_test = train_test_split(X, params, test_size=0.2,
                                                                random_state=SEED)

    length, shift = 200, 1
    H = np.hstack([_create_hankel(X_train[i, :, 0].flatten(), length, shift)
                   for i in range(len(X_train))])
    uh, sh, _ = randomized_svd(H, n_components=16)
    n_tdc = 4
    U = uh[:, :n_tdc]
    S = sh[:n_tdc]
    T_h = T - length * shift + 1

    def tdc(x):
        return (_create_hankel(x, length, shift).T @ (U * 1 / S))[:, :n_tdc]

    if case == "stiffness":
        param_tdc = [np.repeat(param_train[i], T)[:T_h] for i in range(len(X_train))]
        param_test_tdc = [np.repeat(param_test[i], T)[:T_h] for i in range(len(X_test))]
        feature_names = [f"u{i + 1}" for i in range(n_tdc)] + ["k2"]
        threshold = 5e-4
    else:
        param_tdc = [np.tile(param_train[i], (T, 1))[:T_h] for i in range(len(X_train))]
        param_test_tdc = [np.tile(param_test[i], (T, 1))[:T_h] for i in range(len(X_test))]
        feature_names = [f"u{i + 1}" for i in range(n_tdc)] + ["a", "b"]
        threshold = 1e-3
    X_tdc = [tdc(X_train[i, :, 0]) for i in range(len(X_train))]
    dX_tdc = [np.gradient(x, dt, axis=0) for x in X_tdc]
    X_test_tdc = [tdc(X_test[i, :, 0]) for i in range(len(X_test))]

    model = ps.SINDy(feature_names=feature_names,
                     feature_library=ps.PolynomialLibrary(degree=3, include_bias=False),
                     optimizer=ps.STLSQ(threshold=threshold))
    model.fit(X_tdc, t=dt, multiple_trajectories=True, u=param_tdc, x_dot=dX_tdc)
    A = model.coefficients()

    idx_test = 0
    correct_param = param_test_tdc[idx_test][0]
    if case == "stiffness":
        state_param = correct_param * (1 + 0.2)
        N_param = 1
    else:
        state_param = np.array([correct_param[0] * (1 - 0.9), correct_param[1] * (1 + 19)])
        N_param = 2
    N_x, N_obs, N_l = n_tdc, 1, T_h

    out_no_noise_axis = X_test[idx_test, :, 0]
    out_axis = legacy.add_noise_with_snr(out_no_noise_axis, 15)

    P = np.identity(N_x + N_param)
    Q = np.identity(N_x + N_param)
    if case == "stiffness":
        P[:2, :2] *= 1e-6
        P[2, 2] *= 1e-6
        P[3, 3] *= 1e-6
        P[-N_param:, -N_param:] *= 1e-2
        Q[:2, :2] *= 1e-10
        Q[2, 2] *= 1e-8
        Q[3, 3] *= 1e-8
        Q[-N_param:, -N_param:] *= 2e-6
        R = np.identity(N_obs) * 5e-4
    else:
        P[:N_x, :N_x] *= 1e-6
        P[-2, -2] *= 1e-1
        P[-1, -1] *= 1e-1
        Q[:2, :2] *= 5e-10
        Q[2, 2] *= 8e-8
        Q[3, 3] *= 8e-8
        Q[-N_param:, -N_param:] *= 1e-11
        R = np.identity(N_obs) * 3e-1
    I = np.identity(N_x + N_param)
    P0 = P.copy()

    xhat_taxis = np.zeros((N_x + N_param, N_l))
    obs_taxis = np.zeros((N_obs, N_l))
    P_taxis = np.zeros((N_x + N_param, N_x + N_param, N_l))
    xhat = np.zeros(N_x + N_param)
    xhat[:N_x] = X_test_tdc[idx_test][0, :]
    xhat[-N_param:] = state_param
    z0 = xhat.copy()
    xhat_taxis[:, 0] = xhat
    obs_taxis[0, 0] = out_axis[0]
    P_taxis[:, :, 0] = P
    e_ = np.zeros(length)
    e_[0] = 1
    H = legacy.jacobian_H(U, S, N_x, n_param=N_param)

    for i0 in range(1, N_l):
        out = out_axis[i0]
        param = xhat[-N_param:]
        xhat_pred = xhat[:-N_param] + dt * model.predict(xhat[:-N_param].reshape(1, -1),
                                                         xhat[-N_param:].reshape(1, -1))
        xhat_pred = np.append(xhat_pred, param)
        F = legacy.jacobian_A(A, variables=xhat)
        P_pred = P + dt * (F @ P + P @ F.T + Q)
        G = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)
        obs = xhat_pred[:-N_param] @ (U * S).T @ e_.T
        xhat = xhat_pred + G @ np.expand_dims(out - obs, axis=0)
        P = (I - G @ H) @ P_pred @ (I - G @ H).T + G @ R @ G.T
        xhat_taxis[:, i0] = xhat
        obs_taxis[0, i0] = obs
        P_taxis[:, :, i0] = P

    return dict(t=t[:N_l], y=out_axis, coefficients=A, U=U, S=S, z0=z0, P0=P0, Q=Q, R=R,
                true_param=np.atleast_1d(correct_param), mean=xhat_taxis, cov=P_taxis,
                obs=obs_taxis)


# --------------------------------------------------------------------------------------
# Shear building (ShearBuilding_test/ShearBuilding.ipynb)
# --------------------------------------------------------------------------------------
def _read_data(prefix, seq_len, n_channels, split):
    for i1 in range(n_channels):
        X_single_dof = np.loadtxt(DATA_DIR / f"{prefix}dof_{i1 + 1}_{split}.csv")
        if i1 == 0:
            n_instances = int(len(X_single_dof) / seq_len)
            X = np.zeros((n_instances, seq_len, n_channels))
        X[:, :, i1] = X_single_dof[: n_instances * seq_len].reshape(n_instances, seq_len)
    return X


def _shear():
    import utils_ShearBuilding as legacy

    N_ou = N_ov = N_oa = 2
    N_o = 6
    N_obs_u = N_obs_v = N_obs_a = 2
    N_obs = 6
    N_f, N_param = 2, 1
    snr = 15

    T_train = 1000
    U_train = _read_data("_U_concat_", T_train, N_ou, "train")
    V_train = _read_data("_V_concat_", T_train, N_ov, "train")
    A_train = _read_data("_A_concat_", T_train, N_oa, "train")
    param_train = _read_data("theta_", T_train, N_param, "train")
    F_train = _read_data("F_", T_train, N_f, "train")

    np.random.seed(SEED)
    T = 40960
    U_test = _read_data("_U_concat_", T, N_ou, "test")
    U_test_noisefree = copy.copy(U_test)
    U_test = legacy.add_noise_with_snr(U_test, snr)
    V_test = _read_data("_V_concat_", T, N_ov, "test")
    V_test_noisefree = copy.copy(V_test)
    V_test = legacy.add_noise_with_snr(V_test, snr)
    A_test = _read_data("_A_concat_", T, N_oa, "test")
    A_test_noisefree = copy.copy(A_test)
    A_test = legacy.add_noise_with_snr(A_test, snr)
    param_test = _read_data("theta_", T, N_param, "test")
    F_test = _read_data("F_", T, N_f, "test")

    scale_V = np.max(np.abs(V_train))
    scale_F = np.max(np.abs(F_train))
    scale_param = np.max(np.abs(param_train))
    U_train_scaled = U_train / scale_V
    scale_x = np.max(np.abs(U_train_scaled))
    x_train = U_train_scaled / scale_x
    dx_train = V_train / scale_V / scale_x
    ddx_train = A_train / scale_V / scale_x
    param_train_scaled = param_train / scale_param
    F_train_scaled = F_train / scale_F

    data_test = {"U": U_test / scale_V, "V": V_test / scale_V, "A": A_test / scale_V,
                 "theta": param_test / scale_param, "F": F_test / scale_F}
    data_test_noisefree = {"U": U_test_noisefree / scale_V, "V": V_test_noisefree / scale_V,
                           "A": A_test_noisefree / scale_V}

    dt = 0.001
    def quadratic(i, j):
        # pysindy's CustomLibrary infers the arity from the explicit argument list
        return lambda x1, x2, x1_dot, x2_dot, w0, F1, F2: (
            (x1, x2, x1_dot, x2_dot, w0)[i] * (x1, x2, x1_dot, x2_dot, w0)[j])

    sindy_library = [lambda x: x] + [quadratic(i, j) for i in range(5) for j in range(i, 5)]
    sindy_library_names = [
        "x1", "x2", "x1'", "x2'", "w0", "F1", "F2",
        "x1^2", "x1*x2", "x1*x1'", "x1*x2'", "x1*w0",
        "x2^2", "x2*x1'", "x2*x2'", "x2*w0",
        "x1'^2", "x1'*x2'", "x1'*w0",
        "x2'^2", "x2'*w0",
        "w0^2",
    ]
    X, X_dot, params = [], [], []
    for i in range(len(x_train)):
        X.append(np.concatenate((x_train[i], dx_train[i] * scale_x), axis=1))
        X_dot.append(np.concatenate((dx_train[i], ddx_train[i] * scale_x), axis=1))
        params.append(np.concatenate((param_train_scaled[i], F_train_scaled[i, :, 0:1],
                                      F_train_scaled[i, :, 1:2]), axis=1))

    model = ps.SINDy(feature_names=sindy_library_names,
                     feature_library=ps.CustomLibrary(library_functions=sindy_library),
                     optimizer=ps.STLSQ(threshold=1e-2))
    model.fit(X, t=dt, u=params, multiple_trajectories=True, x_dot=X_dot)
    coeffs = model.coefficients()
    B = coeffs[:, 5:7]
    A = np.concatenate((coeffs[:, 0:5], coeffs[:, 7:]), axis=1)
    n_features = A.shape[1]

    Hsindy_coeff = np.zeros((N_obs, n_features))
    Hsindy_f_coeff = np.zeros((N_obs, N_f))
    Hsindy_coeff[:N_obs_u, :N_ou] = np.eye(2)
    Hsindy_coeff[N_obs_u:N_obs_u + N_obs_v, N_ou:N_ou + N_ov] = np.eye(2)
    Hsindy_coeff[-N_obs_a:, :] = A[-N_oa:, :]
    Hsindy_f_coeff[-N_obs_a:, :] = B[-N_oa:, :]

    idx_test = 9
    inp = np.concatenate([data_test[k][idx_test] for k in data_test], axis=-1)
    inp[:, 0:N_ou] /= scale_x
    inp_no_noise = np.concatenate([data_test_noisefree[k][idx_test] for k in data_test_noisefree],
                                  axis=-1)
    inp_no_noise[:, 0:N_ou] /= scale_x

    n_z = N_ou + N_ov + N_param
    xhat_taxis = np.zeros((n_z, T))
    a_axis = np.zeros((N_oa, T))
    P_taxis = np.zeros((n_z, n_z, T))
    F_taxis = np.array(inp[:, -N_f:]).T
    P = np.diag([1e-5] * 2 + [1e-5] * 2 + [2e-5])
    Q = np.diag([1e-4] * 2 + [1e-4] * 2 + [1e-8])
    R = np.diag([5e-1] * 2 + [5e-3] * 2 + [5e0] * 2)
    I = np.identity(n_z)
    P0 = P.copy()
    xhat = legacy.ass_xhat(inp, idx_test, 0, N_param, N_ou, N_ov, N_f, -0.2)
    z0 = xhat.copy()
    xhat_taxis[:, 0] = xhat
    P_taxis[:, :, 0] = P

    for k in range(1, T):
        out = legacy.output_obs(inp, k, 2, 2, 2, 2, 2, 2)
        theta = xhat[-N_param:]
        inp_f, inp_f_1 = legacy.ass_inputs_Euler_forward(k, F_taxis, N_f)
        xhat_pred = xhat[:-N_param] + dt * (legacy.model_Asindy(A, variables=xhat)
                                            + legacy.model_Bsindy_f(B, variables=inp_f_1))
        xhat_pred = np.append(xhat_pred, theta)
        F = legacy.jacobian_A(A, variables=xhat)
        P_pred = P + dt * (F @ P + P @ F.T + Q)
        H = legacy.jacobian_H(Hsindy_coeff, variables=xhat_pred)
        G = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)
        obs = (legacy.model_Hsindy(Hsindy_coeff, variables=xhat_pred)
               + legacy.model_Hsindy_f(Hsindy_f_coeff, variables=inp_f))
        xhat = xhat_pred + G @ (out - obs)
        P = (I - G @ H) @ P_pred @ (I - G @ H).T + G @ R @ G.T
        xhat_taxis[:, k] = xhat
        P_taxis[:, :, k] = P
        a_axis[:, k] = obs[-N_oa:]

    return dict(coefficients=coeffs, inp=inp, inp_no_noise=inp_no_noise, z0=z0, P0=P0, Q=Q,
                R=R, mean=xhat_taxis, cov=P_taxis, acc=a_axis,
                scales=np.array([scale_V, scale_F, scale_param, scale_x]))


CASES = {
    "selkov": lambda: _selkov(poor=False),
    "selkov_poor": lambda: _selkov(poor=True),
    "stiffness": lambda: _coupled("stiffness"),
    "coupling": lambda: _coupled("coupling"),
    "shear": _shear,
}


def main(names):
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    for name in names:
        print(f"running legacy pipeline: {name}", flush=True)
        np.savez(REFERENCE_DIR / f"{name}.npz", **CASES[name]())


if __name__ == "__main__":
    main(sys.argv[1:] or list(CASES))
