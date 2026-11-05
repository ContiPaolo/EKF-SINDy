"""The notebook pipelines written with ``ekfsindy`` (used by the regression tests)."""

import numpy as np
import pysindy as ps
from scipy.integrate import odeint
from sklearn.model_selection import train_test_split

from ekfsindy import AugmentedEKF, SparseModel, SymbolicLibrary, observation, systems
from ekfsindy.data import TimeDelayEmbedding, add_noise_with_snr, read_dof_csv

SEED = 1


def selkov(poor=False):
    system = {"rho": 0.92, "sigma1": 0.10, "sigma2": 0.10, "k2": 1.0, "b122_1": 1.0,
              "b122_2": 1.0, "t0": 0.0, "T_train": 100.0, "dt": 0.10}
    t_train = np.arange(system["t0"], system["T_train"], system["dt"])
    np.random.seed(SEED)
    u1_0, u2_0 = np.random.normal(0.2, 2, 20), np.random.normal(0.2, 2, 20)
    X_train = [odeint(systems.selkov, [u1_0[i], u2_0[i]], t_train, args=(system,))
               for i in range(16)]

    sindy = ps.SINDy(feature_names=["u1", "u2"], feature_library=ps.PolynomialLibrary(degree=3),
                     optimizer=ps.STLSQ(threshold=0.2 if poor else 5e-2))
    sindy.fit(X_train, t=system["dt"], multiple_trajectories=True)

    system.update(rho_start=0.92, rho_end=0.70, T1=50.0, T_sweep=50.0, T=300.0)
    t = np.arange(system["t0"], system["T"], system["dt"])
    np.random.seed(SEED)
    y0 = [np.random.normal(0.2, 2, 1)[0], np.random.normal(0.2, 2, 1)[0]]
    X = odeint(systems.selkov, y0, t, args=(system, True))
    Y = np.column_stack([add_noise_with_snr(X[:, 0], 25), add_noise_with_snr(X[:, 1], 25)])

    library = SymbolicLibrary.from_feature_names(sindy.get_feature_names(), ["u1", "u2"])
    coefficients = sindy.coefficients().copy()
    if poor:
        coefficients[0, 1] = coefficients[1, 1] = 1e-9
        p0_xi = [5e-4, 1e-3, 1e-3, 1e-4, 5e-4, 1e-3]
        q_xi = [5e-6, 1e-9, 1e-10, 1e-11, 1e-9, 1e-12]
    else:
        p0_xi = [5e-4, 1e-3, 5e-4, 1e-3, 1e-4, 5e-4, 1e-3]
        q_xi = [5e-6, 1e-9, 1e-10, 1e-10, 1e-11, 1e-9, 1e-12]
    model = SparseModel.from_pysindy(sindy, library, estimated="nonzero", coefficients=coefficients)

    ekf = AugmentedEKF(model, observation.select_states([0, 1]),
                       Q=np.diag([8e-7, 8e-7] + q_xi), R=np.diag([5e-4, 5e-4]),
                       dt=system["dt"], integrator="rk4")
    z0 = model.augmented_state(Y[0])
    P0 = np.diag([1e-8, 1e-8] + p0_xi)
    return {"t": t, "y": Y, "z0": z0, "P0": P0, "result": ekf.run(z0, P0, Y), "model": model}


def coupled(case):
    system = {"k1": 1.0, "c1": 2e-2, "c2": 1.95e-2, "gamma": 1e-3, "t0": 0.0, "T_end": 200.0,
              "dt": 1e-2}
    np.random.seed(SEED)
    if case == "stiffness":
        system.update(alpha=-1e-1, beta=2e-3)
        n_ics = 16
        u1_0, u2_0 = np.random.normal(-2, 1, n_ics), np.random.normal(3, 1, n_ics)
        params = np.linspace(1.0, 2.0, n_ics)[:, None]
        param_names = ["k2"]
    else:
        from pyDOE import lhs

        system.update(k2=1.95**2)
        n_ics = 20
        u1_0, u2_0 = np.random.normal(-2, 1, n_ics), np.random.normal(3, 1, n_ics)
        samples = lhs(2, samples=n_ics)
        log_range = np.log(0.005) + (np.log(0.5) - np.log(0.005)) * samples
        params = np.column_stack([-np.exp(log_range[:, 0]), np.exp(log_range[:, 1])])
        param_names = ["alpha", "beta"]

    dt = system["dt"]
    t = np.arange(system["t0"], system["T_end"], dt)
    X = []
    for i in range(n_ics):
        system.update(dict(zip(param_names, params[i])))
        X.append(odeint(systems.coupled_oscillators, [u1_0[i], 0.0, u2_0[i], 0.0], t,
                        args=(system,)))
    X_train, X_test, p_train, p_test = train_test_split(np.array(X), params, test_size=0.2,
                                                        random_state=SEED)

    tde = TimeDelayEmbedding(length=200, n_coordinates=4).fit(X_train[:, :, 0])
    T_h = tde.n_samples(len(t))
    Z_train = [tde.transform(x[:, 0]) for x in X_train]
    dZ_train = [np.gradient(z, dt, axis=0) for z in Z_train]
    P_train = [np.tile(p, (T_h, 1)) for p in p_train]

    variables = ["x1", "x2", "x3", "x4"] + param_names
    library = SymbolicLibrary.polynomial(variables, degree=3, include_bias=False)
    sindy = ps.SINDy(feature_library=ps.PolynomialLibrary(degree=3, include_bias=False),
                     feature_names=variables,
                     optimizer=ps.STLSQ(threshold=5e-4 if case == "stiffness" else 1e-3))
    sindy.fit(Z_train, t=dt, multiple_trajectories=True, u=P_train, x_dot=dZ_train)
    model = SparseModel.from_pysindy(sindy, library, n_params=len(param_names))

    idx_test = 0
    y = add_noise_with_snr(X_test[idx_test, :, 0], 15)[:T_h]
    if case == "stiffness":
        p_init = p_test[idx_test] * 1.2
        P0 = np.diag([1e-6] * 4 + [1e-2])
        Q = np.diag([1e-10, 1e-10, 1e-8, 1e-8, 2e-6])
        R = [[5e-4]]
    else:
        p_init = p_test[idx_test] * np.array([1 - 0.9, 1 + 19])
        P0 = np.diag([1e-6] * 4 + [1e-1, 1e-1])
        Q = np.diag([5e-10, 5e-10, 8e-8, 8e-8, 1e-11, 1e-11])
        R = [[3e-1]]

    ekf = AugmentedEKF(model, observation.linear(tde.observation_matrix), Q=Q, R=R, dt=dt)
    z0 = model.augmented_state(tde.transform(X_test[idx_test, :, 0])[0], p_init)
    return {"t": t[:T_h], "y": y, "z0": z0, "P0": P0, "result": ekf.run(z0, P0, y), "model": model,
                "true_param": p_test[idx_test]}


def shear(root=None):
    T_train, T_test, idx_test = 1000, 40960, 9
    U_train = read_dof_csv("_U_concat_", "train", T_train, 2, root)
    V_train = read_dof_csv("_V_concat_", "train", T_train, 2, root)
    A_train = read_dof_csv("_A_concat_", "train", T_train, 2, root)
    theta_train = read_dof_csv("theta_", "train", T_train, 1, root)
    F_train = read_dof_csv("F_", "train", T_train, 2, root)

    np.random.seed(SEED)
    U_test = add_noise_with_snr(read_dof_csv("_U_concat_", "test", T_test, 2, root), 15)
    V_test = add_noise_with_snr(read_dof_csv("_V_concat_", "test", T_test, 2, root), 15)
    A_test = add_noise_with_snr(read_dof_csv("_A_concat_", "test", T_test, 2, root), 15)
    theta_test = read_dof_csv("theta_", "test", T_test, 1, root)
    F_test = read_dof_csv("F_", "test", T_test, 2, root)

    scale_V, scale_F = np.max(np.abs(V_train)), np.max(np.abs(F_train))
    scale_theta = np.max(np.abs(theta_train))
    scale_x = np.max(np.abs(U_train / scale_V))

    variables = ["x1", "x2", "v1", "v2", "w0", "F1", "F2"]
    linear_terms = SymbolicLibrary(variables, variables[:])
    quadratic = SymbolicLibrary.polynomial(variables[:5], degree=2, include_bias=False)
    library = linear_terms + SymbolicLibrary(variables, quadratic.expressions[5:])

    X = [np.concatenate([U_train[i] / scale_V / scale_x, V_train[i] / scale_V], axis=1)
         for i in range(len(U_train))]
    X_dot = [np.concatenate([V_train[i] / scale_V / scale_x, A_train[i] / scale_V], axis=1)
             for i in range(len(U_train))]
    inputs = [np.concatenate([theta_train[i] / scale_theta, F_train[i] / scale_F], axis=1)
              for i in range(len(U_train))]
    sindy = ps.SINDy(feature_library=library.to_pysindy(), optimizer=ps.STLSQ(threshold=1e-2))
    sindy.fit(X, t=1e-3, u=inputs, multiple_trajectories=True, x_dot=X_dot)
    model = SparseModel.from_pysindy(sindy, library, n_params=1, n_inputs=2)

    y = np.concatenate([U_test[idx_test] / scale_V / scale_x, V_test[idx_test] / scale_V,
                        A_test[idx_test] / scale_V], axis=1)
    u = F_test[idx_test] / scale_F
    theta = theta_test[idx_test, 0, 0] / scale_theta

    h = observation.states_and_rates(model, state_indices=[0, 1, 2, 3], rate_indices=[2, 3])
    ekf = AugmentedEKF(model, h, Q=np.diag([1e-4] * 4 + [1e-8]),
                       R=np.diag([5e-1] * 2 + [5e-3] * 2 + [5e0] * 2), dt=1e-3)
    z0 = model.augmented_state(y[0, :4], theta * 1.2)
    P0 = np.diag([1e-5] * 4 + [2e-5])
    return {"y": y, "u": u, "z0": z0, "P0": P0, "result": ekf.run(z0, P0, y, u), "model": model}
