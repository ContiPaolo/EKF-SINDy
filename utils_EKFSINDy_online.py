import os
import sklearn
import numpy as np
import pandas as pd
import matplotlib
import filterpy
import scipy

from matplotlib import pyplot as plt

def extract_coefficients(coeffs, selected_entries, equation_indices, sindy_library, sindy_library_names, sindy_library_du1, sindy_library_du2, n_eqs):
    """
    Extracts non-zero coefficients and associated SINDy library terms for specific equations.

    Parameters:
    - coeffs: numpy.ndarray, the coefficients identified by SINDy.
    - selected_entries: list, indices of coefficients to extract.
    - equation_indices: numpy.ndarray, array indicating which equations the coefficients belong to.
    - sindy_library: list, SINDy library terms.
    - sindy_library_names: list, names of SINDy library terms.
    - sindy_library_du1: list, derivatives of SINDy terms w.r.t. u1.
    - sindy_library_du2: list, derivatives of SINDy terms w.r.t. u2.
    - n_eqs: int, number of equations.

    Returns:
    - B_map: numpy.ndarray, mapping of coefficients to terms.
    - unique_terms: numpy.ndarray, unique indices of SINDy library terms.
    - sindy_terms_per_eq: list, symbolic SINDy terms for each equation.
    - sindy_term_names_per_eq: list, names of terms for each equation.
    - coeff_matrix: numpy.ndarray, coefficient matrix for the extracted terms.
    - sindy_derivatives_u1: list, derivatives of SINDy terms w.r.t. u1.
    - sindy_derivatives_u2: list, derivatives of SINDy terms w.r.t. u2.
    """
    # Extract nonzero coefficients
    non_zero_indices_all = np.nonzero(coeffs)

    # Extract the ones you want to montior indicated by 'selected_entries'
    non_zero_indices = (
        non_zero_indices_all[0][selected_entries],
        non_zero_indices_all[1][selected_entries],
    )
    unique_terms = np.unique(non_zero_indices[1]) # termini che entrano da entrambe le parti (?)

    term_offset = 0
    B_map = [] #Boolean map
    coeff_matrix = []
    sindy_terms_per_eq = []
    sindy_term_names_per_eq = []
    sindy_derivatives_u1 = []
    sindy_derivatives_u2 = []

    if len(unique_terms) > 0:
        B_map = np.zeros((coeffs.shape[0], coeffs.shape[1], len(unique_terms))) #len(unique_terms): #termini che stiamo monitorando
        coeff_matrix = np.zeros((coeffs.shape[0], len(unique_terms)))

        for eq_idx in range(n_eqs): #definisci dentro
            # Extract indices and terms specific to the current equation
            current_eq_indices = np.array(np.where(eq_idx == equation_indices))  #mette 1 ai termini in base all'equazione dove sono
            # 0 == (0,0,0,1,1,1) --> (1,1,1,0,0,0)
            eq_terms = non_zero_indices[1][current_eq_indices[0]] #prende i termini non zero dell'equazione che ci interessa

            eq_sindy_terms = []
            eq_term_names = []
            eq_derivative_u1 = []
            eq_derivative_u2 = []

            for term_idx in range(len(current_eq_indices[0])):
                # Map coefficients and library terms
                B_map[eq_idx, eq_terms[term_idx], term_idx] = 1.0
                global_term_idx = term_idx

                eq_sindy_terms.append(sindy_library[eq_terms[term_idx]])
                eq_term_names.append(sindy_library_names[eq_terms[term_idx]])
                eq_derivative_u1.append(sindy_library_du1[eq_terms[term_idx]])
                eq_derivative_u2.append(sindy_library_du2[eq_terms[term_idx]])

            if eq_idx > 0:
                sindy_terms_per_eq.append(eq_sindy_terms)
                sindy_term_names_per_eq.append(eq_term_names)
                sindy_derivatives_u1.append(eq_derivative_u1)
                sindy_derivatives_u2.append(eq_derivative_u2)
            else:
                sindy_terms_per_eq = [eq_sindy_terms]
                sindy_term_names_per_eq = [eq_term_names]
                sindy_derivatives_u1 = [eq_derivative_u1]
                sindy_derivatives_u2 = [eq_derivative_u2]

            term_offset += len(current_eq_indices[0])
            coeff_matrix[eq_idx, :] = coeffs[eq_idx, :] @ B_map[eq_idx, :, :]

    return (
        B_map,
        unique_terms,
        sindy_terms_per_eq,
        sindy_term_names_per_eq,
        coeff_matrix,
        sindy_derivatives_u1,
        sindy_derivatives_u2,
    )

# %% define the transition model (coincides with the observation model)
def model_Asindy(A, dyn_state, t_, sindy_terms):
    '''
    A: matrix of SINDy coefficients for the states and parameters (transition equation)
    dyn_state: [u1, u2]
    '''
    u1, u2 = dyn_state

    # Initialize the derivative of the dynamic state
    dyn_state_t_ = np.zeros(dyn_state.shape)

    # If SINDy terms are provided, evaluate them
    if sindy_terms:  # if not empty
        sindy_terms_eval = np.zeros(shape=(A.shape[1]))
        for eq_idx in np.arange(len(sindy_terms)):
            for term_idx in np.arange(len(sindy_terms[eq_idx])):
                sindy_terms_eval[term_idx] = sindy_terms[eq_idx][term_idx](u1, u2)
            # Update the dynamic state derivative
            dyn_state_t_[eq_idx] = A[eq_idx][:] @ sindy_terms_eval

    return dyn_state_t_

# %% update the transition model
def updateA(A, param_state, eqOfInterest):
    param_offset = 0
    # Iterate through each equation
    for eq_idx in np.arange(n_eqs):
        eq_indices = np.array(np.where(eq_idx == eqOfInterest))
        # Update coefficients for each term in the equation
        for term_idx in np.arange(len(eq_indices[0])):
            A[eq_idx, term_idx] = param_state[term_idx + param_offset]
        if eq_idx == 0:
            param_offset += 1
        param_offset += term_idx

    return A

# %% compute the Jacobian of the SINDy library
def compute_J_library(sindy_terms, dyn_state, t_, n_features, sindy_derivatives_u1, sindy_derivatives_u2):
    u1, u2 = dyn_state
    n_state = len(dyn_state)

    # Initialize the Jacobian matrix for the SINDy library
    J_library = np.zeros((n_state, n_features, n_state))

    # Iterate through each state and compute the derivatives
    for state_idx in np.arange(n_state):
        for term_idx in np.arange(len(sindy_terms[state_idx])):
            J_library[state_idx, term_idx, 0] = sindy_derivatives_u1[state_idx][term_idx](u1, u2)  # dA/du1
            J_library[state_idx, term_idx, 1] = sindy_derivatives_u2[state_idx][term_idx](u1, u2)  # dA/du2

    return J_library

# %% create the Jacobian matrix for the dynamics equations
def jacobian_A_A_out(A, A_out, dyn_state, param_state, t_, sindy_terms, sindy_terms_out, sindy_derivatives_u1, sindy_derivatives_u2, sindy_derivatives_u1_out, sindy_derivatives_u2_out, n_eqs):
    u1, u2 = dyn_state
    n = A.shape[0]  # number of equations
    n_features = A.shape[1]  # number of features

    # Compute the Jacobian of the SINDy library for the main terms
    J_library = compute_J_library(sindy_terms, dyn_state, t_, n_features, sindy_derivatives_u1, sindy_derivatives_u2)

    # Compute the Jacobian for the output terms if provided
    if sindy_terms_out:  # if not empty
        n_features_out = A_out.shape[1]  # number of features
        J_library_out = compute_J_library(sindy_terms_out, dyn_state, t_, n_features_out, sindy_derivatives_u1_out, sindy_derivatives_u2_out)

    # Initialize the Jacobian matrix for the dynamics equations
    F = np.zeros(shape=(n, n))
    for eq_idx in np.arange(n):
        # Compute the contribution from the main terms
        F[eq_idx, :] = A[eq_idx][:] @ J_library[eq_idx, :, :]  # F_xx
        # Add the contribution from the output terms if available
        if sindy_terms_out:
            F[eq_idx, :] += A_out[eq_idx][:] @ J_library_out[eq_idx, :, :]

    # Add zero columns for the parameter states
    zero_columns = np.zeros((len(dyn_state), len(param_state)))
    F = np.hstack((F, zero_columns))

    param_offset = 0
    # Update the Jacobian with respect to parameter states
    for eq_idx in np.arange(n_eqs):
        for term_idx in np.arange(len(sindy_terms[eq_idx])):
            F[eq_idx, len(dyn_state) + term_idx + param_offset] += sindy_terms[eq_idx][term_idx](u1, u2)
        if eq_idx == 0:
            param_offset += 1
        param_offset += term_idx

    # Append rows of zeros for parameter state derivatives
    for _ in np.arange(len(param_state)):
        F = np.append(F, [np.zeros(n + len(param_state))], axis=0)

    return F

# %% create the Jacobian matrix for the observation equations
def jacobian_H_Hout(h_coeffs, h_coeffs_out, dyn_state, param_state, t_plus_1, sindy_derivatives_u1, sindy_derivatives_u2, sindy_derivatives_u1_out, sindy_derivatives_u2_out, sindy_terms, sindy_terms_out):
    n = h_coeffs.shape[0]  # number of equations
    n_features = h_coeffs.shape[1]  # number of features

    # Compute the Jacobian of the SINDy library for the main terms
    J_library = compute_J_library(sindy_terms, dyn_state, t_plus_1, n_features, sindy_derivatives_u1, sindy_derivatives_u2)

    # Compute the Jacobian for the output terms if provided
    if sindy_terms_out:
        n_features_out = h_coeffs_out.shape[1]  # number of features
        J_library_out = compute_J_library(sindy_terms_out, dyn_state, t_plus_1, n_features_out, sindy_derivatives_u1_out, sindy_derivatives_u2_out)

    # Initialize the Jacobian matrix for the observation equations
    H = np.zeros(shape=(n, n))
    for eq_idx in np.arange(n):

        # Compute the contribution from the main terms
        H[eq_idx, :] = h_coeffs[eq_idx][:] @ J_library[eq_idx, :, :]  # H_xx
        # Add the contribution from the output terms if available
        if sindy_terms_out:
            H[eq_idx, :] += h_coeffs_out[eq_idx][:] @ J_library_out[eq_idx, :, :]

    # Add zero columns for the parameter states
    zero_columns = np.zeros((n, len(param_state)))
    H = np.hstack((H, zero_columns))

    return H

# %% define numerical integration scheme
def numInt(dyn_state, param_state, dt, t_, sindy_terms, sindy_terms_out, Aupd, A_out, P, Q, sindy_derivatives_u1, sindy_derivatives_u2, sindy_derivatives_u1_out, sindy_derivatives_u2_out, method):

    if method == 'EF':
        # Mean state prediction
        xhat_pred = dyn_state + dt * (model_Asindy(Aupd, dyn_state, t_, sindy_terms) + model_Asindy(A_out, dyn_state, t_, sindy_terms_out))

        # Covariance prediction
        F = jacobian_A_A_out(Aupd, A_out, dyn_state, param_state, t_, sindy_terms, sindy_terms_out, sindy_derivatives_u1, sindy_derivatives_u2, sindy_derivatives_u1_out, sindy_derivatives_u2_out)  # Jacobian computation
        P_pred = P + dt * (np.matmul(F, P) + np.matmul(P, F.transpose()) + Q)

    elif method == 'RK4':
        # Mean state prediction using Runge-Kutta 4th order method
        xk1 = model_Asindy(Aupd, dyn_state, t_, sindy_terms) + model_Asindy(A_out, dyn_state, t_, sindy_terms_out)
        xk2 = model_Asindy(Aupd, dyn_state + (dt / 2) * xk1, t_ + dt / 2, sindy_terms) + model_Asindy(A_out, dyn_state + (dt / 2) * xk1, t_ + dt / 2, sindy_terms_out)
        xk3 = model_Asindy(Aupd, dyn_state + (dt / 2) * xk2, t_ + dt / 2, sindy_terms) + model_Asindy(A_out, dyn_state + (dt / 2) * xk2, t_ + dt / 2, sindy_terms_out)
        xk4 = model_Asindy(Aupd, dyn_state + dt * xk3, t_ + dt, sindy_terms) + model_Asindy(A_out, dyn_state + dt * xk3, t_ + dt, sindy_terms_out)

        xhat_pred = dyn_state + (dt / 6) * (xk1 + 2 * xk2 + 2 * xk3 + xk4)

        # Covariance prediction using Runge-Kutta 4th order method
        F = jacobian_A_A_out(Aupd, A_out, dyn_state, param_state, t_, sindy_terms, sindy_terms_out, sindy_derivatives_u1, sindy_derivatives_u2, sindy_derivatives_u1_out, sindy_derivatives_u2_out)  # Jacobian computation
        Fk1 = jacobian_A_A_out(Aupd, A_out, dyn_state + (dt / 2) * xk1, param_state, t_ + dt / 2, sindy_terms, sindy_terms_out, sindy_derivatives_u1, sindy_derivatives_u2, sindy_derivatives_u1_out, sindy_derivatives_u2_out)
        Fk2 = jacobian_A_A_out(Aupd, A_out, dyn_state + (dt / 2) * xk2, param_state, t_ + dt / 2, sindy_terms, sindy_terms_out, sindy_derivatives_u1, sindy_derivatives_u2, sindy_derivatives_u1_out, sindy_derivatives_u2_out)
        Fk3 = jacobian_A_A_out(Aupd, A_out, dyn_state + dt * xk3, param_state, t_ + dt, sindy_terms, sindy_terms_out, sindy_derivatives_u1, sindy_derivatives_u2, sindy_derivatives_u1_out, sindy_derivatives_u2_out)

        Pk1 = np.matmul(F, P) + np.matmul(P, F.transpose()) + Q
        Pk2 = np.matmul(Fk1, (P + Pk1 * (dt / 2))) + np.matmul(P + Pk1 * (dt / 2), Fk1.transpose()) + Q
        Pk3 = np.matmul(Fk2, (P + Pk2 * (dt / 2))) + np.matmul(P + Pk2 * (dt / 2), Fk2.transpose()) + Q
        Pk4 = np.matmul(Fk3, (P + Pk3 * dt)) + np.matmul(P + Pk3 * dt, Fk3.transpose()) + Q

        P_pred = P + (dt / 6) * (Pk1 + 2 * Pk2 + 2 * Pk3 + Pk4)

    return xhat_pred, P_pred

# %%
def plot_outcomes(t_axis, X_data_obs, xhat_taxis, xhat_taxis_piu_sigma, xhat_taxis_meno_sigma, N_x, N_param, true_coeff, coeff_names, system, ntsp, ntst):
    variable_names = ['$u_1$', '$u_2$']
    for i1 in np.arange(N_x):
        plt.figure(figsize=[6, 3])
        plt.plot(t_axis[ntsp:ntst], X_data_obs[ntsp:ntst, i1], 'r--', label = 'mean estimate')
        plt.plot(t_axis[ntsp:ntst], xhat_taxis[i1, ntsp:ntst], 'k', label = 'true value')
        plt.fill_between(t_axis[ntsp:ntst], xhat_taxis_piu_sigma[i1, ntsp:ntst], xhat_taxis_meno_sigma[i1, ntsp:ntst], color='red', alpha=0.2, label=f"mean $\\pm 1.96\sigma_x$")
        plt.xlabel('Time')
        plt.ylabel(f'{variable_names[i1]}')
        plt.title(variable_names[i1])
        plt.legend()
        plt.show()

    for i1 in np.arange(N_param):  # Plot linear stiffness/ coupling terms
        plt.figure(figsize=[6, 3])
        plt.plot(t_axis[ntsp + 1:ntst], xhat_taxis[i1 + N_x, ntsp + 1:ntst], 'k', label = 'true value')
        # Step transition for rho
        if i1 == 0:
          rho_start, rho_end, T, T1, T_sweep  = system['rho_start'], system['rho_end'], system['T'], system['T1'], system['T_sweep']
          rho_step = np.piecewise(t,[t < T1, (t >= T1) & (t <= T1 + T_sweep), t > T1 + T_sweep],[rho_start, lambda t: rho_start + (rho_end - rho_start) * (t - T1) / T_sweep, rho_end])
          plt.plot(t_axis[ntsp + 1:ntst], rho_step[ntsp + 1:ntst], 'r--', label = 'mean estimate')
        else:
            plt.plot(t_axis[ntsp + 1:ntst], np.ones(ntst - ntsp - 1) * true_coeff[i1], 'r--',label = 'mean estimate')
        plt.fill_between(t_axis[ntsp + 1:ntst], xhat_taxis_piu_sigma[i1 + N_x, ntsp + 1:ntst], xhat_taxis_meno_sigma[i1 + N_x, ntsp + 1:ntst], color='red', alpha=0.2, label=f"mean $\\pm 1.96\sigma_x$")
        plt.xlabel('Time')
        plt.ylabel(f'{coeff_names[i1]}')
        plt.title(coeff_names[i1])
        y_limits = [np.min([true_coeff[i1], np.min(xhat_taxis_meno_sigma[i1 + N_x, ntsp + 1:ntst])]),
                    np.max([true_coeff[i1], np.max(xhat_taxis_piu_sigma[i1 + N_x, ntsp:ntst])])]
        plt.ylim(y_limits[0] * (1 - np.sign(y_limits[0]) * 0.05), y_limits[1] * (1 + np.sign(y_limits[1]) * 0.05))
        plt.legend()
        plt.show()

    return t_axis[ntsp:ntst], rho_step

# %% plot phase space
def plot_phase_space(t, X_data_obs, xhat_taxis, xhat_taxis_piu_sigma, xhat_taxis_meno_sigma, N_x, plot_equil, plot_bounds, true_coeff, ntsp, ntst):
    variable_names = ['real', 'estimate']
    plt.figure(figsize=[6, 3])
    plt.plot(X_data_obs[ntsp:ntst, 0], X_data_obs[ntsp:ntst, 1], 'k', label=variable_names[0])
    plt.plot(xhat_taxis[0, ntsp:ntst], xhat_taxis[1, ntsp:ntst], 'r--', label=variable_names[1])
    if plot_bounds:
        plt.plot(xhat_taxis_piu_sigma[0, ntsp:ntst], xhat_taxis_piu_sigma[1, ntsp:ntst], 'k', linestyle='--')
        plt.plot(xhat_taxis_meno_sigma[0, ntsp:ntst], xhat_taxis_meno_sigma[1, ntsp:ntst], 'k', linestyle='--')
    if plot_equil:
        # Note: Estimated equilibrium does not account for the use of two different sigma parameters
        real_equil = [true_coeff[0] / (true_coeff[4] + true_coeff[0] ** 2), true_coeff[0]]
        estimated_equil = [xhat_taxis[N_x, ntst - 1] / (xhat_taxis[N_x + 4, ntst - 1] + xhat_taxis[N_x, ntst - 1] ** 2), xhat_taxis[N_x, ntst - 1]]
        plt.plot(estimated_equil[0], estimated_equil[1], marker='+', markersize=10, color='k', markeredgewidth=3)
        plt.plot(real_equil[0], real_equil[1], marker='+', markersize=10, color='r--', markeredgewidth=3)
    plt.legend()
    plt.show()

    return