#%%
%matplotlib widget

import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy.io import savemat
import cloudpickle

from utils_DEKF import read_data,concatenate_input,add_noise_with_snr,concatenate_data,list_for_postprocessing

# attenzione: è possibile cambiare il metodo di interpolazione di scipy.integrate.solve_ivp per permettere di utilizzare un
# integratore implicito

def ass_xhat(inp,i0):
    xhat=np.zeros(N_ou+N_ov+N_theta)
    xhat[0:N_ou+N_ov]=inp[i0,0:N_ou+N_ov]
    for i1 in np.arange(N_theta):
        if inp[i0,-N_theta-N_f+i1] > 0:
            xhat[-N_theta+i1]=inp[i0,-N_theta-N_f+i1]*(1-theta_diff)
        else:
            xhat[-N_theta+i1]=inp[i0,-N_theta-N_f+i1]*(1+theta_diff)
    return xhat

def ass_inputs_Euler_forward(i0,F_taxis):
    inp_f = F_taxis[-N_f:,i0]
    inp_f_1 = F_taxis[-N_f:,i0-1]

    return inp_f, inp_f_1

def output_obs(inp,i0,N_obs_u,N_obs_v,N_obs_a,N_obs_u_inp,N_obs_v_inp,N_obs_a_inp):
    out = np.zeros((N_obs_u+N_obs_v+N_obs_a))
    out[:N_obs_u] = inp[i0,:N_obs_u]
    out[N_obs_u:N_obs_u+N_obs_v] = inp[i0,N_obs_u_inp:N_obs_u_inp+N_obs_v]
    out[N_obs_u+N_obs_v:N_obs_u+N_obs_v+N_obs_a] = inp[i0,N_obs_u_inp+N_obs_v_inp:N_obs_u_inp+N_obs_v_inp+N_obs_a]
    return out

def jacobian_A(A, variables):
    '''
    A: matrix of SINDy coefficients for the states and parameters (dynamic equation)
    variables: [x1, x2, x1_dot, x2_dot, w0] values at which the jacobian is evaluated
    '''

    n = A.shape[0] #number of equations
    n_features = A.shape[1] #number of features
    n_var = len(variables) 
    J_library = np.zeros((n_features, n_var))
    x1, x2, x1_dot, x2_dot, w0 = variables

    #dA/dx1
    J_library[:,0] = np.array([ 1., 0., 0., 0., 0., 
                        2.*x1, x2, x1_dot, x2_dot, w0,
                        0., 0., 0., 0.,
                        0., 0.,0.,
                        0., 0., 
                        0.,])
    #dA/dx2
    J_library[:,1] = np.array([ 0., 1., 0., 0., 0.,
                        0., x1, 0., 0., 0.,
                        2.*x2, x1_dot, x2_dot, w0,
                        0., 0., 0.,
                        0., 0.,
                        0.])
    #dA/dx1_dot
    J_library[:,2] = np.array([ 0., 0., 1., 0., 0.,
                        0., 0., x1, 0., 0.,
                        0., x2, 0., 0.,
                        2.*x1_dot, x2_dot, w0,
                        0., 0.,
                        0.])
    #dA/dx2_dot
    J_library[:,3] = np.array([ 0., 0., 0., 1., 0.,
                        0., 0., 0., x1, 0.,
                        0., 0., x2, 0.,
                        0., x1_dot, 0.,
                        2.*x2_dot, w0,
                        0.])
    #dA/w0
    J_library[:,4] = np.array([ 0., 0., 0., 0., 1.,
                        0., 0., 0., 0., x1,
                        0., 0., 0., x2,
                        0., 0., x1_dot,
                        0., x2_dot,
                        2.*w0])
    
    F = A @ J_library

    for i3 in range(N_theta):
        F = np.append(F, [np.zeros(n+1)], axis=0)
        F[n+i3,n_var-N_theta+i3] = 0

    return F

def model_Asindy(A, variables):
    '''
    A: matrix of SINDy coefficients for the states and parameters (transition equation)
    variables: [x1, x2, x1_dot, x2_dot, w0]
    '''

    #N_eqs = A.shape[0] #number of equations
    #n_features = A.shape[1] #number of variables
    x1, x2, x1_dot, x2_dot, w0 = variables

    sindy_library_A = np.array([x1, x2, x1_dot, x2_dot, w0,
                                x1*x1, x1*x2, x1*x1_dot, x1*x2_dot, x1*w0,
                                x2*x2, x2*x1_dot, x2*x2_dot, x2*w0,
                                x1_dot*x1_dot, x1_dot*x2_dot, x1_dot*w0,
                                x2_dot*x2_dot, x2_dot*w0,
                                w0*w0
                                ])
    return A @ sindy_library_A

def model_Bsindy_f(B, variables):
    '''
    B: matrix of SINDy coefficients for the states and parameters (observation equation)
    variables: [F1,F2]
    '''

    #N_eqs = A.shape[0] #number of equations
    #n_features = Hsindy_f_coeff.shape[1] #number of variables
    F1, F2 = variables

    sindy_library_B_f = np.array([F1,F2
                                ])

    return B @ sindy_library_B_f

def model_Hsindy(Hsindy_coeff, variables):
    '''
    Hsindy_coeff: matrix of SINDy coefficients for the states and parameters (observation equation)
    variables: [x1, x2, x1_dot, x2_dot, w0]
    '''

    #N_obs = Hsindy_coeff.shape[0] #number of observations
    #n_features = Hsindy_coeff.shape[1] #number of variables
    x1, x2, x1_dot, x2_dot, w0 = variables

    sindy_library_H = np.array([x1, x2, x1_dot, x2_dot, w0,
                                x1*x1, x1*x2, x1*x1_dot, x1*x2_dot, x1*w0,
                                x2*x2, x2*x1_dot, x2*x2_dot, x2*w0,
                                x1_dot*x1_dot, x1_dot*x2_dot, x1_dot*w0,
                                x2_dot*x2_dot, x2_dot*w0,
                                w0*w0
                                ])

    return Hsindy_coeff @ sindy_library_H

def model_Hsindy_f(Hsindy_f_coeff, variables):
    '''
    Hsindy_f_coeff: matrix of SINDy coefficients for the states and parameters (observation equation)
    variables: [F1,F2]
    '''

    #N_obs = Hsindy_f_coeff.shape[0] #number of observations
    #n_features = Hsindy_f_coeff.shape[1] #number of variables
    F1, F2 = variables

    sindy_library_H_f = np.array([F1,F2
                                ])

    return Hsindy_f_coeff @ sindy_library_H_f

def jacobian_H(Hsindy_coeff, variables):
    '''
    Hsindy_coeff: matrix of SINDy coefficients for the states and parameters (observation equation)
    variables: [x1, x2, x1_dot, x2_dot, w0] values at which the jacobian is evaluated
    '''

    #N_obs = Hsindy_coeff.shape[0] #number of observations
    n_features = Hsindy_coeff.shape[1] #number of features
    n_var = len(variables) 
    H_library = np.zeros((n_features, n_var))
    x1, x2, x1_dot, x2_dot, w0 = variables

    #dH/dx1
    H_library[:,0] = np.array([ 1., 0., 0., 0., 0.,
                                2.*x1, x2, x1_dot, x2_dot, w0,
                                0., 0., 0., 0.,
                                0., 0.,0.,
                                0., 0., 
                                0.])
    #dH/dx2
    H_library[:,1] = np.array([ 0., 1., 0., 0., 0.,
                                0., x1, 0., 0., 0.,
                                2.*x2, x1_dot, x2_dot, w0,
                                0., 0., 0.,
                                0., 0.,
                                0.])
    #dH/dx1_dot
    H_library[:,2] = np.array([ 0., 0., 1., 0., 0.,
                                0., 0., x1, 0., 0.,
                                0., x2, 0., 0.,
                                2.*x1_dot, x2_dot, w0,
                                0., 0.,
                                0.])
    #dH/dx2_dot
    H_library[:,3] = np.array([ 0., 0., 0., 1., 0.,
                                0., 0., 0., x1, 0.,
                                0., 0., x2, 0.,
                                0., x1_dot, 0.,
                                2.*x2_dot, w0,
                                0.])
    #dH/w0
    H_library[:,4] = np.array([ 0., 0., 0., 0., 1.,
                                0., 0., 0., 0., x1,
                                0., 0., 0., x2,
                                0., 0., x1_dot,
                                0., x2_dot,
                                2.*w0])

    return Hsindy_coeff @ H_library

def print_model(library_names, coeffs, threshold = 1e-2, feature_names = ["x1", "x2", "x1'", "x2'", "w0", "F1", "F2"], precision = 3):
    for i in range(coeffs.shape[0]):
        stringa = feature_names[i] + "' = "
        for j in range(coeffs.shape[1]):
            if np.abs(coeffs[i,j]) > threshold:
                stringa += str(round(coeffs[i,j], precision)) + ' ' + library_names[j] + " + "
        print(stringa[:-2])

#%%
root_ID = r"D:\Luca\Dati\DEKF"
problem_ID = r"telaio_2DOF"
model_ID = r"3_20DEKF"
data_ID = r"3_15DEKF"

data_root_ID = root_ID + '\\' + problem_ID + '\\' + data_ID
data_root_sindy_ID = root_ID + '\\' + problem_ID + '\\' + model_ID


#%%
t0  = 0.0
dt  = 0.01
ta  = 40.96
t_axis = np.arange(t0, ta, dt)

load_displ = 1
load_veloc = 1
load_accel = 1
load_load  = 1

load_no_noise_data = 0
add_noise = 1
snr_noise_displ = 15
snr_noise_veloc = 15
snr_noise_accel = 15


N_ou = 2
N_ov = 2
N_oa = 2

scale_for_u = 1
scale_for_a = 1

# si preferisce mantenere la scrittura più generale ####################################################################################
H_vibr_data_only = 1 # si potrebbe scrivere un altro modello sindy per trovare la relazione tra le variabili cinematiche del problema e
                     # altri dati misurati (es. valori di deformazione letti tramite strain gauges)
# Tuttavia, dato che Sindy identifica il sistema dinamico a partire dai dati,
# sarà semmpre verificata (o ci si potrà ridurre al)la condizione N_o = N_obs.
# Se si volesse cambiare la condizione, sarà necessario farlo anche sul file di Sindy
N_obs_u = 2
N_obs_v = 2
N_obs_a = 2
#N_obs_a = 0
N_obs = N_obs_u + N_obs_v + N_obs_a
# displacement observations
Sd = np.array([[1,0],[0,1]])   # Sd [N_obs_u x N_ou]
# velocity observations
Sv = np.array([[1,0],[0,1]])   # Sv [N_obs_v x N_ov]
# acceleration observations
Sa = np.array([[1,0],[0,1]])   # Sa [N_obs_a x N_oa]
#Sa = np.array([[0,0],[0,0]])   # Sa [N_obs_a x N_oa]
# presenti nel file di input
N_obs_u_inp = 2
N_obs_v_inp = 2
N_obs_a_inp = 2
# #####################################################################################################################################

#%%
N_l = 4096 #lunghezza serie
N_i = 20   #numero istanze
N_o = N_ou*load_displ+N_ov*load_veloc+N_oa*load_accel  #numero osservazioni
N_f = 2   #numero termini forza
N_theta = 1  #numero parametri

test_performed = 17  #number of the test to be performed 

theta_diff = -0.2 #sottostima (se >0) relativa del parametro iniziale

# load data ###########################################################################################################################
che = 0; data = 0
if load_displ: #displacement
    U_data, _ = read_data(data_root_ID,'_U_concat_',N_l,N_ou)
    if load_no_noise_data or add_noise:
        data_no_noise = copy.copy(U_data)
        U_data = add_noise_with_snr(U_data,snr_noise_displ)
    data = U_data; che = 1        

if load_veloc: #velocity
    V_data, _ = read_data(data_root_ID,'_V_concat_',N_l,N_ov)
    if load_no_noise_data or add_noise:
        V_data_no_noise = copy.copy(V_data)
        V_data = add_noise_with_snr(V_data,snr_noise_veloc)
    che,data = concatenate_input(data,V_data,che)
    che,data_no_noise = concatenate_input(data_no_noise,V_data_no_noise,che)
    
if load_accel: #acceleration
    A_data, _ = read_data(data_root_ID,'_A_concat_',N_l,N_oa)
    if load_no_noise_data or add_noise:
        A_data_no_noise = copy.copy(A_data)
        A_data = add_noise_with_snr(A_data,snr_noise_accel)
    che,data = concatenate_input(data,A_data,che)
    che,data_no_noise = concatenate_input(data_no_noise,A_data_no_noise,che)

theta, _ = read_data(data_root_ID,'theta_',N_l,N_theta)
if load_load:  #force
    F_data, _ = read_data(data_root_ID,'F_',N_l,N_f)
data = concatenate_data(data,F_data,theta,N_l,N_i,N_o,N_theta,N_f)
# end load data #######################################################################################################################

#%% valori di scala di ingresso
scale_U = np.max(np.abs(U_data))
scale_V = np.max(np.abs(V_data))
scale_A = np.max(np.abs(A_data))
scale_F = np.max(np.abs(F_data))
scale_theta = np.max(np.abs(theta))
x_   = U_data / scale_V
ddx_ = A_data / scale_V

scale_x   = np.max(np.abs(x_))
scale_ddx = np.max(np.abs(ddx_))

#%% Routine per caricare i dati senza rumore
# # load no noise data ####################################################################################################################
# if load_no_noise_data:
#     che = 0; data_no_noise = 0
#     if load_displ: #displacement
#         U_data_no_noise, _ = read_data_no_noise(data_root_ID,'_U_concat_',N_l,N_ou)
#         data_no_noise = U_data_no_noise; che = 1
#     if load_veloc: #velocity
#         V_data_no_noise, _ = read_data_no_noise(data_root_ID,'_V_concat_',N_l,N_ov)
#         che,data_no_noise = concatenate_input(data_no_noise,V_data_no_noise,che)  
#     if load_accel: #acceleration
#         A_data_no_noise, _ = read_data_no_noise(data_root_ID,'_A_concat_',N_l,N_oa)
#         che,data_no_noise = concatenate_input(data_no_noise,A_data_no_noise,che)    
# # end load no noise data ################################################################################################################

if add_noise:
    inp_no_noise = data_no_noise[test_performed,:,:]

    inp_no_noise[:,:N_o] = inp_no_noise[:,:N_o] / scale_V
    inp_no_noise[:,N_o:N_o+N_theta] = inp_no_noise[:,N_o:N_o+N_theta] / scale_theta
    inp_no_noise[:,-N_f:] = inp_no_noise[:,-N_f:] / scale_F

    if scale_for_u:
        inp_no_noise[:,:N_obs_u] = inp_no_noise[:,:N_obs_u] / scale_x 

#%% create a list for postprocessing ####################################################################################################
obs_labels = [None] * (N_o+N_theta)
i2 = 0
i2,obs_labels=list_for_postprocessing(load_displ,obs_labels,N_ou,'displ',i2)
i2,obs_labels=list_for_postprocessing(load_veloc,obs_labels,N_ov,'veloc',i2)
i2,obs_labels=list_for_postprocessing(load_accel,obs_labels,N_oa,'accel',i2)
i2,obs_labels=list_for_postprocessing(1,obs_labels,N_theta,'theta',i2)
# end list for postprocessing #########################################################################################################

#%% SINDy

#%% import precomputed SINDy matrices for dynamics
sindy_loadID = data_root_sindy_ID + '\\sindy_model_dynamics.pkl'
with open(sindy_loadID, 'rb') as f:
    sindy_model = cloudpickle.load(f)

coeffs = sindy_model.coefficients()

sindy_library_names = [
    "x1", "x2", "x1'", "x2'", "w0", "F1", "F2",
    "x1^2", "x1*x2", "x1*x1'", "x1*x2'", "x1*w0",
    "x2^2", "x2*x1'", "x2*x2'", "x2*w0",
    "x1'^2", "x1'*x2'", "x1'*w0",
    "x2'^2", "x2'*w0",
    "w0^2"
]

sindy_model.print(precision = 3)
#%%
print_model(sindy_library_names, coeffs, threshold = 1e-4, feature_names = ["x1", "x2", "x1'", "x2'", "w0", "F1", "F2"])


# %%Compute the A and B matrices (related to the evolution dynamics for x and x_dot, and for the theta and F)
#B is the matrix corresponding to the columns relative to F1 and F2
B = coeffs[:,5:7]
#A is the matrix corresponding to the columns relative to the states and parameters
A = np.concatenate((coeffs[:,0:5], coeffs[:,7:]), axis = 1)

#%% import precomputed matrices for observation
if H_vibr_data_only:
    Hsindy_saveID = data_root_sindy_ID + '\\sindy_Hmodel_coeffs.npy'
    Hsindy_f_saveID = data_root_sindy_ID + '\\sindy_Hmodel_f_coeffs.npy'
    Hsindy_coeff = np.load(Hsindy_saveID)
    Hsindy_f_coeff = np.load(Hsindy_f_saveID)

#else # si dovrebbe caricare un altro modello SINDy

#%% set the test performed
inp = data[test_performed,:,:]

inp[:,:N_o] = inp[:,:N_o] / scale_V
inp[:,N_o:N_o+N_theta] = inp[:,N_o:N_o+N_theta] / scale_theta
inp[:,-N_f:] = inp[:,-N_f:] / scale_F

#%% scale the displacement
if scale_for_u:
    inp[:,:N_obs_u] = inp[:,:N_obs_u] / scale_x

#%% scale the acceleration
# if scale_for_a:
#     inp[:,N_obs_u+N_obs_v:N_obs_u+N_obs_v+N_obs_a] = inp[:,N_obs_u+N_obs_v:N_obs_u+N_obs_v+N_obs_a] / scale_ddx

#%% parametri per la calilbrazione del filtro °°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°
# displacement (riscalato), velocity, acceleration test 1 3_20 + 3_15 (add_noise) funziona benissimo
p0_ou = 1e-8
p0_ov = 1e-8
p0_theta = 1e-2
q_ou = 5e-3
q_ov = 5e-3
q_theta = 1e-8
r_ou = 1e-3
r_ov = 1e-3
r_oa = 1e+3
# °°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°

#%% perform the hybrid EKF

xhat_taxis = np.zeros(shape=(N_ou+N_ov+N_theta,N_l))
a_axis  = np.zeros(shape=(N_oa,N_l))
P_taxis = np.zeros(shape=(N_ou+N_ov+N_theta,N_ou+N_ov+N_theta,N_l))
F_taxis = np.array(inp[:,-N_f:]).transpose()

xhat_taxis_piu_sigma  = np.zeros(shape=(N_ou+N_ov+N_theta,N_l))
xhat_taxis_meno_sigma = np.zeros(shape=(N_ou+N_ov+N_theta,N_l))

P = np.identity(N_ou+N_ov+N_theta)
P[:N_ou,:N_ou] = P[:N_ou,:N_ou]*p0_ou
P[N_ou:N_ou+N_ov,N_ou:N_ou+N_ov] = P[N_ou:N_ou+N_ov,N_ou:N_ou+N_ov]*p0_ov
P[-N_theta:,-N_theta:] = P[-N_theta:,-N_theta:]*p0_theta    

Q = np.identity(N_ou+N_ov+N_theta)
Q[:N_ou,:N_ou] = Q[:N_ou,:N_ou]*q_ou
Q[N_ou:N_ou+N_ov,N_ou:N_ou+N_ov] = Q[N_ou:N_ou+N_ov,N_ou:N_ou+N_ov]*q_ov
Q[-N_theta:,-N_theta:] = Q[-N_theta:,-N_theta:]*q_theta

R = np.identity(N_obs)
for i1 in range(N_obs_u):
    R[i1,i1] = r_ou
for i1 in range(N_obs_v):
    R[N_obs_u+i1,N_obs_u+i1] = r_ov
for i1 in range(N_obs_a):
    R[N_obs_u+N_obs_v+i1,N_obs_u+N_obs_v+i1] = r_oa

I = np.zeros(shape=(N_ou+N_ov+N_theta,N_ou+N_ov+N_theta))
for i1 in range(N_ou+N_ov+N_theta):
    I[i1,i1] = np.ones(1)

xhat = ass_xhat(inp,0)

xhat_taxis[:,0] = xhat
P_taxis[:,:,0]  = P

#%%
t0  = 0.0
dt  = 0.01
ta  = 40.96
t_axis = np.arange(t0, ta, dt)

n_time_steps = 4096

t_plus_1 = t0

#%% Run filter
for i0 in range(1,N_l):
    t = t_plus_1
    t_plus_1 = t + dt

    out = output_obs(inp,i0,N_obs_u,N_obs_v,N_obs_a,N_obs_u_inp,N_obs_v_inp,N_obs_a_inp)
    theta = xhat[-N_theta:]

    inp_f, inp_f_1 = ass_inputs_Euler_forward(i0,F_taxis)

    # Prediction phase
    xhat_pred = xhat[0:-N_theta] + dt*(model_Asindy(A,variables=xhat)+model_Bsindy_f(B,variables=inp_f_1))
    xhat_pred = np.append(xhat_pred,theta)
    F = jacobian_A(A,variables=xhat)

    P_pred = P + dt*(np.matmul(F,P) + np.matmul(P,F.transpose()) + Q)
    # end Predictor phase

    # Corrector phase
    H = jacobian_H(Hsindy_coeff,variables=xhat_pred)
    G = np.matmul(np.matmul(P_pred,H.transpose()), np.linalg.inv(np.matmul(np.matmul(H,P_pred),H.transpose())+R) ) # Kalman gain computation

    obs  = model_Hsindy(Hsindy_coeff,variables=xhat_pred) + model_Hsindy_f(Hsindy_f_coeff,variables=inp_f)
    
    out[-N_obs_a:] = out[-N_obs_a:] / scale_ddx
    obs[-N_obs_a:] = obs[-N_obs_a:] / scale_ddx
    xhat = xhat_pred + np.matmul(G, (out-obs) )

    P = np.matmul( np.matmul(I - np.matmul(G, H),  P_pred), (I - np.matmul(G, H)).transpose() ) + np.matmul(np.matmul(G,R),G.transpose())
    # end Corrector phase

    xhat_taxis[:,i0] = xhat
    P_taxis[:,:,i0]  = P
    a_axis[:,i0]     = obs[-N_oa:]*scale_ddx

#%% Plot the results ##########################################################################
for i5 in range(N_ou+N_ov+N_theta):
    # assumption - diagonal covariance matrix  
    xhat_taxis_piu_sigma[i5,:]  = xhat_taxis[i5,:] + 1.96 * np.sqrt(P_taxis[i5,i5,:])
    xhat_taxis_meno_sigma[i5,:] = xhat_taxis[i5,:] - 1.96 * np.sqrt(P_taxis[i5,i5,:])

#%%
n_time_steps_start_plot = 0
n_time_steps_stop_plot = 4095

#%% theta
plt.figure(figsize = [8,4])
plt.plot(t_axis[n_time_steps_start_plot:n_time_steps_stop_plot],xhat_taxis[4,n_time_steps_start_plot:n_time_steps_stop_plot],'k')
plt.plot(t_axis[n_time_steps_start_plot:n_time_steps_stop_plot],xhat_taxis_piu_sigma[4,n_time_steps_start_plot:n_time_steps_stop_plot],'k',linestyle='--')
plt.plot(t_axis[n_time_steps_start_plot:n_time_steps_stop_plot],xhat_taxis_meno_sigma[4,n_time_steps_start_plot:n_time_steps_stop_plot],'k',linestyle='--')
plt.plot(t_axis[n_time_steps_start_plot:n_time_steps_stop_plot],inp[n_time_steps_start_plot:n_time_steps_stop_plot,6],'orange')
plt.show()

#%% displacement
plt.figure(figsize = [8,4])
plt.plot(t_axis[n_time_steps_start_plot:n_time_steps_stop_plot],xhat_taxis[0,n_time_steps_start_plot:n_time_steps_stop_plot],'k')
plt.plot(t_axis[n_time_steps_start_plot:n_time_steps_stop_plot],xhat_taxis_piu_sigma[0,n_time_steps_start_plot:n_time_steps_stop_plot],'k',linestyle='--')
plt.plot(t_axis[n_time_steps_start_plot:n_time_steps_stop_plot],xhat_taxis_meno_sigma[0,n_time_steps_start_plot:n_time_steps_stop_plot],'k',linestyle='--')
plt.plot(t_axis[n_time_steps_start_plot:n_time_steps_stop_plot],inp[n_time_steps_start_plot:n_time_steps_stop_plot,0],'orange')
if load_no_noise_data or add_noise:
    plt.plot(t_axis[n_time_steps_start_plot:n_time_steps_stop_plot],inp_no_noise[n_time_steps_start_plot:n_time_steps_stop_plot,0],'gray')
plt.show()

#%% velocity
plt.figure(figsize = [8,4])
plt.plot(t_axis[n_time_steps_start_plot:n_time_steps_stop_plot],xhat_taxis[2,n_time_steps_start_plot:n_time_steps_stop_plot],'k')
plt.plot(t_axis[n_time_steps_start_plot:n_time_steps_stop_plot],xhat_taxis_piu_sigma[2,n_time_steps_start_plot:n_time_steps_stop_plot],'k',linestyle='--')
plt.plot(t_axis[n_time_steps_start_plot:n_time_steps_stop_plot],xhat_taxis_meno_sigma[2,n_time_steps_start_plot:n_time_steps_stop_plot],'k',linestyle='--')
plt.plot(t_axis[n_time_steps_start_plot:n_time_steps_stop_plot],inp[n_time_steps_start_plot:n_time_steps_stop_plot,2],'orange')
if load_no_noise_data or add_noise:
    plt.plot(t_axis[n_time_steps_start_plot:n_time_steps_stop_plot],inp_no_noise[n_time_steps_start_plot:n_time_steps_stop_plot,2],'gray')
plt.show()

#%% acceleration
    plt.figure(figsize = [8,4])
plt.plot(t_axis[n_time_steps_start_plot:n_time_steps_stop_plot],a_axis[0,n_time_steps_start_plot:n_time_steps_stop_plot],'k')
plt.plot(t_axis[n_time_steps_start_plot:n_time_steps_stop_plot],inp[n_time_steps_start_plot:n_time_steps_stop_plot,4],'orange')
if load_no_noise_data or add_noise:
    plt.plot(t_axis[n_time_steps_start_plot:n_time_steps_stop_plot],inp_no_noise[n_time_steps_start_plot:n_time_steps_stop_plot,4],'gray')
plt.show()

#%% Rescale the observation for output representation
if scale_for_u:
    inp[:,:N_obs_u] = inp[:,:N_obs_u] * scale_x
    xhat_taxis[:N_obs_u,:] = xhat_taxis[:N_obs_u,:] * scale_x
    xhat_taxis_piu_sigma[:N_obs_u,:] = xhat_taxis_piu_sigma[:N_obs_u,:] * scale_x
    xhat_taxis_meno_sigma[:N_obs_u,:] = xhat_taxis_meno_sigma[:N_obs_u,:] * scale_x

inp[:,:N_o] = inp[:,:N_o] * scale_V
inp[:,N_o:N_o+N_theta] = inp[:,N_o:N_o+N_theta] * scale_theta
inp[:,-N_f:] = inp[:,-N_f:] * scale_F

xhat_taxis[:N_ou+N_ov,:] = xhat_taxis[:N_ou+N_ov,:] * scale_V
xhat_taxis[-N_theta:,:] = xhat_taxis[-N_theta:,:] * scale_theta
xhat_taxis_piu_sigma[-N_theta:,:] = xhat_taxis_piu_sigma[-N_theta:,:] * scale_theta
xhat_taxis_meno_sigma[-N_theta:,:] = xhat_taxis_meno_sigma[-N_theta:,:] * scale_theta
xhat_taxis_piu_sigma[:N_ou+N_ov,:] = xhat_taxis_piu_sigma[:N_ou+N_ov,:] * scale_V
xhat_taxis_meno_sigma[:N_ou+N_ov,:] = xhat_taxis_meno_sigma[:N_ou+N_ov,:] * scale_V
a_axis = a_axis *scale_V

if load_no_noise_data or add_noise:
    inp_no_noise[:,:N_o] = inp_no_noise[:,:N_o] * scale_V
    inp_no_noise[:,N_o:N_o+N_theta] = inp_no_noise[:,N_o:N_o+N_theta] * scale_theta
    inp_no_noise[:,-N_f:] = inp_no_noise[:,-N_f:] * scale_F
    if scale_for_u:
        inp_no_noise[:,:N_obs_u] = inp_no_noise[:,:N_obs_u] * scale_x

#%% Save the data for importing them in Matlab (MAT extension)
save_mat_data_ID =  data_root_ID + '\\' + model_ID + '_output_for_matlab.mat'
if load_no_noise_data or add_noise:
    savemat(save_mat_data_ID, {'t_axis': t_axis, 'xhat_taxis': xhat_taxis, 'xhat_taxis_piu_sigma': xhat_taxis_piu_sigma, 'xhat_taxis_meno_sigma': xhat_taxis_meno_sigma,'a_axis':a_axis,'inp':inp,'inp_no_noise':inp_no_noise})
else:
    savemat(save_mat_data_ID, {'t_axis': t_axis, 'xhat_taxis': xhat_taxis, 'xhat_taxis_piu_sigma': xhat_taxis_piu_sigma, 'xhat_taxis_meno_sigma': xhat_taxis_meno_sigma,'a_axis':a_axis,'inp':inp})

# %%
