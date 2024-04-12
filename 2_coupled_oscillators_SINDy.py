#%%
######################       LIBRARIES       ######################
#%matplotlib widget

import pysindy as ps
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from scipy.integrate import odeint
from scipy.linalg import hankel
from pyDOE import lhs
from scipy.io import savemat
from sklearn.utils.extmath import randomized_svd
from sklearn.model_selection import train_test_split

from utils_DEKF import add_noise_with_snr, jacobian_A, jacobian_H

seed = 1

#%% data root #TODO: change root
root_ID = r"D:\Luca\Dati\DEKF"
problem_ID = r"2DOF_nonLinear"
data_ID = r"2_1SENDy"
data_root_ID = root_ID + '\\' + problem_ID + '\\' + data_ID

#%%
######################       PROBLEM SETUP       ######################
'''
We consider two coupled oscillators with the following equations of motion:
u1'' + ω1^2 u1 + 2ξω1 u1' + α u2 = 0,
u2'' + ω2^2 u2 + 2ξω2 u2' + α u1 + β u1^2 + γ u2^3 = 0,

which can be rewritten at first order as:
u1' = v1,
v1' = - ω1^2 u1 - 2ξω1 v1 - α u2,
u2' = v2,
v2' = - ω2^2 u2 - 2ξω2 v2 - α u1 - β u1^2 - γ u2^3.

We suppose we are only able to measure the position of the first oscillator u1, 
and we want to
(Alg. 1) identify the equations of motion from the data (SINDy)
(Alg. 2) estimate the states and the linear and quadratic, α and β, coupling parameters
'''

system = {
    #Natural frequency of the oscillators:
    'w1' : 1.,
    'w2' : 1.95,

    #Damping:
    'csi1' : 1e-2,
    'csi2' : 5e-3,

    #Coupling:
    #To be estimated

    #Cubic nonlinearity:
    'gamma' : 1e-3,

    #Time span:
    't0' : 0.,
    'T' : 200.,

    #Time step:
    'dt' : 1e-2
}

#Define the right-hand side of the ODE:
def f(y,t, system = system, forced = True):

    u1, v1, u2, v2 = y

    #Retrieve parameters:  
    w1 = system['w1']
    w2 = system['w2']
    csi1 = system['csi1']
    csi2 = system['csi2']
    gamma = system['gamma']
    beta = system['beta']
    alpha = system['alpha']

    du1 = v1
    dv1 = - (w1**2)*u1 -2.*csi1*w1*v1  - alpha*u2 
    du2 = v2
    dv2 = - (w2**2)*u2  -2.*csi2*w2*v2 - beta*(u1**2) - alpha*u1 - gamma*(u2**3) 

    return np.array([du1, dv1, du2, dv2])

#Define the time frame:
t0, T, dt = system['t0'], system['T'], system['dt']

t = np.arange(t0,T,dt)
Nt = len(t)

        
# %%
######################       GENERATE DATA       ######################
#Number of instances
n_ics = 20

#Generate random intial conditions for the displacement, while we consider zero initial velocities:
np.random.seed(seed=seed)  
u1_0 = np.random.normal(-2,1,n_ics)
u2_0 = np.random.normal(3,1,n_ics)

v1_0 = np.repeat(0.,n_ics)
v2_0 = np.repeat(0.,n_ics)

#Generate equispaced parameters for α and β, in the range P_α = [-0.05, -0.5] and P_β =  [0.001, 0.01]
min_alpha, max_alpha = 0.005, 0.5
min_beta, max_beta = 0.005, 0.5
samples = lhs(2, samples=n_ics)
#Log space the data to have more samples in the lower range
alpha = np.log(min_alpha) + (np.log(max_alpha) - np.log(min_alpha))*samples[:,0]
beta = np.log(min_beta) + (np.log(max_beta) - np.log(min_beta))*samples[:,1]
alpha = -np.exp(alpha)
beta = np.exp(beta)

param_coupling = np.concatenate((alpha.reshape(-1,1), beta.reshape(-1,1)), axis = 1)


X = []

for num_sim in range(n_ics):
    #Set initial conditions:
    y0 = np.array([u1_0[num_sim], v1_0[num_sim], u2_0[num_sim], v2_0[num_sim]])
    #Retrieve coupling parameters:
    system['alpha'], system['beta'] = param_coupling[num_sim]
    #Solve the ODE:
    sol = odeint(f, y0, t) 
    #Store the solution:
    X.append(sol)

    u1, v1, u2, v2 = np.hsplit(sol, 4)

    #Plot the solution:
    plt.figure(figsize=(7,3))
    plt.plot(t,u1, label = '$u_1$')
    plt.plot(t,u2, label = '$u_2$')
    plt.legend(fontsize = 12)
    plt.title('Simulation #' + str(num_sim+1), fontsize = 12)
    plt.xlabel('t', fontsize = 12)

    plt.show()

X = np.array(X)

#Split the data into training and test sets:
X_train, X_test, param_train, param_test = train_test_split(X, param_coupling, test_size=0.2, random_state=seed)
n_train, n_test = len(X_train), len(X_test)

# %%
######################       TIME-DELAY EMBEDDING       ######################

def create_hankel(vector, length, shift = 1):
    p = len(vector)-(length*shift)+1
    H = vector[0:p]
    for i in range(1,length):
        H = np.vstack([H, vector[(i*shift):(i*shift)+p]])
    return H

#Create Hankel matrix from observations of u1:
length, shift = 200, 1

for i in range(n_train):
    u1 = X_train[i,:,0]
    H1 = create_hankel(u1.flatten(), length, shift)
    if i == 0:
        H = H1
    else:
        H = np.hstack([H,H1])

#Compute SVD:
uh, sh, vh = randomized_svd(H, n_components=16)

#Plot singular values:
plt.figure()
plt.plot(sh/np.sum(sh),'r*-')
plt.ylabel("$s_\lambda / \sum_{i}s_i$", fontsize = 14)
plt.xlabel("$\lambda$", fontsize = 14)

# %%
######################      TIME-DELAY COORDINATES       ######################
#Select the number of time-delay coordinates:
n_tdc = 4

U = uh[:,:n_tdc]
S = sh[:n_tdc]
Nt_h = Nt - length*shift + 1 #time length of the new coordinates

#Compute time-delay coordinates:
u_h = H.T @  (U * 1/S)

#Plot the first time-delay coordinate:
plt.figure(figsize = [13,16])
for i in range(4):
    plt.subplot(421 + 2*i)
    plt.plot(X_train[0,:Nt_h,i], 'b')
    plt.title(f'Original variable {i+1}', fontsize = 12)

    plt.subplot(421 + 2*i+1)
    plt.plot(u_h[i*Nt_h:(i+1)*Nt_h,i], 'r')
    plt.title(f'Time-delay coordinate {i+1}', fontsize = 12)

plt.figure(figsize = [13,8])
for i in range(2):
    plt.subplot(221 + i)
    plt.plot(X_train[0,:Nt_h,i], X_train[0,:Nt_h,i+1], 'b')
    plt.title(f'Original variables {2*i+1} and {2*i+2}', fontsize = 12)
    
    plt.subplot(221 +i + 2)
    plt.plot(u_h[i*Nt_h:(i+1)*Nt_h,i], u_h[i*Nt_h:(i+1)*Nt_h,i+1], 'r')
    plt.title(f'Time-delay coordinates {2*i+1} and {2*i+2}', fontsize = 12)


# %%
######################      PREPARE DATA for SINDy       ######################

X_tdc = []
dX_tdc = []
param_tdc = []

X_test_tdc = []
dX_test_tdc = []
param_test_tdc = []

for i in range(n_train):
    u1_ = X_train[i,:,0]
    H_ = create_hankel(u1_, length, shift)
    u_h = H_.T @  (U * 1/S)

    X_tdc.append(u_h[:,:n_tdc])
    dX_tdc.append(np.gradient(u_h[:,:n_tdc], dt, axis = 0))
    param_tdc.append(np.tile(param_train[i], (Nt,1))[:Nt_h])
    feature_names = ["u" + str(i+1) for i in range(n_tdc)]
    feature_names +=  ["a", "b"]

for i in range(n_test):
    u1_ = X_test[i,:,0]
    H_ = create_hankel(u1_, length, shift)
    u_h = H_.T @  (U * 1/S)

    X_test_tdc.append(u_h[:,:n_tdc])
    dX_test_tdc.append(np.gradient(u_h[:,:n_tdc], dt, axis = 0))
    param_test_tdc.append(np.tile(param_test[i], (Nt,1))[:Nt_h])


# %%
######################      CREATE SINDy model       ######################
model = ps.SINDy(feature_names  = feature_names, feature_library= ps.PolynomialLibrary(degree = 3, include_bias=False), optimizer=ps.STLSQ(threshold=1e-3))
model.fit(X_tdc, t=dt, multiple_trajectories=True, u = param_tdc, x_dot = dX_tdc) 

model.print()
A = model.coefficients()
n_features = A.shape[1] #number of features

# %%
# to reproduce the results
#model = handModel

# %%
######################      PREDICT       ######################
idx_test = 0
#idx_test_fake = 0
#fake_param = param_test_tdc[idx_test_fake][0]
idx_test_fake_a, idx_test_fake_b = 0, 0
fake_param = np.array([param_test_tdc[idx_test_fake_a][0][0], param_test_tdc[idx_test_fake_b][0][1]])#param_test_tdc[idx_test_fake][0]
print('Real parameter:', param_test_tdc[idx_test][0], 'Fake parameter:', fake_param)

#Predict the evolution of the time-delay coordinates:
fake_params = np.tile(fake_param, (Nt_h,1))
pred = model.simulate(X_test_tdc[idx_test][0,:], t= t[:Nt_h], u = fake_params)

#Plot the evolution of the time-delay coordinates:
plt.figure(figsize = [8,16])
for i in range(4):
    plt.subplot(411 +i)
    plt.plot(X_test_tdc[idx_test][:,i], 'b', label = 'true')
    plt.plot(pred[:,i], 'r--', label = 'pred')
    plt.title(f'Time-delay coordinate {i+1}', fontsize = 12)
    plt.legend(fontsize = 12)

#%%
######################      RECONSTRUCTION       ######################
#Reconstruct observed variable:
x_rec = pred @ (U * S).T

#Plot the reconstructed variable:
plt.figure(figsize = [8,4])
plt.plot(X_test[idx_test,:,0], 'b', label = 'true')
plt.plot(x_rec[:,0], 'r--', label = 'pred')
plt.title(f'Reconstructed variable', fontsize = 12)
plt.legend(fontsize = 12)

print('MSE', mean_squared_error(X_test[idx_test,:Nt_h-1,0], x_rec[:Nt_h,0]))


######################      OUTPUT INFO       ######################

'''
OUTPUTS:
    model:          SINDy model
    X_tdc:          time delayed coordinates. It is a list (of length n_train=12, that is the number of training time-series),
                    where each element is an array of shape (Nt_tdc, n_coordinates) = (190001, 4), that is the time-history of length Nt_tdc,
                    of the n_coordinates time-delayed coordinates.
    dX_tdc:         derivatives of the time delayed coordinates.
    param_tdc:      list of parameters, i.e. natural frequencies of the hidden oscillator. Same shape of the data 
                    (but they are constant time histories as parameter is not varying in time).

    X_tdc_test:     Analogous structure but on test data.
    param_tdc_test: " "
    U, S:           Matrix of left singular values (of the SVD of the Hankel matrix) and singular value matrix, respectively.
                    They are used to go from the original cooridnates to the time-delayed coordinates and viceversa (via projection).
            

'''


#%%

# Kalman filter part - select the test case
param_a_rel_error = -0.9# -0.9 
param_b_rel_error = 19

idx_test = 0   #0 con 20 #10 con 100
correct_param = param_test_tdc[idx_test][0]

state_param = np.zeros(2)
state_param[0] = correct_param[0] *(1+param_a_rel_error)
state_param[1] = correct_param[1] *(1+param_b_rel_error)

N_x     = n_tdc # numero variabili di stato uguale al numero di time-delay coordinates
N_param = 2     # numero parametri da identificare
N_obs   = 1     # numero quantità osservate

N_l = Nt_h # lunghezza dell'asse temporale

add_noise = 1
snr_noise_displ = 15
snr_noise_veloc = 15

if add_noise:
    out_no_noise_axis = X_test[idx_test,:,0]
    out_axis = add_noise_with_snr(out_no_noise_axis,snr_noise_displ)
else:
    out_no_noise_axis = X_test[idx_test,:,0]
    out_axis = X_test[idx_test,:,0]


#%% TEST and PLOT the starting state_param fake parameter
print('Real parameter:', correct_param, 'Fake parameter:', state_param)

#Predict the evolution of the time-delay coordinates:
param_axis_fake = np.zeros(shape=(N_l,2))
param_axis_fake[:,0]= np.ones(N_l)*state_param[0]
param_axis_fake[:,1]= np.ones(N_l)*state_param[1]
pred = model.simulate(X_test_tdc[idx_test][0,:], t= t[:Nt_h], u = param_axis_fake)

#Plot the evolution of the time-delay coordinates:
plt.figure(figsize = [8,16])
for i in range(n_tdc):
    plt.subplot(411 +i)
    plt.plot(X_test_tdc[idx_test][:,i], 'b', label = 'true')
    plt.plot(pred[:,i], 'r--', label = 'pred')
    plt.title(f'Time-delay coordinate {i+1}', fontsize = 12)
    plt.legend(fontsize = 12)
plt.show()

#Reconstruct observed variable:
x_rec = pred @ (U * S).T

#Plot the reconstructed variable:
plt.figure(figsize = [8,4])
plt.plot(X_test[idx_test,:,0], 'black', label = 'true')
plt.plot(x_rec[:,0], 'r--', label = 'pred')
plt.title(f'Reconstructed variable', fontsize = 12)
plt.legend(fontsize = 12)

print('MSE', mean_squared_error(X_test[idx_test,:Nt_h-1,0], x_rec[:Nt_h,0]))
#%%
# Kalman filter initialisation

# n_ics = 20   u1^2  #0 per 20
# initialisation values of the uncertainty associated to the state variables
p0_x     = 1e-6
p0_param_1 = 1e-1
p0_param_2 = 1e-1
# process noise
q_x_1      = 5e-10 # prime due variabili embedded
q_x_2      = 5e-8#2e-8 #TODO
q_x_3      = 5e-8#3e-8  #TODO
q_param  = 1e-11
# measurement noise
r_x_obs  = 3e-1

# # n_ics = 20   u1^2  #3 per 20
# # parameters for Kalman filter tuning 
# N_x_inc = []
# # initialisation values of the uncertainty associated to the state variables
# p0_x     = 1e-6
# p0_param_1 = 1e-1
# p0_param_2 = 1e-2
# # process noise
# q_x_1      = 1e-8
# q_x_2      = 4e-9
# q_x_3      = 4e-9
# q_param  = 1e-11
# # measurement noise
# r_x_obs  = 3e-1

t_axis = np.arange(t0, (dt*N_l), dt)

xhat_taxis = np.zeros(shape=(N_x+N_param,N_l))
xhat_corr_taxis = np.zeros(shape=(N_x+N_param,N_l))
obs_taxis  = np.zeros(shape=(N_obs,N_l))
obs_check_taxis = np.zeros(shape=(N_obs,N_l))
P_taxis    = np.zeros(shape=(N_x+N_param,N_x+N_param,N_l))

xhat_taxis_piu_sigma  = np.zeros(shape=(N_x+N_param,N_l))
xhat_taxis_meno_sigma = np.zeros(shape=(N_x+N_param,N_l))

# inizialisation of the covariance matrix
I_cov = np.identity(N_x+N_param)
P = np.identity(N_x+N_param)
P[:N_x,:N_x] = P[:N_x,:N_x]*p0_x
P[-2,-2] = P[-2,-2]*p0_param_1
P[-1,-1] = P[-1,-1]*p0_param_2

# inizialisation of the process noise matrix
Q = np.identity(N_x+N_param)
Q[:2,:2] = Q[:2,:2]*q_x_1
Q[2,2] = Q[2,2]*q_x_2
Q[3,3] = Q[3,3]*q_x_3
Q[-N_param:,-N_param:] = Q[-N_param:,-N_param:]*q_param

# inizialisation of the measurement noise matrix
R = np.identity(N_obs)*r_x_obs

# identity matrix definition (dimension of the state variables)
I = np.zeros(shape=(N_x+N_param,N_x+N_param))
for i1 in range(N_x+N_param):
    I[i1,i1] = np.ones(1)

# inizialisation of the state vector
xhat = np.zeros(shape=(N_x+N_param))
xhat[0:N_x] = X_test_tdc[idx_test][0,:]
xhat[-N_param:] = state_param

xhat_taxis[:,0] = xhat
obs_taxis[0,0]  = out_axis[0]
obs_check_taxis[0,0]  = out_axis[0]
P_taxis[:,:,0]  = P

e_ = np.zeros(length)
e_[0] = 1

H = jacobian_H(U,S,N_x)

#%%
# Kalman filter run
t_plus_1 = t0

for i0 in range(1,N_l):
    t_ = t_plus_1
    t_plus_1 = t_ + dt

    out = out_axis[i0]
    param = xhat[-N_param:]

    if i0 == 1500:
        aaaa = 1

    # Prediction phase
    #xhat_pred = xhat[0:-N_param] + dt*(model_Asindy(A,variables=xhat))
    xhat_pred = xhat[0:-N_param] + dt*(model.predict(xhat[0:-N_param].reshape(1,-1), xhat[-N_param:].reshape(1,-1)))
    xhat_pred = np.append(xhat_pred,param)
    F = jacobian_A(A,variables=xhat)

    P_pred = P + dt*(np.matmul(F,P) + np.matmul(P,F.transpose()) + Q)
    # end Predictor phase

    # Corrector phase
    # H = jacobian_H(U,S,N_x) Computed once for all
    G = np.matmul(np.matmul(P_pred,H.transpose()), np.linalg.inv(np.matmul(np.matmul(H,P_pred),H.transpose())+R) ) # Kalman gain computation
    obs = xhat_pred[:-N_param] @ (U * S).T @ (e_).T
    if N_obs == 1:
        xhat = xhat_pred + np.matmul(G, np.expand_dims(out-obs,axis=0) )
    else:
        xhat = xhat_pred + np.matmul(G, (out-obs) )

    obs_check = xhat[:-N_param] @ (U * S).T @ (e_).T

    P = np.matmul( np.matmul(I - np.matmul(G, H),  P_pred), (I - np.matmul(G, H)).transpose() ) + np.matmul(np.matmul(G,R),G.transpose())
    # end Corrector phase

    xhat_corr_taxis[:,i0] = np.matmul(G, np.expand_dims(out-obs,axis=0) )
    xhat_taxis[:,i0] = xhat
    obs_taxis[0,i0]  = obs
    obs_check_taxis[0,i0]  = obs_check
    P_taxis[:,:,i0]  = P


#%% Plot the results ##########################################################################
for i5 in range(N_x+N_param):
    # assumption - diagonal covariance matrix  
    xhat_taxis_piu_sigma[i5,:]  = xhat_taxis[i5,:] + 1.96 * np.sqrt(P_taxis[i5,i5,:])
    xhat_taxis_meno_sigma[i5,:] = xhat_taxis[i5,:] - 1.96 * np.sqrt(P_taxis[i5,i5,:])


#%%
n_time_steps_start_plot = 0
n_time_steps_stop_plot = N_l

param_axis = np.zeros(shape=(N_l,2))
param_axis[:,0] = np.ones(N_l)*param_test_tdc[idx_test][0,0]
param_axis[:,1] = np.ones(N_l)*param_test_tdc[idx_test][0,1]

#%% alpha
param_index = N_x
plt.figure(figsize = [8,4])
plt.plot(t_axis[n_time_steps_start_plot:n_time_steps_stop_plot],xhat_taxis[param_index,n_time_steps_start_plot:n_time_steps_stop_plot],'k')
plt.plot(t_axis[n_time_steps_start_plot:n_time_steps_stop_plot],xhat_taxis_piu_sigma[param_index,n_time_steps_start_plot:n_time_steps_stop_plot],'k',linestyle='--')
plt.plot(t_axis[n_time_steps_start_plot:n_time_steps_stop_plot],xhat_taxis_meno_sigma[param_index,n_time_steps_start_plot:n_time_steps_stop_plot],'k',linestyle='--')
plt.plot(t_axis[n_time_steps_start_plot:n_time_steps_stop_plot],param_axis[n_time_steps_start_plot:n_time_steps_stop_plot,0],'orange')
plt.show()

#%% beta
param_index = N_x +1
plt.figure(figsize = [8,4])
plt.plot(t_axis[n_time_steps_start_plot:n_time_steps_stop_plot],xhat_taxis[param_index,n_time_steps_start_plot:n_time_steps_stop_plot],'k')
plt.plot(t_axis[n_time_steps_start_plot:n_time_steps_stop_plot],xhat_taxis_piu_sigma[param_index,n_time_steps_start_plot:n_time_steps_stop_plot],'k',linestyle='--')
plt.plot(t_axis[n_time_steps_start_plot:n_time_steps_stop_plot],xhat_taxis_meno_sigma[param_index,n_time_steps_start_plot:n_time_steps_stop_plot],'k',linestyle='--')
plt.plot(t_axis[n_time_steps_start_plot:n_time_steps_stop_plot],param_axis[n_time_steps_start_plot:n_time_steps_stop_plot,1],'orange')
plt.show()

#%% embedded 1
plt.figure(figsize = [8,4])
plt.plot(t_axis[n_time_steps_start_plot:n_time_steps_stop_plot],xhat_taxis[0,n_time_steps_start_plot:n_time_steps_stop_plot],'k')
plt.plot(t_axis[n_time_steps_start_plot:n_time_steps_stop_plot],xhat_taxis_piu_sigma[0,n_time_steps_start_plot:n_time_steps_stop_plot],'k',linestyle='--')
plt.plot(t_axis[n_time_steps_start_plot:n_time_steps_stop_plot],xhat_taxis_meno_sigma[0,n_time_steps_start_plot:n_time_steps_stop_plot],'k',linestyle='--')
plt.plot(t_axis[n_time_steps_start_plot:n_time_steps_stop_plot],X_test_tdc[idx_test][n_time_steps_start_plot:n_time_steps_stop_plot,0],'orange')
plt.show()

#%% embedded 2
plt.figure(figsize = [8,4])
plt.plot(t_axis[n_time_steps_start_plot:n_time_steps_stop_plot],xhat_taxis[1,n_time_steps_start_plot:n_time_steps_stop_plot],'k')
plt.plot(t_axis[n_time_steps_start_plot:n_time_steps_stop_plot],xhat_taxis_piu_sigma[1,n_time_steps_start_plot:n_time_steps_stop_plot],'k',linestyle='--')
plt.plot(t_axis[n_time_steps_start_plot:n_time_steps_stop_plot],xhat_taxis_meno_sigma[1,n_time_steps_start_plot:n_time_steps_stop_plot],'k',linestyle='--')
plt.plot(t_axis[n_time_steps_start_plot:n_time_steps_stop_plot],X_test_tdc[idx_test][n_time_steps_start_plot:n_time_steps_stop_plot,1],'orange')

#%% embedded 3
if N_x>2:
    plt.figure(figsize = [8,4])
    plt.plot(t_axis[n_time_steps_start_plot:n_time_steps_stop_plot],xhat_taxis[2,n_time_steps_start_plot:n_time_steps_stop_plot],'k')
    plt.plot(t_axis[n_time_steps_start_plot:n_time_steps_stop_plot],xhat_taxis_piu_sigma[2,n_time_steps_start_plot:n_time_steps_stop_plot],'k',linestyle='--')
    plt.plot(t_axis[n_time_steps_start_plot:n_time_steps_stop_plot],xhat_taxis_meno_sigma[2,n_time_steps_start_plot:n_time_steps_stop_plot],'k',linestyle='--')
    plt.plot(t_axis[n_time_steps_start_plot:n_time_steps_stop_plot],X_test_tdc[idx_test][n_time_steps_start_plot:n_time_steps_stop_plot,2],'orange')


#%% embedded 4
if N_x>3:
    plt.figure(figsize = [8,4])
    plt.plot(t_axis[n_time_steps_start_plot:n_time_steps_stop_plot],xhat_taxis[3,n_time_steps_start_plot:n_time_steps_stop_plot],'k')
    plt.plot(t_axis[n_time_steps_start_plot:n_time_steps_stop_plot],xhat_taxis_piu_sigma[3,n_time_steps_start_plot:n_time_steps_stop_plot],'k',linestyle='--')
    plt.plot(t_axis[n_time_steps_start_plot:n_time_steps_stop_plot],xhat_taxis_meno_sigma[3,n_time_steps_start_plot:n_time_steps_stop_plot],'k',linestyle='--')
    plt.plot(t_axis[n_time_steps_start_plot:n_time_steps_stop_plot],X_test_tdc[idx_test][n_time_steps_start_plot:n_time_steps_stop_plot,3],'orange')

#%% observed variable
plt.figure(figsize = [8,4])
plt.plot(t_axis[n_time_steps_start_plot:n_time_steps_stop_plot],obs_taxis[0,n_time_steps_start_plot:n_time_steps_stop_plot],'k')
plt.plot(t_axis[n_time_steps_start_plot:n_time_steps_stop_plot],out_axis[n_time_steps_start_plot:n_time_steps_stop_plot],'orange')
if add_noise:
    plt.plot(t_axis[n_time_steps_start_plot:n_time_steps_stop_plot],out_no_noise_axis[n_time_steps_start_plot:n_time_steps_stop_plot],'gray')

#%% Save the data for importing them in Matlab (MAT extension)
save_mat_data_ID =  data_root_ID + '\\' + data_ID + '_output_for_matlab.mat'
if add_noise:
    savemat(save_mat_data_ID, {'t_axis': t_axis, 'xhat_taxis': xhat_taxis, 'xhat_taxis_piu_sigma': xhat_taxis_piu_sigma, 'xhat_taxis_meno_sigma': xhat_taxis_meno_sigma,'param_axis':param_axis,'X_test_tdc':X_test_tdc[idx_test],'out_axis':out_axis,'out_no_noise_axis':out_no_noise_axis,'obs_taxis':obs_taxis})
else:
    savemat(save_mat_data_ID, {'t_axis': t_axis, 'xhat_taxis': xhat_taxis, 'xhat_taxis_piu_sigma': xhat_taxis_piu_sigma, 'xhat_taxis_meno_sigma': xhat_taxis_meno_sigma,'param_axis':param_axis,'X_test_tdc':X_test_tdc[idx_test],'out_axis':out_axis,'obs_taxis':obs_taxis})

# %%
