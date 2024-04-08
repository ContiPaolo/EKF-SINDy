#%%
######################       LIBRARIES       ######################
%matplotlib widget

import pysindy as ps
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from scipy.integrate import odeint
from scipy.linalg import hankel
from scipy.io import savemat
from sklearn.utils.extmath import randomized_svd
from sklearn.model_selection import train_test_split

from utils_DEKF import add_noise_with_snr

seed = 1
#TODO: check testing instances are distributed kinda uniformly over the domain

#%% data root
root_ID = r"D:\Luca\Dati\DEKF"
problem_ID = r"2DOF_nonLinear"
data_ID = r"1_4SENDy"
data_root_ID = root_ID + '\\' + problem_ID + '\\' + data_ID

#%%
######################       PROBLEM SETUP       ######################
'''
We consider two coupled oscillators with the following equations of motion:
u1'' + ω1^2 u1 + 2ξω1 u1' + α u2 = 0,
u2'' + ω2^2 u2 + 2ξω2 u2' + α u1 + σ u1^2 + γ u2^3 = 0,

which can be rewritten at first order as:
u1' = v1,
v1' = - ω1^2 u1 - 2ξω1 v1 - α u2,
u2' = v2,
v2' = - ω2^2 u2 - 2ξω2 v2 - α u1 - σ u1^2 - γ u2^3.

We suppose we are only able to measure the position of the first oscillator u1, 
and we want to
(1) identify the equations of motion from the data
(2) estimate the natural frequencies ω2 of the hidden oscillator.
'''

system = {
    #Natural frequency of the observed oscillator:
    'w1' : 1.,
    #w2 instead will vary and it is the parameter we want to estimate

    #Damping:
    'csi1' : 1e-2,
    'csi2' : 5e-3,

    #Coupling:
    'sigma' : 2e-3, #quadratic
    'alpha' : -1e-1, #linear 
    #TODO: - increasing the coupling makes time-delay embedding hard to capture the behavior with just 4 modes
    #      - decreasing the coupling makes the observed oscillator less sensitive wrt changes in parameters in the second (hidden) one

    #Nonlinearities:
    'gamma' : 1e-3, #cubic

    #Time:
    't0' : 0.,
    'T' : 200.,

    #Time interval:
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
    sigma = system['sigma']
    alpha = system['alpha']

    du1 = v1
    dv1 = - (w1**2)*u1 -2.*csi1*w1*v1  - alpha*u2 
    du2 = v2
    dv2 = - (w2**2)*u2  -2.*csi2*w2*v2 - sigma*(u1**2) - alpha*u1 - gamma*(u2**3) 

    return np.array([du1, dv1, du2, dv2])

#Define the time frame:
t0, T, dt = system['t0'], system['T'], system['dt']

t = np.arange(t0,T,dt)
Nt = len(t)

        
# %%
######################       GENERATE DATA       ######################
#Number of instances
n_ics = 16

#Generate random ICs, while we consider zero initial velocities:
'''np.random.seed(seed=seed)
min_u, max_u = -3, 3 

u1_0 = np.random.uniform(min_u,max_u,n_ics)
u2_0 = np.random.uniform(min_u, max_u,n_ics)

min_x1, max_x1 = -3, 3
#random sign variable
sign = np.random.choice([-1,1],size = n_ics)
min_scalex2, max_scalex2 = 1, 3
scale = np.random.uniform(min_scalex2, max_scalex2,n_ics)

u1_0 = np.random.uniform(min_x1,max_x1,n_ics)
u2_0 = sign * scale * u1_0
#u1_0_train = np.random.uniform(min_x1,min_x1,n_ics)
#u2_0_train = np.random.uniform(min_x2, max_x2,n_ics)'''

np.random.seed(seed=seed)  
u1_0 = np.random.normal(-2,1,n_ics)
u2_0 = np.random.normal(3,1,n_ics)
#TODO: give high initial value to the hidden oscillator such that its dynamic contribution is significant

v1_0 = np.repeat(0.,n_ics)
v2_0 = np.repeat(0.,n_ics)

#Generate equispaced parameters (natural frequency of the hidden oscilator, ω2) in the range P =  [1., 2.]
min_w2, max_w2 = 1., 2.
w2= np.linspace(min_w2, max_w2, n_ics, endpoint = True)

#Generate data
X = []

for i in range(n_ics):
    y0 = np.array([u1_0[i], v1_0[i], u2_0[i], v2_0[i]])
    system['w2'] = w2[i]
    sol = odeint(f, y0, t) 
    X.append(sol)

    u1, v1, u2, v2 = np.hsplit(sol, 4)

    #Plot the solution:
    # plt.figure(figsize=(7,3))
    # plt.plot(t,u1, label = '$u_1$')
    # plt.plot(t,u2, label = '$u_2$')
    # plt.legend(fontsize = 12)
    # plt.xlabel('t', fontsize = 12)

    # plt.show()

X = np.array(X)
X_train, X_test, w2_train, w2_test = train_test_split(X, w2, test_size=0.2, random_state=seed)
n_train, n_test = len(X_train), len(X_test)

print(w2_test)
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
#TODO: length can be increased

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
# plt.figure(figsize = [13,16])
# for i in range(n_tdc):
#     plt.subplot(421 + 2*i)
#     plt.plot(X_train[0,:Nt_h,i], 'b')
#     plt.title(f'Original variable {i+1}', fontsize = 12)

#     plt.subplot(421 + 2*i+1)
#     plt.plot(u_h[i*Nt_h:(i+1)*Nt_h,i], 'r')
#     plt.title(f'Time-delay coordinate {i+1}', fontsize = 12)

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
    param_tdc.append(np.repeat(w2_train[i], Nt)[:Nt_h])
    feature_names = ["u" + str(i+1) for i in range(n_tdc)]
    feature_names +=  ["w2"]

for i in range(n_test):
    u1_ = X_test[i,:,0]
    H_ = create_hankel(u1_, length, shift)
    u_h = H_.T @  (U * 1/S)

    #Baolo
    X_test_tdc.append(u_h[:,:n_tdc])
    dX_test_tdc.append(np.gradient(u_h[:,:n_tdc], dt, axis = 0))
    param_test_tdc.append(np.repeat(w2_test[i], Nt)[:Nt_h])

    #Luca
    # X_test_tdc.append(u_h[:-1,:n_tdc])
    # dX_test_tdc.append(u_h[1:,:n_tdc])
    # param_test_tdc.append(np.repeat(w2_test[i], Nt)[:Nt_h-1])


# %%
#####################       GENERATE DATA for SINDy già trainata ##############
# NON RETRAINARE SINDy
#Number of instances
n_ics = 4

#Generate random ICs, while we consider zero initial velocities:
'''np.random.seed(seed=seed)
min_u, max_u = -3, 3 

u1_0 = np.random.uniform(min_u,max_u,n_ics)
u2_0 = np.random.uniform(min_u, max_u,n_ics)

min_x1, max_x1 = -3, 3
#random sign variable
sign = np.random.choice([-1,1],size = n_ics)
min_scalex2, max_scalex2 = 1, 3
scale = np.random.uniform(min_scalex2, max_scalex2,n_ics)

u1_0 = np.random.uniform(min_x1,max_x1,n_ics)
u2_0 = sign * scale * u1_0
#u1_0_train = np.random.uniform(min_x1,min_x1,n_ics)
#u2_0_train = np.random.uniform(min_x2, max_x2,n_ics)'''

np.random.seed(seed=seed)  
u1_0 = np.random.normal(-2,1,n_ics)
u2_0 = np.random.normal(3,1,n_ics)
#TODO: give high initial value to the hidden oscillator such that its dynamic contribution is significant

v1_0 = np.repeat(0.,n_ics)
v2_0 = np.repeat(0.,n_ics)

#Generate equispaced parameters (natural frequency of the hidden oscilator, ω2) in the range P =  [1., 2.]
min_w2, max_w2 = 2.1, 2.3
w2= np.linspace(min_w2, max_w2, n_ics, endpoint = True)

#Generate data
X = []

for i in range(n_ics):
    y0 = np.array([u1_0[i], v1_0[i], u2_0[i], v2_0[i]])
    system['w2'] = w2[i]
    sol = odeint(f, y0, t) 
    X.append(sol)

    u1, v1, u2, v2 = np.hsplit(sol, 4)

    #Plot the solution:
    # plt.figure(figsize=(7,3))
    # plt.plot(t,u1, label = '$u_1$')
    # plt.plot(t,u2, label = '$u_2$')
    # plt.legend(fontsize = 12)
    # plt.xlabel('t', fontsize = 12)

    # plt.show()

X = np.array(X)
X_train, X_test, w2_train, w2_test = train_test_split(X, w2, test_size=0.2, random_state=seed)
n_train, n_test = len(X_train), len(X_test)

print(w2_test)

def create_hankel(vector, length, shift = 1):
    p = len(vector)-(length*shift)+1
    H = vector[0:p]
    for i in range(1,length):
        H = np.vstack([H, vector[(i*shift):(i*shift)+p]])
    return H

#Create Hankel matrix from observations of u1:
length, shift = 200, 1
#TODO: length can be increased

for i in range(n_train):
    u1 = X_train[i,:,0]
    H1 = create_hankel(u1.flatten(), length, shift)
    if i == 0:
        H = H1
    else:
        H = np.hstack([H,H1])

#Compute time-delay coordinates:
u_h = H.T @  (U * 1/S)

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
    param_tdc.append(np.repeat(w2_train[i], Nt)[:Nt_h])
    feature_names = ["u" + str(i+1) for i in range(n_tdc)]
    feature_names +=  ["w2"]

for i in range(n_test):
    u1_ = X_test[i,:,0]
    H_ = create_hankel(u1_, length, shift)
    u_h = H_.T @  (U * 1/S)

    #Baolo
    X_test_tdc.append(u_h[:,:n_tdc])
    dX_test_tdc.append(np.gradient(u_h[:,:n_tdc], dt, axis = 0))
    param_test_tdc.append(np.repeat(w2_test[i], Nt)[:Nt_h])





# %%
# ######################      CREATE SINDy model       ######################
# model = ps.SINDy(feature_names  = feature_names, feature_library= ps.PolynomialLibrary(degree = 3), optimizer=ps.STLSQ(threshold=5e-4))
# model.fit(X_tdc, t=dt, multiple_trajectories=True, u = param_tdc, x_dot = dX_tdc) 

# model.print()

# %%
######################  (a mano)    CREATE SINDy model       ######################
if n_tdc==2:
    feature_names = ["x1", "x2", "w2"]
    sindy_library = [lambda x: x, 
                    lambda x1, x2, w2: x1*x1, #secondo grado
                    lambda x1, x2, w2: x1*x2,
                    lambda x1, x2, w2: x1*w2,

                    lambda x1, x2, w2: x2*x2,
                    lambda x1, x2, w2: x2*w2,

                    lambda x1, x2, w2: w2*w2,
                    
                    lambda x1, x2, w2: x1*x1*x1, 
                    lambda x1, x2, w2: x1*x1*x2,
                    lambda x1, x2, w2: x1*x1*w2,

                    lambda x1, x2, w2: x2*x2*x1, 
                    lambda x1, x2, w2: x2*x2*x2,
                    lambda x1, x2, w2: x2*x2*w2,
                    
                    lambda x1, x2, w2: w2*w2*x1,
                    lambda x1, x2, w2: w2*w2*x2,
                    lambda x1, x2, w2: w2*w2*w2,
                    
                    lambda x1, x2, w2: x1*x2*w2,            
                    ]
    sindy_library_names = [
        "x1", "x2", "w2",

        "x1^2", "x1*x2", "x1*w2",
        "x2^2", "x2*w2",
        "w2^2",

        "x1^3", "x1^2*x2", "x1^2*w2",
        "x2^2*x1", "x2^3", "x2^2*w2",
        "w2^2*x1", "w2^2*x2", "w2^3",

        "x1*x2*w2"
    ]

elif n_tdc==3:
    feature_names = ["x1", "x2", "x3", "w2"]
    sindy_library = [lambda x: x, 
                    lambda x1, x2, x3, w2: x1*x1, #secondo grado
                    lambda x1, x2, x3, w2: x1*x2,
                    lambda x1, x2, x3, w2: x1*x3,
                    lambda x1, x2, x3, w2: x1*w2,

                    lambda x1, x2, x3, w2: x2*x2,
                    lambda x1, x2, x3, w2: x2*x3,
                    lambda x1, x2, x3, w2: x2*w2,

                    lambda x1, x2, x3, w2: x3*x3,
                    lambda x1, x2, x3, w2: x3*w2,

                    lambda x1, x2, x3, w2: w2*w2,
                    
                    lambda x1, x2, x3, w2: x1*x1*x1, 
                    lambda x1, x2, x3, w2: x1*x1*x2,
                    lambda x1, x2, x3, w2: x1*x1*x3,
                    lambda x1, x2, x3, w2: x1*x1*w2,

                    lambda x1, x2, x3, w2: x2*x2*x1, 
                    lambda x1, x2, x3, w2: x2*x2*x2,
                    lambda x1, x2, x3, w2: x2*x2*x3,
                    lambda x1, x2, x3, w2: x2*x2*w2,

                    lambda x1, x2, x3, w2: x3*x3*x1,
                    lambda x1, x2, x3, w2: x3*x3*x2,
                    lambda x1, x2, x3, w2: x3*x3*x3,
                    lambda x1, x2, x3, w2: x3*x3*w2,
                    
                    lambda x1, x2, x3, w2: w2*w2*x1,
                    lambda x1, x2, x3, w2: w2*w2*x2,
                    lambda x1, x2, x3, w2: w2*w2*x3,
                    lambda x1, x2, x3, w2: w2*w2*w2,
                    
                    lambda x1, x2, x3, w2: x1*x2*x3,
                    lambda x1, x2, x3, w2: x1*x2*w2,
                    lambda x1, x2, x3, w2: x1*x3*w2,

                    lambda x1, x2, x3, w2: x2*x3*w2              
                    ]

    sindy_library_names = [
        "x1", "x2", "x3", "w2",

        "x1^2", "x1*x2", "x1*x3", "x1*w2",
        "x2^2", "x2*x3", "x2*w2",
        "x3^2", "x3*w2",
        "w2^2",

        "x1^3", "x1^2*x2", "x1^2*x3", "x1^2*w2",
        "x2^2*x1", "x2^3", "x2^2*x3", "x2^2*w2",
        "x3^2*x1", "x3^2*x2", "x3^3", "x3^2*w2",
        "w2^2*x1", "w2^2*x2", "w2^2*x3", "w2^3",

        "x1*x2*x3", "x1*x2*w2","x1*x3*w2",
        
        "x2*x3*w2"   
    ]

elif n_tdc==4:
    feature_names = ["x1", "x2", "x3", "x4", "w2"]
    sindy_library = [lambda x: x, 
                    lambda x1, x2, x3, x4, w2: x1*x1, #secondo grado
                    lambda x1, x2, x3, x4, w2: x1*x2,
                    lambda x1, x2, x3, x4, w2: x1*x3,
                    lambda x1, x2, x3, x4, w2: x1*x4,
                    lambda x1, x2, x3, x4, w2: x1*w2,

                    lambda x1, x2, x3, x4, w2: x2*x2,
                    lambda x1, x2, x3, x4, w2: x2*x3,
                    lambda x1, x2, x3, x4, w2: x2*x4,
                    lambda x1, x2, x3, x4, w2: x2*w2,

                    lambda x1, x2, x3, x4, w2: x3*x3,
                    lambda x1, x2, x3, x4, w2: x3*x4,
                    lambda x1, x2, x3, x4, w2: x3*w2,

                    lambda x1, x2, x3, x4, w2: x4*x4,
                    lambda x1, x2, x3, x4, w2: x4*w2,

                    lambda x1, x2, x3, x4, w2: w2*w2,
                    
                    lambda x1, x2, x3, x4, w2: x1*x1*x1, 
                    lambda x1, x2, x3, x4, w2: x1*x1*x2,
                    lambda x1, x2, x3, x4, w2: x1*x1*x3,
                    lambda x1, x2, x3, x4, w2: x1*x1*x4,
                    lambda x1, x2, x3, x4, w2: x1*x1*w2,

                    lambda x1, x2, x3, x4, w2: x2*x2*x1, 
                    lambda x1, x2, x3, x4, w2: x2*x2*x2,
                    lambda x1, x2, x3, x4, w2: x2*x2*x3,
                    lambda x1, x2, x3, x4, w2: x2*x2*x4,
                    lambda x1, x2, x3, x4, w2: x2*x2*w2,

                    lambda x1, x2, x3, x4, w2: x3*x3*x1,
                    lambda x1, x2, x3, x4, w2: x3*x3*x2,
                    lambda x1, x2, x3, x4, w2: x3*x3*x3,
                    lambda x1, x2, x3, x4, w2: x3*x3*x4,
                    lambda x1, x2, x3, x4, w2: x3*x3*w2,


                    lambda x1, x2, x3, x4, w2: x4*x4*x1,
                    lambda x1, x2, x3, x4, w2: x4*x4*x2,
                    lambda x1, x2, x3, x4, w2: x4*x4*x3,
                    lambda x1, x2, x3, x4, w2: x4*x4*x4,
                    lambda x1, x2, x3, x4, w2: x4*x4*w2,
                    
                    lambda x1, x2, x3, x4, w2: w2*w2*x1,
                    lambda x1, x2, x3, x4, w2: w2*w2*x2,
                    lambda x1, x2, x3, x4, w2: w2*w2*x3,
                    lambda x1, x2, x3, x4, w2: w2*w2*x4,
                    lambda x1, x2, x3, x4, w2: w2*w2*w2,
                    
                    lambda x1, x2, x3, x4, w2: x1*x2*x3,
                    lambda x1, x2, x3, x4, w2: x1*x2*x4,
                    lambda x1, x2, x3, x4, w2: x1*x2*w2,
                    lambda x1, x2, x3, x4, w2: x1*x3*x4,
                    lambda x1, x2, x3, x4, w2: x1*x3*w2,
                    lambda x1, x2, x3, x4, w2: x1*x4*w2,

                    lambda x1, x2, x3, x4, w2: x2*x3*x4,
                    lambda x1, x2, x3, x4, w2: x2*x3*w2,
                    lambda x1, x2, x3, x4, w2: x2*x4*w2,                 

                    lambda x1, x2, x3, x4, w2: x3*x4*w2                  
                    ]

    sindy_library_names = [
        "x1", "x2", "x3", "x4", "w2",

        "x1^2", "x1*x2", "x1*x3", "x1*x4", "x1*w2",
        "x2^2", "x2*x3", "x2*x4", "x2*w2",
        "x3^2", "x3*x4", "x3*w2",
        "x4^2", "x4*w2",
        "w2^2",

        "x1^3", "x1^2*x2", "x1^2*x3", "x1^2*x4", "x1^2*w2",
        "x2^2*x1", "x2^3", "x2^2*x3", "x2^2*x4", "x2^2*w2",
        "x3^2*x1", "x3^2*x2", "x3^3", "x3^2*x4", "x3^2*w2",
        "x4^2*x1", "x4^2*x2", "x4^2*x3", "x4^3", "x4^2*w2",
        "w2^2*x1", "w2^2*x2", "w2^2*x3", "w2^2*x4", "w2^3",

        "x1*x2*x3", "x1*x2*x4", "x1*x2*w2", "x1*x3*x4", "x1*x3*w2", "x1*x4*w2",
        
        "x2*x3*x4", "x2*x3*w2", "x2*x4*w2",
        
        "x3*x4*w2"   
    ]

handModel = ps.SINDy(feature_names = sindy_library_names, feature_library= ps.CustomLibrary(library_functions = sindy_library),optimizer=ps.STLSQ(threshold=5e-4))
handModel.fit(X_tdc, t=dt, multiple_trajectories=True, u = param_tdc, x_dot = dX_tdc) 
A = handModel.coefficients()
n_features = A.shape[1] #number of features

#%%
# sindy model
def model_Asindy(A, variables):
    '''
    A: matrix of SINDy coefficients for the states and parameters (transition equation)
    variables: [x1, x2, x3, x4, w2]
    '''

    if len(variables)==3:
        x1, x2, w2 = variables

        sindy_library_A = np.array([x1,x2,w2,
                                    x1*x1,x1*x2,x1*w2,
                                    x2*x2,x2*w2,
                                    w2*w2,
                                    x1*x1*x1, x1*x1*x2, x1*x1*w2,
                                    x2*x2*x1, x2*x2*x2, x2*x2*w2,
                                    w2*w2*x1, w2*w2*x2, w2*w2*w2,
                                    x1*x2*w2])
        
    elif len(variables)==4:
        x1, x2, x3, w2 = variables

        sindy_library_A = np.array([x1,x2,x3,w2,
                                    x1*x1,x1*x2,x1*x3,x1*w2,
                                    x2*x2,x2*x3,x2*w2,
                                    x3*x3,x3*w2,
                                    w2*w2,
                                    x1*x1*x1, x1*x1*x2, x1*x1*x3, x1*x1*w2,
                                    x2*x2*x1, x2*x2*x2, x2*x2*x3, x2*x2*w2,
                                    x3*x3*x1, x3*x3*x2, x3*x3*x3, x3*x3*w2,
                                    w2*w2*x1, w2*w2*x2, w2*w2*x3, w2*w2*w2,
                                    x1*x2*x3, x1*x2*w2, x1*x3*w2,                               
                                    x2*x3*w2])

    elif len(variables)==5:
        x1, x2, x3, x4, w2 = variables

        sindy_library_A = np.array([x1,x2,x3,x4,w2,
                                    x1*x1,x1*x2,x1*x3,x1*x4,x1*w2,
                                    x2*x2,x2*x3,x2*x4,x2*w2,
                                    x3*x3,x3*x4,x3*w2,
                                    x4*x4,x4*w2,
                                    w2*w2,
                                    x1*x1*x1, x1*x1*x2, x1*x1*x3, x1*x1*x4, x1*x1*w2,
                                    x2*x2*x1, x2*x2*x2, x2*x2*x3, x2*x2*x4, x2*x2*w2,
                                    x3*x3*x1, x3*x3*x2, x3*x3*x3, x3*x3*x4, x3*x3*w2,
                                    x4*x4*x1, x4*x4*x2, x4*x4*x3, x4*x4*x4, x4*x4*w2,
                                    w2*w2*x1, w2*w2*x2, w2*w2*x3, w2*w2*x4, w2*w2*w2,
                                    x1*x2*x3, x1*x2*x4, x1*x2*w2, x1*x3*x4, x1*x3*w2, x1*x4*w2,                                
                                    x2*x3*x4, x2*x3*w2, x2*x4*w2,                                
                                    x3*x4*w2])
    return A @ sindy_library_A

#%%
# jacobian matrix A

def jacobian_A(A, variables):
    '''
    A: matrix of SINDy coefficients for the states and parameters (dynamic equation)
    variables: [x1,x2,x3,x4,w2] values at which the jacobian is evaluated
    x1,x2,x3,x4 are the time-delay coordinates
    w2 is the parameter to be identified
    '''

    n = A.shape[0] #number of equations
    n_features = A.shape[1] #number of features
    n_var = len(variables)
    J_library = np.zeros((n_features, n_var))

    if n_var==3:
        x1, x2, w2 = variables
        #dA/dx1
        J_library[:,0] = np.array([1., 0., 0.,
                            2.*x1, x2, w2,
                            0., 0.,
                            0.,
                            3.*x1*x1,2.*x1*x2,2.*x1*w2,
                            x2*x2,         0.,      0.,
                            w2*w2,         0.,      0.,
                            x2*w2])

        #dA/dx2
        J_library[:,1] = np.array([0., 1., 0.,
                            0., x1, 0.,
                            2.*x2, w2,
                            0.,
                            0.,         x1*x1,      0.,
                            2.*x2*x1,3.*x2*x2,2.*x2*w2,
                            0.,         w2*w2,      0.,
                            x1*w2])
        
        #dA/dw2
        J_library[:,2] = np.array([0., 0., 1.,
                            0., 0., x1,
                            0., x2,
                            2.*w2,
                            0.,           0.,     x1*x1,
                            0.,           0.,     x2*x2,
                            2.*w2*x1, 2.*w2*x2,3.*w2*w2,
                            x1*x2])
        
    elif n_var==4:
        x1, x2, x3, w2 = variables

        #dA/dx1
        J_library[:,0] = np.array([1., 0., 0., 0.,
                            2.*x1, x2, x3, w2,
                            0., 0.,0.,
                            0., 0., 
                            0.,
                            3.*x1*x1,2.*x1*x2,2.*x1*x3,2.*x1*w2,
                            x2*x2,         0.,      0.,      0.,
                            x3*x3,         0.,      0.,      0.,
                            w2*w2,         0.,      0.,      0.,
                            x2*x3,      x2*w2,   x3*w2,                        
                            0.])

        #dA/dx2
        J_library[:,1] = np.array([0., 1., 0., 0.,
                            0., x1, 0., 0.,
                            2.*x2, x3, w2,
                            0., 0.,
                            0.,
                            0.,         x1*x1,      0.,      0.,
                            2.*x2*x1,3.*x2*x2,2.*x2*x3,2.*x2*w2,
                            0.,         x3*x3,      0.,      0.,
                            0.,         w2*w2,      0.,      0.,
                            x1*x3,      x1*w2,      0.,                        
                            x3*w2])
        
        #dA/dx3
        J_library[:,2] = np.array([ 0., 0., 1.,0.,
                            0., 0., x1, 0.,
                            0., x2, 0.,
                            2.*x3, w2,
                            0.,
                            0.,           0.,    x1*x1,      0.,
                            0.,           0.,    x2*x2,      0.,
                            2.*x3*x1,2.*x3*x2,3.*x3*x3,2.*x3*w2,
                            0.,           0.,    w2*w2,      0.,
                            x1*x2,        0.,    x1*w2,                       
                            x2*w2])
        
        #dA/dw2
        J_library[:,3] = np.array([0., 0., 0., 1.,
                            0., 0., 0., x1,
                            0., 0., x2,
                            0., x3,
                            2.*w2,
                            0.,           0.,       0.,      x1*x1,
                            0.,           0.,       0.,      x2*x2,
                            0.,           0.,       0.,      x3*x3,
                            2.*w2*x1, 2.*w2*x2,2.*w2*x3,  3.*w2*w2,
                            0.,        x1*x2,    x1*x3,                      
                            x2*x3])

    elif n_var==5:
        x1, x2, x3, x4, w2 = variables

        #dA/dx1
        J_library[:,0] = np.array([1., 0., 0., 0., 0.,
                            2.*x1, x2, x3, x4, w2,
                            0., 0., 0., 0.,
                            0., 0.,0.,
                            0., 0., 
                            0.,
                            3.*x1*x1,2.*x1*x2,2.*x1*x3,2.*x1*x4,2.*x1*w2,
                            x2*x2,         0.,     0.,      0.,      0.,
                            x3*x3,         0.,     0.,      0.,      0.,
                            x4*x4,         0.,     0.,      0.,      0.,
                            w2*w2,         0.,     0.,      0.,      0.,
                            x2*x3,      x2*x4,   x2*w2,   x3*x4,   x3*w2,   x4*w2,                        
                            0.,            0.,      0.,                        
                            0.])

        #dA/dx2
        J_library[:,1] = np.array([0., 1., 0., 0., 0.,
                            0., x1, 0., 0., 0.,
                            2.*x2, x3, x4, w2,
                            0., 0., 0.,
                            0., 0.,
                            0.,
                            0.,         x1*x1,      0.,      0.,      0.,
                            2.*x2*x1,3.*x2*x2,2.*x2*x3,2.*x2*x4,2.*x2*w2,
                            0.,         x3*x3,      0.,      0.,      0.,
                            0.,         x4*x4,      0.,      0.,      0.,
                            0.,         w2*w2,      0.,      0.,      0.,
                            x1*x3,     x1*x4,    x1*w2,      0.,      0.,      0.,                        
                            x3*x4,     x3*w2,    x4*w2,                        
                            0.])
        
        #dA/dx3
        J_library[:,2] = np.array([ 0., 0., 1., 0., 0.,
                            0., 0., x1, 0., 0.,
                            0., x2, 0., 0.,
                            2.*x3, x4, w2,
                            0., 0.,
                            0.,
                            0.,           0.,    x1*x1,      0.,      0.,
                            0.,           0.,    x2*x2,      0.,      0.,
                            2.*x3*x1,2.*x3*x2,3.*x3*x3,2.*x3*x4,2.*x3*w2,
                            0.,           0.,    x4*x4,      0.,      0.,
                            0.,           0.,    w2*w2,      0.,      0.,
                            x1*x2,        0.,       0.,   x1*x4,   x1*w2,     0.,                        
                            x2*x4,     x2*w2,       0.,                        
                            x4*w2])
        
        #dA/dx4
        J_library[:,3] = np.array([ 0., 0., 0., 1., 0.,
                            0., 0., 0., x1, 0.,
                            0., 0., x2, 0.,
                            0., x3, 0.,
                            2.*x4, w2,
                            0.,
                            0.,           0.,       0.,    x1*x1,      0.,
                            0.,           0.,       0.,    x2*x2,      0.,
                            0.,           0.,       0.,    x3*x3,      0.,
                            2.*x4*x1, 2.*x4*x2,2.*x4*x3,3.*x4*x4,2.*x4*w2,
                            0.,           0.,       0.,    w2*w2,      0.,
                            0.,        x1*x2,       0.,    x1*x3,      0.,  x1*w2,                        
                            x2*x3,        0.,    x2*w2,                        
                            x3*w2])
        
        #dA/dw2
        J_library[:,4] = np.array([0., 0., 0., 0., 1.,
                            0., 0., 0., 0., x1,
                            0., 0., 0., x2,
                            0., 0., x3,
                            0., x4,
                            2.*w2,
                            0.,           0.,       0.,      0.,    x1*x1,
                            0.,           0.,       0.,      0.,    x2*x2,
                            0.,           0.,       0.,      0.,    x3*x3,
                            0.,           0.,       0.,      0.,    x4*x4,
                            2.*w2*x1, 2.*w2*x2,2.*w2*x3,2.*w2*x4,3.*w2*w2,
                            0.,           0.,    x1*x2,      0.,   x1*x3,  x1*x4,                        
                            0.,        x2*x3,    x2*x4,                        
                            x3*x4])

    F = A @ J_library

    N_param = 1

    for i3 in range(N_param):
        F = np.append(F, [np.zeros(n+N_param)], axis=0)
        F[n+i3,n_var-N_param+i3] = 0
    
    return F

#%%
# jacobian matrix H

def jacobian_H(U,S,N_x):
    '''
    U, S: matrices obtained by operating the svd of the time delay embedding
    xhat_pred @ (U * S).T
    The Jacobian is equivalent to compute (e)4_1.T (U * S).T (e)200_1
                                          (e)4_2.T (U * S).T (e)200_1 etc.
    '''
    #n = U.shape[0] #number of time-delays (=length)
    #n_variables = U.shape[1] #number of time delays embeddings    
    if N_x==2:
    
        H = np.zeros((1,3))
        che = (U * S).T

        #dH/dx1
        H[0,0] = che[0,0]
        #dH/dx2
        H[0,1] = che[1,0]

    if N_x==3:
    
        H = np.zeros((1,4))
        che = (U * S).T

        #dH/dx1
        H[0,0] = che[0,0]
        #dH/dx2
        H[0,1] = che[1,0]
        #dH/dx3
        H[0,2] = che[2,0]

    if N_x==4:
    
        H = np.zeros((1,5))
        che = (U * S).T

        #dH/dx1
        H[0,0] = che[0,0]
        #dH/dx2
        H[0,1] = che[1,0]
        #dH/dx3
        H[0,2] = che[2,0]
        #dH/dx4
        H[0,3] = che[3,0]

    return H


# %%
# to reproduce the results
model = handModel


# %%
######################      PREDICT  x_test      ######################
idx_test = 0
idx_test_fake = 0
fake_param = param_test_tdc[idx_test_fake][0]
print('Real parameter:', param_test_tdc[idx_test][0], 'Fake parameter:', fake_param)

#Predict the evolution of the time-delay coordinates:
pred = model.simulate(X_test_tdc[idx_test][0,:], t= t[:Nt_h], u = param_test_tdc[idx_test_fake])

#Plot the evolution of the time-delay coordinates:
# plt.figure(figsize = [8,16])
# for i in range(n_tdc):
#     plt.subplot(411 +i)
#     plt.plot(X_test_tdc[idx_test][:,i], 'b', label = 'true')
#     plt.plot(pred[:,i], 'r--', label = 'pred')
#     plt.title(f'Time-delay coordinate {i+1}', fontsize = 12)
#     plt.legend(fontsize = 12)

#%%
######################      RECONSTRUCTION x_test      ######################
#Reconstruct observed variable:
x_rec = pred @ (U * S).T

#Plot the reconstructed variable:
plt.figure(figsize = [8,4])
plt.plot(X_test[idx_test,:,0], 'b', label = 'true')
plt.plot(x_rec[:,0], 'r--', label = 'pred')
plt.title(f'Reconstructed variable', fontsize = 12)
plt.legend(fontsize = 12)
plt.show()

print('MSE', mean_squared_error(X_test[idx_test,:Nt_h-1,0], x_rec[:Nt_h,0]))






# %%
######################      PREDICT  x_train     ######################
idx_train = 2
idx_train_fake = 2
fake_param = param_tdc[idx_train][0]
print('Real parameter:', param_tdc[idx_train][0], 'Fake parameter:', fake_param)

#Predict the evolution of the time-delay coordinates:
pred = model.simulate(X_tdc[idx_train][0,:], t= t[:Nt_h], u = param_tdc[idx_train_fake])

#Plot the evolution of the time-delay coordinates:
# plt.figure(figsize = [8,16])
# for i in range(n_tdc):
#     plt.subplot(411 +i)
#     plt.plot(X_tdc[idx_train][:,i], 'b', label = 'true')
#     plt.plot(pred[:,i], 'r--', label = 'pred')
#     plt.title(f'Time-delay coordinate {i+1}', fontsize = 12)
#     plt.legend(fontsize = 12)

#%%
######################      RECONSTRUCTION  x_train      ######################
#Reconstruct observed variable:
x_rec = pred @ (U * S).T

#Plot the reconstructed variable:
plt.figure(figsize = [8,4])
plt.plot(X_train[idx_train,:,0], 'b', label = 'true')
plt.plot(x_rec[:,0], 'r--', label = 'pred')
plt.title(f'Reconstructed variable', fontsize = 12)
plt.legend(fontsize = 12)

print('MSE', mean_squared_error(X_train[idx_train,:Nt_h-1,0], x_rec[:,0]))

######################      OUTPUT INFO       ######################

'''
OUTPUTS:
    model:          SINDy model
    X_tdc:          time delayed coordinates. It is a list (of length n_train=12, that is the number of training time-series),
                    where each element is an array of shape (Nt_tdc, n_coordinates) = (190001, n_tdc), that is the time-history of length Nt_tdc,
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
param_rel_error = 0.2#0.2#-0.2
idx_test = 0#1#0
correct_param = param_test_tdc[idx_test][0]

state_param = correct_param *(1+param_rel_error)

N_x     = n_tdc # numero variabili di stato uguale al numero di time-delay coordinates
N_param = 1     # numero parametri da identificare
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
param_axis_fake = np.ones(N_l)*state_param
pred = model.simulate(X_test_tdc[idx_test][0,:], t= t[:Nt_h], u = param_axis_fake)

#Plot the evolution of the time-delay coordinates:
plt.figure(figsize = [8,16])
for i in range(n_tdc):
    plt.subplot(411 +i)
    plt.plot(X_test_tdc[idx_test][:,i], 'b', label = 'true')
    plt.plot(pred[:,i], 'r--', label = 'pred')
    plt.title(f'Time-delay coordinate {i+1}', fontsize = 12)
    plt.legend(fontsize = 12)

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

#1_2SENDy
# # parameters for Kalman filter tuning  --> NEW VALUES PB snr 15
# # initialisation values of the uncertainty associated to the state variables
# p0_x     = 1e-6
# p0_param = 1e-2
# # process noise
# q_x      = 1e-7
# q_param  = 1e-7
# # measurement noise
# r_x_obs  = 1e+0

# #1_3SENDy
# # parameters for Kalman filter tuning  --> NEW VALUES PB snr 15 # tolerance 5e-4
# # initialisation values of the uncertainty associated to the state variables
# p0_x_1     = 1e-6
# p0_x_2     = 1e-6
# p0_x_3     = 1e-6
# p0_param = 1e-2
# # process noise
# q_x_1      = 1e-9 # prime due variabili embedded
# q_x_2      = 4e-8
# q_x_3      = 8e-8
# q_param  = 1e-6
# # measurement noise
# r_x_obs  = 5e-3

#1_4SENDy
# parameters for Kalman filter tuning  --> NEW VALUES PB snr 15 # tolerance 5e-4
# initialisation values of the uncertainty associated to the state variables
p0_x_1     = 1e-6
p0_x_2     = 1e-6
p0_x_3     = 1e-6
p0_param = 1e-2
# process noise
q_x_1      = 1e-10 # prime due variabili embedded
q_x_2      = 1e-8
q_x_3      = 1e-8
q_param  = 2e-6
# measurement noise
r_x_obs  = 5e-4


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
P[:2,:2] = P[:2,:2]*p0_x_1
P[2,2] = P[2,2]*p0_x_2
P[3,3] = P[3,3]*p0_x_3
P[-N_param:,-N_param:] = P[-N_param:,-N_param:]*p0_param

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
    xhat_pred = xhat[0:-N_param] + dt*(model_Asindy(A,variables=xhat))
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

param_index = N_x

#%% w2
plt.figure(figsize = [8,4])
param_axis = np.ones(N_l)*param_test_tdc[idx_test][0]
plt.plot(t_axis[n_time_steps_start_plot:n_time_steps_stop_plot],xhat_taxis[param_index,n_time_steps_start_plot:n_time_steps_stop_plot],'k')
plt.plot(t_axis[n_time_steps_start_plot:n_time_steps_stop_plot],xhat_taxis_piu_sigma[param_index,n_time_steps_start_plot:n_time_steps_stop_plot],'k',linestyle='--')
plt.plot(t_axis[n_time_steps_start_plot:n_time_steps_stop_plot],xhat_taxis_meno_sigma[param_index,n_time_steps_start_plot:n_time_steps_stop_plot],'k',linestyle='--')
plt.plot(t_axis[n_time_steps_start_plot:n_time_steps_stop_plot],param_axis[n_time_steps_start_plot:n_time_steps_stop_plot],'orange')
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
plt.show()

#%% embedded 3
if N_x>2:
    plt.figure(figsize = [8,4])
    plt.plot(t_axis[n_time_steps_start_plot:n_time_steps_stop_plot],xhat_taxis[2,n_time_steps_start_plot:n_time_steps_stop_plot],'k')
    plt.plot(t_axis[n_time_steps_start_plot:n_time_steps_stop_plot],xhat_taxis_piu_sigma[2,n_time_steps_start_plot:n_time_steps_stop_plot],'k',linestyle='--')
    plt.plot(t_axis[n_time_steps_start_plot:n_time_steps_stop_plot],xhat_taxis_meno_sigma[2,n_time_steps_start_plot:n_time_steps_stop_plot],'k',linestyle='--')
    plt.plot(t_axis[n_time_steps_start_plot:n_time_steps_stop_plot],X_test_tdc[idx_test][n_time_steps_start_plot:n_time_steps_stop_plot,2],'orange')
    plt.show()

#%% embedded 4
if N_x>3:
    plt.figure(figsize = [8,4])
    plt.plot(t_axis[n_time_steps_start_plot:n_time_steps_stop_plot],xhat_taxis[3,n_time_steps_start_plot:n_time_steps_stop_plot],'k')
    plt.plot(t_axis[n_time_steps_start_plot:n_time_steps_stop_plot],xhat_taxis_piu_sigma[3,n_time_steps_start_plot:n_time_steps_stop_plot],'k',linestyle='--')
    plt.plot(t_axis[n_time_steps_start_plot:n_time_steps_stop_plot],xhat_taxis_meno_sigma[3,n_time_steps_start_plot:n_time_steps_stop_plot],'k',linestyle='--')
    plt.plot(t_axis[n_time_steps_start_plot:n_time_steps_stop_plot],X_test_tdc[idx_test][n_time_steps_start_plot:n_time_steps_stop_plot,3],'orange')
    plt.show()

#%% observed variable
plt.figure(figsize = [8,4])
plt.plot(t_axis[n_time_steps_start_plot:n_time_steps_stop_plot],obs_taxis[0,n_time_steps_start_plot:n_time_steps_stop_plot],'k')
plt.plot(t_axis[n_time_steps_start_plot:n_time_steps_stop_plot],out_axis[n_time_steps_start_plot:n_time_steps_stop_plot],'orange')
if add_noise:
    plt.plot(t_axis[n_time_steps_start_plot:n_time_steps_stop_plot],out_no_noise_axis[n_time_steps_start_plot:n_time_steps_stop_plot],'gray')
plt.show()

#%% Save the data for importing them in Matlab (MAT extension)
save_mat_data_ID =  data_root_ID + '\\' + data_ID + '_output_for_matlab.mat'
if add_noise:
    savemat(save_mat_data_ID, {'t_axis': t_axis, 'xhat_taxis': xhat_taxis, 'xhat_taxis_piu_sigma': xhat_taxis_piu_sigma, 'xhat_taxis_meno_sigma': xhat_taxis_meno_sigma,'param_axis':param_axis,'X_test_tdc':X_test_tdc[idx_test],'out_axis':out_axis,'out_no_noise_axis':out_no_noise_axis,'obs_taxis':obs_taxis})
else:
    savemat(save_mat_data_ID, {'t_axis': t_axis, 'xhat_taxis': xhat_taxis, 'xhat_taxis_piu_sigma': xhat_taxis_piu_sigma, 'xhat_taxis_meno_sigma': xhat_taxis_meno_sigma,'param_axis':param_axis,'X_test_tdc':X_test_tdc[idx_test],'out_axis':out_axis,'obs_taxis':obs_taxis})

# %%
