#%%
######################       LIBRARIES       ######################
import pysindy as ps
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from scipy.integrate import odeint
from scipy.linalg import hankel
from sklearn.utils.extmath import randomized_svd
from sklearn.model_selection import train_test_split

seed = 29

#%%
######################       PROBLEM SETUP       ######################
'''
We consider two coupled oscillators with the following equations of motion:
u1'' + ω1^2 u1 + 2ξω1 u1' + α u2 = 0,
u2'' + ω2^2 u2 + 2ξω2 u2' + α u1 + σ x1^2 + γ x2^3 = 0,

which can be rewritten at first order as:
u1' = v1,
v1' = - ω1^2 u1 - 2ξω1 v1 - α u2,
u2' = v2,
v2' = - ω2^2 u2 - 2ξω2 v2 - α u1 - σ x1^2 - γ x2^3.

We suppose we are only able to measure the position of the first oscillator x1, 
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
    'sigma' : 1e-3, #quadratic
    'alpha' : -1e-2, #linear

    #Nonlinearities:
    'gamma' : 1e-4, #cubic

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
u1_0 = np.random.normal(3.,1,n_ics)
u2_0 = np.random.normal(-2.,1,n_ics)

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
    plt.figure(figsize=(7,3))
    plt.plot(t,u1, label = '$u_1$')
    plt.plot(t,u2, label = '$u_2$')
    plt.legend(fontsize = 12)
    plt.xlabel('t', fontsize = 12)

    plt.show()

X = np.array(X)
X_train, X_test, w2_train, w2_test = train_test_split(X, w2, test_size=0.2, random_state=seed)
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
length, shift = 300, 5

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
plt.figure(figsize = [13,16])
for i in range(4):
    plt.subplot(421 + 2*i)
    plt.plot(X_train[0,:Nt_h,i], 'b')
    plt.title(f'Original variable {i+1}', fontsize = 12)

    plt.subplot(421 + 2*i+1)
    plt.plot(u_h[i*Nt_h:(i+1)*Nt_h,i], 'r')
    plt.title(f'Time-delay coordinate {i+1}', fontsize = 12)

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

    X_test_tdc.append(u_h[:,:n_tdc])
    dX_test_tdc.append(np.gradient(u_h[:,:n_tdc], dt, axis = 0))
    param_test_tdc.append(np.repeat(w2_test[i], Nt)[:Nt_h])


# %%
######################      CREATE SINDy model       ######################
model = ps.SINDy(feature_names  = feature_names, feature_library= ps.PolynomialLibrary(degree = 3), optimizer=ps.STLSQ(threshold=5e-4))
model.fit(X_tdc, t=dt, multiple_trajectories=True, u = param_tdc, x_dot = dX_tdc) 

model.print()
# %%
######################      PREDICT       ######################
idx_test = 0
idx_test_fake = 1
fake_param = param_test_tdc[idx_test_fake][0]
print('Real parameter:', param_test_tdc[idx_test][0], 'Fake parameter:', fake_param)

#Predict the evolution of the time-delay coordinates:
pred = model.simulate(X_test_tdc[idx_test][0,:], t= t[:Nt_h], u = param_test_tdc[idx_test_fake])

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

# %%
######################      PREDICT       ######################
idx_train = 0
idx_train_fake = 0
fake_param = param_tdc[idx_train][0]
print('Real parameter:', param_tdc[idx_train][0], 'Fake parameter:', fake_param)

#Predict the evolution of the time-delay coordinates:
pred = model.simulate(X_tdc[idx_train][0,:], t= t[:Nt_h], u = param_tdc[idx_train_fake])

#Plot the evolution of the time-delay coordinates:
plt.figure(figsize = [8,16])
for i in range(4):
    plt.subplot(411 +i)
    plt.plot(X_tdc[idx_train][:,i], 'b', label = 'true')
    plt.plot(pred[:,i], 'r--', label = 'pred')
    plt.title(f'Time-delay coordinate {i+1}', fontsize = 12)
    plt.legend(fontsize = 12)

#%%
######################      RECONSTRUCTION       ######################
#Reconstruct observed variable:
x_rec = pred @ (U * S).T

#Plot the reconstructed variable:
plt.figure(figsize = [8,4])
plt.plot(X_train[idx_train,:,0], 'b', label = 'true')
plt.plot(x_rec[:,0], 'r--', label = 'pred')
plt.title(f'Reconstructed variable', fontsize = 12)
plt.legend(fontsize = 12)

print('MSE', mean_squared_error(X_test[idx_test,:Nt_h-1,0], x_rec[:Nt_h,0]))

# %%
