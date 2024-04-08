#%%
import matplotlib.pyplot as plt
import numpy as np
import pysindy as ps
import cloudpickle
from joblib import dump

#%%
# load observation data ################################################################################################################
def read_data(data_root_ID,ID_sensor,seq_len,n_channels):
 
    for i1 in range(n_channels):
        i0 = i1+1
        data_path  = data_root_ID + '\\' + ID_sensor + 'dof_' + str(i0) + '.csv'
 
        print("Sensor {:s} to be loaded: {:d}".format(ID_sensor,i0))
        X_single_dof = np.genfromtxt(data_path)
        print("Loaded sensor: {:s} {:d}".format(ID_sensor,i0))
 
        X_single_dof.astype(np.float32)
 
        if i1 == 0:
            n_instances = len(X_single_dof) / seq_len
            n_instances = int(n_instances)
            X = np.zeros((n_instances, seq_len, n_channels))
 
        i4 = 0
        for i3 in range(n_instances):
            X_single_label = X_single_dof[i4 : (i4 + seq_len)]
            X[i3, 0 : (seq_len), i1] = X_single_label
            i4 = i4 + seq_len
    # Return
    return X, n_instances

#%%
# standardize data #####################################################################################################################
def standardize_data(path_save,data,data_string,i1):
    mean_data = [np.mean(data)]
    std_data  = [np.std(data)]
    stand_data   = (data - mean_data) / std_data
    np.savetxt(path_save + '\\mean_' + data_string + '_' + str(i1) + '.csv',mean_data,delimiter=',')
    np.savetxt(path_save + '\\std_' + data_string + '_' + str(i1) + '.csv',mean_data,delimiter=',')
    return mean_data,std_data,stand_data

#%%
# concatenate data #####################################################################################################################
def concatenate_data(data_o,F_data,theta,N_l,N_i,N_o,N_theta,N_f):
    data = np.zeros((N_i,N_l,N_o+N_theta+N_f))
    for i1 in range(N_o):
        data[:,:,i1] = data_o[:,:,i1]
    for i1 in range(N_theta):
        data[:,:,N_o+i1] = theta[:,:,i1]
    for i1 in range(N_f):
        data[:,:,N_o+N_theta+i1] = F_data[:,:,i1]
    return data

# concatenate input ###################################################################################################################
def concatenate_input(input,new_input,che):
    if che:
        input = np.concatenate((input,new_input),axis=2)
    else:
        input = new_input
    che = 1
    return che,input
#%%
root_ID = r"D:\Luca\Dati\DEKF"
problem_ID = r"telaio_2DOF"
data_ID = r"3_14DEKF"
save_ID = r"3_20DEKF"

data_root_ID = root_ID + '\\' + problem_ID + '\\' + data_ID
save_root_ID = root_ID + '\\' + problem_ID + '\\' + save_ID

load_displ = 1
load_veloc = 1
load_accel = 1
load_load  = 1   
 
N_ou = 2
N_ov = 2
N_oa = 2

# si preferisce mantenere la scrittura più generale ####################################################################################
H_vibr_data_only = 1 # si potrebbe scrivere un altro modello sindy per trovare la relazione tra le variabili cinematiche del problema e
                     # altri dati misurati (es. valori di deformazione letti tramite strain gauges)
# Tuttavia, dato che Sindy identifica il sistema dinamico a partire dai dati,
# sarà semmpre verificata (o ci si potrà ridurre al)la condizione N_o = N_obs
N_obs_u = 2
N_obs_v = 2
N_obs_a = 2
N_obs = N_obs_u + N_obs_v + N_obs_a
# displacement observations
Sd = np.array([[1,0],[0,1]])   # Sd [N_obs_u x N_ou]
# velocity observations
Sv = np.array([[1,0],[0,1]])   # Sv [N_obs_v x N_ov]
# acceleration observations
Sa = np.array([[1,0],[0,1]])   # Sa [N_obs_a x N_oa]
#Sa = np.array([[0,0],[0,0]])   # Sa [N_obs_a x N_oa]
# #####################################################################################################################################

N_l = 100 #lunghezza serie
N_i = 5000 #numero istanze
N_o = N_ou*load_displ+N_ov*load_veloc+N_oa*load_accel  #numero osservazioni
N_f = 2   #numero termini forza
N_theta = 1  #numero parametri

# load data ###########################################################################################################################
che = 0; data = 0
if load_displ: #displacement
    U_data, _ = read_data(data_root_ID,'_U_concat_',N_l,N_ou)
    for i1 in range(N_ou):
        U_data_i1 = U_data[:,:,i1]
        #mean_U,std_U,U_data_i1=standardize_data(data_root_ID,U_data_i1,'U',i1)
        U_data[:,:,i1] = U_data_i1
    data = U_data; che = 1
if load_veloc: #velocity
    V_data, _ = read_data(data_root_ID,'_V_concat_',N_l,N_ov)
    for i1 in range(N_ov):
        V_data_i1 = V_data[:,:,i1]
        #mean_V,std_V,V_data_i1=standardize_data(data_root_ID,V_data_i1,'V',i1)
        V_data[:,:,i1] = V_data_i1
    che,data = concatenate_input(data,V_data,che)  
if load_accel: #acceleration
    A_data, _ = read_data(data_root_ID,'_A_concat_',N_l,N_oa)
    for i1 in range(N_oa):
        A_data_i1 = A_data[:,:,i1]
        #mean_A,std_A,A_data_i1=standardize_data(data_root_ID,A_data_i1,'A',i1)
        A_data[:,:,i1] = A_data_i1
    che,data = concatenate_input(data,A_data,che)
theta, _ = read_data(data_root_ID,'theta_',N_l,N_theta)
for i1 in range(N_theta):
    theta_i1 = theta[:,:,i1]
    #mean_theta,std_theta,theta_i1=standardize_data(data_root_ID,theta_i1,'theta',i1)
    theta[:,:,i1] = theta_i1
if load_load:  #force
    F_data, _ = read_data(data_root_ID,'F_',N_l,N_f)
    for i1 in range(N_f):
        F_data_i1 = F_data[:,:,i1]
        #mean_F,std_F,F_data_i1=standardize_data(data_root_ID,F_data_i1,'F',i1)
        F_data[:,:,i1] = F_data_i1
data = concatenate_data(data,F_data,theta,N_l,N_i,N_o,N_theta,N_f)
# end load data #######################################################################################################################

#%% SINDy

#TODO: Find an appropriate scaling for the data and forcing/parameter terms 
scale_V = np.max(np.abs(V_data))
scale_F = np.max(np.abs(F_data))
scale_theta = np.max(np.abs(theta))
x = U_data / scale_V
dx = V_data / scale_V
ddx = A_data / scale_V
F = F_data / scale_F
Theta = theta / scale_theta

#%% Scale the displacement
scale_x = np.max(np.abs(x))
x = x / scale_x

#%% Scale the acceleration
scale_ddx = np.max(np.abs(ddx))
ddx = ddx / scale_ddx

#%%
X = []
X_dot = []
U = []
n_ics = 5000
dt = 0.01
n_timesteps = 100

for i in range(n_ics):
    X.append(np.concatenate((x[i], dx[i]), axis = 1))
    X_dot.append(np.concatenate((dx[i]/scale_x,ddx[i]*scale_ddx), axis = 1))
    U.append(np.concatenate((Theta[i], F[i,:,0:1],F[i,:,1:2]), axis = 1))

#TODO bisognerebbe forzare i coefficienti di F1 e F2 ad essere uguali

feature_names = ["x1", "x2", "x1'", "x2'", "w0", "F1", "F2"]
sindy_library = [lambda x: x, 
                 lambda x1, x2, x1_dot, x2_dot, w0, F1, F2: x1*x1,
                 lambda x1, x2, x1_dot, x2_dot, w0, F1, F2: x1*x2,
                 lambda x1, x2, x1_dot, x2_dot, w0, F1, F2: x1*x1_dot,
                 lambda x1, x2, x1_dot, x2_dot, w0, F1, F2: x1*x2_dot,
                 lambda x1, x2, x1_dot, x2_dot, w0, F1, F2: x1*w0,

                 lambda x1, x2, x1_dot, x2_dot, w0, F1, F2: x2*x2,
                 lambda x1, x2, x1_dot, x2_dot, w0, F1, F2: x2*x1_dot,
                 lambda x1, x2, x1_dot, x2_dot, w0, F1, F2: x2*x2_dot,
                 lambda x1, x2, x1_dot, x2_dot, w0, F1, F2: x2*w0,

                 lambda x1, x2, x1_dot, x2_dot, w0, F1, F2: x1_dot*x1_dot,
                 lambda x1, x2, x1_dot, x2_dot, w0, F1, F2: x1_dot*x2_dot,
                 lambda x1, x2, x1_dot, x2_dot, w0, F1, F2: x1_dot*w0,

                 lambda x1, x2, x1_dot, x2_dot, w0, F1, F2: x2_dot*x2_dot,
                 lambda x1, x2, x1_dot, x2_dot, w0, F1, F2: x2_dot*w0,

                 lambda x1, x2, x1_dot, x2_dot, w0, F1, F2: w0*w0]

sindy_library_names = [
    "x1", "x2", "x1'", "x2'", "w0", "F1", "F2",
    "x1^2", "x1*x2", "x1*x1'", "x1*x2'", "x1*w0",
    "x2^2", "x2*x1'", "x2*x2'", "x2*w0",
    "x1'^2", "x1'*x2'", "x1'*w0",
    "x2'^2", "x2'*w0",
    "w0^2"
]
model = ps.SINDy(feature_names = sindy_library_names, feature_library= ps.CustomLibrary(library_functions = sindy_library), optimizer=ps.STLSQ(threshold=1e-2))
model.fit(X, t=dt, u = U, multiple_trajectories=True, x_dot = X_dot)
coeffs = model.coefficients()

model.print(precision = 3)
#%%
def print_model(library_names, coeffs, threshold = 1e-2, feature_names = ["x1", "x2", "x1'", "x2'", "w0", "F1", "F2"], precision = 3):
    for i in range(coeffs.shape[0]):
        stringa = feature_names[i] + "' = "
        for j in range(coeffs.shape[1]):
            if np.abs(coeffs[i,j]) > threshold:
                stringa += str(round(coeffs[i,j], precision)) + ' ' + library_names[j] + " + "
        print(stringa[:-2])

#%%
print_model(sindy_library_names, coeffs, threshold = 1e-4, feature_names = ["x1", "x2", "x1'", "x2'", "w0", "F1", "F2"])

#%% Predict
times_test = np.arange(0, n_timesteps*dt, dt)
n_test = 2
x0 = np.concatenate((x[n_test], dx[n_test]), axis = 1)[0]
pred = model.simulate(x0, t = times_test, u = U[n_test][:n_timesteps][:])


#%%
#Plot solution
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(times_test[:n_timesteps], x[n_test][:n_timesteps,0],'b-',  label = "x1")
plt.plot(times_test[1:n_timesteps], pred[:n_timesteps-1,0], 'r--', label = "x1 pred")
plt.legend()

plt.subplot(1,2,2)
plt.plot(times_test[:n_timesteps], x[n_test][:n_timesteps,1], 'b-', label = "x2")
plt.plot(times_test[1:n_timesteps], pred[:n_timesteps-1,1], 'r--', label = "x2 pred")
plt.legend()
plt.show()

#%%
#Plot solution
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(times_test[:n_timesteps], dx[n_test][:n_timesteps,0],'b-',  label = "x1'")
plt.plot(times_test[1:n_timesteps], pred[:n_timesteps-1,2], 'r--', label = "x1' pred")
plt.legend()

plt.subplot(1,2,2)
plt.plot(times_test[:n_timesteps], dx[n_test][:n_timesteps,1], 'b-', label = "x2'")
plt.plot(times_test[1:n_timesteps], pred[:n_timesteps-1,3], 'r--', label = "x2' pred")
plt.legend()
plt.show()


# %%
#B is the matrix corresponding to the columns relative to F1 and F2
B = coeffs[:,5:7]
#A is the matrix corresponding to the columns relative to the states and parameters
A = np.concatenate((coeffs[:,0:5], coeffs[:,7:]), axis = 1)
n_features = A.shape[1] #number of features

def jacobian_A(A, variables):
    '''
    A: matrix of SINDy coefficients for the states and parameters
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
                        0.])
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
    return A @ J_library
#%%
J = jacobian_A(A, variables=[1., 1., 1., 1., 1.])

#%% Save the sindy model to a file
sindy_saveID = save_root_ID + '\\sindy_model_dynamics.pkl'
with open(sindy_saveID, 'wb') as f:
    cloudpickle.dump(model, f)

#%% Determine and save the H operator
if H_vibr_data_only: # si misurano dati vibrazionali

    Hsindy_coeff = np.zeros(shape=(N_obs,n_features))
    Hsindy_f_coeff = np.zeros(shape=(N_obs,N_f))

    # sindy_library_names = [
    #     "x1", "x2", "x1'", "x2'", "w0",
    #     "x1^2", "x1*x2", "x1*x1'", "x1*x2'", "x1*w0",
    #     "x2^2", "x2*x1'", "x2*x2'", "x2*w0",
    #     "x1'^2", "x1'*x2'", "x1'*w0",
    #     "x2'^2", "x2'*w0",
    #     "w0^2"
    # ]

    if N_obs_u > 0:
        Hsindy_coeff[:N_obs_u,:N_ou] = Sd                         # u
    if N_obs_v > 0:
        Hsindy_coeff[N_obs_u:N_obs_u+N_obs_v,N_ou:N_ou+N_ov] = Sv #v

    if N_obs_a > 0:
        Hsindy_coeff[-N_obs_a:,:] = np.matmul(Sa, A[-N_oa:,:])    # a
        Hsindy_f_coeff[-N_obs_a:,:] = np.matmul(Sa, B[-N_oa:,:])

    Hsindy_saveID = save_root_ID + '\\sindy_Hmodel_coeffs.npy'
    Hsindy_f_saveID = save_root_ID + '\\sindy_Hmodel_f_coeffs.npy'
    np.save(Hsindy_saveID,Hsindy_coeff)
    np.save(Hsindy_f_saveID,Hsindy_f_coeff)

#else: fitta un altro modello SINDy

# %%
