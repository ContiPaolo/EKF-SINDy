import os
import sklearn
import numpy as np
import pandas as pd
import matplotlib
import filterpy
import scipy

from tensorflow.python.ops.gen_math_ops import mean
from matplotlib import pyplot as plt

# load observation data ################################################################################################################
def read_data(data_root_ID,ID_sensor,seq_len,n_channels, train_test):

    for i1 in range(n_channels):
        i0 = i1+1
        data_path  = data_root_ID + '\\' + ID_sensor + 'dof_' + str(i0) + '_' + train_test + '.csv'

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


# standardize data #####################################################################################################################
def standardize_data(path_save,data,data_string,i1):
    mean_data = [np.mean(data)]
    std_data  = [np.std(data)]
    stand_data   = (data - mean_data) / std_data
    np.savetxt(path_save + '\\mean_' + data_string + '_' + str(i1) + '.csv',mean_data,delimiter=',')
    np.savetxt(path_save + '\\std_' + data_string + '_' + str(i1) + '.csv',mean_data,delimiter=',')
    return mean_data,std_data,stand_data

# standardize data using given mean and std ############################################################################################
def standardize_data_given_mean_std(data_train_string,data,data_string):    
    mean_string = data_train_string + 'mean_' + data_string + '.csv'
    std_string = data_train_string + 'std_' + data_string + '.csv'
    mean_data = np.genfromtxt(mean_string)
    std_data = np.genfromtxt(std_string)
    stand_data = (data - mean_data) / std_data
    return stand_data
    

# concatenate input ###################################################################################################################
def concatenate_input(input,new_input,che):
    if che:
        input = np.concatenate((input,new_input),axis=2)
    else:
        input = new_input
    che = 1
    return che,input

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

# data for alpha ######################################################################################################################
def data_for_alpha(data,N_theta,N_f):
    N_b = data.shape[0]
    N_l = data.shape[1]
    N_c = data.shape[2]
    input_data = np.zeros((N_b*(N_l-1),N_c))
    output_data = np.zeros((N_b*(N_l-1),N_c-N_theta-N_f))
    for i0 in range(N_b):
        for i1 in range(N_l-1):
            input_data[i0*(N_l-1)+i1,:] = data[i0,i1,:]
            output_data[i0*(N_l-1)+i1,:] = data[i0,i1+1,:-N_theta-N_f]
    return input_data,output_data

# shape data for LSTM ##################################################################################################################
def shapeLSTM(data,N_theta,N_f):
    N_b = data.shape[0]
    N_l = data.shape[1]
    N_c = data.shape[2]
    input_data  = np.zeros((N_b,N_l-1,N_c))
    output_data = np.zeros((N_b,N_l-1,N_c-N_theta-N_f))
    for i0 in range(N_b):
        input_data[i0,:,:]  = data[i0,:-1,:]
        output_data[i0,:,:] = data[i0,1:,:-N_theta-N_f]
    return input_data,output_data

# list for postprocessing ####################################################################################
def list_for_postprocessing(load_if,obs_labels,N_oo,label,i0):
    if load_if:
        for i1 in range(N_oo):
            obs_labels[i0] = label + ' ' + str(i1)
            i0 += 1
    return i0,obs_labels

# plot reconstructed signal function (postprocessing) ########################################################
def plot_reconstruction_multi_filter(t_axis,state,stateP,obs,obs_labels,N_o,N_theta,path_save):

    for n_o in np.arange(N_o+N_theta):

        state_low = state[n_o,:] - 1.96 *  np.sqrt(stateP[n_o,n_o,:])
        state_up  = state[n_o,:] + 1.96 *  np.sqrt(stateP[n_o,n_o,:])

        matplotlib.style.use('classic')
        matplotlib.rc('font',  size=16, family='serif')
        matplotlib.rc('axes',  titlesize=16)
        matplotlib.rc('text',  usetex=True)
        matplotlib.rc('lines', linewidth=1)
        plt.plot(t_axis,obs[:,n_o],'m',t_axis,state[n_o,:],'k',t_axis,state_low,'k-.',t_axis,state_up,'k-.')
        
        plt.xlabel(r'time [s]', fontsize=24)
        plt.ylabel(obs_labels[n_o], fontsize=24)
        plt.legend([r'observation',r'state'],loc='upper right',fontsize=24)
        fig_save_1  = path_save + '\\obs_state_' + obs_labels[n_o]  + '.png'
        fig_save_2  = path_save + '\\obs_state_' + obs_labels[n_o]  + '.pdf'
        plt.savefig(fig_save_1, bbox_inches='tight')
        plt.savefig(fig_save_2, bbox_inches='tight')
        plt.close()


def add_noise_with_snr(signal, snr):
    # the signal-to-noise ratio is here defined as the ratio between the mean square of the signal
    # and the mean square of the noise component
    
    # SNR is not expressed in decibel
    # P_signal = np.mean(np.square(signal))
    # signal_dev = P_signal / snr

    # SNR in decibel
    P_signal = np.mean(np.square(signal))
    P_signal_dB = 10*np.log10(P_signal)
    P_noise_dB  = P_signal_dB-snr
    signal_dev  = 10 ** (P_noise_dB/10)

    noise = np.random.normal(0, np.sqrt(signal_dev), signal.shape)
    noisy_signal = signal + noise

    # plt.plot(noisy_signal[0,:],'orange')
    # plt.plot(signal[0,:],'black')

    return noisy_signal

# load observation data ################################################################################################################
def read_data_no_noise(data_root_ID,ID_sensor,seq_len,n_channels):

    for i1 in range(n_channels):
        i0 = i1+1
        data_path  = data_root_ID + '\\' + ID_sensor + 'dof_no_noise_' + str(i0) + '.csv'

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


# Create the jacobian matrix for the dynamic equation ##############################################################################
def jacobian_A(A, variables, N_param = 1):
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

    for i3 in range(N_param):
        F = np.append(F, [np.zeros(n+1)], axis=0)
        F[n+i3,n_var-N_param+i3] = 0

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

# Create the jacobian matrix for the observation equation ##############################################################################
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

def ass_xhat(inp, idx_test, i0,N_param, N_ou, N_ov, N_f, param_diff):
    xhat=np.zeros(N_ou+N_ov+N_param)
    xhat[0:N_ou+N_ov]=inp[i0,0:N_ou+N_ov]
    for i1 in np.arange(N_param):
        if inp[i0,-N_param-N_f+i1] > 0:
            xhat[-N_param+i1]=inp[i0,-N_param-N_f+i1]*(1-param_diff)
        else:
            xhat[-N_param+i1]=inp[i0,-N_param-N_f+i1]*(1+param_diff)
    return xhat

def ass_inputs_Euler_forward(i0,F_taxis, N_f):
    inp_f = F_taxis[-N_f:,i0]
    inp_f_1 = F_taxis[-N_f:,i0-1]

    return inp_f, inp_f_1

def output_obs(inp,i0,N_obs_u,N_obs_v,N_obs_a,N_obs_u_inp,N_obs_v_inp,N_obs_a_inp):
    out = np.zeros((N_obs_u+N_obs_v+N_obs_a))
    out[:N_obs_u] = inp[i0,:N_obs_u]
    out[N_obs_u:N_obs_u+N_obs_v] = inp[i0,N_obs_u_inp:N_obs_u_inp+N_obs_v]
    out[N_obs_u+N_obs_v:N_obs_u+N_obs_v+N_obs_a] = inp[i0,N_obs_u_inp+N_obs_v_inp:N_obs_u_inp+N_obs_v_inp+N_obs_a]
    return out

'''# inizialisation of the state vector
def ass_xhat(inp,i0):
    xhat=np.zeros(N_ou+N_ov+N_param)
    xhat[0:N_ou+N_ov]=inp[i0,0:N_ou+N_ov]
    for i1 in np.arange(N_param):
        if inp[i0,-N_param-N_f+i1] > 0:
            xhat[-N_param+i1]=inp[i0,-N_param-N_f+i1]*(1-param_rel_error)
        else:
            xhat[-N_param+i1]=inp[i0,-N_param-N_f+i1]*(1+param_rel_error)
    return xhat
xhat = ass_xhat(inp,0)'''