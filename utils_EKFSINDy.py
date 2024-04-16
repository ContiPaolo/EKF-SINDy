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

   

    if n_var == 6:
        x1, x2, x3, x4, al, be = variables
        #dA/dx1
        J_library[:,0] = np.array([
                            1., 0., 0., 0., 0., 0.,
                                
                            2.*x1, x2, x3, x4, al, be,
                            0., 0., 0., 0., 0.,
                            0., 0., 0., 0.,
                            0., 0., 0.,
                            0., 0.,
                            0.,

                            3.*x1*x1,2.*x1*x2,2.*x1*x3,2.*x1*x4,2.*x1*al,2.*x1*be,
                            x2*x2, x2*x3, x2*x4, x2*al, x2*be,
                            x3*x3, x3*x4, x3*al, x3*be,
                            x4*x4, x4*al, x4*be,
                            al*al, al*be,
                            be*be,
                            0., 0., 0., 0., 0.,
                            0., 0., 0., 0.,
                            0., 0., 0.,
                            0., 0.,
                            0.,
                            0., 0., 0., 0.,
                            0., 0., 0.,
                            0., 0.,
                            0.,
                            0., 0., 0.,
                            0., 0.,
                            0.,
                            0., 0.,
                            0.,
                            0.])

        #dA/dx2
        J_library[:,1] = np.array([
                            0., 1., 0., 0., 0., 0.,

                            0., x1, 0., 0., 0., 0.,
                            2.*x2, x3, x4, al, be,
                            0., 0., 0., 0.,
                            0., 0., 0.,
                            0., 0.,
                            0.,
                        
                            0., x1*x1, 0., 0., 0., 0.,
                            2*x1*x2, x1*x3, x1*x4, x1*al, x1*be,
                            0., 0., 0., 0.,
                            0., 0., 0.,
                            0., 0.,
                            0.,
                            3.*x2*x2, 2.*x2*x3, 2.*x2*x4, 2.*x2*al, 2.*x2*be,
                            x3*x3, x3*x4, x3*al, x3*be,
                            x4*x4, x4*al, x4*be,
                            al*al, al*be,
                            be*be,
                            0., 0., 0., 0.,
                            0., 0., 0.,
                            0., 0.,
                            0.,
                            0., 0., 0.,
                            0., 0.,
                            0., 
                            0., 0.,
                            0.,
                            0.])

        #dA/dx3
        J_library[:,2] = np.array([ 
                            0., 0., 1., 0., 0., 0.,

                            0., 0., x1, 0., 0., 0.,
                            0., x2, 0., 0., 0.,
                            2.*x3, x4, al, be,
                            0., 0., 0.,
                            0., 0.,
                            0.,

                            0., 0., x1*x1, 0., 0., 0.,
                            0., x1*x2, 0., 0., 0.,
                            2*x1*x3, x1*x4, x1*al, x1*be,
                            0., 0., 0.,
                            0., 0.,
                            0.,
                            0., x2*x2, 0., 0., 0.,
                            2*x2*x3, x2*x4, x2*al, x2*be,
                            0., 0., 0.,
                            0., 0.,
                            0.,
                            3.*x3*x3, 2.*x3*x4, 2.*x3*al, 2.*x3*be,
                            x4*x4, x4*al, x4*be,
                            al*al, al*be,
                            be*be,
                            0., 0., 0.,
                            0., 0.,
                            0.,
                            0.,0.,
                            0.,
                            0.])
        
        #dA/dx4
        J_library[:,3] = np.array([ 
                            0., 0., 0., 1., 0., 0.,

                            0., 0., 0., x1, 0., 0.,
                            0., 0., x2, 0., 0.,
                            0., x3, 0., 0.,
                            2.*x4,  al, be,
                            0., 0., 
                            0.,

                            0., 0., 0., x1*x1, 0., 0.,
                            0., 0., x1*x2, 0., 0.,
                            0., x1*x3, 0., 0.,
                            2*x1*x4, x1*al, x1*be,
                            0., 0.,
                            0.,
                            0., 0., x2*x2, 0., 0.,
                            0., x2*x3, 0., 0.,
                            2*x2*x4, x2*al, x2*be,
                            0., 0., 
                            0.,
                            0., x3*x3, 0., 0., 
                            2*x3*x4, x3*al, x3*be,
                            0., 0.,
                            0.,
                            3*x4*x4, 2*x4*al, 2*x4*be,
                            al*al, al*be,
                            be*be,
                            0., 0.,
                            0.,
                            0.])
        
        #dA/dal
        J_library[:,4] = np.array([
                            0., 0., 0., 0., 1., 0.,

                            0., 0., 0., 0., x1, 0.,
                            0., 0., 0., x2, 0.,
                            0., 0., x3, 0.,
                            0., x4, 0.,
                            2.*al, be,
                            0.,

                            0., 0., 0., 0., x1*x1, 0.,
                            0., 0., 0., x1*x2, 0.,
                            0., 0., x1*x3, 0.,
                            0., x1*x4, 0.,
                            2*x1*al, x1*be,
                            0., 
                            0., 0., 0., x2*x2, 0.,
                            0., 0., x2*x3, 0.,
                            0., x2*x4, 0.,
                            2*x2*al, x2*be,
                            0.,
                            0., 0., x3*x3, 0., 
                            0., x3*x4, 0.,
                            2*x3*al, x3*be,
                            0.,
                            0., x4*x4, 0.,
                            2*x4*al, x4*be,
                            0.,
                            3.*al*al, 2.*al*be,
                            0.,
                            0.])
        
        #dA/dbe
        J_library[:,5] = np.array([
                            0., 0., 0., 0., 0., 1.,

                            0., 0., 0., 0., 0., x1,
                            0., 0., 0., 0., x2,
                            0., 0., 0., x3,
                            0., 0., x4,
                            0., al,
                            2.*be,

                            0., 0., 0., 0., 0., x1*x1,
                            0., 0., 0., 0., x1*x2,
                            0., 0., 0., x1*x3,
                            0., 0., x1*x4,
                            0., x1*al,
                            2.*x1*be,
                            0., 0., 0., 0., x2*x2,
                            0., 0., 0., x2*x3,
                            0., 0., x2*x4,
                            0., x2*al,
                            2.*x2*be,
                            0., 0., 0., x3*x3,
                            0., 0., x3*x4,
                            0., x3*al,
                            2.*x3*be,
                            0., 0., x4*x4,
                            0., x4*al,
                            2.*x4*be,
                            0., al*al,
                            2*al*be,
                            3.*be*be])

        F = A @ J_library

        N_param = 2

    elif n_var == 5:
        x1, x2, x3, x4, k = variables
        #dA/dx1
        J_library[:,0] = np.array([
                                    1., 0., 0., 0., 0., 
                                        
                                    2.*x1, x2, x3, x4, k, 
                                    0., 0., 0., 0., 
                                    0., 0., 0., 
                                    0., 0., 
                                    0., 

                                    3.*x1*x1,2.*x1*x2,2.*x1*x3,2.*x1*x4,2.*x1*k,
                                    x2*x2, x2*x3, x2*x4, x2*k,
                                    x3*x3, x3*x4, x3*k, 
                                    x4*x4, x4*k, 
                                    k*k, 
                                    0., 0., 0., 0.,
                                    0., 0., 0., 
                                    0., 0., 
                                    0., 
                                    0., 0., 0.,
                                    0., 0., 
                                    0.,
                                    0., 0., 
                                    0., 
                                    0.])
        #dA/dx2
        J_library[:,1] = np.array([
                                    0., 1., 0., 0., 0., 
                                        
                                    0., x1, 0., 0., 0., 
                                    2.*x2, x3, x4, k, 
                                    0., 0., 0., 
                                    0., 0., 
                                    0., 
                                    
                                    0., x1*x1, 0., 0., 0., 
                                    2*x1*x2, x1*x3, x1*x4, x1*k, 
                                    0., 0., 0., 
                                    0., 0., 
                                    0., 
                                    3.*x2*x2, 2.*x2*x3, 2.*x2*x4, 2.*x2*k, 
                                    x3*x3, x3*x4, x3*k, 
                                    x4*x4, x4*k, 
                                    k*k, 
                                    0., 0., 0., 
                                    0., 0., 
                                    0., 
                                    0., 0., 
                                    0., 
                                    0.])
        
        #dA/dx3
        J_library[:,2] = np.array([
                                    0., 0., 1., 0., 0., 
                                        
                                    0., 0., x1, 0., 0., 
                                    0., x2, 0., 0.,
                                    2.*x3, x4, k, 
                                    0., 0., 
                                    0., 
                                    
                                    0., 0., x1*x1, 0., 0., 
                                    0., x1*x2, 0., 0., 
                                    2*x1*x3, x1*x4, x1*k,
                                    0., 0.,
                                    0.,
                                    0., x2*x2, 0., 0.,
                                    2*x2*x3, x2*x4, x2*k,
                                    0., 0.,
                                    0.,
                                    3.*x3*x3, 2.*x3*x4, 2.*x3*k,
                                    x4*x4, x4*k, 
                                    k*k,
                                    0., 0.,
                                    0.,
                                    0.])
        
        #dA/dx4
        J_library[:,3] = np.array([
                                    0., 0., 0., 1., 0., 
                                        
                                    0., 0., 0., x1, 0., 
                                    0., 0., x2, 0., 
                                    0., x3, 0., 
                                    2.*x4, k, 
                                    0., 
                                    
                                    0., 0., 0., x1*x1, 0.,
                                    0., 0., x1*x2, 0., 
                                    0., x1*x3, 0.,
                                    2*x1*x4, x1*k,
                                    0.,
                                    0., 0., x2*x2, 0.,
                                    0., x2*x3, 0.,
                                    2*x2*x4, x2*k,
                                    0.,
                                    0., x3*x3, 0.,
                                    2*x3*x4, x3*k,
                                    0.,
                                    3*x4*x4, 2*x4*k,
                                    k*k,
                                    0.])
        
        #dA/dk
        J_library[:,4] = np.array([
                                    0., 0., 0., 0., 1., 
                                        
                                    0., 0., 0., 0., x1, 
                                    0., 0., 0., x2,  
                                    0., 0., x3,
                                    0., x4, 
                                    2.*k,
                                    
                                    0., 0., 0., 0., x1*x1, 
                                    0., 0., 0., x1*x2, 
                                    0., 0., x1*x3,
                                    0., x1*x4,
                                    x1*k, 
                                    0., 0., 0., x2*x2, 
                                    0., 0., x2*x3,
                                    0., x2*x4, 
                                    x2*k, 
                                    0., 0., x3*x3, 
                                    0., x3*x4, 
                                    x3*k, 
                                    0., x4*x4, 
                                    x4*k, 
                                    3.*k*k]) 

        F = A @ J_library

        N_param = 1


    for i3 in range(N_param):
        F = np.append(F, [np.zeros(n+N_param)], axis=0)
        F[n+i3,n_var-N_param+i3] = 0
    
    return F


# Create the jacobian matrix for the observation equation ##############################################################################
def jacobian_H(U,S,N_x, n_param = 2):
    '''
    U, S: matrices obtained by operating the svd of the time delay embedding
    xhat_pred @ (U * S).T
    The Jacobian is equivalent to compute (e)4_1.T (U * S).T (e)200_1
                                          (e)4_2.T (U * S).T (e)200_1 etc.
    '''
    #n = U.shape[0] #number of time-delays (=length)
    #n_variables = U.shape[1] #number of time delays embeddings
    H = np.zeros((1,N_x + n_param))
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

