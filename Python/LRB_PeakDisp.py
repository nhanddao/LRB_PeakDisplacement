# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 08:26:35 2024
@author: nhand
"""

import numpy as np
from pickle import load
from keras.models import load_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
#===============================================
# function to calculate peak displacement per equivalent linear force procedure
# Equation 22 in https://doi.org/10.3311/PPci.19894
# fd= normalized characteristic strength of the isolation system
# Td= period correspondent to the post-yield stiffness
# k1d= K1/Kd= ratio between the initial stiffness and the post-yield stiffness
# S1 (measured in g)= 1s-period spectral acceleration
def ELF(fd, Td, k1d, S1):
    return 0.0767*S1**1.4838*Td**0.5113/fd**0.4929/k1d**0.03006
#===============================================
# function to calculate expected peak displacement for mixed ground motion groups
# fd= normalized characteristic strength of the isolation system
# Td= period correspondent to the post-yield stiffness
# k1d= K1/Kd= ratio between the initial stiffness and the post-yield stiffness
# S3 (measured in g)= 3s-period spectral acceleration
def D50_MixedGM(fd, Td, k1d, S3):
    return 0.55*S3**1.216*Td**0.491/fd**0.363/k1d**0.1417
#===============================================
# function to calculate 90th percentile peak displacement for mixed ground motion groups
# fd= normalized characteristic strength of the isolation system
# Td= period correspondent to the post-yield stiffness
# k1d= K1/Kd= ratio between the initial stiffness and the post-yield stiffness
# S3 (measured in g)= 3s-period spectral acceleration
def D90_MixedGM(fd, Td, k1d, S3):
    return 0.617*S3**1.152*Td**0.465/fd**0.344/k1d**0.1341
#===============================================
# function to calculate 95th percentile peak displacement for mixed ground motion groups
# fd= normalized characteristic strength of the isolation system
# Td= period correspondent to the post-yield stiffness
# k1d= K1/Kd= ratio between the initial stiffness and the post-yield stiffness
# S3 (measured in g)= 3s-period spectral acceleration
def D95_MixedGM(fd, Td, k1d, S3):
    return 0.644*S3**1.125*Td**0.454/fd**0.336/k1d**0.131
#===============================================
# function to calculate 50th percentile peak displacement for pulse-like ground motion groups
# fd= normalized characteristic strength of the isolation system
# Td= period correspondent to the post-yield stiffness
# k1d= K1/Kd= ratio between the initial stiffness and the post-yield stiffness
# S3 (measured in g)= 3s-period spectral acceleration
def D50_PulseLikeGM(fd, Td, k1d, S3):
    return 0.501*S3**1.309*Td**0.569/fd**0.419/k1d**0.1312
#===============================================
# function to calculate 90th percentile peak displacement for pulse-like ground motion groups
# fd= normalized characteristic strength of the isolation system
# Td= period correspondent to the post-yield stiffness
# k1d= K1/Kd= ratio between the initial stiffness and the post-yield stiffness
# S3 (measured in g)= 3s-period spectral acceleration
def D90_PulseLikeGM(fd, Td, k1d, S3):
    return 0.541*S3**1.2404*Td**0.539/fd**0.397/k1d**0.1243
#===============================================
# function to calculate 95th percentile peak displacement for pulse-like ground motion groups
# fd= normalized characteristic strength of the isolation system
# Td= period correspondent to the post-yield stiffness
# k1d= K1/Kd= ratio between the initial stiffness and the post-yield stiffness
# S3 (measured in g)= 3s-period spectral acceleration
def D95_PulseLikeGM(fd, Td, k1d, S3):
    return 0.546*S3**1.186*Td**0.515/fd**0.38/k1d**0.1188
#===============================================
# function to calculate 50th percentile peak displacement for no-pulse ground motion groups
# fd= normalized characteristic strength of the isolation system
# Td= period correspondent to the post-yield stiffness
# k1d= K1/Kd= ratio between the initial stiffness and the post-yield stiffness
# S3 (measured in g)= 3s-period spectral acceleration
def D50_NoPulseGM(fd, Td, k1d, S3):
    return 0.511*S3**1.101*Td**0.486/fd**0.326/k1d**0.1393
#===============================================
# function to calculate 90th percentile peak displacement for no-pulse ground motion groups
# fd= normalized characteristic strength of the isolation system
# Td= period correspondent to the post-yield stiffness
# k1d= K1/Kd= ratio between the initial stiffness and the post-yield stiffness
# S3 (measured in g)= 3s-period spectral acceleration
def D90_NoPulseGM(fd, Td, k1d, S3):
    return 0.594*S3**1.084*Td**0.479/fd**0.321/k1d**0.1372
#===============================================
# function to calculate 95th percentile peak displacement for no-pulse ground motion groups
# fd= normalized characteristic strength of the isolation system
# Td= period correspondent to the post-yield stiffness
# k1d= K1/Kd= ratio between the initial stiffness and the post-yield stiffness
# S3 (measured in g)= 3s-period spectral acceleration
def D95_NoPulseGM(fd, Td, k1d, S3):
    return 0.611*S3**1.0447*Td**0.461/fd**0.309/k1d**0.1323
#===============================================
# function to calculate 50th percentile peak displacement for any ground motion type
# fd= normalized characteristic strength of the isolation system
# Td= period correspondent to the post-yield stiffness
# k1d= K1/Kd= ratio between the initial stiffness and the post-yield stiffness
# S3 (measured in g)= 3s-period spectral acceleration
def D50(fd, Td, k1d, S3):
    return 0.56*S3**1.2701*Td**0.522/fd**0.367/k1d**0.1318
#===============================================
# function to calculate 90th percentile peak displacement for any ground motion type
# fd= normalized characteristic strength of the isolation system
# Td= period correspondent to the post-yield stiffness
# k1d= K1/Kd= ratio between the initial stiffness and the post-yield stiffness
# S3 (measured in g)= 3s-period spectral acceleration
def D90(fd, Td, k1d, S3):
    return 0.622*S3**1.1891*Td**0.489/fd**0.344/k1d**0.1234
#===============================================
# function to calculate 95th percentile peak displacement for any ground motion type
# fd= normalized characteristic strength of the isolation system
# Td= period correspondent to the post-yield stiffness
# k1d= K1/Kd= ratio between the initial stiffness and the post-yield stiffness
# S3 (measured in g)= 3s-period spectral acceleration
def D95(fd, Td, k1d, S3):
    return 0.64*S3**1.153*Td**0.474/fd**0.333/k1d**0.1196
#===============================================
# function to predict peak displacement using ANN model trained from the mixed ground motion group
# The below 3 files must be put in the same folder to this file:
#   1. S1To5_Mixed_InputScaler.pkl
#   2. S1To5_Mixed_OutputScaler.pkl
#   3. S1To5_Mixed_ANN_Model.keras
# fd= normalized characteristic strength of the isolation system
# Td= period correspondent to the post-yield stiffness
# k1d= K1/Kd= ratio between the initial stiffness and the post-yield stiffness
# S1To5 (measured in g)= spectral acceleration at 1s- to 5s periods, step= 0.5 s
# return expected peak displacement DM, which is a scalar
# usage example:
# DM = ANN_MixedGM(0.02, 2.0, 5.0, np.array([9.135, 6.132, 4.4718, 3.504, 2.6858, 2.2533, 1.7622, 1.6084, 1.4265])/9.81)
def ANN_MixedGM(fd, Td, k1d, S1To5):
    # load scaler
    input_scaler= load(open('S1To5_Mixed_InputScaler.pkl', 'rb'))
    output_scaler= load(open('S1To5_Mixed_OutputScaler.pkl', 'rb'))
    # create input vector
    X= np.concatenate((np.array([fd]), np.array([Td]), np.array([k1d]), np.ravel(S1To5)),axis=0)
    # scale inputs
    X_transf= input_scaler.transform([X])
    # load ANN model
    model= load_model('S1To5_Mixed_ANN_Model.keras')
    # predict
    Y_transf= model.predict(X_transf, verbose=None)
    # transform the prediction
    Y= output_scaler.inverse_transform(Y_transf)
    return Y[0][0]
#===============================================
# function to predict peak displacement using ANN model trained from the data of pulse-like ground motion group
# The below 3 files must be put in the same folder to this file:
#   1. S1To5_PulseLike_InputScaler.pkl
#   2. S1To5_PulseLike_OutputScaler.pkl
#   3. S1To5_PulseLike_ANN_Model.keras
# fd= normalized characteristic strength of the isolation system
# Td= period correspondent to the post-yield stiffness
# k1d= K1/Kd= ratio between the initial stiffness and the post-yield stiffness
# S1To5 (measured in g)= spectral acceleration at 1s- to 5s periods, step= 0.5 s
# return expected peak displacement DM, which is a scalar
# usage example:
# DM = ANN_PulseLikeGM(0.02, 2.0, 5.0, np.array([9.135, 6.132, 4.4718, 3.504, 2.6858, 2.2533, 1.7622, 1.6084, 1.4265])/9.81)
def ANN_PulseLikeGM(fd, Td, k1d, S1To5):
    # load scaler
    input_scaler= load(open('S1To5_PulseLike_InputScaler.pkl', 'rb'))
    output_scaler= load(open('S1To5_PulseLike_OutputScaler.pkl', 'rb'))
    # create input vector
    X= np.concatenate((np.array([fd]), np.array([Td]), np.array([k1d]), np.ravel(S1To5)),axis=0)
    # scale inputs
    X_transf= input_scaler.transform([X])
    # load ANN model
    model= load_model('S1To5_PulseLike_ANN_Model.keras')
    # predict
    Y_transf= model.predict(X_transf, verbose=None)
    # transform the prediction
    Y= output_scaler.inverse_transform(Y_transf)
    return Y[0][0]
#===============================================
# function to predict peak displacement using ANN model trained from the data of no-pulse ground motion group
# The below 3 files must be put in the same folder to this file:
#   1. S1To5_NoPulse_InputScaler.pkl
#   2. S1To5_NoPulse_OutputScaler.pkl
#   3. S1To5_NoPulse_ANN_Model.keras
# fd= normalized characteristic strength of the isolation system
# Td= period correspondent to the post-yield stiffness
# k1d= K1/Kd= ratio between the initial stiffness and the post-yield stiffness
# S1To5 (measured in g)= spectral acceleration at 1s- to 5s periods, step= 0.5 s
# return expected peak displacement DM, which is a scalar
# usage example:
# DM = ANN_NoPulseGM(0.02, 2.0, 5.0, np.array([9.135, 6.132, 4.4718, 3.504, 2.6858, 2.2533, 1.7622, 1.6084, 1.4265])/9.81)
def ANN_NoPulseGM(fd, Td, k1d, S1To5):
    # load scaler
    input_scaler= load(open('S1To5_NoPulse_InputScaler.pkl', 'rb'))
    output_scaler= load(open('S1To5_NoPulse_OutputScaler.pkl', 'rb'))
    # create input vector
    X= np.concatenate((np.array([fd]), np.array([Td]), np.array([k1d]), np.ravel(S1To5)),axis=0)
    # scale inputs
    X_transf= input_scaler.transform([X])
    # load ANN model
    model= load_model('S1To5_NoPulse_ANN_Model.keras')
    # predict
    Y_transf= model.predict(X_transf, verbose=None)
    # transform the prediction
    Y= output_scaler.inverse_transform(Y_transf)
    return Y[0][0]
#===============================================
# function to predict peak displacement using ANN model trained from the data across all ground motion group
# The below 3 files must be put in the same folder to this file:
#   1. S1To5_Combined_InputScaler.pkl
#   2. S1To5_Combined_OutputScaler.pkl
#   3. S1To5_Combined_ANN_Model.keras
# fd= normalized characteristic strength of the isolation system
# Td= period correspondent to the post-yield stiffness
# k1d= K1/Kd= ratio between the initial stiffness and the post-yield stiffness
# S1To5 (measured in g)= spectral acceleration at 1s- to 5s periods, step= 0.5 s
# return expected peak displacement DM, which is a scalar
# usage example:
# DM = ANN_AllGMTypes(0.02, 2.0, 5.0, np.array([9.135, 6.132, 4.4718, 3.504, 2.6858, 2.2533, 1.7622, 1.6084, 1.4265])/9.81)
def ANN_AllGMTypes(fd, Td, k1d, S1To5):
    # load scaler
    input_scaler= load(open('S1To5_Combined_InputScaler.pkl', 'rb'))
    output_scaler= load(open('S1To5_Combined_OutputScaler.pkl', 'rb'))
    # create input vector
    X= np.concatenate((np.array([fd]), np.array([Td]), np.array([k1d]), np.ravel(S1To5)),axis=0)
    # scale inputs
    X_transf= input_scaler.transform([X])
    # load ANN model
    model= load_model('S1To5_Combined_ANN_Model.keras')
    # predict
    Y_transf= model.predict(X_transf, verbose=None)
    # transform the prediction
    Y= output_scaler.inverse_transform(Y_transf)
    return Y[0][0]
#===============================================