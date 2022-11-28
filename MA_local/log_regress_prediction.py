import time
import pandas as pd
import numpy  as np 
# import tensorflow as tf
import os
# from tensorflow import keras
from numpy import array 
# from keras.layers import LSTM,GRU,ConvLSTM2D
# from keras.layers import Dense,Dropout,Flatten,TimeDistributed
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
# from keras.layers.convolutional import Conv1D
# from keras.layers.convolutional import MaxPooling1D 
import json
import random
import pickle




def normalize(x):
    return (x-min(x))/(max(x)-min(x))  



def split_sequences(sequences, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequences)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out-1
        if out_end_ix > len(sequences):
            break
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1:out_end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)

    return array(X), array(y)


def read_data(string, n_steps_in, n_steps_out, n_features, architecture):
    
    t_win = n_steps_in*n_steps_out
    Z = pd.read_csv(string)
    Z=Z.to_numpy()   
    
    if architecture == 'CNN_LSTM':
        X, y = split_sequences(Z, t_win, n_steps_out )
        X = X.reshape((X.shape[0], n_steps_in, n_steps_out, n_features))
    elif architecture == 'ConvLSTM':
        X, y = split_sequences(Z, t_win, n_steps_out )
        X = X.reshape((X.shape[0], n_steps_in, 1, n_steps_out, n_features))
    else:
        X, y = split_sequences(Z, n_steps_in, n_steps_out)   
    
    n_train=int(0.7*len(X))     
    X_train=X[0: n_train,];         y_train = y[0:n_train,]
    X_test =X[n_train: len(X),];    y_test  = y[n_train:len(X),]
    X_train.shape
    X_test.shape 
    X_train[0,]
    
    return X_train,y_train,X_test,y_test 



def read_data_ML(string):
    Z = pd.read_csv(string)
    Z=Z.to_numpy()
     
    Z.shape  
    n_train=int(0.7*len(Z))
    
    X_train=Z[0: n_train,0:-1];         y_train = Z[0:n_train,-1]
    X_test =Z[n_train: len(Z),0:-1];    y_test  = Z[n_train:len(Z),-1]
    X_train.shape
    X_test.shape 
    X_train[0,] 
    
    return X_train,y_train,X_test,y_test 



input_path = "/home/doktormatte/MA_SciComp/ACN_1/Occup/39_shifted_red_header.csv"
X_train,y_train,X_test,y_test = read_data_ML(input_path)


loaded_model = pickle.load(open('/home/doktormatte/Dropbox/Dokumente/Studium/MA_SciComp/log_regress_model.sav', 'rb'))
X_test = X_test[6000:7000]
y_test = y_test[6000:7000]

n_steps_out=36
t_target = n_steps_out

m,n=X_test.shape
yhat=np.zeros([m,t_target])            
y_obs=np.zeros([m,t_target])
for kk in range(m-t_target) :
    y_obs[kk,:]=y_test[kk:kk+t_target]    
    
out = np.zeros((3,1))
    
scores1= np.zeros(m,float)
scores_F1= np.zeros([m,3],float)            
n_sample=m-n_steps_out     
for i in range(n_sample): 
    X_test_temp=X_test.copy();  
    X_test_temp=np.append(X_test_temp,[[0]*13,[0]*13,[0]*13,],0)
    for j in range(n_steps_out):   
        temp11=X_test_temp[i+j,:].reshape(1, -1)                    
        yhat[i,j] = loaded_model.predict(temp11) 
        #if i+j+3<m:
        rng1=[ i+j+3 -_ii for _ii in range(0, 3) ]
        rng2=[ _jj for _jj in range(-3,0) ] # only t-3,-2,-1 are considered
        X_test_temp[rng1,rng2]=yhat[i,j]  
        out = np.append(out, X_test_temp[rng1,rng2])
        # print(X_test_temp[rng1,rng2])





