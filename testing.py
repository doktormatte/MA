import math
import time
import pandas as pd
import numpy  as np 
# from matplotlib import pyplot
# from tensorflow import keras
from numpy import array 

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


def read_data(string,n_steps_in,n_steps_out,n_features):
    
    Z = pd.read_csv(string)
    Z = Z.to_numpy()   
    
    X, y = split_sequences(Z, n_steps_in, n_steps_out) 
    
    n_train=int(0.7*len(X))     
    X_train=X[0: n_train,];         y_train = y[0:n_train,]
    X_test =X[n_train: len(X),];    y_test  = y[n_train:len(X),]
    X_train.shape
    X_test.shape 
    X_train[0,]
    
    return X_train,y_train,X_test,y_test 


file_name = "/home/doktormatte/MA_SciComp/Boulder/try.csv"

X_train,y_train,X_test,y_test = read_data(file_name,3,3,100)