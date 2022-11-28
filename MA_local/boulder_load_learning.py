import math
import time
import pandas as pd
import numpy  as np 
#from matplotlib import pyplot
import matplotlib.pyplot as plt
from tensorflow import keras
from numpy import array 
from keras.models import Sequential 
from keras.layers import LSTM,GRU,ConvLSTM2D
from keras.layers import RepeatVector
from keras.layers import Dense,Dropout,Flatten,TimeDistributed
from keras.layers import BatchNormalization 
from keras.layers import Bidirectional
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D 
from keras.models import Model 
from keras.layers import Input
from tensorflow.keras.layers import concatenate


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
    Z=Z.to_numpy()   
    
    X, y = split_sequences(Z, n_steps_in, n_steps_out) 
    
    n_train=int(0.7*len(X))     
    X_train=X[0: n_train,];         y_train = y[0:n_train,]
    X_test =X[n_train: len(X),];    y_test  = y[n_train:len(X),]
    X_train.shape
    X_test.shape 
    X_train[0,]
    
    return X_train,y_train,X_test,y_test 


#df_boulder = pd.read_csv("/home/doktormatte/MA_SciComp/Boulder/Loads/1.csv")

#X_train,y_train,X_test,y_test = read_data("/home/doktormatte/MA_SciComp/Boulder/Loads/1.csv", 3, 3, 100)
# X_train,y_train,X_test,y_test = read_data("/home/doktormatte/MA_SciComp/Boulder/Loads/2.csv", 3, 3, 4)
accuracies = []
for i in range(1,52):
    

    X_train,y_train,X_test,y_test = read_data("/home/doktormatte/MA_SciComp/ACN/Loads/" + str(i) + "_red.csv", 3, 3, 4)
    
    
    
    model = keras.Sequential()
    model.add(
        keras.layers.Bidirectional(
            keras.layers.LSTM(
                units = 100,
                input_shape=(X_train.shape[1], X_train.shape[2])
                )
            )
        )
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(rate=0.4)) 
    model.add(Dense(3, activation='sigmoid'))
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=30,
        shuffle=False
        )
    
    test_pred = model.predict(X_test)
    accuracy = 1.0 - np.sqrt((((test_pred-y_test)**2).mean(axis=1)).mean())
    accuracies.append(accuracy)

print(accuracies)



