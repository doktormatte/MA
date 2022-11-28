import math
import time
import pandas as pd
import numpy  as np 
#from matplotlib import pyplot
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from numpy import array 
from keras.models import Sequential 
from keras.layers import LSTM,GRU,ConvLSTM2D
from keras.layers import RepeatVector
from keras.layers import Dense,Dropout,Flatten,TimeDistributed
from keras.layers import Dense,Dropout,TimeDistributed
from keras.layers import BatchNormalization 
from keras.layers import Bidirectional
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D 
from keras.models import Model 
from keras.layers import Input
from tensorflow.keras.layers import concatenate
# from tensorflow.keras.layers import Flatten
import json
import random
import time

# gpu_devices = tf.config.experimental.list_physical_devices('GPU')
# for device in gpu_devices:
#     tf.config.experimental.set_memory_growth(device, True)


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


#df_boulder = pd.read_csv("/home/doktormatte/MA_SciComp/Boulder/Loads/1.csv")

#X_train,y_train,X_test,y_test = read_data("/home/doktormatte/MA_SciComp/Boulder/Loads/1.csv", 3, 3, 100)
# X_train,y_train,X_test,y_test = read_data("/home/doktormatte/MA_SciComp/Boulder/Loads/2.csv", 3, 3, 4)

accuracies = []
models = []

architectures = ['LSTM', 'GRU', 'BiLSTM', 'Stacked', 'Conv1D', 'CNN_LSTM', 'ConvLSTM']


dirs = ['ACN_1', 'ACN_2', 'Boulder', 'Palo_Alto']
# dirs = ['ACN_2']

summary_cols = ['names','layers','dataset','accuracy']
summary = []
iteration = 0
for i in range(100):

    try:
        
        for dirname in dirs:
            iteration += 1
            print('\n')
            print('ITERATION ' + str(iteration))
            print(dirname)
            print('\n')
            
            n_features = 11
            n_steps_in = 3
            n_steps_out = 3
            po_size = 2
            nf_1 = 32
            nf_2 = 16
            ker_size = 4
        
            stacked = random.randint(0,1)
            nodes_1 = random.randint(8,128)
            nodes_2 = random.randint(4,64)
            dense_1 = random.randint(4,128)
            activation = random.randint(0,1)
            dropout = random.randint(1,60)/100.0            
            epochs = 80
            batch_size = random.randint(64,256)
            architecture = random.choice(architectures)            
            if architecture == 'Conv1D' or architecture == 'ConvLSTM':            
                ker_size=1            
            
            X_train,y_train,X_test,y_test = read_data("/home/doktormatte/MA_SciComp/" + dirname + "/Loads/sum_red.csv", n_steps_in, n_steps_out, n_features, architecture)
            
            model = keras.Sequential()
            
            if architecture == 'Conv1D':
                model.add(Conv1D(filters=nf_1, kernel_size=ker_size, activation='relu', input_shape=(n_steps_in, n_features))) 
                model.add(Conv1D(filters=nf_2, kernel_size=ker_size, activation='relu')) 
                model.add(MaxPooling1D(pool_size=po_size))
                model.add(Flatten())   
                model.add(Dense(dense_1, activation='relu'))
                model.add(Dropout(dropout)) 
                model.add(Dense(n_steps_out, activation='sigmoid'))
                model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
                
            if architecture == 'ConvLSTM':
                model.add(ConvLSTM2D(filters=nf_1, kernel_size=(1,ker_size),activation='relu', input_shape=(n_steps_in, 1, n_steps_out, n_features)))
                model.add(Flatten())
                model.add(Dense(dense_1, activation='relu'))
                model.add(Dropout(dropout))  
                model.add(Dense(n_steps_out, activation='sigmoid'))
                model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
            
            if architecture == 'CNN_LSTM':
                model.add(TimeDistributed(Conv1D(filters=nf_1, kernel_size=ker_size,padding='same',activation='relu'), input_shape=(n_steps_in, n_steps_out, n_features)))
                model.add(TimeDistributed(MaxPooling1D(pool_size=po_size)))
                model.add(TimeDistributed(Conv1D(filters=nf_2, kernel_size=ker_size, padding='same',activation='relu'))) 
                model.add(TimeDistributed(MaxPooling1D(pool_size=po_size,padding='same')))
                model.add(TimeDistributed(Flatten())) 
                model.add(LSTM(nodes_1)) 
                model.add(Dense(dense_1, activation='relu'))
                model.add(Dropout(dropout))   
                model.add(Dense(n_steps_out, activation='sigmoid'))
                model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])            

            if architecture == 'Stacked':
                if stacked == 1:
                    model.add(keras.layers.LSTM(units = nodes_1,return_sequences=True,input_shape=(X_train.shape[1],X_train.shape[2])))
                    model.add(keras.layers.LSTM(nodes_2))
                else:
                    model.add(keras.layers.GRU(units = nodes_1,return_sequences=True,input_shape=(X_train.shape[1],X_train.shape[2])))
                    model.add(keras.layers.GRU(nodes_2))             
                model.add(Dense(dense_1, activation='relu'))
                model.add(Dropout(dropout))  
                model.add(Dense(n_steps_out, activation='sigmoid'))                
                model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
                
            if architecture == 'BiLSTM':
                model.add(keras.layers.Bidirectional(keras.layers.LSTM(units=nodes_1,input_shape=(X_train.shape[1], X_train.shape[2]))))
                model.add(Dense(dense_1, activation='relu'))
                model.add(Dropout(dropout))  
                model.add(Dense(n_steps_out, activation='sigmoid'))                
                model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
            
            if architecture == 'GRU':
                model.add(keras.layers.GRU(units = nodes_1,input_shape=(X_train.shape[1],X_train.shape[2])))
                model.add(Dense(dense_1, activation='relu'))
                model.add(Dropout(dropout))  
                model.add(Dense(n_steps_out, activation='sigmoid'))                
                model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
                
            if architecture == 'LSTM':
                model.add(keras.layers.LSTM(units = nodes_1,input_shape=(X_train.shape[1],X_train.shape[2])))
                model.add(Dense(dense_1, activation='relu'))
                model.add(Dropout(dropout))  
                model.add(Dense(n_steps_out, activation='sigmoid'))                
                model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])      
            
            
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=False
                )
            
            test_pred = model.predict(X_test)
            accuracy = 1.0 - np.sqrt((((test_pred-y_test)**2).mean(axis=1)).mean())
            
            
            a = model.get_config()
            b = json.dumps(a)
            df = pd.read_json(b)
            for index, row in df.iterrows():
                meta = [row['name'], row['layers'], dirname, accuracy]
                summary.append(meta)  
    except KeyboardInterrupt:
        break


timestr = time.strftime("%Y%m%d_%H%M%S")

df_summary = pd.DataFrame(summary, columns=summary_cols)
df_summary.to_csv("/home/doktormatte/MA_SciComp/exp_res_" + timestr + ".csv", encoding='utf-8')



    # accuracies.append(accuracy)
    # models.append(model)

# print(accuracies)
# pd.DataFrame(accuracies).to_csv("/home/doktormatte/MA_SciComp/results_LSTM_stacked_bsize64.csv", encoding='utf-8')

# print("ACN")




                            