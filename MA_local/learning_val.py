# import math
import time
import pandas as pd
import numpy  as np 
import tensorflow as tf
#from matplotlib import pyplot
# import matplotlib.pyplot as plt
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.Session(config=config)
import os

from tensorflow import keras
from numpy import array 
# from keras.models import Sequential 
from keras.layers import LSTM,GRU,ConvLSTM2D
# from keras.layers import RepeatVector
from keras.layers import Dense,Dropout,Flatten,TimeDistributed
# from keras.layers import Dense,Dropout,TimeDistributed
# from keras.layers import BatchNormalization 
# from keras.layers import Bidirectional
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D 
# from keras.models import Model 
# from keras.layers import Input
# from tensorflow.keras.layers import concatenate
# from tensorflow.keras.layers import Flatten
import json
import random
# import time



SEED = 0
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    # random.seed(seed)
    tf.random.set_seed(seed)
    # np.random.seed(seed)
    
def set_global_determinism(seed=SEED):
    set_seeds(seed=seed)
    # os.environ['TF_DETERMINISTIC_OPS'] = '1'
    # os.environ['TF_CUDNN_DETERMINISTIC'] = '1'    
    # tf.config.threading.set_inter_op_parallelism_threads(1)
    # tf.config.threading.set_intra_op_parallelism_threads(1)


set_global_determinism(seed=SEED)


def pause():
    input("Press the <ENTER> key to continue...")

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


def normalize(x):
    return (x-min(x))/(max(x)-min(x))  


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
dirs = ['ACN_1', 'ACN_2', 'Boulder', 'Palo_Alto', 'Dundee', 'Perth_Kinross']
summary_cols = ['names','layers','dataset','accuracy']

iteration = 0



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
epochs = 5
batch_size = random.randint(64,256)

X_train,y_train,X_test,y_test = read_data("/home/doktormatte/MA_SciComp/ACN_1/Loads/sum.csv", n_steps_in, n_steps_out, n_features, 'LSTM')
X_train[:,:,10] = normalize(X_train[:,:,10])



model = keras.Sequential()
model.add(keras.layers.LSTM(units = nodes_1,input_shape=(X_train.shape[1],X_train.shape[2])))
model.add(Dense(dense_1, activation='relu'))
model.add(Dropout(dropout))  
model.add(Dense(n_steps_out, activation='sigmoid'))                
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])     

history = model.fit(X_train, y_train,epochs=epochs,batch_size=batch_size,shuffle=False)
temp = model.predict(X_test, verbose=2)
m,n=temp.shape 
t_target = n_steps_out
   
yhat=np.zeros((m,t_target))
y_obs=np.array(y_test[0:m,0:t_target])
scores1= np.zeros(m)

for i in np.arange(m):  
    for j in np.arange(t_target):  
            yhat[i][j] = temp[i][j]        
    val = 1-np.sqrt(((yhat[i,]-y_obs[i,:])**2).mean())
    scores1[i]=val       
 
_mean1 = np.mean(scores1)     
# res[_iter,:]=[nf_1, nf_2, ker_size, po_size, nodes_1,dropout,
#               n_epoch,bat_size, _mean1 ]  




