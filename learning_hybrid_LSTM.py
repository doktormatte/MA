import math
import time
import pandas as pd
import numpy  as np 
#from matplotlib import pyplot
#import matplotlib.pyplot as plt
import tensorflow as tf
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.Session(config=config)
import os
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

def split_sequences(sequences, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out-1
		# check if we are beyond the dataset
		if out_end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1:out_end_ix, -1]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)



def read_data(string,string2, n_steps_in,n_steps_out,n_features):
     
    Z = pd.read_csv(string)
    Z = Z.to_numpy()     

    X, y = split_sequences(Z[:,3:5], n_steps_in, n_steps_out)        
    
    n_train = int(0.7*len(X))
    Z1 = pd.read_csv(string2)
    Z1 = Z1.to_numpy()
    Z1 = Z1.transpose()
    Z2 = np.concatenate((Z1,Z1),axis=1) 
    X2 = np.zeros([len(Z),3+96],float)  
  
    for i in range(len(Z)-n_steps_in): 
     if Z[i+n_steps_in-1,-1] == 0:             
          qq = np.array(Z2[0][0:96])
          X2[i] = np.append(Z[i+n_steps_in-1][0:3],qq) 
     else:            
         qq = np.array(Z2[1][0:96])
         X2[i] = np.append(Z[i+n_steps_in-1][0:3],qq) 
    
    X_train = X[0: n_train,]
    y_train = y[0:n_train,]
    X_test = X[n_train: len(X),]
    y_test  = y[n_train:len(X),]
    X2_train = X2[0: n_train,]
    X2_test = X2[n_train: len(X),]
    
    return X_train,y_train,X_test,y_test,X2_train,X2_test


# station_1 = '/home/doktormatte/Dropbox/Dokumente/Studium/MA_SciComp/Code/Multistep-Electric-Vehicle-Charging-Station-Occupancy-Prediction--main/mixed_LSTM/data_chg_1.csv'
# station_2 = '/home/doktormatte/Dropbox/Dokumente/Studium/MA_SciComp/Code/Multistep-Electric-Vehicle-Charging-Station-Occupancy-Prediction--main/mixed_LSTM/data_chg_pred_occ_t_1.csv'




dirs = ['ACN_1', 'ACN_2', 'Boulder', 'Palo_Alto', 'Dundee', 'Perth_Kinross']
# dirs = ['ACN_1']
iteration = 0
summary_cols = ['names','layers','dataset','accuracy']

for i in range(200):
    try:
        # summary = []
        summary = pd.DataFrame(columns=summary_cols)
        for i in range(5):
            try:            
                for dirname in dirs:                
                    
                    n_steps_in = 12
                    n_features = 1
                    n_steps_out = 6
                    dropout_1 = random.randint(1,60)/100.0 
                    dropout_2 = random.randint(1,60)/100.0 
                    n_n_lstm = random.randint(8,128)
                    bat_size = random.randint(64,512)    
                    n_epoch = 60
                    
                    dense_1 = random.randint(4,128)
                    dense_2 = random.randint(4,128)
                    dense_3 = random.randint(4,128)
                    dense_4 = random.randint(4,128)                                          
                    
                    accuracies = []
                    model = None
                    for num in range(1,53):
                        try:
                            station_1 = '/home/doktormatte/MA_SciComp/' + dirname + '/Occup/' + str(num) + '_occup.csv'
                            station_2 = '/home/doktormatte/MA_SciComp/' + dirname + '/Occup/' + str(num) + '_averages.csv'
                            X_train,y_train,X_test,y_test,X2_train,X2_test=read_data(station_1,station_2,n_steps_in,n_steps_out,n_features)
                        except Exception:
                            continue                    
                        
                        iteration += 1
                        print('\n')
                        print('ITERATION ' + str(iteration))
                        print(dirname)
                        print('\n')                    
                        
                        input1 = keras.Input(shape=(n_steps_in, n_features))
                        input2 = keras.Input(shape=(99,))  
                        model_LSTM=LSTM(n_n_lstm)(input1)
                        model_LSTM=Dropout(dropout_1)(model_LSTM)
                        model_LSTM=Dense(dense_1, activation='relu')(model_LSTM)
    
                        meta_layer = keras.layers.Dense(99, activation="relu")(input2)
                        meta_layer = keras.layers.Dense(dense_2, activation="relu")(meta_layer)    
                        meta_layer = keras.layers.Dense(dense_3, activation="relu")(meta_layer)
                        model_merge = keras.layers.concatenate([model_LSTM, meta_layer])
                        model_merge = Dense(dense_4, activation='relu')(model_merge)
                        model_merge = Dropout(dropout_2)(model_merge)    
                        output = Dense(n_steps_out, activation='sigmoid')(model_merge)
                        model = Model(inputs=[input1, input2], outputs=output) 
                        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
                        history = model.fit([X_train, X2_train], y_train, epochs=n_epoch, batch_size=bat_size,shuffle=False)
                    
                        test_pred = model.predict([X_test,X2_test])
                        accuracy = 1.0 - np.sqrt((((test_pred-y_test)**2).mean(axis=1)).mean())
                        accuracies.append(accuracy)
                        
                    if not (model is None):
                        a = model.get_config()
                        results = pd.DataFrame(columns=summary_cols)
                        results['layers'] = a['layers']
                        results['names'] = a['name']
                        results['dataset'] = dirname
                        results['accuracy'] = np.mean(accuracies)
                        summary = pd.concat([summary, results])
                    
                    
            except KeyboardInterrupt:
                print('\n')
                print('interrupt to continue ...')
                print('\n')
                loop_forever = True
                while loop_forever:                        
                    try:
                        time.sleep(60)
                    except KeyboardInterrupt:
                        loop_forever = False                    
                continue
    
    
        timestr = time.strftime("%Y%m%d_%H%M%S")       
        summary.to_csv("/home/doktormatte/MA_SciComp/occup_hybrid_exp_res_" + timestr + ".csv", encoding='utf-8')
        
    except KeyboardInterrupt:
        break



# X_train,y_train,X_test,y_test,X2_train,X2_test=read_data(station_1,station_2,n_steps_in,n_steps_out,n_features)




