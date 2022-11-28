import math
import time
import pandas as pd
import numpy  as np 
import sys
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

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score



SEED = 0
env = 'local'
env = 'cloud'

import_path = '/home/doktormatte/Dropbox/Dokumente/Studium/MA_SciComp/Data/'
export_path = '/home/doktormatte/Dropbox/Dokumente/Studium/MA_SciComp/Data/'
if env == 'cloud':
    import_path = '/content/data/Data/'
    export_path = '/drive/MyDrive/'



def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)

    
def set_global_determinism(seed=SEED):
    set_seeds(seed=seed)

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

    X, y = split_sequences(Z[:,10:12], n_steps_in, n_steps_out)        
    
    n_train = int(0.7*len(X))
    Z1 = pd.read_csv(string2)
    Z1 = Z1.to_numpy()
    Z1 = Z1.transpose()
    Z2 = np.concatenate((Z1,Z1),axis=1) 
    X2 = np.zeros([len(Z),10+96],float)  
  
    for i in range(len(Z)-n_steps_in): 
     if Z[i+n_steps_in-1,-1] == 0:             
          qq = np.array(Z2[0][0:96])
          X2[i] = np.append(Z[i+n_steps_in-1][0:10],qq) 
     else:            
         qq = np.array(Z2[1][0:96])
         X2[i] = np.append(Z[i+n_steps_in-1][0:10],qq) 
    
    X_train = X[0: n_train,]
    y_train = y[0:n_train,]
    X_test = X[n_train: len(X),]
    y_test = y[n_train:len(X),]
    X2_train = X2[0: n_train,]
    X2_test = X2[n_train: len(X),]
    
    return X_train,y_train,X_test,y_test,X2_train,X2_test


# station_1 = '/home/doktormatte/Dropbox/Dokumente/Studium/MA_SciComp/Code/Multistep-Electric-Vehicle-Charging-Station-Occupancy-Prediction--main/mixed_LSTM/data_chg_1.csv'
# station_2 = '/home/doktormatte/Dropbox/Dokumente/Studium/MA_SciComp/Code/Multistep-Electric-Vehicle-Charging-Station-Occupancy-Prediction--main/mixed_LSTM/data_chg_pred_occ_t_1.csv'




dirs = ['ACN_1', 'ACN_2', 'Boulder', 'Palo_Alto', 'Dundee', 'Perth_Kinross']
# dirs = ['Dundee', 'Perth_Kinross']
random.shuffle(dirs)
# dirs = ['ACN_1']
iteration = 0
summary_cols = ['names','layers','dataset','accuracy','scores']

for i in range(2000):    
    summary = pd.DataFrame(columns=summary_cols)
    for i in range(5):
        try:          
            n_steps_in = 3
            n_features = 1
            n_steps_out = 3
            dropout_1 = random.randint(1,60)/100.0 
            dropout_2 = random.randint(1,60)/100.0 
            n_n_lstm = random.randint(8,128)
            bat_size = random.randint(1,512)    
            n_epoch = 60
            
            dense_1 = random.randint(4,128)
            dense_2 = random.randint(4,128)
            dense_3 = random.randint(4,128)
            dense_4 = random.randint(4,128)  
            
            for dirname in dirs:                                                                      
                
                accuracies = []
                F1_scores = np.empty((1, 3))
                model = None
                for num in range(1,53):
                    try:
                        station_1 = import_path + dirname + '/Occup/' + str(num) + '_red_header.csv'
                        station_2 = import_path + dirname + '/Occup/' + str(num) + '_averages.csv'
                        X_train,y_train,X_test,y_test,X2_train,X2_test=read_data(station_1,station_2,n_steps_in,n_steps_out,n_features)
                    except Exception:
                        continue                    
                    
                    iteration += 1
                    print('\n')
                    print('ITERATION ' + str(iteration) + ' hybrid occup')
                    print(dirname)
                    print('\n')                    
                    
                    input1 = keras.Input(shape=(n_steps_in, n_features))
                    input2 = keras.Input(shape=(106,))  
                    model_LSTM=LSTM(n_n_lstm)(input1)
                    model_LSTM=Dropout(dropout_1)(model_LSTM)
                    model_LSTM=Dense(dense_1, activation='relu')(model_LSTM)

                    meta_layer = keras.layers.Dense(106, activation="relu")(input2)
                    meta_layer = keras.layers.Dense(dense_2, activation="relu")(meta_layer)    
                    meta_layer = keras.layers.Dense(dense_3, activation="relu")(meta_layer)
                    model_merge = keras.layers.concatenate([model_LSTM, meta_layer])
                    model_merge = Dense(dense_4, activation='relu')(model_merge)
                    model_merge = Dropout(dropout_2)(model_merge)    
                    output = Dense(n_steps_out, activation='sigmoid')(model_merge)
                    model = Model(inputs=[input1, input2], outputs=output) 
                    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
                    history = model.fit([X_train, X2_train], y_train, epochs=n_epoch, batch_size=bat_size,shuffle=False)
                    
                    temp = model.predict([X_test,X2_test])
                    m,n=temp.shape 
                    t_target = n_steps_out
                       
                    yhat=np.zeros((m,t_target))
                    y_obs=np.array(y_test[0:m,0:t_target])
                    scores= np.zeros(m)
                    scores_F1 = np.zeros([m,3],float)
                    
                    for i in np.arange(m):  
                        for j in np.arange(t_target):  
                                if temp[i][j]>= 0.5:
                                    yhat[i][j]= 1                
                        val = 1.0 - sum(abs(yhat[i,]-y_obs[i,:]))/t_target     
                        scores_F1[i,0] = precision_score(y_obs[i,:], yhat[i,],zero_division=1)
                        scores_F1[i,1] = recall_score(y_obs[i,:], yhat[i,],zero_division=1)
                        scores_F1[i,2] = f1_score(y_obs[i,:], yhat[i,],zero_division=1)                             
                        scores[i] = val     
                     
                    mean = np.mean(scores)   
                    accuracy = mean
                    accuracies.append(accuracy)
                    mean_F1 = np.mean(scores_F1, axis=0)  
                    F1_scores = np.concatenate([F1_scores, np.array([mean_F1])], axis=0)
                    
                    a = model.get_config()
                    results = pd.DataFrame(columns=summary_cols)
                    results['layers'] = a['layers']
                    results['names'] = a['name']
                    results['dataset'] = dirname + '_' + str(num)
                    results['accuracy'] = accuracy
                    results['scores'] = " ".join(str(x) for x in list(mean_F1))
                    timestr = time.strftime("%Y%m%d_%H%M%S")       
                    results.to_csv(export_path + "occup_hybrid_exp_res_" + timestr + ".csv", encoding='utf-8')                
                    
                    
                    
                    
                    
            #     if not (model is None):
            #         a = model.get_config()
            #         results = pd.DataFrame(columns=summary_cols)
            #         results['layers'] = a['layers']
            #         results['names'] = a['name']
            #         results['dataset'] = dirname
            #         results['accuracy'] = np.mean(accuracies)
            #         results['scores'] = " ".join(str(x) for x in list(np.mean(F1_scores,axis=0)))  
            #         summary = pd.concat([summary, results])
                
            # timestr = time.strftime("%Y%m%d_%H%M%S")       
            # summary.to_csv(export_path + "occup_hybrid_exp_res_" + timestr + ".csv", encoding='utf-8')                
            # summary = pd.DataFrame(columns=summary_cols)    
            
            
        except KeyboardInterrupt:
            # sys.exit()
            # timestr = time.strftime("%Y%m%d_%H%M%S")       
            # summary.to_csv(export_path + "occup_hybrid_exp_res_" + timestr + ".csv", encoding='utf-8')                
            # summary = pd.DataFrame(columns=summary_cols)
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





# X_train,y_train,X_test,y_test,X2_train,X2_test=read_data(station_1,station_2,n_steps_in,n_steps_out,n_features)




