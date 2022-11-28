import time
import pandas as pd
import numpy  as np 
import tensorflow as tf
import os

from tensorflow import keras
from numpy import array 
from keras.layers import LSTM,GRU,ConvLSTM2D
from keras.layers import Dense,Dropout,Flatten,TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D 

import json
import random

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
# dirs = ['Dundee', 'Perth_Kinross']
random.shuffle(dirs)
summary_cols = ['names','layers','dataset','accuracy', 'scores']
train_test_split = 0.7

iteration = 0


for i in range(2000):
    try:

        summary = []
        for i in range(5):
            try:            
                for dirname in dirs:                
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
                    epochs = 60
                    batch_size = random.randint(64,256)
                    architecture = random.choice(architectures)            
                    if architecture == 'Conv1D' or architecture == 'ConvLSTM':            
                        ker_size=1            
                    
                    accuracies = []
                    model = None
                    for j in range(1,53):
                        # num = random.randint(1,53)
                        num = j
                        try:
                            X_train,y_train,X_test,y_test = read_data(import_path + dirname + '/Occup/' + str(num) + '_red.csv', n_steps_in, n_steps_out, n_features, architecture)
                        except Exception:
                            continue                    
                        
                        iteration += 1
                        print('\n')
                        print('ITERATION ' + str(iteration) + ' non-hybrid occup')
                        print(dirname)
                        print('\n')
                        
                        model = keras.Sequential()
                        
                        if architecture == 'Conv1D':
                            model.add(Conv1D(filters=nf_1, kernel_size=ker_size, activation='relu', input_shape=(n_steps_in, n_features))) 
                            model.add(Conv1D(filters=nf_2, kernel_size=ker_size, activation='relu')) 
                            model.add(MaxPooling1D(pool_size=po_size))
                            model.add(Flatten())   
                            model.add(Dense(dense_1, activation='relu'))
                            model.add(Dropout(dropout)) 
                            model.add(Dense(n_steps_out, activation='sigmoid'))
                            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
                            
                        if architecture == 'ConvLSTM':
                            model.add(ConvLSTM2D(filters=nf_1, kernel_size=(1,ker_size),activation='relu', input_shape=(n_steps_in, 1, n_steps_out, n_features)))
                            model.add(Flatten())
                            model.add(Dense(dense_1, activation='relu'))
                            model.add(Dropout(dropout))  
                            model.add(Dense(n_steps_out, activation='sigmoid'))
                            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
                        
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
                            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])            
            
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
                            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
                            
                        if architecture == 'BiLSTM':
                            model.add(keras.layers.Bidirectional(keras.layers.LSTM(units=nodes_1,input_shape=(X_train.shape[1], X_train.shape[2]))))
                            model.add(Dense(dense_1, activation='relu'))
                            model.add(Dropout(dropout))  
                            model.add(Dense(n_steps_out, activation='sigmoid'))                
                            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
                        
                        if architecture == 'GRU':
                            model.add(keras.layers.GRU(units = nodes_1,input_shape=(X_train.shape[1],X_train.shape[2])))
                            model.add(Dense(dense_1, activation='relu'))
                            model.add(Dropout(dropout))  
                            model.add(Dense(n_steps_out, activation='sigmoid'))                
                            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
                            
                        if architecture == 'LSTM':
                            model.add(keras.layers.LSTM(units = nodes_1,input_shape=(X_train.shape[1],X_train.shape[2])))
                            model.add(Dense(dense_1, activation='relu'))
                            model.add(Dropout(dropout))  
                            model.add(Dense(n_steps_out, activation='sigmoid'))                
                            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])      
                        
                        
                        history = model.fit(
                            X_train, y_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            shuffle=False
                            )
                        
                    
                        temp = model.predict(X_test)
                        m,n = temp.shape 
                        t_target = n_steps_out
                           
                        yhat = np.zeros((m,t_target))
                        y_obs = np.array(y_test[0:m,0:t_target])
                        scores = np.zeros(m)
                        scores_F1 = np.zeros([m,3],float)
                        
                        for i in np.arange(m):  
                            for j in np.arange(t_target):  
                                    if temp[i][j] >= 0.5:
                                        yhat[i][j] = 1                
                            val = 1.0 - sum(abs(yhat[i,]-y_obs[i,:]))/t_target     
                            scores_F1[i,0] = precision_score(y_obs[i,:], yhat[i,],zero_division=1)
                            scores_F1[i,1] = recall_score(y_obs[i,:], yhat[i,],zero_division=1)
                            scores_F1[i,2] = f1_score(y_obs[i,:], yhat[i,],zero_division=1)                             
                            scores[i] = val       
                         
                        mean = np.mean(scores)   
                        mean_F1 = np.mean(scores_F1,axis=0)  
                        
                        a = model.get_config()
                        results = pd.DataFrame(columns=summary_cols)
                        results['layers'] = a['layers']
                        results['names'] = a['name']
                        results['dataset'] = dirname + '_' + str(num)
                        results['accuracy'] = mean
                        results['scores'] = " ".join(str(x) for x in list(mean_F1))
                        timestr = time.strftime("%Y%m%d_%H%M%S")       
                        results.to_csv(export_path + "occup_exp_res_" + timestr + ".csv", encoding='utf-8') 
                        
                        # if not(model is None):                               
                                
                        #     a = model.get_config()
                        #     b = json.dumps(a)
                        #     df = pd.read_json(b)
                        #     for index, row in df.iterrows():
                        #         meta = [row['name'], row['layers'], dirname + '_' + str(num), mean, mean_F1]
                        #         summary.append(meta)
                            
                    
                    
                    
            except KeyboardInterrupt:
                # print(summary)
                # timestr = time.strftime("%Y%m%d_%H%M%S")                
                # df_summary = pd.DataFrame(summary, columns=summary_cols)
                # df_summary.to_csv('/drive/MyDrive/occup_exp_res_' + timestr + '.csv', encoding='utf-8')
                # summary = []
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
    
    
        # timestr = time.strftime("%Y%m%d_%H%M%S")
    
        # df_summary = pd.DataFrame(summary, columns=summary_cols)
        # df_summary.to_csv('/drive/MyDrive/occup_exp_res_' + timestr + '.csv', encoding='utf-8')
    
    except KeyboardInterrupt:
        break



                            
