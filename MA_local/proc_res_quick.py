import time
import pandas as pd
import numpy  as np 
import os


df = pd.read_csv('/home/doktormatte/Dropbox/Dokumente/Studium/MA_SciComp/log_regr_occup_exp_res_20221123_034139.csv')


datasets = ['ACN_1','ACN_2','Palo_Alto','Boulder','Dundee','Perth_Kinross']

for dataset in datasets:
    
    precs = []
    recalls = []
    f1_scores = []
    
    
    df_current = df[df.dataset.str.contains(dataset) == True]
    a = list(df_current.scores)
    
    for entry in a:
        precs.append(float(entry.split(' ')[0]))
        recalls.append(float(entry.split(' ')[1]))
        f1_scores.append(float(entry.split(' ')[2]))
        
    print(dataset + ' prec is: ' + str(np.mean(precs)))
    print(dataset + ' recall is: ' + str(np.mean(recalls)))
    print(dataset + ' f1_score is: ' + str(np.mean(f1_scores)))
    