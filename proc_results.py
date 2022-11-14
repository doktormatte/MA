import time
import pandas as pd
import numpy  as np 
import os

path = '/home/doktormatte/MA_SciComp/'
filenames = os.listdir(path)

results_cols = ['dataset', 'hybrid_occup', 'non_hybrid_occup', 'log_regr_occup', 'hybrid_loads', 'non_hybrid_loads']

summary_cols = ['names','layers','dataset','accuracy']

res_non_hybrid_loads = pd.DataFrame(columns=summary_cols)
res_non_hybrid_occup = pd.DataFrame(columns=summary_cols)
res_hybrid_loads = pd.DataFrame(columns=summary_cols)
res_hybrid_occup = pd.DataFrame(columns=summary_cols)
res_log_regr = pd.DataFrame(columns=summary_cols)





for filename in filenames:
    results = None
    if not ('lock' in filename):
        if 'csv' in filename:
            results = pd.read_csv(path + filename)
            # print(filename)
            if 'load' in filename:
                if 'hybrid' in filename:                        
                    res_hybrid_loads = pd.concat([res_hybrid_loads, results])                    
                else:                        
                    res_non_hybrid_loads = pd.concat([res_non_hybrid_loads, results]) 
            if 'occup' in filename:
                if 'hybrid' in filename:
                    res_hybrid_occup = pd.concat([res_hybrid_occup, results])      
                else:
                    res_non_hybrid_occup = pd.concat([res_non_hybrid_occup, results])     
            if 'log_regr' in filename:                
                res_log_regr = pd.concat([res_log_regr, results])   


datasets = ['ACN_1','ACN_2','Palo_Alto','Boulder','Dundee','Perth_Kinross']
results_DL = {'non_hybrid_loads': res_non_hybrid_loads, 'non_hybrid_occup': res_non_hybrid_occup, 'hybrid_loads': res_hybrid_loads, 'hybrid_occup': res_hybrid_occup}
results_DL = {'non_hybrid_occup': res_non_hybrid_occup, 'hybrid_occup': res_hybrid_occup}
# results_DL = {'non_hybrid_loads': res_non_hybrid_loads, 'hybrid_loads': res_hybrid_loads}
accuracies_all = {}

for dataset in datasets:
    # accuracies_all[dataset + '_log_regr_occup'] = 1.0 - np.mean(res_log_regr[res_log_regr.dataset.str.contains(dataset)==True].accuracy)
    for key in results_DL:
        result_DL = results_DL[key]
        dataset_res_DL = result_DL[result_DL.dataset.str.contains(dataset)==True]
        mean_accuracies = []
        if len(set(dataset_res_DL.names)) <= 5:
            accuracies_all[dataset + '_' + key] = np.mean(dataset_res_DL.accuracy)
        else:
            for model_name in set(dataset_res_DL.names):
                mean_accuracies.append(np.mean(dataset_res_DL[dataset_res_DL.names == model_name].accuracy))
            accuracies_all[dataset + '_' + key] = np.mean(sorted(mean_accuracies)[:-5])
        # print(dataset + ' ' + key)
        # print(len(set(dataset_res_DL.names)))
        
    
        



# for dataset in datasets:


# log_regr_acc_ACN_1 = np.mean(res_log_regr[res_log_regr.dataset.str.contains('ACN_1')==True].accuracy)
# log_regr_acc_ACN_2 = np.mean(res_log_regr[res_log_regr.dataset.str.contains('ACN_2')==True].accuracy)
# log_regr_acc_Palo_Alto = np.mean(res_log_regr[res_log_regr.dataset.str.contains('Palo_Alto')==True].accuracy)
# log_regr_acc_Boulder = np.mean(res_log_regr[res_log_regr.dataset.str.contains('Boulder')==True].accuracy)
# log_regr_acc_Dundee = np.mean(res_log_regr[res_log_regr.dataset.str.contains('Dundee')==True].accuracy)
# log_regr_acc_Perth_Kinross = np.mean(res_log_regr[res_log_regr.dataset.str.contains('Perth_Kinross')==True].accuracy)
