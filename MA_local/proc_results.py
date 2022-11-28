import time
import pandas as pd
import numpy  as np 
import os

path = '/home/doktormatte/Dropbox/Dokumente/Studium/MA_SciComp/Results_11_23_22/'
# path = '/home/doktormatte/gdrive/'
filename = os.listdir(path)

results_cols = ['dataset', 'hybrid_occup', 'non_hybrid_occup', 'log_regr_occup', 'hybrid_loads', 'non_hybrid_loads']

summary_cols_loads = ['name','layers','dataset','rmse','mae','r_squared']
summary_cols_occup = ['name','layers','dataset','accuracy','score']

name_accuracies_occup_cols = ['name', 'dataset', 'accuracy']

res_non_hybrid_loads = pd.DataFrame(columns=summary_cols_loads)
res_non_hybrid_occup = pd.DataFrame(columns=summary_cols_occup)
res_hybrid_loads = pd.DataFrame(columns=summary_cols_loads)
res_hybrid_occup = pd.DataFrame(columns=summary_cols_occup)
# res_log_regr = pd.DataFrame(columns=summary_cols)





    
for filename in filename:
    results = None
    try:
        if not ('lock' in filename):
            if 'csv' in filename:
                results = pd.read_csv(path + filename)
                # print(filename)
                if 'load' in filename:
                    if 'hybrid' in filename:                        
                        res_hybrid_loads = pd.concat([res_hybrid_loads, results]) 
                        # print(filename)
                    else:                        
                        res_non_hybrid_loads = pd.concat([res_non_hybrid_loads, results]) 
                        # print(filename)
                if 'occup' in filename:
                    if 'hybrid' in filename:
                        res_hybrid_occup = pd.concat([res_hybrid_occup, results])      
                        # print(filename)
                    else:
                        res_non_hybrid_occup = pd.concat([res_non_hybrid_occup, results])     
                        # print(filename)
    except:
        continue
            # if 'log_regr' in filename:                
            #     res_log_regr = pd.concat([res_log_regr, results])   
            

res_non_hybrid_occup.name = res_non_hybrid_occup.names
res_hybrid_occup.name = res_hybrid_occup.names
res_non_hybrid_loads.name = res_non_hybrid_loads.names
res_hybrid_loads.name = res_hybrid_loads.names


datasets = ['ACN_1','ACN_2','Palo_Alto','Boulder','Dundee','Perth_Kinross']
# results_DL = {'non_hybrid_loads': res_non_hybrid_loads, 'non_hybrid_occup': res_non_hybrid_occup, 'hybrid_loads': res_hybrid_loads, 'hybrid_occup': res_hybrid_occup}
results_DL_occup = {'non_hybrid_occup': res_non_hybrid_occup, 'hybrid_occup': res_hybrid_occup}
results_DL_loads = {'non_hybrid_loads': res_non_hybrid_loads, 'hybrid_loads': res_hybrid_loads}
accuracies_all = {}
precs_all = {}
recalls_all = {}
f1_score_all = {}
model_accuracies_occup = {}


name_accuracies_occup = pd.DataFrame(columns=summary_cols_occup)
for name in set(list(res_non_hybrid_occup.name)):
    model_results = res_non_hybrid_occup[res_non_hybrid_occup.name == name]
    for dataset in datasets:
        dataset_model_results = model_results[model_results.dataset.str.contains(dataset)==True]       
        row_model_results = pd.DataFrame(columns=summary_cols_occup)
        row_model_results['name'] = [name]
        row_model_results['dataset'] = dataset
        row_model_results['accuracy'] = dataset_model_results.accuracy[:1]
        name_accuracies_occup = pd.concat([name_accuracies_occup, row_model_results])
        
        
bad_names = []        
for name in set(list(res_non_hybrid_occup.name)):
    a = name_accuracies_occup[name_accuracies_occup.name == name]
    # print(len(a[a.accuracy.isnull()]))
    if len(a[a.accuracy.isnull()]) > 2:
        bad_names.append(name)

for bad_name in bad_names:
    name_accuracies_occup = name_accuracies_occup[~name_accuracies_occup.name.str.contains(bad_name)] 
    
mean_accuracies_non_hybrid_occup = pd.DataFrame(columns=['name','accuracy'])
for name in set(list(name_accuracies_occup.name)):
    a = name_accuracies_occup[name_accuracies_occup.name == name]
    row = pd.DataFrame(columns=['name','accuracy'])
    row['name'] = [name]
    row['accuracy'] = np.mean(a.accuracy)
    mean_accuracies_non_hybrid_occup = pd.concat([mean_accuracies_non_hybrid_occup, row])
    
    

best_models_non_hybrid_occup = mean_accuracies_non_hybrid_occup.sort_values('accuracy')[-5:]
best_models_non_hybrid_occup['precision'] = 0.0
best_models_non_hybrid_occup['recall'] = 0.0
best_models_non_hybrid_occup['f1_score'] = 0.0
for model in best_models_non_hybrid_occup.name:
    mean_precs = []
    mean_recalls = []
    mean_f1_score = []
    a = res_non_hybrid_occup[res_non_hybrid_occup.name == model]
    scores = a.scores
    for score in scores:
        mean_precs.append(float(score.split(' ')[0]))
        mean_recalls.append(float(score.split(' ')[1]))
        mean_f1_score.append(float(score.split(' ')[2]))
    # print(scores)
    best_models_non_hybrid_occup.loc[best_models_non_hybrid_occup['name'] == model, 'precision'] = np.mean(mean_precs)
    best_models_non_hybrid_occup.loc[best_models_non_hybrid_occup['name'] == model, 'recall'] = np.mean(mean_recalls)
    best_models_non_hybrid_occup.loc[best_models_non_hybrid_occup['name'] == model, 'f1_score'] = np.mean(mean_f1_score)
    
    
    
    
    
    
    
    
name_accuracies_occup = pd.DataFrame(columns=summary_cols_occup)
for name in set(list(res_hybrid_occup.name)):
    model_results = res_hybrid_occup[res_hybrid_occup.name == name]
    for dataset in datasets:
        dataset_model_results = model_results[model_results.dataset.str.contains(dataset)==True]       
        row_model_results = pd.DataFrame(columns=summary_cols_occup)
        row_model_results['name'] = [name]
        row_model_results['dataset'] = dataset
        row_model_results['accuracy'] = dataset_model_results.accuracy[:1]
        name_accuracies_occup = pd.concat([name_accuracies_occup, row_model_results])
        
        
bad_names = []        
for name in set(list(res_hybrid_occup.name)):
    a = name_accuracies_occup[name_accuracies_occup.name == name]
    # print(len(a[a.accuracy.isnull()]))
    if len(a[a.accuracy.isnull()]) > 2:
        bad_names.append(name)
    
for bad_name in bad_names:
    name_accuracies_occup = name_accuracies_occup[~name_accuracies_occup.name.str.contains(bad_name)] 
    
mean_accuracies_hybrid_occup = pd.DataFrame(columns=['name','accuracy'])
for name in set(list(name_accuracies_occup.name)):
    a = name_accuracies_occup[name_accuracies_occup.name == name]
    row = pd.DataFrame(columns=['name','accuracy'])
    row['name'] = [name]
    row['accuracy'] = np.mean(a.accuracy)
    mean_accuracies_hybrid_occup = pd.concat([mean_accuracies_hybrid_occup, row])
    
    
best_models_hybrid_occup = mean_accuracies_hybrid_occup.sort_values('accuracy')[-5:]
best_models_hybrid_occup['precision'] = 0.0
best_models_hybrid_occup['recall'] = 0.0
best_models_hybrid_occup['f1_score'] = 0.0
for model in best_models_hybrid_occup.name:
    mean_precs = []
    mean_recalls = []
    mean_f1_score = []
    a = res_hybrid_occup[res_hybrid_occup.name == model]
    scores = a.scores
    for score in scores:
        mean_precs.append(float(score.split(' ')[0]))
        mean_recalls.append(float(score.split(' ')[1]))
        mean_f1_score.append(float(score.split(' ')[2]))
    # print(scores)
    best_models_hybrid_occup.loc[best_models_hybrid_occup['name'] == model, 'precision'] = np.mean(mean_precs)
    best_models_hybrid_occup.loc[best_models_hybrid_occup['name'] == model, 'recall'] = np.mean(mean_recalls)
    best_models_hybrid_occup.loc[best_models_hybrid_occup['name'] == model, 'f1_score'] = np.mean(mean_f1_score)
    
    
    
    
