import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, f1_score, precision_score, recall_score, accuracy_score


dirs = ['ACN_2', 'ACN_1', 'Boulder', 'Dundee', 'Palo_Alto', 'Perth_Kinross']
modes = ['Loads', 'Occup']
# modes = ['Loads']
# modes = ['Occup']


cols_loads = ['dataset', 'mode', 'rmse', 'mae', 'r_squared']
cols_occup = ['dataset', 'mode', 'accuracy', 'precision', 'recall', 'f1_score']
df_accuracies_loads = pd.DataFrame(columns=cols_loads)
df_accuracies_occup = pd.DataFrame(columns=cols_occup)


for mode in modes:    
    for dirname in dirs:        
        if mode == 'Loads':
            df = pd.read_csv('/home/doktormatte/MA_SciComp/' + dirname + '/' + mode + '/sum_red.csv', names = list(range(12)))
            n_steps_out = 3
            vals = np.array(df[10])
            pred = vals[:-96]
            vals = vals[96:]
            errors = []
            for i in range(0,len(vals),n_steps_out):
                try:                    
                    a = np.array([vals[i], vals[i+1], vals[i+2]])
                    b = np.array([pred[i], pred[i+1], pred[i+2]])
                    errors.append(np.mean((a-b)**2))                    
                except Exception:
                    pass                
            errors = np.array(errors)
            rmse = np.sqrt(np.mean(errors))   
            
            maes = []
            for j in range(0,len(vals),n_steps_out):                
                try:
                    a = np.array([vals[j], vals[j+1], vals[j+2]])
                    b = np.array([pred[j], pred[j+1], pred[j+2]])
                    maes.append(mean_absolute_error(a,b))
                except Exception:
                    pass
            mae = np.mean(maes)
            
            r_squared = r2_score(vals,pred)
            
            
            row_loads = pd.DataFrame([{'dataset': dirname, 'mode': mode, 'rmse': rmse, 'mae': mae, 'r_squared': r_squared}])
            df_accuracies_loads = pd.concat([df_accuracies_loads, row_loads])
            
        else:           
        
            for num in range(1,53):
                try:
                    
                    df = pd.read_csv ('/home/doktormatte/MA_SciComp/' + dirname + '/' + mode + '/' + str(num) + '_red.csv', names = list(range(12)))            
                    n_steps_out = 3
                    vals = np.array(df[10])
                    pred = vals[:-96]
                    vals = vals[96:]
                    maes = []
                    precs = []
                    recs = []
                    f1_scores = []
                    for i in range(0,len(vals),n_steps_out):
                        try:                    
                            a = np.array([vals[i], vals[i+1], vals[i+2]])
                            b = np.array([pred[i], pred[i+1], pred[i+2]])
                            
                            mae = sum(abs(a-b))/n_steps_out                            
                            maes.append(mae)     
                            
                            prec = precision_score(a, b, zero_division=1)
                            precs.append(prec)
                            
                            rec = recall_score(a, b, zero_division=1)
                            recs.append(rec)                           
                            
                            f1_scores.append(f1_score(a, b, zero_division=1))
                            
                        except Exception as e:
                            print(e)
                            pass         
                        
                        
                    
                    
                    
                    # errors = np.array(errors)
                    # error = np.sqrt(np.mean(errors))       
                    
                    row_occup = pd.DataFrame([{'dataset': dirname + '_' + str(num), 'mode': mode, 'accuracy': np.mean(maes), 'precision': np.mean(precs), 'recall': np.mean(recs), 'f1_score': np.mean(f1_scores)}])
                    df_accuracies_occup = pd.concat([df_accuracies_occup, row_occup])
                    
                except Exception:
                    pass
                
                
file_name_loads = '/home/doktormatte/MA_SciComp/res_naive_1d_loads.csv'        
df_accuracies_loads.to_csv(file_name_loads, encoding='utf-8', index=False)

file_name_occup = '/home/doktormatte/MA_SciComp/res_naive_1d_occup.csv'        
df_accuracies_occup.to_csv(file_name_occup, encoding='utf-8', index=False)
            
            
            
            
