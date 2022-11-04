import pandas as pd
import numpy as np


dirs = ['ACN_2', 'ACN_1', 'Boulder', 'Dundee', 'Palo_Alto', 'Perth_Kinross']
modes = ['Loads', 'Occup']


cols = ['dataset', 'mode', 'accuracy']
df_accuracies = pd.DataFrame(columns=cols)


for mode in modes:    
    for dirname in dirs:        
        if mode == 'Loads':
            df = pd.read_csv('/home/doktormatte/MA_SciComp/' + dirname + '/' + mode + '/sum_red.csv', names = list(range(12)))
            n_steps_out = 3
            vals = np.array(df[10])
            pred = vals[:-(96*7)]
            vals = vals[96*7:]
            errors = []
            for i in range(0,len(vals),n_steps_out):
                try:                    
                    a = np.array([vals[i], vals[i+1], vals[i+2]])
                    b = np.array([pred[i], pred[i+1], pred[i+2]])
                    errors.append(np.mean((a-b)**2))                    
                except Exception:
                    pass                
            errors = np.array(errors)
            error = np.sqrt(np.mean(errors))   
            
            row = pd.DataFrame([{'dataset': dirname, 'mode': mode, 'accuracy': 1.0-error}])
            df_accuracies = pd.concat([df_accuracies, row])
            
        else:           
        
            for num in range(1,53):
                try:
                    
                    df = pd.read_csv ('/home/doktormatte/MA_SciComp/' + dirname + '/' + mode + '/' + str(num) + '_red.csv', names = list(range(12)))            
                    n_steps_out = 3
                    vals = np.array(df[10])
                    pred = vals[:-(96*7)]
                    vals = vals[96*7:]
                    errors = []
                    for i in range(0,len(vals),n_steps_out):
                        try:                    
                            a = np.array([vals[i], vals[i+1], vals[i+2]])
                            b = np.array([pred[i], pred[i+1], pred[i+2]])
                            errors.append(np.mean((a-b)**2))                    
                        except Exception:
                            pass                
                    errors = np.array(errors)
                    error = np.sqrt(np.mean(errors))       
                    
                    row = pd.DataFrame([{'dataset': dirname + '_' + str(num), 'mode': mode, 'accuracy': 1.0-error}])
                    df_accuracies = pd.concat([df_accuracies, row])
                    
                except Exception:
                    pass
                
                
file_name = '/home/doktormatte/MA_SciComp/res_naive_1w.csv'        
df_accuracies.to_csv(file_name, encoding='utf-8', index=False)
            
            
            
            
