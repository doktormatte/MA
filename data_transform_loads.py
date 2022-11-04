import pandas as pd
from numpy import array 
# from sklearn.preprocessing import MinMaxScaler

# dirs = ['Perth_Kinross']
dirs = ['ACN_1', 'ACN_2', 'Boulder', 'Dundee', 'Palo_Alto', 'Perth_Kinross']
train_test_split = 0.7

# dirs = ['Perth_Kinross']
# modes = ['Loads', 'Occup']
modes = ['Loads']
# scaler = MinMaxScaler(feature_range=(0, 1))

for dirname in dirs:
    for mode in modes:
        for num in range(1,53):
            try:
                cols = list(range(112))           
                df = pd.read_csv ('/home/doktormatte/MA_SciComp/' + dirname + '/' + mode + '/' + str(num) + '.csv', names = cols)            
                df = df.drop(columns=df.iloc[:, 15:111])      
                df = df.drop(columns=df.iloc[:, :4])          

                file_name = '/home/doktormatte/MA_SciComp/' + dirname + '/' + mode + '/' + str(num) + '_red.csv'
                df.to_csv(file_name, encoding='utf-8', index=False, header=False)
            except Exception:
                pass
            
            
            
for dirname in dirs:
    for mode in modes:
        sum_loads = pd.read_csv ('/home/doktormatte/MA_SciComp/' + dirname + '/' + mode + '/1_red.csv', names = list(range(12)))
        # df_init = pd.read_csv ('/home/doktormatte/MA_SciComp/' + dirname + '/' + mode + '/1_red.csv', names = cols)
        for column in sum_loads:
            if column == 10 or column == 11:
                sum_loads[column] = 0.0        
        
        for num in range(1,53):
            try:                
                df = pd.read_csv ('/home/doktormatte/MA_SciComp/' + dirname + '/' + mode + '/' + str(num) + '_red.csv', names = list(range(12)))                            
                for column in df:
                    if column == 10 or column == 11:
                        sum_loads[column] += df[column]  
            except Exception:
                pass
        file_name = '/home/doktormatte/MA_SciComp/' + dirname + '/' + mode + '/sum.csv'
        sum_loads.to_csv(file_name, encoding='utf-8', index=False, header=False)
        
        
        
        
        
for dirname in dirs:
    for mode in modes:
        df = pd.read_csv ('/home/doktormatte/MA_SciComp/' + dirname + '/' + mode + '/sum.csv', names = list(range(12)))
        df_norm = pd.DataFrame()
        for column in df:
            n_train=int(train_test_split*len(df[column])) 
            if column == 10 or column == 11:                
                x = df[column]
                x_train = x[:n_train]
                x_train = (x_train-min(x_train))/(max(x_train)-min(x_train))                
                x_test = x[n_train:]
                x_test = (x_test-min(x_test))/(max(x_test)-min(x_test))                               
                df_norm[column] = x_train + x_test
            else:
                y = df[column]
                df_norm[column] = y
        
        file_name = '/home/doktormatte/MA_SciComp/' + dirname + '/' + mode + '/sum_red.csv'
        df_norm.to_csv(file_name, encoding='utf-8', index=False, header=False)
        
            
# cols = list(range(1,102))            
# df = pd.read_csv ('/home/doktormatte/MA_SciComp/ACN/Loads/51.csv', names = cols)            
# df = df.drop(columns=df.iloc[:, 4:100])       

# df_norm = pd.DataFrame()
# for column in df:
#     df[column] = df[column].astype('float32')
#     # df_norm[column] = scaler.fit_transform(df[column].values.reshape(-1,1))
#     x = df[column]
#     x_std = (x-min(x))/(max(x)-min(x))    
#     df_norm[column] = x_std
#     # print(df[column])
# file_name = '/home/doktormatte/MA_SciComp/ACN/Loads/51_red.csv'
# df_norm.to_csv(file_name, encoding='utf-8', index=False, header=False)