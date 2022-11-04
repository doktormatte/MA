import pandas as pd
from numpy import array 
# from sklearn.preprocessing import MinMaxScaler

# dirs = ['Perth_Kinross']
# dirs = ['ACN_2', 'ACN_1', 'Boulder', 'Dundee', 'Palo_Alto', 'Perth_Kinross']
dirs = ['Perth_Kinross', 'Dundee']
# dirs = ['ACN_2', 'ACN_1', 'Boulder', 'Palo_Alto']
# dirs = ['Boulder', 'Palo_Alto']

# dirs = ['Palo_Alto']
# modes = ['Loads']
modes = ['Occup']
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
            
            
            
# for dirname in dirs:
#     for mode in modes:
#         sum_loads = pd.read_csv ('/home/doktormatte/MA_SciComp/' + dirname + '/' + mode + '/1_red.csv', names = list(range(12)))
#         # df_init = pd.read_csv ('/home/doktormatte/MA_SciComp/' + dirname + '/' + mode + '/1_red.csv', names = cols)
#         for column in sum_loads:
#             if column == 10 or column == 11:
#                 sum_loads[column] = 0.0        
        
#         for num in range(1,53):
#             try:                
#                 df = pd.read_csv ('/home/doktormatte/MA_SciComp/' + dirname + '/' + mode + '/' + str(num) + '_red.csv', names = list(range(12)))                            
#                 for column in df:
#                     if column == 10 or column == 11:
#                         sum_loads[column] += df[column]  
#             except Exception:
#                 pass
#         file_name = '/home/doktormatte/MA_SciComp/' + dirname + '/' + mode + '/sum.csv'
#         sum_loads.to_csv(file_name, encoding='utf-8', index=False, header=False)
        
        
        
        
        
# for dirname in dirs:
#     for mode in modes:
#         df = pd.read_csv ('/home/doktormatte/MA_SciComp/' + dirname + '/' + mode + '/sum.csv', names = list(range(12)))
#         df_norm = pd.DataFrame()
#         for column in df:
#             if column == 10 or column == 11:                
#                 x = df[column]
#                 x_std = (x-min(x))/(max(x)-min(x))    
#                 df_norm[column] = x_std
#             else:
#                 y = df[column]
#                 df_norm[column] = y
        
#         file_name = '/home/doktormatte/MA_SciComp/' + dirname + '/' + mode + '/sum_red.csv'
#         df_norm.to_csv(file_name, encoding='utf-8', index=False, header=False)


# for dirname in dirs:
#     for mode in modes:
#         for num in range(1,53):
#             try:
#                 df_test = pd.read_csv('/home/doktormatte/MA_SciComp/' + dirname + '/Occup/' + str(num) +'.csv', names = list(range(112)))

#                 df_test_weekday = df_test[df_test[4] == 0]
#                 df_test_weekday = df_test_weekday.iloc[:,15:111]
#                 weekday_avg = df_test_weekday.iloc[0,:]

#                 df_test_weekend = df_test[df_test[4] == 1]
#                 df_test_weekend = df_test_weekend.iloc[:,15:111]
#                 weekend_avg = df_test_weekend.iloc[0,:]

#                 averages = pd.DataFrame()
#                 averages['weekday'] = weekday_avg
#                 averages['weekend'] = weekend_avg
#                 averages.to_csv('/home/doktormatte/MA_SciComp/' + dirname + '/Occup/' + str(num) +'_averages.csv', index=False)

#                 df_test = df_test.drop(columns=df_test.iloc[:, 15:111])      
#                 df_test = df_test.drop(columns=df_test.iloc[:, 2:4])
#                 df_test = df_test.drop(columns=df_test.iloc[:, 3:12])
#                 header = ['t','dayofweek','weekend','y_t_1','y']
#                 df_test.to_csv('/home/doktormatte/MA_SciComp/' + dirname + '/Occup/' + str(num) +'_occup.csv',encoding='utf-8', index=False, header=header)
#             except Exception:
#                 pass


        
            
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