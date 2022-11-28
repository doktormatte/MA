import json
import datetime
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt



# dirs = ['ACN_1', 'ACN_2', 'Boulder', 'Palo_Alto', 'Dundee', 'Perth_Kinross']
# occup_ratios = dict.fromkeys(dirs)

# for dirname in dirs:
#     occup_ratios[dirname] = {}
#     df_counts = pd.read_csv('/home/doktormatte/MA_SciComp/' + dirname + '/Occup/session_counts.csv')
    
#     for num in range(1,53):
#         try:
            
            
            
#             df_occup = pd.read_csv('/home/doktormatte/MA_SciComp/' + dirname + '/Occup/' + str(num) + '_red_header.csv')
#             count = df_counts[df_counts['num'] == num]['count'].values[0]
#             df_occup = df_occup.astype({'k':'int'})
#             occup_ratio = len(df_occup[df_occup['k'] == 1])/len(df_occup['k'])
#             occup_ratios[dirname][num] = [count, occup_ratio]
            
            
#         except Exception as e:
#             # print(e)
#             pass


df_sessions = pd.read_csv('/home/doktormatte/MA_SciComp/Palo_Alto/Occup/session_counts.csv')
key_x = list(df_sessions['count'])
key_y = list(df_sessions['check'])
plt.xlabel('#sessions')
plt.ylabel('occupancy')
plt.scatter(key_x, key_y)
plt.show()

# for key in occup_ratios:
#     key_x = []
#     key_y = []
#     for stat_num in occup_ratios[key]:
#         key_x.append(occup_ratios[key][stat_num][0])
#         key_y.append(occup_ratios[key][stat_num][1])
    
#     plt.title(key)
#     plt.xlabel('#sessions')
#     plt.ylabel('occupancy')
#     plt.scatter(key_x, key_y)
#     plt.show()