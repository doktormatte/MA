import datetime
import pandas as pd
import numpy as np
from scipy import stats
import sys 
from workalendar.europe import Scotland

def roundTime(dt=None):    
    dt = dt.to_pydatetime()
    roundTo = 15*60    
    if dt == None : dt = datetime.datetime.now()
    seconds = (dt.replace(tzinfo=None) - dt.min).seconds
    rounding = (seconds+roundTo/2) // roundTo * roundTo
    return dt + datetime.timedelta(0,rounding-seconds,-dt.microsecond)

def create_start_timestamp(row):
    return row['Start Date'][:10] + " " + row['Start Time']

def create_end_timestamp(row):
    return row['End Date'][:10] + " " + row['End Time']


def conv_entries(arr):  
    int_arr = [int(x) for x in arr]
    if int_arr[2] > 29:
        int_arr[1] += 1        
    int_arr[2] = 0
    mins = int_arr[0]*60 + int_arr[1]
    rem = mins % 15
    if rem > 7:
        mins += (15-rem)
    else:
        mins -= rem        
    return mins

def calc_quarter_load(row):
    dur = row['Session_Duration']
    total = row['Total kWh']
    return (total/dur)*15.0


def conv_timestamp(ts):
    time_arr = str(ts)[-8:].split(':')
    hours = int(time_arr[0])
    mins = int(time_arr[1])
    return (hours*60+mins)//15

def get_day_of_week(ts):
    return ts.weekday()

def get_day_of_month(ts):
    return ts.day

def get_day_of_year(ts):
    return ts.timetuple().tm_yday

def get_day_index(ts):
    idx = ts.weekday()
    if idx == 6:
        return 0
    return idx+1

def get_weekend(ts):
    if ts.weekday() > 4:
        return 1
    return 0

def get_sin(x, x_max):
    return np.sin(2.0*np.pi*x/x_max)

def get_cos(x, x_max):
    return np.cos(2.0*np.pi*x/x_max)


def get_holiday(ts):
    year = ts.year
    cal = Scotland()
    holidays = cal.holidays(year)
    for holiday in holidays:
        if holiday[0] == datetime.date.fromtimestamp(datetime.datetime.timestamp(ts)):
            return 1
    return 0   

def calc_session_dur(row):
    sess_dur = (row['Total kWh']/50.0)*60.0
    if sess_dur < 1.0:
        return 1.0
    return float(round(sess_dur))
    

def calc_energy_per_min(row):
    # return (row['Total kWh']/60.0)*float(row['Session_Duration'])
    return row['Total kWh']/row['Session_Duration']
 



def calc_total_dur(row):
    timedelta = row['End_Timestamp'] - row['Start_Timestamp']
    return timedelta.total_seconds() / 60

def calc_minute_load(row):
    return row['Total kWh']/row['Session_Duration']


def adjust_ts(row):
    if row['Start_Timestamp'] == row['End_Timestamp']:
        return row['Start_Timestamp'] + datetime.timedelta(minutes = 15)
    return row['End_Timestamp']


def add_to_backbones(row, stat_name):    
    iters_load = int(row['Session_Duration'])
    backbone_load = stat_backbones[stat_name][0]
    for i in range(iters_load):
        # if (backbone_load.loc[backbone_load['date_time'] == row['Start_Timestamp'] + datetime.timedelta(minutes=i), 'value'] != 0):
        #     print(row['Start_Timestamp'])
        backbone_load.loc[backbone_load['date_time'] == row['Start_Timestamp'] + datetime.timedelta(minutes=i), 'value'] += row['Energy_per_min']     
    
    
    iters_occup = int(row['Total_Duration'])
    backbone_occup = stat_backbones[stat_name][1]
    for i in range(iters_occup):
        backbone_occup.loc[backbone_occup['date_time'] == row['Start_Timestamp'] + datetime.timedelta(minutes=i), 'value'] = 1


   

   

df_pk = pd.read_csv (r'/home/doktormatte/Dropbox/Dokumente/Studium/MA_SciComp/Data/PerthKinross_2016-2019.csv')

rapid_chargers = ['APT Triple Rapid Charger', 'Siemens Triple Rapid Charger']
df_pk = df_pk[df_pk['Model'].isin(rapid_chargers) == True]


df_pk = df_pk[df_pk['Total kWh'].notna()]
df_pk = df_pk[df_pk['Total kWh'] > 0.0]

df_pk['Start_Timestamp'] = df_pk.apply(lambda row: create_start_timestamp(row), axis=1)
df_pk['End_Timestamp'] = df_pk.apply(lambda row: create_end_timestamp(row), axis=1)


cutoff = datetime.datetime.strptime("2017-07-01 00:00", '%Y-%m-%d %H:%M')
df_pk['Start_Timestamp'] = df_pk['Start_Timestamp'].map(lambda x: datetime.datetime.strptime(x, '%d/%m/%Y %H:%M'))
df_pk['End_Timestamp'] = df_pk['End_Timestamp'].map(lambda x: datetime.datetime.strptime(x, '%d/%m/%Y %H:%M'))
df_pk = df_pk[df_pk['Start_Timestamp'] >= cutoff]

df_pk['Start_Timestamp'] = df_pk['Start_Timestamp'].dt.to_pydatetime()
# df_pk['Start_Timestamp'] = df_pk['Start_Timestamp'].map(roundTime)

df_pk['End_Timestamp'] = df_pk['End_Timestamp'].dt.to_pydatetime()
# df_pk['End_Timestamp'] = df_pk['End_Timestamp'].map(roundTime)



df_pk = df_pk[df_pk.Start_Timestamp < df_pk.End_Timestamp]

df_pk['Total_Duration'] = df_pk.apply(lambda row: calc_total_dur(row), axis=1)


df_pk['z_score_dur'] = np.abs(stats.zscore(df_pk['Total_Duration']))
df_pk = df_pk[df_pk.z_score_dur <= 3.0]

df_pk['Session_Duration'] = df_pk.apply(lambda row: calc_session_dur(row), axis=1)
df_pk['Energy_per_min'] = df_pk.apply(lambda row: calc_energy_per_min(row), axis=1)

df_pk['z_score_ene'] = np.abs(stats.zscore(df_pk['Energy_per_min']))
df_pk = df_pk[df_pk.z_score_ene <= 3.0]


stations = list(set(list(df_pk['CP ID'])))
# stations = [50245]
start = cutoff
end = datetime.datetime.strptime("2019-09-01 00:00", '%Y-%m-%d %H:%M')


stat_backbones = dict.fromkeys(stations)
load_weekday_averages = dict.fromkeys(stations)
load_weekend_averages = dict.fromkeys(stations)

occup_weekday_averages = dict.fromkeys(stations)
occup_weekend_averages = dict.fromkeys(stations)

glob_week_averages = pd.DataFrame({'timeslot': list(range(96))})
glob_week_averages['avg_value'] = 0.0 

glob_weekend_averages = pd.DataFrame({'timeslot': list(range(96))})
glob_weekend_averages['avg_value'] = 0.0 


# sys.exit()

for stat_name in stations:
    
    backbone_load = pd.DataFrame({'date_time': pd.date_range(start, end, freq="1min")})
    backbone_load = backbone_load.loc[:len(backbone_load)-len(backbone_load)%15-1,:]
    backbone_load.set_index('date_time')
    backbone_load['timeslot'] = backbone_load['date_time'].apply(conv_timestamp)
    backbone_load['day_index'] = backbone_load['date_time'].apply(get_day_index)
    backbone_load['weekend'] = backbone_load['date_time'].apply(get_weekend)
    backbone_load['value'] = 0.0
    
    
    backbone_occup = pd.DataFrame({'date_time': pd.date_range(start, end, freq="1min")})
    backbone_occup = backbone_occup.loc[:len(backbone_occup)-len(backbone_occup)%15-1,:]
    backbone_occup.set_index('date_time')
    backbone_occup['timeslot'] = backbone_occup['date_time'].apply(conv_timestamp)
    backbone_occup['day_index'] = backbone_occup['date_time'].apply(get_day_index)
    backbone_occup['weekend'] = backbone_occup['date_time'].apply(get_weekend) 
    backbone_occup['value'] = 0        
    
    stat_backbones[stat_name] = [backbone_load, backbone_occup, None, None]
    
 
  
 
for stat_name in stations:

    df_pk_stat = df_pk[df_pk['CP ID'] == stat_name]        
    df_pk_stat.apply(lambda row: add_to_backbones(row, stat_name), axis=1)  
    print(stat_name)
    


for stat_name in stations:
    
    backbone_load_min = stat_backbones[stat_name][0]
    values_load_min = backbone_load_min['value']     
    backbone_load_quarter = pd.DataFrame({'date_time': pd.date_range(start, end + datetime.timedelta(minutes=15), freq="15min")})
    backbone_load_quarter = backbone_load_quarter.loc[:int(len(values_load_min)//15)-1,:]
    backbone_load_quarter.set_index('date_time')
    
    backbone_load_quarter['timeslot'] = backbone_load_quarter['date_time'].apply(conv_timestamp)
    max_timeslot = max(backbone_load_quarter['timeslot'])
    backbone_load_quarter['day_of_week'] = backbone_load_quarter['date_time'].apply(get_day_of_week)
    max_day_of_week = max(backbone_load_quarter['day_of_week'])
    backbone_load_quarter['day_of_month'] = backbone_load_quarter['date_time'].apply(get_day_of_month)
    max_day_of_month = max(backbone_load_quarter['day_of_month'])
    backbone_load_quarter['day_of_year'] = backbone_load_quarter['date_time'].apply(get_day_of_year)
    max_day_of_year = max(backbone_load_quarter['day_of_year'])
    backbone_load_quarter['weekend'] = backbone_load_quarter['date_time'].apply(get_weekend)
    backbone_load_quarter['holiday'] = backbone_load_quarter['date_time'].apply(get_holiday)
      
    backbone_load_quarter['timeslot_sin'] = backbone_load_quarter.apply(lambda x: get_sin(x['timeslot'], max_timeslot),axis=1)
    backbone_load_quarter['timeslot_cos'] = backbone_load_quarter.apply(lambda x: get_cos(x['timeslot'], max_timeslot),axis=1)
    backbone_load_quarter['day_of_week_sin'] = backbone_load_quarter.apply(lambda x: get_sin(x['day_of_week'], max_day_of_week),axis=1)
    backbone_load_quarter['day_of_week_cos'] = backbone_load_quarter.apply(lambda x: get_cos(x['day_of_week'], max_day_of_week),axis=1)
    backbone_load_quarter['day_of_month_sin'] = backbone_load_quarter.apply(lambda x: get_sin(x['day_of_month'], max_day_of_month),axis=1)
    backbone_load_quarter['day_of_month_cos'] = backbone_load_quarter.apply(lambda x: get_cos(x['day_of_month'], max_day_of_month),axis=1)    
    backbone_load_quarter['day_of_year_sin'] = backbone_load_quarter.apply(lambda x: get_sin(x['day_of_year'], max_day_of_year),axis=1)
    backbone_load_quarter['day_of_year_cos'] = backbone_load_quarter.apply(lambda x: get_cos(x['day_of_year'], max_day_of_year),axis=1)      

    backbone_load_quarter['value'] = 0.0        
    quarter_loads = []           
    sum_loads = 0.0    
    for i in range(len(values_load_min)):
        sum_loads += values_load_min[i]        
        if (i+1)%15 == 0:
            quarter_loads.append(sum_loads)                 
            sum_loads = 0.0       
    backbone_load_quarter['value'] = quarter_loads        
    stat_backbones[stat_name][2] = backbone_load_quarter
    
    
    
    
    
    backbone_occup_min = stat_backbones[stat_name][1]
    # backbone_occup_min.to_csv('/home/doktormatte/MA_SciComp/test_occup.csv', encoding='utf-8')
    values_occup_min = backbone_occup_min['value'] 
    backbone_occup_quarter = pd.DataFrame({'date_time': pd.date_range(start, end + datetime.timedelta(minutes=15), freq="15min")})
    backbone_occup_quarter = backbone_occup_quarter.loc[:int(len(values_load_min)//15)-1,:]
    backbone_occup_quarter.set_index('date_time')
    
    backbone_occup_quarter['timeslot'] = backbone_occup_quarter['date_time'].apply(conv_timestamp)
    max_timeslot = max(backbone_occup_quarter['timeslot'])
    backbone_occup_quarter['day_of_week'] = backbone_occup_quarter['date_time'].apply(get_day_of_week)
    max_day_of_week = max(backbone_occup_quarter['day_of_week'])
    backbone_occup_quarter['day_of_month'] = backbone_occup_quarter['date_time'].apply(get_day_of_month)
    max_day_of_month = max(backbone_occup_quarter['day_of_month'])
    backbone_occup_quarter['day_of_year'] = backbone_occup_quarter['date_time'].apply(get_day_of_year)
    max_day_of_year = max(backbone_occup_quarter['day_of_year'])
    backbone_occup_quarter['weekend'] = backbone_occup_quarter['date_time'].apply(get_weekend)
    backbone_occup_quarter['holiday'] = backbone_occup_quarter['date_time'].apply(get_holiday)
      
    backbone_occup_quarter['timeslot_sin'] = backbone_occup_quarter.apply(lambda x: get_sin(x['timeslot'], max_timeslot),axis=1)
    backbone_occup_quarter['timeslot_cos'] = backbone_occup_quarter.apply(lambda x: get_cos(x['timeslot'], max_timeslot),axis=1)
    backbone_occup_quarter['day_of_week_sin'] = backbone_occup_quarter.apply(lambda x: get_sin(x['day_of_week'], max_day_of_week),axis=1)
    backbone_occup_quarter['day_of_week_cos'] = backbone_occup_quarter.apply(lambda x: get_cos(x['day_of_week'], max_day_of_week),axis=1)
    backbone_occup_quarter['day_of_month_sin'] = backbone_occup_quarter.apply(lambda x: get_sin(x['day_of_month'], max_day_of_month),axis=1)
    backbone_occup_quarter['day_of_month_cos'] = backbone_occup_quarter.apply(lambda x: get_cos(x['day_of_month'], max_day_of_month),axis=1)    
    backbone_occup_quarter['day_of_year_sin'] = backbone_occup_quarter.apply(lambda x: get_sin(x['day_of_year'], max_day_of_year),axis=1)
    backbone_occup_quarter['day_of_year_cos'] = backbone_occup_quarter.apply(lambda x: get_cos(x['day_of_year'], max_day_of_year),axis=1) 
    

    backbone_occup_quarter['value'] = 0
    quarter_occup = []
    sum_occup = 0
    for i in range(len(values_occup_min)):
        sum_occup += values_occup_min[i]        
        if (i+1)%15 == 0:
            if sum_occup > 0:                
                quarter_occup.append(1)  
            else:
                quarter_occup.append(0)  
            sum_occup = 0            
    backbone_occup_quarter['value'] = quarter_occup    
    stat_backbones[stat_name][3] = backbone_occup_quarter
    

for stat_name in stations:
    
    df_pk_stat = df_pk[df_pk['CP ID'] == stat_name]        
    df_pk_stat.apply(lambda row: add_to_backbones(row, stat_name), axis=1)        
    occup_avg_weekday = pd.DataFrame({'timeslot': list(range(96))})
    occup_avg_weekday['avg_value'] = 0.0      
    occup_avg_weekend = pd.DataFrame({'timeslot': list(range(96))})
    occup_avg_weekend['avg_value'] = 0.0      
    
    load_avg_weekday = pd.DataFrame({'timeslot': list(range(96))})
    load_avg_weekday['avg_value'] = 0.0      
    load_avg_weekend = pd.DataFrame({'timeslot': list(range(96))})
    load_avg_weekend['avg_value'] = 0.0    
    
    backbone_load = stat_backbones[stat_name][2]
    train_test_split = 0.7
    n_train = int(train_test_split*len(backbone_load)) 
    backbone_load_clipped = backbone_load[:n_train]
    load_weekday_averages[stat_name] = load_avg_weekday
    load_weekend_averages[stat_name] = load_avg_weekend
    
    for i in range(96): 
        avg_value_load = backbone_load_clipped[(backbone_load_clipped.timeslot == i) & (backbone_load_clipped.weekend == 0)].value.sum()
        load_avg_weekday.loc[load_avg_weekday['timeslot'] == i, 'avg_value']  = avg_value_load
    for i in range(96): 
        load_avg_weekday.loc[load_avg_weekday['timeslot'] == i, 'avg_value'] /= len(backbone_load_clipped[(backbone_load_clipped.timeslot == i) & (backbone_load_clipped.weekend == 0)])
    load_weekday_averages[stat_name] = load_avg_weekday
    
    for i in range(96): 
        avg_value_load = backbone_load_clipped[(backbone_load_clipped.timeslot == i) & (backbone_load_clipped.weekend == 1)].value.sum()
        load_avg_weekend.loc[load_avg_weekend['timeslot'] == i, 'avg_value']  = avg_value_load
    for i in range(96): 
        load_avg_weekend.loc[load_avg_weekend['timeslot'] == i, 'avg_value'] /= len(backbone_load_clipped[(backbone_load_clipped.timeslot == i) & (backbone_load_clipped.weekend == 1)])
    load_weekend_averages[stat_name] = load_avg_weekend
    
    
    backbone_occup = stat_backbones[stat_name][3]
    n_train = int(train_test_split*len(backbone_occup)) 
    backbone_occup_clipped = backbone_occup[:n_train]
    
    for i in range(96):    
        avg_value_occup = len(backbone_occup_clipped[(backbone_occup_clipped.timeslot == i) & (backbone_occup_clipped.value == 1) & (backbone_occup_clipped.weekend == 0)]) / len(backbone_occup_clipped[(backbone_occup_clipped.timeslot == i) & (backbone_occup_clipped.weekend == 0)])
        occup_avg_weekday.loc[occup_avg_weekday['timeslot'] == i, 'avg_value'] = avg_value_occup
        
        # glob_week_averages.loc[glob_week_averages['timeslot'] == i, 'avg_value_occup'] += avg_value_occup 
    occup_weekday_averages[stat_name] = occup_avg_weekday
        
    for i in range(96):    
        avg_value_occup = len(backbone_occup_clipped[(backbone_occup_clipped.timeslot == i) & (backbone_occup_clipped.value == 1) & (backbone_occup_clipped.weekend == 1)]) / len(backbone_occup_clipped[(backbone_occup_clipped.timeslot == i) & (backbone_occup_clipped.weekend == 1)])        
        occup_avg_weekend.loc[occup_avg_weekend['timeslot'] == i, 'avg_value'] = avg_value_occup
        
        # glob_weekend_averages.loc[glob_weekend_averages['timeslot'] == i, 'avg_value'] += avg_value  
    occup_weekend_averages[stat_name] = occup_avg_weekend
    
    print(stat_name)
    
    

    
    
print('\n')
backbone_num = 1

for stat_name in stations:
    
   backbone_load = stat_backbones[stat_name][2]
   
   backbone_load_weekday = backbone_load[backbone_load['weekend'] == 0]
   load_avg_weekday = load_weekday_averages[stat_name].T.drop(labels='timeslot', axis=0)
   load_weekday_data = pd.concat([load_avg_weekday]*len(backbone_load_weekday), ignore_index=True)
   backbone_load_weekday = pd.concat([backbone_load_weekday.reset_index(drop=True), load_weekday_data.reset_index(drop=True)], axis=1)
   
   backbone_load_weekend = backbone_load[backbone_load['weekend'] == 1]
   load_avg_weekend = load_weekend_averages[stat_name].T.drop(labels='timeslot', axis=0)
   load_weekend_data = pd.concat([load_avg_weekend]*len(backbone_load_weekend), ignore_index=True)
   backbone_load_weekend = pd.concat([backbone_load_weekend.reset_index(drop=True), load_weekend_data.reset_index(drop=True)], axis=1)
   
   sorted_backbone_load = pd.concat([backbone_load_weekday, backbone_load_weekend]).sort_values(by=['date_time'], ascending=True)
   sorted_backbone_load['value_shifted'] = np.roll(sorted_backbone_load['value'],-1)
   sorted_backbone_load.drop('date_time', axis=1, inplace=True)
   
   file_name_load = "/home/doktormatte/MA_SciComp/Perth_Kinross/Loads/" + str(backbone_num) + ".csv"
   # file_name_load = "/home/doktormatte/MA_SciComp/Boulder/Loads/" + stat_name.replace('/', '') + ".csv"
   sorted_backbone_load.to_csv(file_name_load, encoding='utf-8', index=False, header=False)
   
   
    
    
   backbone_occup = stat_backbones[stat_name][3]
   
   backbone_occup_weekday = backbone_occup[backbone_occup['weekend'] == 0]
   occup_avg_weekday = occup_weekday_averages[stat_name].T.drop(labels='timeslot', axis=0)
   occup_weekday_data = pd.concat([occup_avg_weekday]*len(backbone_occup_weekday), ignore_index=True)
   backbone_occup_weekday = pd.concat([backbone_occup_weekday.reset_index(drop=True), occup_weekday_data.reset_index(drop=True)], axis=1)
   
   backbone_occup_weekend = backbone_occup[backbone_occup['weekend'] == 1]
   occup_avg_weekend = occup_weekend_averages[stat_name].T.drop(labels='timeslot', axis=0)
   occup_weekend_data = pd.concat([occup_avg_weekend]*len(backbone_occup_weekend), ignore_index=True)
   backbone_occup_weekend = pd.concat([backbone_occup_weekend.reset_index(drop=True), occup_weekend_data.reset_index(drop=True)], axis=1)
   
   sorted_backbone_occup = pd.concat([backbone_occup_weekday, backbone_occup_weekend]).sort_values(by=['date_time'], ascending=True)
   sorted_backbone_occup['value_shifted'] = np.roll(sorted_backbone_occup['value'],-1)
   sorted_backbone_occup.drop('date_time', axis=1, inplace=True)
   
   
   
   file_name_occup = "/home/doktormatte/MA_SciComp/Perth_Kinross/Occup/" + str(backbone_num) + ".csv"
   
   # file_name_occup = "/home/doktormatte/MA_SciComp/Boulder/Occup/" + stat_name.replace('/', '') + ".csv"
   sorted_backbone_occup.to_csv(file_name_occup, encoding='utf-8', index=False, header=False)
   
   
   backbone_num += 1  
   
   print(stat_name)






# df_pk['New_End_Timestamp'] = df_pk.apply(lambda row: adjust_ts(row), axis=1)
# df_pk['End_Timestamp'] = df_pk['New_End_Timestamp']




# df_pk['Session_Duration'] = df_pk.apply(lambda row: calc_sessions_dur(row), axis=1)
# df_pk['Total_Duration'] = df_pk.apply(lambda row: calc_total_dur(row), axis=1)

# df_pk['z_score'] = np.abs(stats.zscore(df_pk['Session_Duration']))
# df_pk = df_pk[df_pk.z_score <= 3.0]

# print(len(df_pk))
# df_pk = df_pk[df_pk['Total_Duration'] >= df_pk['Session_Duration']]
# print(len(df_pk))




# df_pk['Load_per_quarter'] = df_pk.apply(lambda row: calc_quarter_load(row), axis=1)

# stations = list(set(list(df_pk['CP ID'])))
# start = cutoff
# end = df_pk['End_Timestamp'].max()


# stat_backbones = dict.fromkeys(stations)
# load_weekday_averages = dict.fromkeys(stations)
# load_weekend_averages = dict.fromkeys(stations)

# occup_weekday_averages = dict.fromkeys(stations)
# occup_weekend_averages = dict.fromkeys(stations)


# glob_week_averages = pd.DataFrame({'timeslot': list(range(96))})
# glob_week_averages['avg_value'] = 0.0 

# glob_weekend_averages = pd.DataFrame({'timeslot': list(range(96))})
# glob_weekend_averages['avg_value'] = 0.0 


# for stat_name in stations:
    
#     backbone_load = pd.DataFrame({'date_time': pd.date_range(start, end, freq="15min")})
#     backbone_load.set_index('date_time')
#     backbone_load['timeslot'] = backbone_load['date_time'].apply(conv_timestamp)
#     backbone_load['day_index'] = backbone_load['date_time'].apply(get_day_index)
#     backbone_load['weekend'] = backbone_load['date_time'].apply(get_weekend)
#     backbone_load['value'] = 0.0
    
    
#     backbone_occup = pd.DataFrame({'date_time': pd.date_range(start, end, freq="15min")})
#     backbone_occup.set_index('date_time')
#     backbone_occup['timeslot'] = backbone_occup['date_time'].apply(conv_timestamp)
#     backbone_occup['day_index'] = backbone_occup['date_time'].apply(get_day_index)
#     backbone_occup['weekend'] = backbone_occup['date_time'].apply(get_weekend) 
#     backbone_occup['value'] = 0        
    
#     stat_backbones[stat_name] = [backbone_load, backbone_occup]
    
    
# for stat_name in stations:

#     df_pk_stat = df_pk[df_pk['CP ID'] == stat_name]        
#     df_pk.apply(lambda row: add_to_backbones(row, stat_name), axis=1)        
#     occup_avg_weekday = pd.DataFrame({'timeslot': list(range(96))})
#     occup_avg_weekday['avg_value'] = 0.0      
#     occup_avg_weekend = pd.DataFrame({'timeslot': list(range(96))})
#     occup_avg_weekend['avg_value'] = 0.0      
    
#     load_avg_weekday = pd.DataFrame({'timeslot': list(range(96))})
#     load_avg_weekday['avg_value'] = 0.0      
#     load_avg_weekend = pd.DataFrame({'timeslot': list(range(96))})
#     load_avg_weekend['avg_value'] = 0.0      
    
    
#     backbone_load = stat_backbones[stat_name][0]
#     load_weekday_averages[stat_name] = load_avg_weekday
#     load_weekend_averages[stat_name] = load_avg_weekend
    
#     for i in range(96): 
#         avg_value_load = backbone_load[(backbone_load.timeslot == i) & (backbone_load.weekend == 0)].value.sum()
#         load_avg_weekday.loc[load_avg_weekday['timeslot'] == i, 'avg_value']  = avg_value_load
#     for i in range(96): 
#         load_avg_weekday.loc[load_avg_weekday['timeslot'] == i, 'avg_value'] /= len(backbone_load[(backbone_load.timeslot == i) & (backbone_load.weekend == 0)])
#     load_weekday_averages[stat_name] = load_avg_weekday
    
#     for i in range(96): 
#         avg_value_load = backbone_load[(backbone_load.timeslot == i) & (backbone_load.weekend == 1)].value.sum()
#         load_avg_weekend.loc[load_avg_weekend['timeslot'] == i, 'avg_value']  = avg_value_load
#     for i in range(96): 
#         load_avg_weekend.loc[load_avg_weekend['timeslot'] == i, 'avg_value'] /= len(backbone_load[(backbone_load.timeslot == i) & (backbone_load.weekend == 1)])
#     load_weekend_averages[stat_name] = load_avg_weekend
    
    
#     backbone_occup = stat_backbones[stat_name][1]
    
#     for i in range(96):    
#         avg_value_occup = len(backbone_occup[(backbone_occup.timeslot == i) & (backbone_occup.value == 1) & (backbone_occup.weekend == 0)]) / len(backbone_occup[(backbone_occup.timeslot == i) & (backbone_occup.weekend == 0)])
#         occup_avg_weekday.loc[occup_avg_weekday['timeslot'] == i, 'avg_value'] = avg_value_occup
        
#         # glob_week_averages.loc[glob_week_averages['timeslot'] == i, 'avg_value_occup'] += avg_value_occup 
#     occup_weekday_averages[stat_name] = occup_avg_weekday
        
#     for i in range(96):    
#         avg_value_occup = len(backbone_occup[(backbone_occup.timeslot == i) & (backbone_occup.value == 1) & (backbone_occup.weekend == 1)]) / len(backbone_occup[(backbone_occup.timeslot == i) & (backbone_occup.weekend == 1)])        
#         occup_avg_weekend.loc[occup_avg_weekend['timeslot'] == i, 'avg_value'] = avg_value_occup
        
#         # glob_weekend_averages.loc[glob_weekend_averages['timeslot'] == i, 'avg_value'] += avg_value  
#     occup_weekend_averages[stat_name] = occup_avg_weekend
    
#     print(stat_name)
    
    
    
# print('\n')
# backbone_num = 1

# for stat_name in stations:
    
#    backbone_load = stat_backbones[stat_name][0]
   
#    backbone_load_weekday = backbone_load[backbone_load['weekend'] == 0]
#    load_avg_weekday = load_weekday_averages[stat_name].T.drop(labels='timeslot', axis=0)
#    load_weekday_data = pd.concat([load_avg_weekday]*len(backbone_load_weekday), ignore_index=True)
#    backbone_load_weekday = pd.concat([backbone_load_weekday.reset_index(drop=True), load_weekday_data.reset_index(drop=True)], axis=1)
   
#    backbone_load_weekend = backbone_load[backbone_load['weekend'] == 1]
#    load_avg_weekend = load_weekend_averages[stat_name].T.drop(labels='timeslot', axis=0)
#    load_weekend_data = pd.concat([load_avg_weekend]*len(backbone_load_weekend), ignore_index=True)
#    backbone_load_weekend = pd.concat([backbone_load_weekend.reset_index(drop=True), load_weekend_data.reset_index(drop=True)], axis=1)
   
#    sorted_backbone_load = pd.concat([backbone_load_weekday, backbone_load_weekend]).sort_values(by=['date_time'], ascending=True)
#    sorted_backbone_load['value_shifted'] = np.roll(sorted_backbone_load['value'],-1)
#    sorted_backbone_load.drop('date_time', axis=1, inplace=True)
   
#    file_name_load = "/home/doktormatte/MA_SciComp/Perth_Kinross/Loads/" + str(backbone_num) + ".csv"
#    # file_name_load = "/home/doktormatte/MA_SciComp/Boulder/Loads/" + stat_name.replace('/', '') + ".csv"
#    sorted_backbone_load.to_csv(file_name_load, encoding='utf-8', index=False, header=False)
   
   
    
    
#    backbone_occup = stat_backbones[stat_name][1]
   
#    backbone_occup_weekday = backbone_occup[backbone_occup['weekend'] == 0]
#    occup_avg_weekday = occup_weekday_averages[stat_name].T.drop(labels='timeslot', axis=0)
#    occup_weekday_data = pd.concat([occup_avg_weekday]*len(backbone_occup_weekday), ignore_index=True)
#    backbone_occup_weekday = pd.concat([backbone_occup_weekday.reset_index(drop=True), occup_weekday_data.reset_index(drop=True)], axis=1)
   
#    backbone_occup_weekend = backbone_occup[backbone_occup['weekend'] == 1]
#    occup_avg_weekend = occup_weekend_averages[stat_name].T.drop(labels='timeslot', axis=0)
#    occup_weekend_data = pd.concat([occup_avg_weekend]*len(backbone_occup_weekend), ignore_index=True)
#    backbone_occup_weekend = pd.concat([backbone_occup_weekend.reset_index(drop=True), occup_weekend_data.reset_index(drop=True)], axis=1)
   
#    sorted_backbone_occup = pd.concat([backbone_occup_weekday, backbone_occup_weekend]).sort_values(by=['date_time'], ascending=True)
#    sorted_backbone_occup['value_shifted'] = np.roll(sorted_backbone_occup['value'],-1)
#    sorted_backbone_occup.drop('date_time', axis=1, inplace=True)
   
   
   
#    file_name_occup = "/home/doktormatte/MA_SciComp/Perth_Kinross/Occup/" + str(backbone_num) + ".csv"
   
#    # file_name_occup = "/home/doktormatte/MA_SciComp/Boulder/Occup/" + stat_name.replace('/', '') + ".csv"
#    sorted_backbone_occup.to_csv(file_name_occup, encoding='utf-8', index=False, header=False)
   
   
#    backbone_num += 1  
   
#    print(stat_name)


