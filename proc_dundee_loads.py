import datetime
import pandas as pd
import numpy as np
from scipy import stats
import sys 

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

def get_day_index(ts):
    idx = ts.weekday()
    if idx == 6:
        return 0
    return idx+1

def get_weekend(ts):
    if ts.weekday() > 4:
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


   

   

df_dundee = pd.read_csv (r'/home/doktormatte/Dropbox/Dokumente/Studium/MA_SciComp/Data/Dundee_all.csv')
exclude_stations = [51548,51547,51549,51550,50912,50914,50913,50262]
rapid_chargers = ["APT 50kW Raption", "APT Triple Rapid Charger", "APT Dual Rapid Charger"]

print(df_dundee.shape)

df_dundee = df_dundee[df_dundee['CP ID'].isin(exclude_stations) == False]
df_dundee = df_dundee[df_dundee['Unnamed: 15'].isin(rapid_chargers) == True]
df_dundee = df_dundee[df_dundee['Total kWh'].notna()]
df_dundee = df_dundee[df_dundee['Total kWh'] > 0.0]

df_dundee['Start_Timestamp'] = df_dundee.apply(lambda row: create_start_timestamp(row), axis=1)
df_dundee['End_Timestamp'] = df_dundee.apply(lambda row: create_end_timestamp(row), axis=1)

cutoff = datetime.datetime.strptime("2017-02-11 00:00", '%Y-%m-%d %H:%M')
df_dundee['Start_Timestamp'] = df_dundee['Start_Timestamp'].map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M'))
df_dundee['End_Timestamp'] = df_dundee['End_Timestamp'].map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M'))
df_dundee = df_dundee[df_dundee['Start_Timestamp'] >= cutoff]



df_dundee['Start_Timestamp'] = df_dundee['Start_Timestamp'].dt.to_pydatetime()
df_dundee['End_Timestamp'] = df_dundee['End_Timestamp'].dt.to_pydatetime()


df_dundee = df_dundee[df_dundee.Start_Timestamp < df_dundee.End_Timestamp]

df_dundee['Total_Duration'] = df_dundee.apply(lambda row: calc_total_dur(row), axis=1)

df_dundee['z_score_dur'] = np.abs(stats.zscore(df_dundee['Total_Duration']))
df_dundee = df_dundee[df_dundee.z_score_dur <= 3.0]

df_dundee['Session_Duration'] = df_dundee.apply(lambda row: calc_session_dur(row), axis=1)
df_dundee['Energy_per_min'] = df_dundee.apply(lambda row: calc_energy_per_min(row), axis=1)

df_dundee['z_score_ene'] = np.abs(stats.zscore(df_dundee['Energy_per_min']))
df_dundee = df_dundee[df_dundee.z_score_ene <= 3.0]


# sys.exit()

# df_dundee['Load_per_quarter'] = df_dundee.apply(lambda row: calc_quarter_load(row), axis=1)

stations = list(set(list(df_dundee['CP ID'])))
stations = [50338.0]
start = cutoff
end = datetime.datetime.strptime("2017-03-31 00:00", '%Y-%m-%d %H:%M')
# end = df_dundee['End_Timestamp'].max()

df_dundee[df_dundee['CP ID'] == 50338.0].to_csv('/home/doktormatte/MA_SciComp/test_loads.csv', encoding='utf-8',header=list(df_dundee))
# sys.exit()  


stat_backbones = dict.fromkeys(stations)
load_weekday_averages = dict.fromkeys(stations)
load_weekend_averages = dict.fromkeys(stations)

glob_week_averages = pd.DataFrame({'timeslot': list(range(96))})
glob_week_averages['avg_value'] = 0.0 

glob_weekend_averages = pd.DataFrame({'timeslot': list(range(96))})
glob_weekend_averages['avg_value'] = 0.0 




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

    df_dundee_stat = df_dundee[df_dundee['CP ID'] == stat_name]        
    df_dundee_stat.apply(lambda row: add_to_backbones(row, stat_name), axis=1)  
    


for stat_name in stations:
    
    backbone_load_min = stat_backbones[stat_name][0]
    values_load_min = backbone_load_min['value']     
    backbone_load_quarter = pd.DataFrame({'date_time': pd.date_range(start, end + datetime.timedelta(minutes=15), freq="15min")})
    backbone_load_quarter = backbone_load_quarter.loc[:int(len(values_load_min)//15)-1,:]
    backbone_load_quarter.set_index('date_time')
    backbone_load_quarter['timeslot'] = backbone_load_quarter['date_time'].apply(conv_timestamp)
    backbone_load_quarter['day_index'] = backbone_load_quarter['date_time'].apply(get_day_index)
    backbone_load_quarter['weekend'] = backbone_load_quarter['date_time'].apply(get_weekend)
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
    backbone_occup_min.to_csv('/home/doktormatte/MA_SciComp/test_occup.csv', encoding='utf-8')
    values_occup_min = backbone_occup_min['value'] 
    backbone_occup_quarter = pd.DataFrame({'date_time': pd.date_range(start, end + datetime.timedelta(minutes=15), freq="15min")})
    backbone_occup_quarter = backbone_occup_quarter.loc[:int(len(values_load_min)//15)-1,:]
    backbone_occup_quarter.set_index('date_time')
    backbone_occup_quarter['timeslot'] = backbone_occup_quarter['date_time'].apply(conv_timestamp)
    backbone_occup_quarter['day_index'] = backbone_occup_quarter['date_time'].apply(get_day_index)
    backbone_occup_quarter['weekend'] = backbone_occup_quarter['date_time'].apply(get_weekend)
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
    
    
    backbone_occup_quarter.to_csv('/home/doktormatte/MA_SciComp/test_quarter_occup.csv', encoding='utf-8')

sys.exit()  

for stat_name in stations:
    
    load_avg_weekday = pd.DataFrame({'timeslot': list(range(96))})
    load_avg_weekday['avg_value'] = 0.0      
    load_avg_weekend = pd.DataFrame({'timeslot': list(range(96))})
    load_avg_weekend['avg_value'] = 0.0      
    
    
    backbone_load = stat_backbones[stat_name][0]
    load_weekday_averages[stat_name] = load_avg_weekday
    load_weekend_averages[stat_name] = load_avg_weekend
    
    for i in range(96): 
        avg_value_load = backbone_load[(backbone_load.timeslot == i) & (backbone_load.weekend == 0)].value.sum()
        load_avg_weekday.loc[load_avg_weekday['timeslot'] == i, 'avg_value']  = avg_value_load
    for i in range(96): 
        load_avg_weekday.loc[load_avg_weekday['timeslot'] == i, 'avg_value'] /= len(backbone_load[(backbone_load.timeslot == i) & (backbone_load.weekend == 0)])
    load_weekday_averages[stat_name] = load_avg_weekday
    
    for i in range(96): 
        avg_value_load = backbone_load[(backbone_load.timeslot == i) & (backbone_load.weekend == 1)].value.sum()
        load_avg_weekend.loc[load_avg_weekend['timeslot'] == i, 'avg_value']  = avg_value_load
    for i in range(96): 
        load_avg_weekend.loc[load_avg_weekend['timeslot'] == i, 'avg_value'] /= len(backbone_load[(backbone_load.timeslot == i) & (backbone_load.weekend == 1)])
    load_weekend_averages[stat_name] = load_avg_weekend   
    print(stat_name)
    
    
sys.exit()    
print('\n')
backbone_num = 1

for stat_name in stations:
    
   backbone_load = stat_backbones[stat_name][0]
   
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
   
   file_name_load = "/home/doktormatte/MA_SciComp/Dundee/Loads/" + str(backbone_num) + ".csv"
   # file_name_load = "/home/doktormatte/MA_SciComp/Boulder/Loads/" + stat_name.replace('/', '') + ".csv"
   sorted_backbone_load.to_csv(file_name_load, encoding='utf-8', index=False, header=False)   
   backbone_num += 1  
   
   print(stat_name)


