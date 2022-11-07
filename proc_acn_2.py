import json
import datetime
import pandas as pd
import numpy as np
from scipy import stats
from workalendar.usa import California

def roundTime(dt=None):    
    dt = dt.to_pydatetime()
    roundTo = 15*60    
    if dt == None : dt = datetime.datetime.now()
    seconds = (dt.replace(tzinfo=None) - dt.min).seconds
    rounding = (seconds+roundTo/2) // roundTo * roundTo
    return dt + datetime.timedelta(0,rounding-seconds,-dt.microsecond)


def strip_time(timestr):
    return timestr[5:-4]

def calc_charging_time(row):
    end = row['doneChargingTime']
    start = row['connectionTime']
    dur_sec = (end - start).total_seconds()
    return dur_sec/60.0

def calc_quarter_load(row):
    dur = row['sessionDuration']
    total = row['kWhDelivered']
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

def get_weekend(ts):
    if ts.weekday() > 4:
        return 1
    return 0

def get_sin(x, x_max):
    return np.sin(2.0*np.pi*x/x_max)

def get_cos(x, x_max):
    return np.cos(2.0*np.pi*x/x_max)

def get_month(ts):
    return ts.month

def get_holiday(ts):
    year = ts.year
    cal = California()
    holidays = cal.holidays(year)
    for holiday in holidays:
        if holiday[0] == datetime.date.fromtimestamp(datetime.datetime.timestamp(ts)):
            return 1
    return 0        
    

def add_to_backbones(row, stat_name):
    delta_load = row['doneChargingTime'] - row['connectionTime']
    iters_load = int(round(delta_load.total_seconds()/60.0)/15.0)

    backbone_load = stat_backbones[stat_name][0]
    for i in range(iters_load):
        backbone_load.loc[backbone_load['date_time'] == row['connectionTime'] + datetime.timedelta(minutes=15*i), 'value'] += row['loadPerQuarter']     
    
    
    delta_occup = row['disconnectTime'] - row['connectionTime']
    iters_occup = int(round(delta_occup.total_seconds()/60.0)/15.0)
    
    backbone_occup = stat_backbones[stat_name][1]
    for i in range(iters_occup):
        backbone_occup.loc[backbone_occup['date_time'] == row['connectionTime'] + datetime.timedelta(minutes=15*i), 'value'] = 1.0   


def add_to_backbone(row):
    iters = int(row['sessionDuration']//15)
    for i in range(iters):
        backbone.loc[backbone['date_time'] == row['connectionTime'] + datetime.timedelta(minutes=15*i), 'value'] += row['loadPerQuarter'] 
    
  

f = open('/home/doktormatte/Dropbox/Dokumente/Studium/MA_SciComp/Data/acndata_sessions_jpl.json')
data = json.load(f)
sites = set()
print(len(data['_items']))


df_acndata = []
columns=['_id','clusterID','connectionTime','disconnectTime','doneChargingTime','kWhDelivered','sessionID', 'siteID', 'spaceID', 'stationID', 'timezone']

for i in data['_items']:    
    df_acndata.append([i['_id'],i['clusterID'],i['connectionTime'],i['disconnectTime'],i['doneChargingTime'],i['kWhDelivered'], i['sessionID'],i['siteID'],i['spaceID'],i['stationID'],i['timezone']])
    
df_acn = pd.DataFrame(df_acndata, columns=columns)
# df_acn.to_csv("/home/doktormatte/Dropbox/Dokumente/Studium/MA_SciComp/Data/acn_jpl.csv", encoding='utf-8')


df_acn = df_acn[df_acn['connectionTime'].notna()]
df_acn = df_acn[df_acn['doneChargingTime'].notna()]

df_acn['connectionTime'] = df_acn['connectionTime'].apply(strip_time)
df_acn['doneChargingTime'] = df_acn['doneChargingTime'].apply(strip_time)
df_acn['disconnectTime'] = df_acn['disconnectTime'].apply(strip_time)

df_acn['connectionTime'] = df_acn['connectionTime'].map(lambda x: datetime.datetime.strptime(x, '%d %b %Y %H:%M:%S'))
df_acn['doneChargingTime'] = df_acn['doneChargingTime'].map(lambda x: datetime.datetime.strptime(x, '%d %b %Y %H:%M:%S'))
df_acn['disconnectTime'] = df_acn['disconnectTime'].map(lambda x: datetime.datetime.strptime(x, '%d %b %Y %H:%M:%S'))

df_acn['connectionTime'] = df_acn['connectionTime'].map(roundTime)
df_acn['doneChargingTime'] = df_acn['doneChargingTime'].map(roundTime)
df_acn['disconnectTime'] = df_acn['disconnectTime'].map(roundTime)

# exclude_stations = ['2-39-83-387', '2-39-82-384', '2-39-82-385', '2-39-81-4550']
# df_acn = df_acn[df_acn['stationID'].isin(exclude_stations) == False]

# df_acn = df_acn[df_acn['stationID'] == '1-1-178-828']



df_acn['sessionDuration'] = df_acn.apply(lambda row: calc_charging_time(row), axis=1)
df_acn = df_acn[df_acn['sessionDuration'] > 0.0]


df_acn['z_score'] = np.abs(stats.zscore(df_acn['sessionDuration']))
df_acn = df_acn[df_acn.z_score <= 3.0]

df_acn['loadPerQuarter'] = df_acn.apply(lambda row: calc_quarter_load(row), axis=1)

#df_acn.to_csv('/home/doktormatte/Dropbox/Dokumente/Studium/MA_SciComp/Data/acn.csv', encoding='utf-8', index=False)
        
# start = df_acn['connectionTime'].min()
start = datetime.datetime.strptime("9/15/2018 00:00", '%m/%d/%Y %H:%M')
end = df_acn['disconnectTime'].max()
# end = datetime.datetime.strptime("3/1/2020 00:00", '%m/%d/%Y %H:%M')

stations = list(set(list(df_acn['stationID'])))

stat_backbones = dict.fromkeys(stations)
load_weekday_averages = dict.fromkeys(stations)
load_weekend_averages = dict.fromkeys(stations)

occup_weekday_averages = dict.fromkeys(stations)
occup_weekend_averages = dict.fromkeys(stations)


glob_week_averages = pd.DataFrame({'timeslot': list(range(96))})
glob_week_averages['avg_value'] = 0.0 

glob_weekend_averages = pd.DataFrame({'timeslot': list(range(96))})
glob_weekend_averages['avg_value'] = 0.0 


for stat_name in stations:
    
    backbone_load = pd.DataFrame({'date_time': pd.date_range(start, end, freq="15min")})
    backbone_load.set_index('date_time')
    
    backbone_load['timeslot'] = backbone_load['date_time'].apply(conv_timestamp)
    max_timeslot = max(backbone_load['timeslot'])
    backbone_load['day_of_week'] = backbone_load['date_time'].apply(get_day_of_week)
    max_day_of_week = max(backbone_load['day_of_week'])
    backbone_load['day_of_month'] = backbone_load['date_time'].apply(get_day_of_month)
    max_day_of_month = max(backbone_load['day_of_month'])
    backbone_load['day_of_year'] = backbone_load['date_time'].apply(get_day_of_year)
    max_day_of_year = max(backbone_load['day_of_year'])
    backbone_load['weekend'] = backbone_load['date_time'].apply(get_weekend)
    backbone_load['holiday'] = backbone_load['date_time'].apply(get_holiday)
      
    backbone_load['timeslot_sin'] = backbone_load.apply(lambda x: get_sin(x['timeslot'], max_timeslot),axis=1)
    backbone_load['timeslot_cos'] = backbone_load.apply(lambda x: get_cos(x['timeslot'], max_timeslot),axis=1)
    backbone_load['day_of_week_sin'] = backbone_load.apply(lambda x: get_sin(x['day_of_week'], max_day_of_week),axis=1)
    backbone_load['day_of_week_cos'] = backbone_load.apply(lambda x: get_cos(x['day_of_week'], max_day_of_week),axis=1)
    backbone_load['day_of_month_sin'] = backbone_load.apply(lambda x: get_sin(x['day_of_month'], max_day_of_month),axis=1)
    backbone_load['day_of_month_cos'] = backbone_load.apply(lambda x: get_cos(x['day_of_month'], max_day_of_month),axis=1)    
    backbone_load['day_of_year_sin'] = backbone_load.apply(lambda x: get_sin(x['day_of_year'], max_day_of_year),axis=1)
    backbone_load['day_of_year_cos'] = backbone_load.apply(lambda x: get_cos(x['day_of_year'], max_day_of_year),axis=1)        
    backbone_load['value'] = 0.0  

    
    
    
    backbone_occup = pd.DataFrame({'date_time': pd.date_range(start, end, freq="15min")})
    backbone_occup.set_index('date_time')
    
    backbone_occup['timeslot'] = backbone_occup['date_time'].apply(conv_timestamp)
    max_timeslot = max(backbone_occup['timeslot'])
    backbone_occup['day_of_week'] = backbone_occup['date_time'].apply(get_day_of_week)
    max_day_of_week = max(backbone_occup['day_of_week'])
    backbone_occup['day_of_month'] = backbone_occup['date_time'].apply(get_day_of_month)
    max_day_of_month = max(backbone_occup['day_of_month'])
    backbone_occup['day_of_year'] = backbone_occup['date_time'].apply(get_day_of_year)
    max_day_of_year = max(backbone_occup['day_of_year'])
    backbone_occup['weekend'] = backbone_occup['date_time'].apply(get_weekend) 
    backbone_occup['holiday'] = backbone_occup['date_time'].apply(get_holiday)
     
    backbone_occup['timeslot_sin'] = backbone_occup.apply(lambda x: get_sin(x['timeslot'], max_timeslot),axis=1)
    backbone_occup['timeslot_cos'] = backbone_occup.apply(lambda x: get_cos(x['timeslot'], max_timeslot),axis=1)
    backbone_occup['day_of_week_sin'] = backbone_occup.apply(lambda x: get_sin(x['day_of_week'], max_day_of_week),axis=1)
    backbone_occup['day_of_week_cos'] = backbone_occup.apply(lambda x: get_cos(x['day_of_week'], max_day_of_week),axis=1)
    backbone_occup['day_of_month_sin'] = backbone_occup.apply(lambda x: get_sin(x['day_of_month'], max_day_of_month),axis=1)
    backbone_occup['day_of_month_cos'] = backbone_occup.apply(lambda x: get_cos(x['day_of_month'], max_day_of_month),axis=1)
    backbone_occup['day_of_year_sin'] = backbone_occup.apply(lambda x: get_sin(x['day_of_year'], max_day_of_year),axis=1)
    backbone_occup['day_of_year_cos'] = backbone_occup.apply(lambda x: get_cos(x['day_of_year'], max_day_of_year),axis=1)
    backbone_occup['value'] = 0             

    stat_backbones[stat_name] = [backbone_load, backbone_occup]
    
    print(stat_name)
    
print('\n')    

for stat_name in stations:

    df_acn_stat = df_acn[df_acn['stationID'] == stat_name]        
    df_acn_stat.apply(lambda row: add_to_backbones(row, stat_name), axis=1)        
    occup_avg_weekday = pd.DataFrame({'timeslot': list(range(96))})
    occup_avg_weekday['avg_value'] = 0.0      
    occup_avg_weekend = pd.DataFrame({'timeslot': list(range(96))})
    occup_avg_weekend['avg_value'] = 0.0      
    
    load_avg_weekday = pd.DataFrame({'timeslot': list(range(96))})
    load_avg_weekday['avg_value'] = 0.0      
    load_avg_weekend = pd.DataFrame({'timeslot': list(range(96))})
    load_avg_weekend['avg_value'] = 0.0    
    
    backbone_load = stat_backbones[stat_name][0]
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
    
    
    backbone_occup = stat_backbones[stat_name][1]
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
   
    file_name_load = "/home/doktormatte/MA_SciComp/ACN_2/Loads/" + str(backbone_num) + ".csv"
    # file_name_load = "/home/doktormatte/MA_SciComp/Boul der/Loads/" + stat_name.replace('/', '') + ".csv"
    sorted_backbone_load.to_csv(file_name_load, encoding='utf-8', index=False, header=False)
   
   
    
    
    backbone_occup = stat_backbones[stat_name][1]
   
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
   
   
   
    file_name_occup = "/home/doktormatte/MA_SciComp/ACN_2/Occup/" + str(backbone_num) + ".csv"
    
   
    # file_name_occup = "/home/doktormatte/MA_SciComp/Boulder/Occup/" + stat_name.replace('/', '') + ".csv"
    sorted_backbone_occup.to_csv(file_name_occup, encoding='utf-8', index=False, header=False)
   
   
    backbone_num += 1  
   
    print(stat_name)










# backbone = pd.DataFrame({'date_time': pd.date_range(start, end, freq="15min")})
# backbone.set_index('date_time')
# backbone['value'] = 0.0

#df_acn.apply(lambda row: add_to_backbone(row), axis=1)

#df_acnObj = df_acnObj.append({'User_ID': 23, 'UserName': 'Riti', 'Action': 'Login'}, ignore_index=True)