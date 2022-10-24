import datetime
import pandas as pd
import numpy as np
from scipy import stats

def roundTime(dt=None):    
    dt = dt.to_pydatetime()
    roundTo = 15*60    
    if dt == None : dt = datetime.datetime.now()
    seconds = (dt.replace(tzinfo=None) - dt.min).seconds
    rounding = (seconds+roundTo/2) // roundTo * roundTo
    return dt + datetime.timedelta(0,rounding-seconds,-dt.microsecond)


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
    dur = row['Charging Time (hh:mm:ss)']
    total = row['Energy (kWh)']
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


def check_time_format(x):
    try:
        datetime.datetime.strptime(x, '%m/%d/%Y %H:%M')        
        return True
    except Exception:
        return False


def add_to_backbones(row, stat_name):
    
    iters_load = int(row['Charging Time (hh:mm:ss)']//15)


    backbone_load = stat_backbones[stat_name][0]
    for i in range(iters_load):
        backbone_load.loc[backbone_load['date_time'] == row['Start Date'] + datetime.timedelta(minutes=15*i), 'value'] += row['Load_per_quarter']     
    
    delta = row['End Date'] - row['Start Date']
    iters_occup = int(round(delta.total_seconds()/60.0)/15.0)
    
    backbone_occup = stat_backbones[stat_name][1]
    for i in range(iters_occup):
        backbone_occup.loc[backbone_occup['date_time'] == row['Start Date'] + datetime.timedelta(minutes=15*i), 'value'] = 1.0   


def add_to_backbone(row):
    iters = row['Charging Time (hh:mm:ss)']//15
    for i in range(iters):
        backbone.loc[backbone['date_time'] == row['Start Date'] + datetime.timedelta(minutes=15*i), 'value'] += row['Load_per_quarter']     

   

df_palo_alto = pd.read_csv (r'/home/doktormatte/Dropbox/Dokumente/Studium/MA_SciComp/Data/ElectricVehicleChargingStationUsageJuly2011Dec2020_PaloAlto.csv')
exclude_stations = ["PALO ALTO CA / TED THOMPSON #4","PALO ALTO CA / TED THOMPSON #3","PALO ALTO CA / CAMBRIDGE #5","PALO ALTO CA / TED THOMPSON #2","PALO ALTO CA / CAMBRIDGE #3","PALO ALTO CA / CAMBRIDGE #4","PALO ALTO CA / BRYANT # 1","PALO ALTO CA / SHERMAN 6","PALO ALTO CA / SHERMAN 7","PALO ALTO CA / SHERMAN 9","PALO ALTO CA / SHERMAN 8","PALO ALTO CA / SHERMAN 4","PALO ALTO CA / SHERMAN 1","PALO ALTO CA / SHERMAN 3","PALO ALTO CA / SHERMAN 14","PALO ALTO CA / SHERMAN 2","PALO ALTO CA / SHERMAN 5","PALO ALTO CA / SHERMAN 15","PALO ALTO CA / SHERMAN 11","PALO ALTO CA / SHERMAN 17"]

print(df_palo_alto.shape)

df_palo_alto = df_palo_alto[df_palo_alto['Station Name'].isin(exclude_stations) == False]


mode = 'rapid'

if mode == "rapid":
    df_palo_alto = df_palo_alto[df_palo_alto['Plug Type'] == 'J1772']
else: 
    df_palo_alto = df_palo_alto[df_palo_alto['Plug Type'] == 'NEMA 5-20R']


df_palo_alto = df_palo_alto[df_palo_alto['Start Date'].notna()]
df_palo_alto = df_palo_alto[df_palo_alto['End Date'].notna()]
df_palo_alto = df_palo_alto[df_palo_alto['Charging Time (hh:mm:ss)'].notna()]

print(df_palo_alto.shape)



cutoff = datetime.datetime.strptime("8/1/2017 00:00", '%m/%d/%Y %H:%M')
df_palo_alto['Start Date'] = df_palo_alto['Start Date'].map(lambda x: datetime.datetime.strptime(x, '%m/%d/%Y %H:%M'))


df_palo_alto = df_palo_alto[df_palo_alto['End Date'].apply(check_time_format) == True]

df_palo_alto['End Date'] = df_palo_alto['End Date'].map(lambda x: datetime.datetime.strptime(x, '%m/%d/%Y %H:%M'))
# df_palo_alto.End_Date___Time = df_palo_alto.End_Date___Time.map(lambda x: datetime.datetime.strptime(x, '%Y/%m/%d %H:%M:%S+%f'))
df_palo_alto = df_palo_alto[df_palo_alto['Start Date'] >= cutoff]
df_palo_alto = df_palo_alto[df_palo_alto['Energy (kWh)'] > 0.0]


df_palo_alto['Charging Time (hh:mm:ss)'] = df_palo_alto['Charging Time (hh:mm:ss)'].map(lambda x: x.split(':'))
df_palo_alto['Charging Time (hh:mm:ss)'] = df_palo_alto['Charging Time (hh:mm:ss)'].map(conv_entries)
df_palo_alto = df_palo_alto[df_palo_alto['Charging Time (hh:mm:ss)'] > 0]
df_palo_alto['z_score'] = np.abs(stats.zscore(df_palo_alto['Charging Time (hh:mm:ss)']))
df_palo_alto = df_palo_alto[df_palo_alto.z_score <= 3.0]

df_palo_alto['Total Duration (hh:mm:ss)'] = df_palo_alto['Total Duration (hh:mm:ss)'].map(lambda x: x.split(':'))
df_palo_alto['Total Duration (hh:mm:ss)'] = df_palo_alto['Total Duration (hh:mm:ss)'].map(conv_entries)
df_palo_alto = df_palo_alto[df_palo_alto['Total Duration (hh:mm:ss)'] > 0]


df_palo_alto['Start Date'] = df_palo_alto['Start Date'].dt.to_pydatetime()
df_palo_alto['Start Date'] = df_palo_alto['Start Date'].map(roundTime)

df_palo_alto['End Date'] = df_palo_alto['End Date'].dt.to_pydatetime()
df_palo_alto['End Date'] = df_palo_alto['End Date'].map(roundTime)


df_palo_alto['Load_per_quarter'] = df_palo_alto.apply(lambda row: calc_quarter_load(row), axis=1)


stations = list(set(list(df_palo_alto['Station Name'])))


# start = df_palo_alto['Start Date'].min()
start = cutoff
end = df_palo_alto['End Date'].max()


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
    backbone_load['day_index'] = backbone_load['date_time'].apply(get_day_index)
    backbone_load['weekend'] = backbone_load['date_time'].apply(get_weekend)
    backbone_load['value'] = 0.0
    
    
    backbone_occup = pd.DataFrame({'date_time': pd.date_range(start, end, freq="15min")})
    backbone_occup.set_index('date_time')
    backbone_occup['timeslot'] = backbone_occup['date_time'].apply(conv_timestamp)
    backbone_occup['day_index'] = backbone_occup['date_time'].apply(get_day_index)
    backbone_occup['weekend'] = backbone_occup['date_time'].apply(get_weekend) 
    backbone_occup['value'] = 0        
    
    stat_backbones[stat_name] = [backbone_load, backbone_occup]
    
    
for stat_name in stations:

    df_palo_alto_stat = df_palo_alto[df_palo_alto['Station Name'] == stat_name]        
    df_palo_alto_stat.apply(lambda row: add_to_backbones(row, stat_name), axis=1)        
    occup_avg_weekday = pd.DataFrame({'timeslot': list(range(96))})
    occup_avg_weekday['avg_value'] = 0.0      
    occup_avg_weekend = pd.DataFrame({'timeslot': list(range(96))})
    occup_avg_weekend['avg_value'] = 0.0      
    
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
    
    
    backbone_occup = stat_backbones[stat_name][1]
    
    for i in range(96):    
        avg_value_occup = len(backbone_occup[(backbone_occup.timeslot == i) & (backbone_occup.value == 1) & (backbone_occup.weekend == 0)]) / len(backbone_occup[(backbone_occup.timeslot == i) & (backbone_occup.weekend == 0)])
        occup_avg_weekday.loc[occup_avg_weekday['timeslot'] == i, 'avg_value'] = avg_value_occup
        
        # glob_week_averages.loc[glob_week_averages['timeslot'] == i, 'avg_value_occup'] += avg_value_occup 
    occup_weekday_averages[stat_name] = occup_avg_weekday
        
    for i in range(96):    
        avg_value_occup = len(backbone_occup[(backbone_occup.timeslot == i) & (backbone_occup.value == 1) & (backbone_occup.weekend == 1)]) / len(backbone_occup[(backbone_occup.timeslot == i) & (backbone_occup.weekend == 1)])        
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
   
   file_name_load = "/home/doktormatte/MA_SciComp/Palo_Alto/Loads/" + str(backbone_num) + ".csv"
   # file_name_load = "/home/doktormatte/MA_SciComp/Boulder/Loads/" + stat_name.replace('/', '') + ".csv"
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
   
   
   
   file_name_occup = "/home/doktormatte/MA_SciComp/Palo_Alto/Occup/" + str(backbone_num) + ".csv"
   
   # file_name_occup = "/home/doktormatte/MA_SciComp/Boulder/Occup/" + stat_name.replace('/', '') + ".csv"
   sorted_backbone_occup.to_csv(file_name_occup, encoding='utf-8', index=False, header=False)
   
   
   backbone_num += 1  
   
   print(stat_name)




# backbone = pd.DataFrame({'date_time': pd.date_range(start, end, freq="15min")})
# backbone.set_index('date_time')
# backbone['value'] = 0.0
# backbone['sessions'] = 0

#df_palo_alto.apply(lambda row: add_to_backbone(row), axis=1)

