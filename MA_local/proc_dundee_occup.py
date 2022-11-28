import datetime
import pandas as pd
import numpy as np

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
    dur = row['Charging Time (hh:mm:ss)']
    total = row['Energy (kWh)']
    return (total/dur)*15.0

def calc_sessions_dur(row):
    return (row['Total kWh']/50.0)*60.0

def calc_minute_load(row):
    return row['Total kWh']/row['Session_Duration']

def calc_total_dur(row):
    dur = (row['End_Timestamp'] - row['Start_Timestamp']).total_seconds()
    # if dur < 0:
    #     print(row['Start_Timestamp'])
    #     print(row['End_Timestamp'])  
    #     print(dur)
    #     print("\n")
    return round(dur/60)
    


def add_to_backbone(row):
    iters = int(row['Session_Duration'])
    load_per_min = 0.8333333333333334
    for i in range(iters):
        backbone.loc[backbone['date_time'] == row['Start_Timestamp'] + datetime.timedelta(minutes=i), 'value'] += load_per_min    

   

dfdundee = pd.read_csv (r'/home/doktormatte/Dropbox/Dokumente/Studium/MA_SciComp/Data/Dundee_all.csv')
exclude_stations = [51548,51547,51549,51550,50912,50914,50913,50262]
rapid_chargers = ["APT 50kW Raption", "APT Triple Rapid Charger", "APT Dual Rapid Charger"]

print(dfdundee.shape)

dfdundee = dfdundee[dfdundee['CP ID'].isin(exclude_stations) == False]
dfdundee = dfdundee[dfdundee['Unnamed: 15'].isin(rapid_chargers) == True]
dfdundee = dfdundee[dfdundee['Total kWh'].notna()]

dfdundee['Start_Timestamp'] = dfdundee.apply(lambda row: create_start_timestamp(row), axis=1)
dfdundee['End_Timestamp'] = dfdundee.apply(lambda row: create_end_timestamp(row), axis=1)


dfdundee['Session_Duration'] = dfdundee.apply(lambda row: calc_sessions_dur(row), axis=1)
dfdundee['Session_Duration'] = round(dfdundee['Session_Duration'])
dfdundee = dfdundee[dfdundee['Session_Duration'] > 0.0]
dfdundee['Minute_Load'] = dfdundee.apply(lambda row: calc_minute_load(row), axis=1)

cutoff = datetime.datetime.strptime("2017-02-11 00:00", '%Y-%m-%d %H:%M')
dfdundee['Start_Timestamp'] = dfdundee['Start_Timestamp'].map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M'))
dfdundee['End_Timestamp'] = dfdundee['End_Timestamp'].map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M'))
dfdundee = dfdundee[dfdundee['Start_Timestamp'] >= cutoff]

dfdundee['Total_Duration'] = dfdundee.apply(lambda row: calc_total_dur(row), axis=1)
dfdundee = dfdundee[dfdundee['Total_Duration'] > 0.0]

# start = dfdundee['Start_Timestamp'].min()
# end = dfdundee['End_Timestamp'].max()
# backbone = pd.DataFrame({'date_time': pd.date_range(start, end, freq="1min")})
# backbone.set_index('date_time')
# backbone['value'] = 0.0

print(dfdundee.shape)




# dfdundee.apply(lambda row: add_to_backbone(row), axis=1)

