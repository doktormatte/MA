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
    dur = row['Charging_Time__hh_mm_ss_']
    total = row['Energy__kWh_']
    return (total/dur)*15.0


def add_to_backbone(row):
    iters = row['Charging_Time__hh_mm_ss_']//15
    for i in range(iters):
        backbone.loc[backbone['date_time'] == row['Start_Date___Time'] + datetime.timedelta(minutes=15*i), 'value'] += row['Load_per_quarter']     

   

dfpk = pd.read_csv (r'/home/doktormatte/Dropbox/Dokumente/Studium/MA_SciComp/Data/PerthKinross_2016-2019.csv')

# read boulder charging data
dfboulder = pd.read_csv (r'/home/doktormatte/Dropbox/Dokumente/Studium/MA_SciComp/Data/Electric_Vehicle_Charging_Station_Energy_Consumption_Boulder.csv')  

# remove charging stations that were activated after cutoff
exclude_stations_boulder = ["BOULDER / EAST REC","BOULDERJUNCTION / JUNCTION ST1","BOULDER / RESERVOIR ST1","BOULDER / RESERVOIR ST2","BOULDER / CARPENTER PARK1","BOULDER / CARPENTER PARK2","BOULDER / AIRPORT ST1","COMM VITALITY / 5050 PEARL 1","BOULDER / VALMONT ST2","BOULDER / VALMONT ST1"]
dfboulder = dfboulder[dfboulder.Station_Name.isin(exclude_stations_boulder) == False]

# remove entries with empty start and end timestamps
dfboulder = dfboulder[dfboulder['Start_Date___Time'].notna()]
dfboulder = dfboulder[dfboulder['End_Date___Time'].notna()]

# transform timestamp strings to timestamps and apply cutoff
dfboulder.Start_Date___Time = dfboulder.Start_Date___Time.map(lambda x: datetime.datetime.strptime(x, '%Y/%m/%d %H:%M:%S+%f'))
dfboulder.End_Date___Time = dfboulder.End_Date___Time.map(lambda x: datetime.datetime.strptime(x, '%Y/%m/%d %H:%M:%S+%f'))
cutoff = datetime.datetime.strptime("2019/01/01 00:00:00+00", '%Y/%m/%d %H:%M:%S+%f')
dfboulder = dfboulder[dfboulder.Start_Date___Time >= cutoff]

# only consider sessions during which energy was used
dfboulder = dfboulder[dfboulder.Energy__kWh_ > 0.0]

# transform charging time into minutes total and round up or down to next quarter hour
dfboulder.Charging_Time__hh_mm_ss_ = dfboulder.Charging_Time__hh_mm_ss_.map(lambda x: x.split(':'))
dfboulder.Charging_Time__hh_mm_ss_ = dfboulder.Charging_Time__hh_mm_ss_.map(conv_entries)
dfboulder = dfboulder[dfboulder.Charging_Time__hh_mm_ss_ > 0]

# calculate charging load per quarter hour
dfboulder['Load_per_quarter'] = dfboulder.apply(lambda row: calc_quarter_load(row), axis=1)

# generate time-axis for final time series
dfboulder.Start_Date___Time = dfboulder.Start_Date___Time.dt.to_pydatetime()
dfboulder.Start_Date___Time = dfboulder.Start_Date___Time.map(roundTime)
start = dfboulder.Start_Date___Time.min()
end = dfboulder.Start_Date___Time.max()
backbone = pd.DataFrame({'date_time': pd.date_range(start, end, freq="15min")})
backbone.set_index('date_time')
backbone['value'] = 0.0

# set aggregated charging load as values of time series
dfboulder.apply(lambda row: add_to_backbone(row), axis=1)

