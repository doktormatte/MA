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

def conv_timestamp(ts):
    time_arr = str(ts)[-8:].split(':')
    hours = int(time_arr[0])
    mins = int(time_arr[1])
    return (hours*60+mins)//15

def get_weekend(ts):
    if ts.weekday() > 4:
        return 1
    return 0

def get_weekday(ts):
    return ts.weekday()
    

def calc_quarter_load(row):
    dur = row['Charging Time (hh:mm:ss)']
    total = row['Energy (kWh)']
    return (total/dur)*15.0


def add_to_backbone(row):
    delta = row['End Date'] - row['Start Date']
    iters = int(delta.total_seconds()//60)//15    
    for i in range(iters):
        backbone.loc[backbone['date_time'] == row['Start Date'] + datetime.timedelta(minutes=15*i), 'value'] = 1   
        

def add_to_avg_weekday(row):    
    
    # thought: no need to iterate !
    for i in range(96):
        avg_weekday.loc[avg_weekday['timeslot'] == i, 'avg_value'] += 1
    

   

df = pd.read_csv (r'/home/doktormatte/Dropbox/Dokumente/Studium/MA_SciComp/Data/ElectricVehicleChargingStationUsageJuly2011Dec2020_PaloAlto.csv')
exclude_stations = ["PALO ALTO CA / TED THOMPSON #4","PALO ALTO CA / TED THOMPSON #3","PALO ALTO CA / CAMBRIDGE #5","PALO ALTO CA / TED THOMPSON #2","PALO ALTO CA / CAMBRIDGE #3","PALO ALTO CA / CAMBRIDGE #4","PALO ALTO CA / BRYANT # 1","PALO ALTO CA / SHERMAN 6","PALO ALTO CA / SHERMAN 7","PALO ALTO CA / SHERMAN 9","PALO ALTO CA / SHERMAN 8","PALO ALTO CA / SHERMAN 4","PALO ALTO CA / SHERMAN 1","PALO ALTO CA / SHERMAN 3","PALO ALTO CA / SHERMAN 14","PALO ALTO CA / SHERMAN 2","PALO ALTO CA / SHERMAN 5","PALO ALTO CA / SHERMAN 15","PALO ALTO CA / SHERMAN 11","PALO ALTO CA / SHERMAN 17"]

print(df.shape)

df = df[df['Station Name'].isin(exclude_stations) == False]
df = df[df['Start Date'].notna()]
df = df[df['End Date'].notna()]
df = df[df['Charging Time (hh:mm:ss)'].notna()]

print(df.shape)



cutoff = datetime.datetime.strptime("8/1/2017 00:00", '%m/%d/%Y %H:%M')
df['Start Date'] = df['Start Date'].map(lambda x: datetime.datetime.strptime(x, '%m/%d/%Y %H:%M'))

df = df[df['End Date'].str.len() > 11]

df['End Date'] = df['End Date'].map(lambda x: datetime.datetime.strptime(x, '%m/%d/%Y %H:%M'))
df = df[df['Start Date'] >= cutoff]
df = df[df['Energy (kWh)'] > 0.0]

stations = set(list(df['Station Name']))


##### begin: this needs to be in loop


for stat_name in stations:
    
    df_copy = df.copy(deep=True)

    #stat_name = 'PALO ALTO CA / MPL #1'
    df_copy = df_copy[df_copy['Station Name'] == stat_name]
    
    df_copy['Start Date'] = df_copy['Start Date'].dt.to_pydatetime()
    df_copy['Start Date'] = df_copy['Start Date'].map(roundTime)
    
    df_copy['End Date'] = df_copy['End Date'].dt.to_pydatetime()
    df_copy['End Date'] = df_copy['End Date'].map(roundTime)
    
    # df_copy = df_copy[df_copy['Start Date'].notna()]
    # df_copy = df_copy[df_copy['End Date'].notna()]
    
    start = df_copy['Start Date'].min()
    end = df_copy['End Date'].max()
    
    print(start)
    print(end)
    
    backbone = pd.DataFrame({'date_time': pd.date_range(start, end, freq="15min")})
    backbone.set_index('date_time')
    backbone['value'] = 0
    backbone['timeslot'] = 0
    
    df_copy.apply(lambda row: add_to_backbone(row), axis=1)
    backbone['timeslot'] = backbone['date_time'].apply(conv_timestamp)
    backbone['weekend'] = backbone['date_time'].apply(get_weekend)
    backbone['weekday'] = backbone['date_time'].apply(get_weekday)
    
    weekday_backbone = backbone[backbone['weekend'] == 0]
    weekend_backbone = backbone[backbone['weekend'] == 1]
    
    avg_weekday = pd.DataFrame({'timeslot': list(range(96))})
    avg_weekday['avg_value'] = 0.0  
    for i in range(96):
        total_occur = weekday_backbone[weekday_backbone.timeslot == i].shape[0]
        occup_occur = weekday_backbone[(weekday_backbone.timeslot == i) & (weekday_backbone.value == 1)].shape[0]
        avg_value = occup_occur/total_occur
        avg_weekday.loc[avg_weekday['timeslot'] == i, 'avg_value'] = avg_value     
    
    
    avg_weekend = pd.DataFrame({'timeslot': list(range(96))})
    avg_weekend['avg_value'] = 0.0
    for i in range(96):
        total_occur = weekend_backbone[weekend_backbone.timeslot == i].shape[0]
        occup_occur = weekend_backbone[(weekend_backbone.timeslot == i) & (weekend_backbone.value == 1)].shape[0]
        avg_value = occup_occur/total_occur
        avg_weekend.loc[avg_weekend['timeslot'] == i, 'avg_value'] = avg_value 




##### end: this needs to be in loop





# for i in range(len(df['End Date'])):    
#     try:
#         datetime.datetime.strptime(df['End Date'][i], '%m/%d/%Y %H:%M')
#     except Exception:
#         print(df['End Date'][i])

# print(df.shape)

# df['Charging Time (hh:mm:ss)'] = df['Charging Time (hh:mm:ss)'].map(lambda x: x.split(':'))
# df['Charging Time (hh:mm:ss)'] = df['Charging Time (hh:mm:ss)'].map(conv_entries)
# df = df[df['Charging Time (hh:mm:ss)'] > 0]
# df['Load_per_quarter'] = df.apply(lambda row: calc_quarter_load(row), axis=1)



# backbone = pd.DataFrame({'date_time': pd.date_range(start, end, freq="15min")})
# backbone.set_index('date_time')
# backbone['value'] = 0.0

#df.apply(lambda row: add_to_backbone(row), axis=1)

