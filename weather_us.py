import requests
import json
from datetime import datetime, timezone, timedelta
import pandas as pd



start = datetime.strptime("2012/01/01", '%Y/%m/%d')
end = datetime.strptime("2022/10/11", '%Y/%m/%d')
#end = datetime.date.today()

backbone = pd.DataFrame({'date_time': pd.date_range(start, end, freq="24h")})

# url = "https://www.wunderground.com/history/daily/KBUR/date/2021-10-12"
# url = "https://www.wunderground.com/history/daily/KFNL/date/2020-01-20"

#sites = ['KBUR','KFNL','KSJC']
sites = ['KBUR']


# url = 'https://www.wunderground.com/history/daily/' + 'KBUR' + '/date/' + str(date)[:-9]
url = 'https://www.wunderground.com/history/daily/' + 'KBUR' + '/date/' + '2020-01-20'



# ts = int('1604750712')
# tz = timezone(-timedelta(hours=4))

# print(datetime.fromtimestamp(ts, tz).strftime('%Y-%m-%d %H:%M:%S'))


weatherdata = []
columns=['valid_time_gmt','day_ind','temp','wx_phrase','pressure_tend','pressure_desc','dewPt', 'heat_index', 'rh', 'pressure', 'vis', 'wc', 'wdir', 'wdir_cardinal', 'gust', 'wspd', 'max_temp', 'min_temp', 'precip_total', 'precip_hrly', 'snow_hrly', 'uv_desc', 'feels_like', 'uv_index', 'clds']
failed_dates = []


for date in backbone.date_time:
    
    try:
        
        time = str(date)[:-9].replace('-','')
        url = 'https://api.weather.com/v1/location/EGPN:9:GB/observations/historical.json?apiKey=e1f10a1e78da46f5b10a1e78da96f525&units=e&startDate=' + time + '&endDate=' + time
        x = requests.get(url)
        html = x.text
        data = json.loads(html)
        observations = data['observations']
    
        for obs in observations:
            ts = int(obs['valid_time_gmt'])
            tz = timezone(-timedelta(hours=0))
            out = datetime.fromtimestamp(ts, tz).strftime('%Y-%m-%d %H:%M:%S')            
            weatherdata.append([out, obs['day_ind'], obs['temp'], obs['wx_phrase'], obs['pressure_tend'], obs['pressure_desc'], obs['dewPt'], obs['heat_index'], obs['rh'], obs['pressure'], obs['vis'], obs['wc'], obs['wdir'], obs['wdir_cardinal'], obs['gust'], obs['wspd'], obs['max_temp'], obs['min_temp'], obs['precip_total'], obs['precip_hrly'], obs['snow_hrly'], obs['uv_desc'], obs['feels_like'], obs['uv_index'], obs['clds']])
    
        print("done " + time)
        
    except Exception as e: 
        print('Failed at ' + str(date))
        failed_dates.append(str(date))
        pass
    
    # print(datetime.fromtimestamp(ts, tz).strftime('%Y-%m-%d %H:%M:%S'))

df_weather = pd.DataFrame(weatherdata, columns=columns)


df_weather.to_csv('EGPN_weather.csv', encoding='utf-8')



# for site in sites:
#     for date in backbone.date_time:
#         url = 'https://www.wunderground.com/history/daily/' + site + '/date/' + str(date)[:-9]
#         x = requests.get(url)
#         html = x.text
        
        
#         one = html.split('observations&q;:')[1]
#         one = one.replace('&q', '')
#         one = one.replace(';', '')
        
#         two = one.split(']')[0] + ']'
        
#         idx = two.find("obsTimeUtc")
#         two = two[:idx+21] + two[idx+31:]
        
#         idx = two.find('obsTimeLocal')
#         two = two[:idx+23] + two[idx+32:]
        
#         two = two.replace('{', '{"')
#         two = two.replace('{"{', '{{')
#         two = two.replace('"{', '{')
#         two = two.replace(':', '":"')
#         two = two.replace(',', '","')
#         two = two.replace('}', '"}')
#         two = two.replace('}"}', '}}')
#         two = two.replace('}"', '}')
#         two = two.replace(':"{', ':{')
        
        
        
#         data = json.loads(two)
#         for item in data:
#             print(item)
#         #print(data)
#         print('Done ' +  site + ' ' + str(date)[:-9])