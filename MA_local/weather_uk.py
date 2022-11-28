import requests
import json
import ast
from datetime import datetime, timezone, timedelta
import pandas as pd



locations = ['@2645365']
# locations = ['uk/dundee']


weatherdata = []
columns = ['date','desc','temp','templow','baro','wind','wd','hum']
failed_dates = []

data_all = ''
for city in locations:
    for year in range(2017,2023):
        for month in range(1,13):
            url = 'https://www.timeanddate.com/weather/' + city + '/historic?month=' + str(month) + '&year=' + str(year)
            x = requests.get(url)
            html = x.text
            one = html.split("data=")
            two = one[1].split('"detail":[')
            three = two[1].split(']')
            data_str = three[0]
            data_all = data_all + data_str + ','
        print('Done ' + city + ' ' + str(year))

data_all = '[' + data_all[:-1] + ']'

data = json.loads(data_all)

tz = timezone(-timedelta(hours=0))

for item in data:
    conv_date = datetime.fromtimestamp(int(item['date'])/1000, tz).strftime('%Y-%m-%d %H:%M:%S')
    try:        
        weatherdata.append([conv_date, item['desc'], item['temp'], item['templow'], item['baro'], item['wind'], item['wd'], item['hum']])
        
    except Exception: 
        # print(e)
        # print('Failed at ' + str(conv_date))
        # failed_dates.append(str(conv_date))
        pass
        
df_weather = pd.DataFrame(weatherdata, columns=columns)


df_weather.to_csv('Perth_weather.csv', encoding='utf-8')


 

 
# url1 = 'https://www.timeanddate.com/weather/uk/' + '@2645365' + '/historic?month=' + '1' + '&year=' + '2017'
# url2 = 'https://www.timeanddate.com/weather/@2645365/historic?month=1&year=2017'
# x = requests.get(url1)