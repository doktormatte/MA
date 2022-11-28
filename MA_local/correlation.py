import datetime
import pandas as pd
import numpy as np
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose


cutoff = datetime.datetime.strptime("2019/03/21 00:00:00+00", '%Y/%m/%d %H:%M:%S+%f')
end = datetime.datetime.strptime("2022/05/01 00:00:00+00", '%Y/%m/%d %H:%M:%S+%f')

# series = pd.read_csv('/home/doktormatte/MA_SciComp/Boulder/Occup/test_occup.csv', header=0, index_col=0)
#series = pd.read_csv('/home/doktormatte/MA_SciComp/Boulder/Occup/test_occup.csv', header=0).squeeze('columns')
series = pd.read_csv('/home/doktormatte/MA_SciComp/Boulder/Loads/test_load.csv', header=0).squeeze('columns')



X = series.values
X = X[7584:-44]
# result = adfuller(X)
# print('ADF Statistic: %f' % result[0])
# print('p-value: %f' % result[1])
# print('Critical Values:')
# for key, value in result[4].items():
# 	print('\t%s: %.3f' % (key, value))

# plot_acf(series)
# pyplot.show()


loads = pd.Series(X, index=pd.date_range(cutoff, end, freq="15min"))


# decomposition = seasonal_decompose(loads)
decomposition = seasonal_decompose(loads, model = 'additive', filt = None)
# decomposition = sm.tsa.seasonal_decompose(X, freq = 11712)
# stl = STL(X, seasonal=11712)