import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

from darts import TimeSeries
from darts.datasets import AirPassengersDataset
from darts.metrics import mape
from darts.models import ExponentialSmoothing, TBATS, AutoARIMA, Theta, Prophet, KalmanForecaster, LinearRegressionModel, RandomForest, RNNModel, TCNModel, TransformerModel, TFTModel

def eval_model(model, train, test):
    model.fit(train)
    forecast = model.predict(len(test))
    print("model {} obtains MAPE: {:.2f}%".format(model, mape(test, forecast)))


ref_ts = pd.read_csv('/home/doktormatte/MA_SciComp/ACN_2/Loads/ref_ts.csv')
ref_values = pd.read_csv('/home/doktormatte/MA_SciComp/ACN_2/Loads/sum_red_header.csv')

ts = pd.DataFrame(columns=['date_time', 'value'])
ts.date_time = ref_ts.date_time
# ts.date_time = ts.date_time.map(lambda x: datetime.datetime.strptime(x, '%d %b %Y %H:%M:%S'))
ts.date_time = ts.date_time.map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
ts.value = ref_values.k


df_series = TimeSeries.from_dataframe(ts, 'date_time', 'value')

train_test_split = 0.7
n_train = int(train_test_split*len(df_series)) 

train = df_series[:n_train]
test = df_series[n_train:n_train+48]



# eval_model(ExponentialSmoothing(), train, test)
# eval_model(TBATS(), train, test)
eval_model(AutoARIMA(), train, test)
eval_model(Theta(), train, test)
eval_model(Prophet(), train, test)
eval_model(KalmanForecaster(), train, test)
eval_model(LinearRegressionModel(), train, test)
eval_model(RandomForest(), train, test)
eval_model(RNNModel(), train, test)
eval_model(TCNModel(), train, test)
eval_model(TransformerModel(), train, test)
eval_model(TFTModel(), train, test)


