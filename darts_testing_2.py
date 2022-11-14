import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

from darts import TimeSeries
from darts.datasets import AirPassengersDataset
from darts.metrics import mape, mse, rmse
from darts.models import (ExponentialSmoothing,
                            TBATS,
                            AutoARIMA,
                            Theta,
                            Prophet,
                            KalmanForecaster,
                            LinearRegressionModel,
                            RandomForest,
                            RNNModel,
                            TCNModel,
                            TransformerModel,
                            TFTModel,
                            BlockRNNModel,
                            NHiTSModel,
                            NBEATSModel)

def eval_model(model, train, test):
    model.fit(train)
    forecast = model.predict(len(test))
    print("model {} obtains MSE: {:.2f}%".format(model, mse(test, forecast)))


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
n_steps_in = 12
n_steps_out = 3

# train = df_series[:n_train]
# test = df_series[n_train:n_train+48]


# We first set aside the first 80% as training series:
train, _ = df_series.split_before(train_test_split)



def eval_model(model, past_covariates=None, future_covariates=None):
    # Past and future covariates are optional because they won't always be used in our tests
    
    # We backtest the model on the last 20% of the flow series, with a horizon of 10 steps:
    backtest = model.historical_forecasts(series=df_series, 
                                          past_covariates=past_covariates,
                                          future_covariates=future_covariates,
                                          start=train_test_split, 
                                          retrain=False,
                                          verbose=True, 
                                          forecast_horizon=n_steps_out)
    
    # flow[-len(backtest)-100:].plot()
    # backtest.plot(label='backtest (n=10)')
    print('Backtest RMSE = {}'.format(rmse(df_series, backtest)))
    
# brnn_no_cov = BlockRNNModel(input_chunk_length=n_steps_in, output_chunk_length=n_steps_out, n_rnn_layers=2)

# brnn_no_cov.fit(train, epochs=50, verbose=True)
# eval_model(brnn_no_cov)


# lstm = RNNModel(model="LSTM",input_chunk_length=n_steps_in, training_length=n_steps_in+n_steps_out, n_rnn_layers=2)
# lstm.fit(train, epochs=3, verbose=True)

# gru = RNNModel(model="GRU",input_chunk_length=n_steps_in, training_length=n_steps_in+n_steps_out, n_rnn_layers=2)
# gru.fit(train, epochs=3, verbose=True)


# tcn = TCNModel(input_chunk_length=n_steps_in, output_chunk_length=n_steps_out, num_layers=4)
# tcn.fit(train, epochs=3, verbose=True)

# tfm = TransformerModel(input_chunk_length=n_steps_in, output_chunk_length=n_steps_out)
# tfm.fit(train, epochs=3, verbose=True)

# tft = TFTModel(input_chunk_length=n_steps_in, output_chunk_length=n_steps_out, add_relative_index=True)
# tft.fit(train, epochs=3, verbose=True)

# nhts = NHiTSModel(input_chunk_length=n_steps_in, output_chunk_length=n_steps_out)
# nhts.fit(train,epochs=3, verbose=True)




# nbts = NBEATSModel(input_chunk_length=n_steps_in, output_chunk_length=n_steps_out)
# nbts.fit(train,epochs=3, verbose=True)

prophet = Prophet()
prophet.fit(train)







# eval_model(ExponentialSmoothing(), train, test)
# eval_model(TBATS(), train, test)
# eval_model(AutoARIMA(), train, test)
# eval_model(Theta(), train, test)


# eval_model(Prophet(), train, test)
# eval_model(KalmanForecaster(), train, test)



# eval_model(LinearRegressionModel(), train, test)
# eval_model(RandomForest(), train, test)
# eval_model(RNNModel(), train, test)
# eval_model(TCNModel(), train, test)
# eval_model(TransformerModel(), train, test)
# eval_model(TFTModel(), train, test)


