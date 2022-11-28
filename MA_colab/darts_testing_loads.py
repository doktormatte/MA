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

# def eval_model(model, train, test):
#     model.fit(train)
#     forecast = model.predict(len(test))
#     print("model {} obtains MSE: {:.2f}%".format(model, mse(test, forecast)))

def eval_model(model, time_series, n_steps_out, past_covariates=None, future_covariates=None):
    # Past and future covariates are optional because they won't always be used in our tests
    
    
    backtest = model.historical_forecasts(series=time_series, 
                                          past_covariates=past_covariates,
                                          future_covariates=future_covariates,
                                          start=0.7, 
                                          retrain=False,
                                          verbose=False, 
                                          forecast_horizon=n_steps_out)
    print('Backtest RMSE = {}'.format(rmse(df_series, backtest)))
    return(rmse(df_series, backtest))



# dirs = ['ACN_1', 'ACN_2', 'Boulder', 'Palo_Alto', 'Dundee', 'Perth_Kinross']
dirs = ['Palo_Alto', 'Dundee', 'Perth_Kinross']
results = []

for dirname in dirs:
    ref_ts = pd.read_csv('/home/doktormatte/MA_SciComp/' + dirname + '/Loads/ref_ts.csv')
    ref_values = pd.read_csv('/home/doktormatte/MA_SciComp/' + dirname + '/Loads/sum_red_header.csv')

    ts = pd.DataFrame(columns=['date_time', 'value'])
    ts.date_time = ref_ts.date_time
    ts.date_time = ts.date_time.map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    ts.value = ref_values.k
    df_series = TimeSeries.from_dataframe(ts, 'date_time', 'value')
    train_test_split = 0.7
    n_train = int(train_test_split*len(df_series)) 
    n_steps_in = 12
    n_steps_out = 3
    
    train, _ = df_series.split_before(train_test_split)    
    
    brnn = BlockRNNModel(input_chunk_length=n_steps_in, output_chunk_length=n_steps_out, n_rnn_layers=2)
    brnn.fit(train, epochs=50, verbose=True)    
    results.append([dirname, 'brnn', eval_model(brnn, df_series, n_steps_out)])
    
    lstm = RNNModel(model="LSTM",input_chunk_length=n_steps_in, training_length=n_steps_in+n_steps_out, n_rnn_layers=2)
    lstm.fit(train, epochs=50, verbose=True)
    results.append([dirname, 'lstm', eval_model(lstm, df_series, n_steps_out)])
    
    gru = RNNModel(model="GRU",input_chunk_length=n_steps_in, training_length=n_steps_in+n_steps_out, n_rnn_layers=2)
    gru.fit(train, epochs=50, verbose=True)
    results.append([dirname, 'gru', eval_model(gru, df_series, n_steps_out)])
    
    tcn = TCNModel(input_chunk_length=n_steps_in, output_chunk_length=n_steps_out, num_layers=4)
    tcn.fit(train, epochs=50, verbose=True)
    results.append([dirname, 'tcn', eval_model(tcn, df_series, n_steps_out)])
    
    tfm = TransformerModel(input_chunk_length=n_steps_in, output_chunk_length=n_steps_out)
    tfm.fit(train, epochs=50, verbose=True)
    results.append([dirname, 'tfm', eval_model(tfm, df_series, n_steps_out)])
    
    nhts = NHiTSModel(input_chunk_length=n_steps_in, output_chunk_length=n_steps_out)
    nhts.fit(train,epochs=50, verbose=True)
    results.append([dirname, 'nhts', eval_model(nhts, df_series, n_steps_out)])    
    
    nbts = NBEATSModel(input_chunk_length=n_steps_in, output_chunk_length=n_steps_out)
    nbts.fit(train,epochs=50, verbose=True)
    results.append([dirname, 'nbts', eval_model(nbts, df_series, n_steps_out)])   
    
    
    
    
    
    
    
    
    



