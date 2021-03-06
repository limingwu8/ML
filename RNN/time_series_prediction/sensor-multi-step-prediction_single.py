# import matplotlib
# matplotlib.use('Agg')
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential,load_model
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot
from numpy import array
import datetime
import numpy as np
import os
import pickle


# date-time parsing function for loading the dataset
def parser(x):
    return datetime.strptime('190'+x, '%Y-%m')

# convert time series into supervised learning problem
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)

# transform series into train and test sets for supervised learning
def prepare_data(series, n_test, n_lag, n_seq):
    # extract raw values
    raw_values = series.values
    # transform data to be stationary
    diff_series = difference(raw_values, 1)
    diff_values = diff_series.values
    diff_values = diff_values.reshape(len(diff_values), 1)
    # rescale values to -1, 1
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_values = scaler.fit_transform(diff_values)
    scaled_values = scaled_values.reshape(len(scaled_values), 1)
    # transform into supervised learning problem X, y
    supervised = series_to_supervised(scaled_values, n_lag, n_seq)
    supervised_values = supervised.values
    # split into train and test sets
    train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
    return scaler, train, test

# fit an LSTM network to training data
def fit_lstm(train, n_lag, n_seq, n_batch, nb_epoch, n_neurons):
    # reshape training into [samples, timesteps, features]
    X, y = train[:, 0:n_lag], train[:, n_lag:]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    # design network
    model = Sequential()
    model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(y.shape[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    # fit network
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=n_batch, verbose=0, shuffle=False)
        model.reset_states()
    return model

# make one forecast with an LSTM,
def forecast_lstm(model, X, n_batch):
    # reshape input pattern to [samples, timesteps, features]
    X = X.reshape(1, 1, len(X))
    # make forecast
    forecast = model.predict(X, batch_size=n_batch)
    # convert to array
    return [x for x in forecast[0, :]]

# evaluate the persistence model
def make_forecasts(model, n_batch, test, n_lag, n_seq):
    forecasts = list()
    for i in range(len(test)):
        X, y = test[i, 0:n_lag], test[i, n_lag:]
        # make forecast
        forecast = forecast_lstm(model, X, n_batch)
        # store the forecast
        forecasts.append(forecast)
    return forecasts

# invert differenced forecast
def inverse_difference(last_ob, forecast):
    # invert first forecast
    inverted = list()
    inverted.append(forecast[0] + last_ob)
    # propagate difference forecast using inverted first value
    for i in range(1, len(forecast)):
        inverted.append(forecast[i] + inverted[i-1])
    return inverted

# inverse data transform on forecasts
def inverse_transform(series, forecasts, scaler, n_test):
    inverted = list()
    for i in range(len(forecasts)):
        # create array from forecast
        forecast = array(forecasts[i])
        forecast = forecast.reshape(1, len(forecast))
        # invert scaling
        inv_scale = scaler.inverse_transform(forecast)
        inv_scale = inv_scale[0, :]
        # invert differencing
        index = len(series) - n_test + i - 1
        last_ob = series.values[index]
        inv_diff = inverse_difference(last_ob, inv_scale)
        # store
        inverted.append(inv_diff)
    return inverted
# evaluate the RMSE for each forecast time step
def evaluate_forecasts(test, forecasts, n_lag, n_seq, sensor_name):
    for i in range(n_seq):
        actual = [row[i] for row in test]
        predicted = [forecast[i] for forecast in forecasts]
        rmse = sqrt(mean_squared_error(actual, predicted))
        rmse_percent = rmse/np.mean(actual)
        if SAVE_INFO:
            # save data to pickle
            pickle.dump(actual, pkl)
            pickle.dump(predicted,pkl)
        print('t+%d RMSE: %f, error percent: %f%%' % ((i+1), rmse, rmse_percent*100))

        if SAVE_INFO:
            logs.write('t+%d RMSE: %f, error percent: %f%%\n' % ((i+1), rmse, rmse_percent*100))


# plot the forecasts in the context of the original dataset
def plot_forecasts(series, forecasts, n_test, file_name, sensor_name, time, n_seq):

    plot_one_line = 1
    fontsize = 12

    # plot the entire dataset in blue
    fig = pyplot.figure()
    forecasts = np.array(forecasts)
    pyplot.plot(time, series.values, label = 'Actual data')
    # only plot the last forecast value
    X = []
    for i in range(1,forecasts.shape[1]+1):

        off_s = len(series) - n_test + i - n_seq
        off_e = off_s + n_test - 1
        X.append(range(off_s,off_e+1))
    X = np.array(X)
    Y = np.array(forecasts)
    for i in range(0,Y.shape[1]):
        index = X[i]
        pyplot.plot(time[index[0]:index[len(index)-1]+1],Y[:,i], label = 'Prediction: t+' + str(i+1))
        if plot_one_line == 1:
            break
    pyplot.title(file_name,fontsize=fontsize)
    pyplot.legend(fontsize = fontsize)
    pyplot.xlabel('Time',fontsize=fontsize)
    pyplot.ylabel(units[sensor_name],fontsize=fontsize)


    # plot zoomed in figure
    fig_zoomed = pyplot.figure()
    # plot original data
    start = X[0][0] - 1
    end = len(series)
    pyplot.plot(time[start:end],series[start:end], label = 'Actual data')
    for i in range(0,Y.shape[1]):
        index = X[i]
        pyplot.plot(time[index[0]:index[len(index)-1]+1],Y[:,i], label = 'Prediction: t+' + str(i+1))
        if plot_one_line == 1:
            break
    pyplot.title(file_name, fontsize=fontsize)
    pyplot.legend(fontsize=fontsize)
    pyplot.xlabel('Time', fontsize=fontsize)
    pyplot.ylabel(units[sensor_name], fontsize=fontsize)

    # show the plot
    fig.show()
    fig_zoomed.show()

    if SAVE_INFO:
        fig.set_size_inches(18.5, 10.5)
        fig.savefig(PATH + file_name + '.png', bbox_inches='tight', dpi=200)
        fig_zoomed.set_size_inches(18.5, 10.5)
        fig_zoomed.savefig(PATH + file_name + '-zoomed.png', bbox_inches='tight', dpi=200)

    pyplot.close(fig)
    pyplot.close(fig_zoomed)


def get_files(file_dir):
    '''
    Args:
        file_dir: file directory
    Returns:
        list of file path
    '''
    dataset_path = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            dataset_path.append(os.path.join(root,file))
    return dataset_path

def run_train(n_lag, n_seq, n_epochs, n_batch, n_neurons, dataset_path):
    path = dataset_path.split('.')[-2]
    sensor_name = path.split('/')[-1]
    file_name = path.split('/')[-1] + '-' + path.split('/')[-2]

    print('processing the dataset of ', file_name)
    if SAVE_INFO:
        logs.write(file_name + '\n')


    # load dataset
    series = read_csv(dataset_path, sep=',')
    header = list(series.columns.values)

    raw_time = series[header[0]]
    raw_values = series[header[1]]

    raw_time = raw_time.values
    raw_datetime = [datetime.datetime.strptime(
        i, "%d-%b-%Y %H:%M:%S") for i in raw_time]
    raw_values = raw_values.values

    series_time = Series(raw_time)
    series_values = Series(raw_values)

    # configure
    n_test = int(0.2 * series.shape[0])

    # prepare data
    scaler, train, test = prepare_data(series_values, n_test, n_lag, n_seq)
    # fit model
    model = fit_lstm(train, n_lag, n_seq, n_batch, n_epochs, n_neurons)
    if SAVE_INFO == 1:
        # save model
        model_name = 'model_'+ file_name + '-' + 'seq_' + str(n_seq) +'.h5'
        model.save(PATH + model_name)

    # make forecasts
    forecasts = make_forecasts(model, n_batch, test, n_lag, n_seq)
    # inverse transform forecasts and test
    forecasts = inverse_transform(series_values, forecasts, scaler, n_test + n_seq - 1)
    actual = [row[n_lag:] for row in test]
    actual = inverse_transform(series_values, actual, scaler, n_test + n_seq - 1)
    # evaluate forecasts
    evaluate_forecasts(actual, forecasts, n_lag, n_seq, file_name)
    # plot forecasts
    plot_forecasts(series_values, forecasts, n_test, file_name, sensor_name, raw_datetime, n_seq)


SAVE_INFO = 1       # 1: save information in file, 0: do not save
RUN_ON_LOCAL = 1    # 1: run on local, 0: run on server
TRAIN = 1           # 1: train model, 0: load model

n_lag = 1
n_seq = 4  # forecast the next n_seq
n_epochs = 1500
n_batch = 1
n_neurons = 1
dataset_path = './dataset/sample_6_hour/OIL_RETURN_TEMPERATURE.csv'
# sensor units
units = {'MAIN_FILTER_IN_PRESSURE':'PSI','MAIN_FILTER_OIL_TEMP':'Fahrenheit',
         'MAIN_FILTER_OUT_PRESSURE':'PSI','OIL_RETURN_TEMPERATURE':'Fahrenheit',
         'TANK_FILTER_IN_PRESSURE':'PSI','TANK_FILTER_OUT_PRESSURE':'PSI',
         'TANK_LEVEL':'Inch','TANK_TEMPERATURE':'Fahrenheit','FT-202B':'Mils',
         'FT-204B':'Mils','PT-203':'Mils','PT-204.HS':'Mils'}

if RUN_ON_LOCAL:
    PATH = '/home/bc/Documents/USS/compare/'
else:
    PATH = '/home/PNW/wu1114/Documents/USS/compare/'

if SAVE_INFO:
    with open(PATH+'logs.txt','w') as logs:
        with open(PATH+'data.pkl','wb') as pkl:
            if TRAIN:
                run_train(n_lag, n_seq, n_epochs, n_batch, n_neurons, dataset_path)

else:
    if TRAIN:
        run_train(n_lag, n_seq, n_epochs, n_batch, n_neurons, dataset_path)
