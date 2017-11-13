import matplotlib
matplotlib.use('Agg')
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
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
def make_forecasts(model, n_batch, train, test, n_lag, n_seq):
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
def plot_forecasts(series, forecasts, n_test, sensor_name):
    # plot the entire dataset in blue
    fig = pyplot.figure()
    pyplot.plot(series.values)
    # only plot the last forecast value
    X = []
    Y = []
    for i in range(len(forecasts)):

        off_s = len(series) - n_test + i - 1
        off_e = off_s + len(forecasts[i]) + 1
        xaxis = [x for x in range(off_s, off_e)]
        yaxis = [series.values[off_s]] + forecasts[i]
        X.append(xaxis)
        Y.append(yaxis)
    X = np.array(X)
    Y = np.array(Y)
    for i in range(1,X.shape[1]):
        pyplot.plot(X[:,i],Y[:,i])

    ## plot zoomed in figure, for the next three steps
    fig_zoomed = pyplot.figure()
    # plot original data
    pyplot.plot(range(X[0, 0], X[X.shape[0] - 1, X.shape[1] - 1] + 1),
                series[X[0, 0]: X[X.shape[0] - 1, X.shape[1] - 1]+1])
    # plot forecasts data
    for i in range(1,X.shape[1]):
        pyplot.plot(X[:,i],Y[:,i])

    ## plot zoomed in figure, for the only next one step
    fig_zoomed_next_one = pyplot.figure()
    # plot original data
    pyplot.plot(range(X[0, 0], X[X.shape[0] - 1, X.shape[1] - 1] + 1),
                series[X[0, 0]: X[X.shape[0] - 1, X.shape[1] - 1] + 1])
    pyplot.plot(X[:,1],Y[:,1])


    # show the plot
    fig.show()
    fig_zoomed.show()
    fig_zoomed_next_one.show()

    if SAVE_INFO:
        fig.set_size_inches(18.5, 10.5)
        fig.savefig(path + sensor_name + '.png', bbox_inches='tight', dpi=200)
        fig_zoomed.set_size_inches(18.5, 10.5)
        fig_zoomed.savefig(path + sensor_name + '-zoomed.png', bbox_inches='tight', dpi=200)
        fig_zoomed_next_one.savefig(path + sensor_name + '-zoomed-next_one_step.png',
                                    bbox_inches='tight', dpi=200)
    pyplot.close(fig)
    pyplot.close(fig_zoomed)
    pyplot.close(fig_zoomed_next_one)


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

def run_train():
    dataset_path = get_files('./dataset/')
    for i in range(len(dataset_path)):
        file_path = dataset_path[i]
        folder_name = file_path.split('/')[-2]
        file_name = file_path.split('/')[-1]
        file_name = file_name.split('.')[0]
        sensor_name = file_name + '-' + folder_name

        print('processing the dataset of ', sensor_name)
        if SAVE_INFO:
            logs.write(file_name + '-' + folder_name + '\n')

        # load dataset
        series = read_csv(file_path, sep=',')
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
        n_lag = 1
        n_seq = 3  # forecast the next n_seq
        n_test = int(0.2 * series.shape[0])
        # n_epochs = 1500
        n_epochs = 1
        n_batch = 1
        n_neurons = 1
        # prepare data
        scaler, train, test = prepare_data(series_values, n_test, n_lag, n_seq)
        # fit model
        model = fit_lstm(train, n_lag, n_seq, n_batch, n_epochs, n_neurons)
        # make forecasts
        forecasts = make_forecasts(model, n_batch, train, test, n_lag, n_seq)
        # inverse transform forecasts and test
        forecasts = inverse_transform(series_values, forecasts, scaler, n_test + 2)
        actual = [row[n_lag:] for row in test]
        actual = inverse_transform(series_values, actual, scaler, n_test + 2)
        # evaluate forecasts
        evaluate_forecasts(actual, forecasts, n_lag, n_seq, sensor_name)
        # plot forecasts
        plot_forecasts(series_values, forecasts, n_test + 2, sensor_name)



SAVE_INFO = 1
RUN_ON_LOCAL = 1

path = ''

if RUN_ON_LOCAL:
    path = '/home/bc/Documents/USS/multi-step-prediction/'
else:
    path = '/home/PNW/wu1114/Documents/USS/multi-step-prediction/'

if SAVE_INFO:
    with open(path+'logs.txt','w') as logs:
        with open(path+'data.pkl','wb') as pkl:
            run_train()
else:
    run_train()