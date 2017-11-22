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
import matplotlib.pyplot as plt
import numpy as np
import re
import os
import datetime


# date-time parsing function for loading the dataset
def parser(x):
    return datetime.strptime('190'+x, '%Y-%m')

# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df

# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

# scale train and test data to [-1, 1]
def scale(train, test):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled

# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]

# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    model.add(LSTM(units=neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        model.reset_states()
    return model

# make a one-step forecast
def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0,0]

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


dataset_path = get_files('./dataset/')

logs = open('/home/bc/Pictures/logsTemp.txt','w')

for i in range(len(dataset_path)):

    path = dataset_path[i]
    folder_name = path.split('/')[-2]
    file_name = path.split('/')[-1]
    file_name = file_name.split('.')[0]

    print('processing the dataset of ', file_name + '-' + folder_name)
    logs.write(file_name + '-' + folder_name + '\n')

    # load dataset
    series = read_csv(path,sep=',')
    header = list(series.columns.values)

    raw_time = series[header[0]]
    raw_values = series[header[1]]

    raw_time = raw_time.values
    raw_datetime = [datetime.datetime.strptime(
        i, "%d-%b-%Y %H:%M:%S") for i in raw_time]
    raw_values = raw_values.values

    test_len = int(len(raw_values)*0.2)
    # transform data to be stationary
    diff_values = difference(raw_values, 1)

    # transform data to be supervised learning
    supervised = timeseries_to_supervised(diff_values, 1)
    supervised_values = supervised.values

    # split data into train and test-sets
    train, test = supervised_values[0:-test_len], supervised_values[-test_len:]

    # transform the scale of the data
    scaler, train_scaled, test_scaled = scale(train, test)

    epoch = [1,10,50,100,300,500]
    fig = []
    fig_zoomed = []
    rmse_error = []
    for e in epoch:
        print('Epoch : ',e)
        logs.write('Epoch : %d\n' % e)
        # fit the model
        lstm_model = fit_lstm(train_scaled, 1, e, 10)
        # forecast the entire training dataset to build up state for forecasting
        train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
        lstm_model.predict(train_reshaped, batch_size=1)

        # walk-forward validation on the test data
        predictions = list()
        for i in range(len(test_scaled)):
            # make one-step forecast
            X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
            yhat = forecast_lstm(lstm_model, 1, X)
            # invert scaling
            yhat = invert_scale(scaler, X, yhat)
            # invert differencing
            yhat = inverse_difference(raw_values, yhat, len(test_scaled) + 1 - i)
            # store forecast
            predictions.append(yhat)
            expected = raw_values[len(train) + i + 1]
            print('timePoint=%d, Predicted=%f, Expected=%f' % (i + 1, yhat, expected))

        # report performance
        rmse = sqrt(mean_squared_error(raw_values[-test_len:], predictions))
        print('Test RMSE: %.3f' % rmse)
        logs.write('Test RMSE: %.3f\n' % rmse)
        rmse_error.append(rmse)

        # save figures
        fig1 = plt.figure()
        plt.title(file_name)
        plt1 = plt.plot(raw_datetime[len(train) + 1 : len(raw_values)],raw_values[-test_len:], label = 'actual')
        plt2 = plt.plot(raw_datetime[len(train) + 1 : len(raw_values)],predictions, label = 'prediction')
        plt.legend()
        fig_zoomed.append(fig1)

        # line plot of observed vs predicted
        fig2 = plt.figure()
        plt.title(file_name)
        plt1 = plt.plot(raw_datetime, raw_values, label='prediction')
        plt2 = plt.plot(raw_datetime[len(train) + 1 : len(raw_values)], predictions,label='actual')
        plt.legend()
        fig.append(fig2)
        # plt.show()

    index_min = np.argmin(rmse_error)
    fig[index_min].set_size_inches(18.5, 10.5)
    fig[index_min].savefig('/home/bc/Pictures/' + file_name + '-' + folder_name + '.png', bbox_inches='tight',dpi = 200)
    fig_zoomed[index_min].set_size_inches(18.5, 10.5)
    fig_zoomed[index_min].savefig('/home/bc/Pictures/' + file_name + '_zoomed-' + folder_name + '.png', bbox_inches='tight',dpi = 200)

logs.close()
