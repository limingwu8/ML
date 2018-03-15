import matplotlib
# matplotlib.use('Agg')
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential,load_model
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot
from numpy import array
import datetime
from matplotlib.dates import DateFormatter
import numpy as np
from scipy import stats
from scipy.stats import rayleigh
import os
import pickle
from collections import OrderedDict

class Sensor:
    units = {'MAIN_FILTER_IN_PRESSURE':'PSI','MAIN_FILTER_OIL_TEMP':'Fahrenheit',
         'MAIN_FILTER_OUT_PRESSURE':'PSI','OIL_RETURN_TEMPERATURE':'Fahrenheit',
         'TANK_FILTER_IN_PRESSURE':'PSI','TANK_FILTER_OUT_PRESSURE':'PSI',
         'TANK_LEVEL':'Inch','TANK_TEMPERATURE':'Fahrenheit','FT-202B':'Mils',
         'FT-204B':'Mils','PT-203':'Mils','PT-204':'Mils'}

    def __init__(self, dataset_path, sensor_name,operating_range, sample_rate, root_path, n_epochs = 1, n_batch = 1,
                 save_info = 0, n_neurons = 1, run_on_local = 1, train = 1, n_lag = 1, n_seq = 1):
        self.n_lag = n_lag
        self.n_seq = n_seq
        self.n_epochs = n_epochs
        self.n_batch = n_batch
        self.n_neurons = n_neurons
        self.dataset_path = dataset_path
        self.sensor_name = sensor_name
        self.operating_range = operating_range
        self.sample_rate = sample_rate
        self.root_path = root_path
        self.save_info = save_info
        self.run_on_local = run_on_local
        self.train = train
        self.init_file_name()

    def get_units(self):
        return self.units

    def init_file_name(self):
        self.dataset_path = os.path.join(self.dataset_path, self.sample_rate, self.sensor_name + '.csv')
        self.file_name = self.sensor_name + '-' + self.sample_rate
        self.file_path = os.path.join(self.root_path, self.sensor_name, self.sample_rate, str(self.n_seq) + '_step')

    def get_files(self, file_dir):
        '''
        Args:
            file_dir: file directory
        Returns:
            list of file path
        '''
        dataset_path = []
        for root, dirs, files in os.walk(file_dir):
            for file in files:
                dataset_path.append(os.path.join(root, file))
        return dataset_path

    # date-time parsing function for loading the dataset
    def parser(self, x):
        return datetime.strptime('190' + x, '%Y-%m')

    # convert time series into supervised learning problem
    def series_to_supervised(self, data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
        # put it all together
        agg = concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    # create a differenced series
    def difference(self, dataset, interval=1):
        diff = list()
        for i in range(interval, len(dataset)):
            value = dataset[i] - dataset[i - interval]
            diff.append(value)
        return Series(diff)

    # if the prediction values are minus, set them zero
    def constrain(self, forecasts):
        for i in range(0, len(forecasts)):
            item = forecasts[i]
            for j in range(0, len(item)):
                if forecasts[i][j] < 0:
                    forecasts[i][j] = 0
        return forecasts

    # transform series into train and test sets for supervised learning
    def prepare_data(self, series, n_test, n_lag, n_seq):
        # extract raw values
        raw_values = series.values
        # transform data to be stationary
        diff_series = self.difference(raw_values, 1)
        diff_values = diff_series.values
        diff_values = diff_values.reshape(len(diff_values), 1)
        # rescale values to -1, 1
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_values = scaler.fit_transform(diff_values)
        scaled_values = scaled_values.reshape(len(scaled_values), 1)
        # transform into supervised learning problem X, y
        supervised = self.series_to_supervised(scaled_values, n_lag, n_seq)
        supervised_values = supervised.values
        # split into train and test sets
        train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
        return scaler, train, test

    # fit an LSTM network to training data
    def fit_lstm(self, train, n_lag, n_seq, n_batch, nb_epoch, n_neurons):
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
            model.fit(X, y, epochs=1, batch_size=n_batch, verbose=2, shuffle=False)
            model.reset_states()
        # model.fit(X, y, epochs=nb_epoch, batch_size=n_batch, verbose=0, shuffle=False)

        return model

    # make one forecast with an LSTM,
    def forecast_lstm(self, model, X, n_batch):
        # reshape input pattern to [samples, timesteps, features]
        X = X.reshape(1, 1, len(X))
        # make forecast
        forecast = model.predict(X, batch_size=n_batch)
        # convert to array
        return [x for x in forecast[0, :]]

    # evaluate the persistence model
    def make_forecasts(self, model, n_batch, test, n_lag, n_seq):
        forecasts = list()
        for i in range(len(test)):
            X, y = test[i, 0:n_lag], test[i, n_lag:]
            # make forecast
            forecast = self.forecast_lstm(model, X, n_batch)
            # store the forecast
            forecasts.append(forecast)
        return forecasts

    # invert differenced forecast
    def inverse_difference(self, last_ob, forecast):
        # invert first forecast
        inverted = list()
        inverted.append(forecast[0] + last_ob)
        # propagate difference forecast using inverted first value
        for i in range(1, len(forecast)):
            inverted.append(forecast[i] + inverted[i - 1])
        return inverted

    # inverse data transform on forecasts
    def inverse_transform(self, series, forecasts, scaler, n_test):
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
            inv_diff = self.inverse_difference(last_ob, inv_scale)
            # store
            inverted.append(inv_diff)
        return inverted

    # evaluate the RMSE for each forecast time step
    def evaluate_forecasts(self, test, forecasts, n_lag, n_seq, sensor_name):
        for i in range(n_seq):
            actual = [row[i] for row in test]
            predicted = [forecast[i] for forecast in forecasts]
            rmse = sqrt(mean_squared_error(actual, predicted))
            rmse_percent = rmse / np.mean(actual)
            if self.save_info & self.train:
                # save data to pickle
                pickle.dump(actual, self.pkl)
                pickle.dump(predicted, self.pkl)
            print('t+%d RMSE: %f, error percent: %f%%' % ((i + 1), rmse, rmse_percent * 100))

            if self.save_info & self.train:
                self.logs.write('t+%d RMSE: %f, error percent: %f%%\n' % ((i + 1), rmse, rmse_percent * 100))

    # plot the forecasts in the context of the original dataset
    def plot_forecasts(self, series, forecasts, n_test, file_name, sensor_name, time, n_seq):

        plot_one_line = 1
        label_fontsize = 35
        axis_fontsize = 30
        linewidth = 5

        # plot the entire dataset in blue
        fig = pyplot.figure()
        ax1 = fig.add_subplot(1, 1, 1)
        # make x label in a specific format
        ax1.xaxis_date()
        ax1.xaxis.set_major_formatter(DateFormatter('%m-%d'))
        forecasts = np.array(forecasts)
        pyplot.plot(time, series.values, label='Actual data', linewidth=linewidth)
        ####################### plot the forecast value #########################
        X = []
        for i in range(1, forecasts.shape[1] + 1):
            off_s = len(series) - n_test + i - n_seq
            off_e = off_s + n_test - 1
            X.append(range(off_s, off_e + 1))
        X = np.array(X)
        Y = np.array(forecasts)
        for i in range(0, Y.shape[1]):
            index = X[i]
            pyplot.plot(time[index[0]:index[len(index) - 1] + 1], Y[:, i], label='Prediction: t+' + str(i + 1), linewidth=linewidth)
            if plot_one_line == 1:
                break
        pyplot.title(file_name, fontsize=label_fontsize)
        pyplot.legend(fontsize=label_fontsize)
        pyplot.xlabel('Time', fontsize=label_fontsize)
        pyplot.ylabel(self.units[sensor_name], fontsize=label_fontsize)
        pyplot.xticks(fontsize=axis_fontsize)
        pyplot.yticks(fontsize=axis_fontsize)

        ######################### plot zoomed in figure ########################
        fig_zoomed = pyplot.figure()
        ax2 = fig_zoomed.add_subplot(1, 1, 1)
        ax2.xaxis_date()
        ax2.xaxis.set_major_formatter(DateFormatter('%m-%d'))
        # plot original data
        start = X[0][0] - 1
        end = len(series)
        pyplot.plot(time[start:end], series[start:end], label='Actual data', linewidth=linewidth)
        for i in range(0, Y.shape[1]):
            index = X[i]
            pyplot.plot(time[index[0]:index[len(index) - 1] + 1], Y[:, i], label='Prediction: t+' + str(i + 1), linewidth=linewidth)
            if plot_one_line == 1:
                break
        pyplot.title(file_name, fontsize=label_fontsize)
        pyplot.legend(fontsize=label_fontsize)
        pyplot.xlabel('Time', fontsize=label_fontsize)
        pyplot.ylabel(self.units[sensor_name], fontsize=label_fontsize)
        pyplot.xticks(fontsize=axis_fontsize)
        pyplot.yticks(fontsize=axis_fontsize)
        # show the plot
        fig.show()
        fig_zoomed.show()

        if self.save_info:
            fig.set_size_inches(18.5, 10.5)
            fig_zoomed.set_size_inches(18.5, 10.5)
            fig.savefig(os.path.join(self.file_path, file_name + '.png'), bbox_inches='tight', dpi=150)
            fig_zoomed.savefig(os.path.join(self.file_path, file_name + '-zoomed.png'), bbox_inches='tight', dpi=150)

        pyplot.close(fig)
        pyplot.close(fig_zoomed)

    def load_dataset(self):
        series = read_csv(self.dataset_path, sep=',')
        header = list(series.columns.values)

        raw_time = series[header[0]]
        raw_values = series[header[1]]

        raw_time = raw_time.values
        raw_datetime = [datetime.datetime.strptime(
            i, "%Y-%m-%d %H:%M:%S") for i in raw_time]
        raw_values = raw_values.values

        series_time = Series(raw_time)
        series_values = Series(raw_values)
        return series, series_values, raw_datetime

    def open_file(self):

        if not os.path.exists(self.file_path):
            try:
                os.makedirs(self.file_path)
            except:
                print('create folder error!')
        try:
            self.logs = open(os.path.join(self.file_path, 'logs.txt'), 'w')
            self.pkl = open(os.path.join(self.file_path, 'data.pkl'),'wb')
        except:
            print('open file error!')
    def close_file(self):
        try:
            self.logs.close()
            self.pkl.close()
        except:
            print('close file error!')

    def run_train(self):
        # create logs files
        self.open_file()

        print('processing the dataset of ', self.file_name)
        if self.save_info:
            self.logs.write(self.file_name + '\n')

        series, series_values, raw_datetime = self.load_dataset()
        # number of testing data, here use Novermber's data as testing
        # n_test = int(0.2 * series.shape[0])
        a = [raw_datetime[i].month == 11 for i in range(0, len(raw_datetime))]
        n_test = len(np.where(a)[0])

        # prepare data
        scaler, train, test = self.prepare_data(series_values, n_test, self.n_lag, self.n_seq)
        # fit model
        model = self.fit_lstm(train, self.n_lag, self.n_seq, self.n_batch, self.n_epochs, self.n_neurons)
        if self.save_info == 1:
            # save model
            model_name = 'model_' + self.file_name + '-' + 'seq_' + str(self.n_seq) + '.h5'
            model.save(os.path.join(self.file_path, model_name))

        # make prediction
        forecasts = self.make_forecasts(model, self.n_batch, test, self.n_lag, self.n_seq)
        # inverse transform forecasts and test
        forecasts = self.inverse_transform(series_values, forecasts, scaler, n_test + self.n_seq - 1)
        forecasts = self.constrain(forecasts)
        actual = [row[self.n_lag:] for row in test]
        actual = self.inverse_transform(series_values, actual, scaler, n_test + self.n_seq - 1)
        # evaluate forecasts
        self.evaluate_forecasts(actual, forecasts, self.n_lag, self.n_seq, self.file_name)
        # plot forecasts
        self.plot_forecasts(series_values, forecasts, n_test, self.file_name, self.sensor_name, raw_datetime, self.n_seq)
        # close file
        self.close_file()

    def run_update(self):
        pass

    def load_model_and_predict(self):
        # load model
        print('loading model ' + self.file_name + '.h5...')
        model = load_model(os.path.join(self.file_path, 'model_' + self.file_name + '-' + 'seq_' + str(self.n_seq) + '.h5'))
        # load dataset
        series, series_values, raw_datetime = self.load_dataset()

        # number of testing data, here use Novermber's data as testing
        a = [raw_datetime[i].month == 11 for i in range(0, len(raw_datetime))]
        n_test = len(np.where(a)[0])
        scaler, train, test = self.prepare_data(series_values, n_test, self.n_lag, self.n_seq)
        # make a prediction
        forecasts = self.make_forecasts(model, self.n_batch, test, self.n_lag, self.n_seq)
        # inverse transform forecasts and test        pyplot.show()
        forecasts = self.inverse_transform(series_values, forecasts, scaler, n_test + self.n_seq - 1)
        forecasts = self.constrain(forecasts)

        actual = [row[self.n_lag:] for row in test]
        actual = self.inverse_transform(series_values, actual, scaler, n_test + self.n_seq - 1)
        # evaluate forecasts
        self.evaluate_forecasts(actual, forecasts, self.n_lag, self.n_seq, self.file_name)
        # plot forecasts
        self.plot_forecasts(series_values, forecasts, n_test, self.file_name, self.sensor_name, raw_datetime, self.n_seq)

    def get_health_score(self):
        print('loading model ' + self.file_name + '.h5...')
        model = load_model(
            os.path.join(self.file_path, 'model_' + self.file_name + '-' + 'seq_' + str(self.n_seq) + '.h5'))
        # load dataset
        series, series_values, raw_datetime = self.load_dataset()

        # number of testing data, here use Novermber's data as testing
        a = [raw_datetime[i].month == 11 for i in range(0, len(raw_datetime))]
        n_test = len(np.where(a)[0])
        scaler, train, test = self.prepare_data(series_values, n_test, self.n_lag, self.n_seq)
        # make a prediction
        forecasts = self.make_forecasts(model, self.n_batch, test, self.n_lag, self.n_seq)
        # inverse transform forecast
        forecasts = self.inverse_transform(series_values, forecasts, scaler, n_test + self.n_seq - 1)
        forecasts = self.constrain(forecasts)
        # for sensor 'FT-202B' and 'PT-203', we should use log transfer to make them looks like Gaussian
        if self.sensor_name in ['FT-202B', 'PT-203', 'FT-204B','PT-204']:
            # use log transform
            # normal, low, high = self.operating_range
            # normal = np.log(normal + 10)
            # low = np.log(low + 10)
            # high = np.log(high + 10)
            # three_sigma = abs(normal-low) if abs(normal-low)>abs(normal-high) else abs(normal-high)
            # mu = normal
            # sigma = three_sigma / 3
            # cdf = stats.norm.cdf(np.log(np.array(forecasts) + 10), loc=mu, scale=sigma)
            # health_index = 1 - abs(cdf - 0.5) * 2
            # time = raw_datetime[-n_test:]

            # use rayleigh distribution
            # if the prediction value is less than the mean of the rayleigh distribution, set health index as 1
            # otherwise the far from the mean, the less the health index is
            health_index = np.zeros((len(forecasts),1))
            mean, var, skew, kurt = rayleigh.stats(moments='mvsk')
            index = forecasts <= mean
            health_index[index] = 1
            index = forecasts > mean
            cdf = rayleigh.cdf(forecasts)
            health_index[index] = (1 - cdf[index])*2
            time = raw_datetime[-n_test:]
        else:
            normal, low, high = self.operating_range
            three_sigma = abs(normal-low) if abs(normal-low)>abs(normal-high) else abs(normal-high)
            mu = normal
            sigma = three_sigma/3
            cdf = stats.norm.cdf(forecasts, loc=mu, scale=sigma)
            health_index = 1 - abs(cdf - 0.5) * 2
            time = raw_datetime[-n_test:]
        if self.save_info:
            # save health index to file
            print('save health index to csv starts...')
            df = pd.DataFrame({'time':time, 'prediction_value':np.squeeze(forecasts), 'health_index':np.squeeze(health_index)}, columns=['time','prediction_value','health_index'])
            df.to_csv(os.path.join(os.curdir,'health_index',self.sensor_name+'.csv'), sep=',', encoding='utf-8',index = False)
            print('save health index to csv done...')
