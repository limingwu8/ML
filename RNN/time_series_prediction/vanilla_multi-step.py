from pandas import DataFrame
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot
import datetime


# date-time parsing function for loading the dataset
def parser(x):
    return datetime.strptime('190' + x, '%Y-%m')


# convert time series into supervised learning problem
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
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


# transform series into train and test sets for supervised learning
def prepare_data(series, n_test, n_lag, n_seq):
    # extract raw values
    raw_values = series.values
    raw_values = raw_values.reshape(len(raw_values), 1)
    # transform into supervised learning problem X, y
    supervised = series_to_supervised(raw_values, n_lag, n_seq)
    supervised_values = supervised.values
    # split into train and test sets
    train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
    return train, test


# make a persistence forecast
def persistence(last_ob, n_seq):
    return [last_ob for i in range(n_seq)]


# evaluate the persistence model
def make_forecasts(train, test, n_lag, n_seq):
    forecasts = list()
    for i in range(len(test)):
        X, y = test[i, 0:n_lag], test[i, n_lag:]
        # make forecast
        forecast = persistence(X[-1], n_seq)
        # store the forecast
        forecasts.append(forecast)
    return forecasts


# evaluate the RMSE for each forecast time step
def evaluate_forecasts(test, forecasts, n_lag, n_seq):
    for i in range(n_seq):
        actual = test[:, (n_lag + i)]
        predicted = [forecast[i] for forecast in forecasts]
        rmse = sqrt(mean_squared_error(actual, predicted))
        print('t+%d RMSE:%f' % ((i + 1), rmse))


# plot the forecasts in the context of the original dataset
def plot_forecasts(series, forecasts, n_test):
    # plot the entire dataset in blue
    pyplot.plot(series.values)
    # plot the forecasts in red
    for i in range(len(forecasts)):
        off_s = len(series) - n_test + i - 1
        off_e = off_s + len(forecasts[i]) + 1
        xaxis = [x for x in range(off_s, off_e)]
        yaxis = [series.values[off_s]] + forecasts[i]
        pyplot.plot(xaxis, yaxis, color='red')
    # show the plot
    pyplot.show()


def run_train(n_seq, dataset_path, sensor_name, sample_rate, root_path,save_info):
    dataset_path += sample_rate + '/' + sensor_name + '.csv'
    sensor_name += '-' + sample_rate
    print('RMSE of ', sensor_name)

    # load dataset
    series = read_csv(dataset_path, header=0, parse_dates=[0], index_col=0, squeeze=True)

    # configure
    n_test = int(0.2 * series.shape[0])  # prepare data
    train, test = prepare_data(series, n_test, n_lag, n_seq)
    # make forecasts
    forecasts = make_forecasts(train, test, n_lag, n_seq)
    # evaluate forecasts
    evaluate_forecasts(test, forecasts, n_lag, n_seq)
    # plot forecasts
    # plot_forecasts(series, forecasts, n_test + n_seq-1)  # load dataset


# sensor units
units = {'MAIN_FILTER_IN_PRESSURE':'PSI','MAIN_FILTER_OIL_TEMP':'Fahrenheit',
         'MAIN_FILTER_OUT_PRESSURE':'PSI','OIL_RETURN_TEMPERATURE':'Fahrenheit',
         'TANK_FILTER_IN_PRESSURE':'PSI','TANK_FILTER_OUT_PRESSURE':'PSI',
         'TANK_LEVEL':'Inch','TANK_TEMPERATURE':'Fahrenheit','FT-202B':'Mils',
         'FT-204B':'Mils','PT-203':'Mils','PT-204.HS':'Mils'}

save_info = 1       # 1: save information in file, 0: do not save
run_on_local = 1    # 1: run on local, 0: run on server

n_lag = 1
dataset_path = './dataset/'
root_path = '/home/bc/Documents/USS/compare/'

sensor_names = {
    'MAIN_FILTER_IN_PRESSURE','MAIN_FILTER_OIL_TEMP','MAIN_FILTER_OUT_PRESSURE','OIL_RETURN_TEMPERATURE',
    'TANK_FILTER_IN_PRESSURE','TANK_FILTER_OUT_PRESSURE','TANK_LEVEL','TANK_TEMPERATURE','FT-202B',
    'FT-204B','PT-203','PT-204.HS'
}
sample_rates_n_seq = {
    'sample_1_hour':(1,48), 'sample_6_hour':(1,8), 'sample_12_hour':(1,4),
    'sample_18_hour':(1,2), 'sample_1_day':(1,2)
}
for name in sensor_names:
    for j in sample_rates_n_seq:
        n_seqs = sample_rates_n_seq[j]
        sample_rate = j
        for n_seq in n_seqs:
            run_train(n_seq = n_seq, dataset_path = dataset_path, sensor_name = name,
                       sample_rate = sample_rate, root_path = root_path, save_info = save_info)