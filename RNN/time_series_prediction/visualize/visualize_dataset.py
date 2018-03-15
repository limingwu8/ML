from pandas import Series
from pandas import read_csv
from pandas import datetime
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import datetime
import os

operating_range = {
    'MAIN_FILTER_IN_PRESSURE':(65,40,90),'MAIN_FILTER_OIL_TEMP':(110,90,130),'MAIN_FILTER_OUT_PRESSURE':(63,40,90),
    'OIL_RETURN_TEMPERATURE':(110,90,130),'TANK_FILTER_IN_PRESSURE':(20,10,30),'TANK_FILTER_OUT_PRESSURE':(18,10,30),
    'TANK_LEVEL':(19,14,23),'TANK_TEMPERATURE':(110,90,130),'FT-202B':(0.5,0,3.5),'FT-204B':(0.5,0,3.5),
    'PT-203':(0.5,0,3.5),'PT-204':(0.5,0,3.5)
}


def load_dataset(dataset_path):
    series = read_csv(dataset_path, sep=',')
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

def get_paths():
    root = 'C:\\Users\\wu1114\\PycharmProjects\\ML\\RNN\\time_series_prediction\\dataset\\csv\\sampled\\sample_1_day\\'
    files = os.listdir(root)
    paths = [(root + s) for s in files ]
    return paths

def plot_save(time, value, sensor_name):
    label_fontsize = 35
    legend_fontsize = 30
    axis_fontsize = 30
    linewidth = 5

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    # make x label in a specific format
    ax1.xaxis_date()
    ax1.xaxis.set_major_formatter(DateFormatter('%m-%d'))
    plt.plot(time,value, linewidth=linewidth, label = 'sensor data')
    plt.xlabel('Time', fontsize=label_fontsize)
    plt.ylabel('Value', fontsize=label_fontsize)
    plt.title(sensor_name, fontsize=label_fontsize)
    plt.xticks(fontsize=axis_fontsize)
    plt.yticks(fontsize=axis_fontsize)
    fig.set_size_inches(18.5, 10.5)
    # draw operating ranges
    normal, low, high = operating_range[sensor_name]
    plt.axhline(y=normal, color='g', linestyle='-',linewidth=linewidth, label = 'normal')
    plt.axhline(y=low, color='orange', linestyle='-',linewidth=linewidth, label = 'low')
    plt.axhline(y=high, color='r', linestyle='-',linewidth=linewidth, label = 'high')
    plt.legend(fontsize=legend_fontsize)

    fig.savefig(sensor_name+'.png', bbox_inches='tight', dpi=150)


if __name__ == '__main__':
    paths = get_paths()
    for path in paths:
        series, series_values, raw_datetime = load_dataset(path)
        sensor_name = os.path.basename(path).split('.')[0]
        plot_save(raw_datetime, series_values,sensor_name)
    print()
