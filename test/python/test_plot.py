from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
import matplotlib.pyplot as plt

series = read_csv('../../RNN/short_term_prediction/sensor_data2.csv', header=0)

plt.plot(series)

plt.show()