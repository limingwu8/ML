import pprint, pickle
from sklearn.metrics import mean_squared_error
from math import sqrt

pkl_file = open('/home/bc/Pictures/multi-step-prediction/neuron=10/data.pkl','rb')

data1 = pickle.load(pkl_file)
pprint.pprint(data1)

data2 = pickle.load(pkl_file)
pprint.pprint(data2)
print('rmse:',sqrt(mean_squared_error(data1, data2)))
pkl_file.close()