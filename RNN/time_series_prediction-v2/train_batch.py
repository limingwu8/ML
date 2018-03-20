# from RNN.time_series_prediction.Sensor import *
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import Sensor
import os
# configuration
save_info = 1       # 1: save information in file, 0: do not save
which_computer = 2    # 0: run on p219, 1: run on civs(windows), 2: run on civs(linux), 3: run on my own laptop(linux), 4: run on bc(linux)
train = 0          # 1: train model, 0: load model

n_lag = 1
n_epochs = 2000
dataset_path = os.path.join(os.curdir, 'dataset','csv', 'sampled')
if which_computer==4:
    root_path = '/home/bc/Documents/USS/compare-v2/'
elif which_computer==1:
    root_path = 'Y:\\USS-RF-Fan-Data-Analytics\\_13_Preliminary-results\\LSTM-preciction\\multi-step-prediction\\compare-v2\\'
elif which_computer==2:
    root_path = '/home/liming/Documents/USS-RF-Fan-Data-Analytics/_13_Preliminary-results/LSTM-preciction/multi-step-prediction/compare-v2/'
elif which_computer==0:
    root_path = '/home/PNW/wu1114/Documents/USS/compare-v2/'
else:
    root_path = ''

sensor_names = {
    'MAIN_FILTER_IN_PRESSURE','MAIN_FILTER_OIL_TEMP','MAIN_FILTER_OUT_PRESSURE','OIL_RETURN_TEMPERATURE',
    'TANK_FILTER_IN_PRESSURE','TANK_FILTER_OUT_PRESSURE','TANK_LEVEL','TANK_TEMPERATURE','FT-202B',
    'FT-204B','PT-203','PT-204'
}
# operating range for each sensor, (normal, low, high)
operating_ranges = {
    'MAIN_FILTER_IN_PRESSURE':(65,40,90),'MAIN_FILTER_OIL_TEMP':(110,90,130),'MAIN_FILTER_OUT_PRESSURE':(63,40,90),
    'OIL_RETURN_TEMPERATURE':(110,90,130),'TANK_FILTER_IN_PRESSURE':(20,10,30),'TANK_FILTER_OUT_PRESSURE':(18,10,30),
    'TANK_LEVEL':(19,14,23),'TANK_TEMPERATURE':(110,90,130),'FT-202B':(0.5,0,3.5),'FT-204B':(0.5,0,3.5),
    'PT-203':(0.5,0,3.5),'PT-204':(0.5,0,3.5)
}
sample_rates_n_seq = {
    'sample_1_day':(1,2)
}
for name in sensor_names:
    operating_range = operating_ranges[name]
    for j in sample_rates_n_seq:
        n_seqs = sample_rates_n_seq[j]
        sample_rate = j
        for s in n_seqs:
            if s!=1:
                break
            s = Sensor.Sensor(n_seq = s, n_epochs= n_epochs, dataset_path = dataset_path, sensor_name = name,
                              operating_range = operating_range,sample_rate = sample_rate, train = train,
                              root_path = root_path, save_info = save_info)
            if train == 1:
                s.run_train()   # train the network
            else:
                # s.get_health_score()  # load .h5 file and make prediction
                # s.get_all_health_score()
                s.load_model_and_predict()