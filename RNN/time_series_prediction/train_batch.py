from RNN.time_series_prediction.Sensor import *


save_info = 1       # 1: save information in file, 0: do not save
run_on_local = 1    # 1: run on local, 0: run on server
train = 1           # 1: train model, 0: load model

n_lag = 1
n_epochs = 1500
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
    if name.__eq__('MAIN_FILTER_OIL_TEMP'):
        for j in sample_rates_n_seq:
            if j == 'sample_1_hour':
                n_seqs = sample_rates_n_seq[j]
                sample_rate = j
                for s in n_seqs:
                    s = Sensor(n_seq = s, n_epochs= n_epochs, dataset_path = dataset_path, sensor_name = name,
                               sample_rate = sample_rate, root_path = root_path, save_info = 1)
                    s.run_train()
