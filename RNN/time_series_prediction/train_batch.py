from RNN.time_series_prediction.Sensor import *

# configuration
save_info = 0       # 1: save information in file, 0: do not save
which_computer = 1    # 0: run on server, 1: run on civs(windows), 2: run on civs(linux), 3: run on my own laptop(linux), 4: run on bc(linux)
train = 0           # 1: train model, 0: load model

n_lag = 1
n_epochs = 1
dataset_path = os.path.join(os.curdir, 'dataset')
if which_computer==4:
    root_path = '/home/bc/Documents/USS/compare/'
elif which_computer==1:
    root_path = 'Y:\\USS-RF-Fan-Data-Analytics\\_13_Preliminary-results\\LSTM-preciction\\multi-step-prediction\\compare\\'
else:
    root_path = ''

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
        for s in n_seqs:
            if s!=1:
                break
            s = Sensor(n_seq = s, n_epochs= n_epochs, dataset_path = dataset_path, sensor_name = name,
                       sample_rate = sample_rate, train = train, root_path = root_path, save_info = save_info)
            if train == 1:
                s.run_train()   # train the network
            else:
                s.load_model_and_predict()  # load .h5 file and make prediction

