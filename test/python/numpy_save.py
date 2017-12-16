import numpy as np

# save to npy
# a = np.arange(8)
# b = np.array([11,22,33,44,55,66])
# c = ['a','b','c','d']
#
# with open('test.npy','wb') as f:
#     np.save(f,a)
#     np.save(f,b)
#     np.save(f,c)

a = np.array({'fuc':np.array([1,2,3,4,]),'conv':np.array([[[1,3,7,8,9]]])})
with open('test.npy','wb') as f:
    np.save(f,a)

# read from npy
data_path = 'test.npy'

data_dict = np.load(data_path, encoding='latin1').item()
keys = sorted(data_dict.keys())

for key in keys:
    print(key)