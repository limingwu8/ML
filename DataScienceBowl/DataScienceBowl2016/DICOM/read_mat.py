import scipy.io
import matplotlib.pylab as plt
import os
import numpy as np
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

path = '/home/liming/Documents/dataset/data'
files = get_files(path)
for file in files:
    print(file)

ds = []
for i in range(len(files)):
    temp = scipy.io.loadmat(files[i])
    img = sol_yxzt
    ds.append(temp)
    plt.imshow(temp)
plt.show()

#
# print(ds.PatientName)
# print(ds.PixelData)
# print(ds.pixel_array)
