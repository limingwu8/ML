import scipy.io
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image


# read mat file
mat = scipy.io.loadmat('dtidata.mat')

dtidata = mat['dtidata']
root = '/home/liming/Documents/dataset/dtidata'

print(dtidata.shape)

for i in range(dtidata.shape[0]):
    for j in range(dtidata.shape[3]):
        A = dtidata[i,:,:,j].astype(np.uint8)
        im = Image.fromarray(A)
        im.save(os.path.join(root, 'img'+str(i)+str(j)+'.png'))
        # plt.imshow(A)
        # plt.show()