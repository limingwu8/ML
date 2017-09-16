from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, adam
from keras.utils import np_utils

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from sklearn.utils import shuffle
from PIL import Image

dir1 = "F://datasets//inputData"
dir2 = "F://datasets//inputDataResized"
imgRows , imgCols = 200, 200

files = os.listdir("F://datasets//cats")
num_files = len(files)

# resize file
for file in files:
    im = Image.open(dir1 + '\\' + file)
    img = im.resize((imgRows, imgCols))
    gray = img.convert("L")
    gray.save(dir2 + '\\' + file, "JPEG")