import os
import sys
import random
import warnings
import re

import numpy as np
import pandas as pd


from tqdm import tqdm
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from PIL import Image
from scipy.misc import imfilter


# Set some parameters
IMG_W = 128
IMG_H = 128
IMG_CHANNELS = 3
TRAIN_PATH = '/root/dataset/dataScienceBowl2018/stage1_train/'
TEST_PATH = '/root/dataset/dataScienceBowl2018/stage1_test/'

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed

import os
import re
IMG_WIDTH= 128
IMG_HEIGHT = 128
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def assemble_masks(path):
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.bool)
    for mask_file in next(os.walk(os.path.join(path, 'masks')))[2]:
        mask_ = Image.open(os.path.join(path, 'masks', mask_file))
        mask_ = mask_.resize((IMG_HEIGHT, IMG_WIDTH))
        mask_ = np.asarray(mask_, dtype = np.bool)
        mask = mask | mask_
    mask = np.expand_dims(mask, axis=-1)
    return mask

TEST_PATH = '/root/dataset/dataScienceBowl2018/stage1_test/'
train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]

# # read training data
X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
for n, id_ in enumerate(train_ids):
    path = os.path.join(TRAIN_PATH + id_)
    img = Image.open(os.path.join(path, 'images', id_ + '.png')).convert("RGB")
    img = img.resize((IMG_HEIGHT, IMG_WIDTH))
    X_train[n] = img
    Y_train[n] = assemble_masks(path)

# read testing data
# sizes_test = []
# X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
# for n, id_ in enumerate(test_ids):
#     path = os.path.join(TEST_PATH, id_, 'images', id_ + '.png')
#     img = Image.open(path).convert("RGB")
#     sizes_test.append(img.size)
#     img = img.resize((IMG_HEIGHT, IMG_WIDTH))
#     X_test[n] = img

def get_contour(X, Y):
    '''
    :param images: images, ndarray, e.g.(500,256,256,3)
    :param labels: masks, ndarray, e.g.(500,256,256,1)
    :return:
    images with contour, e.g.(500,256,256,3)
    '''
    num_imgs = X.shape[0]
    contour_images = np.zeros((num_imgs, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    contour_labels = np.zeros((num_imgs, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    contours = np.zeros((num_imgs, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

    for i in range(0, num_imgs):
        x = X[i]
        y = Y[i]
        c = imfilter(np.squeeze(y), "find_edges")
        c = np.expand_dims(c, axis=-1)
        contour_images[i] = (x | c)
        contour_labels[i] = (y | c)
        contours[i] = c

    return contour_images, contour_labels, contours
X_train_contour, Y_train_contour, contour = get_contour(X_train, Y_train)
print()