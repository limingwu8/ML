import os
import re
IMG_W = 128
IMG_H = 128
from PIL import Image
import numpy as np
TEST_PATH = '/root/dataset/dataScienceBowl2018/stage1_test/'

# read testing data
test_dims = []
test_ids = []
X_test = []
temp = []
for root, dirs, files in os.walk(TEST_PATH):
    parent_dir_name = re.split(r'/', root)[-1]
    if parent_dir_name == 'images':
        img = Image.open(os.path.join(root, file))
        test_dims.append(img.size)
        test_ids.append(file)
        img = img.resize((IMG_W, IMG_H))
        X_train.append(np.asarray(img))
        temp.clear()