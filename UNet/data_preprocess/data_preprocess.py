
# coding: utf-8

# In[9]:

import os
import re
import math
import numpy as np


# In[10]:

def get_files(file_dir, val_ratio):
    '''
    Args:
        file_dir: file directory
        val_radio: the ratio of validation data, e.g. 0.2
    Returns:
        list of training images , training labels, validation images and validation labels
    '''
    images = []
    temp = []
    for root, dirs, files in os.walk(file_dir):
        # image directories
        for file in files:
            images.append(os.path.join(root,file))
        label = re.split(r'/',root)[-1]
        if (label == 'Empty') or (label == 'Occupied'):
            temp.append(root)
    # assign labels based on the folder name
    labels = []
    for folder in temp:
        n_img = len(os.listdir(folder))
        lastword = folder.split('/')[-1]

        if lastword=='Empty':
            labels = np.append(labels, int(n_img)*[0])
        else:
            labels = np.append(labels,int(n_img)*[1])

    # put the images and labels and shuffle them
    temp = np.array([images,labels])
    temp = temp.transpose()
    np.random.shuffle(temp)

    all_image_list = list(temp[:, 0])
    all_label_list = list(temp[:, 1])

    # just use 500 images to test, when training, delete the following two lines
    # all_image_list = all_image_list[:30000]
    # all_label_list = all_label_list[:30000]

    # split data to training data and validation data
    n_sample = len(all_label_list)  # number of all samples
    n_val = math.ceil(n_sample * val_ratio)  # number of validation samples
    n_train = n_sample - n_val  # number of training samples

    tra_images = all_image_list[0:n_train]
    tra_labels = all_label_list[0:n_train]
    tra_labels = [int(float(i)) for i in tra_labels]
    val_images = all_image_list[n_train:]
    val_labels = all_label_list[n_train:]
    val_labels = [int(float(i)) for i in val_labels]

    return tra_images, tra_labels, val_images, val_labels


# In[11]:

file_dir = '/root/dataset/dataScienceBowl2018/stage1_train'
val_ratio = 0.1
get_files(file_dir, val_ratio)


# In[ ]:



