"""
UNet
Common utility functions and classes
"""

import os
import sys
import numpy as np
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
import matplotlib.pylab as plt


# Base Configuration class
# Don't use this class directly. Instead, sub-class it and override

class Config():

    name = None

    img_width = 256
    img_height = 256

    img_channel = 3

    batch_size = 16

    learning_rate = 1e-3
    learning_momentum = 0.9
    weight_decay = 1e-4

    shuffle = False

    def __init__(self):
        self.IMAGE_SHAPE = np.array([
            self.img_width, self.img_height, self.img_channel
        ])

    def display(self):
        """Display Configuration values"""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")

# Configurations

class Option(Config):
    """Configuration for training on Kaggle Data Science Bowl 2018
    Derived from the base Config class and overrides specific values
    """
    name = "DSB2018"

    # root dir of training and validation set
    root_dir = '/home/liming/Documents/dataset/dataScienceBowl2018/combined'

    # root dir of testing set
    test_dir = '/home/liming/Documents/dataset/dataScienceBowl2018/stage1_test'
    num_workers = 1     # number of threads for data loading, set to 1 if using CUDA
    shuffle = True      # shuffle the data set
    batch_size = 1      # GTX1060 3G Memory
    is_train = True     # True for training, False for testing
    shuffle = True

    n_gpu = 1

    learning_rate = 1e-3
    weight_decay = 1e-4

    pin_memory = True   # store data in CPU pin buffer rather than memory. when using CUDA, set to True

    is_cuda = torch.cuda.is_available()  # True --> GPU
    num_gpus = torch.cuda.device_count()  # number of GPUs
    checkpoint_dir = ""  # dir to save checkpoints
    dtype = torch.cuda.FloatTensor if is_cuda else torch.Tensor  # data type

"""
Dataset orgnization:
Read images and masks, combine separated mask into one
Write images and combined masks into specific folder
"""
class Utils(object):
    """
    Initialize image parameters from DSB2018Config class
    """
    def __init__(self, src_root_path, dest_root_path):
        self.opt = Option
        self.src_root_path = src_root_path
        self.dest_root_path = dest_root_path

    # Combine all separated masks into one mask
    def assemble_masks(self, path):
        # mask = np.zeros((self.config.IMG_HEIGHT, self.config.IMG_WIDTH), dtype=np.uint8)
        mask = None
        for i, mask_file in enumerate(next(os.walk(os.path.join(path, 'masks')))[2]):
            mask_ = Image.open(os.path.join(path, 'masks', mask_file)).convert("RGB")
            # mask_ = mask_.resize((self.config.IMG_HEIGHT, self.config.IMG_WIDTH))
            mask_ = np.asarray(mask_)
            if i == 0:
                mask = mask_
                continue
            mask = mask | mask_
        # mask = np.expand_dims(mask, axis=-1)
        return mask

    #
    def prepare_training_data(self):
        '''
        Args:
            file_dir: root file directory
        Returns:
            list of training images, in ndarray, e.g. (560,256,256,4)
            list of training labels, in ndarray, e.g. (560,256,256,1)
        '''
        # get imageId
        train_ids = next(os.walk(self.src_root_path))[1]

        # read training data
        X_train = []
        Y_train = []
        print('reading training data starts...')
        sys.stdout.flush()
        for n, id_ in tqdm(enumerate(train_ids)):
            path = os.path.join(self.src_root_path, id_)
            img = Image.open(os.path.join(path, 'images', id_ + '.png')).convert("RGB")
            # img = img.resize((self.config.IMG_HEIGHT, self.config.IMG_WIDTH))
            # X_train[n] = img
            # Y_train[n] = self.assemble_masks(path)
            X_train.append(np.asarray(img))
            Y_train.append(self.assemble_masks(path))

        self.X_train = X_train
        self.Y_train = Y_train
        self.train_ids = train_ids
        print('reading training data done...')

    # save image and combined masks into a new folder
    def save_training_data(self):
        print('writing training data starts...')
        for i in tqdm(range(0, len(self.train_ids))):
            # create directories if they are not exists
            id = self.train_ids[i]
            id_dir = os.path.join(self.dest_root_path, id)
            if not os.path.exists(id_dir):
                os.mkdir(id_dir)

            image = self.X_train[i]
            mask = self.Y_train[i]
            Image.fromarray(image).save(os.path.join(id_dir, 'image.png'))
            Image.fromarray(mask).save(os.path.join(id_dir, 'mask.png'))

        print('writing training data done...')

    def prepare_testing_data(self):
        '''
        Args:
            file_dir: root file directory
        Returns:
            list of training images, in ndarray, e.g. (560,256,256,4)
            list of training labels, in ndarray, e.g. (560,256,256,1)
        '''
        # get imageId
        train_ids = next(os.walk(self.src_root_path))[1]

        # read training data
        X_train = []
        Y_train = []
        print('reading training data starts...')
        sys.stdout.flush()
        for n, id_ in tqdm(enumerate(train_ids)):
            path = os.path.join(self.src_root_path, id_)
            img = Image.open(os.path.join(path, 'images', id_ + '.png')).convert("RGB")
            # img = img.resize((self.config.IMG_HEIGHT, self.config.IMG_WIDTH))
            # X_train[n] = img
            # Y_train[n] = self.assemble_masks(path)
            X_train.append(np.asarray(img))
            Y_train.append(self.assemble_masks(path))

        self.X_train = X_train
        self.Y_train = Y_train
        self.train_ids = train_ids
        print('reading training data done...')

    # save image and combined masks into a new folder
    def save_testing_data(self):
        print('writing training data starts...')
        for i in tqdm(range(0, len(self.train_ids))):
            # create directories if they are not exists
            id = self.train_ids[i]
            id_dir = os.path.join(self.dest_root_path, id)
            if not os.path.exists(id_dir):
                os.mkdir(id_dir)

            image = self.X_train[i]
            mask = self.Y_train[i]
            Image.fromarray(image).save(os.path.join(id_dir, 'image.png'))
            Image.fromarray(mask).save(os.path.join(id_dir, 'mask.png'))

        print('writing training data done...')

if __name__ == '__main__':
    """ Prepare training data
    read data from src_root_path, save combined data to dest_root_path
    """
    src_root_path = '/home/liming/Documents/dataset/dataScienceBowl2018/stage1_train'
    dest_root_path = '/home/liming/Documents/dataset/dataScienceBowl2018/combined'
    util = Utils(src_root_path, dest_root_path)
    util.prepare_training_data()
    util.save_training_data()

    """ Prepare testing data
    read data from src_root_path, save combined data to dest_root_path
    """
    src_root_path = '/home/liming/Documents/dataset/dataScienceBowl2018/stage1_train'
    dest_root_path = '/home/liming/Documents/dataset/dataScienceBowl2018/testing_data'
    util = Utils(src_root_path, dest_root_path)
    util.prepare_training_data()
    util.save_training_data()