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
from skimage.morphology import label
import pandas as pd
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
    test_dir = '/home/liming/Documents/dataset/dataScienceBowl2018/testing_data'

    # save segmenting results (prediction masks) to this folder
    results_dir = '/home/liming/Documents/dataset/dataScienceBowl2018/results'

    num_workers = 1     	# number of threads for data loading
    shuffle = True      	# shuffle the data set
    batch_size = 2     		# GTX1060 3G Memory
    epochs = 150			# number of epochs to train
    is_train = False     	# True for training, False for making prediction
    save_model = True   	# True for saving the model, False for not saving the model

    n_gpu = 1				# number of GPUs

    learning_rate = 1e-3	# learning rage
    weight_decay = 1e-4		# weight decay

    pin_memory = True   	# use pinned (page-locked) memory. when using CUDA, set to True

    is_cuda = torch.cuda.is_available()  	# True --> GPU
    num_gpus = torch.cuda.device_count()  	# number of GPUs
    checkpoint_dir = "./checkpoint"  		# dir to save checkpoints
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
    def __init__(self, stage1_train_src, stage1_train_dest, stage1_test_src, stage1_test_dest):
        self.opt = Option
        self.stage1_train_src = stage1_train_src
        self.stage1_train_dest = stage1_train_dest
        self.stage1_test_src = stage1_test_src
        self.stage1_test_dest = stage1_test_dest

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

    # read all training data and save them to other folder
    def prepare_training_data(self):
        # get imageId
        train_ids = next(os.walk(self.stage1_train_src))[1]

        # read training data
        X_train = []
        Y_train = []
        print('reading training data starts...')
        sys.stdout.flush()
        for n, id_ in tqdm(enumerate(train_ids)):
            path = os.path.join(self.stage1_train_src, id_)
            dest = os.path.join(self.stage1_train_dest, id_)
            img = Image.open(os.path.join(path, 'images', id_ + '.png')).convert("RGB")
            mask = self.assemble_masks(path)
            img.save(os.path.join(dest, 'image.png'))
            Image.fromarray(mask).save(os.path.join(dest, 'mask.png'))

        print('reading training data done...')

    # read testing data and save them to other folder
    def prepare_testing_data(self):
        # get imageId
        test_ids = next(os.walk(self.stage1_test_src))[1]
        # read training data
        print('reading testing data starts...')
        sys.stdout.flush()
        for n, id_ in tqdm(enumerate(test_ids)):
            path = os.path.join(self.stage1_test_src, id_, 'images', id_+'.png')
            dest = os.path.join(self.stage1_test_dest, id_)
            if not os.path.exists(dest):
                os.mkdir(dest)
            img = Image.open(path).convert("RGB")
            img.save(os.path.join(dest, 'image.png'))

        print('reading testing data done...')

# Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)

def encode_and_save(preds_test_upsampled, test_ids):
    """
    Use run-length-encoding encode the prediction masks and save to csv file for submitting
    :param preds_test_upsampled: list, for each elements, numpy array (Width, Height)
    :param test_ids: list, for each elements, image id
    :return:
        save to csv file
    """
    new_test_ids = []
    rles = []
    for n, id_ in enumerate(test_ids):
        rle = list(prob_to_rles(preds_test_upsampled[n]))
        rles.extend(rle)
        new_test_ids.extend([id_] * len(rle))

    sub = pd.DataFrame()
    sub['ImageId'] = new_test_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
    sub.to_csv('sub-dsbowl2018.csv', index=False)

if __name__ == '__main__':
    """ Prepare training data and testing data
    read data and overlay masks and save to destination path
    """
    stage1_train_src = '/home/liming/Documents/dataset/dataScienceBowl2018/stage1_train'
    stage1_train_dest = '/home/liming/Documents/dataset/dataScienceBowl2018/combined'
    stage1_test_src = '/home/liming/Documents/dataset/dataScienceBowl2018/stage1_test'
    stage1_test_dest = '/home/liming/Documents/dataset/dataScienceBowl2018/testing_data'

    util = Utils(stage1_train_src, stage1_train_dest, stage1_test_src, stage1_test_dest)
    util.prepare_training_data()
    util.prepare_testing_data()