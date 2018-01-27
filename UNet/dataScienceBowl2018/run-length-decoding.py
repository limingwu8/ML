import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# revised from https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background, this is only one submask of the mask

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')

def rle_to_mask(image_id, rle, size):
    '''

    :param image_id: image id
    :param rle: rles corresponding to the image id
    :param size: image size corresponding to the image id
    :return: mask corresponding to the image id
    '''
    mask = np.zeros((size[0],size[1]), dtype = np.bool)
    for r in rle:
        mask = mask | rle_decode(r, size)
    return mask

def csv_to_mask(csv_path, image_ids, sizes):
    '''

    :param csv_path: the path of run-length-encoding CSV file
    :param image_ids: image id
    :param sizes: the original image size corresponding to image id
    :return: masks of all image id
    '''
    df = pd.read_csv(csv_path)
    df = df.values
    imageIds_all = np.squeeze(df[:,0])
    rles = np.squeeze(df[:,1])
    masks = []
    for n, image_id in enumerate(image_ids):
        index = np.where(imageIds_all == image_id)
        rle = rles[index]
        size = sizes[n]
        masks.append(rle_to_mask(image_id, rle, size))
    return masks

if __name__ == '__main__':
    # prepare testing data, for convenience, I just read from pickle
    pkl_file = open('size_test.pkl', 'rb')
    image_ids = pickle.load(pkl_file)
    sizes = pickle.load(pkl_file)
    pkl_file.close()

    csv_path = '/root/PycharmProjects/ML/UNet/dataScienceBowl2018/sub-dsbowl2018-4.csv'
    masks = csv_to_mask(csv_path,image_ids, sizes)

