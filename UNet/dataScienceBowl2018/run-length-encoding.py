import numpy as np
import pickle
from skimage.morphology import label
import pandas as pd
import matplotlib.pyplot as plt

# Run-length encoding revised from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
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
    '''
        x is a image with value between 0 and 1, choose value > cutoff as masks
    :param x: masks
    :param cutoff: threshold
    :return: masks, True and False matrix
    '''
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)

def img_to_rles(image_ids, masks):
    '''
        convert masks to rles
    :param image_ids: imageIds, list
    :param masks: (W,H) images with values between 0 and 1
    :return: imageIds and rles corresponding to imageIds
    '''
    new_test_ids = []
    rles = []
    for n, id_ in enumerate(image_ids):
        rle = list(prob_to_rles(masks[n]))
        rles.extend(rle)
        new_test_ids.extend([id_] * len(rle))
    return new_test_ids, rles

def rles_to_csv(ids, rles):
    '''
        convert ids and rles to csv
    :param ids: imageIds, list
    :param rles: run-length encodings, list
    '''
    sub = pd.DataFrame()
    sub['ImageId'] = ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
    sub.to_csv('test.csv', index=False)

if __name__ == '__main__':
    # prepare testing data, for convenience, I just read from pickle
    pkl_file = open('data.pkl', 'rb')
    test_ids = pickle.load(pkl_file)
    preds_test_upsampled = pickle.load(pkl_file)
    pkl_file.close()
    #############################################################
    # test rle algorithm, before testing the following function #
    # on your own dataset, make sure test_ids are image ids and #
    # preds_test_upsampled are prediction results from UNet     #
    ids, rles = img_to_rles(test_ids, preds_test_upsampled)     #
    rles_to_csv(ids, rles)                                      #
    #############################################################