# write to TFRecords and read from TFRecords
import tensorflow as tf
import numpy as np
import os
import re
import math
import matplotlib.pyplot as plt
import skimage.io as io
import scipy.misc
from PIL import Image
from PKLot.mark_image import *
from PKLot.data_augmentation import *


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


def get_batch(image_path, label_path, image_W, image_H, batch_size, capacity):
    '''
    Args:
        image: list type
        label: list type
        image_W: image width
        image_H: image height
        batch_size: batch size
        capacity: the maximum elements in queue
    Returns:
        image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
        label_batch: 1D tensor [batch_size], dtype=tf.int32
    '''

    image = tf.cast(image_path, tf.string)
    label = tf.cast(label_path, tf.int32)

    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label])

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=1)

    ######################################
    # data argumentation should go to here
    ######################################

    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)

    # if you want to test the generated batches of images, you might want to comment the following line.
    # 如果想看到正常的图片，请注释掉111行（标准化）和 126行（image_batch = tf.cast(image_batch, tf.float32)）
    # 训练时不要注释掉！
    image = tf.image.per_image_standardization(image)

    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=64,
                                              capacity=capacity)

    # you can also use shuffle_batch
    #    image_batch, label_batch = tf.train.shuffle_batch([image,label],
    #                                                      batch_size=BATCH_SIZE,
    #                                                      num_threads=64,
    #                                                      capacity=CAPACITY,
    #                                                      min_after_dequeue=CAPACITY-1)

    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)

    return image_batch, label_batch


def int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_tfrecord(images,labels,save_dir,name):
    ''' convert all images and labels to one tfrecord file
    :param images: list of image directories, string type
    :param labels: list of labels, int type
    :param save_dir: the directory to save tfrecord file
    :param name: the name of tfrecord file
    :return: no return
    '''

    filename = os.path.join(save_dir, name + '.tfrecords')
    n_samples = len(labels)
    if np.shape(images)[0] != n_samples:
        raise ValueError('Images size does not match label size.')

    # starts to transform, needs some time...
    writer = tf.python_io.TFRecordWriter(filename)
    print('\nTransform start...')
    for i in np.arange(0,n_samples):
        try:
            # type image must be ndarray
            img = Image.open(images[i])
            img = img.resize((28,28))
            img = img.convert("L")
            img = np.array(img.getdata(), dtype='uint8').reshape(img.size[0],img.size[1])

            image_raw = img.tostring()
            label = int(labels[i])
            example = tf.train.Example(
                features = tf.train.Features(
                    feature = {'label':int64_feature(label),
                               'image_raw': bytes_feature(image_raw)}
                )
            )
            writer.write(example.SerializeToString())
        except IOError as e:
            print('Could not read:', images[i])
            print('error: %s' %e)
            print('skip it!\n')
    writer.close()
    print('Transform done!')

def read_and_decode(tfrecords_file, batch_size):
    ''' read and decode tfrecord file, generate (image, label) batches
    :param tfrecords_file: the directory of tfrecord file
    :param batch_size: number of images in each batch
    :return:
        image: 4D tensor - [batch_size, width, height, channel]
        label: 1D tensor - [batch_size]
    '''
    # make an input Queue from the tfrecord file
    filename_queue = tf.train.string_input_producer([tfrecords_file], num_epochs=None)

    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(filename_queue)
    img_features = tf.parse_single_example(
        serialized_example,
        features={
            'label':tf.FixedLenFeature([],tf.int64),
            'image_raw':tf.FixedLenFeature([],tf.string),
        }
    )
    image = tf.decode_raw(img_features['image_raw'], tf.uint8)

    #######################################################
    # you can put data augmentation here, I didn't use it #
    ###################################################################

    # this image size should be the same as the format in .tfrecords file
    image = tf.reshape(image, [28, 28, 1])
    image = tf.cast(image, tf.float32)
    image = tf.image.per_image_standardization(image)
    label = tf.cast(img_features['label'], tf.int32)
    image_batch, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=64,
        capacity=2000
    )

    # augment_data(image_batch, label_batch)

    return image_batch, tf.reshape(label_batch, [batch_size])

def plot_images(images, labels, title):
    '''plot one batch size'''
    for i in np.arange(0, BATCH_SIZE):
        image = images[i].reshape(28,28)
        plt.subplot(1,BATCH_SIZE,i+1)
        plt.axis('off')
        plt.title(chr(ord('0') + labels[i]), fontsize=14)
        plt.subplots_adjust(top=1.5)
        plt.imshow(image)
    plt.show()

if __name__ == '__main__':


    dataset_path = '/home/bc/Documents/datasets/PKLot/PKLotSegmented'
    dataset_path = '/home/bc/Documents/datasets/PKLot/PKLotSegmented'
    tfrecords_path = '/home/bc/Documents/datasets/PKLot/tfrecords'

    train_tfrecords_name = 'train'
    val_tfrecords_name = 'val'

    BATCH_SIZE = 10
    val_ratio = 0.2


    train_image_list, train_label_list, val_image_list, val_label_list = get_files(dataset_path,val_ratio)
    # write training data to a tfrecords, write validation data to another tfrecords
    convert_to_tfrecord(train_image_list,train_label_list,tfrecords_path,train_tfrecords_name)
    convert_to_tfrecord(val_image_list, val_label_list, tfrecords_path, val_tfrecords_name)


    # read data from tfrecords and display them
    image_batch, label_batch = read_and_decode(os.path.join(tfrecords_path, train_tfrecords_name + '.tfrecords'), BATCH_SIZE)
    image_batch, label_batch = read_and_decode(os.path.join(tfrecords_path, val_tfrecords_name + '.tfrecords'), BATCH_SIZE)

    with tf.Session() as sess:
        i = 0
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            while not coord.should_stop() and i<1:
                image, label = sess.run([image_batch, label_batch])
                plot_images(image, label, 'shuffled train images')
                i+=1
        except tf.errors.OutOfRangeError:
            print('done!')
        finally:
            coord.request_stop()
        coord.join(threads)
