# write to TFRecords and read from TFRecords
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import skimage.io as io

def get_file(file_dir):
    '''Get full image directory and corresponding labels
    :param file_dir: file directory
    :return:
        images: image directories, list, string
        labels: label, list, int
    '''
    images = []
    temp = []
    for root, dirs, files in os.walk(file_dir):
        # image directories
        for file in files:
            images.append(os.path.join(root,file))
        for dir in dirs:
            temp.append(os.path.join(root,dir))
    # assign labels based on the folder name
    labels = []
    for folder in temp:
        n_img = len(os.listdir(folder))
        letter = folder.split('\\')[-1]

        if letter=='0':
            labels = np.append(labels, n_img*[0])
        else:
            labels = np.append(labels,n_img*[1])
    # put the images and labels and shuffle them
    temp = np.array([images,labels])
    temp = temp.transpose()
    np.random.shuffle(temp)

    image_list = list(temp[:,0])
    label_list = list(temp[:,1])
    label_list = [int(float(i)) for i in label_list]

    return image_list, label_list

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
            image = io.imread(images[i])    # type image much be ndarray
            image_raw = image.tostring()
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
    filename_queue = tf.train.string_input_producer([tfrecords_file])

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

    # all the images are 200*200, you can change the image size here.
    image = tf.reshape(image, [200, 200])
    label = tf.cast(img_features['label'], tf.int32)
    image_batch, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=64,
        capacity=2000
    )
    return image_batch, tf.reshape(label_batch, [batch_size])

def plot_images(images, labels, title):
    '''plot one batch size'''
    for i in np.arange(0, BATCH_SIZE):
        plt.subplot(1,5,i+1)
        plt.axis('off')
        plt.title(chr(ord('0') + labels[i]-1), fontsize=14)
        plt.subplots_adjust(top=1.5)
        plt.imshow(images[i])
    plt.show()
if __name__ == '__main__':
    BATCH_SIZE = 5
    cwd = os.getcwd()
    print('current work directory',cwd)
    folder = cwd + '\\data'
    # image_list, label_list = get_file(folder)
    # print(image_list)
    # print(label_list)
    # convert_to_tfrecord(image_list,label_list,folder,'dataset')
    image_batch, label_batch = read_and_decode(folder + '\\dataset.tfrecords', BATCH_SIZE)

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
