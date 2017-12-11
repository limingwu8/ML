import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import xml.etree.cElementTree as ET
import tensorflow as tf

img_path = '/home/bc/Documents/datasets/PKLot/PKLot/PUCPR/Cloudy/2012-09-12/2012-09-12_07_34_01.jpg'
xml_path = '/home/bc/Documents/datasets/PKLot/PKLot/PUCPR/Cloudy/2012-09-12/2012-09-12_07_34_01.xml'


def fill_color(img_path,xml_path):

    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)

    # get xml tree
    tree = ET.ElementTree(file=xml_path)

    # get position of each slot
    position = [{'x':int(elem.attrib['x']),'y':int(elem.attrib['y'])} for elem in tree.iter(tag = 'point')]
    position = tuple(position[i:i+4] for i in range(0,len(position),4))

    # get label of each slot
    labels = tuple(int(elem.attrib['occupied']) for elem in tree.iter(tag = 'space'))

    for i in range(len(labels)):
        color = 'red'
        if labels[i] == '0':
            color = 'green'
        draw.polygon([(position[i][0]['x'], position[i][0]['y']),
                      (position[i][1]['x'], position[i][1]['y']),
                      (position[i][2]['x'], position[i][2]['y']),
                      (position[i][3]['x'], position[i][3]['y'])], outline = color)
    return img

def segment(img_path,xml_path):
    '''
    segment images from a big image base on the information in the xml file
    :param img_path: the path of the big image
    :param xml_path: the xml file of the big image
    :return:
        segments: 3-d numpy array, uint8, 28x28
        labels: 1-d numpy array, integer, only 0 and 1
    '''
    img = Image.open(img_path)
    # crop = img.crop((100,100,300,300))

    # get xml tree
    tree = ET.ElementTree(file=xml_path)

    # get center, size and label of the each image
    center = tuple({'x':int(elem.attrib['x']),'y':int(elem.attrib['y'])} for elem in tree.iter(tag = 'center'))
    size = tuple({'w':int(elem.attrib['w']),'h':int(elem.attrib['h'])} for elem in tree.iter(tag = 'size'))
    labels = tuple(int(elem.attrib['occupied']) for elem in tree.iter(tag='space'))

    segments  = []

    for i in range(len(labels)):
        area = (center[i]['x'] - size[i]['w'] / 2, center[i]['y'] - size[i]['h'] / 2,
                center[i]['x'] + size[i]['w'] / 2, center[i]['y'] + size[i]['h'] / 2)
        segs = img.crop(area)
        segs = segs.resize((28, 28))
        segs = segs.convert("L")
        segs = np.array(segs.getdata(), dtype='float32').reshape(segs.size[0], segs.size[1])
        segments.append(segs)

    segments_images = np.array(segments)
    segments_labels = np.array(labels)

    # convert numpy array to tensor
    segments_images = tf.convert_to_tensor(segments_images, tf.float32)
    segments_labels = tf.convert_to_tensor(segments_labels, tf.int32)

    # preprocess
    segments_images = tf.image.per_image_standardization(segments_images)

    return segments_images, segments_labels

def segment_nparray(img_path,xml_path):
    '''
    segment images from a big image base on the information in the xml file
    :param img_path: the path of the big image
    :param xml_path: the xml file of the big image
    :return:
        segments: 3-d numpy array, uint8, 28x28
        labels: 1-d numpy array, integer, only 0 and 1
    '''
    img = Image.open(img_path)
    # crop = img.crop((100,100,300,300))

    # get xml tree
    tree = ET.ElementTree(file=xml_path)

    # get center, size and label of the each image
    center = tuple({'x':int(elem.attrib['x']),'y':int(elem.attrib['y'])} for elem in tree.iter(tag = 'center'))
    size = tuple({'w':int(elem.attrib['w']),'h':int(elem.attrib['h'])} for elem in tree.iter(tag = 'size'))
    labels = tuple(int(elem.attrib['occupied']) for elem in tree.iter(tag='space'))

    segments  = []

    for i in range(len(labels)):
        area = (center[i]['x'] - size[i]['w'] / 2, center[i]['y'] - size[i]['h'] / 2,
                center[i]['x'] + size[i]['w'] / 2, center[i]['y'] + size[i]['h'] / 2)
        segs = img.crop(area)
        segs = segs.resize((28, 28))
        segs = segs.convert("L")
        segs = np.array(segs.getdata(), dtype='float32').reshape(segs.size[0], segs.size[1])
        segments.append(segs)

    segments_images = np.array(segments)
    segments_labels = np.array(labels)

    segments_images = per_image_standardization(segments_images)

    return segments_images, segments_labels

def per_image_standardization(images):
    images = images.reshape(-1,28*28)
    for i in range(0,len(images)):
        image = images[i]
        adjusted_stddev = max(np.std(image),1.0/np.sqrt(len(image)))
        image = (image - np.mean(image))/adjusted_stddev
        images[i] = image
    return images.reshape(-1,28,28)

if __name__ == '__main__':
    pass
    # img = fill_color(img_path,xml_path)
    # img,segments = segment(img_path,xml_path)

    # plt.imshow(img)
    # plt.show()
    img, segments = segment_nparray(img_path, xml_path)




