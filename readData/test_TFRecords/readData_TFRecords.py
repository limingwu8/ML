# use TFRecords to read data
import os
import tensorflow as tf
from PIL import Image
# cwd = os.getcwd()
# print("current working directory:", cwd)
#
# writer = tf.python_io.TFRecordWriter("train.tfrecords")
# classes = os.listdir(cwd)
#
# for index, name in enumerate(classes):
#     class_path = cwd + '\\' + name + "\\"
#     for img_name in os.listdir(class_path):
#         img_path = class_path + img_name
#         img = Image.open(img_path)
#         img = img.resize((224, 224))
#         img_raw = img.tobytes()              #将图片转化为原生bytes
#         example = tf.train.Example(features=tf.train.Features(feature={
#             "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
#             'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
#         }))
#         writer.write(example.SerializeToString())  #序列化为字符串
# writer.close()

# use queue to read data
def read_and_decode(filename):
    # create a queue base on filename
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [224, 224, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)

    return img,label

if __name__ == '__main__':
    img,label = read_and_decode("train.tfrecords")

    # img_batch, label_batch = tf.train.shuffle_batch([img,label],
    #                                                 batch_size=30,
    #                                                 capacity=2000,
    #                                                 min_after_dequeue=1000)
    init = tf.initialize_all_variables()
    with tf.Session() as s:
        s.run(init)
        # threads = tf.train.start_queue_runners(sess=s)
        for i in range(3):
            # val, l = s.run([img_batch,label_batch])
            val, l = s.run()
            print(val.shape,l)
