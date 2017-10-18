import tensorflow as tf

image_raw_data = tf.gfile.FastGFile('C:\\Users\\Administrator\\Desktop\\aaa.jpg').read()

with tf.Session() as sess:
    img_data = tf.image.decode_png(image_raw_data)
    print(img_data.eval())
