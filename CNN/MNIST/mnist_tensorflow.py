import os
# os.environ['LD_LIBRARY_PATH'] = "/usr/local/cuda-9.0/lib64"
# os.environ['PATH'] = '/home/liming/anaconda3/envs/tf/bin:/usr/local/cuda-9.0/bin:/home/liming/anaconda3/bin:/home/liming/bin:/home/liming/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin'
# os.environ.update()
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32,[None,784])
y_ = tf.placeholder(tf.float32,[None,10])

# Weight and bias initialization
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

# Convolution and Pooling
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

# First Convolution and Max Pooling
W_conv1 = weight_variable([5,5,1,32])   # output 32 feature maps
b_conv1 = bias_variable([32])
x_image = tf.reshape(x,[-1,28,28,1])   # input size 28x28x1
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)   # output size 28x28x32
h_pool1 = max_pool_2x2(h_conv1)     # output size 14x14x32

# Second Convolution and Max Pooling
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)   # output size 14x14x64
h_pool2 = max_pool_2x2(h_conv2)     # output size 7x7x64

# Densely Connected Layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout, reduce over fitting
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout Layer, just like the softmax layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# Train and evaluate model
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

# add summary to tensorboard
tf.summary.scalar('accuracy', accuracy)
tf.summary.scalar('cross_entropy',cross_entropy)
merged = tf.summary.merge_all()

with tf.Session() as sess:

    train_writer = tf.summary.FileWriter('./logs/train/', sess.graph)
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            # train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            summary, train_accuracy = sess.run([merged,accuracy], feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            train_writer.add_summary(summary, i)
            print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))