import os.path
import math

import numpy as np
import tensorflow as tf

from CNN.CIFAR10.cifar10_input import *
from CNN.CIFAR10.cifar10_model import *

BATCH_SIZE = 128
learning_rate = 0.05
MAX_STEP = 100  # with this setting, it took less than 30 mins on my laptop to train.

# %% Train the model on the training data
def train():
    my_global_step = tf.Variable(0, name='global_step', trainable=False)

    data_dir = '/home/liming/Documents/datasets/cifar-10-batches-bin/'
    log_dir = '/home/liming/PycharmProjects/ML/CNN/CIFAR10/logs/'

    images, labels = read_cifar10(data_dir=data_dir,
                                                is_train=True,
                                                batch_size=BATCH_SIZE,
                                                shuffle=True)
    logits = inference(images, BATCH_SIZE)

    loss = losses(logits, labels)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, global_step=my_global_step)

    saver = tf.train.Saver(tf.global_variables())
    summary_op = tf.summary.merge_all()

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            _, loss_value = sess.run([train_op, loss])

            if step % 50 == 0:
                print('Step: %d, loss: %.4f' % (step, loss_value))

            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()


# %% To test the model on the test data
def evaluate():
    with tf.Graph().as_default():

        test_dir = '/home/liming/Documents/datasets/cifar-10-batches-bin/'
        log_dir = '/home/liming/PycharmProjects/ML/CNN/CIFAR10/logs/'
        n_test = 10000

        # reading test data
        images, labels = read_cifar10(data_dir=test_dir,
                                        is_train=False,
                                        batch_size=BATCH_SIZE,
                                        shuffle=False)

        logits = inference(images, BATCH_SIZE)
        top_k_op = tf.nn.in_top_k(logits, labels, 1)
        saver = tf.train.Saver(tf.global_variables())

        with tf.Session() as sess:

            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
                return

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                num_iter = int(math.ceil(n_test / BATCH_SIZE))
                true_count = 0
                total_sample_count = num_iter * BATCH_SIZE
                step = 0

                while step < num_iter and not coord.should_stop():
                    predictions = sess.run([top_k_op])
                    true_count += np.sum(predictions)
                    step += 1
                    precision = true_count / total_sample_count
                print('precision = %.3f' % precision)
            except Exception as e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
                coord.join(threads)

if __name__=='__main__':
    # train()
    evaluate()