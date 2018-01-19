import os
import os.path

import numpy as np
import tensorflow as tf

import input_data
import VGG
import utils

# %%
IMG_W = 32
IMG_H = 32
N_CLASSES = 10
BATCH_SIZE = 32
learning_rate = 0.01
MAX_STEP = 15000  # it took me about one hour to complete the training.
IS_PRETRAIN = True


# %%   Training
def train():
    pre_trained_weights = './/vgg16_pretrain//vgg16.npy'
    data_dir = '/home/liming/Documents/datasets/cifar-10-batches-bin/'
    train_log_dir = './/logs//train//'
    val_log_dir = './/logs//val//'

    with tf.name_scope('input'):
        tra_image_batch, tra_label_batch = input_data.read_cifar10(data_dir=data_dir,
                                                                   is_train=True,
                                                                   batch_size=BATCH_SIZE,
                                                                   shuffle=True)
        val_image_batch, val_label_batch = input_data.read_cifar10(data_dir=data_dir,
                                                                   is_train=False,
                                                                   batch_size=BATCH_SIZE,
                                                                   shuffle=False)

    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])
    y_ = tf.placeholder(tf.int16, shape=[BATCH_SIZE, N_CLASSES])

    logits = VGG.VGG16N(x, N_CLASSES, IS_PRETRAIN)
    loss = utils.loss(logits, y_)
    accuracy = utils.accuracy(logits, y_)

    my_global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = utils.optimize(loss, learning_rate, my_global_step)

    saver = tf.train.Saver(tf.global_variables())
    tf.summary.scalar('loss',loss)
    tf.summary.scalar('accuracy',accuracy)
    summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # load the parameter file, assign the parameters, skip the specific layers
        utils.load_with_skip(pre_trained_weights, sess, ['fc6', 'fc7', 'fc8'])
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        tra_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
        val_summary_writer = tf.summary.FileWriter(val_log_dir, sess.graph)
        try:
            for step in np.arange(MAX_STEP):
                if coord.should_stop():
                    break

                tra_images, tra_labels = sess.run([tra_image_batch, tra_label_batch])
                summary, _, tra_loss, tra_acc = sess.run([summary_op, train_op, loss, accuracy],
                                                feed_dict={x: tra_images, y_: tra_labels})
                tra_summary_writer.add_summary(summary, step)
                if step % 50 == 0 or (step + 1) == MAX_STEP:
                    print('Step: %d, loss: %.4f, accuracy: %.4f%%' % (step, tra_loss, tra_acc))
                    # summary_str = sess.run(summary_op)
                    # tra_summary_writer.add_summary(summary_str, step)

                if step % 200 == 0 or (step + 1) == MAX_STEP:
                    val_images, val_labels = sess.run([val_image_batch, val_label_batch])
                    summary, val_loss, val_acc = sess.run([summary_op, loss, accuracy],
                                                 feed_dict={x: val_images, y_: val_labels})
                    print('**  Step %d, val loss = %.2f, val accuracy = %.2f%%  **' % (step, val_loss, val_acc))
                    val_summary_writer.add_summary(summary, step)

                if step % 2000 == 0 or (step + 1) == MAX_STEP:
                    checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()

        coord.join(threads)
        sess.close()


# %%   Test the accuracy on test dataset. got about 85.69% accuracy.
import math


def evaluate():
    with tf.Graph().as_default():

        #        log_dir = 'C://Users//kevin//Documents//tensorflow//VGG//logsvgg//train//'
        log_dir = 'C:/Users/kevin/Documents/tensorflow/VGG/logs/train/'
        test_dir = './/data//cifar-10-batches-bin//'
        n_test = 10000

        images, labels = input_data.read_cifar10(data_dir=test_dir,
                                                 is_train=False,
                                                 batch_size=BATCH_SIZE,
                                                 shuffle=False)

        logits = VGG.VGG16N(images, N_CLASSES, IS_PRETRAIN)
        correct = utils.num_correct_prediction(logits, labels)
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
                print('\nEvaluating......')
                num_step = int(math.floor(n_test / BATCH_SIZE))
                num_sample = num_step * BATCH_SIZE
                step = 0
                total_correct = 0
                while step < num_step and not coord.should_stop():
                    batch_correct = sess.run(correct)
                    total_correct += np.sum(batch_correct)
                    step += 1
                print('Total testing samples: %d' % num_sample)
                print('Total correct predictions: %d' % total_correct)
                print('Average accuracy: %.2f%%' % (100 * total_correct / num_sample))
            except Exception as e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
                coord.join(threads)

if __name__ == '__main__':
    train()