#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 08:50:40 2017

@author: liming
"""

#%%

import os
import numpy as np
import tensorflow as tf
from temp2 import input_data
from temp2 import model

#%%
N_CLASSES = 2
IMG_W = 208
IMG_H = 208
BATCH_SIZE = 16
CAPACITY = 2000
MAX_STEP = 6000
learning_rate = 0.0001
ratio = 0.2

#%%

def run_training():
    
    train_dir = '/home/liming/Documents/cats_vs_dogs/data/train/'
    logs_train_dir = '/home/liming/PycharmProjects/ML/dogs_vs_cats/logs/train/'
    logs_val_dir = '/home/liming/PycharmProjects/ML/dogs_vs_cats/logs/val/'

    train, train_label, val, val_label = input_data.get_files(train_dir, ratio)
    train_batch, train_label_batch = input_data.get_batch(train,
                                                         train_label,
                                                         IMG_W,
                                                         IMG_H,
                                                         BATCH_SIZE,
                                                         CAPACITY)
    val_batch, val_label_batch = input_data.get_batch(val,
                                                     val_label,
                                                     IMG_W,
                                                     IMG_H,
                                                     BATCH_SIZE,
                                                     CAPACITY)

    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3],name='placeholder_X')
    y_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE],name='placeholder_Y')

    logits = model.inference(x, BATCH_SIZE, N_CLASSES)
    loss = model.losses(logits, y_)
    train_op = model.trainning(loss, learning_rate)
    acc = model.evaluation(logits, y_)


    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
        val_writer = tf.summary.FileWriter(logs_val_dir, sess.graph)

        try:
            for step in np.arange(MAX_STEP):
                if coord.should_stop():
                    break

                tra_images, tra_labels = sess.run([train_batch, train_label_batch])
                _, tra_loss, tra_acc = sess.run([train_op, loss, acc], feed_dict={x: tra_images, y_: tra_labels})
                if step % 50 == 0:
                    print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (step, tra_loss, tra_acc * 100.0))
                    # summary_str = sess.run(summary_op)
                    # train_writer.add_summary(summary_str, step)

                if step % 200 == 0 or (step + 1) == MAX_STEP:
                    val_images, val_labels = sess.run([val_batch, val_label_batch])
                    val_loss, val_acc = sess.run([loss, acc],
                                                 feed_dict={x: val_images, y_: val_labels})
                    print('**  Step %d, val loss = %.2f, val accuracy = %.2f%%  **' % (step, val_loss, val_acc * 100.0))
                    # summary_str = sess.run(summary_op)
                    # val_writer.add_summary(summary_str, step)

                if step % 2000 == 0 or (step + 1) == MAX_STEP:
                    checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()
        coord.join(threads)
    
if __name__ == '__main__':
    run_training()













