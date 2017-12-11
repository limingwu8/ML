#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 10:51:09 2017

@author: liming

This model imatates the google cifar10 codes
www.github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10.py
"""

#%%

import tensorflow as tf

#%%

def inference(images, batch_size, n_classes):
    ''' build the model
    Args:
        images: one batch of images, 4D tensor, tf.float32, 
                [batch_size, width, height, channels]
        batch_size: how many images in one batch
    Returns:
        output tensor with the computed logits, float, [batch_size, n_classes]
    '''

    # conv1
    # shape = [kernel size, kernel size, channels, kernel numbers]
    with tf.variable_scope('conv1') as scope:
        weights = tf.get_variable('weights',
                                  shape = [3,3,1,32],
                                  dtype = tf.float32,
                                  initializer = tf.truncated_normal_initializer(stddev = 0.1, dtype = tf.float32))
        biases = tf.get_variable('biases',
                                 shape = [32],
                                 dtype = tf.float32,
                                 initializer = tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(images, weights, strides = [1,1,1,1], padding = 'SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        
        conv1 = tf.nn.relu(pre_activation, name = scope.name)
        
    # pool1 and norm1    
    with tf.variable_scope('pooling_norm1') as scope:
        pool1 = tf.nn.max_pool(conv1, ksize = [1,2,2,1], strides = [1,2,2,1],
                               padding = 'SAME', name = 'pooling1')
        norm1 = tf.nn.lrn(pool1, depth_radius = 4, bias = 1.0, 
                          alpha = 0.001/9.0, beta = 0.75, name = 'norm1')
    
    # conv2
    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable('weights',
                                  shape = [3,3,32,32],
                                  dtype = tf.float32,
                                  initializer = tf.truncated_normal_initializer(stddev = 0.1, dtype = tf.float32))
        
        biases = tf.get_variable('biases',
                                 shape = [32],
                                 dtype = tf.float32,
                                 initializer = tf.truncated_normal_initializer(0.1))
        conv = tf.nn.conv2d(norm1, weights, strides = [1,1,1,1], padding = 'SAME')
        conv = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(conv, name = scope.name)
        
    # pool2 and norm2
    with tf.variable_scope('pooling_norm2') as scope:
        norm2 = tf.nn.lrn(conv2, depth_radius = 4, bias = 1.0, alpha = 0.001/9.0,
                          beta = 0.75, name = 'norm2')
        pool2 = tf.nn.max_pool(norm2, ksize = [1,2,2,1], strides = [1,1,1,1],
                               padding = 'SAME', name = 'pooling2')
    
    # local3
    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(pool2, shape = [batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable('weights',
                                  shape = [dim, 128],
                                  dtype = tf.float32,
                                  initializer = tf.truncated_normal_initializer(stddev = 0.005, dtype = tf.float32))
        biases = tf.get_variable('biases',
                                 shape = [128],
                                 dtype = tf.float32,
                                 initializer = tf.truncated_normal_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name = scope.name)
        
    # local4
    with tf.variable_scope('local4') as scope:
        weights = tf.get_variable('weights',
                                  shape = [128, 128],
                                  dtype = tf.float32,
                                  initializer = tf.truncated_normal_initializer(stddev = 0.005, dtype = tf.float32))
        biases = tf.get_variable('biases',
                                 shape = [128],
                                 dtype = tf.float32,
                                 initializer = tf.truncated_normal_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name = scope.name)
        
    # softmax
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('weights',
                                  shape = [128, n_classes],
                                  dtype = tf.float32,
                                  initializer = tf.truncated_normal_initializer(stddev = 0.005, dtype = tf.float32))
        biases = tf.get_variable('biases',
                                 shape = [n_classes],
                                 dtype = tf.float32,
                                 initializer = tf.truncated_normal_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(local4,weights), biases, name = 'softmax_linear')
        
    
    return softmax_linear

def losses(logits, labels):
    ''' Compute loss from logits and labels
    Args:
        logits: logits tensor, float, [batch_size, n_classes]
        labels: label tensor, tf.int32, [batch_size]
    Returns:
        loss tensor of float type
    '''
    
    with tf.variable_scope('loss') as scope:
        # do not need one hot encoding if use the sparse softmax
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits,
                                                                       labels = labels,
                                                                       name = 'xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name = 'loss')
        # show in tensor board
        tf.summary.scalar(scope.name + '/loss', loss)
        
        return loss

def training(loss, learning_rate):
    '''
    Args:
        loss: loss tensor, from losses()
        
    Returns:
        train_op: the op for training
    '''
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
        global_step = tf.Variable(0, name = 'global_step', trainable = False)
        train_op = optimizer.minimize(loss, global_step = global_step)
    
    return train_op
    
    
def evaluation(logits, labels):
    '''
    Args:
        logits: Logits tensor, float [batch_size, n_classes]
        labels: Labels tensor, int32 [batch_size], with value in the range [0, n_classes]
    Returns:
        A scalar int32 tensor with the number of example (out of batch_size) that
        were predicted correctly
    '''
    with tf.variable_scope('accuracy') as scope:
        # 1 means use the largest number of prediction
        # e.g. the probability of 1 is 0.7, the probability of 0 is 0.3
        # choose 1, compare with label
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name + '/accuracy', accuracy)
        
    return accuracy
