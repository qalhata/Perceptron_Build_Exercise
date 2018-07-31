# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 17:49:15 2018

@author: Shabaka
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data directory", one_hot=True)

# BUild the Neural Network # Tensorflow already uses numpy so we're good
# Tensorflow carries out computations for us - using the same
# computation graph. Automatic differentiation
# Build the computational graph = Set input data and ground truth
# as placeholders

with tf.variable_scope('input', reuse=True):
    X = tf.placeholder(tf.float32, shape=[None, 784], name='X')
    y_ = tf.placeholder(tf.float32, shape=[None, 10], name='y_')
    # 10 is one-hot

with tf.variable_scope('layer', reuse=True):
    W = tf.get_variable('weights', shape=[784, 10],
                        initializer=tf.truncated_normal_initializer())
    b = tf.get_variable('biases', shape=[10],
                        initializer=tf.zeros_initializer())
    y = tf.matmul(X, W) + b

with tf.variable_scope('output', reuse=True):
    with tf.variable_scope('loss', reuse=True):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
        tf.summary.scalar('loss', cross_entropy)

    with tf.variable_scope('accuracy', reuse=True):
        corr_prediction = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_, axis=1))
        accuracy = tf.reduce_mean(tf.cast(corr_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)


# Tensorflow is a ststic computational graph module
# We feed in data in batchs which we have given placeholders

# Next we take the output of the network and compare
# with grnd truth layer using softmax layer and
# cross entropy loss function

# then we run gradient descent update - we use a tensorflow

with tf.variable_scope('train', reuse=True):
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# this tells tf what we want to minimise (cross ent and)

sess = tf.InteractiveSession()
# initialise all variables
sess.run(tf.global_variables_initializer())

merged = tf.summary.merge_all()    # plot multiple values
train_writer = tf.summary.FileWriter('logs', graph=sess.graph)

# Training a thousand epochs for example

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(256)
    summary, loss, _ = sess.run([merged, cross_entropy, train_step],
                                feed_dict={X: batch_xs, y: batch_ys})
    if (i+1) % 100 == 0:
        print('Iteration {0}: {1:.4f}'.format(i+1, loss))
    train_writer.add_summary(summary, i)
    train_writer.flush()

# check accuracy of test data

print(sess.run(accuracy,
               feed_dict={X: mnist.test.images, y_: mnist.test.labels}))


train_writer.close()
