from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

NUM_DATA = 10000
NUM_TRAINING_DATA = 7000
NUM_VALIDATION_DATA = 2000
NUM_TEST_DATA = 1000

if __name__=='__main__':

    np.random.seed(1)
    dataset = np.random.rand(NUM_DATA,2)
    dataindex = np.arange(1,NUM_DATA)
    np.random.shuffle(dataindex)

    training_set = dataset[dataindex[0:NUM_TRAINING_DATA], :]
    validation_set = dataset[dataindex[NUM_TRAINING_DATA:NUM_TRAINING_DATA+NUM_VALIDATION_DATA], :]
    test_set = dataset[dataindex[NUM_TRAINING_DATA+NUM_VALIDATION_DATA:NUM_DATA], :]

    training_ans = np.dot(training_set, np.array([[1],[1]],dtype=np.float32))
    validation_ans = np.dot(validation_set, np.array([[1],[1]],dtype=np.float32))
    test_ans = np.dot(test_set, np.array([[1],[1]],dtype=np.float32))

    x = tf.placeholder(tf.float32, shape=[None, 2])
    y_ = tf.placeholder(tf.float32, shape=[None, 1])

    W1 = tf.Variable(tf.zeros([2, 10]))
    b1 = tf.Variable(tf.zeros([10]))

    W2 = tf.Variable(tf.zeros([10, 20]))
    b2 = tf.Variable(tf.zeros([20]))

    W3 = tf.Variable(tf.zeros([20, 1]))
    b3 = tf.Variable(tf.zeros([1]))

    a1 = tf.nn.sigmoid(tf.matmul(W1,x) + b1)
    a2 = tf.nn.sigmoid(tf.matmul(W2,a1) + b2)
    y = tf.nn.sigmoid(tf.matmul(W3,a2) + b3)

    coss = tf.reduce_mean((y - y_) * (y - y_))

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(coss)

    for i in range(1000):
        batch = training_set[i*10:(i+1)*10,:]
        train_step.run(feed_dict={x: batch[0], y_: training_ans[i*10:(i+1)*10]})

    print(coss.eval(feed_dict={x: test_set, y_: test_ans}))