from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

NUM_DATA = 10000
NUM_TRAINING_DATA = 7000
NUM_VALIDATION_DATA = 2000
NUM_TEST_DATA = 1000
BATCH_SIZE = 50

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

    W1 = tf.Variable(tf.truncated_normal([2, 10], stddev=0.1))
    b1 = tf.Variable(tf.zeros([10]))

    W2 = tf.Variable(tf.truncated_normal([10, 20], stddev=0.1))
    b2 = tf.Variable(tf.zeros([20]))

    W3 = tf.Variable(tf.truncated_normal([20, 1], stddev=0.1))
    b3 = tf.Variable(tf.zeros([1]))

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    a1 = tf.nn.relu6(tf.matmul(x, W1) + b1)
    a2 = tf.nn.relu6(tf.matmul(a1,W2) + b2)
    y = tf.matmul(a2,W3) + b3

    coss = tf.reduce_mean((tf.nn.l2_loss(y-y_)))

    train_step = tf.train.GradientDescentOptimizer(0.005).minimize(coss)

    for epoch in range(100):
        train_index = np.arange(1, NUM_TRAINING_DATA)
        np.random.shuffle(train_index)
        run_training_set = training_set[train_index,:]
        run_train_ans = training_ans[train_index,:]
        for i in range(int(NUM_TRAINING_DATA/BATCH_SIZE)):
            train_step.run(feed_dict={x: run_training_set[i*BATCH_SIZE:(i+1)*BATCH_SIZE,:],
                                      y_: run_train_ans[i*BATCH_SIZE:(i+1)*BATCH_SIZE,:]})

    print('This is a toy example of floating adding using DNN')
    print('Average loss on test set: ', coss.eval(feed_dict={x: test_set, y_: test_ans}))
    test_by_hand = np.array([[0.2,0.6],[0.1,0.3]], dtype=np.float32)
    test_by_hand.shape=(2,2)
    test_by_hand_ans = np.array([0, 0], dtype=np.float32)
    test_by_hand_ans.shape = (2, 1)
    print('Manual test at: \n', test_by_hand, '\nResult: \n', sess.run(y, feed_dict={x:test_by_hand, y_: test_by_hand_ans}))