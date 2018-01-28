from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

NUM_DATA = 10000
NUM_TRAINING_DATA = 8000
NUM_VALIDATION_DATA = 1000
NUM_TEST_DATA = 1000
BATCH_SIZE = 50


def dnn(layers, x, y_, act_func = tf.nn.relu6):
    if layers == None:
        print('Error layer configuration. Exit...')
        exit(-1)
    layer_output = x
    for i in range(len(layers)-1):
        this_units_num = layers[i]
        next_units_num = layers[i+1]
        w = tf.get_variable("Layer_"+str(i)+"_weight", dtype=tf.float32,
                            initializer=tf.truncated_normal(shape=(this_units_num, next_units_num), stddev=0.1))
        b = tf.get_variable("Layer_"+str(i)+"_bias", dtype=tf.float32,
                            initializer=tf.zeros(shape=(next_units_num)))
        if i != len(layers)-2:
            layer_output = act_func(tf.matmul(layer_output, w) + b)
        else:
            layer_output = tf.matmul(layer_output, w) + b

    coss = tf.nn.l2_loss(y_ - layer_output)
    return layer_output, coss


def main():
    np.random.seed(1)
    dataset = np.random.rand(NUM_DATA,2)
    dataindex = np.arange(NUM_DATA)
    np.random.shuffle(dataindex)

    training_set = dataset[dataindex[0:NUM_TRAINING_DATA], :]
    validation_set = dataset[dataindex[NUM_TRAINING_DATA:NUM_TRAINING_DATA+NUM_VALIDATION_DATA], :]
    test_set = dataset[dataindex[NUM_TRAINING_DATA+NUM_VALIDATION_DATA:NUM_DATA], :]

    training_ans = np.fromiter(map(lambda x: x[0] * x[1], training_set),dtype=np.float32)
    validation_ans = np.fromiter(map(lambda x: x[0] * x[1], validation_set),dtype=np.float32)
    test_ans = np.fromiter(map(lambda x: x[0] * x[1], test_set),dtype=np.float32)
    training_ans.shape = (NUM_TRAINING_DATA,1)
    validation_ans.shape = (NUM_VALIDATION_DATA,1)
    test_ans.shape = (NUM_TEST_DATA,1)

    layers = [2, 4, 8, 16, 32, 1]
    x = tf.placeholder(tf.float32, shape=[None, layers[0]])
    y_ = tf.placeholder(tf.float32, shape=[None, layers[-1]])
    layer_output, coss = dnn(layers, x, y_, tf.nn.relu)
    train_step = tf.train.AdamOptimizer(2e-4).minimize(coss)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(400):
            train_index = np.arange(1, NUM_TRAINING_DATA)
            np.random.shuffle(train_index)
            run_training_set = training_set[train_index,:]
            run_train_ans = training_ans[train_index,:]
            for i in range(int(NUM_TRAINING_DATA/BATCH_SIZE)):
                sess.run(train_step, feed_dict={x: run_training_set[i*BATCH_SIZE:(i+1)*BATCH_SIZE,:],
                                                y_: run_train_ans[i*BATCH_SIZE:(i+1)*BATCH_SIZE,:]})
            print('Average loss on test set: ', sess.run(coss, feed_dict={x: test_set, y_: test_ans}))

        print('This is a toy example of floating adding using DNN')
        print('Average loss on test set: ', sess.run(coss, feed_dict={x: test_set, y_: test_ans}))
        test_by_hand = np.array([[0.2,0.6],[0.1,0.3]], dtype=np.float32)
        test_by_hand.shape=(2,2)
        test_by_hand_ans = np.array([0.12, 0.03], dtype=np.float32)
        test_by_hand_ans.shape = (2, 1)
        print('Manual test at: \n', test_by_hand, '\nResult: \n', test_by_hand_ans)
        print(sess.run(layer_output, feed_dict={x: test_by_hand, y_: test_by_hand_ans}))


if __name__=='__main__':
    main()

