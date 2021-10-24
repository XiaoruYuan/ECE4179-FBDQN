# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import random
import cv2
import sys
sys.path.append('game/')
import wrapped_flappy_bird as fb
from collections import deque

# Hyper parameters
ACTIONS = 2
GAMMA = 0.99
OBSERVE = 10000
EXPLORE = 3000000   # was 2000000
TRAINING = 4030000  # added training time to stop
INITIAL_EPSILON = 0.1
FINAL_EPSILON = 0.0001
REPLAY_MEMORY = 50000
BATCH = 32
IMAGE_SIZE = 80

S = tf.placeholder(dtype=tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 4], name='S')
A = tf.placeholder(dtype=tf.float32, shape=[None, ACTIONS], name='A')
Y = tf.placeholder(dtype=tf.float32, shape=[None], name='Y')
k_initializer = tf.truncated_normal_initializer(0, 0.01)
b_initializer = tf.constant_initializer(0.01)


def conv2d(inputs, kernel_size, filters, strides):
    return tf.layers.conv2d(inputs, kernel_size=kernel_size, filters=filters, strides=strides, padding='same', kernel_initializer=k_initializer, bias_initializer=b_initializer)


def max_pool(inputs):
    return tf.layers.max_pooling2d(inputs, pool_size=2, strides=2, padding='same')


def relu(inputs):
    return tf.nn.relu(inputs)


# Define network
h0 = max_pool(relu(conv2d(S, 8, 32, 4)))
h0 = relu(conv2d(h0, 4, 64, 2))
h0 = relu(conv2d(h0, 3, 64, 1))
h0 = tf.contrib.layers.flatten(h0)
h0 = tf.layers.dense(h0, units=512, activation=tf.nn.relu, bias_initializer=b_initializer)

# Define cost function
Q = tf.layers.dense(h0, units=ACTIONS, bias_initializer=b_initializer, name='Q')
Q_ = tf.reduce_sum(tf.multiply(Q, A), axis=1)
loss = tf.losses.mean_squared_error(Y, Q_)
optimizer = tf.train.AdamOptimizer(1e-6).minimize(loss)

# Start a new game
game_state = fb.GameState()
# Store previous observation
D = deque()

# Get the first state
do_nothing = np.zeros(ACTIONS)
do_nothing[0] = 1
img, reward, terminal = game_state.frame_step(do_nothing)
img = cv2.cvtColor(cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE)), cv2.COLOR_BGR2GRAY)
_, img = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
S0 = np.stack((img, img, img, img), axis=2)

# Save and load network
sess = tf.Session()
sess.run(tf.global_variables_initializer())

t = 0
success = 0
saver = tf.train.Saver()
epsilon = INITIAL_EPSILON
highest_score = 0
prev_state = False
curr_score = 0

# Start training
while t <= TRAINING:
    # scale down epsilon
    if epsilon > FINAL_EPSILON and t > OBSERVE:
        epsilon = INITIAL_EPSILON - (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE * (t - OBSERVE)

    Qv = sess.run(Q, feed_dict={S: [S0]})[0]
    Av = np.zeros(ACTIONS)
    if np.random.random() <= epsilon:
        action_index = np.random.randint(ACTIONS)
    else:
        action_index = np.argmax(Qv) 
    Av[action_index] = 1

    img, reward, terminal = game_state.frame_step(Av)
    if reward == 1:
        success += 1
        if prev_state:
            curr_score += 1
        else:
            curr_score = 1
    elif reward == -1:
        if highest_score < curr_score:
            highest_score = curr_score
        curr_score = 0

    img = cv2.cvtColor(cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE)), cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
    img = np.reshape(img, (IMAGE_SIZE, IMAGE_SIZE, 1))
    S1 = np.append(S0[:, :, 1:], img, axis=2)

    D.append((S0, Av, reward, S1, terminal))
    if len(D) > REPLAY_MEMORY:
        D.popleft()

    # only train if done observing
    if t > OBSERVE:
        # sample a minimum batch to train on
        minibatch = random.sample(D, BATCH)

        # get the batch variables
        S_batch = [d[0] for d in minibatch]
        A_batch = [d[1] for d in minibatch]
        R_batch = [d[2] for d in minibatch]
        S_batch_next = [d[3] for d in minibatch]
        T_batch = [d[4] for d in minibatch]

        Y_batch = []
        Q_batch_next = sess.run(Q, feed_dict={S: S_batch_next})
        for i in range(BATCH):
            # if terminal, only equals reward
            if T_batch[i]:
                Y_batch.append(R_batch[i])
            else:
                Y_batch.append(R_batch[i] + GAMMA * np.max(Q_batch_next[i]))

        # perform gradient step
        sess.run(optimizer, feed_dict={S: S_batch, A: A_batch, Y: Y_batch})

    # update the old values
    S0 = S1
    t += 1

    # save progress every 10000 iterations
    if t > OBSERVE and t % 10000 == 0:
        saver.save(sess, './flappy_bird_dqn', global_step=t)

    # print info
    if t <= OBSERVE:
        state = 'observe'
    elif t <= OBSERVE + EXPLORE:
        state = 'explore'
    else:
        state = 'train'
    print('Current Step %d Success %d State %s Epsilon %.6f Action %d Reward %f Q_MAX %f' % (t, success, state, epsilon, action_index, reward, np.max(Qv)))
    print('Highest score so far', highest_score)



