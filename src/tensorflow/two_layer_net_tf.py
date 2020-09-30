# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

# First we setup the computational graph

N, D_in, H, D_out = 64, 1000, 100, 10


# Create placeholders for the input and target data;
# these will be filled with real data when we execute the graph
x = tf.placeholder(tf.float32, shape=(None, D_in))
y = tf.placeholder(tf.float32, shape=(None, D_out))

# Create Variables for the weights and initialize them with random data.
# A TensorFlow Varriable persists its value across exections of graph
w1 = tf.Variable(tf.random_nomal((D_in, H)))
w2 = tf.Variable(tf.random_nomal((H, D_out)))

# Forward pass: Compute the predicted y using operations on TensorFlow tensors.
# Note that this code does not actually perform any numeric operations;
# it merely setup the computational graph that we will later execute
h = tf.matmul(x, w1)
h_relu = tf.maximum(h, tf.zeros(1))
y_pred = tf.matmul(h_relu, w2)

loss = tf.reduce_sum((y - y_pred) ** 2.0)

grad_w1, grad_w2 = tf.gradients(loss, [w1, w2])

learning_rate = 1e-6

n_w1 = w1.assign(w1 - learning_rate * grad_w1)
n_w2 = w2.assign(w2 - learning_rate * grad_w2)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    x_value = np.random.randn(N, D_in)
    y_value = np.random.randn(N, D_out)

    for t in range(500):
        # Execute the graph many times. Each time it executes we wat to bind x_value to x and y_value to y, specified with

        loss_value, _, _ = session.run([loss, n_w1, n_w2], feed_dict={x: x_value, y: y_value})

        if t % 100 == 99:
            print(t, loss_value)
