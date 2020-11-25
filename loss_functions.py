from math import pi
import tensorflow as tf

import pdb

import utils


def square_dist(opts, sample_x, sample_y):
    """
    Wrapper to compute square distance
    """
    norms_x = tf.reduce_sum(tf.square(sample_x), axis=-1, keepdims=True)
    norms_y = tf.reduce_sum(tf.square(sample_y), axis=-1, keepdims=True)

    squared_dist = norms_x + tf.transpose(norms_y) \
                    - 2. * tf.matmul(sample_x,sample_y,transpose_b=True)
    return tf.nn.relu(squared_dist)


def square_dist_broadcast(opts, sample_x, sample_y):
    """
    Wrapper to compute square distance
    """
    x = tf.expand_dims(sample_x,axis=1)
    y = tf.expand_dims(sample_y,axis=0)
    squared_dist = tf.reduce_sum(tf.square(x - y),axis=-1)
    return squared_dist


def l2_cost(x1, x2):
    # c(x,y) = ||x - y||_2
    cost = tf.reduce_sum(tf.square(x1 - x2), axis=-1)
    cost = tf.sqrt(1e-10 + cost)
    return cost


def l2sq_cost(x1,x2):
    # c(x,y) = sum_i(||x - y||_2^2[:,i])
    cost = tf.reduce_sum(tf.square(x1 - x2), axis=-1)
    return cost


def l2sq_norm_cost(x1, x2):
    # c(x,y) = mean_i(||x - y||_2^2[:,i])
    cost = tf.reduce_mean(tf.square(x1 - x2), axis=-1)
    return cost


def l1_cost(x1, x2):
    # c(x,y) = ||x - y||_1
    cost = tf.reduce_sum(tf.abs(x1 - x2), axis=-1)
    return cost

def xentropy_cost(labels, logits):
    # c(z,x) = z * -log(x) + (1 - z) * -log(1 - x)
    # where x = logits, z = labels
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    return tf.reduce_sum(cross_entropy,axis=-1)
