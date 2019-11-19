from math import pi
import tensorflow as tf

import utils


def kl_penalty(pz_mean, pz_sigma, encoded_mean, encoded_sigma):
    """
    Compute KL divergence between prior and variational distribution
    """
    kl = encoded_sigma / pz_sigma \
        + tf.square(pz_mean - encoded_mean) / pz_sigma - 1. \
        + tf.log(pz_sigma) - tf.log(encoded_sigma)
    kl = 0.5 * tf.reduce_sum(kl,axis=-1)
    return tf.reduce_mean(kl)


def mc_kl_penalty(samples, q_mean, q_Sigma, p_mean, p_Sigma):
    """
    Compute MC log density ratio
    """
    kl = tf.log(q_Sigma) - tf.log(p_Sigma) \
        + tf.square(samples - q_mean) / q_Sigma \
        - tf.square(samples - p_mean) / p_Sigma
    kl = -0.5 * tf.reduce_sum(kl,axis=-1)
    return tf.reduce_mean(kl)


def Xentropy_penalty(samples, mean, sigma):
    """
    Compute Xentropy for gaussian using MC
    """
    loglikelihood = tf.log(2*pi) + tf.log(sigma) + tf.square(samples-mean) / sigma
    loglikelihood = -0.5 * tf.reduce_sum(loglikelihood,axis=-1)
    return tf.reduce_mean(loglikelihood)


def entropy_penalty(samples, mean, sigma):
    """
    Compute entropy for gaussian
    """
    entropy = tf.log(sigma) + 1. + tf.log(2*pi)
    entropy = 0.5 * tf.reduce_sum(entropy,axis=-1)
    return tf.reduce_mean(entropy)


def matching_penalty(opts, samples_pz, samples_qz):
    """
    Compute the WAE's matching penalty
    (add here other penalty if any)
    """
    macth_penalty = mmd_penalty(opts, samples_pz, samples_qz)
    return macth_penalty


def mmd_penalty(opts, sample_qz, sample_pz):
    sigma2_p = opts['pz_scale'] ** 2
    kernel = opts['mmd_kernel']
    n = utils.get_batch_size(sample_qz)
    n = tf.cast(n, tf.int32)
    nf = tf.cast(n, tf.float32)
    half_size = (n * n - n) / 2

    distances_pz = square_dist(opts, sample_pz, sample_pz)
    distances_qz = square_dist(opts, sample_qz, sample_qz)
    distances = square_dist(opts, sample_qz, sample_pz)

    if opts['mmd_kernel'] == 'RBF':
        # Median heuristic for the sigma^2 of Gaussian kernel
        sigma2_k = tf.nn.top_k(
            tf.reshape(distances, [-1]), half_size).values[half_size - 1]
        sigma2_k += tf.nn.top_k(
            tf.reshape(distances_qz, [-1]), half_size).values[half_size - 1]
        # Maximal heuristic for the sigma^2 of Gaussian kernel
        # sigma2_k = tf.nn.top_k(tf.reshape(distances_qz, [-1]), 1).values[0]
        # sigma2_k += tf.nn.top_k(tf.reshape(distances, [-1]), 1).values[0]
        # sigma2_k = opts['latent_space_dim'] * sigma2_p
        if opts['verbose']:
            sigma2_k = tf.Print(sigma2_k, [sigma2_k], 'Kernel width:')
        res1 = tf.exp( - distances_qz / 2. / sigma2_k)
        res1 += tf.exp( - distances_pz / 2. / sigma2_k)
        res1 = tf.multiply(res1, 1. - tf.eye(n))
        res1 = tf.reduce_sum(res1) / (nf * nf - nf)
        res2 = tf.exp( - distances / 2. / sigma2_k)
        res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
        stat = res1 - res2
    elif opts['mmd_kernel'] == 'IMQ':
        Cbase = 2 * opts['zdim'] * sigma2_p
        stat = 0.
        for scale in [.1, .2, .5, 1., 2., 5., 10.]:
            C = Cbase * scale
            res1 = C / (C + distances_qz)
            res1 += C / (C + distances_pz)
            res1 = tf.multiply(res1, 1. - tf.eye(n))
            res1 = tf.reduce_sum(res1) / (nf * nf - nf)
            res2 = C / (C + distances)
            res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
            stat += res1 - res2
    elif opts['mmd_kernel'] == 'RQ':
        stat = 0.
        for scale in [.1, .2, .5, 1., 2., 5., 10.]:
            res1 = (1. + distances_qz / scale / 2.) ** (-scale)
            res1 += (1. + distances_pz / scale / 2.) ** (-scale)
            res1 = tf.multiply(res1, 1. - tf.eye(n))
            res1 = tf.reduce_sum(res1) / (nf * nf - nf)
            res2 = (1. + distances / scale / 2.) ** (-scale)
            res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
            stat += res1 - res2

    return stat


def square_dist(opts, sample_x, sample_y):
    """
    Wrapper to compute square distance
    """
    norms_x = tf.reduce_sum(tf.square(sample_x), axis=-1, keepdims=True)
    norms_y = tf.reduce_sum(tf.square(sample_y), axis=-1, keepdims=True)

    squared_dist = norms_x + tf.transpose(norms_y) \
                    - 2. * tf.matmul(sample_x,sample_y,transpose_b=True)
    return tf.nn.relu(squared_dist)


def square_dist_v2(opts, sample_x, sample_y):
    """
    Wrapper to compute square distance
    """
    x = tf.expand_dims(sample_x,axis=1)
    y = tf.expand_dims(sample_y,axis=0)
    squared_dist = tf.reduce_sum(tf.square(x - y),axis=-1)
    return squared_dist


def reconstruction_loss(opts, x1, x2):
    """
    Compute the WAE's reconstruction losses for the top layer
    x1: image data             [batch,im_dim]
    x2: image reconstruction   [batch,im_dim]
    """
    # Flatten last dim input
    x1 = tf.layers.flatten(x1)
    x2 = tf.layers.flatten(x2)
    # Compute chosen cost
    if opts['cost'] == 'l2':
        cost = l2_cost(x1, x2)
    elif opts['cost'] == 'l2sq':
        cost = l2sq_cost(x1, x2)
    elif opts['cost'] == 'l2sq_norm':
        cost = l2sq_norm_cost(x1, x2)
    elif opts['cost'] == 'l1':
        cost = l1_cost(x1, x2)
    else:
        assert False, 'Unknown cost function %s' % opts['obs_cost']
    # Compute loss
    loss = tf.reduce_mean(cost)
    return loss


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
    eps = 1e-8
    labels = tf.layers.flatten(labels)
    # cross_entropy = - (labels * tf.log(preds+eps) + (1. - labels) * tf.log(1 - (preds+eps)))
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    return tf.reduce_mean(tf.reduce_sum(cross_entropy,axis=-1))
