import numpy as np
import tensorflow as tf
from math import ceil, sqrt

import ops.linear
import ops.conv2d
import ops.deconv2d
import ops.batchnorm
import ops.layernorm
import ops._ops
import ops.resnet
from datahandler import datashapes
from sampling_functions import sample_gaussian

import logging
import pdb

def encoder(opts, input, output_dim, scope=None, reuse=False,
                                            is_training=False,
                                            dropout_rate=1.):
    with tf.variable_scope(scope, reuse=reuse):
        if opts['network']['e_arch'] == 'mlp':
            # Encoder uses only fully connected layers with ReLus
            outputs = mlp_encoder(opts, input, output_dim,
                                            reuse,
                                            is_training,
                                            dropout_rate)
        elif opts['network']['e_arch'] == 'dcgan':
            # Fully convolutional architecture similar to DCGAN
            outputs = dcgan_encoder(opts, input, output_dim,
                                            reuse,
                                            is_training,
                                            dropout_rate)
        elif opts['network']['e_arch'] == 'dcgan_v2':
            # Fully convolutional architecture similar to Wasserstein GAN
            outputs = dcgan_v2_encoder(opts, input, output_dim,
                                            reuse,
                                            is_training,
                                            dropout_rate)
        elif opts['network']['e_arch'] == 'conv_locatello':
            # Fully convolutional architecture similar to Locatello & al.
            outputs = locatello_encoder(opts, input, output_dim,
                                            reuse,
                                            is_training,
                                            dropout_rate)
        elif opts['network']['e_arch'] == 'resnet':
            assert False, 'To Do'
            # Resnet archi similar to Improved training of WAGAN
            outputs = resnet_encoder(opts, input, output_dim,
                                            reuse,
                                            is_training,
                                            dropout_rate)
        elif opts['network']['e_arch'] == 'resnet_v2':
            assert False, 'To Do'
            # Full conv Resnet archi similar to Improved training of WAGAN
            outputs = resnet_v2_encoder(opts, input, output_dim,
                                            reuse,
                                            is_training,
                                            dropout_rate)
        else:
            raise ValueError('%s : Unknown encoder architecture' % opts['network']['e_arch'])

    mean, logSigma = tf.split(outputs,2,axis=-1)
    logSigma = tf.clip_by_value(logSigma, -20, 500)
    Sigma = tf.nn.softplus(logSigma)
    mean = tf.layers.flatten(mean)
    Sigma = tf.layers.flatten(Sigma)

    if opts['encoder'] == 'det':
        z = mean
    elif opts['encoder'] == 'gauss':
        qz_params = tf.concat((mean, Sigma), axis=-1)
        z = sample_gaussian(qz_params, 'tensorflow')
    else:
        assert False, 'Unknown encoder %s' % opts['encoder']

    return z, mean, Sigma


def decoder(opts, input, output_dim, scope=None, reuse=False,
                                            is_training=False,
                                            dropout_rate=1.):
    with tf.variable_scope(scope, reuse=reuse):
        if opts['network']['d_arch'] == 'mlp':
            # Encoder uses only fully connected layers with ReLus
            outputs = mlp_decoder(opts, input, output_dim,
                                            reuse,
                                            is_training,
                                            dropout_rate)
        elif opts['network']['d_arch'] == 'dcgan':
            # Fully convolutional architecture similar to DCGAN
            outputs = dcgan_decoder(opts, input, output_dim,
                                            reuse,
                                            is_training,
                                            dropout_rate)
        elif opts['network']['d_arch'] == 'dcgan_v2':
            # Fully convolutional architecture similar to improve Wasserstein nGAN
            outputs = dcgan_v2_decoder(opts, input, output_dim,
                                            reuse,
                                            is_training,
                                            dropout_rate)
        elif opts['network']['d_arch'] == 'conv_locatello':
            # Fully convolutional architecture similar to Locatello & al.
            outputs = locatello_decoder(opts, input, output_dim,
                                            reuse,
                                            is_training,
                                            dropout_rate)
        elif opts['network']['d_arch'] == 'resnet':
            assert False, 'To Do'
            # Fully convolutional architecture similar to improve Wasserstein nGAN
            outputs = resnet_decoder(opts, input, output_dim,
                                            reuse,
                                            is_training,
                                            dropout_rate)
        elif opts['network']['d_arch'] == 'resnet_v2':
            assert False, 'To Do'
            # Fully convolutional architecture similar to improve Wasserstein nGAN
            outputs = resnet_v2_decoder(opts, input, output_dim,
                                            reuse,
                                            is_training,
                                            dropout_rate)
        else:
            raise ValueError('%s Unknown encoder architecture for mixtures' % opts['network']['d_arch'])

    mean, logSigma = tf.split(outputs,2,axis=-1)
    logSigma = tf.clip_by_value(logSigma, -20, 500)
    Sigma = tf.nn.softplus(logSigma)

    mean = tf.layers.flatten(mean)
    Sigma = tf.layers.flatten(Sigma)

    # if opts['decoder'] == 'det':
    #     x = mean
    # elif opts['decoder'] == 'gauss':
    #     px_params = tf.concat((mean, Sigma), axis=-1)
    #     x = sample_gaussian(px_params, 'tensorflow')
    # elif opts['decoder'] == 'bernoulli':
    #     assert False, 'Bernoulli decoder not implemented yet.'
    #     mean = tf.nn.sigmoid(mean)
    #     # x = sample_bernoulli(mean)
    # else:
    #     assert False, 'Unknown decoder %s' % opts['decoder']

    if opts['model'] == 'TCWAE_MWS' or opts['model'] == 'TCWAE_GAN':
        if opts['input_normalize_sym']:
            x = tf.nn.tanh(mean)
        else:
            x = tf.nn.sigmoid(mean)
    else:
        x = tf.nn.sigmoid(mean)        
    x = tf.reshape(x, [-1] + datashapes[opts['dataset']])

    return x, mean, Sigma


def mlp_encoder(opts, input, output_dim, reuse=False,
                                            is_training=False,
                                            dropout_rate=1.):
    layer_x = input
    for i in range(opts['network']['e_nlayers']):
        layer_x = ops.linear.Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                    opts['network']['e_nfilters'][i], init=opts['mlp_init'], scope='hid{}/lin'.format(i))
        # Note for mlp, batchnorm and layernorm are equivalent
        if opts['normalization']=='batchnorm':
            layer_x = ops.batchnorm.Batchnorm_layers(
                opts, layer_x, 'hid%d/bn' % i, is_training, reuse)
        elif opts['normalization']=='layernorm':
            layer_x = ops.layernorm.Layernorm(
                opts, layer_x, 'hid%d/bn' % i, reuse)
        layer_x = ops._ops.non_linear(layer_x,opts['network']['e_nonlinearity'])
        layer_x = tf.nn.dropout(layer_x, keep_prob=dropout_rate)
    outputs = ops.linear.Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                output_dim, init=opts['mlp_init'], scope='hid_final')

    return outputs


def locatello_encoder(opts, input, output_dim, reuse=False,
                                            is_training=False,
                                            dropout_rate=1.):
    """
    Archi used by Locatello & al.
    """
    layer_x = input
    # Conv block
    for i in range(opts['network']['e_nlayers']):
        layer_x = ops.conv2d.Conv2d(opts, layer_x,layer_x.get_shape().as_list()[-1],opts['network']['e_nfilters'][i],
                opts['network']['filter_size'][i],stride=2,scope='hid{}/conv'.format(i+1),init=opts['conv_init'])
        if opts['normalization']=='batchnorm':
            layer_x = ops.batchnorm.Batchnorm_layers(
                opts, layer_x, 'hid%d/bn' % i, is_training, reuse)
        layer_x = ops._ops.non_linear(layer_x,'relu')
    # 256 FC layer
    layer_x = tf.reshape(layer_x,[-1,np.prod(layer_x.get_shape().as_list()[1:])])
    layer_x = ops.linear.Linear(opts,layer_x,np.prod(layer_x.get_shape().as_list()[1:]),
                256, scope='hid_fc')
    if opts['normalization']=='batchnorm':
        layer_x = ops.batchnorm.Batchnorm_layers(
            opts, layer_x, 'hid_bn' , is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # Final FC
    outputs = ops.linear.Linear(opts,layer_x,np.prod(layer_x.get_shape().as_list()[1:]),
                output_dim, scope='hid_final')

    return outputs


def mlp_decoder(opts, input, output_dim, reuse, is_training,
                                            dropout_rate=1.):
    # Architecture with only fully connected layers and ReLUs
    layer_x = input
    for i in range(opts['network']['d_nlayers']):
        layer_x = ops.linear.Linear(opts, layer_x,np.prod(layer_x.get_shape().as_list()[1:]),
                    opts['network']['d_nfilters'][opts['network']['d_nlayers']-1-i], init=opts['mlp_init'], scope='hid%d/lin' % i)
        layer_x = ops._ops.non_linear(layer_x,opts['network']['d_nonlinearity'])
        # Note for mlp, batchnorm and layernorm are equivalent
        if opts['normalization']=='batchnorm':
            layer_x = ops.batchnorm.Batchnorm_layers(
                opts, layer_x, 'hid%d/bn' % i, is_training, reuse)
        elif opts['normalization']=='layernorm':
            layer_x = ops.layernorm.Layernorm(
                opts, layer_x, 'hid%d/bn' % i, reuse)
        layer_x = tf.nn.dropout(layer_x, keep_prob=dropout_rate)
    outputs = ops.linear.Linear(opts, layer_x,np.prod(layer_x.get_shape().as_list()[1:]),
                np.prod(output_dim), init=opts['mlp_init'], scope='hid_final')

    return outputs


def  locatello_decoder(opts, input, output_dim, reuse,
                                            is_training,
                                            dropout_rate=1.):
    """
    Archi used by Locatello & al.
    """

    # batch_size
    batch_size = tf.shape(input)[0]
    # Linear layers
    h0 = ops.linear.Linear(opts,input,np.prod(input.get_shape().as_list()[1:]),
                256, scope='hid0/lin0')
    if opts['normalization']=='batchnorm':
        h0 = ops.batchnorm.Batchnorm_layers(
            opts, h0, 'hid0/bn0', is_training, reuse)
    h0 = ops._ops.non_linear(h0,'relu')
    h1 = ops.linear.Linear(opts,h0,np.prod(h0.get_shape().as_list()[1:]),
                4*4*64, scope='hid0/lin1')
    if opts['normalization']=='batchnorm':
        h1 = ops.batchnorm.Batchnorm_layers(
            opts, h1, 'hid0/bn0', is_training, reuse)
    h1 = ops._ops.non_linear(h1,'relu')
    h1 = tf.reshape(h1, [-1, 4, 4, 64])
    layer_x = h1
    # Conv block
    for i in range(opts['network']['d_nlayers'] - 1):
        _out_shape = [batch_size, 2*layer_x.get_shape().as_list()[1],
                        2*layer_x.get_shape().as_list()[2],
                        opts['network']['d_nfilters'][opts['network']['d_nlayers']-1-i]]
        layer_x = ops.deconv2d.Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1], _out_shape,
                   opts['network']['filter_size'][opts['network']['d_nlayers']-1-i], stride=2, scope='hid%d/deconv' % i, init= opts['conv_init'])
        if opts['normalization']=='batchnorm':
            layer_x = ops.batchnorm.Batchnorm_layers(
                opts, layer_x, 'hid%d/bn' % (i+2), is_training, reuse)
        layer_x = ops._ops.non_linear(layer_x,'relu')
    outputs = ops.deconv2d.Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1], [batch_size,]+output_dim,
                opts['network']['filter_size'][0], stride=2, scope='hid_final/deconv', init= opts['conv_init'])

    return outputs


def discriminator(opts, input, is_training, dropout_rate=1.):
    """
    Discriminator network for FactorVAE.
    Archtecture is the same than icml paper
    """

    layer_x = tf.layers.flatten(input)
    for i in range(6):
        layer_x = ops.linear.Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                    1000, init='glorot_uniform', scope='hid{}/lin'.format(i))
        # Note for mlp, batchnorm and layernorm are equivalent
        if opts['normalization']=='batchnorm':
            layer_x = ops.batchnorm.Batchnorm_layers(
                opts, layer_x, 'hid%d/bn' % i, is_training)
        layer_x = ops._ops.non_linear(layer_x,'leaky_relu')
        layer_x = tf.nn.dropout(layer_x, keep_prob=dropout_rate)
    logits = ops.linear.Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                2, init='glorot_uniform', scope='hid_final')
    probs = tf.nn.softmax(logits)

    return logits, probs
