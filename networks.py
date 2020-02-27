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

    if opts['decoder'] == 'det':
        x = mean
    elif opts['decoder'] == 'gauss':
        px_params = tf.concat((mean, Sigma), axis=-1)
        x = sample_gaussian(px_params, 'tensorflow')
    elif opts['decoder'] == 'bernoulli':
        assert False, 'Bernoulli decoder not implemented yet.'
        mean = tf.nn.sigmoid(mean)
        # x = sample_bernoulli(mean)
    else:
        assert False, 'Unknown decoder %s' % opts['decoder']

    if opts['input_normalize_sym']:
        x = tf.nn.tanh(x)
    else:
        x = tf.nn.sigmoid(x)
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

def dcgan_encoder(opts, input, output_dim, reuse=False,
                                            is_training=False,
                                            dropout_rate=1.):
    """
    DCGAN style network.
    Final dense layer with output of size output_dim.
    """
    layer_x = input
    # Conv block
    for i in range(opts['network']['e_nlayers']):
        if opts['network']['downsample'][i]:
            layer_x = ops.conv2d.Conv2d(opts, layer_x,layer_x.get_shape().as_list()[-1],opts['network']['e_nfilters'][i],
                    opts['network']['filter_size'][i],stride=2,scope='hid{}/conv'.format(i+1),init=opts['conv_init'])
        else:
            layer_x = ops.conv2d.Conv2d(opts,layer_x,layer_x.get_shape().as_list()[-1],opts['network']['e_nfilters'][i],
                    opts['network']['filter_size'][i],stride=1,scope='hid{}/conv'.format(i),init=opts['conv_init'])
        if opts['normalization']=='batchnorm':
            layer_x = ops.batchnorm.Batchnorm_layers(
                opts, layer_x, 'hid%d/bn' % i, is_training, reuse)
        elif opts['normalization']=='layernorm':
            layer_x = ops.layernorm.Layernorm(
                opts, layer_x, 'hid%d/bn' % i, reuse)
        layer_x = ops._ops.non_linear(layer_x,opts['network']['e_nonlinearity'])
        layer_x = tf.nn.dropout(layer_x, keep_prob=dropout_rate)
    # Final linear
    layer_x = tf.reshape(layer_x,[-1,np.prod(output_dim)])
    outputs = ops.linear.Linear(opts,layer_x,np.prod(layer_x.get_shape().as_list()[1:]),
                output_dim, scope='hid_final')

    return outputs

def dcgan_v2_encoder(opts, input, output_dim, reuse=False,
                                            is_training=False,
                                            dropout_rate=1.):
    """
    Fully convolutional DCGAN style network.
    Final conv layer ouput give size of output.
    """
    layer_x = input
    # Conv block
    for i in range(opts['network']['e_nlayers']):
        if opts['network']['downsample'][i]:
            layer_x = ops.conv2d.Conv2d(opts, layer_x,layer_x.get_shape().as_list()[-1],opts['network']['e_nfilters'][i],
                    opts['network']['filter_size'][i],stride=2,scope='hid{}/conv'.format(i+1),init=opts['conv_init'])
        else:
            layer_x = ops.conv2d.Conv2d(opts,layer_x,layer_x.get_shape().as_list()[-1],opts['network']['e_nfilters'][i],
                    opts['network']['filter_size'][i],stride=1,scope='hid{}/conv'.format(i),init=opts['conv_init'])
        if opts['normalization']=='batchnorm':
            layer_x = ops.batchnorm.Batchnorm_layers(
                opts, layer_x, 'hid%d/bn' % i, is_training, reuse)
        elif opts['normalization']=='layernorm':
            layer_x = ops.layernorm.Layernorm(
                opts, layer_x, 'hid%d/bn' % i, reuse)
        layer_x = ops._ops.non_linear(layer_x,opts['network']['e_nonlinearity'])
        layer_x = tf.nn.dropout(layer_x, keep_prob=dropout_rate)
    # Final conv linear
    assert len(opts['network']['downsample'])>opts['network']['e_nlayers'], \
                'Need to pass nlayers+1 filters & downsample description'
    if opts['network']['downsample'][i+1]:
        outputs = ops.conv2d.Conv2d(opts, layer_x,layer_x.get_shape().as_list()[-1],2*opts['network']['e_nfilters'][i+1],
                opts['network']['filter_size'][i],stride=2,scope='hid_final',init=opts['conv_init'])
    else:
        outputs = ops.conv2d.Conv2d(opts,layer_x,layer_x.get_shape().as_list()[-1],opts['network']['e_nfilters'][i+1],
                opts['network']['filter_size'][i],stride=1,scope='hid_final',init=opts['conv_init'])
    assert np.prod(outputs.get_shape().as_list()[-1])==output_dim, 'latent dimension mismatch'

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
        layer_x = ops._ops.non_linear(layer_x,'relu')
        layer_x = tf.nn.dropout(layer_x, keep_prob=dropout_rate)
    # 256 FC layer
    layer_x = tf.reshape(layer_x,[-1,np.prod(layer_x.get_shape().as_list()[1:])])
    layer_x = ops.linear.Linear(opts,layer_x,np.prod(layer_x.get_shape().as_list()[1:]),
                256, scope='hid_fc')
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # Final FC
    outputs = ops.linear.Linear(opts,layer_x,np.prod(layer_x.get_shape().as_list()[1:]),
                output_dim, scope='hid_final')

    return outputs

def resnet_encoder(opts, input, output_dim, reuse=False,
                                            is_training=False,
                                            dropout_rate=1.):
    """
    Same than dcgan_v2 but with residual connection.
    output_dim:     dim of output latent
    features_dim:   shape of input FEATURES [w,h,c]
    """

    layer_x = input
    # -- Reshapping to features_dim if needed
    if layer_x.get_shape().as_list()[1:-1]!=features_dim[:-1]:
        layer_x = ops.linear.Linear(opts,layer_x,np.prod(layer_x.get_shape().as_list()[1:]),
                    np.prod(features_dim), scope='hid_linear_init')
        layer_x = tf.reshape(layer_x,[-1,]+features_dim)
    # -- Conv block
    conv = layer_x
    for i in range(num_layers-1):
        conv = ops.conv2d.Conv2d(opts,conv,conv.get_shape().as_list()[-1],num_units,
                filter_size,stride=1,scope='hid{}/conv'.format(i),init=opts['conv_init'])
        if opts['normalization']=='batchnorm':
            conv = ops.batchnorm.Batchnorm_layers(
                opts, conv, 'hid%d/bn' % (i+1), is_training, reuse)
        elif opts['normalization']=='layernorm':
            conv = ops.layernorm.Layernorm(
                opts, conv, 'hid%d/bn' % (i+1), reuse)
        conv = ops._ops.non_linear(conv,opts['network']['e_nonlinearity'])
        conv = tf.nn.dropout(conv, keep_prob=dropout_rate)
    # Last conv resampling
    if resample=='down':
        conv = ops.conv2d.Conv2d(opts, conv,conv.get_shape().as_list()[-1],2*num_units,
                filter_size,stride=2,scope='hid{}/conv'.format(i+1),init=opts['conv_init'])
    elif resample==None:
        conv = ops.conv2d.Conv2d(opts, conv,conv.get_shape().as_list()[-1],num_units,
                filter_size,stride=1,scope='hid{}/conv'.format(i+1),init=opts['conv_init'])
    else:
        assert False, 'Resample %s not allowed for encoder' % resample
    out_shape = conv.get_shape().as_list()[1:]
    # -- Shortcut
    if resample=='down':
        shortcut = ops.conv2d.Conv2d(opts, layer_x,layer_x.get_shape().as_list()[-1],2*num_units,
                filter_size,stride=2,scope='hid_shortcut',init=opts['conv_init'])
    elif resample==None:
        if conv.get_shape().as_list()[1:]==layer_x.get_shape().as_list()[1:]:
            shortcut = layer_x
        else:
            shortcut = ops.conv2d.Conv2d(opts, layer_x,layer_x.get_shape().as_list()[-1],num_units,
                    filter_size,stride=1,scope='hid_shortcut',init=opts['conv_init'])
    else:
        assert False, 'Resample %s not allowed for encoder' % resample
    # -- Resnet output
    outputs = conv + shortcut
    if opts['normalization']=='batchnorm':
        outputs = ops.batchnorm.Batchnorm_layers(
            opts, outputs, 'hid%d/bn' % (i+2), is_training, reuse)
    elif opts['normalization']=='layernorm':
        outputs = ops.layernorm.Layernorm(
            opts, outputs, 'hid%d/bn' % (i+2), reuse)
    outputs = ops._ops.non_linear(outputs,opts['network']['e_nonlinearity'])
    outputs = tf.nn.dropout(outputs, keep_prob=dropout_rate)
    outputs = ops.linear.Linear(opts,outputs,np.prod(outputs.get_shape().as_list()[1:]),
                output_dim, scope='hid_final')

    return outputs, out_shape

def resnet_v2_encoder(opts, input, output_dim, reuse=False,
                                            is_training=False,
                                            dropout_rate=1.):
    """
    Full conv resnet.
    output_dim:     number of output channels for intermediate latents / dimension of top latent
    features_dim:   shape of input [w,h,c]
    """

    # batch_size
    batch_size = tf.shape(input)[0]
    # -- Reshapping to if needed features dim
    if input.get_shape().as_list()[1:]!=features_dim:
        layer_x = tf.reshape(input,[-1,]+features_dim)
    else:
        layer_x = input
    # -- Conv block
    conv = layer_x
    for i in range(num_layers-1):
        conv = ops.conv2d.Conv2d(opts,conv,conv.get_shape().as_list()[-1],num_units,
                filter_size,stride=1,scope='hid{}/conv'.format(i),init=opts['conv_init'])
        if opts['normalization']=='batchnorm':
            conv = ops.batchnorm.Batchnorm_layers(
                opts, conv, 'hid%d/bn' % (i+1), is_training, reuse)
        elif opts['normalization']=='layernorm':
            conv = ops.layernorm.Layernorm(
                opts, conv, 'hid%d/bn' % (i+1), reuse)
        conv = ops._ops.non_linear(conv,opts['network']['e_nonlinearity'])
        conv = tf.nn.dropout(conv, keep_prob=dropout_rate)
    # Last conv resampling
    if resample=='down':
        conv = ops.conv2d.Conv2d(opts, conv,conv.get_shape().as_list()[-1],2*num_units,
                filter_size,stride=2,scope='hid{}/conv'.format(i+1),init=opts['conv_init'])
    elif resample==None:
        conv = ops.conv2d.Conv2d(opts, conv,conv.get_shape().as_list()[-1],num_units,
                filter_size,stride=1,scope='hid{}/conv'.format(i+1),init=opts['conv_init'])
    else:
        assert False, 'Resample %s not allowed for encoder' % resample
    # -- Shortcut
    if resample=='down':
        shortcut = ops.conv2d.Conv2d(opts, layer_x,layer_x.get_shape().as_list()[-1],2*num_units,
                filter_size,stride=2,scope='hid_shortcut',init='normilized_glorot')
    elif resample==None:
        if conv.get_shape().as_list()[1:]==layer_x.get_shape().as_list()[1:]:
            shortcut = layer_x
        else:
            shortcut = ops.conv2d.Conv2d(opts, layer_x,layer_x.get_shape().as_list()[-1],num_units,
                    filter_size,stride=1,scope='hid_shortcut',init='normilized_glorot')
    else:
        assert False, 'Resample %s not allowed for encoder' % resample
    # -- Resnet output
    outputs = conv + shortcut
    if opts['normalization']=='batchnorm':
        outputs = ops.batchnorm.Batchnorm_layers(
            opts, outputs, 'hid%d/bn' % (i+2), is_training, reuse)
    elif opts['normalization']=='layernorm':
        outputs = ops.layernorm.Layernorm(
            opts, outputs, 'hid%d/bn' % (i+2), reuse)
    outputs = ops._ops.non_linear(outputs,opts['network']['e_nonlinearity'])
    outputs = tf.nn.dropout(outputs, keep_prob=dropout_rate)

    # Shape
    if top_latent:
        output_dim = int(output_dim / outputs.get_shape().as_list()[1] / outputs.get_shape().as_list()[2])
    out_shape = outputs.get_shape().as_list()[1:-1] + [int(output_dim/2),]
    # last hidden layer
    if last_archi=='dense':
        # -- dense layer
        outputs = ops.linear.Linear(opts,outputs,np.prod(outputs.get_shape().as_list()[1:]),
                    2*np.prod(out_shape), scope='hid_final')
    elif last_archi=='conv1x1':
        # -- 1x1 conv
        outputs = ops.conv2d.Conv2d(opts,outputs,outputs.get_shape().as_list()[-1],output_dim,
                1,stride=1,scope='hid_final',init=opts['conv_init'])
    elif last_archi=='conv':
        # -- conv
        outputs = ops.conv2d.Conv2d(opts,outputs,outputs.get_shape().as_list()[-1],output_dim,
                filter_size,stride=1,scope='hid_final',init=opts['conv_init'])
    else:
        assert False, 'Unknown last_archi %s ' % last_archi

    return outputs, out_shap


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


def  dcgan_decoder(opts, input, output_dim, reuse,
                                            is_training,
                                            dropout_rate=1.):
    """
    DCGAN style network with stride 2 at each hidden deconvolution layers.
    First dense layer reshape to [out_h/2**num_layers,out_w/2**num_layers,num_units].
    Then num_layers deconvolutions with stride 2 and num_units filters.
    Last deconvolution output a 3-d latent code [out_h,out_w,2].
    """

    batch_size = tf.shape(input)[0]
    height = output_dim[0] / 2**np.sum(opts['network']['upsample'])
    width = output_dim[1] / 2**np.sum(opts['network']['upsample'])
    h0 = ops.linear.Linear(opts,input,np.prod(input.get_shape().as_list()[1:]),
            opts['network']['d_nfilters'][-1] * ceil(height) * ceil(width), scope='hid0/lin')
    if opts['normalization']=='batchnorm':
        h0 = ops.batchnorm.Batchnorm_layers(
            opts, h0, 'hid0/bn_lin', is_training, reuse)
    elif opts['normalization']=='layernorm':
        h0 = ops.layernorm.Layernorm(
            opts, h0, 'hid0/bn_lin', reuse)
    h0 = tf.reshape(h0, [-1, ceil(height), ceil(width), opts['network']['d_nfilters'][-1]])
    h0 = ops._ops.non_linear(h0,opts['network']['d_nonlinearity'])
    layer_x = h0
    for i in range(opts['network']['d_nlayers'] - 1):
        scale = 2**(i + 1)
        _out_shape = [batch_size, ceil(height * scale), ceil(width * scale),
                                    opts['network']['d_nfilters'][opts['network']['d_nlayers']-1-i]]
        layer_x = ops.deconv2d.Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1], _out_shape,
                   opts['network']['filter_size'][opts['network']['d_nlayers']-1-i], scope='hid%d/deconv' % i, init= opts['conv_init'])
        if opts['normalization']=='batchnorm':
            layer_x = ops.batchnorm.Batchnorm_layers(
                opts, layer_x, 'hid%d/bn' % i, is_training, reuse)
        elif opts['normalization']=='layernorm':
            layer_x = ops.layernorm.Layernorm(
                opts, layer_x, 'hid%d/bn' % i, reuse)
        layer_x = ops._ops.non_linear(layer_x,opts['network']['d_nonlinearity'])
        layer_x = tf.nn.dropout(layer_x, keep_prob=dropout_rate)
    _out_shape = [batch_size] + list(output_dim)
    outputs = ops.deconv2d.Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1], _out_shape,
                opts['network']['filter_size'][0], scope='hid_final/deconv', init= opts['conv_init'])

    return outputs

def  dcgan_v2_decoder(opts, input, archi, num_layers, num_units,
                                                        filter_size,
                                                        output_dim,
                                                        features_dim,
                                                        resample,
                                                        reuse,
                                                        is_training,
                                                        dropout_rate=1.):
    """
    DCGAN style network. First deconvolution layer can have stride 2.
    First dense layer reshape to [features_dim[0]/2,features_dim[1]/2,2*num_units] if resampling up
    or [features_dim[0],features_dim[1],num_units] if no resampling.
    Then num_layers-1 deconvolutions with num_units filters.
    Final dense layer with output's dimension of output_dim.
    """

    # batch_size
    batch_size = tf.shape(input)[0]
    # Reshapping to linear
    if resample=='up':
        if num_units!=features_dim[2]:
            logging.error('num units decoder not matching num_units decoder')
        # handeling padding
        if features_dim[0]%2==0:
            reshape = [int(features_dim[0]/2),int(features_dim[1]/2),2*num_units]
        else:
            reshape = [int((features_dim[0]+1)/2),int((features_dim[1]+1)/2),2*num_units]
    elif resample==None:
        if num_units!=features_dim[2]:
            logging.error('num units decoder not matching num_units decoder')
        reshape = [features_dim[0], features_dim[1], num_units]
    else:
        assert False, 'Resample {} not allowed for encoder'.format(resample)
    h0 = ops.linear.Linear(opts,input,np.prod(input.get_shape().as_list()[1:]),
            np.prod(reshape), scope='hid0/lin')
    h0 = tf.reshape(h0, [-1,]+ reshape)
    if opts['normalization']=='batchnorm':
        h0 = ops.batchnorm.Batchnorm_layers(
                    opts, h0, 'hid0/bn', is_training, reuse)
    elif opts['normalization']=='layernorm':
        h0 = ops.layernorm.Layernorm(
                    opts, h0, 'hid0/bn', reuse)
    h0 = ops._ops.non_linear(h0,opts['network']['d_nonlinearity'])
    layer_x = h0
    # First deconv resampling
    if resample=='up':
        layer_x = ops.deconv2d.Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1], [batch_size,]+features_dim,
                    filter_size, stride=2, scope='hid0/deconv', init=opts['conv_init'])
    elif resample==None:
        # layer_x = ops.deconv2d.Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1], [-1,]+features_dim,
        #             filter_size, stride=1, scope='hid0/deconv', init=opts['conv_init'])
        layer_x = ops.conv2d.Conv2d(opts, layer_x,layer_x.get_shape().as_list()[-1], num_units,
                filter_size, stride=1, scope='hid0/deconv', init=opts['conv_init'])

    else:
        assert False, 'Resample {} not allowed for encoder'.format(resample)
    # Deconv block
    for i in range(num_layers - 1):
        if opts['normalization']=='batchnorm':
            layer_x = ops.batchnorm.Batchnorm_layers(
                        opts, layer_x, 'hid%d/bn' % (i+1), is_training, reuse)
        elif opts['normalization']=='layernorm':
            layer_x = ops.layernorm.Layernorm(
                        opts, layer_x, 'hid%d/bn' % (i+1), reuse)
        layer_x = ops._ops.non_linear(layer_x,opts['network']['d_nonlinearity'])
        layer_x = tf.nn.dropout(layer_x, keep_prob=dropout_rate)
        # layer_x = ops.deconv2d.Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1], [-1,]+features_dim,
        #             filter_size, stride=1, scope='hid%d/deconv' % (i+1), init= opts['conv_init'])
        layer_x = ops.conv2d.Conv2d(opts, layer_x,layer_x.get_shape().as_list()[-1], num_units,
                filter_size, stride=1, scope='hid%d/deconv' % (i+1), init=opts['conv_init'])
    # Final linear
    outputs = ops.linear.Linear(opts,layer_x,np.prod(layer_x.get_shape().as_list()[1:]),
                np.prod(output_dim), scope='hid_final')

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
    h0 = ops._ops.non_linear(h0,'relu')
    h1 = ops.linear.Linear(opts,h0,np.prod(h0.get_shape().as_list()[1:]),
                4*4*64, scope='hid0/lin1')
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
        layer_x = ops._ops.non_linear(layer_x,'relu')
        layer_x = tf.nn.dropout(layer_x, keep_prob=dropout_rate)
    outputs = ops.deconv2d.Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1], [batch_size,]+output_dim,
                opts['network']['filter_size'][0], stride=2, scope='hid_final/deconv', init= opts['conv_init'])

    return outputs

def  resnet_decoder(opts, input, archi, num_layers, num_units,
                                                        filter_size,
                                                        output_dim,
                                                        features_dim,
                                                        resample,
                                                        reuse,
                                                        is_training,
                                                        dropout_rate=1.):
    """
    Same than dcgan_v2 but with residual connection.
    Final hidden layer can be dense layer, 1x1 conv or big-kernel conv.
    output_dim:     shape/dim of output latent
    features_dim:   shape of ouput features [w,h,c]
    """

    # batch_size
    batch_size = tf.shape(input)[0]
    # -- Reshapping to features dim
    if resample=='up':
        if num_units!=features_dim[2]:
            logging.error('num units decoder not matching num_units decoder')
        if features_dim[0]%2==0:
            reshape = [int(features_dim[0]/2),int(features_dim[1]/2),2*num_units]
        else:
            reshape = [int((features_dim[0]+1)/2),int((features_dim[1]+1)/2),2*num_units]
    elif resample==None:
        if num_units!=features_dim[2]:
            logging.error('num units decoder not matching num_units decoder')
        reshape = [features_dim[0], features_dim[1], num_units]
    else:
        assert False, 'Resample {} not allowed for encoder'.format(resample)
    layer_x = ops.linear.Linear(opts,input,np.prod(input.get_shape().as_list()[1:]),
            np.prod(reshape), scope='hid0/lin')
    layer_x = tf.reshape(layer_x, [-1,]+ reshape)
    # -- Conv block
    conv = layer_x
    # First deconv resampling
    if resample=='up':
        output_shape = [batch_size,features_dim[0],features_dim[1],num_units]
        conv = ops.deconv2d.Deconv2D(opts, conv, conv.get_shape().as_list()[-1], output_shape,
                    filter_size, stride=2, scope='hid0/deconv', init=opts['conv_init'])
    elif resample==None:
        conv = ops.conv2d.Conv2d(opts, conv,conv.get_shape().as_list()[-1], num_units,
                filter_size, stride=1, scope='hid0/deconv', init=opts['conv_init'])
    else:
        assert False, 'Resample {} not allowed for encoder'.format(resample)
    # Deconv block
    for i in range(num_layers - 1):
        if opts['normalization']=='batchnorm':
            conv = ops.batchnorm.Batchnorm_layers(
                        opts, conv, 'hid%d/bn' % (i+1), is_training, reuse)
        elif opts['normalization']=='layernorm':
            conv = ops.layernorm.Layernorm(
                        opts, conv, 'hid%d/bn' % (i+1), reuse)
        conv = ops._ops.non_linear(conv,opts['network']['d_nonlinearity'])
        # conv = tf.nn.dropout(conv, keep_prob=dropout_rate)
        conv = ops.conv2d.Conv2d(opts, conv,conv.get_shape().as_list()[-1], num_units,
                filter_size, stride=1, scope='hid%d/deconv' % (i+1), init=opts['conv_init'])
    # -- Shortcut
    if resample=='up':
        output_shape = [batch_size,features_dim[0],features_dim[1],num_units]
        shortcut = ops.deconv2d.Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1], output_shape,
                    filter_size, stride=2, scope='hid_shortcut', init=opts['conv_init'])
    elif resample==None:
        if conv.get_shape().as_list()[-1]==layer_x.get_shape().as_list()[-1]:
            shortcut = layer_x
        else:
            shortcut = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1], num_units,
                        filter_size, stride=1, scope='hid_shortcut', init=opts['conv_init'])
    else:
        assert False, 'Resample {} not allowed for encoder'.format(resample)
    # -- Resnet output
    outputs = conv + shortcut
    if opts['normalization']=='batchnorm':
        outputs = ops.batchnorm.Batchnorm_layers(
                    opts, outputs, 'hid%d/bn' % (i+2), is_training, reuse)
    elif opts['normalization']=='layernorm':
        outputs = ops.layernorm.Layernorm(
                    opts, outputs, 'hid%d/bn' % (i+2), reuse)
    outputs = ops._ops.non_linear(outputs,opts['network']['d_nonlinearity'])
    if np.prod(output_dim)==2*np.prod(features_dim):
        outputs = ops.conv2d.Conv2d(opts, outputs,outputs.get_shape().as_list()[-1], output_dim[-1],
                filter_size, stride=1, scope='hid_final', init=opts['conv_init'])
    else:
        outputs = ops.linear.Linear(opts,outputs,np.prod(outputs.get_shape().as_list()[1:]),
                    np.prod(output_dim), scope='hid_final')

    return outputs

def  resnet_v2_decoder(opts, input, archi, num_layers, num_units,
                                                        filter_size,
                                                        output_dim,
                                                        features_dim,
                                                        resample,
                                                        last_archi,
                                                        reuse,
                                                        is_training,
                                                        dropout_rate=1.):
    """
    Full conv resnet
    output_dim:     number of output channels
    features_dim:   shape of input latent [w,h,c]
    """

    # batch_size
    batch_size = tf.shape(input)[0]
    # -- Reshapping to features dim
    layer_x = tf.reshape(input,[-1,]+features_dim)

    conv = layer_x
    # First deconv resampling
    if resample=='up':
        output_shape = [batch_size,2*features_dim[0],2*features_dim[1],int(num_units/2)]
        conv = ops.deconv2d.Deconv2D(opts, conv, conv.get_shape().as_list()[-1], output_shape,
                    filter_size, stride=2, scope='hid0/deconv', init=opts['conv_init'])
    elif resample==None:
        conv = ops.conv2d.Conv2d(opts, conv,conv.get_shape().as_list()[-1], num_units,
                filter_size, stride=1, scope='hid0/deconv', init=opts['conv_init'])
    else:
        assert False, 'Resample {} not allowed for encoder'.format(resample)
    # Deconv block
    for i in range(num_layers - 1):
        if opts['normalization']=='batchnorm':
            conv = ops.batchnorm.Batchnorm_layers(
                        opts, conv, 'hid%d/bn' % (i+1), is_training, reuse)
        elif opts['normalization']=='layernorm':
            conv = ops.layernorm.Layernorm(
                        opts, conv, 'hid%d/bn' % (i+1), reuse)
        conv = ops._ops.non_linear(conv,opts['network']['d_nonlinearity'])
        # conv = tf.nn.dropout(conv, keep_prob=dropout_rate)
        conv = ops.conv2d.Conv2d(opts, conv,conv.get_shape().as_list()[-1], num_units,
                filter_size, stride=1, scope='hid%d/deconv' % (i+1), init=opts['conv_init'])
    # -- Shortcut
    if resample=='up':
        output_shape = [batch_size,2*features_dim[0],2*features_dim[1],num_units]
        shortcut = ops.deconv2d.Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1], output_shape,
                    filter_size, stride=2, scope='hid_shortcut', init='normilized_glorot')
    elif resample==None:
        if conv.get_shape().as_list()[-1]==layer_x.get_shape().as_list()[-1]:
            shortcut = layer_x
        else:
            shortcut = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1], num_units,
                        filter_size, stride=1, scope='hid_shortcut', init='normilized_glorot')
    else:
        assert False, 'Resample {} not allowed for encoder'.format(resample)
    # -- Resnet output
    outputs = conv + shortcut
    if opts['normalization']=='batchnorm':
        outputs = ops.batchnorm.Batchnorm_layers(
                    opts, outputs, 'hid%d/bn' % (i+2), is_training, reuse)
    elif opts['normalization']=='layernorm':
        outputs = ops.layernorm.Layernorm(
                    opts, outputs, 'hid%d/bn' % (i+2), reuse)
    outputs = ops._ops.non_linear(outputs,opts['network']['d_nonlinearity'])

    # last hidden layer
    if last_archi=='dense':
        # -- dense layer
        if np.prod(output_dim)==2*np.prod(datashapes[opts['dataset']]):
            outputs = ops.conv2d.Conv2d(opts, outputs,outputs.get_shape().as_list()[-1],output_dim[-1],
                    filter_size,stride=1,scope='hid_final',init=opts['conv_init'])
        else:
            output_shape = [outputs.get_shape().as_list()[1],outputs.get_shape().as_list()[2],output_dim[-1]]
            outputs = ops.linear.Linear(opts,outputs,np.prod(outputs.get_shape().as_list()[1:]),
                        np.prod(output_shape), scope='hid_final')
    elif last_archi=='conv1x1':
        # -- 1x1 conv
        outputs = ops.conv2d.Conv2d(opts, outputs,outputs.get_shape().as_list()[-1],output_dim[-1],
                1,stride=1,scope='hid_final',init=opts['conv_init'])
    elif last_archi=='conv':
        # -- conv
        outputs = ops.conv2d.Conv2d(opts, outputs,outputs.get_shape().as_list()[-1],output_dim[-1],
                filter_size,stride=1,scope='hid_final',init=opts['conv_init'])
    else:
        assert False, 'Unknown last_archi %s ' % last_archi


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
