import sys
import time
import os
from math import sqrt, cos, sin, pi, ceil
import numpy as np
import tensorflow as tf
#import tensorflow_probability as tfp

import pdb

def sample_pz(opts,pz_params,batch_size=100):
    if opts['prior']=='gaussian' or opts['prior']=='implicit':
        noise = sample_gaussian(pz_params, 'numpy', batch_size)
    elif opts['prior']=='dirichlet':
        noise = sample_dirichlet(pz_params, batch_size)
    else:
        assert False, 'Unknown prior %s' % opts['prior']
    return noise

def sample_gaussian(params, typ='numpy', batch_size=100):
    """
    Sample noise from gaussian distribution with parameters
    means and covs
    """
    if typ =='tensorflow':
        means, covs = tf.split(params,2,axis=-1)
        shape = tf.shape(means)
        eps = tf.random.normal(shape, dtype=tf.float32)
        noise = means + tf.multiply(eps,tf.sqrt(1e-10+covs))
    elif typ =='numpy':
        means, covs = np.split(params,2,axis=-1)
        if len(np.shape(means))<2:
            shape = (batch_size,)+np.shape(means)
        else:
            shape = np.shape(means)
        eps = np.random.normal(0.,1.,shape).astype(np.float32)
        noise = means + np.multiply(eps,np.sqrt(1e-10+covs))
    return noise

def sample_dirichlet(alpha, batch_size=100):
    """
    Sample noise from dirichlet distribution with parameters
    alpha
    """
    return np.random.dirichlet(alpha, batch_size)

def sample_unif(shape, minval=0, maxval=None, dtype=tf.float32):
    """
    Sample noise from Unif[minval,maxval]
    """
    return tf.random.uniform(shape, minval, maxval, dtype)

def sample_bernoulli(params):
    """
    Sample noise from Bernoulli distribution with mean parameters
    params
    """
    assert False, 'tfp not available on cluster gpu yet'
    """
    shape = tf.shape(params)
    bernoulli_dist = tfp.distributions.Bernoulli(logits=params, dtype=tf.float32)
    return bernoulli_dist.sample()
    """

def linespace(opts, n, anchors, std=1.0):
    """
    Genereate linear grid space
        - n:  Num of steps in the interpolation
        - anchors[nanchors,zdim]: encoding
    Return:
    linespce[nanchors,n,zdim]: list of linear interpolations for each latent dim
    """
    nanchors = np.shape(anchors)[0]
    zdim = np.shape(anchors)[-1]
    identity = np.eye(zdim)
    linespce = []
    if opts['dataset'] == 'celebA':
        # if opts['model'] == 'TCWAE_MWS' or opts['model'] == 'BetaTCVAE':
        #     std_range = 6*std
        # else:
        std_range = 4*std
    else:
        std_range = 2*std
    for i in range(zdim):
        start = anchors-std_range*identity[i]
        stop = anchors+std_range*identity[i]
        inter = []
        for j in range(nanchors):
            interpolation = np.linspace(start[j],stop[j],n,endpoint=True)
            inter.append(interpolation)
            # int_m = np.linspace(start[j],anchors[j],int(n/2.),endpoint=False)
            # if n%2==0:
            #     int_p = np.linspace(anchors[j],stop[j],int(n/2.),endpoint=True)
            # else:
            #     int_p = np.linspace(anchors[j],stop[j],int(n/2.)+1,endpoint=True)
            # inter.append(np.concatenate((int_m,int_p),axis=0))
        linespce.append(np.array(inter))
    linespce = np.stack(linespce,axis=1)

    return linespce

def generate_linespace(opts, n, mode, anchors):
    """
    Genereate latent linear grid space
    """
    nanchors = np.shape(anchors)[0]
    dim_to_interpolate = min(opts['nmixtures'],opts['zdim'])
    if mode=='transformation':
        assert np.shape(anchors)[1]==0, 'Zdim needs to be 2 to plot transformation'
        ymin, xmin = np.amin(anchors,axis=0)
        ymax, xmax = np.amax(anchors,axis=0)
        x = np.linspace(1.1*xmin,1.1*xmax,n)
        y = np.linspace(1.1*ymin,1.1*ymax,n)
        linespce = np.stack(np.meshgrid(y,x)).T
    elif mode=='points_interpolation':
        assert np.shape(anchors)[0]%2==0, 'Need an ode number of anchors points'
        axs = [[np.linspace(anchors[2*k,d],anchors[2*k+1,d],n) for d in range(dim_to_interpolate)] for k in range(int(nanchors/2))]
        linespce = []
        for i in range(len(axs)):
            crd = np.stack([np.asarray(axs[i][j]) for j in range(dim_to_interpolate)],axis=0).T
            coord = np.zeros((crd.shape[0],opts['zdim']))
            coord[:,:crd.shape[1]] = crd
            linespce.append(coord)
        linespace = np.asarray(linespce)
    elif mode=='priors_interpolation':
        axs = [[np.linspace(anchors[0,d],anchors[k,d],n) for d in range(dim_to_interpolate)] for k in range(1,nanchors)]
        linespce = []
        for i in range(len(axs)):
            crd = np.stack([np.asarray(axs[i][j]) for j in range(dim_to_interpolate)],axis=0).T
            coord = np.zeros((crd.shape[0],opts['zdim']))
            coord[:,:crd.shape[1]] = crd
            linespce.append(coord)
        linespace = np.asarray(linespce)
    else:
        assert False, 'Unknown mode %s for vizualisation' % opts['mode']
    return linespace
