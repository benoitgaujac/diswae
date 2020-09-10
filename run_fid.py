import os
from datetime import datetime
import logging
import argparse
import configs
from train import Run
from datahandler import DataHandler
import utils
import itertools

import tensorflow as tf

import pdb

parser = argparse.ArgumentParser()
# Args for experiment
# Data paths
parser.add_argument("--dataset", default='dsprites',
                    help='dataset')
parser.add_argument("--data_dir", type=str, default='../data',
                    help='directory in which data is stored')
parser.add_argument("--out_dir", type=str, default='code_outputs',
                    help='root_directory in which outputs are saved')
parser.add_argument("--res_dir", type=str, default='res',
                    help='directory in which exp. res are saved')
parser.add_argument("--weights_file")
# model set up
parser.add_argument("--model", default='TCWAE_MWS',
                    help='model to train [WAE/BetaVAE/...]')
parser.add_argument("--net_archi", default='conv_locatello',
                    help='networks architecture [mlp/conv_locatello]')
# fid setup
parser.add_argument("--compute_stats", action='store_true', default=False,
                    help='wether compute the stats of the dataset or not.')
parser.add_argument("--fid_inputs", default='samples',
                    help='inputs to compute FID')

FLAGS = parser.parse_args()


# --- Network architectures
mlp_config = { 'e_arch': 'mlp' , 'e_nlayers': 2, 'e_nfilters': [1200, 1200], 'e_nonlinearity': 'relu',
        'd_arch': 'mlp' , 'd_nlayers': 3, 'd_nfilters': [1200, 1200, 1200], 'd_nonlinearity': 'tanh'}

conv_config = { 'e_arch': 'conv_locatello' , 'e_nlayers': 4, 'e_nfilters': [32,32,64,64], 'e_nonlinearity': 'relu',
        'd_arch': 'conv_locatello' , 'd_nlayers': 4, 'd_nfilters': [32,32,32,64], 'd_nonlinearity': 'relu',
        'filter_size': [4,4,4,4]}

net_configs = {'mlp': mlp_config, 'conv_locatello': conv_config}


def main():

    # Select dataset to use
    if FLAGS.dataset == 'dsprites':
        opts = configs.config_dsprites
    elif FLAGS.dataset == 'noisydsprites':
        opts = configs.config_noisydsprites
    elif FLAGS.dataset == 'screamdsprites':
        opts = configs.config_screamdsprites
    elif FLAGS.dataset == 'smallNORB':
        opts = configs.config_smallNORB
    elif FLAGS.dataset == '3dshapes':
        opts = configs.config_3dshapes
    elif FLAGS.dataset == '3Dchairs':
        opts = configs.config_3Dchairs
    elif FLAGS.dataset == 'celebA':
        opts = configs.config_celebA
    elif FLAGS.dataset == 'mnist':
        opts = configs.config_mnist
    else:
        assert False, 'Unknown dataset'

    # Set method param
    opts['data_dir'] = FLAGS.data_dir
    opts['fid'] = True
    opts['network'] = net_configs[FLAGS.net_archi]

    # Model set up
    opts['model'] = FLAGS.model
    if FLAGS.dataset == 'celebA':
        opts['zdim'] = 32
    elif FLAGS.dataset == '3Dchairs':
        opts['zdim'] = 16
    else:
        opts['zdim'] = 10

    # Create directories
    opts['out_dir'] = FLAGS.out_dir
    out_subdir = os.path.join(opts['out_dir'], opts['model'])
    opts['exp_dir'] = os.path.join(out_subdir, FLAGS.res_dir)
    if not tf.io.gfile.isdir(opts['exp_dir']):
        raise Exception("Experiment doesn't exist!")

    #Reset tf graph
    tf.reset_default_graph()

    # Loading the dataset
    data = DataHandler(opts)
    assert data.train_size >= opts['batch_size'], 'Training set too small'

    # init method
    run = Run(opts, data)

    # get fid
    run.fid_score(opts['exp_dir'], FLAGS.weights_file, FLAGS.compute_stats, FLAGS.fid_inputs)

main()
