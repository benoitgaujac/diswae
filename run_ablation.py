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
parser.add_argument("--model", default='TCWAE_MWS',
                    help='model to train [WAE/BetaVAE/...]')
parser.add_argument("--mode", default='train',
                    help='mode to run [train/vizu/fid/test]')
parser.add_argument("--data_dir", type=str, default='../data',
                    help='directory in which data is stored')
parser.add_argument("--out_dir", type=str, default='code_outputs',
                    help='root_directory in which outputs are saved')
parser.add_argument("--exp_dir", type=str, default='res',
                    help='directory in which exp. outputs are saved')
parser.add_argument("--num_it", type=int, default=300000,
                    help='iteration number')
parser.add_argument("--idx", type=int, default=0,
                    help='idx latent reg weight setup')
parser.add_argument("--save_model", default='True',
                    help='save final model weights [True/False]')
parser.add_argument("--save_data", default='True',
                    help='save training data [True/False]')
parser.add_argument("--weights_file")
parser.add_argument('--gpu_id', default='cpu',
                    help='gpu id for DGX box. Default is cpu')


FLAGS = parser.parse_args()


# --- Network architectures
mlp_config = { 'e_arch': 'mlp' , 'e_nlayers': 2, 'e_nfilters': [1200, 1200], 'e_nonlinearity': 'relu',
        'd_arch': 'mlp' , 'd_nlayers': 3, 'd_nfilters': [1200, 1200, 1200], 'd_nonlinearity': 'tanh'}

conv_config = { 'e_arch': 'conv_locatello' , 'e_nlayers': 4, 'e_nfilters': [32,32,64,64], 'e_nonlinearity': 'relu',
        'd_arch': 'conv_locatello' , 'd_nlayers': 4, 'd_nfilters': [32,32,32,64], 'd_nonlinearity': 'relu',
        'filter_size': [4,4,4,4], 'downsample': [None,None,None,None], 'upsample': [None,None,None,None]}

net_configs = {'mlp': mlp_config, 'conv_locatello': conv_config}


def main():

    # Select dataset to use
    opts = configs.config_dsprites

    # Select training method
    if FLAGS.model:
        opts['model'] = FLAGS.model

    # Data directory
    opts['data_dir'] = FLAGS.data_dir

    # Mode
    if FLAGS.mode=='fid':
        opts['fid'] = True
    else:
        opts['fid'] = False

    # Opt set up
    opts['lr'] = 0.0004

    # Model set up
    opts['zdim'] = 10
    opts['batch_size'] = 64
    opts['cost'] = 'xentropy'
    # Objective Function Coefficients
    lmba = [1, 2, 5, 10, 15, 20, 25, 50, 100, 150]
    opts['obj_fn_coeffs'] = lmba[FLAGS.idx-1]
    if opts['model']!='WAE':
        lmba = list(zip(lmba,lmba))
        opts['obj_fn_coeffs'] = list(lmba[FLAGS.idx-1])
    # Penalty Sigma_q
    opts['pen_enc_sigma'] = True
    opts['lambda_pen_enc_sigma'] = 1.
    # NN set up
    opts['network'] = net_configs['conv_locatello']

    # Create directories
    if FLAGS.out_dir:
        opts['out_dir'] = FLAGS.out_dir
    if FLAGS.exp_dir:
        opts['exp_dir'] = FLAGS.exp_dir
    if opts['model'] == 'TCWAE_MWS' or opts['model'] == 'TCWAE_GAN':
        exp_dir = os.path.join(opts['out_dir'],
                               opts['model'],
                               '{}_{}_{}_{:%Y_%m_%d_%H_%M}'.format(
                                    opts['exp_dir'],
                                    opts['obj_fn_coeffs'][0],
                                    opts['obj_fn_coeffs'][1],datetime.now()), )
    else :
        exp_dir = os.path.join(opts['out_dir'],
                               opts['model'],
                               '{}_{}_{:%Y_%m_%d_%H_%M}'.format(
                                    opts['exp_dir'],
                                    opts['obj_fn_coeffs'],
                                    datetime.now()), )
    opts['exp_dir'] = exp_dir
    if not tf.gfile.IsDirectory(exp_dir):
        utils.create_dir(exp_dir)
        utils.create_dir(os.path.join(exp_dir, 'checkpoints'))

    # Verbose
    logging.basicConfig(filename=os.path.join(exp_dir,'outputs.log'),
        level=logging.INFO, format='%(asctime)s - %(message)s')

    # Loading the dataset
    data = DataHandler(opts)
    assert data.num_points >= opts['batch_size'], 'Training set too small'

    # Experiemnts set up
    opts['epoch_num'] = int(FLAGS.num_it / int(data.num_points/opts['batch_size']))
    opts['print_every'] = int(opts['epoch_num'] / 2.) * int(data.num_points/opts['batch_size'])-1
    opts['evaluate_every'] = int(opts['print_every'] / 2.) + 1
    opts['save_every'] = 1000000000
    if FLAGS.save_model=='True':
        opts['save_final'] = True
    else:
        opts['save_final'] = False
    if FLAGS.save_data=='True':
        opts['save_train_data'] = True
    else:
        opts['save_train_data'] = False
    opts['vizu_encSigma'] = False


    #Reset tf graph
    tf.reset_default_graph()

    run = Run(opts)

    # Training/testing/vizu
    if FLAGS.mode=="train":
        # Dumping all the configs to the text file
        with utils.o_gfile((exp_dir, 'params.txt'), 'w') as text:
            text.write('Parameters:\n')
            for key in opts:
                text.write('%s : %s\n' % (key, opts[key]))
        run.train(data, FLAGS.weights_file)
    else:
        assert False, 'Unknown mode %s' % FLAGS.mode


main()
