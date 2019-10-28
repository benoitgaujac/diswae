import os
import sys
import logging
import argparse
import configs
from train import WAE
from datahandler import DataHandler
import utils

import tensorflow as tf
import itertools

import pdb

parser = argparse.ArgumentParser()
# Args for experiment
parser.add_argument("--model", default='WAE',
                    help='model to train [WAE/BetaVAE/...]')
parser.add_argument("--mode", default='train',
                    help='mode to run [train/vizu/fid/test]')
parser.add_argument("--exp", default='mnist',
                    help='dataset [mnist/cifar10/].'\
                    ' celebA/dsprites Not implemented yet')
parser.add_argument("--data_dir", type=str, default='../data',
                    help='directory in which data is stored')
parser.add_argument("--out_dir", type=str, default='code_outputs',
                    help='root_directory in which outputs are saved')
parser.add_argument("--exp_dir", type=str, default='results',
                    help='directory in which exp. outputs are saved')
parser.add_argument("--enum", type=int, default=100,
                    help='epoch number')
parser.add_argument("--enet_archi", default='mlp',
                    help='encoder networks architecture [mlp/dcgan_v2/resnet]')
parser.add_argument("--dnet_archi", default='mlp',
                    help='decoder networks architecture [mlp/dcgan_v2/resnet]')
parser.add_argument("--idx_lmba", type=int, default=0,
                    help='idx lambda setup')
parser.add_argument("--idx_beta", type=float, default=10.,
                    help='idx lambda setup')
parser.add_argument("--weights_file")
parser.add_argument('--gpu_id', default='cpu',
                    help='gpu id for DGX box. Default is cpu')


FLAGS = parser.parse_args()

def main():

    # Select dataset to use
    if FLAGS.exp == 'dsprites':
        opts = configs.config_dsprites
    elif FLAGS.exp == 'smallNORB':
        opts = configs.config_smallNORB
    elif FLAGS.exp == '3Dchairs':
        opts = configs.config_3Dchairs
    elif FLAGS.exp == 'celebA':
        assert False, 'CelebA dataset not implemented yet.'
        opts = configs.config_celebA
    elif FLAGS.exp == 'mnist':
        opts = configs.config_mnist
    else:
        assert False, 'Unknown experiment dataset'

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

    # Experiemnts set up
    opts['epoch_num'] = FLAGS.enum
    opts['print_every'] = 60000
    opts['save_every_epoch'] = 1000000
    opts['save_final'] = False
    opts['save_train_data'] = False
    opts['vizu_encSigma'] = True

    # Model set up
    opts['zdim'] = 10

    # Objective Function Coefficients
    if opts['model'] == 'BetaVAE':
        beta = [1, 3, 10, 20, 30, 40, 50, 75, 100]
        opts['beta'] = beta[FLAGS.idx_beta-1]
    elif opts['model'] == 'WAE':
        lmba = [1, 3, 10, 20, 30, 40, 50, 75, 100]
        opts['lambda'] = lmba[FLAGS.idx_lmba-1]
    elif opts['model'] == 'disWAE':
        # Penalty
        lmba0 = [10**i for i in range(-2,3)]
        lmba1 = [10**i for i in range(-2,3)]
        lmba = list(itertools.product(lmba0,lmba1))
        opts['lambda'] = lmba[FLAGS.idx_lmba-1]
    else:
        assert False, 'unknown model {}'.format(opts['model'])
    opts['pen_enc_sigma'] = False
    opts['lambda_pen_enc_sigma'] = 0.1

    # NN set up
    opts['filter_size'] = [4,4,4,4]
    opts['mlp_init'] = 'glorot_uniform' #normal, he, glorot, glorot_he, glorot_uniform, ('uniform', range)
    opts['e_arch'] = FLAGS.enet_archi # mlp, dcgan, dcgan_v2, resnet
    opts['e_nlayers'] = 4
    opts['downsample'] = [None,]*opts['e_nlayers'] #None, True
    opts['e_nfilters'] = [32,32,64,64]
    opts['e_nonlinearity'] = 'relu' # soft_plus, relu, leaky_relu, tanh
    opts['d_arch'] =  FLAGS.enet_archi # mlp, dcgan, dcgan_v2, resnet
    opts['upsample'] = [None,]*opts['d_nlayers'] #None, up
    opts['d_nlayers'] = 4
    opts['d_nfilters'] = [32,32,32,64]
    opts['d_nonlinearity'] = 'relu' # soft_plus, relu, leaky_relu, tanh

    # Create directories
    if FLAGS.out_dir:
        opts['out_dir'] = FLAGS.out_dir
    if FLAGS.exp_dir:
        opts['exp_dir'] = FLAGS.exp_dir
    if opts['model'] == 'BetaVAE':
        exp_dir = os.path.join(opts['out_dir'],
                               opts['model'],
                               '{}_{}_{:%Y_%m_%d_%H_%M}'.format(
                                    opts['beta'],opts['exp_dir'],
                                    datetime.now()), )
    elif opts['model'] == 'WAE':
        exp_dir = os.path.join(opts['out_dir'],
                               opts['model'],
                               '{}_{}_{:%Y_%m_%d_%H_%M}'.format(
                                    opts['lambda'],
                                    opts['exp_dir'],
                                    datetime.now()), )
    elif opts['model'] == 'disWAE':
        exp_dir = os.path.join(opts['out_dir'],
                               opts['model'],
                               '{}_{}_{}_{:%Y_%m_%d_%H_%M}'.format(
                                    opts['lambda'][0],
                                    opts['lambda'][1],
                                    opts['exp_dir'],datetime.now()), )
    else:
        assert False, 'unknown model {}'.format(opts['model'])

    opts['exp_dir'] = exp_dir
    if not tf.gfile.IsDirectory(exp_dir):
        utils.create_dir(exp_dir)
        utils.create_dir(os.path.join(exp_dir, 'checkpoints'))

    # Verbose
    logging.basicConfig(filename=os.path.join(out_dir,'outputs.log'),
        level=logging.INFO, format='%(asctime)s - %(message)s')

    # Loading the dataset
    data = DataHandler(opts)
    assert data.num_points >= opts['batch_size'], 'Training set too small'

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
