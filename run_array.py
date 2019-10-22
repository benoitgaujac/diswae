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
parser.add_argument("--work_dir")
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
    if FLAGS.exp == 'celebA':
        opts = configs.config_celebA
    elif FLAGS.exp == 'celebA_small':
        opts = configs.config_celebA_small
    elif FLAGS.exp == 'mnist':
        opts = configs.config_mnist
    elif FLAGS.exp == 'mnist_small':
        opts = configs.config_mnist_small
    elif FLAGS.exp == 'cifar10':
        opts = configs.config_cifar10
    elif FLAGS.exp == 'dsprites':
        opts = configs.config_dsprites
    elif FLAGS.exp == 'smallNORB':
        opts = configs.config_smallNORB
    elif FLAGS.exp == 'grassli':
        opts = configs.config_grassli
    elif FLAGS.exp == 'grassli_small':
        opts = configs.config_grassli_small
    else:
        assert False, 'Unknown experiment dataset'

    # Select training method
    if FLAGS.model:
        opts['model'] = FLAGS.model

    # Data directory
    opts['data_dir'] = FLAGS.data_dir
    # Working directory
    if FLAGS.work_dir:
        opts['work_dir'] = FLAGS.work_dir

    # Mode
    if FLAGS.mode=='fid':
        opts['fid'] = True
    else:
        opts['fid'] = False

    # Experiemnts set up
    opts['epoch_num'] = FLAGS.enum
    opts['print_every'] = 30000
    opts['save_every_epoch'] = 1000000
    opts['save_final'] = False
    opts['save_train_data'] = False

    # Model set up
    opts['zdim'] = 10

    # Objective Function Coefficients
    if opts['model'] == 'WAE':
        # Penalty
        lmba0 = [10**i for i in range(-2,3)]
        lmba1 = [10**i for i in range(-2,3)]
        lmba = list(itertools.product(lmba0,lmba1))
        opts['lambda'] = lmba[FLAGS.idx_lmba-1]
    elif opts['model'] == 'BetaVAE':
        beta = [10**i for i in range(-2,3)]
        opts['beta'] = beta[FLAGS.idx_beta-1]

    # NN set up
    opts['filter_size'] = [4,4,4,4]
    opts['mlp_init'] = 'glorot_uniform' #normal, he, glorot, glorot_he, glorot_uniform, ('uniform', range)
    opts['e_arch'] = FLAGS.enet_archi # mlp, dcgan, dcgan_v2, resnet
    opts['e_nlayers'] = 2
    opts['downsample'] = [None,]*opts['e_nlayers'] #None, True
    opts['e_nfilters'] = [1200,1200] #[32,32,64,64]
    opts['e_nonlinearity'] = 'relu' # soft_plus, relu, leaky_relu, tanh
    opts['d_arch'] =  FLAGS.enet_archi # mlp, dcgan, dcgan_v2, resnet
    opts['upsample'] = [None,]*opts['d_nlayers'] #None, up
    opts['d_nlayers'] = 3
    opts['d_nfilters'] = [1200,1200,1200] #[32,32,64,64]
    opts['d_nonlinearity'] = 'tanh' # soft_plus, relu, leaky_relu, tanh

    # Create directories
    if not tf.gfile.IsDirectory(opts['model']):
        utils.create_dir(opts['model'])
    work_dir = os.path.join(opts['model'],opts['work_dir'])
    opts['work_dir'] = work_dir
    if not tf.gfile.IsDirectory(work_dir):
        utils.create_dir(work_dir)
        utils.create_dir(os.path.join(work_dir, 'checkpoints'))

    # Verbose
    logging.basicConfig(filename=os.path.join(work_dir,'outputs.log'),
        level=logging.INFO, format='%(asctime)s - %(message)s')

    # Loading the dataset
    data = DataHandler(opts)
    assert data.num_points >= opts['batch_size'], 'Training set too small'

    #Reset tf graph
    tf.reset_default_graph()

    # build WAE/VAE
    if opts['model']=='WAE':
        model = WAE(opts)
    elif opts['model']=='BetaVAE':
        model = BetaVAE(opts)
    else:
        assert False, 'Unknown methdo %s' % opts['model']

    # Training/testing/vizu
    if FLAGS.mode=="train":
        # Dumping all the configs to the text file
        with utils.o_gfile((work_dir, 'params.txt'), 'w') as text:
            text.write('Parameters:\n')
            for key in opts:
                text.write('%s : %s\n' % (key, opts[key]))
        model.train(data, FLAGS.weights_file, model=FLAGS.model)
    # elif FLAGS.mode=="vizu":
    #     opts['rec_loss_nsamples'] = 1
    #     opts['sample_recons'] = False
    #     wae.latent_interpolation(data, opts['work_dir'], FLAGS.weights_file)
    # elif FLAGS.mode=="fid":
    #     wae.fid_score(data, opts['work_dir'], FLAGS.weights_file)
    # elif FLAGS.mode=="test":
    #     wae.test_losses(data, opts['work_dir'], FLAGS.weights_file)
    # elif FLAGS.mode=="vlae_exp":
    #     wae.vlae_experiment(data, opts['work_dir'], FLAGS.weights_file)
    else:
        assert False, 'Unknown mode %s' % FLAGS.mode

main()
