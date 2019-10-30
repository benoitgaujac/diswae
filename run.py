import os
from datetime import datetime
import logging
import argparse
import configs
from train import Run
from datahandler import DataHandler
import utils
import pdb

import tensorflow as tf

# --- Args for experiment
parser = argparse.ArgumentParser()

parser.add_argument("--model", default='WAE',
                    help='model to train [WAE/disWAE/BetaVAE/...]')
parser.add_argument("--mode", default='train',
                    help='mode to run [train/vizu/fid/test]')
parser.add_argument("--exp", default='dsprites',
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
parser.add_argument("--net_archi", default='mlp',
                    help='networks architecture [mlp/conv_locatello]')
parser.add_argument("--lmba0", type=float, default=10.,
                    help='lambda dimension wise match for WAE')
parser.add_argument("--lmba1", type=float, default=10.,
                    help='lambda hsic for WAE')
parser.add_argument("--beta", type=float, default=10.,
                    help='beta coeff for BetaVAE model')
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


# --- main
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
    opts['print_every'] = 200 #30000
    opts['save_every_epoch'] = 1000000
    opts['save_final'] = False
    opts['save_train_data'] = False

    # Model set up
    opts['zdim'] = 10

    # Objective Function Coefficients
    if opts['model'] == 'WAE':
        opts['obj_fn_coeffs'] = FLAGS.lmba0
    if opts['model'] == 'disWAE':
        opts['obj_fn_coeffs'] = [FLAGS.lmba0, FLAGS.lmba1]
    elif opts['model'] == 'BetaVAE':
        opts['obj_fn_coeffs'] = FLAGS.beta
    opts['pen_enc_sigma'] = True
    opts['lambda_pen_enc_sigma'] = 0.5

    # NN set up
    opts['network'] = net_configs[FLAGS.net_archi]

    # Create directories
    if FLAGS.out_dir:
        opts['out_dir'] = FLAGS.out_dir
    if FLAGS.exp_dir:
        opts['exp_dir'] = FLAGS.exp_dir
    exp_dir = os.path.join(opts['out_dir'],
                           opts['model'],
                           '{}_{:%Y_%m_%d_%H_%M}'.format(opts['exp_dir'],datetime.now()), )
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
