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
parser.add_argument("--dataset", default='dsprites',
                    help='dataset')
parser.add_argument("--data_dir", type=str, default='../data',
                    help='directory in which data is stored')
parser.add_argument("--out_dir", type=str, default='code_outputs',
                    help='root_directory in which outputs are saved')
parser.add_argument("--res_dir", type=str, default='res',
                    help='directory in which exp. res are saved')
parser.add_argument("--num_it", type=int, default=300000,
                    help='iteration number')
parser.add_argument("--net_archi", default='mlp',
                    help='networks architecture [mlp/conv_locatello]')
parser.add_argument("--id", type=int, default=0,
                    help='exp id corresponding to latent reg weight setup')
parser.add_argument("--sigma_pen", action='store_true', default=False,
                    help='penalization of Sigma_q')
parser.add_argument("--sigma_pen_val", type=float, default=0.01,
                    help='value of penalization of Sigma_q')
parser.add_argument("--cost", default='xentropy',
                    help='ground cost [l2, l2sq, l2sq_norm, l1, xentropy]')
parser.add_argument('--fid', action='store_true', default=False,
                    help='compute FID score')
parser.add_argument('--save_model', action='store_false', default=True,
                    help='save final model weights [True/False]')
parser.add_argument("--save_data", action='store_false', default=True,
                    help='save training data')
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
    opts['fid'] = FLAGS.fid
    opts['cost'] = FLAGS.cost #l2, l2sq, l2sq_norm, l1, xentropy
    opts['network'] = net_configs[FLAGS.net_archi]
    opts['pen_enc_sigma'] = FLAGS.sigma_pen
    opts['lambda_pen_enc_sigma'] = FLAGS.sigma_pen_val

    # Model set up
    opts['model'] = FLAGS.model
    if FLAGS.dataset == 'celebA':
        opts['zdim'] = 32
        opts['batch_size'] = 128
        opts['lr'] = 0.0001
    elif FLAGS.dataset == '3Dchairs':
        opts['zdim'] = 16
        opts['batch_size'] = 128
        opts['lr'] = 0.0001
    else:
        opts['zdim'] = 10
        opts['batch_size'] = 64
        opts['lr'] = 0.0004
    if opts['model'][-3:]=='VAE':
        opts['input_normalize_sym'] = False

    # Objective Function Coefficients
    if FLAGS.dataset == 'celebA':
        if opts['model'][-3:]=='VAE':
            beta = [1, 5, 10, 15, 20, 25, 50]
            opts['obj_fn_coeffs'] = beta[FLAGS.id-1]
        else:
            beta = [1, 2, 5, 10, 15, 25, 50]
            gamma = [1, 2, 5, 10, 15, 25, 50]
            lmba = list(itertools.product(beta,gamma))
            opts['obj_fn_coeffs'] = list(lmba[FLAGS.id-1])
    elif FLAGS.dataset == '3Dchairs':
        if opts['model'][-3:]=='VAE':
            beta = [1, 2, 5, 10, 20, 50, 100]
            opts['obj_fn_coeffs'] = beta[FLAGS.id-1]
        else:
            beta = [1, 2, 5, 10, 15, 20, 25]
            gamma = [1, 2, 5, 10, 15, 20, 25]
            lmba = list(itertools.product(beta,gamma))
            opts['obj_fn_coeffs'] = list(lmba[FLAGS.id-1])
    else:
        if opts['model'] == 'BetaVAE' or model == 'BetaTCVAE':
            beta = [1, 2, 4, 6, 8, 10]
            opts['obj_fn_coeffs'] = beta[FLAGS.id-1]
        elif opts['model']=='FactorVAE':
            beta = [1, 10, 25, 50, 75, 100]
            opts['obj_fn_coeffs'] = beta[FLAGS.id-1]
        else:
            if opts['cost'] == 'xentropy':
                beta = [1, 5, 10, 25, 50, 100]
                gamma = [1, 5, 10, 25, 50, 100]
            else:
                beta = [1, 2, 4, 6, 8, 10]
                gamma = [1, 2, 4, 6, 8, 10]
            lmba = list(itertools.product(beta,gamma))
            opts['obj_fn_coeffs'] = list(lmba[FLAGS.id-1])
    # Create directories
    opts['out_dir'] = FLAGS.out_dir
    opts['exp_dir'] = FLAGS.res_dir
    if opts['model'] == 'disWAE' or opts['model'] == 'TCWAE_MWS' or opts['model'] == 'TCWAE_GAN':
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
    opts['print_every'] = int(opts['epoch_num'] / 5.) * int(data.num_points/opts['batch_size'])-1
    opts['evaluate_every'] = int(opts['print_every'] / 2.) + 1
    opts['save_every'] = 10000000000
    opts['save_final'] = FLAGS.save_model
    opts['save_train_data'] = FLAGS.save_data
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
