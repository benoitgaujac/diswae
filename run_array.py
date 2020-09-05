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
parser.add_argument("--net_archi", default='conv_locatello',
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
    opts['fid'] = FLAGS.fid
    opts['cost'] = FLAGS.cost #l2, l2sq, l2sq_norm, l1, xentropy
    opts['network'] = net_configs[FLAGS.net_archi]
    opts['pen_enc_sigma'] = FLAGS.sigma_pen
    opts['lambda_pen_enc_sigma'] = FLAGS.sigma_pen_val

    # Model set up
    opts['model'] = FLAGS.model
    if opts['model'][-3:]=='VAE':
        opts['input_normalize_sym']=False
    if FLAGS.dataset == 'celebA':
        opts['zdim'] = 32
        opts['lr'] = 0.0001
    elif FLAGS.dataset == '3Dchairs':
        opts['zdim'] = 16
        opts['lr'] = 0.0001
    else:
        opts['zdim'] = 10
        opts['lr'] = 0.0005
    if opts['model'][-3:]=='VAE':
        opts['input_normalize_sym'] = False

    # Objective Function Coefficients
    if FLAGS.dataset == 'celebA':
        if opts['model'][-3:]=='VAE':
            beta = [1, 5, 10, 15, 20, 25, 50]
            coef_id = (FLAGS.id-1) % len(beta)
            opts['obj_fn_coeffs'] = beta[coef_id]
        else:
            beta = [1, 2, 5, 10, 15, 25, 50]
            gamma = [1, 2, 5, 10, 15, 25, 50]
            lmba = list(itertools.product(beta,gamma))
            coef_id = (FLAGS.id-1) % len(lmba)
            opts['obj_fn_coeffs'] = list(lmba[coef_id])
    elif FLAGS.dataset == '3Dchairs':
        if opts['model'][-3:]=='VAE':
            beta = [1, 2, 5, 10, 20, 50, 100]
            coef_id = (FLAGS.id-1) % len(beta)
            opts['obj_fn_coeffs'] = beta[coef_id]
        else:
            beta = [1, 2, 5, 10, 15, 20, 25]
            gamma = [1, 2, 5, 10, 15, 20, 25]
            lmba = list(itertools.product(beta,gamma))
            coef_id = (FLAGS.id-1) % len(lmba)
            opts['obj_fn_coeffs'] = list(lmba[coef_id])
    elif FLAGS.dataset == 'dsprites':
        if opts['model'] == 'BetaVAE' or opts['model'] == 'BetaTCVAE':
            beta = [1, 2, 4, 6, 8, 10]
            coef_id = (FLAGS.id-1) % len(beta)
            opts['obj_fn_coeffs'] = beta[coef_id]
        elif opts['model']=='FactorVAE':
            beta = [1, 10, 25, 50, 75, 100]
            coef_id = (FLAGS.id-1) % len(beta)
            opts['obj_fn_coeffs'] = beta[coef_id]
        elif opts['model']=='WAE':
            if opts['cost'] == 'xentropy':
                beta = [1, 5, 10, 25, 50, 100]
            else:
                beta = [0.1, 0.5, 1, 2, 4, 8]
            coef_id = (FLAGS.id-1) % len(beta)
            opts['obj_fn_coeffs'] = beta[coef_id]
        else:
            if opts['cost'] == 'xentropy':
                beta = [1, 5, 10, 25, 50, 100]
                gamma = [1, 5, 10, 25, 50, 100]
            else:
                beta = [0.1, 0.5, 1, 2, 4, 8]
                gamma = [0.1, 0.5, 1, 2, 4, 8]
            lmba = list(itertools.product(beta,gamma))
            coef_id = (FLAGS.id-1) % len(lmba)
            opts['obj_fn_coeffs'] = list(lmba[coef_id])
    else:
        if opts['model'] == 'BetaVAE' or opts['model'] == 'BetaTCVAE':
            beta = [1, 2, 4, 6, 8, 10]
            coef_id = (FLAGS.id-1) % len(beta)
            opts['obj_fn_coeffs'] = beta[coef_id]
        elif opts['model']=='FactorVAE':
            beta = [1, 10, 25, 50, 75, 100]
            coef_id = (FLAGS.id-1) % len(beta)
            opts['obj_fn_coeffs'] = beta[coef_id]
        elif opts['model']=='WAE':
            if opts['cost'] == 'xentropy':
                beta = [1, 5, 10, 25, 50, 100]
            else:
                beta = [1, 2, 4, 6, 8, 10]
            coef_id = (FLAGS.id-1) % len(beta)
            opts['obj_fn_coeffs'] = beta[coef_id]
        else:
            if opts['cost'] == 'xentropy':
                beta = [1, 5, 10, 25, 50, 100]
                gamma = [1, 5, 10, 25, 50, 100]
            else:
                beta = [1, 2, 4, 6 ,8 ,10]
                gamma = [1, 2, 4, 6 ,8 ,10]
            lmba = list(itertools.product(beta,gamma))
            coef_id = (FLAGS.id-1) % len(lmba)
            opts['obj_fn_coeffs'] = list(lmba[coef_id])
    # Create directories
    opts['out_dir'] = FLAGS.out_dir
    if not tf.io.gfile.isdir(opts['out_dir']):
        utils.create_dir(opts['out_dir'])
    out_subdir = os.path.join(opts['out_dir'], opts['model'])
    if not tf.io.gfile.isdir(out_subdir):
        utils.create_dir(out_subdir)
    opts['exp_dir'] = FLAGS.res_dir
    if opts['model'] == 'disWAE' or opts['model'] == 'TCWAE_MWS' or opts['model'] == 'TCWAE_GAN':
        exp_dir = os.path.join(out_subdir,
                               '{}_{}_{}_{:%Y_%m_%d_%H_%M}'.format(
                                    opts['exp_dir'],
                                    opts['obj_fn_coeffs'][0],
                                    opts['obj_fn_coeffs'][1],datetime.now()), )
    else :
        exp_dir = os.path.join(out_subdir,
                               '{}_{}_{:%Y_%m_%d_%H_%M}'.format(
                                    opts['exp_dir'],
                                    opts['obj_fn_coeffs'],
                                    datetime.now()), )
    opts['exp_dir'] = exp_dir
    if not tf.io.gfile.isdir(exp_dir):
        utils.create_dir(exp_dir)
        utils.create_dir(os.path.join(exp_dir, 'checkpoints'))

    # Verbose
    logging.basicConfig(filename=os.path.join(exp_dir,'outputs.log'),
        level=logging.INFO, format='%(asctime)s - %(message)s')

    # Experiemnts set up
    opts['it_num'] = FLAGS.num_it
    opts['print_every'] = int(opts['it_num'] / 5.)
    opts['evaluate_every'] = int(opts['print_every'] / 2.) + 1
    opts['save_every'] = 10000000000
    opts['save_final'] = FLAGS.save_model
    opts['save_train_data'] = FLAGS.save_data
    opts['vizu_encSigma'] = False

    #Reset tf graph
    tf.reset_default_graph()

    # Loading the dataset
    data = DataHandler(opts)
    assert data.train_size >= opts['batch_size'], 'Training set too small'

    # inti method
    run = Run(opts, data, FLAGS.weights_file)

    # Training/testing/vizu
    if FLAGS.mode=="train":
        # Dumping all the configs to the text file
        with utils.o_gfile((exp_dir, 'params.txt'), 'w') as text:
            text.write('Parameters:\n')
            for key in opts:
                text.write('%s : %s\n' % (key, opts[key]))
        run.train()
    else:
        assert False, 'Unknown mode %s' % FLAGS.mode


main()
