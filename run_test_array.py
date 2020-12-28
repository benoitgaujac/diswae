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
parser.add_argument("--stage_to_scratch", action='store_true', default=False,
                    help='stage data to scracht')
parser.add_argument("--scratch_dir", type=str, default='scratch0',
                    help='scratch directory in which data is staged')
parser.add_argument("--out_dir", type=str, default='code_outputs',
                    help='root_directory in which outputs are saved')
parser.add_argument("--res_dir", type=str, default='res',
                    help='directory in which exp. res are saved')
parser.add_argument("--num_it", type=int, default=300000,
                    help='iteration number')
parser.add_argument("--net_archi",
                    help='networks architecture [mlp/conv_locatello/conv_rae]')
parser.add_argument("--id", type=int, default=0,
                    help='exp id corresponding to latent reg weight setup')
parser.add_argument("--cost", default='xentropy',
                    help='ground cost [l2, l2sq, l2sq_norm, l1, xentropy]')
parser.add_argument("--gamma", type=float, default=1.0,
                    help='latent KL regularizer')
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

conv_locatello = { 'e_arch': 'conv_locatello' , 'e_nlayers': 4, 'e_nfilters': [32,32,64,64], 'e_nonlinearity': 'relu',
        'd_arch': 'conv_locatello' , 'd_nlayers': 4, 'd_nfilters': [32,32,32,64], 'd_nonlinearity': 'relu',
        'filter_size': [4,4,4,4]}
conv_rae = { 'e_arch': 'conv_rae' , 'e_nlayers': 4, 'e_nfilters': [32,64,128,256], 'e_nonlinearity': 'relu',
        'd_arch': 'conv_rae' , 'd_nlayers': 3, 'd_nfilters': [32,64,128], 'd_nonlinearity': 'relu',
        'filter_size': [4,4,4,4]}

net_configs = { 'mlp': mlp_config,
                'conv_locatello': conv_locatello,
                'conv_rae': conv_rae}


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
    opts['fid'] = FLAGS.fid
    opts['cost'] = FLAGS.cost #l2, l2sq, l2sq_norm, l1, xentropy
    if FLAGS.net_archi:
        opts['network'] = net_configs[FLAGS.net_archi]
    else:
        if FLAGS.dataset == 'celebA':
            opts['network'] = net_configs['conv_rae']
        else:
            opts['network'] = net_configs['conv_locatello']
    # Model set up
    opts['model'] = FLAGS.model
    if FLAGS.dataset == 'celebA':
        opts['zdim'] = 32
    elif FLAGS.dataset == '3Dchairs':
        opts['zdim'] = 16
    else:
        opts['zdim'] = 10
    opts['lr'] = 0.0001

    # Objective Function Coefficients
    if FLAGS.dataset == 'celebA':
        if opts['model']=='BetaTCVAE':
            beta = [1, 2, 5, 10, 15]
            coef_id = (FLAGS.id-1) % len(beta)
            opts['obj_fn_coeffs'] = beta[coef_id]
        elif opts['model']=='FactorVAE':
            beta = [1, 5, 10, 25, 50]
            coef_id = (FLAGS.id-1) % len(beta)
            opts['obj_fn_coeffs'] = beta[coef_id]
        elif opts['model']=='TCWAE_MWS':
            beta = [1, 2, 5, 10, 15]
            coef_id = (FLAGS.id-1) % len(beta)
            opts['obj_fn_coeffs'] = [beta[coef_id], FLAGS.gamma]
        elif opts['model']=='TCWAE_GAN':
            beta = [1, 5, 10, 25, 50]
            coef_id = (FLAGS.id-1) % len(beta)
            opts['obj_fn_coeffs'] = [beta[coef_id], FLAGS.gamma]
        else:
            raise Exception('Unknown {} model for celebA'.format(opts['model']))
    elif FLAGS.dataset == '3Dchairs':
        if opts['model'][-3:]=='VAE':
            beta = [1, 2, 5, 10, 20, 50]
            coef_id = (FLAGS.id-1) % len(beta)
            opts['obj_fn_coeffs'] = beta[coef_id]
        else:
            beta = [1, 2, 5, 10, 20, 50]
            coef_id = (FLAGS.id-1) % len(beta)
            opts['obj_fn_coeffs'] = [beta[coef_id], FLAGS.gamma]
    else:
        if opts['model']=='BetaTCVAE':
            beta = [1, 2, 4, 6, 8, 10]
            coef_id = (FLAGS.id-1) % len(beta)
            opts['obj_fn_coeffs'] = beta[coef_id]
        elif opts['model']=='FactorVAE':
            beta = [1, 10, 25, 50, 75, 100]
            coef_id = (FLAGS.id-1) % len(beta)
            opts['obj_fn_coeffs'] = beta[coef_id]
        elif opts['model']=='WAE':
            beta = [1, 5, 10, 25, 50, 100]
            coef_id = (FLAGS.id-1) % len(beta)
            opts['obj_fn_coeffs'] = beta[coef_id]
        elif opts['model'] == 'TCWAE_MWS':
            if opts['cost']=='xent':
                beta = [1, 2, 4, 6, 8, 10]
            else:
                beta = [0.1, 0.25, 0.5, 0.75, 1, 2]
            coef_id = (FLAGS.id-1) % len(beta)
            opts['obj_fn_coeffs'] = [beta[coef_id], FLAGS.gamma]
        elif opts['model']=='TCWAE_GAN':
            if opts['cost']=='xent':
                beta = [1, 10, 25, 50, 75, 100]
            else:
                beta = [0.1, 1, 2.5, 5, 7.5, 10]
            coef_id = (FLAGS.id-1) % len(beta)
            opts['obj_fn_coeffs'] = [beta[coef_id], FLAGS.gamma]
        else:
            raise NotImplementedError('Model type not recognised')

    # Create directories
    results_dir = 'results'
    if not tf.io.gfile.isdir(results_dir):
        utils.create_dir(results_dir)
    opts['out_dir'] = os.path.join(results_dir,FLAGS.out_dir)
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
    opts['print_every'] = int(opts['it_num'] / 2.)
    opts['evaluate_every'] = int(opts['it_num'] / 4.)
    opts['save_every'] = 10000000000
    opts['save_final'] = FLAGS.save_model
    opts['save_train_data'] = FLAGS.save_data
    opts['vizu_encSigma'] = False

    #Reset tf graph
    tf.reset_default_graph()

    # Loading the dataset
    opts['data_dir'] = FLAGS.data_dir
    opts['stage_to_scratch'] = FLAGS.stage_to_scratch
    opts['scratch_dir'] = FLAGS.scratch_dir
    data = DataHandler(opts)
    assert data.train_size >= opts['batch_size'], 'Training set too small'

    # inti method
    run = Run(opts, data)

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
