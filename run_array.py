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
parser.add_argument("--exp", default='dsprites',
                    help='dataset [mnist/cifar10/].'\
                    ' celebA/dsprites Not implemented yet')
parser.add_argument("--data_dir", type=str, default='../data',
                    help='directory in which data is stored')
parser.add_argument("--out_dir", type=str, default='code_outputs',
                    help='root_directory in which outputs are saved')
parser.add_argument("--exp_dir", type=str, default='res',
                    help='directory in which exp. outputs are saved')
parser.add_argument("--num_it", type=int, default=300000,
                    help='iteration number')
parser.add_argument("--net_archi", default='mlp',
                    help='networks architecture [mlp/conv_locatello]')
parser.add_argument("--idx", type=int, default=0,
                    help='idx latent reg weight setup')
parser.add_argument("--sigma_pen", default='False',
                    help='penalization of Sigma_q')
parser.add_argument("--sp", type=float, default=0.01,
                    help='value of penalization of Sigma_q')
parser.add_argument("--cost", default='xentropy',
                    help='ground cost [l2, l2sq, l2sq_norm, l1, xentropy]')
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
    if FLAGS.exp == 'dsprites':
        opts = configs.config_dsprites
    elif FLAGS.exp == 'smallNORB':
        opts = configs.config_smallNORB
    elif FLAGS.exp == '3dshapes':
        opts = configs.config_3dshapes
    elif FLAGS.exp == '3Dchairs':
        opts = configs.config_3Dchairs
    elif FLAGS.exp == 'celebA':
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

    # Opt set up
    opts['lr'] = 0.0001

    # Model set up
    if FLAGS.exp == 'celebA':
        opts['zdim'] = 32
    else:
        opts['zdim'] = 10
    opts['batch_size'] = 128
    opts['cost'] = FLAGS.cost #l2, l2sq, l2sq_norm, l1, xentropy
    if opts['model']!='TCWAE_MWS' and opts['model']!='TCWAE_GAN' and opts['model']!='WAE':
        opts['input_normalize_sym'] = False
    # Objective Function Coefficients
    if opts['model'] == 'BetaVAE':
        beta = [1, 2, 4, 6, 8, 10, 20]
        opts['obj_fn_coeffs'] = beta[FLAGS.idx-1]
    elif opts['model'] == 'BetaTCVAE':
        if FLAGS.exp == 'celebA':
            beta = [1, 5, 10, 15, 20, 50]
        elif FLAGS.exp == 'dsprites':
            beta = [8, 8, 8, 8, 8]
        elif FLAGS.exp == 'smallNORB':
            beta = [4, 4, 4, 4, 4]
        else:
            beta = [1, 2, 4, 6, 8, 10, 20]
        opts['obj_fn_coeffs'] = beta[FLAGS.idx-1]
    elif opts['model'] == 'FactorVAE':
        beta = [1,]
        if FLAGS.exp == 'celebA':
            gamma = [1, 5, 10, 15, 20, 50]
        elif FLAGS.exp == 'dsprites':
            gamma = [100, 100, 100, 100, 100]
        elif FLAGS.exp == 'smallNORB':
            gamma = [10, 10, 10, 10, 10]
        else:
            gamma = [1, 10, 20, 30, 40, 50, 100]
        lmba = list(itertools.product(beta,gamma))
        opts['obj_fn_coeffs'] = list(lmba[FLAGS.idx-1])
    elif opts['model'] == 'WAE':
        if opts['cost'] == 'xentropy':
            # toy experiment with xent
            if FLAGS.exp == 'dsprites':
                # lmba = [1, 10, 20, 50, 100, 150, 200]
                lmba = [200, 200, 200, 200, 200]
            elif FLAGS.exp == 'smallNORB':
                lmba = [100, 100, 100, 100, 100]
                # lmba = [1, 50, 100, 150, 200, 500, 1000]
            else:
                lmba = [1, 10, 20, 50, 100, 150, 200]
        else:
            # real word experiment with l2^2
            if FLAGS.exp == 'smallNORB':
                lmba = [1, 10, 20, 50, 100, 150, 200]
            else:
                lmba = [1, 10, 20, 50, 100, 150, 200]
        opts['obj_fn_coeffs'] = lmba[FLAGS.idx-1]
    elif opts['model'] == 'TCWAE_MWS' or opts['model'] == 'TCWAE_GAN':
        if FLAGS.exp == 'smallNORB':
            if opts['cost'] == 'xentropy':
                if opts['model'] == 'TCWAE_MWS':
                    lmba0 = [10, 10, 10]
                    lmba1 = [25, 25]
                elif opts['model'] == 'TCWAE_GAN':
                    lmba0 = [25, 25, 25]
                    lmba1 = [1, 1]
                else:
                    lmba0 = [1, 5, 10, 20, 25, 50, 100]
                    lmba1 = [1, 5, 10, 20, 25, 50, 100]
            elif opts['cost'] == 'l1':
                if opts['model'] == 'TCWAE_MWS':
                    lmba0 = [2, 2, 2]
                    lmba1 = [4, 4]
                elif opts['model'] == 'TCWAE_GAN':
                    lmba0 = [4, 4, 4]
                    lmba1 = [1, 1]
                else:
                    lmba0 = [1, 2, 4 ,6, 8, 10]
                    lmba1 = [1, 2, 4 ,6, 8, 10]
            elif opts['cost'] == 'l2':
                lmba0 = [.1, .2, .4 ,.6, .8, 1.]
                lmba1 = [.1, .2, .4 ,.6, .8, 1.]
            elif opts['cost'] == 'l2sq':
                if opts['model'] == 'TCWAE_MWS':
                    lmba0 = [4, 4, 4]
                    lmba1 = [8, 8]
                elif opts['model'] == 'TCWAE_GAN':
                    lmba0 = [4, 4, 4]
                    lmba1 = [1, 1]
                else:
                    lmba0 = [1, 2, 4 ,6, 8, 10]
                    lmba1 = [1, 2, 4 ,6, 8, 10]
            else:
                lmba0 = [1, 2, 4 ,6, 8, 10]
                lmba1 = [1, 2, 4 ,6, 8, 10]
        elif FLAGS.exp == 'dsprites':
            if opts['cost'] == 'xentropy':
                if opts['model'] == 'TCWAE_MWS':
                    lmba0 = [50, 50,]
                    lmba1 = [100, 100, 100]
                elif opts['model'] == 'TCWAE_GAN':
                    lmba0 = [100, 100,]
                    lmba1 = [5, 5, 5]
                else:
                    lmba0 = [1, 10, 20, 50, 75, 100, 150]
                    lmba1 = [1, 10, 20, 50, 75, 100, 150]
            elif opts['cost'] == 'l1':
                lmba0 = [2, 4 ,6, 8, 10]
                lmba1 = [2, 4 ,6, 8, 10]
            elif opts['cost'] == 'l2':
                lmba0 = [0.0001, 0.001, .01, .1]
                lmba1 = [0.0001, 0.001, .01, .1]
                opts['lr'] = 0.00001
            elif opts['cost'] == 'l2sq':
                lmba0 = [.1, .5, 1, 2, 4, 6]
                lmba1 = [.1, .5, 1, 2, 4, 6]
            else:
                lmba0 = [.01, .05, .1, .5, 1]
                lmba1 = [.05, .1, .5, 1, 5, 10]
        elif FLAGS.exp == '3Dchairs':
            if opts['model'] == 'TCWAE_MWS':
                lmba0 = [1, 2, 4, 6, 10, 15]
                lmba1 = [2, 4, 6, 8, 10]
            elif opts['model'] == 'TCWAE_GAN':
                lmba0 = [5, 10, 15, 20]
                lmba1 = [2, 4, 6, 8]
            else:
                lmba0 = [2, 4, 6, 8, 10]
                lmba1 = [1, 2, 4, 6]
            opts['lr'] = 0.0004
            opts['batch_size'] = 256
        elif FLAGS.exp == 'celebA':
            if opts['model'] == 'TCWAE_MWS':
                lmba0 = [1, 2, 5, 10, 15, 20]
                lmba1 = [1, 2, 5, 10, 15, 20]
            elif opts['model'] == 'TCWAE_GAN':
                lmba0 = [1, 2, 5, 10, 15, 20]
                lmba1 = [1, 2, 5, 10, 15, 20]
            else:
                lmba0 = [1, 2, 4, 6, 8, 10]
                lmba1 = [1, 2, 4, 6, 8, 10]
            opts['lr'] = 0.0001
            opts['batch_size'] = 128
        else:
            lmba0 = [1, 2, 4, 6, 8, 10]
            lmba1 = [1, 4, 8, 12, 16, 20]
            opts['batch_size'] = 256
            opts['lr'] = 0.0004
        lmba = list(itertools.product(lmba0,lmba1))
        opts['obj_fn_coeffs'] = list(lmba[FLAGS.idx-1])
    elif opts['model'] == 'disWAE':
        # Penalty
        if FLAGS.exp == 'dsprites':
            lmba0 = [1, 50, 100, 200, 300, 400, 600, 800, 900, 1000]
            lmba1 = [1, 5, 10, 20, 40, 100, 200, 400, 600, 1000]
        elif FLAGS.exp == 'smallNORB':
            lmba0 = [1, 5, 10, 20, 50, 75, 100, 250, 500, 1000]
            lmba1 = [1, 10, 50, 75, 100, 150, 200, 400, 600, 1000]
        elif FLAGS.exp == '3dshapes':
            lmba0 = [1, 50, 100, 150, 250, 500, 750, 1000, 1500, 2000]
            lmba1 = [1, 10, 20, 50, 100, 250, 500, 750, 1000, 1500]
        else :
            lmba0 = [1, 10, 50, 100, 250, 500, 750, 1000, 1500]
            lmba1 = [1, 10, 50, 100, 250, 500, 750, 1000, 1500]
        lmba = list(itertools.product(lmba0,lmba1))
        opts['obj_fn_coeffs'] = list(lmba[FLAGS.idx-1])
    else:
        assert False, 'unknown model {}'.format(opts['model'])
    # Penalty Sigma_q
    opts['pen_enc_sigma'] = FLAGS.sigma_pen=='True'
    if FLAGS.exp == 'celebA':
        opts['lambda_pen_enc_sigma'] = FLAGS.sp
    elif FLAGS.exp == 'smallNORB':
        opts['lambda_pen_enc_sigma'] = .2
    elif FLAGS.exp == '3Dchairs':
        opts['lambda_pen_enc_sigma'] = FLAGS.sp
    else:
        opts['lambda_pen_enc_sigma'] = 1.

    # NN set up
    opts['network'] = net_configs[FLAGS.net_archi]

    # Create directories
    if FLAGS.out_dir:
        opts['out_dir'] = FLAGS.out_dir
    if FLAGS.exp_dir:
        opts['exp_dir'] = FLAGS.exp_dir
    if opts['model'] == 'FactorVAE' or opts['model'] == 'disWAE' or opts['model'] == 'TCWAE_MWS' or opts['model'] == 'TCWAE_GAN':
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
