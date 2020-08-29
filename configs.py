import copy
from math import pow, sqrt

### Default common config
config = {}
# Outputs set up
config['verbose'] = False
config['save_every'] = 1000
config['save_final'] = True
config['save_train_data'] = True
config['print_every'] = 100
config['evaluate_every'] = int(config['print_every'] / 2)
config['vizu_embedded'] = False
config['embedding'] = 'umap' #vizualisation method of the embeddings: pca, umap
config['vizu_encSigma'] = False
config['vizu_interpolation'] = True
config['fid'] = False
config['out_dir'] = 'code_outputs'
config['plot_num_pics'] = 100
config['plot_num_cols'] = 10
# Experiment set up
config['train_dataset_size'] = -1
config['batch_size'] = 100
config['trbuffer_size'] = 70000
config['epoch_num'] = 100
config['model'] = 'TCWAE_MWS' #WAE, BetaVAE
config['use_trained'] = False #train from pre-trained model
# Opt set up
config['optimizer'] = 'adam' # adam, sgd
config['adam_beta1'] = 0.9
config['adam_beta2'] = 0.999
config['lr'] = 0.0001
config['lr_adv'] = 1e-08
config['normalization'] = 'none' #batchnorm, layernorm, none
config['batch_norm_eps'] = 1e-05
config['batch_norm_momentum'] = 0.99
config['dropout_rate'] = 1.
# Objective set up
config['cost'] = 'l2sq' #l2, l2sq, l2sq_norm, l1, xentropy
config['mmd_kernel'] = 'IMQ' # RBF, IMQ
config['pen_enc_sigma'] = False
config['lambda_pen_enc_sigma'] = 0.001
# Model set up
config['pz_scale'] = 1.
config['prior'] = 'gaussian' # dirichlet, gaussian
config['encoder'] = 'gauss' # deterministic, gaussian
config['decoder'] = 'det' # deterministic, gaussian
# lambda set up
config['lambda_schedule'] = 'constant' # adaptive, constant
# NN set up
config['init_std'] = 0.099999
config['init_bias'] = 0.0
config['mlp_init'] = 'glorot_uniform' #normal, he, glorot, glorot_he, glorot_uniform, ('uniform', range)
config['conv_init'] = 'glorot_uniform' #he, glorot, normilized_glorot, truncated_norm


### DSprites config
config_dsprites = config.copy()
# Data set up
config_dsprites['dataset'] = 'dsprites'
config_dsprites['DSprites_data_source_url'] = 'https://github.com/deepmind/dsprites-dataset/blob/master/'
config_dsprites['input_normalize_sym'] = False
config_dsprites['true_gen_model'] = True #If synthetic data with true gen. model known: True, False
config_dsprites['dataset_size'] = 737280
# Model set up
config_dsprites['zdim'] = 10
# lambda set up
config_dsprites['lambda'] = [100,10]


### Noisy DSprites config
config_noisydsprites = config_dsprites.copy()
config_noisydsprites['dataset'] = 'noisydsprites'


### Scream DSprites config
config_screamdsprites = config_dsprites.copy()
config_screamdsprites['dataset'] = 'screamdsprites'
# config_screamdsprites['true_gen_model'] = False #If synthetic data with true gen. model known: True, False
# config_screamdsprites['train_dataset_size'] = 1000
# config_screamdsprites['trbuffer_size'] = 500


### 3dshapes config
config_3dshapes = config.copy()
# Data set up
config_3dshapes['dataset'] = '3dshapes'
config_3dshapes['3dshapes_data_source_url'] = 'https://storage.cloud.google.com/3d-shapes/3dshapes.h5'
config_3dshapes['input_normalize_sym'] = False
config_3dshapes['true_gen_model'] = True #If synthetic data with true gen. model known: True, False
config_3dshapes['dataset_size'] = 480000
# Model set up
config_3dshapes['zdim'] = 10
# lambda set up
config_3dshapes['lambda'] = [10,1]


### smallNORB config
config_smallNORB = config.copy()
# Data set up
config_smallNORB['dataset'] = 'smallNORB'
config_smallNORB['smallNORB_data_source_url'] = 'https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/'
config_smallNORB['input_normalize_sym'] = False
config_smallNORB['true_gen_model'] = True #If synthetic data with true gen. model known: True, False
config_smallNORB['dataset_size'] = 48600
# Model set up
config_smallNORB['zdim'] = 10
# lambda set up
config_smallNORB['lambda'] = [10,1]


### 3Dchairs config
config_3Dchairs = config.copy()
# Data set up
config_3Dchairs['dataset'] = '3Dchairs'
config_3Dchairs['3Dchairs_data_source_url'] = 'https://www.di.ens.fr/willow/research/seeing3Dchairs/data/'
config_3Dchairs['input_normalize_sym'] = False
config_3Dchairs['true_gen_model'] = False #If synthetic data with true gen. model known: True, False
config_3Dchairs['dataset_size'] = 86366
# Model set up
config_3Dchairs['zdim'] = 10
# lambda set up
config_3Dchairs['lambda'] = [8,2]


### celebA config
config_celebA = config.copy()
# Data set up
config_celebA['dataset'] = 'celebA'
config_celebA['celebA_data_source_url'] = 'https://docs.google.com/uc?export=download'
config_celebA['celebA_crop'] = 'closecrop' # closecrop, resizecrop
config_celebA['input_normalize_sym'] = True
config_celebA['true_gen_model'] = False #If synthetic data with true gen. model known: True, False
config_celebA['dataset_size'] = 202599
# Model set up
config_celebA['zdim'] = 32
# lambda set up
config_celebA['lambda'] = [10,2]


### MNIST config
config_mnist = config.copy()
# Data set up
config_mnist['dataset'] = 'mnist'
config_mnist['MNIST_data_source_url'] = 'http://yann.lecun.com/exdb/mnist/'
config_mnist['input_normalize_sym'] = False
# Model set up
config_mnist['zdim'] = 8
# lambda set up
config_mnist['lambda'] = [10,10]
