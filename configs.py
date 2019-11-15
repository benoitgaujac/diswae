import copy
from math import pow, sqrt


### DSprites config
config_dsprites = {}
# Outputs set up
config_dsprites['verbose'] = False
config_dsprites['save_every'] = 10000
config_dsprites['save_final'] = True
config_dsprites['save_train_data'] = True
config_dsprites['evaluate_every'] = 100
config_dsprites['plot_every'] = 100
config_dsprites['vizu_embedded'] = False
config_dsprites['embedding'] = 'umap' #vizualisation method of the embeddings: pca, umap
config_dsprites['vizu_encSigma'] = True
config_dsprites['vizu_interpolation'] = True
config_dsprites['fid'] = False
config_dsprites['out_dir'] = 'results_dsprites'
config_dsprites['plot_num_pics'] = 100
config_dsprites['plot_num_cols'] = 10
# Data set up
config_dsprites['dataset'] = 'dsprites'
config_dsprites['DSprites_data_source_url'] = 'https://github.com/deepmind/dsprites-dataset/blob/master/'
config_dsprites['input_normalize_sym'] = False
config_dsprites['num_factors'] = 5
# Experiment set up
config_dsprites['train_dataset_size'] = -1
config_dsprites['batch_size'] = 128
config_dsprites['epoch_num'] = 101
config_dsprites['model'] = 'WAE' #WAE, BetaVAE
config_dsprites['use_trained'] = False #train from pre-trained model
# Opt set up
config_dsprites['optimizer'] = 'adam' # adam, sgd
config_dsprites['adam_beta1'] = 0.9
config_dsprites['adam_beta2'] = 0.999
config_dsprites['lr'] = 0.0004
config_dsprites['lr_adv'] = 0.0008
config_dsprites['normalization'] = 'batchnorm' #batchnorm, layernorm, none
config_dsprites['batch_norm_eps'] = 1e-05
config_dsprites['batch_norm_momentum'] = 0.99
config_dsprites['dropout_rate'] = 1.
# Objective set up
config_dsprites['cost'] = 'l2sq' #l2, l2sq, l2sq_norm, l1
config_dsprites['mmd_kernel'] = 'IMQ' # RBF, IMQ
config_dsprites['pen_enc_sigma'] = True
config_dsprites['lambda_pen_enc_sigma'] = 0.001
# Model set up
config_dsprites['zdim'] = 10
config_dsprites['pz_scale'] = 1.
config_dsprites['prior'] = 'gaussian' # dirichlet, gaussian
config_dsprites['encoder'] = 'gauss' # deterministic, gaussian
config_dsprites['decoder'] = 'det' # deterministic, gaussian
# lambda set up
config_dsprites['lambda'] = [10,10]
config_dsprites['lambda_schedule'] = 'constant' # adaptive, constant
# NN set up
config_dsprites['init_std'] = 0.099999
config_dsprites['init_bias'] = 0.0
config_dsprites['mlp_init'] = 'glorot_uniform' #normal, he, glorot, glorot_he, glorot_uniform, ('uniform', range)
config_dsprites['conv_init'] = 'normilized_glorot' #he, glorot, normilized_glorot, truncated_norm


### smallNORB config
config_smallNORB = {}
# Outputs set up
config_smallNORB['verbose'] = False
config_smallNORB['save_every'] = 10000
config_smallNORB['save_final'] = True
config_smallNORB['save_train_data'] = True
config_smallNORB['print_every'] = 100
config_smallNORB['vizu_embedded'] = False
config_smallNORB['embedding'] = 'umap' #vizualisation method of the embeddings: pca, umap
config_smallNORB['vizu_encSigma'] = True
config_smallNORB['vizu_interpolation'] = True
config_smallNORB['fid'] = False
config_smallNORB['out_dir'] = 'results_mnist'
config_smallNORB['plot_num_pics'] = 100
config_smallNORB['plot_num_cols'] = 10
# Data set up
config_smallNORB['dataset'] = 'smallNORB'
config_smallNORB['smallNORB_data_source_url'] = 'https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/'
config_smallNORB['input_normalize_sym'] = False
# Experiment set up
config_smallNORB['train_dataset_size'] = -1
config_smallNORB['batch_size'] = 128
config_smallNORB['epoch_num'] = 101
config_smallNORB['method'] = 'wae' #vae, wae
config_smallNORB['use_trained'] = False #train from pre-trained model
# Opt set up
config_smallNORB['optimizer'] = 'adam' # adam, sgd
config_smallNORB['adam_beta1'] = 0.9
config_smallNORB['adam_beta2'] = 0.999
config_smallNORB['lr'] = 0.001
config_smallNORB['lr_adv'] = 0.0008
config_smallNORB['normalization'] = 'batchnorm' #batchnorm, layernorm, none
config_smallNORB['batch_norm_eps'] = 1e-05
config_smallNORB['batch_norm_momentum'] = 0.99
config_smallNORB['dropout_rate'] = 1.
# Objective set up
config_smallNORB['cost'] = 'l2sq' #l2, l2sq, l2sq_norm, l1
config_smallNORB['mmd_kernel'] = 'IMQ' # RBF, IMQ
config_smallNORB['pen_enc_sigma'] = False
config_smallNORB['lambda_pen_enc_sigma'] = 0.001
# Model set up
config_smallNORB['zdim'] = 10
config_smallNORB['pz_scale'] = 1.
config_smallNORB['prior'] = 'gaussian' # dirichlet, gaussian
config_smallNORB['encoder'] = 'gauss' # deterministic, gaussian
config_smallNORB['decoder'] = 'det' # deterministic, gaussian
# lambda set up
config_smallNORB['lambda'] = [10,10]
config_smallNORB['lambda_schedule'] = 'constant' # adaptive, constant
# NN set up
config_smallNORB['init_std'] = 0.099999
config_smallNORB['init_bias'] = 0.0
config_smallNORB['mlp_init'] = 'glorot_uniform' #normal, he, glorot, glorot_he, glorot_uniform, ('uniform', range)
config_smallNORB['conv_init'] = 'normilized_glorot' #he, glorot, normilized_glorot, truncated_norm


### 3Dchairs config
config_3Dchairs = {}
# Outputs set up
config_3Dchairs['verbose'] = False
config_3Dchairs['save_every'] = 10000
config_3Dchairs['save_final'] = True
config_3Dchairs['save_train_data'] = True
config_3Dchairs['print_every'] = 100
config_3Dchairs['vizu_embedded'] = False
config_3Dchairs['embedding'] = 'umap' #vizualisation method of the embeddings: pca, umap
config_3Dchairs['vizu_encSigma'] = True
config_3Dchairs['vizu_interpolation'] = True
config_3Dchairs['fid'] = False
config_3Dchairs['out_dir'] = 'results_mnist'
config_3Dchairs['plot_num_pics'] = 100
config_3Dchairs['plot_num_cols'] = 10
# Data set up
config_3Dchairs['dataset'] = '3Dchairs'
config_3Dchairs['3Dchairs_data_source_url'] = 'https://www.di.ens.fr/willow/research/seeing3Dchairs/data/'
config_3Dchairs['input_normalize_sym'] = False
# Experiment set up
config_3Dchairs['train_dataset_size'] = -1
config_3Dchairs['batch_size'] = 128
config_3Dchairs['epoch_num'] = 101
config_3Dchairs['method'] = 'wae' #vae, wae
config_3Dchairs['use_trained'] = False #train from pre-trained model
# Opt set up
config_3Dchairs['optimizer'] = 'adam' # adam, sgd
config_3Dchairs['adam_beta1'] = 0.9
config_3Dchairs['adam_beta2'] = 0.999
config_3Dchairs['lr'] = 0.0001
config_3Dchairs['lr_adv'] = 0.0008
config_3Dchairs['normalization'] = 'batchnorm' #batchnorm, layernorm, none
config_3Dchairs['batch_norm_eps'] = 1e-05
config_3Dchairs['batch_norm_momentum'] = 0.99
config_3Dchairs['dropout_rate'] = 1.
# Objective set up
config_3Dchairs['cost'] = 'l2sq' #l2, l2sq, l2sq_norm, l1
config_3Dchairs['mmd_kernel'] = 'IMQ' # RBF, IMQ
config_3Dchairs['pen_enc_sigma'] = False
config_3Dchairs['lambda_pen_enc_sigma'] = 0.001
# Model set up
config_3Dchairs['zdim'] = 10
config_3Dchairs['pz_scale'] = 1.
config_3Dchairs['prior'] = 'gaussian' # dirichlet, gaussian
config_3Dchairs['encoder'] = 'gauss' # deterministic, gaussian
config_3Dchairs['decoder'] = 'det' # deterministic, gaussian
# lambda set up
config_3Dchairs['lambda'] = [10,10]
config_3Dchairs['lambda_schedule'] = 'constant' # adaptive, constant
# NN set up
config_3Dchairs['init_std'] = 0.099999
config_3Dchairs['init_bias'] = 0.0
config_3Dchairs['mlp_init'] = 'glorot_uniform' #normal, he, glorot, glorot_he, glorot_uniform, ('uniform', range)
config_3Dchairs['conv_init'] = 'normilized_glorot' #he, glorot, normilized_glorot, truncated_norm


### celebA config
config_celebA = {}
# Outputs set up
config_celebA['verbose'] = False
config_celebA['save_every'] = 10000
config_celebA['save_final'] = True
config_celebA['save_train_data'] = True
config_celebA['print_every'] = 100
config_celebA['vizu_embedded'] = False
config_celebA['embedding'] = 'umap' #vizualisation method of the embeddings: pca, umap
config_celebA['vizu_encSigma'] = False
config_celebA['vizu_interpolation'] = True
config_celebA['fid'] = False
config_celebA['out_dir'] = 'results_mnist'
config_celebA['plot_num_pics'] = 100
config_celebA['plot_num_cols'] = 10
# Data set up
config_celebA['dataset'] = 'celebA'
config_celebA['celebA_data_source_url'] = 'https://docs.google.com/uc?export=download'
config_celebA['celebA_crop'] = 'closecrop' # closecrop, resizecrop
config_celebA['input_normalize_sym'] = True
# Experiment set up
config_celebA['train_dataset_size'] = -1
config_celebA['batch_size'] = 128
config_celebA['epoch_num'] = 101
config_celebA['method'] = 'wae' #vae, wae
config_celebA['use_trained'] = False #train from pre-trained model
# Opt set up
config_celebA['optimizer'] = 'adam' # adam, sgd
config_celebA['adam_beta1'] = 0.9
config_celebA['adam_beta2'] = 0.999
config_celebA['lr'] = 0.0001
config_celebA['lr_adv'] = 0.0008
config_celebA['normalization'] = 'batchnorm' #batchnorm, layernorm, none
config_celebA['batch_norm_eps'] = 1e-05
config_celebA['batch_norm_momentum'] = 0.99
config_celebA['dropout_rate'] = 1.
# Objective set up
config_celebA['cost'] = 'l2sq' #l2, l2sq, l2sq_norm, l1
config_celebA['mmd_kernel'] = 'IMQ' # RBF, IMQ
config_celebA['pen_enc_sigma'] = False
config_celebA['lambda_pen_enc_sigma'] = 0.001
# Model set up
config_celebA['zdim'] = 10
config_celebA['pz_scale'] = 1.
config_celebA['prior'] = 'gaussian' # dirichlet, gaussian
config_celebA['encoder'] = 'gauss' # deterministic, gaussian
config_celebA['decoder'] = 'det' # deterministic, gaussian
# lambda set up
config_celebA['lambda'] = [10,10]
config_celebA['lambda_schedule'] = 'constant' # adaptive, constant
# NN set up
config_celebA['init_std'] = 0.099999
config_celebA['init_bias'] = 0.0
config_celebA['mlp_init'] = 'glorot_uniform' #normal, he, glorot, glorot_he, glorot_uniform, ('uniform', range)
config_celebA['conv_init'] = 'normilized_glorot' #he, glorot, normilized_glorot, truncated_norm


### MNIST config
config_mnist = {}
# Outputs set up
config_mnist['verbose'] = False
config_mnist['save_every'] = 10000
config_mnist['save_final'] = True
config_mnist['save_train_data'] = True
config_mnist['print_every'] = 100
config_mnist['vizu_embedded'] = False
config_mnist['embedding'] = 'umap' #vizualisation method of the embeddings: pca, umap
config_mnist['vizu_encSigma'] = False
config_mnist['vizu_interpolation'] = True
config_mnist['fid'] = False
config_mnist['out_dir'] = 'results_mnist'
config_mnist['plot_num_pics'] = 100
config_mnist['plot_num_cols'] = 10
# Data set up
config_mnist['dataset'] = 'mnist'
config_mnist['MNIST_data_source_url'] = 'http://yann.lecun.com/exdb/mnist/'
config_mnist['input_normalize_sym'] = False
# Experiment set up
config_mnist['train_dataset_size'] = -1
config_mnist['batch_size'] = 128
config_mnist['epoch_num'] = 101
config_mnist['method'] = 'wae' #vae, wae
config_mnist['use_trained'] = False #train from pre-trained model
# Opt set up
config_mnist['optimizer'] = 'adam' # adam, sgd
config_mnist['adam_beta1'] = 0.5
config_mnist['lr'] = 0.001
config_mnist['lr_adv'] = 0.0008
config_mnist['normalization'] = 'batchnorm' #batchnorm, layernorm, none
config_mnist['batch_norm_eps'] = 1e-05
config_mnist['batch_norm_momentum'] = 0.99
config_mnist['dropout_rate'] = 1.
# Objective set up
config_mnist['cost'] = 'l2sq' #l2, l2sq, l2sq_norm, l1
config_mnist['mmd_kernel'] = 'IMQ' # RBF, IMQ
config_mnist['pen_enc_sigma'] = False
config_mnist['lambda_pen_enc_sigma'] = 0.001
# Model set up
config_mnist['zdim'] = 8
config_mnist['pz_scale'] = 1.
config_mnist['prior'] = 'gaussian' # dirichlet, gaussian
config_mnist['encoder'] = 'gauss' # deterministic, gaussian
config_mnist['decoder'] = 'det' # deterministic, gaussian
# lambda set up
config_mnist['lambda'] = [10,10]
config_mnist['lambda_schedule'] = 'constant' # adaptive, constant
# NN set up
config_mnist['init_std'] = 0.099999
config_mnist['init_bias'] = 0.0
config_mnist['mlp_init'] = 'glorot_uniform' #normal, he, glorot, glorot_he, glorot_uniform, ('uniform', range)
config_mnist['conv_init'] = 'he' #he, glorot, normilized_glorot, truncated_norm


### CIFAR 10 config
config_svhn = {}
# Outputs set up
config_svhn['verbose'] = False
config_svhn['save_every'] = 2000
config_svhn['print_every'] = 200000
config_svhn['save_final'] = True
config_svhn['save_train_data'] = False
config_svhn['vizu_sinkhorn'] = False
config_svhn['vizu_embedded'] = True
config_svhn['embedding'] = 'umap' #vizualisation method of the embeddings: pca, umap
config_svhn['vizu_encSigma'] = False
config_svhn['fid'] = False
config_svhn['out_dir'] = 'results_svhn'
config_svhn['plot_num_pics'] = 100
config_svhn['plot_num_cols'] = 15
# Data set up
config_svhn['dataset'] = 'svhn'
config_svhn['SVHN_data_source_url'] = 'http://ufldl.stanford.edu/housenumbers/'
config_svhn['input_normalize_sym'] = False
# Experiment set up
config_svhn['train_dataset_size'] = -1
config_svhn['use_extra'] = False
config_svhn['batch_size'] = 128
config_svhn['epoch_num'] = 4120
config_svhn['method'] = 'wae' #vae, wae
config_svhn['use_trained'] = False #train from pre-trained model
# Opt set up
config_svhn['optimizer'] = 'adam' # adam, sgd
config_svhn['adam_beta1'] = 0.5
config_svhn['lr'] = 0.0002
config_svhn['lr_adv'] = 0.0008
config_svhn['normalization'] = 'batchnorm' #batchnorm, layernorm, none
config_svhn['batch_norm_eps'] = 1e-05
config_svhn['batch_norm_momentum'] = 0.99
config_svhn['dropout_rate'] = 1.
# Objective set up
config_svhn['coef_rec'] = 1. # coef recon loss
config_svhn['cost'] = 'l2sq' #l2, l2sq, l2sq_norm, l1
config_svhn['mmd_kernel'] = 'IMQ' # RBF, IMQ
# Model set up
config_svhn['nlatents'] = 8
config_svhn['zdim'] = [64,49,36,25,16,9,4,2]
config_svhn['pz_scale'] = 1.
config_svhn['prior'] = 'gaussian' # dirichlet or gaussian
config_svhn['encoder'] = ['gauss','gauss','gauss','gauss','gauss','gauss','gauss','gauss'] # deterministic, gaussian
config_svhn['decoder'] = ['det','gauss','gauss','gauss','gauss','gauss','gauss','gauss'] # deterministic, gaussian
# lambda set up
config_svhn['lambda_scalar'] = 10.
config_svhn['lambda'] = [1/config_svhn['zdim'][i] for i in range(config_svhn['nlatents'])]
config_svhn['lambda'].append(0.0001/config_svhn['zdim'][-1])
config_svhn['lambda_schedule'] = 'constant' # adaptive, constant
# NN set up
config_svhn['init_std'] = 0.099999
config_svhn['init_bias'] = 0.0
config_svhn['mlp_init'] = 'glorot_he' #normal, he, glorot, glorot_he, glorot_uniform, ('uniform', range)
config_svhn['conv_init'] = 'he' #he, glorot, normilized_glorot, truncated_norm
config_svhn['e_nlatents'] = config_svhn['nlatents'] #config_mnist['nlatents']
