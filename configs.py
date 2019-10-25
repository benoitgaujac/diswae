import copy
from math import pow, sqrt


### DSprites config
config_dsprites = {}
# Outputs set up
config_dsprites['verbose'] = False
config_dsprites['save_every'] = 10000
config_dsprites['save_final'] = True
config_dsprites['save_train_data'] = True
config_dsprites['print_every'] = 100
config_dsprites['vizu_embedded'] = False
config_dsprites['embedding'] = 'umap' #vizualisation method of the embeddings: pca, umap
config_dsprites['vizu_encSigma'] = False
config_dsprites['vizu_interpolation'] = True
config_dsprites['fid'] = False
config_dsprites['out_dir'] = 'results_mnist'
config_dsprites['plot_num_pics'] = 100
config_dsprites['plot_num_cols'] = 10
# Data set up
config_dsprites['dataset'] = 'dsprites'
config_dsprites['DSprites_data_source_url'] = 'https://github.com/deepmind/dsprites-dataset/blob/master/'
config_dsprites['input_normalize_sym'] = False
# Experiment set up
config_dsprites['train_dataset_size'] = -1
config_dsprites['batch_size'] = 128
config_dsprites['epoch_num'] = 101
config_dsprites['model'] = 'WAE' #WAE, BetaVAE
config_dsprites['use_trained'] = False #train from pre-trained model
config_dsprites['e_pretrain'] = False #pretrained the encoder parameters
config_dsprites['e_pretrain_it'] = 1000
config_dsprites['e_pretrain_sample_size'] = 200
# Opt set up
config_dsprites['optimizer'] = 'adam' # adam, sgd
config_dsprites['adam_beta1'] = 0.9
config_dsprites['adam_beta2'] = 0.999
config_dsprites['lr'] = 0.0001
config_dsprites['lr_adv'] = 0.0008
config_dsprites['e_norm'] = 'batchnorm' #batchnorm, layernorm, none
config_dsprites['d_norm'] = 'batchnorm' #batchnorm, layernorm, none
config_dsprites['batch_norm_eps'] = 1e-05
config_dsprites['batch_norm_momentum'] = 0.99
config_dsprites['dropout_rate'] = 1.
# Objective set up
config_dsprites['cost'] = 'l2sq' #l2, l2sq, l2sq_norm, l1
config_dsprites['penalty'] = 'mmd' #sinkhorn, mmd
config_dsprites['pen'] = 'wae' #wae, wae_mmd
config_dsprites['epsilon'] = 0.1 #Sinkhorn regularization parameters
config_dsprites['L'] = 30 #Sinkhorn iteration
config_dsprites['mmd_kernel'] = 'IMQ' # RBF, IMQ
config_dsprites['pen'] = 'wae' # wae, wae_mmd
config_dsprites['pen_enc_sigma'] = False
config_dsprites['lambda_pen_enc_sigma'] = 0.001
config_dsprites['pen_dec_sigma'] = False
config_dsprites['lambda_pen_dec_sigma'] = 0.0005
# Model set up
config_dsprites['zdim'] = 10
config_dsprites['pz_scale'] = 1.
config_dsprites['prior'] = 'gaussian' # dirichlet, gaussian
# lambda set up
config_dsprites['lambda'] = [10,10]
config_dsprites['lambda_schedule'] = 'constant' # adaptive, constant
# NN set up
config_dsprites['init_std'] = 0.99999
config_dsprites['init_bias'] = 0.0
config_dsprites['mlp_init'] = 'glorot_uniform' #normal, he, glorot, glorot_he, glorot_uniform, ('uniform', range)
config_dsprites['conv_init'] = 'normilized_glorot' #he, glorot, normilized_glorot, truncated_norm
config_dsprites['filter_size'] = [4,4,4,4]
# encoder
config_dsprites['encoder'] = 'gauss' # deterministic, gaussian
config_dsprites['e_arch'] = 'conv_locatello' # mlp, dcgan
config_dsprites['e_nlayers'] = 4
config_dsprites['downsample'] = [None,]*config_dsprites['e_nlayers'] #None, True
config_dsprites['e_nfilters'] = [32,32,64,64]
config_dsprites['e_nonlinearity'] = 'leaky_relu' # soft_plus, relu, leaky_relu, tanh
# decoder
config_dsprites['decoder'] = 'det' # deterministic, gaussian
config_dsprites['d_arch'] = 'conv_locatello' # mlp, dcgan
config_dsprites['d_nlayers'] = 4
config_dsprites['upsample'] = [None,]*config_dsprites['d_nlayers'] #None, up
config_dsprites['d_nfilters'] = [32,32,32,64]
config_dsprites['d_nonlinearity'] = 'relu' # soft_plus, relu, leaky_relu, tanh


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
config_smallNORB['vizu_encSigma'] = False
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
config_smallNORB['e_pretrain'] = False #pretrained the encoder parameters
config_smallNORB['e_pretrain_it'] = 1000
config_smallNORB['e_pretrain_sample_size'] = 200
# Opt set up
config_smallNORB['optimizer'] = 'adam' # adam, sgd
config_smallNORB['adam_beta1'] = 0.9
config_smallNORB['adam_beta2'] = 0.999
config_smallNORB['lr'] = 0.0001
config_smallNORB['lr_adv'] = 0.0008
config_smallNORB['e_norm'] = 'batchnorm' #batchnorm, layernorm, none
config_smallNORB['d_norm'] = 'batchnorm' #batchnorm, layernorm, none
config_smallNORB['batch_norm_eps'] = 1e-05
config_smallNORB['batch_norm_momentum'] = 0.99
config_smallNORB['dropout_rate'] = 1.
# Objective set up
config_smallNORB['cost'] = 'l2sq' #l2, l2sq, l2sq_norm, l1
config_smallNORB['penalty'] = 'mmd' #sinkhorn, mmd
config_smallNORB['pen'] = 'wae' #wae, wae_mmd
config_smallNORB['epsilon'] = 0.1 #Sinkhorn regularization parameters
config_smallNORB['L'] = 30 #Sinkhorn iteration
config_smallNORB['mmd_kernel'] = 'IMQ' # RBF, IMQ
config_smallNORB['pen'] = 'wae' # wae, wae_mmd
config_smallNORB['pen_enc_sigma'] = False
config_smallNORB['lambda_pen_enc_sigma'] = 0.001
config_smallNORB['pen_dec_sigma'] = False
config_smallNORB['lambda_pen_dec_sigma'] = 0.0005
# Model set up
config_smallNORB['zdim'] = 10
config_smallNORB['pz_scale'] = 1.
config_smallNORB['prior'] = 'gaussian' # dirichlet, gaussian
# lambda set up
config_smallNORB['lambda'] = [10,10]
config_smallNORB['lambda_schedule'] = 'constant' # adaptive, constant
# NN set up
config_smallNORB['init_std'] = 0.99999
config_smallNORB['init_bias'] = 0.0
config_smallNORB['mlp_init'] = 'glorot_uniform' #normal, he, glorot, glorot_he, glorot_uniform, ('uniform', range)
config_smallNORB['conv_init'] = 'normilized_glorot' #he, glorot, normilized_glorot, truncated_norm
config_smallNORB['filter_size'] = [4,4,4,4]
# encoder
config_smallNORB['encoder'] = 'gauss' # deterministic, gaussian
config_smallNORB['e_arch'] = 'conv_locatello' # mlp, dcgan
config_smallNORB['e_nlayers'] = 4
config_smallNORB['downsample'] = [None,]*config_smallNORB['e_nlayers'] #None, True
config_smallNORB['e_nfilters'] = [32,32,64,64]
config_smallNORB['e_nonlinearity'] = 'leaky_relu' # soft_plus, relu, leaky_relu, tanh
# decoder
config_smallNORB['decoder'] = 'det' # deterministic, gaussian
config_smallNORB['d_arch'] = 'conv_locatello' # mlp, dcgan
config_smallNORB['d_nlayers'] = 4
config_smallNORB['upsample'] = [None,]*config_smallNORB['d_nlayers'] #None, up
config_smallNORB['d_nfilters'] = [32,32,32,64]
config_smallNORB['d_nonlinearity'] = 'relu' # soft_plus, relu, leaky_relu, tanh


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
config_3Dchairs['vizu_encSigma'] = False
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
config_3Dchairs['e_pretrain'] = False #pretrained the encoder parameters
config_3Dchairs['e_pretrain_it'] = 1000
config_3Dchairs['e_pretrain_sample_size'] = 200
# Opt set up
config_3Dchairs['optimizer'] = 'adam' # adam, sgd
config_3Dchairs['adam_beta1'] = 0.9
config_3Dchairs['adam_beta2'] = 0.999
config_3Dchairs['lr'] = 0.0001
config_3Dchairs['lr_adv'] = 0.0008
config_3Dchairs['e_norm'] = 'batchnorm' #batchnorm, layernorm, none
config_3Dchairs['d_norm'] = 'batchnorm' #batchnorm, layernorm, none
config_3Dchairs['batch_norm_eps'] = 1e-05
config_3Dchairs['batch_norm_momentum'] = 0.99
config_3Dchairs['dropout_rate'] = 1.
# Objective set up
config_3Dchairs['cost'] = 'l2sq' #l2, l2sq, l2sq_norm, l1
config_3Dchairs['penalty'] = 'mmd' #sinkhorn, mmd
config_3Dchairs['pen'] = 'wae' #wae, wae_mmd
config_3Dchairs['epsilon'] = 0.1 #Sinkhorn regularization parameters
config_3Dchairs['L'] = 30 #Sinkhorn iteration
config_3Dchairs['mmd_kernel'] = 'IMQ' # RBF, IMQ
config_3Dchairs['pen'] = 'wae' # wae, wae_mmd
config_3Dchairs['pen_enc_sigma'] = False
config_3Dchairs['lambda_pen_enc_sigma'] = 0.001
config_3Dchairs['pen_dec_sigma'] = False
config_3Dchairs['lambda_pen_dec_sigma'] = 0.0005
# Model set up
config_3Dchairs['zdim'] = 10
config_3Dchairs['pz_scale'] = 1.
config_3Dchairs['prior'] = 'gaussian' # dirichlet, gaussian
# lambda set up
config_3Dchairs['lambda'] = [10,10]
config_3Dchairs['lambda_schedule'] = 'constant' # adaptive, constant
# NN set up
config_3Dchairs['init_std'] = 0.99999
config_3Dchairs['init_bias'] = 0.0
config_3Dchairs['mlp_init'] = 'glorot_uniform' #normal, he, glorot, glorot_he, glorot_uniform, ('uniform', range)
config_3Dchairs['conv_init'] = 'normilized_glorot' #he, glorot, normilized_glorot, truncated_norm
config_3Dchairs['filter_size'] = [4,4,4,4]
# encoder
config_3Dchairs['encoder'] = 'gauss' # deterministic, gaussian
config_3Dchairs['e_arch'] = 'conv_locatello' # mlp, dcgan
config_3Dchairs['e_nlayers'] = 4
config_3Dchairs['downsample'] = [None,]*config_3Dchairs['e_nlayers'] #None, True
config_3Dchairs['e_nfilters'] = [32,32,64,64]
config_3Dchairs['e_nonlinearity'] = 'leaky_relu' # soft_plus, relu, leaky_relu, tanh
# decoder
config_3Dchairs['decoder'] = 'det' # deterministic, gaussian
config_3Dchairs['d_arch'] = 'conv_locatello' # mlp, dcgan
config_3Dchairs['d_nlayers'] = 4
config_3Dchairs['upsample'] = [None,]*config_3Dchairs['d_nlayers'] #None, up
config_3Dchairs['d_nfilters'] = [32,32,32,64]
config_3Dchairs['d_nonlinearity'] = 'relu' # soft_plus, relu, leaky_relu, tanh


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
config_celebA['e_pretrain'] = False #pretrained the encoder parameters
config_celebA['e_pretrain_it'] = 1000
config_celebA['e_pretrain_sample_size'] = 200
# Opt set up
config_celebA['optimizer'] = 'adam' # adam, sgd
config_celebA['adam_beta1'] = 0.9
config_celebA['adam_beta2'] = 0.999
config_celebA['lr'] = 0.0001
config_celebA['lr_adv'] = 0.0008
config_celebA['e_norm'] = 'batchnorm' #batchnorm, layernorm, none
config_celebA['d_norm'] = 'batchnorm' #batchnorm, layernorm, none
config_celebA['batch_norm_eps'] = 1e-05
config_celebA['batch_norm_momentum'] = 0.99
config_celebA['dropout_rate'] = 1.
# Objective set up
config_celebA['cost'] = 'l2sq' #l2, l2sq, l2sq_norm, l1
config_celebA['penalty'] = 'mmd' #sinkhorn, mmd
config_celebA['pen'] = 'wae' #wae, wae_mmd
config_celebA['epsilon'] = 0.1 #Sinkhorn regularization parameters
config_celebA['L'] = 30 #Sinkhorn iteration
config_celebA['mmd_kernel'] = 'IMQ' # RBF, IMQ
config_celebA['pen'] = 'wae' # wae, wae_mmd
config_celebA['pen_enc_sigma'] = False
config_celebA['lambda_pen_enc_sigma'] = 0.001
config_celebA['pen_dec_sigma'] = False
config_celebA['lambda_pen_dec_sigma'] = 0.0005
# Model set up
config_celebA['zdim'] = 10
config_celebA['pz_scale'] = 1.
config_celebA['prior'] = 'gaussian' # dirichlet, gaussian
# lambda set up
config_celebA['lambda'] = [10,10]
config_celebA['lambda_schedule'] = 'constant' # adaptive, constant
# NN set up
config_celebA['init_std'] = 0.99999
config_celebA['init_bias'] = 0.0
config_celebA['mlp_init'] = 'glorot_uniform' #normal, he, glorot, glorot_he, glorot_uniform, ('uniform', range)
config_celebA['conv_init'] = 'normilized_glorot' #he, glorot, normilized_glorot, truncated_norm
config_celebA['filter_size'] = [4,4,4,4]
# encoder
config_celebA['encoder'] = 'gauss' # deterministic, gaussian
config_celebA['e_arch'] = 'conv_locatello' # mlp, dcgan
config_celebA['e_nlayers'] = 4
config_celebA['downsample'] = [None,]*config_celebA['e_nlayers'] #None, True
config_celebA['e_nfilters'] = [32,32,64,64]
config_celebA['e_nonlinearity'] = 'leaky_relu' # soft_plus, relu, leaky_relu, tanh
# decoder
config_celebA['decoder'] = 'det' # deterministic, gaussian
config_celebA['d_arch'] = 'conv_locatello' # mlp, dcgan
config_celebA['d_nlayers'] = 4
config_celebA['upsample'] = [None,]*config_celebA['d_nlayers'] #None, up
config_celebA['d_nfilters'] = [32,32,32,64]
config_celebA['d_nonlinearity'] = 'relu' # soft_plus, relu, leaky_relu, tanh


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
# Experiment set up
config_mnist['train_dataset_size'] = -1
config_mnist['batch_size'] = 128
config_mnist['epoch_num'] = 101
config_mnist['method'] = 'wae' #vae, wae
config_mnist['use_trained'] = False #train from pre-trained model
config_mnist['e_pretrain'] = False #pretrained the encoder parameters
config_mnist['e_pretrain_it'] = 1000
config_mnist['e_pretrain_sample_size'] = 200
# Opt set up
config_mnist['optimizer'] = 'adam' # adam, sgd
config_mnist['adam_beta1'] = 0.5
config_mnist['lr'] = 0.001
config_mnist['lr_adv'] = 0.0008
config_mnist['e_norm'] = 'batchnorm' #batchnorm, layernorm, none
config_mnist['d_norm'] = 'batchnorm' #batchnorm, layernorm, none
config_mnist['batch_norm_eps'] = 1e-05
config_mnist['batch_norm_momentum'] = 0.99
config_mnist['dropout_rate'] = 1.
# Objective set up
config_mnist['cost'] = 'l2sq' #l2, l2sq, l2sq_norm, l1
config_mnist['penalty'] = 'mmd' #sinkhorn, mmd
config_mnist['pen'] = 'wae' #wae, wae_mmd
config_mnist['epsilon'] = 0.1 #Sinkhorn regularization parameters
config_mnist['L'] = 30 #Sinkhorn iteration
config_mnist['mmd_kernel'] = 'IMQ' # RBF, IMQ
config_mnist['pen'] = 'wae' # wae, wae_mmd
config_mnist['pen_enc_sigma'] = False
config_mnist['lambda_pen_enc_sigma'] = 0.001
config_mnist['pen_dec_sigma'] = False
config_mnist['lambda_pen_dec_sigma'] = 0.0005
# Model set up
config_mnist['zdim'] = 8
config_mnist['pz_scale'] = 1.
config_mnist['prior'] = 'gaussian' # dirichlet, gaussian
# lambda set up
config_mnist['lambda'] = [10,10]
config_mnist['lambda_schedule'] = 'constant' # adaptive, constant
# NN set up
config_mnist['init_std'] = 0.99999
config_mnist['init_bias'] = 0.0
config_mnist['mlp_init'] = 'glorot_uniform' #normal, he, glorot, glorot_he, glorot_uniform, ('uniform', range)
config_mnist['conv_init'] = 'he' #he, glorot, normilized_glorot, truncated_norm
config_mnist['filter_size'] = [5,3]
# encoder
config_mnist['encoder'] = 'gauss' # deterministic, gaussian
config_mnist['e_arch'] = 'mlp' # mlp, dcgan
config_mnist['e_nlayers'] = 2
config_mnist['downsample'] = [None,]*config_mnist['e_nlayers'] #None, True
config_mnist['e_nfilters'] = [512,256]
config_mnist['e_nonlinearity'] = 'leaky_relu' # soft_plus, relu, leaky_relu, tanh
# decoder
config_mnist['decoder'] = 'det' # deterministic, gaussian
config_mnist['d_arch'] = 'mlp' # mlp, dcgan
config_mnist['d_nlayers'] = 2
config_mnist['upsample'] = [None,]*config_mnist['d_nlayers'] #None, up
config_mnist['d_nfilters'] = [512,256]
config_mnist['d_nonlinearity'] = 'relu' # soft_plus, relu, leaky_relu, tanh


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
# Experiment set up
config_svhn['train_dataset_size'] = -1
config_svhn['use_extra'] = False
config_svhn['batch_size'] = 128
config_svhn['epoch_num'] = 4120
config_svhn['method'] = 'wae' #vae, wae
config_svhn['use_trained'] = False #train from pre-trained model
config_svhn['e_pretrain'] = False #pretrained the encoder parameters
config_svhn['e_pretrain_sample_size'] = 200
config_svhn['e_pretrain_it'] = 1000
# Opt set up
config_svhn['optimizer'] = 'adam' # adam, sgd
config_svhn['adam_beta1'] = 0.5
config_svhn['lr'] = 0.0002
config_svhn['lr_adv'] = 0.0008
config_svhn['e_norm'] = 'batchnorm' #batchnorm, layernorm, none
config_svhn['d_norm'] = 'layernorm' #batchnorm, layernorm, none
config_svhn['batch_norm_eps'] = 1e-05
config_svhn['batch_norm_momentum'] = 0.99
config_svhn['dropout_rate'] = 1.
# Objective set up
config_svhn['coef_rec'] = 1. # coef recon loss
config_svhn['cost'] = 'l2sq' #l2, l2sq, l2sq_norm, l1
config_svhn['penalty'] = 'mmd' #sinkhorn, mmd
config_svhn['pen'] = 'wae' #wae, wae_mmd
config_svhn['epsilon'] = 0.1 #Sinkhorn regularization parameters
config_svhn['L'] = 30 #Sinkhorn iteration
config_svhn['mmd_kernel'] = 'IMQ' # RBF, IMQ
# Model set up
config_svhn['nlatents'] = 8
config_svhn['zdim'] = [64,49,36,25,16,9,4,2]
config_svhn['pz_scale'] = 1.
config_svhn['prior'] = 'gaussian' # dirichlet or gaussian
# lambda set up
config_svhn['lambda_scalar'] = 10.
config_svhn['lambda'] = [1/config_svhn['zdim'][i] for i in range(config_svhn['nlatents'])]
config_svhn['lambda'].append(0.0001/config_svhn['zdim'][-1])
config_svhn['lambda_schedule'] = 'constant' # adaptive, constant
# NN set up
config_svhn['init_std'] = 0.0099999
config_svhn['init_bias'] = 0.0
config_svhn['mlp_init'] = 'glorot_he' #normal, he, glorot, glorot_he, glorot_uniform, ('uniform', range)
config_svhn['conv_init'] = 'he' #he, glorot, normilized_glorot, truncated_norm
config_svhn['filter_size'] = [5,3,3,3,3,3,3,3]
config_svhn['last_archi'] = ['conv1x1','conv1x1','conv1x1','conv1x1','conv1x1','conv1x1','dense']
# encoder
config_svhn['e_nlatents'] = config_svhn['nlatents'] #config_mnist['nlatents']
config_svhn['encoder'] = ['gauss','gauss','gauss','gauss','gauss','gauss','gauss','gauss'] # deterministic, gaussian
config_svhn['e_arch'] = ['dcgan','dcgan','dcgan','dcgan','dcgan','dcgan','dcgan','dcgan'] # mlp, dcgan, ali, began
config_svhn['e_nlayers'] = [2,2,2,2,2,2,2,2]
config_svhn['e_nfilters'] = [96,96,64,64,32,32,32,32]
config_svhn['e_nonlinearity'] = 'leaky_relu' # soft_plus, relu, leaky_relu, tanh
# decoder
config_svhn['decoder'] = ['det','gauss','gauss','gauss','gauss','gauss','gauss','gauss'] # deterministic, gaussian
config_svhn['d_arch'] = ['dcgan','dcgan','dcgan','dcgan','dcgan','dcgan','dcgan','dcgan'] # mlp, dcgan, dcgan_mod, ali, began
config_svhn['d_nlayers'] = [2,2,2,2,2,2,2,2]
config_svhn['d_nfilters'] = [96,96,64,64,32,32,32,32]
config_svhn['d_nonlinearity'] = 'relu' # soft_plus, relu, leaky_relu, tanh
