import copy
from math import pow, sqrt

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
config_mnist['work_dir'] = 'results_mnist'
config_mnist['plot_num_pics'] = 100
config_mnist['plot_num_cols'] = 10

# Data set up
config_mnist['dataset'] = 'mnist'
config_mnist['input_normalize_sym'] = False
config_mnist['MNIST_data_source_url'] = 'http://yann.lecun.com/exdb/mnist/'
config_mnist['Zalando_data_source_url'] = 'http://fashionChallenging Common Assumptions in the Unsupervised Learning of Disentangled Representations-mnist.s3-website.eu-central-1.amazonaws.com/'

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


config_mnist['encoder'] = 'gauss' # deterministic, gaussian
config_mnist['e_arch'] = 'mlp' # mlp, dcgan
config_mnist['e_nlayers'] = 2
config_mnist['downsample'] = [None,]*config_mnist['e_nlayers'] #None, True
config_mnist['e_nfilters'] = [512,256]
config_mnist['e_nonlinearity'] = 'leaky_relu' # soft_plus, relu, leaky_relu, tanh

config_mnist['decoder'] = 'det' # deterministic, gaussian
config_mnist['d_arch'] = 'mlp' # mlp, dcgan
config_mnist['d_nlayers'] = 2
config_mnist['upsample'] = [None,]*config_mnist['d_nlayers'] #None, up
config_mnist['d_nfilters'] = [512,256]
config_mnist['d_nonlinearity'] = 'relu' # soft_plus, relu, leaky_relu, tanh


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
config_dsprites['work_dir'] = 'results_mnist'
config_dsprites['plot_num_pics'] = 100
config_dsprites['plot_num_cols'] = 10

# Data set up
config_dsprites['dataset'] = 'dsprites'
config_dsprites['input_normalize_sym'] = False
config_dsprites['DSprites_data_source_url'] = 'https://github.com/deepmind/dsprites-dataset/blob/master/'

# Experiment set up
config_dsprites['train_dataset_size'] = -1
config_dsprites['batch_size'] = 128
config_dsprites['epoch_num'] = 101
config_dsprites['method'] = 'wae' #vae, wae
config_dsprites['use_trained'] = False #train from pre-trained model
config_dsprites['e_pretrain'] = False #pretrained the encoder parameters
config_dsprites['e_pretrain_it'] = 1000
config_dsprites['e_pretrain_sample_size'] = 200

# Opt set up
config_dsprites['optimizer'] = 'adam' # adam, sgd
config_dsprites['adam_beta1'] = 0.9
config_dsprites['adam_beta2'] = 0.999
config_dsprites['lr'] = 0.001
config_dsprites['lr_adv'] = 0.0008
config_dsprites['e_norm'] = 'batchnorm' #batchnorm, layernorm, none
config_dsprites['d_norm'] = 'batchnorm' #batchnorm, layernorm, none
config_dsprites['batch_norm_eps'] = 1e-05
config_dsprites['batch_norm_momentum'] = 0.99

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

config_dsprites['encoder'] = 'gauss' # deterministic, gaussian
config_dsprites['e_arch'] = 'conv_locatello' # mlp, dcgan
config_dsprites['e_nlayers'] = 4
config_dsprites['downsample'] = [None,]*config_dsprites['e_nlayers'] #None, True
config_dsprites['e_nfilters'] = [32,32,64,64]
config_dsprites['e_nonlinearity'] = 'leaky_relu' # soft_plus, relu, leaky_relu, tanh

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
config_smallNORB['work_dir'] = 'results_mnist'
config_smallNORB['plot_num_pics'] = 100
config_smallNORB['plot_num_cols'] = 10

# Data set up
config_smallNORB['dataset'] = 'smallNORB'
config_smallNORB['input_normalize_sym'] = False
config_smallNORB['smallNORB_data_source_url'] = 'https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/'

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
config_smallNORB['lr'] = 0.001
config_smallNORB['lr_adv'] = 0.0008
config_smallNORB['e_norm'] = 'batchnorm' #batchnorm, layernorm, none
config_smallNORB['d_norm'] = 'batchnorm' #batchnorm, layernorm, none
config_smallNORB['batch_norm_eps'] = 1e-05
config_smallNORB['batch_norm_momentum'] = 0.99

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

config_smallNORB['encoder'] = 'gauss' # deterministic, gaussian
config_smallNORB['e_arch'] = 'conv_locatello' # mlp, dcgan
config_smallNORB['e_nlayers'] = 4
config_smallNORB['downsample'] = [None,]*config_smallNORB['e_nlayers'] #None, True
config_smallNORB['e_nfilters'] = [32,32,64,64]
config_smallNORB['e_nonlinearity'] = 'leaky_relu' # soft_plus, relu, leaky_relu, tanh

config_smallNORB['decoder'] = 'det' # deterministic, gaussian
config_smallNORB['d_arch'] = 'conv_locatello' # mlp, dcgan
config_smallNORB['d_nlayers'] = 4
config_smallNORB['upsample'] = [None,]*config_smallNORB['d_nlayers'] #None, up
config_smallNORB['d_nfilters'] = [32,32,32,64]
config_smallNORB['d_nonlinearity'] = 'relu' # soft_plus, relu, leaky_relu, tanh


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
config_svhn['work_dir'] = 'results_svhn'
config_svhn['plot_num_pics'] = 100
config_svhn['plot_num_cols'] = 15

# Data set up
config_svhn['dataset'] = 'svhn'
config_svhn['input_normalize_sym'] = False
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


config_svhn['e_nlatents'] = config_svhn['nlatents'] #config_mnist['nlatents']
config_svhn['encoder'] = ['gauss','gauss','gauss','gauss','gauss','gauss','gauss','gauss'] # deterministic, gaussian
config_svhn['e_arch'] = ['dcgan','dcgan','dcgan','dcgan','dcgan','dcgan','dcgan','dcgan'] # mlp, dcgan, ali, began
config_svhn['e_nlayers'] = [2,2,2,2,2,2,2,2]
config_svhn['e_nfilters'] = [96,96,64,64,32,32,32,32]
config_svhn['e_nonlinearity'] = 'leaky_relu' # soft_plus, relu, leaky_relu, tanh


config_svhn['decoder'] = ['det','gauss','gauss','gauss','gauss','gauss','gauss','gauss'] # deterministic, gaussian
config_svhn['d_arch'] = ['dcgan','dcgan','dcgan','dcgan','dcgan','dcgan','dcgan','dcgan'] # mlp, dcgan, dcgan_mod, ali, began
config_svhn['d_nlayers'] = [2,2,2,2,2,2,2,2]
config_svhn['d_nfilters'] = [96,96,64,64,32,32,32,32]
config_svhn['d_nonlinearity'] = 'relu' # soft_plus, relu, leaky_relu, tanh


### CIFAR 10 config
config_cifar10 = {}
# Outputs set up
config_cifar10['verbose'] = False
config_cifar10['save_every'] = 2000
config_cifar10['print_every'] = 200000
config_cifar10['save_final'] = True
config_cifar10['save_train_data'] = False
config_cifar10['vizu_sinkhorn'] = False
config_cifar10['vizu_embedded'] = True
config_cifar10['embedding'] = 'umap' #vizualisation method of the embeddings: pca, umap
config_cifar10['vizu_encSigma'] = False
config_cifar10['fid'] = False
config_cifar10['work_dir'] = 'results_cifar'
config_cifar10['plot_num_pics'] = 100
config_cifar10['plot_num_cols'] = 10

# Data set up
config_cifar10['dataset'] = 'cifar10'
config_cifar10['input_normalize_sym'] = False
config_cifar10['cifar10_data_source_url'] = 'https://www.cs.toronto.edu/~kriz/'

# Experiment set up
config_cifar10['train_dataset_size'] = -1
config_cifar10['batch_size'] = 128
config_cifar10['epoch_num'] = 4120
config_cifar10['method'] = 'wae' #vae, wae
config_cifar10['use_trained'] = False #train from pre-trained model
config_cifar10['e_pretrain'] = False #pretrained the encoder parameters
config_cifar10['e_pretrain_sample_size'] = 200
config_cifar10['e_pretrain_it'] = 1000

# Opt set up
config_cifar10['optimizer'] = 'adam' # adam, sgd
config_cifar10['adam_beta1'] = 0.5
config_cifar10['lr'] = 0.0001
config_cifar10['lr_adv'] = 0.0008
config_cifar10['e_norm'] = 'batchnorm' #batchnorm, layernorm, none
config_cifar10['d_norm'] = 'layernorm' #batchnorm, layernorm, none
config_cifar10['batch_norm_eps'] = 1e-05
config_cifar10['batch_norm_momentum'] = 0.99

# Objective set up
config_cifar10['coef_rec'] = 1. # coef recon loss
config_cifar10['cost'] = 'l2sq' #l2, l2sq, l2sq_norm, l1
config_cifar10['penalty'] = 'mmd' #sinkhorn, mmd
config_cifar10['pen'] = 'wae' #wae, wae_mmd
config_cifar10['epsilon'] = 0.1 #Sinkhorn regularization parameters
config_cifar10['L'] = 30 #Sinkhorn iteration
config_cifar10['mmd_kernel'] = 'RQ' # RBF, IMQ, RQ

# Model set up
config_cifar10['nlatents'] = 8
config_cifar10['zdim'] = [64,49,36,25,16,9,4,2]
config_cifar10['pz_scale'] = 1.
config_cifar10['prior'] = 'gaussian' # dirichlet or gaussian

# lambda set up
config_cifar10['lambda_scalar'] = 10.
config_cifar10['lambda'] = [1/config_cifar10['zdim'][i] for i in range(config_cifar10['nlatents'])]
config_cifar10['lambda'].append(0.0001/config_cifar10['zdim'][-1])
config_cifar10['lambda_schedule'] = 'constant' # adaptive, constant

# NN set up
config_cifar10['init_std'] = 0.0099999
config_cifar10['init_bias'] = 0.0
config_cifar10['mlp_init'] = 'glorot_he' #normal, he, glorot, glorot_he, glorot_uniform, ('uniform', range)
config_cifar10['conv_init'] = 'he' #he, glorot, normilized_glorot, truncated_norm
config_cifar10['filter_size'] = [5,3,3,3,3,3,3,3]
config_cifar10['last_archi'] = ['conv1x1','conv1x1','conv1x1','conv1x1','conv1x1','conv1x1','dense']


config_cifar10['e_nlatents'] = config_cifar10['nlatents'] #config_mnist['nlatents']
config_cifar10['encoder'] = ['gauss','gauss','gauss','gauss','gauss','gauss','gauss','gauss'] # deterministic, gaussian
config_cifar10['e_arch'] = ['dcgan','dcgan','dcgan','dcgan','dcgan','dcgan','dcgan','dcgan'] # mlp, dcgan, ali, began
config_cifar10['e_nlayers'] = [2,2,2,2,2,2,2,2]
config_cifar10['e_nfilters'] = [96,96,64,64,32,32,32,32]
config_cifar10['e_nonlinearity'] = 'leaky_relu' # soft_plus, relu, leaky_relu, tanh


config_cifar10['decoder'] = ['det','gauss','gauss','gauss','gauss','gauss','gauss','gauss'] # deterministic, gaussian
config_cifar10['d_arch'] = ['dcgan','dcgan','dcgan','dcgan','dcgan','dcgan','dcgan','dcgan'] # mlp, dcgan, dcgan_mod, ali, began
config_cifar10['d_nlayers'] = [2,2,2,2,2,2,2,2]
config_cifar10['d_nfilters'] = [96,96,64,64,32,32,32,32]
config_cifar10['d_nonlinearity'] = 'relu' # soft_plus, relu, leaky_relu, tanh
