"""
Auto-Encoder models
"""
import os
import sys
import logging

import numpy as np
import tensorflow as tf
from sklearn import linear_model

import utils
from sampling_functions import sample_pz, linespace
from plot_functions import save_train, save_test_smallnorb, save_test_celeba, save_dimwise_traversals
from plot_functions import plot_embedded, plot_encSigma, plot_interpolation
import models
from datahandler import datashapes

# Path to inception model and stats for training set
sys.path.append('../TTUR')
sys.path.append('../inception')
import fid
inception_path = '../inception'
inception_model = os.path.join(inception_path, 'classify_image_graph_def.pb')
layername = 'FID_Inception_Net/pool_3:0'


import pdb

class Run(object):

    def __init__(self, opts):

        logging.error('Building the Tensorflow Graph')

        # --- Create session
        # self.sess = tf.Session()
        self.opts = opts

        # --- Data shape
        assert opts['dataset'] in datashapes, 'Unknown dataset.'
        self.data_shape = datashapes[opts['dataset']]

        # --- Placeholders
        self.add_model_placeholders()
        self.add_training_placeholders()

        # --- Initialize prior parameters
        mean = np.zeros(opts['zdim'], dtype='float32')
        Sigma = np.ones(opts['zdim'], dtype='float32')
        self.pz_params = np.concatenate([mean, Sigma], axis=0)

        # --- Instantiate Model
        if opts['model'] == 'BetaVAE':
            self.model = models.BetaVAE(opts)
            self.obj_fn_coeffs = self.beta
        elif opts['model'] == 'BetaTCVAE':
            self.model = models.BetaTCVAE(opts)
            self.obj_fn_coeffs = self.beta
        elif opts['model'] == 'FactorVAE':
            self.model = models.FactorVAE(opts)
            self.obj_fn_coeffs = self.beta
        elif opts['model'] == 'WAE':
            self.model = models.WAE(opts)
            self.obj_fn_coeffs = self.lmbd
        elif opts['model'] == 'TCWAE_MWS':
            self.model = models.TCWAE_MWS(opts)
            self.obj_fn_coeffs = (self.lmbd1, self.lmbd2)
        elif opts['model'] == 'TCWAE_GAN':
            self.model = models.TCWAE_GAN(opts)
            self.obj_fn_coeffs = (self.lmbd1, self.lmbd2)
        elif opts['model'] == 'disWAE':
            self.model = models.disWAE(opts)
            self.obj_fn_coeffs = (self.lmbd1, self.lmbd2)
        else:
            raise NotImplementedError()

        # --- Define Objective
        self.objective, self.loss_reconstruct, self.divergences, self.enc_sigmastats \
            = self.model.loss(inputs=self.batch,
                              samples=self.samples_pz,
                              loss_coeffs=self.obj_fn_coeffs,
                              is_training=self.is_training)

        self.z_samples, self.z_mean, self.z_logvar, self.recon_x, _, _ = self.model.forward_pass(inputs=self.batch,
                                                                          is_training=self.is_training,
                                                                          reuse=True)

        self.generated_x = self.model.sample_x_from_prior(noise=self.samples_pz)

        self.kl_to_prior = self.model.dimewise_kl_to_prior(self.z_samples,
                                                    self.z_mean,
                                                    self.z_logvar)

        self.mse = self.model.MSE(self.batch, self.recon_x)

        # FID score
        if opts['fid']:
            self.blurriness = self.compute_blurriness()
            self.inception_graph = tf.Graph()
            self.inception_sess = tf.Session(graph=self.inception_graph)
            with self.inception_graph.as_default():
                self.create_inception_graph()
            self.inception_layer = self._get_inception_layer()

        # --- Optimizers, savers, etc
        self.add_optimizers()
        self.add_savers()
        self.initializer = tf.global_variables_initializer()

    def add_model_placeholders(self):
        opts = self.opts
        shape = self.data_shape
        self.batch = tf.placeholder(tf.float32,
                                        [None] + shape,
                                        name='points_ph')
        self.samples_pz = tf.placeholder(tf.float32,
                                         [None] + [opts['zdim'],],
                                         name='noise_ph')

    def add_training_placeholders(self):
        self.lr_decay = tf.placeholder(tf.float32, name='rate_decay_ph')
        self.is_training = tf.placeholder(tf.bool, name='is_training_ph')
        self.batch_size = tf.placeholder(tf.int32, name='batch_size_ph')
        if self.opts['model']=='BetaVAE' or self.opts['model'] == 'BetaTCVAE' or self.opts['model'] == 'FactorVAE':
            self.beta = tf.placeholder(tf.float32, name='beta_ph')
        elif self.opts['model']=='WAE':
            self.lmbd = tf.placeholder(tf.float32, name='lambda_ph')
        else:
            self.lmbd1 = tf.placeholder(tf.float32, name='lambda1_ph')
            self.lmbd2 = tf.placeholder(tf.float32, name='lambda2_ph')

    def compute_blurriness(self):
        images = self.batch
        # First convert to greyscale
        if self.data_shape[-1] > 1:
            # We have RGB
            images = tf.image.rgb_to_grayscale(images)
        # Next convolve with the Laplace filter
        lap_filter = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        lap_filter = lap_filter.reshape([3, 3, 1, 1])
        conv = tf.nn.conv2d(images, lap_filter, strides=[1, 1, 1, 1],
                                                padding='VALID')
        _, lapvar = tf.nn.moments(conv, axes=[1, 2, 3])
        return lapvar

    def create_inception_graph(self):
        # Create inception graph
        with tf.gfile.FastGFile( inception_model, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString( f.read())
            _ = tf.import_graph_def( graph_def, name='FID_Inception_Net')

    def _get_inception_layer(self):
        # Get inception activation layer (and reshape for batching)
        pool3 = self.inception_sess.graph.get_tensor_by_name(layername)
        ops_pool3 = pool3.graph.get_operations()
        for op_idx, op in enumerate(ops_pool3):
            for o in op.outputs:
                shape = o.get_shape()
                if shape._dims != []:
                  shape = [s.value for s in shape]
                  new_shape = []
                  for j, s in enumerate(shape):
                    if s == 1 and j == 0:
                      new_shape.append(None)
                    else:
                      new_shape.append(s)
                  o.__dict__['_shape_val'] = tf.TensorShape(new_shape)
        return pool3

    def compute_mig(self,z_mean,labels):
        """MIG metric.
        Compute the discrete mutual information between
        mean latent codes and factors as in ICML 2019"""
        opts = self.opts
        # Discretize enc_mean
        discretized_z_mean = utils.discretizer(np.transpose(z_mean), 20)
        # mutual discrete information
        mutual_info = utils.discrete_mutual_info(discretized_z_mean,np.transpose(labels))
        # laten entropy
        entropy = utils.discrete_entropy(np.transpose(labels))
        # mig metric
        assert mutual_info.shape[0] == discretized_z_mean.shape[0]
        assert mutual_info.shape[1] == labels.shape[1]
        sorted_mutual_info = np.sort(mutual_info, axis=0)[::-1]
        mig = (sorted_mutual_info[0, :] - sorted_mutual_info[1, :]) / entropy

        return np.mean(mig)

    def generate_factorVAE_minibatch(self, data, global_variances, active_dims):
        opts = self.opts
        batch_size = 64
        # sample batch of factors
        factors = utils.sample_factors(batch_size, data.factor_sizes)
        # sample factor idx
        factor_index = np.random.randint(len(data.factor_indices))
        factor_index = data.factor_indices[factor_index]
        # fixing the selected factor across batch
        factors[:, factor_index] = factors[0, factor_index]
        # sample batch of images with fix selected factor
        # batch_images = utils.sample_images(opts, opts['dataset'], data, factors)
        batch_images = data.sample_observations_from_factors(opts['dataset'], factors)
        # encode images
        z = self.sess.run(self.z_samples, feed_dict={self.batch: batch_images,
                                                self.is_training: False})
        # get variance per dimension and vote
        local_variances = np.var(z, axis=0, ddof=1)
        argmin = np.argmin(local_variances[active_dims] / global_variances[active_dims])

        return factor_index, argmin

    def compute_factorVAE(self, data, codes):
        """Compute FactorVAE metric"""
        opts = self.opts
        threshold = .05
        # Compute global variance and pruning dimensions
        global_variances = np.var(codes, axis=0, ddof=1)
        active_dims = np.sqrt(global_variances)>=threshold
        # Generate classifier training set and build classifier
        training_size = 10000
        # training_size = 100
        votes = np.zeros((len(data.factor_sizes), opts['zdim']),dtype=np.int32)
        for i in range(training_size):
            factor, vote = self.generate_factorVAE_minibatch(data,
                                                    global_variances,
                                                    active_dims)
            votes[factor, vote] += 1
            # print('{} training points generated'.format(i+1))
        classifier = np.argmax(votes, axis=0)
        other_index = np.arange(votes.shape[1])
        # Generate classifier eval set and get eval accuracy
        eval_size = 5000
        # eval_size = 50
        votes = np.zeros((len(data.factor_sizes), opts['zdim']),dtype=np.int32)
        for i in range(eval_size):
            factor, vote = self.generate_factorVAE_minibatch(data,
                                                    global_variances,
                                                    active_dims)
            votes[factor, vote] += 1
            # print('{} eval points generated'.format(i+1))
        acc = np.sum(votes[classifier, other_index]) * 1. / np.sum(votes)

        return acc

    def generate_SAP_minibatch(self, data, num_points):
        opts = self.opts
        batch_size = 500
        # batch_size = 50
        representations = None
        factors = None
        i = 0
        while i < num_points:
            num_points_iter = min(num_points - i, batch_size)
            # sample batch of factors
            current_factors = utils.sample_factors(num_points_iter, data.factor_sizes)
            # sample batch of images from factors
            # batch_images = utils.sample_images(opts, opts['dataset'], data, current_factors)
            batch_images = data.sample_observations_from_factors(opts['dataset'], current_factors)
            # encode images
            current_z = self.sess.run(self.z_samples, feed_dict={
                                                self.batch: batch_images,
                                                self.is_training: False})
            if i == 0:
                factors = current_factors
                z = current_z
            else:
                factors = np.vstack((factors, current_factors))
                z = np.vstack((z,current_z))
            i += num_points_iter

        return z, factors

    def compute_SAP(self, data):
        """Compute SAP metric"""
        opts = self.opts
        # Generate training set
        training_size = 10000
        # training_size = 100
        mus, ys = self.generate_SAP_minibatch(data, training_size)
        # Generate testing set
        testing_size = 5000
        # testing_size = 50
        mus_test, ys_test = self.generate_SAP_minibatch(data, testing_size)
        # Computing score matrix
        score_matrix = utils.compute_score_matrix(mus, ys, mus_test, ys_test)
        # average diff top 2 predictive latent dim for each factor
        sorted_score_matric = np.sort(score_matrix, axis=0)
        sap = np.mean(sorted_score_matric[-1, :] - sorted_score_matric[-2, :])

        return sap

    def generate_betaVAE_minibatch(self, data):
        opts = self.opts
        batch_size = 64
        # sample 2 batches of factors
        factors_1 = utils.sample_factors(batch_size, data.factor_sizes)
        factors_2 = utils.sample_factors(batch_size, data.factor_sizes)
        # sample factor idx
        factor_index = np.random.randint(len(data.factor_indices))
        factor_index = data.factor_indices[factor_index]
        # fixing the selected factor across batch
        factors_1[:, factor_index] = factors_2[:, factor_index]
        # sample images with fix selected factor
        # images_1 = utils.sample_images(opts, opts['dataset'], data, factors_1)
        # images_2 = utils.sample_images(opts, opts['dataset'], data, factors_2)
        images_1 = data.sample_observations_from_factors(opts['dataset'], factors_1)
        images_2 = data.sample_observations_from_factors(opts['dataset'], factors_2)
        # encode images
        z_1 = self.sess.run(self.z_samples, feed_dict={self.batch: images_1,
                                                self.is_training: False})
        z_2 = self.sess.run(self.z_samples, feed_dict={self.batch: images_2,
                                                self.is_training: False})
        # Compute the feature vector based on differences in representation.
        feature_vector = np.mean(np.abs(z_1 - z_2), axis=0)

        return feature_vector, factor_index

    def compute_betaVAE(self, data):
        """Compute betaVAE metric"""
        opts = self.opts
        # Generate classifier training set and build classifier
        training_size = 10000
        # training_size = 100
        x_train = np.zeros((training_size,opts['zdim']))
        y_train = np.zeros((training_size,))
        for i in range(training_size):
            x_train[i], y_train[i] = self.generate_betaVAE_minibatch(data)
        # logging.info("Training sklearn model.")
        model = linear_model.LogisticRegression()
        model.fit(x_train, y_train)
        # Generate classifier eval set and get eval accuracy
        eval_size = 5000
        # eval_size = 50
        x_eval = np.zeros((eval_size,opts['zdim']))
        y_eval = np.zeros((eval_size,))
        for i in range(eval_size):
            x_eval[i], y_eval[i] = self.generate_betaVAE_minibatch(sess, data)
        acc = model.score(x_eval, y_eval)

        return acc

    def add_savers(self):
        saver = tf.train.Saver(max_to_keep=10)
        self.saver = saver

    def optimizer(self, lr, decay=1.):
        opts = self.opts
        lr *= decay
        if opts['optimizer'] == 'sgd':
            return tf.train.GradientDescentOptimizer(lr)
        elif opts['optimizer'] == 'adam':
            return tf.train.AdamOptimizer(lr, beta1=opts['adam_beta1'], beta2=opts['adam_beta2'])
        else:
            assert False, 'Unknown optimizer.'

    def discr_optimizer(self, lr=0.0001):
        return tf.train.AdamOptimizer(lr, beta1=0.5, beta2=0.9,)

    def add_optimizers(self):
        opts = self.opts
        lr = opts['lr']
        opt = self.optimizer(lr, self.lr_decay)
        encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                scope='encoder')
        decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                scope='decoder')
        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # discriminator opt if needed
        if self.opts['model']=='FactorVAE' or self.opts['model']=='TCWAE_GAN':
            if opts['dataset']=='celebA' or opts['dataset']=='3Dchairs':
                discr_opt = self.discr_optimizer(0.00001)
            else:
                discr_opt = self.discr_optimizer()
            discr_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                    scope='discriminator')
            vae_opt = opt.minimize(loss=self.objective,var_list=encoder_vars + decoder_vars)
            discriminator_opt = discr_opt.minimize(loss=-self.model.discr_loss,var_list=discr_vars)
            # self.opt = tf.group(vae_opt, discriminator_opt, update_ops)
            self.opt = tf.group(vae_opt, discriminator_opt)
        else:
            self.opt = opt.minimize(loss=self.objective,var_list=encoder_vars + decoder_vars)
            # with tf.control_dependencies(update_ops):
            #     self.opt = opt.minimize(loss=self.objective,var_list=encoder_vars + decoder_vars)

    def train(self, data, WEIGHTS_FILE):
        """
        Train top-down model with chosen method
        """
        opts = self.opts
        logging.error('Training {}'.format(self.opts['model']))
        exp_dir = opts['exp_dir']

        # writer = tf.summary.FileWriter(exp_dir)

        # - Init sess and load trained weights if needed
        self.sess = tf.Session()
        if opts['use_trained']:
            if not tf.gfile.Exists(WEIGHTS_FILE+".meta"):
                raise Exception("weights file doesn't exist")
            self.saver.restore(self.sess, WEIGHTS_FILE)
        else:
            self.sess.run(self.initializer)
        self.sess.graph.finalize()

        # - Set up for training
        train_size = data.train_size
        batches_num = int(train_size/opts['batch_size'])
        logging.error('Train size: {}, Batch num.: {}, Epoch num: {}'.format(train_size, batches_num, opts['epoch_num']))
        npics = opts['plot_num_pics']
        im_shape = datashapes[opts['dataset']]
        fixed_noise = sample_pz(opts, self.pz_params, npics)
        # anchors_ids = np.random.choice(npics, 5, replace=True)
        anchors_ids = [0, 4, 6, 12, 39]

        # - Init all monitoring variables
        Loss, Loss_test, Loss_rec, Loss_rec_test = [], [], [], []
        Divergences, Divergences_test = [], []
        betaVAE, MIG, factorVAE, SAP = [], [], [], []
        MSE = []
        if opts['vizu_encSigma']:
            enc_Sigmas = []

        decay, counter = 1., 0
        # decay_steps, decay_rate = int(batches_num * opts['epoch_num'] / 5), 0.95
        decay_steps, decay_rate = 500000, 0.95
        wait, wait_lambda = 0, 0
        for epoch in range(opts['epoch_num']):
            # Saver
            if counter > 0 and counter % opts['save_every'] == 0:
                self.saver.save(self.sess,
                                os.path.join(exp_dir, 'checkpoints', 'trained-wae'),
                                global_step=counter)

            #####  TRAINING LOOP #####
            for it in range(batches_num):
                # Sample batches of data points and Pz noise
                data_ids = np.random.choice(train_size, opts['batch_size'], replace=True)
                batch_images = data.get_batch_img(data_ids,'train').astype(np.float32)
                batch_pz_samples = sample_pz(opts, self.pz_params, opts['batch_size'])
                # Feeding dictionary
                feed_dict = {self.batch: batch_images,
                             self.samples_pz: batch_pz_samples,
                             self.lr_decay: decay,
                             self.obj_fn_coeffs: opts['obj_fn_coeffs'],
                             self.is_training: True}
                _ = self.sess.run(self.opt, feed_dict=feed_dict)

                ##### TESTING LOOP #####
                if (counter+1)%opts['evaluate_every'] == 0 or (counter+1) == 100:
                    print("Epoch {}, Iteration {}".format(epoch, it+1))
                    # Train losses
                    data_ids = np.random.choice(train_size, opts['batch_size'], replace=True)
                    batch_images = data.get_batch_img(data_ids,'train').astype(np.float32)
                    batch_pz_samples = sample_pz(opts, self.pz_params, opts['batch_size'])
                    # Feeding dictionary
                    feed_dict = {self.batch: batch_images,
                                 self.samples_pz: batch_pz_samples,
                                 self.lr_decay: decay,
                                 self.obj_fn_coeffs: opts['obj_fn_coeffs'],
                                 self.is_training: True}
                    [loss, loss_rec, divergences, enc_sigmastats] = self.sess.run([
                                                self.objective,
                                                self.loss_reconstruct,
                                                self.divergences,
                                                self.enc_sigmastats],
                                                feed_dict=feed_dict)
                    Loss.append(loss)
                    Loss_rec.append(loss_rec)
                    Divergences.append(divergences)

                    # Encoded Sigma
                    if opts['vizu_encSigma']:
                        enc_Sigmas.append(enc_sigmastats)

                    # Test losses
                    test_size = data.test_size
                    batch_size_te = min(test_size,500)
                    batches_num_te = int(test_size/batch_size_te)+1
                    loss_test, loss_rec_test, mse = 0., 0., 0.
                    if opts['true_gen_model']:
                        codes, codes_mean = np.zeros((batches_num_te*batch_size_te,opts['zdim'])), np.zeros((batches_num_te*batch_size_te,opts['zdim']))
                        labels = np.zeros((batches_num_te*batch_size_te,len(data.factor_indices)))
                    if type(divergences)==list:
                        divergences_test = np.zeros(len(divergences))
                    else:
                        divergences_test = 0.
                    kl_to_prior = np.zeros(opts['zdim'])
                    for it_ in range(batches_num_te):
                        # Sample batches of data points
                        data_ids = np.random.choice(test_size, batch_size_te, replace=True)
                        batch_images_test = data.get_batch_img(data_ids, 'test').astype(np.float32)
                        batch_pz_samples_test = sample_pz(opts, self.pz_params, batch_size_te)
                        test_feed_dict = {self.batch: batch_images_test,
                                          self.samples_pz: batch_pz_samples_test,
                                          self.obj_fn_coeffs: opts['obj_fn_coeffs'],
                                          self.is_training: False}
                        [loss, l_rec, m, divergences, z, z_mean, kl] = self.sess.run([self.objective,
                                                         self.loss_reconstruct,
                                                         self.mse,
                                                         self.divergences,
                                                         self.z_samples,
                                                         self.z_mean,
                                                         self.kl_to_prior],
                                                        feed_dict=test_feed_dict)
                        loss_test += loss / batches_num_te
                        loss_rec_test += l_rec / batches_num_te
                        mse += m / batches_num_te
                        divergences_test += np.array(divergences) / batches_num_te
                        kl_to_prior += kl / batches_num_te
                        if opts['true_gen_model']:
                            codes[batch_size_te*it_:batch_size_te*(it_+1)] = z
                            codes_mean[batch_size_te*it_:batch_size_te*(it_+1)] = z_mean
                            labels[batch_size_te*it_:batch_size_te*(it_+1)] = data.get_batch_label(data_ids,'test')[:,data.factor_indices]
                    Loss_test.append(loss_test)
                    Loss_rec_test.append(loss_rec_test)
                    MSE.append(mse)
                    Divergences_test.append(divergences_test.tolist())
                    if opts['true_gen_model']:
                        # betaVAE.append(self.compute_betaVAE(data))
                        MIG.append(self.compute_mig(codes_mean, labels))
                        factorVAE.append(self.compute_factorVAE(data, codes))
                        SAP.append(self.compute_SAP(data))
                        betaVAE.append(0.)

                if (counter+1)%opts['print_every'] == 0 or (counter+1) == 100:
                    # Plot vizualizations
                    # Auto-encoding test images & samples generated by the model
                    [reconstructions_test, latents_test, generations] = self.sess.run(
                                                [self.recon_x,
                                                 self.z_samples,
                                                 self.generated_x],
                                                feed_dict={self.batch: data.get_batch_img(range(opts['plot_num_pics']), 'vizu'),
                                                           self.samples_pz: fixed_noise,
                                                           self.is_training: False})
                    # Auto-encoding training images
                    reconstructions_train = self.sess.run(self.recon_x,
                                                          feed_dict={self.batch: data.get_batch_img(range(200,200+npics), 'train'),
                                                                     self.is_training: False})

                    # - Plotting embeddings, Sigma, latent interpolation, and saving
                    # Embeddings
                    if opts['vizu_embedded'] and counter > 1:
                        plot_embedded(opts, [latents_test[:npics]], [fixed_noise],
                                      data.get_batch_label(range(npics), 'vizu'),
                                      exp_dir, 'embedded_e%04d_mb%05d.png' % (epoch, it))
                    # Encoded sigma
                    if opts['vizu_encSigma'] and counter > 1:
                        plot_encSigma(opts,
                                      enc_Sigmas,
                                      exp_dir,
                                      'encSigma_e%04d_mb%05d.png' % (epoch, it))
                    # Encode anchors points and interpolate
                    if opts['vizu_interpolation']:
                        num_steps = 15

                        enc_var = np.ones(opts['zdim'])
                        # crate linespace
                        enc_interpolation = linespace(opts, num_steps,  # shape: [nanchors, zdim, nsteps, zdim]
                                                      anchors=latents_test[anchors_ids],
                                                      std=enc_var)
                        # reconstructing
                        dec_interpolation = self.sess.run(self.generated_x,
                                                          feed_dict={self.samples_pz: np.reshape(enc_interpolation,
                                                                                                 [-1, opts['zdim']]),
                                                                     self.is_training: False})
                        inter_anchors = np.reshape(dec_interpolation, [-1, opts['zdim'], num_steps]+im_shape)
                        kl_to_prior_sorted = np.argsort(kl_to_prior)[::-1]
                        plot_interpolation(opts, inter_anchors[:,kl_to_prior_sorted], exp_dir,
                                           'inter_e%04d_mb%05d.png' % (epoch, it))

                    # Saving plots
                    save_train(opts,
                              data.get_batch_img(range(200,200+npics), 'train'),    # train images
                              data.get_batch_img(range(npics), 'vizu'),             # test images
                              reconstructions_train, reconstructions_test,          # reconstructions
                              generations,                                          # model samples
                              Loss, Loss_test,                                      # loss
                              Loss_rec, Loss_rec_test,                              # rec loss
                              betaVAE, MIG, factorVAE, SAP,                         # disentangle metrics
                              Divergences, Divergences_test,                        # divergence terms
                              exp_dir,                                              # working directory
                              'res_e%04d_mb%05d.png' % (epoch, it))                 # filename


                    # Printing various loss values
                    debug_str = '\n EPOCH: %d/%d, BATCH:%d/%d' % (epoch,
                                                               opts['epoch_num'],
                                                               it + 1,
                                                               batches_num)
                    logging.error(debug_str)
                    debug_str = 'TRAIN LOSS=%.3f, TEST LOSS=%.3f' % (Loss[-1],Loss_test[-1])
                    logging.error(debug_str)
                    if opts['true_gen_model']:
                        debug_str = 'betaVAE=%.3f, MIG=%.3f, factorVAE=%.3f, SAP=%.3f' % (
                                                    betaVAE[-1],
                                                    MIG[-1],
                                                    factorVAE[-1],
                                                    SAP[-1])
                        logging.error(debug_str)
                    if opts['model'] == 'BetaVAE':
                        debug_str = 'REC=%.3f, TEST REC=%.3f, beta*KL=%10.3e, beta*TEST KL=%10.3e, \n '  % (
                                                    Loss_rec[-1],
                                                    Loss_rec_test[-1],
                                                    Divergences[-1],
                                                    Divergences_test[-1])
                        logging.error(debug_str)
                    elif opts['model'] == 'BetaTCVAE':
                        debug_str = 'REC=%.3f, TEST REC=%.3f, b*TC=%10.3e, TEST b*TC=%10.3e, KL=%10.3e, TEST KL=%10.3e, \n '  % (
                                                    Loss_rec[-1],
                                                    Loss_rec_test[-1],
                                                    Divergences[-1][0],
                                                    Divergences_test[-1][0],
                                                    Divergences[-1][1],
                                                    Divergences_test[-1][1])
                        logging.error(debug_str)
                    elif opts['model'] == 'FactorVAE':
                        debug_str = 'REC=%.3f, TEST REC=%.3f, g*TC=%10.3e, TEST g*TC=%10.3e, b*KL=%10.3e, TEST KL=%10.3e, \n '  % (
                                                    Loss_rec[-1],
                                                    Loss_rec_test[-1],
                                                    Divergences[-1][1],
                                                    Divergences_test[-1][1],
                                                    Divergences[-1][0],
                                                    Divergences_test[-1][0])
                        logging.error(debug_str)
                    elif opts['model'] == 'WAE':
                        debug_str = 'REC=%.3f, TEST REC=%.3f, l*MMD=%10.3e, l*TEST MMD=%10.3e \n ' % (
                                                    Loss_rec[-1],
                                                    Loss_rec_test[-1],
                                                    Divergences[-1],
                                                    Divergences_test[-1])
                        logging.error(debug_str)
                    elif opts['model'] == 'disWAE':
                        debug_str = 'TRAIN: REC=%.3f,l1*HSIC=%10.3e, l2*DIMWISE=%10.3e, WAE=%10.3e' % (
                                                    Loss_rec[-1],
                                                    Divergences[-1][0],
                                                    Divergences[-1][1],
                                                    Divergences[-1][2])
                        logging.error(debug_str)
                        debug_str = 'TEST : REC=%.3f, l1*HSIC=%10.3e, l2*DIMWISE=%10.3e, WAE=%10.3e \n ' % (
                                                    Loss_rec_test[-1],
                                                    Divergences_test[-1][0],
                                                    Divergences_test[-1][1],
                                                    Divergences_test[-1][2])
                        logging.error(debug_str)
                    elif opts['model'] == 'TCWAE_MWS' or opts['model'] == 'TCWAE_GAN':
                        debug_str = 'TRAIN: REC=%.3f,l1*TC=%10.3e, l2*DIMWISE=%10.3e, WAE=%10.3e' % (
                                                    Loss_rec[-1],
                                                    Divergences[-1][0],
                                                    Divergences[-1][1],
                                                    Divergences[-1][2])
                        logging.error(debug_str)
                        debug_str = 'TEST : REC=%.3f, l1*TC=%10.3e, l2*DIMWISE=%10.3e, WAE=%10.3e \n ' % (
                                                    Loss_rec_test[-1],
                                                    Divergences_test[-1][0],
                                                    Divergences_test[-1][1],
                                                    Divergences_test[-1][2])
                        logging.error(debug_str)
                    else:
                        raise NotImplementedError('Model type not recognised')

                # - Update learning rate if necessary and counter
                # if False:
                if opts['dataset']=='celebA' or opts['dataset']=='3Dchairs':
                    if (counter+1) % decay_steps == 0:
                        decay = decay_rate ** (int(counter / decay_steps))
                        logging.error('Reduction in lr: %f\n' % decay)
                        """
                        # If no significant progress was made in last 20 epochs
                        # then decrease the learning rate.
                        if np.mean(Loss_rec[-20:]) < np.mean(Loss_rec[-20 * batches_num:])-1.*np.var(Loss_rec[-20 * batches_num:]):
                            wait = 0
                        else:
                            wait += 1
                        if wait > 20 * batches_num:
                            decay = max(decay  / 1.33, 1e-6)
                            logging.error('Reduction in lr: %f\n' % decay)
                            print('')
                            wait = 0
                        """

                # - Update regularizer if necessary
                if opts['lambda_schedule'] == 'adaptive':
                    if epoch >= .0 and len(Loss_rec) > 0:
                        if wait_lambda > 1000 * batches_num + 1:
                            # opts['lambda'] = list(2*np.array(opts['lambda']))
                            opts['lambda'][-1] = 2*opts['lambda'][-1]
                            wae_lambda = opts['lambda']
                            logging.error('Lambda updated to %s\n' % wae_lambda)
                            print('')
                            wait_lambda = 0
                        else:
                            wait_lambda += 1

                # logging
                if (counter+1)%50000 ==0 :
                    logging.error('Train it.: {}/{} \n'.format(counter+1,opts['epoch_num']*batches_num))

                counter += 1

        # - Finale losses & scores
        # Feeding dictionary
        feed_dict = {self.batch: batch_images,
                     self.samples_pz: batch_pz_samples,
                     self.lr_decay: decay,
                     self.obj_fn_coeffs: opts['obj_fn_coeffs'],
                     self.is_training: True}
        [loss, loss_rec, divergences] = self.sess.run([
                                    self.objective,
                                    self.loss_reconstruct,
                                    self.divergences],
                                    feed_dict=feed_dict)
        # Train losses
        Loss.append(loss)
        Loss_rec.append(loss_rec)
        Divergences.append(divergences)
        # Test losses
        loss_test, loss_rec_test, mse = 0., 0., 0.
        if opts['true_gen_model']:
            codes, codes_mean = np.zeros((batches_num_te*batch_size_te,opts['zdim'])), np.zeros((batches_num_te*batch_size_te,opts['zdim']))
            labels = np.zeros((batches_num_te*batch_size_te,len(data.factor_indices)))
        if type(divergences)==list:
            divergences_test = np.zeros(len(divergences))
        else:
            divergences_test = 0.
        kl_to_prior = np.zeros(opts['zdim'])
        for it_ in range(batches_num_te):
            # Sample batches of data points
            data_ids = np.random.choice(test_size, batch_size_te, replace=True)
            batch_images_test = data.get_batch_img(data_ids, 'test').astype(np.float32)
            batch_pz_samples_test = sample_pz(opts, self.pz_params, batch_size_te)
            test_feed_dict = {self.batch: batch_images_test,
                              self.samples_pz: batch_pz_samples_test,
                              self.obj_fn_coeffs: opts['obj_fn_coeffs'],
                              self.is_training: False}
            [loss, l_rec, m, divergences, z, z_mean, kl] = self.sess.run([self.objective,
                                             self.loss_reconstruct,
                                             self.mse,
                                             self.divergences,
                                             self.z_samples,
                                             self.z_mean,
                                             self.kl_to_prior],
                                            feed_dict=test_feed_dict)
            loss_test += loss / batches_num_te
            loss_rec_test += l_rec / batches_num_te
            mse += m / batches_num_te
            divergences_test += np.array(divergences) / batches_num_te
            kl_to_prior += kl / batches_num_te
            if opts['true_gen_model']:
                codes[batch_size_te*it_:batch_size_te*(it_+1)] = z
                codes_mean[batch_size_te*it_:batch_size_te*(it_+1)] = z_mean
                labels[batch_size_te*it_:batch_size_te*(it_+1)] = data.get_batch_label(data_ids,'test')[:,data.factor_indices]
        Loss_test.append(loss_test)
        Loss_rec_test.append(loss_rec_test)
        MSE.append(mse)
        Divergences_test.append(divergences_test.tolist())
        if opts['true_gen_model']:
            betaVAE.append(self.compute_betaVAE(self.sess, data))
            MIG.append(self.compute_mig(codes_mean, labels))
            factorVAE.append(self.compute_factorVAE(self.sess, data, codes))
            SAP.append(self.compute_SAP(self.sess, data))
        # Printing various loss values
        logging.error(' \n Training done.')
        debug_str = 'EPOCH: %d/%d, BATCH:%d/%d' % (epoch,
                                                   opts['epoch_num'],
                                                   it + 1,
                                                   batches_num)
        logging.error(debug_str)
        debug_str = 'TRAIN LOSS=%.3f, TEST LOSS=%.3f' % (Loss[-1],Loss_test[-1])
        logging.error(debug_str)
        if opts['true_gen_model']:
            debug_str = 'betaVAE=%.3f, MIG=%.3f, factorVAE=%.3f, SAP=%.3f' % (
                                        betaVAE[-1],
                                        MIG[-1],
                                        factorVAE[-1],
                                        SAP[-1])
            logging.error(debug_str)
        if opts['model'] == 'BetaVAE':
            debug_str = 'REC=%.3f, TEST REC=%.3f, beta*KL=%10.3e, beta*TEST KL=%10.3e, \n '  % (
                                        Loss_rec[-1],
                                        Loss_rec_test[-1],
                                        Divergences[-1],
                                        Divergences_test[-1])
            logging.error(debug_str)
        elif opts['model'] == 'BetaTCVAE':
            debug_str = 'REC=%.3f, TEST REC=%.3f, b*TC=%10.3e, TEST b*TC=%10.3e, KL=%10.3e, TEST KL=%10.3e, \n '  % (
                                        Loss_rec[-1],
                                        Loss_rec_test[-1],
                                        Divergences[-1][0],
                                        Divergences_test[-1][0],
                                        Divergences[-1][1],
                                        Divergences_test[-1][1])
            logging.error(debug_str)
        elif opts['model'] == 'FactorVAE':
            debug_str = 'REC=%.3f, TEST REC=%.3f, b*KL=%10.3e, TEST b*KL=%10.3e, g*TC=%10.3e, TEST g*TC=%10.3e, \n '  % (
                                        Loss_rec[-1],
                                        Loss_rec_test[-1],
                                        Divergences[-1][0],
                                        Divergences_test[-1][0],
                                        Divergences[-1][1],
                                        Divergences_test[-1][1])
            logging.error(debug_str)
        elif opts['model'] == 'WAE':
            debug_str = 'REC=%.3f, TEST REC=%.3f, l*MMD=%10.3e, l*TEST MMD=%10.3e \n ' % (
                                        Loss_rec[-1],
                                        Loss_rec_test[-1],
                                        Divergences[-1],
                                        Divergences_test[-1])
            logging.error(debug_str)
        elif opts['model'] == 'disWAE':
            debug_str = 'TRAIN: REC=%.3f,l1*HSIC=%10.3e, l2*DIMWISE=%10.3e, WAE=%10.3e' % (
                                        Loss_rec[-1],
                                        Divergences[-1][0],
                                        Divergences[-1][1],
                                        Divergences[-1][2])
            logging.error(debug_str)
            debug_str = 'TEST : REC=%.3f, l1*HSIC=%10.3e, l2*DIMWISE=%10.3e, WAE=%10.3e \n ' % (
                                        Loss_rec_test[-1],
                                        Divergences_test[-1][0],
                                        Divergences_test[-1][1],
                                        Divergences_test[-1][2])
            logging.error(debug_str)
        elif opts['model'] == 'TCWAE_MWS' or opts['model'] == 'TCWAE_GAN':
            debug_str = 'TRAIN: REC=%.3f,l1*TC=%10.3e, l2*DIMWISE=%10.3e, WAE=%10.3e' % (
                                        Loss_rec[-1],
                                        Divergences[-1][0],
                                        Divergences[-1][1],
                                        Divergences[-1][2])
            logging.error(debug_str)
            debug_str = 'TEST : REC=%.3f, l1*TC=%10.3e, l2*DIMWISE=%10.3e, WAE=%10.3e \n ' % (
                                        Loss_rec_test[-1],
                                        Divergences_test[-1][0],
                                        Divergences_test[-1][1],
                                        Divergences_test[-1][2])
            logging.error(debug_str)
        else:
            raise NotImplementedError('Model type not recognised')

        # - Save the final model
        if opts['save_final'] and epoch > 0:
            self.saver.save(self.sess, os.path.join(exp_dir,
                                                'checkpoints',
                                                'trained-{}-final'.format(opts['model'])),
                                                global_step=counter)
        # - save training data
        if opts['save_train_data']:
            data_dir = 'train_data'
            save_path = os.path.join(exp_dir, data_dir)
            utils.create_dir(save_path)
            name = 'res_train_final'
            np.savez(os.path.join(save_path, name),
                    loss=np.array(Loss[-1]), loss_test=np.array(Loss_test[-1]),
                    loss_rec=np.array(Loss_rec[-1]), loss_rec_test=np.array(Loss_rec_test[-1]), mse = np.array(MSE[-1]),
                    divergences=np.array(Divergences[-1]), divergences_test=np.array(Divergences_test[-1]),
                    mig=np.array(MIG[-1]), factorVAE=np.array(factorVAE[-1]), sap=np.array(SAP[-1]))

    def test(self, data, WEIGHTS_PATH, verbose):
        """
        Test model and save different metrics
        """

        opts = self.opts

        # - Load trained weights
        if not tf.gfile.Exists(WEIGHTS_PATH+".meta"):
            raise Exception("weights file doesn't exist")
        self.saver.restore(self.sess, WEIGHTS_PATH)

        # - Set up
        # test_size = np.shape(data.test_data)[0]
        # batch_size_te = min(test_size,1000)
        test_size = 1000
        batch_size_te = min(test_size,100)
        batches_num_te = int(test_size/batch_size_te)+1

        # - Init all monitoring variables
        Loss, Loss_rec = 0., 0.
        Divergences = []
        betaVAE, MIG, factorVAE, SAP = 0., 0., 0., 0.
        real_blurr, blurr, fid_scores = 0., 0., 0.
        if opts['true_gen_model']:
            codes, codes_mean = np.zeros((batches_num_te*batch_size_te,opts['zdim'])), np.zeros((batches_num_te*batch_size_te,opts['zdim']))
            labels = np.zeros((batches_num_te*batch_size_te,len(data.factor_indices)))

        # - Testing loop
        for it_ in range(batches_num_te):
            # Sample batches of data points
            data_ids = np.random.choice(test_size, batch_size_te, replace=True)
            batch_images_test = data.test_data[data_ids].astype(np.float32)
            batch_pz_samples_test = sample_pz(opts, self.pz_params, batch_size_te)
            test_feed_dict = {self.batch: batch_images_test,
                              self.samples_pz: batch_pz_samples_test,
                              self.obj_fn_coeffs: opts['obj_fn_coeffs'],
                              self.is_training: False}
            [loss, l_rec, divergences, z, z_mean, samples] = self.sess.run([self.objective,
                                             self.loss_reconstruct,
                                             self.divergences,
                                             self.z_samples,
                                             self.z_mean,
                                             self.generated_x],
                                            feed_dict=test_feed_dict)
            Loss += loss / batches_num_te
            Loss_rec += l_rec / batches_num_te
            Divergences.append(divergences)
            # fid score
            if opts['fid']:
                # Load inception mean samples for train set
                trained_stats = os.path.join(inception_path, 'fid_stats.npz')
                # Load trained stats
                f = np.load(trained_stats)
                self.mu_train, self.sigma_train = f['mu'][:], f['sigma'][:]
                f.close()
                # Compute bluriness of real data
                real_blurriness = self.sess.run(self.blurriness,
                                            feed_dict={ self.batch: batch_images_test})
                real_blurr += np.mean(real_blurriness) / batches_num_te
                # Compute gen blur
                gen_blurr = self.sess.run(self.blurriness,
                                            feed_dict={self.batch: samples})
                blurr += np.mean(gen_blurr) / batches_num_te
                # Compute FID score
                # First convert to RGB
                if np.shape(samples)[-1] == 1:
                    # We have greyscale
                    samples = self.sess.run(tf.image.grayscale_to_rgb(samples))
                preds_incep = self.inception_sess.run(self.inception_layer,
                              feed_dict={'FID_Inception_Net/ExpandDims:0': samples})
                preds_incep = preds_incep.reshape((batch_size_te,-1))
                mu_gen = np.mean(preds_incep, axis=0)
                sigma_gen = np.cov(preds_incep, rowvar=False)
                fid_score = fid.calculate_frechet_distance(mu_gen, sigma_gen,
                                            self.mu_train,
                                            self.sigma_train,
                                            eps=1e-6)
                fid_scores += fid_score / batches_num_te

            # storing labels and factors
            if opts['true_gen_model']:
                codes[batch_size_te*it_:batch_size_te*(it_+1)] = z
                codes_mean[batch_size_te*it_:batch_size_te*(it_+1)] = z_mean
                batch_labels_test = data.test_labels[data_ids][:,data.factor_indices]
                labels[batch_size_te*it_:batch_size_te*(it_+1)] = data.test_labels[data_ids][:,data.factor_indices]
        Divergences = np.mean(Divergences, axis=0)

        # - Compute metrics
        if opts['true_gen_model']:
            # betaVAE = self.compute_betaVAE(self.sess, data)
            MIG = self.compute_mig(codes_mean, labels)
            factorVAE = self.compute_factorVAE(self.sess, data, codes)
            SAP = self.compute_SAP(self.sess, data)

        # - Printing various loss values
        if verbose=='high':
            debug_str = 'Testing done.'
            logging.error(debug_str)
            if opts['true_gen_model']:
                debug_str = 'betaVAE=%.3f, MIG=%.3f, factorVAE=%.3f, SAP=%.3f' % (
                                            betaVAE,
                                            MIG,
                                            factorVAE,
                                            SAP)
                logging.error(debug_str)
            if opts['fid']:
                debug_str = 'Real blurr=%10.3e, blurr=%10.3e, FID=%.3f \n ' % (
                                            real_blurr,
                                            blurr,
                                            fid_scores)
                logging.error(debug_str)

            if opts['model'] == 'BetaVAE':
                debug_str = 'LOSS=%.3f, REC=%.3f, beta*KL=%10.3e \n '  % (
                                            Loss,
                                            Loss_rec,
                                            Divergences)
                logging.error(debug_str)
            elif opts['model'] == 'BetaTCVAE':
                debug_str = 'LOSS=%.3f, REC=%.3f, b*TC=%10.3e, KL=%10.3e \n '  % (
                                            Loss,
                                            Loss_rec,
                                            Divergences[0],
                                            Divergences[1])
                logging.error(debug_str)
            elif opts['model'] == 'FactorVAE':
                debug_str = 'LOSS=%.3f, REC=%.3f, b*KL=%10.3e, g*TC=%10.3e, \n '  % (
                                            Loss,
                                            Loss_rec,
                                            Divergences[0],
                                            Divergences[1])
                logging.error(debug_str)
            elif opts['model'] == 'WAE':
                debug_str = 'LOSS=%.3f, REC=%.3f, b*MMD=%10.3e \n ' % (
                                            Loss,
                                            Loss_rec,
                                            Divergences)
                logging.error(debug_str)
            elif opts['model'] == 'disWAE':
                debug_str = 'LOSS=%.3f, REC=%.3f, b*HSIC=%10.3e, g*DIMWISE=%10.3e, WAE=%10.3e' % (
                                            Loss,
                                            Loss_rec,
                                            Divergences[0],
                                            Divergences[1],
                                            Divergences[2])
                logging.error(debug_str)
            elif opts['model'] == 'TCWAE_MWS' or opts['model'] == 'TCWAE_GAN':
                debug_str = 'LOSS=%.3f, REC=%.3f,l1*TC=%10.3e, l2*DIMWISE=%10.3e, WAE=%10.3e' % (
                                            Loss,
                                            Loss_rec,
                                            Divergences[0],
                                            Divergences[1],
                                            Divergences[2])
                logging.error(debug_str)
            else:
                raise NotImplementedError('Model type not recognised')


        # - save testing data
        data_dir = 'test_data'
        save_path = os.path.join(opts['exp_dir'], data_dir)
        utils.create_dir(save_path)
        name = 'res_test_final'
        np.savez(os.path.join(save_path, name),
                loss=np.array(Loss),
                loss_rec=np.array(Loss_rec),
                divergences=Divergences,
                betavae=np.array(betaVAE),
                mig=np.array(MIG),
                factorVAE=np.array(factorVAE),
                sap=np.array(SAP),
                real_blurr=np.array(real_blurr),
                blurr=np.array(blurr),
                fid=np.array(fid_scores))

    def plot(self, data, WEIGHTS_PATH):
        """
        Plots reconstructions, latent transversals and model samples
        """

        opts = self.opts

        # - Load trained weights
        if not tf.gfile.Exists(WEIGHTS_PATH+".meta"):
            raise Exception("weights file doesn't exist")
        self.saver.restore(self.sess, WEIGHTS_PATH)

        # - Set up
        im_shape = datashapes[opts['dataset']]
        if opts['dataset']=='celebA' or opts['dataset']=='3Dchairs':
            num_pics = 100
            num_steps = 7
        else:
            num_pics = 20
            num_steps = 10
        fixed_noise = sample_pz(opts, self.pz_params, num_pics)


        # - Auto-encoding test images & samples generated by the model
        [reconstructions, latents, generations] = self.sess.run(
                                    [self.recon_x,
                                     self.z_samples,
                                     self.generated_x],
                                    feed_dict={self.batch: data.data[self.plot_data_idx],
                                               self.samples_pz: fixed_noise,
                                               self.is_training: False})
        # - get kl(q(z_i),p(z_i)) on test data to plot latent traversals
        test_size = self.test_size
        batch_size_te = min(test_size,1000)
        batches_num_te = int(test_size/batch_size_te)+1
        kl_to_prior = np.zeros(opts['zdim'])
        for it_ in range(batches_num_te):
            data_ids = np.random.choice(test_size, batch_size_te, replace=True)
            batch_images_test = data.get_batch_img(data_ids, 'test').astype(np.float32)
            kl = self.sess.run(self.kl_to_prior, feed_dict={
                                                self.batch: batch_images_test,
                                                self.is_training: False})
            kl_to_prior += kl / batches_num_te

        # - Latent transversals
        enc_var = np.ones(opts['zdim'])
        # create latent linespacel
        if opts['dataset']=='celebA' :
            idx = [0,3,20,26,40,49]
            latent_transversal = linespace(opts, num_steps,  # shape: [nanchors, zdim, nsteps, zdim]
                                    anchors=latents[idx],
                                    std=enc_var)
            # latent_transversal = latent_transversal[:,:,::-1]
        elif opts['dataset']=='3Dchairs':
            idx = [48,4,21,44,1]
            latent_transversal = linespace(opts, num_steps,  # shape: [nanchors, zdim, nsteps, zdim]
                                    anchors=latents[idx],
                                    std=enc_var)
            latent_transversal = latent_transversal[:,:,::-1]
        else:
            latent_transversal = linespace(opts, num_steps,  # shape: [nanchors, zdim, nsteps, zdim]
                                    anchors=latents[::2],
                                    std=enc_var)
        # - Reconstructing latent transversals
        obs_transversal = self.sess.run(self.generated_x,
                                    feed_dict={self.samples_pz: np.reshape(latent_transversal,[-1, opts['zdim']]),
                                               self.is_training: False})
        obs_transversal = np.reshape(obs_transversal, [-1, opts['zdim'], num_steps]+im_shape)
        kl_to_prior_sorted = np.argsort(kl_to_prior)[::-1]
        obs_transversal = obs_transversal[:,kl_to_prior_sorted]

        # - ploting and saving
        if opts['dataset']=='celebA' or opts['dataset']=='3Dchairs':
            save_test_celeba(opts, data.data[self.plot_data_idx],
                                        reconstructions,
                                        obs_transversal,
                                        generations,
                                        opts['exp_dir'])
            save_dimwise_traversals(opts, obs_transversal, opts['exp_dir'])
        else:
            save_test_smallnorb(opts, data.data[self.plot_data_idx],
                                        reconstructions,
                                        obs_transversal,
                                        generations,
                                        opts['exp_dir'])
