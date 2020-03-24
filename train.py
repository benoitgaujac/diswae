"""
Auto-Encoder models
"""
import os
import logging

import numpy as np
import tensorflow as tf
from sklearn import linear_model

import utils
from sampling_functions import sample_pz, linespace
from plot_functions import save_train, save_test
from plot_functions import plot_embedded, plot_encSigma, plot_interpolation
import models
from datahandler import datashapes

import pdb

class Run(object):

    def __init__(self, opts):

        logging.error('Building the Tensorflow Graph')

        # --- Create session
        self.sess = tf.Session()
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
            self.obj_fn_coeffs = (self.lmbd1, self.lmbd2)
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
        self.objective, self.loss_reconstruct, self.divergences, self.recon_x, self.enc_z, self.enc_sigmastats \
            = self.model.loss(inputs=self.batch,
                              samples=self.samples_pz,
                              loss_coeffs=self.obj_fn_coeffs,
                              is_training=self.is_training,
                              dropout_rate=self.dropout_rate)

        self.z_samples, self.z_mean, _, _, _, _ = self.model.forward_pass(inputs=self.batch,
                                                                          is_training=self.is_training,
                                                                          dropout_rate=self.dropout_rate,
                                                                          reuse=True)

        self.generated_x = self.model.sample_x_from_prior(noise=self.samples_pz)

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
        self.dropout_rate = tf.placeholder(tf.float32, name='dropout_rate_ph')
        self.batch_size = tf.placeholder(tf.int32, name='batch_size_ph')
        if self.opts['model']=='BetaVAE' or self.opts['model'] == 'BetaTCVAE':
            self.beta = tf.placeholder(tf.float32, name='beta_ph')
        elif self.opts['model']=='WAE':
            self.lmbd = tf.placeholder(tf.float32, name='lambda_ph')
        else:
            self.lmbd1 = tf.placeholder(tf.float32, name='lambda1_ph')
            self.lmbd2 = tf.placeholder(tf.float32, name='lambda2_ph')

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

    def generate_factorVAE_minibatch(self, sess, data, global_variances, active_dims):
        opts = self.opts
        batch_size = 64
        # sample batch of factors
        factors = utils.sample_factors(opts, batch_size, data)
        # sample factor idx
        factor_index = np.random.randint(len(data.factor_indices))
        factor_index = data.factor_indices[factor_index]
        # fixing the selected factor across batch
        factors[:, factor_index] = factors[0, factor_index]
        # sample batch of images with fix selected factor
        batch_images = utils.sample_images(opts, opts['dataset'], data, factors)
        # encode images
        z = sess.run(self.z_samples, feed_dict={self.batch: batch_images,
                                                self.dropout_rate: 1.,
                                                self.is_training: False})
        # get variance per dimension and vote
        local_variances = np.var(z, axis=0, ddof=1)
        argmin = np.argmin(local_variances[active_dims] / global_variances[active_dims])

        return factor_index, argmin

    def compute_factorVAE(self, sess, data, codes):
        """Compute FactorVAE metric"""
        opts = self.opts
        threshold = .05
        # Compute global variance and pruning dimensions
        global_variances = np.var(codes, axis=0, ddof=1)
        active_dims = np.sqrt(global_variances)>=threshold
        # Generate classifier training set and build classifier
        # training_size = 100
        training_size = 10000
        votes = np.zeros((len(data.factor_sizes), opts['zdim']),dtype=np.int32)
        for i in range(training_size):
            factor, vote = self.generate_factorVAE_minibatch(sess,
                                                    data,
                                                    global_variances,
                                                    active_dims)
            votes[factor, vote] += 1
            # print('{} training points generated'.format(i+1))
        classifier = np.argmax(votes, axis=0)
        other_index = np.arange(votes.shape[1])
        # Generate classifier eval set and get eval accuracy
        # eval_size = 50
        eval_size = 5000
        votes = np.zeros((len(data.factor_sizes), opts['zdim']),dtype=np.int32)
        for i in range(eval_size):
            factor, vote = self.generate_factorVAE_minibatch(sess,
                                                    data,
                                                    global_variances,
                                                    active_dims)
            votes[factor, vote] += 1
            # print('{} eval points generated'.format(i+1))
        acc = np.sum(votes[classifier, other_index]) * 1. / np.sum(votes)

        return acc

    def generate_SAP_minibatch(self, sess, data, num_points):
        opts = self.opts
        batch_size = 500
        # batch_size = 50
        representations = None
        factors = None
        i = 0
        while i < num_points:
            num_points_iter = min(num_points - i, batch_size)
            # sample batch of factors
            current_factors = utils.sample_factors(opts, num_points_iter, data)
            # sample batch of images from factors
            batch_images = utils.sample_images(opts, opts['dataset'], data, current_factors)
            # encode images
            current_z = sess.run(self.z_samples, feed_dict={
                                                self.batch: batch_images,
                                                self.dropout_rate: 1.,
                                                self.is_training: False})
            if i == 0:
                factors = current_factors
                z = current_z
            else:
                factors = np.vstack((factors, current_factors))
                z = np.vstack((z,current_z))
            i += num_points_iter

        return z, factors

    def compute_SAP(self, sess, data):
        """Compute SAP metric"""
        opts = self.opts
        # Generate training set
        # training_size = 100
        training_size = 10000
        mus, ys = self.generate_SAP_minibatch(sess, data, training_size)
        # Generate testing set
        # testing_size = 50
        testing_size = 5000
        mus_test, ys_test = self.generate_SAP_minibatch(sess, data, testing_size)
        # Computing score matrix
        score_matrix = utils.compute_score_matrix(mus, ys, mus_test, ys_test)
        # average diff top 2 predictive latent dim for each factor
        sorted_score_matric = np.sort(score_matrix, axis=0)
        sap = np.mean(sorted_score_matric[-1, :] - sorted_score_matric[-2, :])

        return sap

    def generate_betaVAE_minibatch(self, sess, data):
        opts = self.opts
        batch_size = 64
        # sample 2 batches of factors
        factors_1 = utils.sample_factors(opts, batch_size, data)
        factors_2 = utils.sample_factors(opts, batch_size, data)
        # sample factor idx
        factor_index = np.random.randint(len(data.factor_indices))
        factor_index = data.factor_indices[factor_index]
        # fixing the selected factor across batch
        factors_1[:, factor_index] = factors_2[:, factor_index]
        # sample images with fix selected factor
        images_1 = utils.sample_images(opts, opts['dataset'], data, factors_1)
        images_2 = utils.sample_images(opts, opts['dataset'], data, factors_2)
        # encode images
        z_1 = sess.run(self.z_samples, feed_dict={self.batch: images_1,
                                                self.dropout_rate: 1.,
                                                self.is_training: False})
        z_2 = sess.run(self.z_samples, feed_dict={self.batch: images_2,
                                                self.dropout_rate: 1.,
                                                self.is_training: False})
        # Compute the feature vector based on differences in representation.
        feature_vector = np.mean(np.abs(z_1 - z_2), axis=0)

        return feature_vector, factor_index

    def compute_betaVAE(self, sess, data):
        """Compute betaVAE metric"""
        opts = self.opts
        # Generate classifier training set and build classifier
        # training_size = 100
        training_size = 10000
        x_train = np.zeros((training_size,opts['zdim']))
        y_train = np.zeros((training_size,))
        for i in range(training_size):
            x_train[i], y_train[i] = self.generate_betaVAE_minibatch(sess, data)
        # logging.info("Training sklearn model.")
        model = linear_model.LogisticRegression()
        model.fit(x_train, y_train)
        # Generate classifier eval set and get eval accuracy
        # eval_size = 50
        eval_size = 5000
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

    def discr_optimizer(self):
        return tf.train.AdamOptimizer(0.0001, beta1=0.5, beta2=0.9,)

    def add_optimizers(self):
        opts = self.opts
        lr = opts['lr']
        opt = self.optimizer(lr, self.lr_decay)
        encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                scope='encoder')
        decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                scope='decoder')
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # discriminator opt if needed
        if self.opts['model']=='FactorVAE' or self.opts['model']=='TCWAE_GAN':
            discr_opt = self.discr_optimizer()
            discr_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                    scope='discriminator')
            vae_opt = opt.minimize(loss=self.objective,var_list=encoder_vars + decoder_vars)
            discriminator_opt = discr_opt.minimize(loss=-self.model.discr_loss,var_list=discr_vars)
            self.opt = tf.group(vae_opt, discriminator_opt, update_ops)
        else:
            with tf.control_dependencies(update_ops):
                self.opt = opt.minimize(loss=self.objective,var_list=encoder_vars + decoder_vars)

    def train(self, data, WEIGHTS_FILE):
        """
        Train top-down model with chosen method
        """
        opts = self.opts
        logging.error('Training {}'.format(self.opts['model']))
        exp_dir = opts['exp_dir']

        # writer = tf.summary.FileWriter(exp_dir)

        # - Init sess and load trained weights if needed
        if opts['use_trained']:
            if not tf.gfile.Exists(WEIGHTS_FILE+".meta"):
                raise Exception("weights file doesn't exist")
            self.saver.restore(self.sess, WEIGHTS_FILE)
        else:
            self.sess.run(self.initializer)

        # - Set up for training
        train_size = data.num_points
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
        if opts['vizu_encSigma']:
            enc_Sigmas = []

        decay, counter = 1., 0
        decay_steps, decay_rate = int(batches_num * opts['epoch_num'] / 5), 0.98
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
                batch_images = data.data[data_ids].astype(np.float32)
                batch_pz_samples = sample_pz(opts, self.pz_params, opts['batch_size'])
                # Feeding dictionary
                feed_dict = {self.batch: batch_images,
                             self.samples_pz: batch_pz_samples,
                             self.lr_decay: decay,
                             self.obj_fn_coeffs: opts['obj_fn_coeffs'],
                             self.dropout_rate: opts['dropout_rate'],
                             self.is_training: True}
                [_, loss, loss_rec, divergences, enc_sigmastats] = self.sess.run([
                                                self.opt,
                                                self.objective,
                                                self.loss_reconstruct,
                                                self.divergences,
                                                self.enc_sigmastats],
                                                feed_dict=feed_dict)

                ##### TESTING LOOP #####
                if (counter+1)%opts['evaluate_every'] == 0 or (counter+1) == 100:
                    print("Epoch {}, Iteration {}".format(epoch, it+1))
                    # batch_size_te = 64
                    test_size = np.shape(data.test_data)[0]
                    batch_size_te = min(test_size,1000)
                    batches_num_te = int(test_size/batch_size_te)+1
                    # Train losses
                    Loss.append(loss)
                    Loss_rec.append(loss_rec)
                    Divergences.append(divergences)

                    # Encoded Sigma
                    if opts['vizu_encSigma']:
                        enc_Sigmas.append(enc_sigmastats)

                    # Test losses
                    loss_test, loss_rec_test = 0., 0.
                    if opts['true_gen_model']:
                        codes, codes_mean = np.zeros((batches_num_te*batch_size_te,opts['zdim'])), np.zeros((batches_num_te*batch_size_te,opts['zdim']))
                        labels = np.zeros((batches_num_te*batch_size_te,len(data.factor_indices)))
                    if type(divergences)==list:
                        divergences_test = np.zeros(len(divergences))
                    else:
                        divergences_test = 0.
                    for it_ in range(batches_num_te):
                        # Sample batches of data points
                        data_ids = np.random.choice(test_size, batch_size_te, replace=True)
                        batch_images_test = data.test_data[data_ids].astype(np.float32)
                        batch_pz_samples_test = sample_pz(opts, self.pz_params, batch_size_te)
                        test_feed_dict = {self.batch: batch_images_test,
                                          self.samples_pz: batch_pz_samples_test,
                                          self.obj_fn_coeffs: opts['obj_fn_coeffs'],
                                          self.dropout_rate: 1.,
                                          self.is_training: False}
                        [loss, l_rec, divergences, z, z_mean] = self.sess.run([self.objective,
                                                         self.loss_reconstruct,
                                                         self.divergences,
                                                         self.z_samples,
                                                         self.z_mean],
                                                        feed_dict=test_feed_dict)
                        loss_test += loss / batches_num_te
                        loss_rec_test += l_rec / batches_num_te
                        divergences_test += np.array(divergences) / batches_num_te
                        if opts['true_gen_model']:
                            codes[batch_size_te*it_:batch_size_te*(it_+1)] = z
                            codes_mean[batch_size_te*it_:batch_size_te*(it_+1)] = z_mean
                            batch_labels_test = data.test_labels[data_ids][:,data.factor_indices]
                            labels[batch_size_te*it_:batch_size_te*(it_+1)] = data.test_labels[data_ids][:,data.factor_indices]
                    Loss_test.append(loss_test)
                    Loss_rec_test.append(loss_rec_test)
                    Divergences_test.append(divergences_test.tolist())
                    if opts['true_gen_model']:
                        betaVAE.append(self.compute_betaVAE(self.sess, data))
                        MIG.append(self.compute_mig(codes_mean, labels))
                        factorVAE.append(self.compute_factorVAE(self.sess, data, codes))
                        SAP.append(self.compute_SAP(self.sess, data))

                if (counter+1)%opts['print_every'] == 0 or (counter+1) == 100:
                    # Plot vizualizations
                    # Auto-encoding test images & samples generated by the model
                    [reconstructions_test, latents_test, generations] = self.sess.run(
                                                [self.recon_x,
                                                 self.enc_z,
                                                 self.generated_x],
                                                feed_dict={self.batch: data.vizu_data[0:opts['plot_num_pics']],       # TODO what is this?
                                                           self.samples_pz: fixed_noise,
                                                           self.dropout_rate: 1.,
                                                           self.is_training: False})
                    # Auto-encoding training images
                    reconstructions_train = self.sess.run(self.recon_x,
                                                          feed_dict={self.batch: data.data[200:200+npics],  # TODO what is this?
                                                                     self.dropout_rate: 1.,
                                                                     self.is_training: False})

                    # - Plotting embeddings, Sigma, latent interpolation, and saving
                    # Embeddings
                    if opts['vizu_embedded'] and counter > 1:
                        plot_embedded(opts, [latents_test[:npics]], [fixed_noise],
                                      data.vizu_labels[:npics],
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
                                                                     self.dropout_rate: 1.,
                                                                     self.is_training: False})
                        inter_anchors = np.reshape(dec_interpolation, [-1, opts['zdim'], num_steps]+im_shape)
                        plot_interpolation(opts, inter_anchors, exp_dir,
                                           'inter_e%04d_mb%05d.png' % (epoch, it))

                    # Saving plots
                    save_train(opts,
                              data.data[200:200+npics], data.vizu_data[:npics],     # images
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

                # - Update learning rate if necessary and counter
                # if counter >= batches_num * opts['epoch_num'] / 5 and counter % decay_steps == 0:
                if False:
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
                    loss_rec=np.array(Loss_rec[-1]), loss_rec_test=np.array(Loss_rec_test[-1]),
                    divergences=np.array(Divergences[-1]), divergences_test=np.array(Divergences_test[-1]),
                    mig=np.array(MIG[-1]), factorVAE=np.array(factorVAE[-1]), sap=np.array(SAP[-1]))

        logging.error('Training done.')

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
        test_size = np.shape(data.test_data)[0]
        batch_size_te = min(test_size,1000)
        batches_num_te = int(test_size/batch_size_te)+1

        # - Init all monitoring variables
        Loss, Loss_rec = 0., 0.
        Divergences = []
        betaVAE, MIG, factorVAE, SAP = 0., 0., 0., 0.
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
                              self.dropout_rate: 1.,
                              self.is_training: False}
            [loss, l_rec, divergences, z, z_mean] = self.sess.run([self.objective,
                                             self.loss_reconstruct,
                                             self.divergences,
                                             self.z_samples,
                                             self.z_mean],
                                            feed_dict=test_feed_dict)
            Loss += loss / batches_num_te
            Loss_rec += l_rec / batches_num_te
            Divergences.append(divergences)
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
                sap=np.array(SAP))

    def plot(self, data, WEIGHTS_PATH):
        """
        Test model and save different metrics
        """

        opts = self.opts

        # - Load trained weights
        if not tf.gfile.Exists(WEIGHTS_PATH+".meta"):
            raise Exception("weights file doesn't exist")
        self.saver.restore(self.sess, WEIGHTS_PATH)

        # - set up
        npics = opts['plot_num_pics']
        im_shape = datashapes[opts['dataset']]
        fixed_noise = sample_pz(opts, self.pz_params, npics)
        anchors_ids = [0, 4, 6, 12, 39]

        # - Auto-encoding test images & samples generated by the model
        [reconstructions_test, latents_test, generations] = self.sess.run(
                                    [self.recon_x,
                                     self.enc_z,
                                     self.generated_x],
                                    feed_dict={self.batch: data.vizu_data[0:opts['plot_num_pics']],
                                               self.samples_pz: fixed_noise,
                                               self.dropout_rate: 1.,
                                               self.is_training: False})

        # Saving plots
        save_test(opts, data.vizu_data[:npics],
                        reconstructions_test,
                        generations, opts['exp_dir'])

        # - Embeddings
        if opts['vizu_embedded']:
            plot_embedded(opts, [latents_test[:npics]], [fixed_noise],
                          data.vizu_labels[:npics],
                          opts['exp_dir'], 'embedded.png', False)

        # - Latent transversals
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
                                                         self.dropout_rate: 1.,
                                                         self.is_training: False})
            inter_anchors = np.reshape(dec_interpolation, [-1, opts['zdim'], num_steps]+im_shape)
            plot_interpolation(opts, inter_anchors, opts['exp_dir'],
                               'latent_trans.png', False)
