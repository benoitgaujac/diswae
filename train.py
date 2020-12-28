"""
Auto-Encoder models
"""
import os
import sys
import logging

import numpy as np
import tensorflow as tf
from sklearn import linear_model
from math import ceil

import utils
from sampling_functions import sample_pz, linespace
from plot_functions import save_train, save_test_smallnorb, save_test_celeba, save_dimwise_traversals
from plot_functions import plot_embedded, plot_encSigma, plot_interpolation
import models
from datahandler import datashapes
from fid.fid import calculate_frechet_distance

import pdb

class Run(object):

    def __init__(self, opts, data):

        logging.error('Building the Tensorflow Graph')
        self.opts = opts

        # --- Data
        assert opts['dataset'] in datashapes, 'Unknown dataset.'
        self.data = data

        # --- Initialize prior parameters
        mean = np.zeros(opts['zdim'], dtype='float32')
        Sigma = np.ones(opts['zdim'], dtype='float32')
        self.pz_params = np.concatenate([mean, Sigma], axis=0)

        # --- Placeholders
        self.add_ph()

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
            self.obj_fn_coeffs = self.beta
        elif opts['model'] == 'TCWAE_MWS':
            self.model = models.TCWAE_MWS(opts)
            self.obj_fn_coeffs = (self.lmbd1, self.lmbd2)
        elif opts['model'] == 'TCWAE_MWS_MI':
            self.model = models.TCWAE_MWS_MI(opts)
            self.obj_fn_coeffs = (self.lmbd1, self.lmbd2)
        elif opts['model'] == 'TCWAE_GAN':
            self.model = models.TCWAE_GAN(opts)
            self.obj_fn_coeffs = (self.lmbd1, self.lmbd2)
        elif opts['model'] == 'TCWAE_GAN_MI':
            self.model = models.TCWAE_GAN_MI(opts)
            self.obj_fn_coeffs = (self.lmbd1, self.lmbd2)
        else:
            raise NotImplementedError()

        # --- Define Objective
        self.loss_reconstruct, self.divergences = self.model.loss(
                                inputs=self.data.next_element,
                                loss_coeffs=self.obj_fn_coeffs,
                                is_training=self.is_training)
        if opts['model'] == 'BetaVAE' or opts['model'] == 'WAE':
            self.objective = self.loss_reconstruct + self.divergences
        else:
            self.objective = self.loss_reconstruct\
                            + self.divergences[0] + self.divergences[1]

        # --- Get batchnorm ops for training only
        self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # --- encode & decode pass for testing
        self.z_samples, self.z_mean, self.z_sigma, self.recon_x, _ =\
            self.model.forward_pass(inputs=self.data.next_element,
                                is_training=self.is_training,
                                reuse=True)

        # --- kl
        self.kl_to_prior = self.model.dimewise_kl_to_prior(self.z_samples,
                                self.z_mean, self.z_sigma)

        # --- MSE
        self.mse = self.model.MSE(self.data.next_element, self.recon_x)

        # --- encode & decode pass for vizu
        self.encoded, self.encoded_mean, _, self.decoded, _ =\
            self.model.forward_pass(inputs=self.inputs_img,
                                is_training=self.is_training,
                                reuse=True)

        # --- Sampling
        self.generated = self.model.sample_x_from_prior(noise=self.pz_samples)

        # --- FID score
        if opts['fid']:
            self.inception_graph = tf.Graph()
            self.inception_sess = tf.Session(graph=self.inception_graph)
            with self.inception_graph.as_default():
                self.create_inception_graph()
            self.inception_layer = self._get_inception_layer()

        # --- Optimizers, savers, etc
        self.add_optimizers()

        # --- Init iteratorssess, saver and load trained weights if needed, else init variables
        self.sess = tf.Session()
        self.train_handle, self.test_handle = self.data.init_iterator(self.sess)
        self.saver = tf.train.Saver(max_to_keep=10)
        self.initializer = tf.global_variables_initializer()
        self.sess.graph.finalize()


    def add_ph(self):
        self.lr_decay = tf.placeholder(tf.float32, name='rate_decay_ph')
        self.is_training = tf.placeholder(tf.bool, name='is_training_ph')
        self.pz_samples = tf.placeholder(tf.float32,
                                         [None] + [self.opts['zdim'],],
                                         name='noise_ph')
        self.inputs_img = tf.placeholder(tf.float32,
                                         [None] + self.data.data_shape,
                                         name='point_ph')
        if self.opts['model'][:5]=='TCWAE':
            self.lmbd1 = tf.placeholder(tf.float32, name='lambda1_ph')
            self.lmbd2 = tf.placeholder(tf.float32, name='lambda2_ph')
        else:
            self.beta = tf.placeholder(tf.float32, name='beta_ph')

    def create_inception_graph(self):
        inception_model = 'classify_image_graph_def.pb'
        inception_path = os.path.join('fid', inception_model)
        # Create inception graph
        with tf.gfile.FastGFile(inception_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString( f.read())
            _ = tf.import_graph_def( graph_def, name='FID_Inception_Net')

    def _get_inception_layer(self):
        # Get inception activation layer (and reshape for batching)
        layername = 'FID_Inception_Net/pool_3:0'
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

    def generate_factorVAE_minibatch(self, global_variances, active_dims):
        opts = self.opts
        batch_size = 64
        # sample batch of factors
        factors = utils.sample_factors(batch_size, self.data.factor_sizes)
        # sample factor idx
        factor_index = np.random.randint(len(self.data.factor_indices))
        factor_index = self.data.factor_indices[factor_index]
        # fixing the selected factor across batch
        factors[:, factor_index] = factors[0, factor_index]
        # sample batch of images with fix selected factor
        batch_images = self.data.sample_observations_from_factors(opts, factors)
        # encode images
        z = self.sess.run(self.encoded, feed_dict={self.inputs_img: batch_images,
                                                self.is_training: False})
        # get variance per dimension and vote
        local_variances = np.var(z, axis=0, ddof=1)
        argmin = np.argmin(local_variances[active_dims] / global_variances[active_dims])

        return factor_index, argmin

    def compute_factorVAE(self, codes):
        """Compute FactorVAE metric"""
        opts = self.opts
        threshold = .05
        # Compute global variance and pruning dimensions
        global_variances = np.var(codes, axis=0, ddof=1)
        active_dims = np.sqrt(global_variances)>=threshold
        # Generate classifier training set and build classifier
        training_size = 5000
        # training_size = 100
        votes = np.zeros((len(self.data.factor_sizes), opts['zdim']),dtype=np.int32)
        for i in range(training_size):
            factor, vote = self.generate_factorVAE_minibatch(global_variances,
                                                    active_dims)
            votes[factor, vote] += 1
            # print('{} training points generated'.format(i+1))
        classifier = np.argmax(votes, axis=0)
        other_index = np.arange(votes.shape[1])
        # Generate classifier eval set and get eval accuracy
        eval_size = 1000
        # eval_size = 50
        votes = np.zeros((len(self.data.factor_sizes), opts['zdim']),dtype=np.int32)
        for i in range(eval_size):
            factor, vote = self.generate_factorVAE_minibatch(global_variances,
                                                    active_dims)
            votes[factor, vote] += 1
            # print('{} eval points generated'.format(i+1))
        acc = np.sum(votes[classifier, other_index]) * 1. / np.sum(votes)

        return acc

    def generate_SAP_minibatch(self, num_points):
        opts = self.opts
        batch_size = 64
        representations = None
        factors = None
        i = 0
        while i < num_points:
            num_points_iter = min(num_points - i, batch_size)
            # sample batch of factors
            current_factors = utils.sample_factors(num_points_iter, self.data.factor_sizes)
            # sample batch of images from factors
            batch_images = self.data.sample_observations_from_factors(opts, current_factors)
            # encode images
            current_z = self.sess.run(self.encoded, feed_dict={
                                                self.inputs_img: batch_images,
                                                self.is_training: False})
            if i == 0:
                factors = current_factors
                z = current_z
            else:
                factors = np.vstack((factors, current_factors))
                z = np.vstack((z,current_z))
            i += num_points_iter

        return z, factors

    def compute_SAP(self, z, y, z_test, y_test):
        """Compute SAP metric"""
        # # Generate training set
        # training_size = 4000
        # # training_size = 100
        # mus, ys = self.generate_SAP_minibatch(training_size)
        # # Generate testing set
        # testing_size = 2000
        # # testing_size = 50
        # mus_test, ys_test = self.generate_SAP_minibatch(testing_size)
        # Computing score matrix
        score_matrix = utils.compute_score_matrix(z, y, z_test, y_test)
        # average diff top 2 predictive latent dim for each factor
        sorted_score_matric = np.sort(score_matrix, axis=0)
        sap = np.mean(sorted_score_matric[-1, :] - sorted_score_matric[-2, :])

        return sap

    def generate_betaVAE_minibatch(self):
        opts = self.opts
        batch_size = 64
        # sample 2 batches of factors
        factors_1 = utils.sample_factors(batch_size, self.data.factor_sizes)
        factors_2 = utils.sample_factors(batch_size, self.data.factor_sizes)
        # sample factor idx
        factor_index = np.random.randint(len(self.data.factor_indices))
        factor_index = self.data.factor_indices[factor_index]
        # fixing the selected factor across batch
        factors_1[:, factor_index] = factors_2[:, factor_index]
        # sample images with fix selected factor
        images_1 = slef.data.sample_observations_from_factors(opts, factors_1)
        images_2 = slef.data.sample_observations_from_factors(opts, factors_2)
        # encode images
        z_1 = self.sess.run(self.z_samples, feed_dict={self.batch: images_1,
                                                self.is_training: False})
        z_2 = self.sess.run(self.z_samples, feed_dict={self.batch: images_2,
                                                self.is_training: False})
        # Compute the feature vector based on differences in representation.
        feature_vector = np.mean(np.abs(z_1 - z_2), axis=0)

        return feature_vector, factor_index

    def compute_betaVAE(self):
        """Compute betaVAE metric"""
        opts = self.opts
        # Generate classifier training set and build classifier
        training_size = 2000
        # training_size = 100
        x_train = np.zeros((training_size,opts['zdim']))
        y_train = np.zeros((training_size,))
        for i in range(training_size):
            x_train[i], y_train[i] = self.generate_betaVAE_minibatch()
        # logging.info("Training sklearn model.")
        model = linear_model.LogisticRegression()
        model.fit(x_train, y_train)
        # Generate classifier eval set and get eval accuracy
        eval_size = 1000
        # eval_size = 50
        x_eval = np.zeros((eval_size,opts['zdim']))
        y_eval = np.zeros((eval_size,))
        for i in range(eval_size):
            x_eval[i], y_eval[i] = self.generate_betaVAE_minibatch()
        acc = model.score(x_eval, y_eval)

        return acc

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
        # discriminator opt if needed
        if self.opts['model'] in ['FactorVAE','TCWAE_GAN','TCWAE_GAN_MI']:
            if opts['dataset']=='celebA' or opts['dataset']=='3Dchairs':
                discr_opt = self.discr_optimizer(0.00001)
            else:
                discr_opt = self.discr_optimizer()
            discr_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                    scope='discriminator')
            with tf.control_dependencies(self.extra_update_ops):
                vae_opt = opt.minimize(loss=self.objective,var_list=encoder_vars + decoder_vars)
                discriminator_opt = discr_opt.minimize(loss=self.model.discr_loss,var_list=discr_vars)
            self.opt = tf.group(vae_opt, discriminator_opt)
        else:
            # self.opt = opt.minimize(loss=self.objective,var_list=encoder_vars + decoder_vars)
            with tf.control_dependencies(self.extra_update_ops):
                self.opt = opt.minimize(loss=self.objective,var_list=encoder_vars + decoder_vars)

    def train(self, MODEL_PATH=None, WEIGHTS_FILE=None):
        """
        Train top-down model with chosen method
        """
        logging.error('\nTraining {}'.format(self.opts['model']))
        exp_dir = self.opts['exp_dir']

        # - Set up for training
        train_size = self.data.train_size
        logging.error('\nTrain size: {}, Batch num.: {}, Ite. num: {}'.format(
                                        train_size,
                                        int(train_size/self.opts['batch_size']),
                                        self.opts['it_num']))
        npics = self.opts['plot_num_pics']
        fixed_noise = sample_pz(self.opts, self.pz_params, npics)
        # anchors_ids = np.random.choice(npics, 5, replace=True)
        anchors_ids = [0, 4, 6, 12, 39]

        # - Init all monitoring variables
        Loss, Loss_test, Loss_rec, Loss_rec_test = [], [], [], []
        Divergences, Divergences_test = [], []
        MSE, MSE_test = [], []
        kl = np.zeros(self.opts['zdim'])
        if self.opts['vizu_encSigma']:
            enc_Sigmas = []
        # - Init decay lr and lambda
        decay = 1.
        # decay_steps, decay_rate = int(batches_num * opts['epoch_num'] / 5), 0.95
        decay_steps, decay_rate = 500000, 0.95
        wait, wait_lambda = 0, 0

        # - Load trained model or init variables
        if self.opts['use_trained']:
            if MODEL_PATH is None or WEIGHTS_FILE is None:
                    raise Exception("No model/weights provided")
            else:
                if not tf.gfile.IsDirectory(MODEL_PATH):
                    raise Exception("model doesn't exist")
                WEIGHTS_PATH = os.path.join(MODEL_PATH,'checkpoints', WEIGHTS_FILE)
                if not tf.gfile.Exists(WEIGHTS_FILE+".meta"):
                    raise Exception("weights file doesn't exist")
                self.saver.restore(self.sess, WEIGHTS_FILE)
        else:
            self.sess.run(self.initializer)

        # - Training
        for it in range(self.opts['it_num']):
            # Saver
            if it > 0 and it % self.opts['save_every'] == 0:
                self.saver.save(self.sess,
                                os.path.join(exp_dir, 'checkpoints', 'trained-wae'),
                                global_step=it)
            #####  TRAINING LOOP #####
            it += 1
            _ = self.sess.run(self.opt, feed_dict={
                                self.data.handle: self.train_handle,
                                self.lr_decay: decay,
                                self.obj_fn_coeffs: self.opts['obj_fn_coeffs'],
                                self.is_training: True})
            ##### TESTING LOOP #####
            if it % self.opts['evaluate_every'] == 0:
                logging.error('\nIteration {}/{}'.format(it, self.opts['it_num']))
                # training losses
                feed_dict={self.data.handle: self.train_handle,
                                self.obj_fn_coeffs: self.opts['obj_fn_coeffs'],
                                self.is_training: False}
                [loss, loss_rec, mse, divergences] = self.sess.run([
                                            self.objective,
                                            self.loss_reconstruct,
                                            self.mse,
                                            self.divergences],
                                            feed_dict=feed_dict)
                Loss.append(loss)
                Loss_rec.append(loss_rec)
                MSE.append(mse)
                Divergences.append(divergences)
                # testing losses
                test_it_num = int(self.data.test_size / self.opts['batch_size'])
                loss, loss_rec, mse = 0., 0., 0.
                if type(divergences)==tuple:
                    divergences = np.zeros(len(divergences))
                else:
                    divergences = np.zeros((1,))
                kl = np.zeros(self.opts['zdim'])
                for it_ in range(test_it_num):
                    test_feed_dict={self.data.handle: self.test_handle,
                                    self.obj_fn_coeffs: self.opts['obj_fn_coeffs'],
                                    self.is_training: False}
                    [l, l_rec, m, div, kl_to_prior] = self.sess.run([self.objective,
                                                self.loss_reconstruct,
                                                self.mse,
                                                self.divergences,
                                                self.kl_to_prior],
                                                feed_dict=test_feed_dict)
                    loss += l / test_it_num
                    loss_rec += l_rec / test_it_num
                    mse += m / test_it_num
                    divergences += np.array(div) / test_it_num
                    kl += kl_to_prior / test_it_num
                Loss_test.append(loss)
                Loss_rec_test.append(loss_rec)
                MSE_test.append(mse)
                Divergences_test.append(divergences)

                # Printing various loss values
                debug_str = 'ITER: %d/%d, ' % (it, self.opts['it_num'])
                logging.error(debug_str)
                debug_str = 'TRAIN LOSS=%.3f, TEST LOSS=%.3f' % (Loss[-1],Loss_test[-1])
                logging.error(debug_str)
                if self.opts['model'] == 'BetaVAE':
                    debug_str = 'REC=%.3f, TEST REC=%.3f, beta*KL=%10.3e, beta*TEST KL=%10.3e, \n '  % (
                                                Loss_rec[-1],
                                                Loss_rec_test[-1],
                                                Divergences[-1][0],
                                                Divergences_test[-1][0])
                    logging.error(debug_str)
                elif self.opts['model'] == 'BetaTCVAE':
                    debug_str = 'REC=%.3f, TEST REC=%.3f, b*TC=%10.3e, TEST b*TC=%10.3e, pseudo-KL=%10.3e, TEST pseudo-KL=%10.3e, \n '  % (
                                                Loss_rec[-1],
                                                Loss_rec_test[-1],
                                                Divergences[-1][0],
                                                Divergences_test[-1][0],
                                                Divergences[-1][1],
                                                Divergences_test[-1][1])
                    logging.error(debug_str)
                elif self.opts['model'] == 'FactorVAE':
                    debug_str = 'REC=%.3f, TEST REC=%.3f, g*TC=%10.3e, TEST g*TC=%10.3e, pseudo-KL=%10.3e, TEST pseudo-KL=%10.3e, \n '  % (
                                                Loss_rec[-1],
                                                Loss_rec_test[-1],
                                                Divergences[-1][1],
                                                Divergences_test[-1][1],
                                                Divergences[-1][0],
                                                Divergences_test[-1][0])
                    logging.error(debug_str)
                elif self.opts['model'] == 'WAE':
                    debug_str = 'REC=%.3f, TEST REC=%.3f, l*MMD=%10.3e, l*TEST MMD=%10.3e \n ' % (
                                                Loss_rec[-1],
                                                Loss_rec_test[-1],
                                                Divergences[-1],
                                                Divergences_test[-1])
                    logging.error(debug_str)
                elif self.opts['model'] == 'TCWAE_MWS' or self.opts['model'] == 'TCWAE_GAN':
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
                elif self.opts['model'] == 'TCWAE_MWS_MI' or self.opts['model'] == 'TCWAE_GAN_MI':
                    debug_str = 'TRAIN: REC=%.3f,l1*TC=%10.3e, l2*pseudo-KL=%10.3e, WAE=%10.3e' % (
                                                Loss_rec[-1],
                                                Divergences[-1][0],
                                                Divergences[-1][1],
                                                Divergences[-1][2])
                    logging.error(debug_str)
                    debug_str = 'TEST : REC=%.3f, l1*TC=%10.3e, l2*pseudo-KL=%10.3e, WAE=%10.3e \n ' % (
                                                Loss_rec_test[-1],
                                                Divergences_test[-1][0],
                                                Divergences_test[-1][1],
                                                Divergences_test[-1][2])
                    logging.error(debug_str)
                else:
                    raise NotImplementedError('Model type not recognised')

            ##### Vizu LOOP #####
            if it % self.opts['print_every'] == 0:
                # Plot vizualizations
                # Auto-encoding test images & samples generated by the model
                [reconstructions_vizu, latents_vizu, generations] = self.sess.run(
                                                [self.decoded, self.encoded, self.generated],
                                                feed_dict={self.inputs_img: self.data.data_vizu,
                                                           self.pz_samples: fixed_noise,
                                                           self.is_training: False})
                # - Plotting latent interpolation, and saving
                # # Embeddings
                # if self.opts['vizu_embedded'] and it > 1:
                #     plot_embedded(self.opts, [latents_vizu[:npics]], [fixed_noise],
                #                             self.data.data_vizu,
                #                             exp_dir, 'embedded_it%07d.png' % (it))

                # # Encoded sigma
                # if self.opts['vizu_encSigma'] and it > 1:
                #     plot_encSigma(self.opts,
                #                   enc_Sigmas,
                #                   exp_dir,
                #                   'encSigma_e%04d_mb%05d.png' % (epoch, buff*batches_num+it+1))

                # Encode anchors points and interpolate
                if self.opts['vizu_interpolation']:
                    num_steps = 15

                    enc_var = np.ones(self.opts['zdim'])
                    # crate linespace
                    enc_interpolation = linespace(self.opts, num_steps,  # shape: [nanchors, zdim, nsteps, zdim]
                                            anchors=latents_vizu[anchors_ids],
                                            std=enc_var)
                    enc_interpolation = np.reshape(enc_interpolation, [-1, self.opts['zdim']])
                    # reconstructing
                    dec_interpolation = self.sess.run(self.generated,
                                            feed_dict={self.pz_samples: enc_interpolation,
                                                       self.is_training: False})
                    inter_anchors = np.reshape(dec_interpolation, [-1, self.opts['zdim'], num_steps]+self.data.data_shape)
                    kl_to_prior_sorted = np.argsort(kl)[::-1]
                    plot_interpolation(self.opts, inter_anchors[:,kl_to_prior_sorted],
                                            exp_dir, 'inter_it%07d.png' % (it))

                # Auto-encoding training images
                reconstructions_train = []
                inputs_tr = []
                for _ in range(2):
                    batch =  self.sess.run(self.data.next_element, feed_dict={self.data.handle: self.train_handle}) # Make sure size is correct
                    rec = self.sess.run(self.decoded, feed_dict={self.inputs_img: batch,
                                                           self.is_training: False})

                    inputs_tr.append(batch)
                    reconstructions_train.append(rec)
                inputs_tr = np.concatenate(inputs_tr, axis=0)[:self.opts['plot_num_pics']]
                reconstructions_train = np.concatenate(reconstructions_train, axis=0)[:self.opts['plot_num_pics']]


                # Saving plots
                save_train(self.opts,
                          inputs_tr, self.data.data_vizu,                       # train/vizu images
                          reconstructions_train, reconstructions_vizu,          # reconstructions
                          generations,                                          # model samples
                          Loss, Loss_test,                                      # loss
                          Loss_rec, Loss_rec_test,                              # rec loss
                          MSE, MSE_test,                                        # mse
                          Divergences, Divergences_test,                        # divergence terms
                          exp_dir,                                              # working directory
                          'res_it%07d.png' % (it))                              # filename

            # - Update learning rate if necessary and it
            # if False:
            if self.opts['dataset']=='celebA' or self.opts['dataset']=='3Dchairs':
                if (it+1) % decay_steps == 0:
                    decay = decay_rate ** (int(it / decay_steps))
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
            if self.opts['lambda_schedule'] == 'adaptive':
                if it >= .0 and len(Loss_rec) > 0:
                    if wait_lambda > 1000 * batches_num + 1:
                        # opts['lambda'] = list(2*np.array(opts['lambda']))
                        self.opts['lambda'][-1] = 2*self.opts['lambda'][-1]
                        wae_lambda = self.opts['lambda']
                        logging.error('Lambda updated to %s\n' % wae_lambda)
                        print('')
                        wait_lambda = 0
                    else:
                        wait_lambda += 1

            # - logging
            if (it)%50000 ==0 :
                logging.error('Train it.: {}/{} \n'.format(it,self.opts['it_num']))

        # - Save the final model
        if self.opts['save_final'] and it > 0:
            self.saver.save(self.sess, os.path.join(exp_dir,
                                                'checkpoints',
                                                'trained-{}-final'.format(self.opts['model'])),
                                                global_step=it)

        # - Finale losses & scores
        feed_dict={self.data.handle: self.train_handle,
                        self.obj_fn_coeffs: self.opts['obj_fn_coeffs'],
                        self.is_training: False}
        [loss, loss_rec, mse, divergences] = self.sess.run([
                                    self.objective,
                                    self.loss_reconstruct,
                                    self.mse,
                                    self.divergences],
                                    feed_dict=feed_dict)
        Loss.append(loss)
        Loss_rec.append(loss_rec)
        MSE.append(mse)
        Divergences.append(divergences)
        # Test losses
        loss, loss_rec, mse = 0., 0., 0.
        if type(divergences)==tuple:
            divergences = np.zeros(len(divergences))
        else:
            divergences = np.zeros((1,))
        kl = np.zeros(self.opts['zdim'])
        for it_ in range(test_it_num):
            test_feed_dict={self.data.handle: self.test_handle,
                            self.obj_fn_coeffs: self.opts['obj_fn_coeffs'],
                            self.is_training: False}
            [l, l_rec, m, div, kl_to_prior] = self.sess.run([self.objective,
                                        self.loss_reconstruct,
                                        self.mse,
                                        self.divergences,
                                        self.kl_to_prior],
                                        feed_dict=test_feed_dict)
            loss += l / test_it_num
            loss_rec += l_rec / test_it_num
            mse += m / test_it_num
            divergences += np.array(div) / test_it_num
            kl += kl_to_prior / test_it_num
        Loss_test.append(loss)
        Loss_rec_test.append(loss_rec)
        MSE_test.append(mse)
        Divergences_test.append(divergences)
        # Disentanglment metrics
        if self.opts['true_gen_model']:
            # generating traing set for MIG and SAP
            tr_set_size = 10000
            # tr_set_size = 100
            tr_batch_size = min(1000,tr_set_size)
            tr_batch_num = int(tr_set_size / tr_batch_size)
            tr_codes = np.zeros((tr_set_size, self.opts['zdim']))
            tr_codes_mean = np.zeros((tr_set_size, self.opts['zdim']))
            tr_labels = np.zeros((tr_set_size,len(self.data.factor_sizes)))
            for it_ in range(tr_batch_num):
                # Sample batches of test_data points
                batch_factors = utils.sample_factors(tr_batch_size, self.data.factor_sizes)
                batch_images = self.data.sample_observations_from_factors(self.opts, batch_factors)
                batch_pz_samples = sample_pz(self.opts, self.pz_params, tr_batch_size)
                test_feed_dict = {self.inputs_img: batch_images,
                                  self.pz_samples: batch_pz_samples,
                                  self.is_training: False}
                [z, z_mean] = self.sess.run([self.encoded, self.encoded_mean],
                                                feed_dict=test_feed_dict)
                tr_codes[tr_batch_size*it_:tr_batch_size*(it_+1)] = z
                tr_codes_mean[tr_batch_size*it_:tr_batch_size*(it_+1)] = z_mean
                tr_labels[tr_batch_size*it_:tr_batch_size*(it_+1)] = batch_factors
            # MIG
            MIG=self.compute_mig(tr_codes_mean, tr_labels)
            # generating testing set for SAP
            te_set_size = 5000
            # te_set_size = 50
            te_batch_size = min(1000,te_set_size)
            te_batch_num = int(te_set_size / tr_batch_size)
            te_codes = np.zeros((te_set_size, self.opts['zdim']))
            te_labels = np.zeros((te_set_size,len(self.data.factor_sizes)))
            for it_ in range(te_batch_num):
                # Sample batches of test_data points
                batch_factors = utils.sample_factors(te_batch_size, self.data.factor_sizes)
                batch_images = self.data.sample_observations_from_factors(self.opts, batch_factors)
                batch_pz_samples = sample_pz(self.opts, self.pz_params, te_batch_size)
                test_feed_dict = {self.inputs_img: batch_images,
                                  self.pz_samples: batch_pz_samples,
                                  self.is_training: False}
                [z, z_mean] = self.sess.run([self.encoded, self.encoded_mean],
                                                feed_dict=test_feed_dict)
                te_codes[te_batch_size*it_:te_batch_size*(it_+1)] = z
                te_labels[te_batch_size*it_:te_batch_size*(it_+1)] = batch_factors
            # SAP
            SAP=self.compute_SAP(tr_codes,tr_labels,te_codes,te_labels)
            # FactorVAE
            factorVAE=self.compute_factorVAE(np.vstack((tr_codes,te_codes)))

        # Printing various loss values
        logging.error('Training done.')
        debug_str = 'TRAIN LOSS=%.3f, TEST LOSS=%.3f' % (Loss[-1],Loss_test[-1])
        logging.error(debug_str)
        if self.opts['true_gen_model']:
            debug_str = 'MIG=%.3f, factorVAE=%.3f, SAP=%.3f' % (
                                                MIG,
                                                factorVAE,
                                                SAP)
            logging.error(debug_str)
        if self.opts['model'] == 'BetaVAE':
            debug_str = 'REC=%.3f, TEST REC=%.3f, b*KL=%10.3e, b*TEST KL=%10.3e'  % (
                                                Loss_rec[-1],
                                                Loss_rec_test[-1],
                                                Divergences[-1],
                                                Divergences_test[-1])
            logging.error(debug_str)
        elif self.opts['model'] in ['BetaTCVAE', 'FactorVAE', 'TCWAE_MWS_MI', 'TCWAE_GAN_MI']:
            debug_str = 'REC=%.3f, TEST REC=%.3f, b*TC=%10.3e, TEST b*TC=%10.3e, KL=%10.3e, TEST KL=%10.3e'  % (
                                                Loss_rec[-1],
                                                Loss_rec_test[-1],
                                                Divergences[-1][0],
                                                Divergences_test[-1][0],
                                                Divergences[-1][1],
                                                Divergences_test[-1][1])
            logging.error(debug_str)
        elif self.opts['model'] == 'WAE':
            debug_str = 'REC=%.3f, TEST REC=%.3f, b*MMD=%10.3e, b*TEST MMD=%10.3e' % (
                                                Loss_rec[-1],
                                                Loss_rec_test[-1],
                                                Divergences[-1],
                                                Divergences_test[-1])
            logging.error(debug_str)
        elif self.opts['model'] == 'TCWAE_MWS' or self.opts['model'] == 'TCWAE_GAN':
            debug_str = 'TRAIN: REC=%.3f,l1*TC=%10.3e, l2*DIMWISE=%10.3e, MMD=%10.3e' % (
                                                Loss_rec[-1],
                                                Divergences[-1][0],
                                                Divergences[-1][1],
                                                Divergences[-1][2])
            logging.error(debug_str)
            debug_str = 'TEST : REC=%.3f, l1*TC=%10.3e, l2*DIMWISE=%10.3e, MMD=%10.3e' % (
                                                Loss_rec_test[-1],
                                                Divergences_test[-1][0],
                                                Divergences_test[-1][1],
                                                Divergences_test[-1][2])
            logging.error(debug_str)
        elif self.opts['model'] == 'TCWAE_MWS_MI' or self.opts['model'] == 'TCWAE_GAN_MI':
            debug_str = 'TRAIN: REC=%.3f,l1*TC=%10.3e, l2*pseudo-KL=%10.3e, WAE=%10.3e' % (
                                                Loss_rec[-1],
                                                Divergences[-1][0],
                                                Divergences[-1][1],
                                                Divergences[-1][2])
            logging.error(debug_str)
            debug_str = 'TEST : REC=%.3f, l1*TC=%10.3e, l2*pseudo-KL=%10.3e, WAE=%10.3e \n ' % (
                                                Loss_rec_test[-1],
                                                Divergences_test[-1][0],
                                                Divergences_test[-1][1],
                                                Divergences_test[-1][2])
            logging.error(debug_str)
        else:
            raise NotImplementedError('Model type not recognised')

        # - save training data
        if self.opts['save_train_data']:
            data_dir = 'train_data'
            save_path = os.path.join(exp_dir, data_dir)
            utils.create_dir(save_path)
            name = 'res_train_final'
            np.savez(os.path.join(save_path, name),
                    loss=np.array(Loss[-1]), loss_test=np.array(Loss_test[-1]),
                    loss_rec=np.array(Loss_rec[-1]), loss_rec_test=np.array(Loss_rec_test[-1]),
                    mse = np.array(MSE[-1]), mse_test = np.array(MSE_test[-1]),
                    divergences=np.array(Divergences[-1]), divergences_test=np.array(Divergences_test[-1]),
                    mig=np.array(MIG), factorVAE=np.array(factorVAE), sap=np.array(SAP))

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
        test_size = data.test_size
        batch_size_te = min(test_size,1000)
        batches_num_te = int(test_size/batch_size_te)+1
        # - Init all monitoring variables
        Loss, Loss_rec, MSE = 0., 0., 0.
        Divergences = []
        MIG, factorVAE, SAP = 0., 0., 0.
        real_blurr, blurr, fid_scores = 0., 0., 0.
        if opts['true_gen_model']:
            codes, codes_mean = np.zeros((batches_num_te*batch_size_te,opts['zdim'])), np.zeros((batches_num_te*batch_size_te,opts['zdim']))
            labels = np.zeros((batches_num_te*batch_size_te,len(data.factor_indices)))
        # - Testing loop
        for it_ in range(batches_num_te):
            # Sample batches of data points
            data_ids = np.random.choice(test_size, batch_size_te, replace=True)
            batch_images_test = data.get_batch_img(data_ids, 'test').astype(np.float32)
            batch_pz_samples_test = sample_pz(opts, self.pz_params, batch_size_te)
            test_feed_dict = {self.batch: batch_images_test,
                              self.samples_pz: batch_pz_samples_test,
                              self.obj_fn_coeffs: opts['obj_fn_coeffs'],
                              self.is_training: False}
            [loss, l_rec, mse, divergences, z, z_mean, samples] = self.sess.run([self.objective,
                                             self.loss_reconstruct,
                                             self.mse,
                                             self.divergences,
                                             self.z_samples,
                                             self.z_mean,
                                             self.generated_x],
                                            feed_dict=test_feed_dict)
            Loss += loss / batches_num_te
            Loss_rec += l_rec / batches_num_te
            MSE += mse / batches_num_te
            if len(Divergences)>0:
                Divergences[-1] += np.array(divergences) / batches_num_te
            else:
                Divergences.append(np.array(divergences) / batches_num_te)
            # storing labels and factors
            if opts['true_gen_model']:
                    codes[batch_size_te*it_:batch_size_te*(it_+1)] = z
                    codes_mean[batch_size_te*it_:batch_size_te*(it_+1)] = z_mean
                    labels[batch_size_te*it_:batch_size_te*(it_+1)] = data.get_batch_label(data_ids,'test')[:,data.factor_indices]

        # - Compute disentanglment metrics
        if opts['true_gen_model']:
            MIG.append(self.compute_mig(codes_mean, labels))
            factorVAE.append(self.compute_factorVAE(data, codes))
            SAP.append(self.compute_SAP(data))

        # - Printing various loss values
        if verbose=='high':
            debug_str = 'Testing done.'
            logging.error(debug_str)
            if opts['true_gen_model']:
                debug_str = 'MIG=%.3f, factorVAE=%.3f, SAP=%.3f' % (
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
                debug_str = 'LOSS=%.3f, REC=%.3f, MSE=%.3f, beta*KL=%10.3e \n '  % (
                                            Loss,
                                            Loss_rec,
                                            MSE,
                                            Divergences)
                logging.error(debug_str)
            elif opts['model'] == 'BetaTCVAE':
                debug_str = 'LOSS=%.3f, REC=%.3f, MSE=%.3f, b*TC=%10.3e, KL=%10.3e \n '  % (
                                            Loss,
                                            Loss_rec,
                                            MSE,
                                            Divergences[0],
                                            Divergences[1])
                logging.error(debug_str)
            elif opts['model'] == 'FactorVAE':
                debug_str = 'LOSS=%.3f, REC=%.3f, MSE=%.3f, b*KL=%10.3e, g*TC=%10.3e, \n '  % (
                                            Loss,
                                            Loss_rec,
                                            MSE,
                                            Divergences[0],
                                            Divergences[1])
                logging.error(debug_str)
            elif opts['model'] == 'WAE':
                debug_str = 'LOSS=%.3f, REC=%.3f, MSE=%.3f, b*MMD=%10.3e \n ' % (
                                            Loss,
                                            Loss_rec,
                                            MSE,
                                            Divergences)
                logging.error(debug_str)
            elif opts['model'] == 'TCWAE_MWS' or opts['model'] == 'TCWAE_GAN':
                debug_str = 'LOSS=%.3f, REC=%.3f,l1*TC=%10.3e, MSE=%.3f, l2*DIMWISE=%10.3e, WAE=%10.3e' % (
                                            Loss,
                                            Loss_rec,
                                            MSE,
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
                mse = np.array(MSE),
                divergences=Divergences,
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

        # --- Load trained weights
        self.saver.restore(self.sess, WEIGHTS_PATH)

        # - Set up
        im_shape = datashapes[opts['dataset']]
        if opts['dataset']=='celebA' or opts['dataset']=='3Dchairs':
            num_pics = opts['plot_num_pics']
            num_steps = 7
        else:
            num_pics = opts['evaluate_num_pics']
            num_steps = 10
        fixed_noise = sample_pz(opts, self.pz_params, num_pics)
        # - Auto-encoding test images & samples generated by the model
        [reconstructions, generations] = self.sess.run(
                                        [self.decoded, self.generated],
                                        feed_dict={self.inputs_img: self.data.data_plot[:num_pics],
                                                   self.pz_samples: fixed_noise,
                                                   self.is_training: False})
        # - get kl(q(z_i),p(z_i)) on test data to plot latent traversals
        test_it_num = int(self.data.test_size / self.opts['batch_size'])
        kl_to_prior = np.zeros(self.opts['zdim'])
        for it_ in range(test_it_num):
            test_feed_dict={self.data.handle: self.test_handle,
                            self.obj_fn_coeffs: self.opts['obj_fn_coeffs'],
                            self.is_training: False}
            kl = self.sess.run(self.kl_to_prior, feed_dict=test_feed_dict)
            kl_to_prior += kl / test_it_num

        # - Latent transversals
        # encodings obs
        latents = self.sess.run(self.encoded, feed_dict={
                            self.inputs_img: self.data.data_plot[:opts['evaluate_num_pics']],
                            self.is_training: False})
        enc_var = np.ones(opts['zdim'])
        # create latent linespacel
        if opts['dataset']=='celebA' :
            latent_transversal = linespace(opts, num_steps,  # shape: [nanchors, zdim, nsteps, zdim]
                                    anchors=latents,
                                    std=enc_var)
            # latent_transversal = latent_transversal[:,:,::-1]
        elif opts['dataset']=='3Dchairs':
            latent_transversal = linespace(opts, num_steps,  # shape: [nanchors, zdim, nsteps, zdim]
                                    anchors=latents,
                                    std=enc_var)
            latent_transversal = latent_transversal[:,:,::-1]
        else:
            latent_transversal = linespace(opts, num_steps,  # shape: [nanchors, zdim, nsteps, zdim]
                                    anchors=latents,
                                    std=enc_var)
        # - Reconstructing latent transversals
        obs_transversal = self.sess.run(self.generated,
                                    feed_dict={self.pz_samples: np.reshape(latent_transversal,[-1, opts['zdim']]),
                                               self.is_training: False})
        obs_transversal = np.reshape(obs_transversal, [-1, opts['zdim'], num_steps]+im_shape)
        kl_to_prior_sorted = np.argsort(kl_to_prior)[::-1]
        obs_transversal = obs_transversal[:,kl_to_prior_sorted]
        # obs_transversal = obs_transversal[:,:,::-1]

        # - ploting and saving
        if opts['dataset']=='celebA' or opts['dataset']=='3Dchairs':
            save_test_celeba(opts, self.data.data_plot[:num_pics],
                                        reconstructions,
                                        obs_transversal,
                                        generations,
                                        opts['exp_dir'])
            save_dimwise_traversals(opts, obs_transversal, opts['exp_dir'])
        else:
            save_test_smallnorb(opts, self.data.data_plot[:num_pics],
                                        reconstructions,
                                        obs_transversal,
                                        generations,
                                        opts['exp_dir'])

    def fid_score(self, MODEL_PATH, WEIGHTS_PATH, COMPUTE_DATASET_STASTICS, fid_inputs='samples'):
        """
        Compute FID score
        """

        opts = self.opts

        # --- Load trained weights
        # if not tf.gfile.IsDirectory(MODEL_PATH):
        #     raise Exception("model doesn't exist")
        # WEIGHTS_PATH = os.path.join(MODEL_PATH,'checkpoints',WEIGHTS_FILE)
        # if not tf.gfile.Exists(WEIGHTS_PATH+".meta"):
        #     raise Exception('weights file {} doesnt exist'.format(WEIGHTS_PATH))
        self.saver.restore(self.sess, WEIGHTS_PATH)

        if self.data.dataset == 'celebA':
            compare_dataset_name = 'celeba_stats.npz'
        elif self.data.dataset == '3Dchairs':
            compare_dataset_name = '3Dchairs_stats.npz'
        else:
            assert False, 'FID not implemented for {} dataset.'.format(self.data.dataset)

        # --- logging
        run_name = os.path.split(opts['exp_dir'])[1]
        run_config = run_name[:run_name.find('2020')-1]
        # logging.error('Computing scores for {}'.format(run_config))

        # --- setup
        full_size = self.data.train_size + self.data.test_size
        test_size = 1000 #self.data.test_size
        batch_size = 100
        batch_num = int(test_size/batch_size)
        fid_dir = 'fid'
        stats_path = os.path.join(fid_dir, compare_dataset_name)
        scores = {}

        # --- MSE
        MSE = []
        num_it = int(self.data.test_size / self.opts['batch_size'])
        for it in range(num_it):
            test_feed_dict={self.data.handle: self.test_handle,
                            self.is_training: False}
            mse = self.sess.run(self.mse, feed_dict=test_feed_dict)
            MSE.append(mse)
        mes_mean, mse_std = np.mean(MSE), np.var(MSE)
        debug_str = 'MSE={:.3f}+\-{:.3f} for {}'.format(mes_mean, mse_std, run_config)
        logging.error(debug_str)
        scores['mse'] = {'mean':mes_mean, 'std':mse_std}

        # --- Compute stats on real dataset if needed
        if COMPUTE_DATASET_STASTICS:
            preds_list = []
            for n in range(batch_num):
                batch_id = np.random.randint(full_size, size=batch_size)
                batch = self.data._sample_observations(batch_id)
                # Convert to RGB if needed
                if np.shape(batch)[-1] == 1:
                    batch = np.repeat(np.expand_dims(batch, axis=-1), 3, axis=-1)
                preds_incep = self.inception_sess.run(self.inception_layer,
                              feed_dict={'FID_Inception_Net/ExpandDims:0': batch})
                preds_incep = preds_incep.reshape((batch_size,-1))
                preds_list.append(preds_incep)
            preds_list = np.concatenate(preds_list, axis=0)
            mu = np.mean(preds_list, axis=0)
            sigma = np.cov(preds_list, rowvar=False)
            # saving stats
            np.savez(stats_path, m=mu, s=sigma)
        else:
            if not os.path.isfile(stats_path):
                raise NotImplementedError('No statistics found for given dataset')
            stats = np.load(stats_path)
            mu = stats['m']
            sigma = stats['s']

        # --- Compute stats for reconstructions or samples
        if fid_inputs == 'reconstructions' or fid_inputs=='all':
            preds_list = []
            for n in range(batch_num):
                batch_id = np.random.randint(full_size, size=batch_size)
                batch = self.data._sample_observations(batch_id)
                recons = self.sess.run(self.decoded, feed_dict={
                                        self.inputs_img: batch,
                                        self.is_training: False})
                if np.shape(recons)[-1] == 1:
                    recons = np.repeat(np.expand_dims(recons, axis=-1), 3, axis=-1)
                preds_incep = self.inception_sess.run(self.inception_layer,
                              feed_dict={'FID_Inception_Net/ExpandDims:0': recons})
                preds_incep = preds_incep.reshape((batch_size,-1))
                preds_list.append(preds_incep)
            preds_list = np.concatenate(preds_list, axis=0)
            mu_model = np.mean(preds_list, axis=0)
            sigma_model = np.cov(preds_list, rowvar=False)
            fid_scores = calculate_frechet_distance(mu, sigma, mu_model, sigma_model)
            scores['reconstructions'] = fid_scores
        if fid_inputs == 'samples' or fid_inputs=='all':
            preds_list = []
            for n in range(batch_num):
                batch = sample_pz(opts, self.pz_params, batch_size)
                samples = self.sess.run(self.generated, feed_dict={
                                        self.pz_samples: batch,
                                        self.is_training: False})
                if np.shape(samples)[-1] == 1:
                    samples = np.repeat(np.expand_dims(samples, axis=-1), 3, axis=-1)
                preds_incep = self.inception_sess.run(self.inception_layer,
                              feed_dict={'FID_Inception_Net/ExpandDims:0': samples})
                preds_incep = preds_incep.reshape((batch_size,-1))
                preds_list.append(preds_incep)
            preds_list = np.concatenate(preds_list, axis=0)
            mu_model = np.mean(preds_list, axis=0)
            sigma_model = np.cov(preds_list, rowvar=False)
            fid_scores = calculate_frechet_distance(mu, sigma, mu_model, sigma_model)
            scores['samples'] = fid_scores

        # --- Logging
        for k in scores:
            if k!='mse':
                debug_str = 'FID={:.3f} for {}'.format(scores[k], k)
                logging.error(debug_str)
        fid_res_dir = os.path.join(MODEL_PATH,fid_dir)
        if not tf.io.gfile.isdir(fid_res_dir):
            utils.create_dir(fid_res_dir)
        filename = 'scores'
        np.savez(os.path.join(fid_res_dir,filename),scores)
