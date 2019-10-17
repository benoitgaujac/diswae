# Copyright 2017 Max Planck Society
# Distributed under the BSD-3 Software license,
# (See accompanying file ./LICENSE.txt or copy at
# https://opensource.org/licenses/BSD-3-Clause)

"""
Wasserstein Auto-Encoder models
"""

import sys
import time
import os
import logging

from math import sqrt, cos, sin, pi
import numpy as np
import tensorflow as tf

import utils
from sampling_functions import sample_pz, sample_gaussian, linespace
from loss_functions import matching_penalty, reconstruction_loss
from plot_functions import save_train, save_latent_interpolation, save_vlae_experiment
from plot_functions import plot_embedded, plot_encSigma, plot_interpolation
from networks import encoder, decoder
from datahandler import datashapes

# Path to inception model and stats for training set
sys.path.append('../TTUR')
sys.path.append('../inception')
import fid
inception_path = '../inception'
inception_model = os.path.join(inception_path, 'classify_image_graph_def.pb')
layername = 'FID_Inception_Net/pool_3:0'

import pdb

class WAE(object):

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
        self.pz_params = np.concatenate([mean,Sigma],axis=0)

        # --- Initialize list container
        encSigmas_stats = []
        decSigmas_stats = []
        pen_enc_sigma, pen_dec_sigma = 0., 0.

        # --- Encoding & decoding
        # - Encoding data points
        enc_mean, enc_Sigma = encoder(self.opts, input=self.points,
                                        output_dim=2*opts['zdim'],
                                        scope='encoder',
                                        reuse=False,
                                        is_training=self.is_training,
                                        dropout_rate=self.dropout_rate)
        if opts['encoder'] == 'det':
            # - deterministic encoder
            self.encoded = enc_mean
        elif opts['encoder'] == 'gauss':
            # - gaussian encoder
            qz_params = tf.concat((enc_mean,enc_Sigma),axis=-1)
            self.encoded = sample_gaussian(opts, qz_params, 'tensorflow')
            # Enc Sigma penalty
            if opts['pen_enc_sigma']:
                pen_enc_sigma += opts['lambda_pen_enc_sigma']*tf.reduce_mean(tf.reduce_sum(tf.abs(tf.log(enc_Sigma)),axis=-1))
            # - Enc Sigma stats
            Sigma_tr = tf.reduce_mean(enc_Sigma,axis=-1)
            Smean, Svar = tf.nn.moments(Sigma_tr,axes=[0])
            Sstats = tf.stack([Smean,Svar],axis=-1)
            encSigmas_stats.append(Sstats)
        else:
            assert False, 'Unknown encoder %s' % opts['encoder']
        # - Decoding encoded points (i.e. reconstruct)
        output_dim = datashapes[opts['dataset']][:-1]+[2*datashapes[opts['dataset']][-1],]
        recon_mean, recon_Sigma = decoder(self.opts, input=self.encoded,
                                        output_dim=output_dim,
                                        scope='decoder',
                                        reuse=False,
                                        is_training=self.is_training,
                                        dropout_rate=self.dropout_rate)
        # - reshaping or resampling reconstructed
        if opts['decoder'] == 'det':
            # - deterministic decoder
            reconstructed = recon_mean
        elif opts['decoder'] == 'gauss':
            # - gaussian decoder
            p_params = tf.concat((recon_mean,recon_Sigma),axis=-1)
            reconstructed = sample_gaussian(opts, p_params, 'tensorflow')
            # - Dec Sigma penalty
            if opts['pen_dec_sigma']:
                pen_dec_sigma += opts['lambda_pen_dec_sigma'] * tf.reduce_mean(tf.reduce_sum(tf.abs(tf.log(recon_Sigma)),axis=-1))
            # Dec Sigma stats
            Sigma_tr = tf.reduce_mean(recon_Sigma,axis=-1)
            Smean, Svar = tf.nn.moments(Sigma_tr,axes=[0])
            Sstats = tf.stack([Smean,Svar],axis=-1)
            decSigmas_stats.append(Sstats)
        else:
            assert False, 'Unknown encoder %s' % opts['decoder']
        if opts['input_normalize_sym']:
            reconstructed = tf.nn.tanh(recon_mean)
        else:
            reconstructed = tf.nn.sigmoid(recon_mean)
        self.reconstructed = tf.reshape(reconstructed,[-1]+datashapes[opts['dataset']])

        # --- reconstruction loss
        self.loss_reconstruct = reconstruction_loss(opts, self.points, reconstructed)

        # --- Sampling from model (only for generation)
        output_dim = datashapes[opts['dataset']][:-1]+[2*datashapes[opts['dataset']][-1],]
        decoded_mean, decoded_Sigma = decoder(self.opts, input=self.samples,
                                output_dim=output_dim,
                                scope='decoder',
                                reuse=True,
                                is_training=self.is_training)
        if opts['decoder'] == 'det':
            decoded = decoded_mean
        elif opts['decoder'] == 'gauss':
            p_params = tf.concat((decoded_mean,decoded_Sigma),axis=-1)
            decoded = sample_gaussian(opts, p_params, 'tensorflow')
        else:
            assert False, 'Unknown decoder %s' % opts['decoder']
        if opts['input_normalize_sym']:
            decoded=tf.nn.tanh(decoded)
        else:
            decoded=tf.nn.sigmoid(decoded)
        self.decoded = tf.reshape(decoded,[-1]+datashapes[opts['dataset']])

        # --- Objectives, penalties, sigma pen, FID
        shuffle_mask = [tf.constant(np.random.choice(np.arange(opts['batch_size']),opts['batch_size'],False)) for i in range(opts['zdim'])]
        shuffled_mean = []
        shuffled_Sigma = []
        for z in range(opts['zdim']):
            shuffled_mean.append(tf.gather(enc_mean[:,z],shuffle_mask[z]))
            shuffled_Sigma.append(tf.gather(enc_Sigma[:,z],shuffle_mask[z]))
        shuffled_mean = tf.stack(shuffled_mean,axis=-1)
        shuffled_Sigma = tf.stack(shuffled_Sigma,axis=-1)
        p_params = tf.concat((shuffled_mean,shuffled_Sigma),axis=-1)
        self.shuffled_encoded = sample_gaussian(opts, p_params, 'tensorflow')
        # - Dimension-wise latent reg
        self.dimension_wise_match_penalty = matching_penalty(opts, self.encoded, self.shuffled_encoded)
        # - Multidim. HSIC
        self.hsic_match_penalty = matching_penalty(opts, self.shuffled_encoded, self.samples)
        # - WAE latent reg
        self.wae_match_penalty = matching_penalty(opts, self.encoded, self.samples)
        # - Compute objs
        self.objective = self.loss_reconstruct \
                         + self.lmbd1 * self.dimension_wise_match_penalty \
                         + self.lmbd2 * self.hsic_match_penalty
        # - Enc Sigma penalty
        if opts['pen_enc_sigma']:
            self.objective += pen_enc_sigma
        # - Dec Sigma penalty
        if opts['pen_dec_sigma']:
            self.objective += pen_dec_sigma
        # - FID score
        if opts['fid']:
            self.blurriness = self.compute_blurriness()
            self.inception_graph = tf.Graph()
            self.inception_sess = tf.Session(graph=self.inception_graph)
            with self.inception_graph.as_default():
                self.create_inception_graph()
            self.inception_layer = self._get_inception_layer()

        # --- Disentangle metrics
        """
        TODO
        # - betaVAE
        self.betaVAE = todo
        # - factorVAE
        self.factorVAE = todo
        # - MIG
        self.mig = todo
        # - DCI
        self.dci = todo
        """

        # --- Optimizers, savers, etc
        self.add_optimizers()
        self.add_savers()
        self.init = tf.global_variables_initializer()

    def add_model_placeholders(self):
        opts = self.opts
        shape = self.data_shape
        self.points = tf.placeholder(tf.float32,
                                        [None] + shape,
                                        name='points_ph')
        self.samples = tf.placeholder(tf.float32,
                                        [None] + [opts['zdim'],],
                                        name='noise_ph')

    def add_training_placeholders(self):
        self.lr_decay = tf.placeholder(tf.float32, name='rate_decay_ph')
        self.is_training = tf.placeholder(tf.bool, name='is_training_ph')
        self.lmbd1 = tf.placeholder(tf.float32, name='lambda1_ph')
        self.lmbd2 = tf.placeholder(tf.float32, name='lambda2_ph')
        self.dropout_rate = tf.placeholder(tf.float32, name='dropout_rate_ph')
        self.batch_size = tf.placeholder(tf.int32, name='batch_size_ph')

    def add_savers(self):
        opts = self.opts
        saver = tf.train.Saver(max_to_keep=10)
        self.saver = saver

    def optimizer(self, lr, decay=1.):
        opts = self.opts
        lr *= decay
        if opts['optimizer'] == 'sgd':
            return tf.train.GradientDescentOptimizer(lr)
        elif opts['optimizer'] == 'adam':
            return tf.train.AdamOptimizer(lr, beta1=opts['adam_beta1'])
        else:
            assert False, 'Unknown optimizer.'

    def add_optimizers(self):
        opts = self.opts
        # WAE optimizer
        lr = opts['lr']
        opt = self.optimizer(lr, self.lr_decay)
        encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                scope='encoder')
        decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                scope='decoder')
        ae_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.wae_opt = opt.minimize(loss=self.objective, var_list=ae_vars)
        # Pretraining optimizer
        if opts['e_pretrain']:
            pre_opt = self.optimizer(0.001)
            self.pre_opt = pre_opt.minimize(loss=self.pre_loss, var_list=encoder_vars)

    def compute_blurriness(self):
        images = self.points
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


    def train(self, data, WEIGHTS_FILE):
        """
        Train top-down model with chosen method
        """

        opts = self.opts
        logging.error('Training WAE')
        work_dir = opts['work_dir']

        # - Init sess and load trained weights if needed
        if opts['use_trained']:
            if not tf.gfile.Exists(WEIGHTS_FILE+".meta"):
                raise Exception("weights file doesn't exist")
            self.saver.restore(self.sess, WEIGHTS_FILE)
        else:
            self.sess.run(self.init)
            if opts['e_pretrain']:
                logging.error('Pretraining the encoder\n')
                self.pretrain_encoder(data)
                print('')

        # - Set up for training
        train_size = data.num_points
        batches_num = int(train_size/opts['batch_size'])
        npics = opts['plot_num_pics']
        im_shape = datashapes[opts['dataset']]
        fixed_noise = sample_pz(opts, self.pz_params, npics)
        anchors_ids = np.random.choice(npics, 5, replace=True)


        if opts['fid']:
            # Load inception mean samples for train set
            trained_stats = os.path.join(inception_path, 'fid_stats.npz')
            # Load trained stats
            f = np.load(trained_stats)
            self.mu_train, self.sigma_train = f['mu'][:], f['sigma'][:]
            f.close()
            # Compute bluriness of real data
            real_blurr = self.sess.run(self.blurriness, feed_dict={
                                                self.points: data.data[:npics]})
            logging.error('Real pictures sharpness = %10.4e' % np.min(real_blurr))
            print('')

        # - Init all monitoring variables
        Loss, Loss_rec, Loss_rec_test = [], [], []
        Loss_hsic, Loss_dim, Loss_wae = [], [], []
        enc_Sigmas, dec_Sigmas = [], []
        mean_blurr, fid_scores = [], [],
        decay, counter = 1., 0
        decay_steps, decay_rate = int(batches_num * opts['epoch_num'] / 5), 0.98
        wait, wait_lambda = 0, 0
        wae_lambda = opts['lambda']
        self.start_time = time.time()
        for epoch in range(opts['epoch_num']):
            # Saver
            if counter>0 and counter % opts['save_every'] == 0:
                self.saver.save(self.sess, os.path.join(
                                                work_dir,'checkpoints',
                                                'trained-wae'),
                                                global_step=counter)
            ##### TRAINING LOOP #####
            for it in range(batches_num):
                # Sample batches of data points and Pz noise
                data_ids = np.random.choice(train_size, opts['batch_size'],
                                                replace=True)
                batch_images = data.data[data_ids].astype(np.float32)
                batch_samples = sample_pz(opts, self.pz_params,
                                                opts['batch_size'])
                # Feeding dictionary
                feed_dict={self.points: batch_images,
                           self.samples: batch_samples,
                           self.lr_decay: decay,
                           self.lmbd1: opts['lambda'][0],
                           self.lmbd2: opts['lambda'][1],
                           self.dropout_rate: opts['dropout_rate'],
                           self.is_training: True}
                # Update encoder and decoder
                [_, loss, loss_rec, loss_hsic, loss_dim, loss_wae] = self.sess.run([
                                                self.wae_opt,
                                                self.objective,
                                                self.loss_reconstruct,
                                                self.hsic_match_penalty,
                                                self.dimension_wise_match_penalty,
                                                self.wae_match_penalty],
                                                feed_dict=feed_dict)

                Loss.append(loss)
                Loss_rec.append(loss_rec)
                Loss_hsic.append(loss_hsic)
                Loss_dim.append(loss_dim)
                Loss_wae.append(loss_wae)
                if opts['vizu_encSigma']:
                    [enc_sigmastats,dec_sigmastats] = self.sess.run(
                                                [self.encSigmas_stats,
                                                self.decSigmas_stats],
                                                feed_dict=feed_dict)
                    enc_Sigmas.append(enc_sigmastats)
                    dec_Sigmas.append(dec_sigmastats)

                ##### TESTING LOOP #####
                if counter % opts['print_every']==0 or counter==100:
                    now = time.time()
                    batch_size_te = 200
                    test_size = np.shape(data.test_data)[0]
                    batches_num_te = int(test_size/batch_size_te)
                    # Test loss
                    loss_rec_test = 0.
                    for it_ in range(batches_num_te):
                        # Sample batches of data points
                        data_ids =  np.random.choice(test_size, batch_size_te,
                                                replace=True)
                        batch_images = data.test_data[data_ids].astype(np.float32)
                        l = self.sess.run(self.loss_reconstruct,
                                                feed_dict={self.points:batch_images,
                                                           self.lmbd1: opts['lambda'][0],
                                                           self.lmbd2: opts['lambda'][1],
                                                           self.dropout_rate: 1.,
                                                           self.is_training:False})
                        loss_rec_test += l / batches_num_te
                    Loss_rec_test.append(loss_rec_test)

                    # Auto-encoding test images & samples generated by the model
                    [reconstructed_test, encoded, samples] = self.sess.run(
                                                #[self.full_reconstructed[-1],
                                                [self.reconstructed,
                                                 self.encoded,
                                                 self.decoded],
                                                feed_dict={self.points:data.test_data[:npics],
                                                           self.samples: fixed_noise,
                                                           self.dropout_rate: 1.,
                                                           self.is_training:False})
                    # Auto-encoding training images
                    reconstructed_train = self.sess.run(self.reconstructed,
                                                feed_dict={self.points:data.data[200:200+npics],
                                                           self.dropout_rate: 1.,
                                                           self.is_training:False})

                    # - Plotting embeddings, Sigma, latent interpolation, FID score and saving
                    # Embeddings
                    if opts['vizu_embedded'] and counter>1:
                        decoded = samples[:-1]
                        decoded = decoded[::-1]
                        decoded.append(fixed_noise)
                        plot_embedded(opts,encoded,decoded, #[fixed_noise,].append(samples)
                                                data.test_labels[:10*npics],
                                                work_dir,'embedded_e%04d_mb%05d.png' % (epoch, it))
                    # Encoded sigma
                    if opts['vizu_encSigma'] and counter>1:
                        plot_encSigma(opts, enc_Sigmas, dec_Sigmas,
                                                work_dir,
                                                'encSigma_e%04d_mb%05d.png' % (epoch, it))
                    # Encode anchors points and interpolate
                    if opts['vizu_interpolation']:
                        logging.error('Latent interpolation..')
                        num_steps = 10
                        # encoded stats
                        enc_mean = np.mean(encoded,axis=0)
                        enc_var = np.mean(np.square(encoded-enc_mean),axis=0)
                        # Selecting anchors
                        data_anchors = data.test_data[anchors_ids]
                        # crate linespace
                        enc_interpolation = linespace(opts, num_steps, # shape: [nanchors, zdim, nsteps, zdim]
                                                    anchors=encoded[anchors_ids],
                                                    std=enc_var)
                        # reconstructing
                        dec_interpolation = self.sess.run(self.decoded,
                                                feed_dict={self.samples: np.reshape(enc_interpolation,[-1,opts['zdim']]),
                                                           self.dropout_rate: 1.,
                                                           self.is_training: False})
                        inter_anchors = np.reshape(dec_interpolation,[-1,opts['zdim'],num_steps]+im_shape)
                        plot_interpolation(opts, inter_anchors, work_dir,
                                                'inter_e%04d_mb%05d.png' % (epoch, it))

                    # FID score
                    if opts['fid']:
                        # Compute FID score
                        gen_blurr = self.sess.run(self.blurriness,
                                                feed_dict={self.points: samples})
                        mean_blurr.append(np.min(gen_blurr))
                        # First convert to RGB
                        if np.shape(flat_samples)[-1] == 1:
                            # We have greyscale
                            flat_samples = self.sess.run(tf.image.grayscale_to_rgb(flat_samples))
                        preds_incep = self.inception_sess.run(self.inception_layer,
                                      feed_dict={'FID_Inception_Net/ExpandDims:0': flat_samples})
                        preds_incep = preds_incep.reshape((npics,-1))
                        mu_gen = np.mean(preds_incep, axis=0)
                        sigma_gen = np.cov(preds_incep, rowvar=False)
                        fid_score = fid.calculate_frechet_distance(mu_gen,
                                                sigma_gen,
                                                self.mu_train,
                                                self.sigma_train,
                                                eps=1e-6)
                        fid_scores.append(fid_score)
                        debug_str = 'FID=%.3f, BLUR=%10.4e' % (
                                                fid_scores[-1],
                                                mean_blurr[-1])
                        logging.error(debug_str)

                    # Printing various loss values
                    debug_str = 'EPOCH: %d/%d, BATCH:%d/%d' % (
                                                epoch + 1, opts['epoch_num'],
                                                it + 1, batches_num)
                    logging.error(debug_str)
                    debug_str = 'TRAIN LOSS=%.3f' % (Loss[-1])
                    logging.error(debug_str)
                    debug_str = 'REC=%.3f, REC TEST=%.3f, HSIC=%10.3e, DIMWISE=%10.3e, WAE=%10.3e\n ' % (
                                                Loss_rec[-1],
                                                Loss_rec_test[-1],
                                                Loss_hsic[-1],
                                                Loss_dim[-1],
                                                Loss_wae[-1])
                    logging.error(debug_str)
                    print('')
                    # Saving plots
                    save_train(opts, data.data[200:200+npics], data.test_data[:npics],  # images
                                     reconstructed_train, reconstructed_test, # reconstructions
                                     samples,  # model samples
                                     Loss,  # loss
                                     Loss_rec, Loss_rec_test,   # rec losses
                                     Loss_hsic, Loss_dim, Loss_wae,  # reg losses
                                     work_dir,  # working directory
                                     'res_e%04d_mb%05d.png' % (epoch, it))  # filename

                # - Update learning rate if necessary and counter
                if counter >= batches_num * opts['epoch_num'] / 5 and counter % decay_steps == 0:
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
                            wait_lambda+=1

                counter += 1

        # - Save the final model
        if opts['save_final'] and epoch > 0:
            self.saver.save(self.sess, os.path.join(work_dir,
                                                'checkpoints',
                                                'trained-wae-final'),
                                                global_step=counter)
        # - save training data
        if opts['save_train_data']:
            data_dir = 'train_data'
            save_path = os.path.join(work_dir,data_dir)
            utils.create_dir(save_path)
            name = 'res_train_final'
            np.savez(os.path.join(save_path,name),
                        data_test=data.data[200:200+npics], data_train=data.test_data[:npics],
                        label_test=data.test_labels[:10*npics],
                        encoded = encoded,
                        rec_train=reconstructed_train, rec_test=reconstructed_test[:npics],
                        samples=samples,
                        loss=np.array(Loss),
                        loss_rec=np.array(Loss_rec), loss_rec_test=np.array(Loss_rec_test),
                        loss_hsci=np.array(Loss_hsic), loss_dim=np.array(Loss_dim), loss_wae=np.array(Loss_wae))


    # def latent_interpolation(self, data, MODEL_PATH, WEIGHTS_FILE):
    #     opts = self.opts
    #
    #     # --- Load trained weights
    #     if not tf.gfile.IsDirectory(MODEL_PATH):
    #         raise Exception("model doesn't exist")
    #     WEIGHTS_PATH = os.path.join(MODEL_PATH,'checkpoints',WEIGHTS_FILE)
    #     if not tf.gfile.Exists(WEIGHTS_PATH+".meta"):
    #         raise Exception("weights file doesn't exist")
    #     self.saver.restore(self.sess, WEIGHTS_PATH)
    #     # Set up
    #     test_size = np.shape(data.test_data)[0]
    #     num_steps = 13 #28
    #     num_anchors = 14 #12
    #     imshape = datashapes[opts['dataset']]
    #
    #     # --- Reconstructions
    #     logging.error('Encoding test images..')
    #     num_pics = 2000
    #     encoded, full_recons = self.sess.run([self.encoded,
    #                             # self.full_reconstructed[-1]],
    #                             self.full_reconstructed],
    #                             feed_dict={self.points:data.test_data[:num_pics],
    #                                        self.dropout_rate: 1.,
    #                                        self.is_training:False})
    #     reconstructed = full_recons[-1]
    #     # data_ids = np.arange(43,43+15,dtype='int32')
    #     data_ids = np.random.choice(num_pics, 30,
    #                                     replace=True)
    #     full_recon = [full_recons[i][data_ids] for i in range(len(full_recons))]
    #     # full_recon = self.sess.run(self.full_reconstructed,
    #     #                        feed_dict={self.points:data.test_data[data_ids],
    #     #                                   self.dropout_rate: 1.,
    #     #                                   self.is_training: False})
    #     full_reconstructed = [data.test_data[data_ids],] + full_recon
    #
    #     if opts['sample_recons']:
    #         if opts['encoder'][0]=='gauss':
    #             data_ids = np.arange(36,37)
    #             sampled_recon = self.sess.run(self.sampled_reconstructed,
    #                                    feed_dict={self.points:data.test_data[data_ids],
    #                                               self.dropout_rate: 1.,
    #                                               self.is_training: False})
    #
    #             sampled_reconstructed = [np.concatenate([data.test_data[data_ids] for i in range(6)]),] + sampled_recon
    #         else:
    #             sampled_reconstructed = None
    #     else:
    #         sampled_reconstructed = None
    #
    #     # --- Encode anchors points and interpolate
    #     logging.error('Anchors interpolation..')
    #     encshape = list(np.shape(encoded[-1])[1:])
    #     #anchors_ids = np.random.choice(num_pics,2*num_anchors,replace=False)
    #     # anchors_ids = np.arange(2*num_anchors,3*num_anchors)
    #     anchors_ids = np.random.choice(num_pics, num_anchors,
    #                                     replace=True)
    #     data_anchors = data.test_data[anchors_ids]
    #     enc_anchors = np.reshape(encoded[-1][anchors_ids],[-1,2]+encshape)
    #     enc_interpolation = linespace(opts, num_steps, anchors=enc_anchors)
    #     num_int = np.shape(enc_interpolation)[1]
    #     if opts['e_nlatents']!=opts['nlatents']:
    #         dec_anchors = self.sess.run(self.anchors_decoded,
    #                                 feed_dict={self.anchors_points: np.reshape(enc_interpolation,[-1,]+encshape),
    #                                            self.dropout_rate: 1.,
    #                                            self.is_training: False})
    #     else:
    #         dec_anchors = self.sess.run(self.decoded[-1],
    #                                 feed_dict={self.samples: np.reshape(enc_interpolation,[-1,]+encshape),
    #                                            self.dropout_rate: 1.,
    #                                            self.is_training: False})
    #     inter_anchors = np.reshape(dec_anchors,[-1,num_int]+imshape)
    #     # adding data
    #     data_anchors = np.reshape(data_anchors,[-1,2]+imshape)
    #     inter_anchors = np.concatenate((np.expand_dims(data_anchors[:,0],axis=1),inter_anchors),axis=1)
    #     inter_anchors = np.concatenate((inter_anchors,np.expand_dims(data_anchors[:,1],axis=1)),axis=1)
    #
    #
    #     if opts['zdim'][-1]==2:
    #         # --- Latent interpolation
    #         logging.error('Latent interpolation..')
    #         if False:
    #             enc_mean = np.mean(encoded[-1],axis=0)
    #             enc_var = np.mean(np.square(encoded[-1]-enc_mean),axis=0)
    #         else:
    #             enc_mean = np.zeros(opts['zdim'][-1], dtype='float32')
    #             enc_var = np.ones(opts['zdim'][-1], dtype='float32')
    #         mins, maxs = enc_mean - 2.*np.sqrt(enc_var), enc_mean + 2.*np.sqrt(enc_var)
    #         x = np.linspace(mins[0], maxs[0], num=num_steps, endpoint=True)
    #         xymin = np.stack([x,mins[1]*np.ones(num_steps)],axis=-1)
    #         xymax = np.stack([x,maxs[1]*np.ones(num_steps)],axis=-1)
    #         latent_anchors = np.stack([xymin,xymax],axis=1)
    #         grid_interpolation = linespace(opts, num_steps,
    #                                 anchors=latent_anchors)
    #         dec_latent = self.sess.run(self.decoded[-1],
    #                                 feed_dict={self.samples: np.reshape(grid_interpolation,[-1,]+list(np.shape(enc_mean))),
    #                                            self.dropout_rate: 1.,
    #                                            self.is_training: False})
    #         inter_latent = np.reshape(dec_latent,[-1,num_steps]+imshape)
    #     else:
    #         inter_latent = None
    #
    #     # --- Samples generation
    #     logging.error('Samples generation..')
    #     num_cols = 15
    #     npics = 7*num_cols #num_cols**2
    #     prior_noise = sample_pz(opts, self.pz_params, npics)
    #     samples = self.sess.run(self.decoded[-1],
    #                            feed_dict={self.samples: prior_noise,
    #                                       self.dropout_rate: 1.,
    #                                       self.is_training: False})
    #     # --- Making & saving plots
    #     logging.error('Saving images..')
    #     save_latent_interpolation(opts, data.test_data[:num_pics],data.test_labels[:num_pics], # data,labels
    #                     encoded, reconstructed[:npics], # encoded, reconstructed points
    #                     full_reconstructed, sampled_reconstructed, # full & sampled recons
    #                     inter_anchors, inter_latent, # anchors and latents interpolation
    #                     samples, # samples
    #                     MODEL_PATH) # working directory
    #
    # def vlae_experiment(self, data, MODEL_PATH, WEIGHTS_FILE):
    #     """
    #     Plot and save different latent interpolation
    #     """
    #
    #     opts = self.opts
    #     num_pics = opts['plot_num_pics']
    #     # num_pics = 16
    #
    #     # --- Sampling fixed noise
    #     fixed_noise = []
    #     for n in range(opts['nlatents']):
    #         mean = np.zeros(opts['zdim'][opts['nlatents']-1-n], dtype='float32')
    #         Sigma = np.ones(opts['zdim'][opts['nlatents']-1-n], dtype='float32')
    #         params = np.concatenate([mean,Sigma],axis=0)
    #         fixed_noise.append(sample_gaussian(opts, params, batch_size=1))
    #
    #     # --- Decoding loop
    #     self.vlae_decoded = []
    #     for m in range(opts['nlatents']):
    #         if m==0:
    #             decoded = tf.convert_to_tensor(sample_pz(opts,self.pz_params,num_pics-1),
    #                                             dtype=tf.float32)
    #             decoded = tf.concat([tf.convert_to_tensor(fixed_noise[0],dtype=tf.float32),decoded],axis=0)
    #         else:
    #             decoded = tf.concat([fixed_noise[0] for i in range(num_pics)],axis=0)
    #         for n in range(opts['nlatents']-1,-1,-1):
    #             # Output dim
    #             if n==0:
    #                 output_dim = datashapes[opts['dataset']][:-1]+[2*datashapes[opts['dataset']][-1],]
    #             else:
    #                 output_dim = [2*opts['zdim'][n-1],]
    #             if opts['d_arch'][n]=='resnet_v2':
    #                 features_dim=self.features_dim[n+1]
    #             else:
    #                 features_dim=self.features_dim[n]
    #             # Decoding
    #             decoded_mean, decoded_Sigma = decoder(opts, input=decoded,
    #                                             archi=opts['d_arch'][n],
    #                                             num_layers=opts['d_nlayers'][n],
    #                                             num_units=opts['d_nfilters'][n],
    #                                             filter_size=opts['filter_size'][n],
    #                                             output_dim=output_dim,
    #                                             features_dim=features_dim,
    #                                             resample=opts['d_resample'][n],
    #                                             last_archi=opts['d_last_archi'][n],
    #                                             scope='decoder/layer_%d' % n,
    #                                             reuse=True,
    #                                             is_training=False)
    #             if opts['decoder'][n] == 'det':
    #                 decoded = decoded_mean
    #             elif opts['decoder'][n] == 'gauss':
    #                 if n==opts['nlatents']-m:
    #                     p_params = tf.concat((decoded_mean,decoded_Sigma),axis=-1)
    #                     decoded = sample_gaussian(opts, p_params, 'tensorflow')
    #                 else:
    #                     decoded =  decoded_mean + tf.multiply(fixed_noise[opts['nlatents']-n],tf.sqrt(1e-10+decoded_Sigma))
    #             else:
    #                 assert False, 'Unknown encoder %s' % opts['decoder'][n]
    #             # reshape and normalize for last decoding
    #             if n==0:
    #                 if opts['input_normalize_sym']:
    #                     decoded=tf.nn.tanh(decoded)
    #                 else:
    #                     decoded=tf.nn.sigmoid(decoded)
    #                 decoded = tf.reshape(decoded,[-1]+datashapes[opts['dataset']])
    #         self.vlae_decoded.append(decoded)
    #
    #     # --- Load trained weights
    #     if not tf.gfile.IsDirectory(MODEL_PATH):
    #         raise Exception("model doesn't exist")
    #     WEIGHTS_PATH = os.path.join(MODEL_PATH,'checkpoints',WEIGHTS_FILE)
    #     if not tf.gfile.Exists(WEIGHTS_PATH+".meta"):
    #         raise Exception("weights file doesn't exist")
    #     self.saver.restore(self.sess, WEIGHTS_PATH)
    #
    #     # --- vlae decoding
    #     decoded = self.sess.run(self.vlae_decoded,feed_dict={})
    #
    #     # --- Making & saving plots
    #     logging.error('Saving images..')
    #     save_vlae_experiment(opts, decoded, MODEL_PATH)
    #
    # def fid_score(self, data, MODEL_PATH, WEIGHTS_FILE):
    #     """
    #     Compute FID score
    #     """
    #
    #     opts = self.opts
    #
    #     # --- Load trained weights
    #     if not tf.gfile.IsDirectory(MODEL_PATH):
    #         raise Exception("model doesn't exist")
    #     WEIGHTS_PATH = os.path.join(MODEL_PATH,'checkpoints',WEIGHTS_FILE)
    #     if not tf.gfile.Exists(WEIGHTS_PATH+".meta"):
    #         raise Exception("weights file doesn't exist")
    #     self.saver.restore(self.sess, WEIGHTS_PATH)
    #
    #     # Setup
    #     batch_size = 1000
    #     batches_num = 1
    #
    #     # Load inception mean samples for train set
    #     trained_stats = os.path.join(inception_path, 'fid_stats.npz')
    #     # Load trained stats
    #     f = np.load(trained_stats)
    #     self.mu_train, self.sigma_train = f['mu'][:], f['sigma'][:]
    #     f.close()
    #     # Compute bluriness of real data
    #     real_blurr = self.sess.run(self.blurriness, feed_dict={
    #                                             self.points: data.test_data[:batch_size]})
    #     real_blurr = np.mean(real_blurr)
    #     # logging.error('Real pictures sharpness = %10.4e' % np.min(real_blurr))
    #     # print('')
    #
    #     # Test loop
    #     now = time.time()
    #     mean_blurr, fid_scores = 0., 0.
    #     for it_ in range(batches_num):
    #         # Samples
    #         noise = sample_pz(opts, self.pz_params, batch_size)
    #         # Sampling
    #         samples = self.sess.run(self.decoded[-1],
    #                                     feed_dict={self.samples: noise,
    #                                     self.is_training:False})
    #         # compute blur
    #         gen_blurr = self.sess.run(self.blurriness,
    #                                     feed_dict={self.points: samples})
    #         mean_blurr+= (np.mean(gen_blurr) / batches_num)
    #         # Compute FID score
    #         # First convert to RGB
    #         if np.shape(samples)[-1] == 1:
    #             # We have greyscale
    #             samples = self.sess.run(tf.image.grayscale_to_rgb(samples))
    #         preds_incep = self.inception_sess.run(self.inception_layer,
    #                       feed_dict={'FID_Inception_Net/ExpandDims:0': samples})
    #         preds_incep = preds_incep.reshape((batch_size,-1))
    #         mu_gen = np.mean(preds_incep, axis=0)
    #         sigma_gen = np.cov(preds_incep, rowvar=False)
    #         fid_score = fid.calculate_frechet_distance(mu_gen,
    #                                     sigma_gen,
    #                                     self.mu_train,
    #                                     self.sigma_train,
    #                                     eps=1e-6)
    #         fid_scores+= (fid_score / batches_num)
    #
    #         # Logging
    #         debug_str = 'FID=%.3f, BLUR=%10.3e, REAL BLUR=%10.3e' % (
    #                                 fid_scores,
    #                                 mean_blurr,
    #                                 real_blurr)
    #         logging.error(debug_str)
    #     name = 'fid'
    #     np.savez(os.path.join(work_dir,name),
    #                 fid=np.array(fid_scores),
    #                 blur=np.array(mean_blurr))
    #
    # def test_losses(self, data, MODEL_PATH, WEIGHTS_FILE):
    #     """
    #     Compute losses
    #     """
    #
    #     opts = self.opts
    #
    #     # --- Load trained weights
    #     if not tf.gfile.IsDirectory(MODEL_PATH):
    #         raise Exception("model doesn't exist")
    #     WEIGHTS_PATH = os.path.join(MODEL_PATH,'checkpoints',WEIGHTS_FILE)
    #     if not tf.gfile.Exists(WEIGHTS_PATH+".meta"):
    #         raise Exception("weights file doesn't exist")
    #     self.saver.restore(self.sess, WEIGHTS_PATH)
    #     work_dir = opts['work_dir']
    #
    #     # Setup
    #     now = time.time()
    #     batch_size_te = 200
    #     test_size = np.shape(data.test_data)[0]
    #     batches_num_te = int(test_size/batch_size_te)
    #     train_size = data.num_points
    #     # Logging stat
    #     debug_str = 'TEST SIZE=%d, BATCH NUM=%d' % (
    #                             test_size,
    #                             batches_num_te)
    #     logging.error(debug_str)
    #
    #     # Test loss
    #     wae_lambda = opts['lambda']
    #     loss, loss_test = 0., 0.
    #     loss_rec, loss_rec_test = 0., 0.
    #     loss_match, loss_match_test = 0., 0.
    #     for it_ in range(batches_num_te):
    #         # Sample batches of test data points
    #         data_ids =  np.random.choice(test_size, batch_size_te,
    #                                 replace=True)
    #         batch_images = data.test_data[data_ids].astype(np.float32)
    #         batch_samples = sample_pz(opts, self.pz_params,
    #                                         batch_size_te)
    #         l, lrec, lmatch = self.sess.run([self.objective,
    #                                             self.loss_reconstruct,
    #                                             self.match_penalty],
    #                             feed_dict={self.points:batch_images,
    #                                             self.samples: batch_samples,
    #                                             self.lmbd: wae_lambda,
    #                                             self.is_training:False})
    #         loss_test += l / batches_num_te
    #         loss_rec_test += lrec / batches_num_te
    #         loss_match_test += lmatch / batches_num_te
    #         # Sample batches of train data points
    #         data_ids =  np.random.choice(train_size, batch_size_te,
    #                                 replace=True)
    #         batch_images = data.data[data_ids].astype(np.float32)
    #         batch_samples = sample_pz(opts, self.pz_params,
    #                                         batch_size_te)
    #         l, lrec, lmatch = self.sess.run([self.objective,
    #                                             self.loss_reconstruct,
    #                                             self.match_penalty],
    #                             feed_dict={self.points:batch_images,
    #                                             self.samples: batch_samples,
    #                                             self.lmbd: wae_lambda,
    #                                             self.is_training:False})
    #         loss += l / batches_num_te
    #         loss_rec += lrec / batches_num_te
    #         loss_match += lmatch / batches_num_te
    #
    #     # Logging
    #     debug_str = 'TRAIN: LOSS=%.3f, REC=%.3f, MATCH=%10.3e' % (
    #                             loss,
    #                             loss_rec,
    #                             loss_match)
    #     logging.error(debug_str)
    #     debug_str = 'TEST: LOSS=%.3f, REC=%.3f, MATCH=%10.3e' % (
    #                             loss_test,
    #                             loss_rec_test,
    #                             loss_match_test)
    #     logging.error(debug_str)
    #
    #     name = 'losses'
    #     np.savez(os.path.join(work_dir,name),
    #                 loss_train=np.array(loss),
    #                 loss_rec_train=np.array(loss_rec),
    #                 loss_match_train=np.array(loss_match),
    #                 loss_test=np.array(loss_test),
    #                 loss_rec_test=np.array(loss_rec_test),
    #                 loss_match_test=np.array(loss_match_test))
