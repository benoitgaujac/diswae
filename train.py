"""
Auto-Encoder models
"""
import os
import logging

import numpy as np
import tensorflow as tf

import utils
from sampling_functions import sample_pz, linespace
from plot_functions import save_train
from plot_functions import plot_embedded, plot_encSigma, plot_interpolation
import models
from datahandler import datashapes


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
        elif opts['model'] == 'WAE':
            self.model = models.WAE(opts)
            self.obj_fn_coeffs = self.lmbd
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
        if self.opts['model']=='BetaVAE':
            self.beta = tf.placeholder(tf.float32, name='beta_ph')
        elif self.opts['model']=='WAE':
            self.lmbd = tf.placeholder(tf.float32, name='lambda_ph')
        elif self.opts['model']=='disWAE':
            self.lmbd1 = tf.placeholder(tf.float32, name='lambda1_ph')
            self.lmbd2 = tf.placeholder(tf.float32, name='lambda2_ph')
        else:
            raise NotImplementedError()

    def add_savers(self):
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
        lr = opts['lr']
        opt = self.optimizer(lr, self.lr_decay)
        vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.opt = opt.minimize(loss=self.objective, var_list=vars)

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
        npics = opts['plot_num_pics']
        im_shape = datashapes[opts['dataset']]
        fixed_noise = sample_pz(opts, self.pz_params, npics)
        anchors_ids = np.random.choice(npics, 5, replace=True)

        # - Init all monitoring variables
        Loss, Loss_rec, Loss_rec_test = [], [], []
        Divergences = []
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

            # Training Loop
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

                Loss.append(loss)
                Loss_rec.append(loss_rec)
                Divergences.append(divergences)

                # loss_train_summary = tf.Summary(value=[tf.Summary.Value(tag="loss_train", simple_value=loss)])
                # loss_rec_train_summary = tf.Summary(value=[tf.Summary.Value(tag="loss_rec_train", simple_value=loss_rec)])
                # writer.add_summary(loss_train_summary, it)
                # writer.add_summary(loss_rec_train_summary, it)

                # summary_vals_train = [tf.Summary.Value(tag="loss_train", simple_value=loss),
                #                       tf.Summary.Value(tag="loss_rec_train", simple_value=loss_rec)]
                #
                # if opts['model'] == 'BetaVAE':
                #     summary_vals_train.append(tf.Summary.Value(tag="kl_train", simple_value=divergences))
                # elif opts['model'] == 'WAE':
                #     summary_vals_train.append(tf.Summary.Value(tag="dim_wise_match_train", simple_value=divergences))
                # elif opts['model'] == 'disWAE':
                #     summary_vals_train.append(tf.Summary.Value(tag="dim_wise_match_train", simple_value=divergences[0]))
                #     summary_vals_train.append(tf.Summary.Value(tag="hsic_match_train", simple_value=divergences[1]))
                #     summary_vals_train.append(tf.Summary.Value(tag="wae_match_train", simple_value=divergences[2]))
                # else:
                #     raise NotImplementedError()

                # if opts['vizu_encSigma']:
                #     enc_Sigmas.append(enc_sigmastats)
                #     summary_vals_train.append(tf.Summary.Value(tag="enc_sigma_mean_train",
                #                                                simple_value=enc_sigmastats[0]))
                #     summary_vals_train.append(tf.Summary.Value(tag="enc_sigma_var_train",
                #                                                simple_value=enc_sigmastats[1]))

                # writer.add_summary(tf.Summary(value=summary_vals_train), it + (epoch * batches_num))

                ##### TESTING LOOP #####
                if (counter+1) % opts['print_every'] == 0 or (counter+1) == 100:
                    print("Epoch {}, Iteration {}".format(epoch, it+1))
                    batch_size_te = 200
                    test_size = np.shape(data.test_data)[0]
                    batches_num_te = int(test_size/batch_size_te)
                    # Test loss
                    loss_rec_test = 0.
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
                        l_rec = self.sess.run(self.loss_reconstruct,
                                              feed_dict=test_feed_dict)
                        loss_rec_test += l_rec / batches_num_te
                    Loss_rec_test.append(loss_rec_test)

                    # Auto-encoding test images & samples generated by the model
                    [reconstructions_test, latents_test, generations_test] = self.sess.run(
                                                [self.recon_x,
                                                 self.enc_z,
                                                 self.generated_x],
                                                feed_dict={self.batch: data.test_data[:npics],       # TODO what is this?
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
                                      data.test_labels[:npics],
                                      exp_dir, 'embedded_e%04d_mb%05d.png' % (epoch, it))
                    # Encoded sigma
                    if opts['vizu_encSigma'] and counter > 1:
                        plot_encSigma(opts,
                                      enc_Sigmas,
                                      exp_dir,
                                      'encSigma_e%04d_mb%05d.png' % (epoch, it))
                    # Encode anchors points and interpolate
                    if opts['vizu_interpolation']:
                        logging.error('Latent interpolation..')
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

                    # Printing various loss values
                    debug_str = 'EPOCH: %d/%d, BATCH:%d/%d' % (epoch,
                                                               opts['epoch_num'],
                                                               it + 1,
                                                               batches_num)
                    logging.error(debug_str)
                    debug_str = 'TRAIN LOSS=%.3f' % (Loss[-1])
                    logging.error(debug_str)
                    if opts['model'] == 'BetaVAE':
                        debug_str = 'REC=%.3f, REC TEST=%.3f, beta KL=%10.3e\n '  % (
                            Loss_rec[-1],
                            Loss_rec_test[-1],
                            Divergences[-1])
                    elif opts['model'] == 'WAE':
                        debug_str = 'REC=%.3f, REC TEST=%.3f, lambda MMD=%10.3e\n ' % (
                                                    Loss_rec[-1],
                                                    Loss_rec_test[-1],
                                                    Divergences[-1])
                    elif opts['model'] == 'disWAE':
                        debug_str = 'REC=%.3f, REC TEST=%.3f, lambda1 HSIC=%10.3e, lambda2DIMWISE=%10.3e, WAE=%10.3e\n ' % (
                                                    Loss_rec[-1],
                                                    Loss_rec_test[-1],
                                                    Divergences[-1][1],
                                                    Divergences[-1][0],
                                                    Divergences[-1][2])
                    else:
                        raise NotImplementedError('Model type not recognised')
                    logging.error(debug_str)

                    # Saving plots
                    save_train(opts,
                               data.data[200:200+npics], data.test_data[:npics],        # images
                               reconstructions_train, reconstructions_test,             # reconstructions
                               generations_test,                                        # model samples
                               Loss,                                                    # loss
                               Loss_rec, Loss_rec_test,                                 # rec losses
                               Divergences,                                             # divergence terms
                               exp_dir,                                                 # working directory
                               'res_e%04d_mb%05d.png' % (epoch, it))                    # filename

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
                            wait_lambda += 1

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
                     data_test=data.data[200:200 + npics], data_train=data.test_data[:npics], encoded=latents_test,
                     rec_train=reconstructions_train, rec_test=reconstructions_test[:npics],
                     samples=generations_test, loss=np.array(Loss), loss_rec=np.array(Loss_rec),
                     loss_rec_test=np.array(Loss_rec_test), divergences=np.array(Divergences))
