import numpy as np
import tensorflow as tf
import math

from networks import encoder, decoder, discriminator, dimwise_discriminator
from datahandler import datashapes
from loss_functions import l2_cost, l2sq_cost, l2sq_norm_cost, l1_cost, xentropy_cost
from loss_functions import square_dist_broadcast
import utils

import pdb

class Model(object):

    def __init__(self, opts):
        self.opts = opts

        self.output_dim = datashapes[self.opts['dataset']][:-1] \
                          + [2 * datashapes[self.opts['dataset']][-1], ]

        self.pz_mean = np.zeros(opts['zdim'], dtype='float32')      # TODO don't hardcode this
        self.pz_sigma = np.ones(opts['zdim'], dtype='float32')


    def forward_pass(self, inputs, is_training, reuse=False):

        enc_z, enc_mean, enc_Sigma = encoder(self.opts,
                                             input=inputs,
                                             output_dim=2 * self.opts['zdim'],
                                             scope='encoder',
                                             reuse=reuse,
                                             is_training=is_training)

        dec_x, dec_mean = decoder(self.opts,input=enc_z, output_dim=self.output_dim,
                                             scope='decoder',
                                             reuse=reuse,
                                             is_training=is_training)
        return enc_z, enc_mean, enc_Sigma, dec_x, dec_mean

    def sample_x_from_prior(self, noise):

        sample_x, _ = decoder(self.opts, input=noise, output_dim=self.output_dim,
                                              scope='decoder',
                                              reuse=True,
                                              is_training=False)
        return sample_x

    def dimewise_kl_to_prior(self,  z, z_mean, z_logvar):
      """Estimate dimension-wise KL(q(z_i),p(z_i)) to plot latent traversals.
      """
      # Compute log(q(z(x_j)|x_i)) for every sample in the batch, which is a
      # tensor of size [batch_size, batch_size, num_latents]. In the following
      # comments, [batch_size, batch_size, num_latents] are indexed by [j, i, l].
      log_qz_prob = utils.gaussian_log_density(
          tf.expand_dims(z, 1), tf.expand_dims(z_mean, 0),
          tf.expand_dims(z_logvar, 0))
      # Compute log q(z(x_j)_l) = log(sum_i(q(z(x_j)_l|x_i))
      # + constant) for each sample in the batch, which is a vector of size
      # [batch_size, num_latents].
      log_qz_l = tf.reduce_logsumexp(log_qz_prob, axis=1, keepdims=False)
      # Compute log p(z_l)
      # + constant) where p~N(0,1), for each sample in the batch, which is a vector of size
      # [batch_size, num_latents].
      log_pz_l = - tf.square(z) / 2.
      return tf.reduce_mean(log_qz_l - log_pz_l, axis=0)

    def MSE(self, inputs, reconstructions):
        # compute MSE between inputs and reconstruction
        square_dist = tf.reduce_sum(tf.square(inputs - reconstructions),axis=[1,2,3])
        return tf.reduce_mean(square_dist)


class BetaVAE(Model):

    def __init__(self, opts):
        super().__init__(opts)

    def kl_penalty(self, pz_mean, pz_sigma, encoded_mean, encoded_sigma): # To check implementation
        """
        Compute KL divergence between prior and variational distribution
        """
        kl = encoded_sigma / pz_sigma \
            + tf.square(pz_mean - encoded_mean) / pz_sigma - 1. \
            + tf.math.log(pz_sigma) - tf.math.log(encoded_sigma)
        kl = 0.5 * tf.reduce_sum(kl, axis=-1)
        return tf.reduce_mean(kl)

    def reconstruction_loss(self, labels, logits):
        """
        Compute Xentropy for bernoulli
        """
        labels = tf.layers.flatten(labels)
        cross_entropy = xentropy_cost(labels, logits)
        return tf.reduce_mean(cross_entropy)

    def loss(self, inputs, loss_coeffs, is_training):

        beta = loss_coeffs

        _, enc_mean, enc_Sigma, _, dec_mean = self.forward_pass(inputs=inputs,
                                              is_training=is_training)

        loss_reconstruct = self.reconstruction_loss(inputs, dec_mean)
        kl = self.kl_penalty(self.pz_mean, self.pz_sigma, enc_mean, enc_Sigma)
        divergences = beta * kl

        return loss_reconstruct


class BetaTCVAE(BetaVAE):

    def __init__(self, opts):
        super().__init__(opts)

    def total_correlation(self, z, z_mean, z_logvar):
      """Estimate of total correlation on a batch.
      Based on ICML paper
      """
      # Compute log(q(z(x_j)|x_i)) for every sample in the batch, which is a
      # tensor of size [batch_size, batch_size, num_latents]. In the following
      # comments, [batch_size, batch_size, num_latents] are indexed by [j, i, l].
      log_qz_prob = utils.gaussian_log_density(
          tf.expand_dims(z, 1), tf.expand_dims(z_mean, 0),
          tf.expand_dims(z_logvar, 0))
      # Compute log prod_l q(z(x_j)_l) = sum_l(log(sum_i(q(z(x_j)_l|x_i)))
      # + constant) for each sample in the batch, which is a vector of size
      # [batch_size,].
      log_qz_product = tf.reduce_sum(
          tf.reduce_logsumexp(log_qz_prob, axis=1, keepdims=False),
          axis=1,
          keepdims=False)
      # Compute log(q(z(x_j))) as log(sum_i(q(z(x_j)|x_i))) + constant =
      # log(sum_i(prod_l q(z(x_j)_l|x_i))) + constant.
      log_qz = tf.reduce_logsumexp(
          tf.reduce_sum(log_qz_prob, axis=2, keepdims=False),
          axis=1,
          keepdims=False)
      return tf.reduce_mean(log_qz - log_qz_product)

    def loss(self, inputs, loss_coeffs, is_training):

        beta = loss_coeffs

        enc_z, enc_mean, enc_Sigma, _, dec_mean = self.forward_pass(inputs=inputs,
                                              is_training=is_training)

        loss_reconstruct = self.reconstruction_loss(inputs, dec_mean)
        kl = self.kl_penalty(self.pz_mean, self.pz_sigma, enc_mean, enc_Sigma)
        tc = self.total_correlation(enc_z, enc_mean, tf.math.log(enc_Sigma))
        divergences = (beta*tc, kl-tc)

        return loss_reconstruct, divergences


class FactorVAE(BetaVAE):
    def __init__(self, opts):
        super().__init__(opts)

    def get_discr_pred(self, inputs, reuse=False):
      """Build and get Dsicriminator preds.
      Based on ICML paper
      """
      with tf.variable_scope("discriminator",reuse=reuse):
          logits, probs = discriminator(self.opts, inputs)
          clipped = tf.clip_by_value(probs, 1e-6, 1 - 1e-6)
      return logits, clipped

    def loss(self, inputs, loss_coeffs, is_training):

        gamma = loss_coeffs

        # --- Encoding and reconstructing
        enc_z, enc_mean, enc_Sigma, _, dec_mean = self.forward_pass(inputs=inputs,
                                                is_training=is_training)
        loss_reconstruct = self.reconstruction_loss(inputs, dec_mean)

        # --- Latent regularization
        # - KL reg
        kl = self.kl_penalty(self.pz_mean, self.pz_sigma, enc_mean, enc_Sigma)
        # - shuffling latent codes
        enc_z_shuffle = []
        seed = 456
        for i in range(enc_z.get_shape()[1]):
            enc_z_shuffle.append(tf.gather(enc_z[:, i], tf.random.shuffle(tf.range(tf.shape(enc_z[:, i])[0]))))
        enc_z_shuffle = tf.stack(enc_z_shuffle, axis=-1, name="encoded_shuffled")
        # - Get discriminator preds
        logits_z, probs_z = self.get_discr_pred(
                                inputs=enc_z)
        _, probs_z_shuffle = self.get_discr_pred(
                                inputs=enc_z_shuffle,
                                reuse=True)
        # - TC loss
        tc = tf.reduce_mean(logits_z[:,0] - logits_z[:,1])
        # - Discr loss
        discr_loss = tf.math.log(probs_z[:,0]) + tf.math.log(probs_z_shuffle[:,1])
        self.discr_loss = - 0.5 * tf.reduce_mean(discr_loss)
        divergences = (gamma*tc, kl)

        return loss_reconstruct, divergences


class WAE(BetaVAE):

    def __init__(self, opts):
        super().__init__(opts)

    def mmd_penalty(self, sample_qz, sample_pz):
        opts = self.opts
        sigma2_p = opts['pz_scale'] ** 2
        kernel = opts['mmd_kernel']
        n = utils.get_batch_size(sample_qz)
        n = tf.cast(n, tf.int32)
        nf = tf.cast(n, tf.float32)
        half_size = tf.cast((n * n - n) / 2,tf.int32)

        distances_pz = square_dist_broadcast(sample_pz, sample_pz)
        distances_qz = square_dist_broadcast(sample_qz, sample_qz)
        distances = square_dist_broadcast(sample_qz, sample_pz)

        # distances_pz = self.square_dist(sample_pz, sample_pz)
        # distances_qz = self.square_dist(sample_qz, sample_qz)
        # distances = self.square_dist(sample_qz, sample_pz)

        if opts['mmd_kernel'] == 'RBF':
            # Median heuristic for the sigma^2 of Gaussian kernel
            sigma2_k = tf.nn.top_k(
                tf.reshape(distances, [-1]), half_size).values[half_size - 1]
            sigma2_k += tf.nn.top_k(
                tf.reshape(distances_qz, [-1]), half_size).values[half_size - 1]
            # Maximal heuristic for the sigma^2 of Gaussian kernel
            # sigma2_k = tf.nn.top_k(tf.reshape(distances_qz, [-1]), 1).values[0]
            # sigma2_k += tf.nn.top_k(tf.reshape(distances, [-1]), 1).values[0]
            # sigma2_k = opts['latent_space_dim'] * sigma2_p
            if opts['verbose']:
                sigma2_k = tf.Print(sigma2_k, [sigma2_k], 'Kernel width:')
            res1 = tf.exp( - distances_qz / 2. / sigma2_k)
            res1 += tf.exp( - distances_pz / 2. / sigma2_k)
            res1 = tf.multiply(res1, 1. - tf.eye(n))
            res1 = tf.reduce_sum(res1) / (nf * nf - nf)
            res2 = tf.exp( - distances / 2. / sigma2_k)
            res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
            stat = res1 - res2
        elif opts['mmd_kernel'] == 'IMQ':
            Cbase = 2 * opts['zdim'] * sigma2_p
            stat = 0.
            for scale in [.1, .2, .5, 1., 2., 5., 10.]:
                C = Cbase * scale
                res1 = C / (C + distances_qz)
                res1 += C / (C + distances_pz)
                res1 = tf.multiply(res1, 1. - tf.eye(n))
                res1 = tf.reduce_sum(res1) / (nf * nf - nf)
                res2 = C / (C + distances)
                res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
                stat += res1 - res2
        elif opts['mmd_kernel'] == 'RQ':
            stat = 0.
            for scale in [.1, .2, .5, 1., 2., 5., 10.]:
                res1 = (1. + distances_qz / scale / 2.) ** (-scale)
                res1 += (1. + distances_pz / scale / 2.) ** (-scale)
                res1 = tf.multiply(res1, 1. - tf.eye(n))
                res1 = tf.reduce_sum(res1) / (nf * nf - nf)
                res2 = (1. + distances / scale / 2.) ** (-scale)
                res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
                stat += res1 - res2
        return stat

    def reconstruction_loss(self, x1, x2, logits):
        opts = self.opts
        # Flatten last dim input
        x1 = tf.layers.flatten(x1)
        x2 = tf.layers.flatten(x2)
        # - Compute chosen cost
        if opts['cost'] == 'l2':
            cost = l2_cost(x1, x2)
        elif opts['cost'] == 'l2sq':
            cost = l2sq_cost(x1, x2)
        elif opts['cost'] == 'l2sq_norm':
            cost = l2sq_norm_cost(x1, x2)
        elif opts['cost'] == 'l1':
            cost = l1_cost(x1, x2)
        elif opts['cost'] == 'xent':
            cost = xentropy_cost(x1, logits)
        else:
            assert False, 'Unknown cost function %s' % opts['cost']
        return tf.reduce_mean(cost)

    def loss(self, inputs, loss_coeffs, is_training):

        lmbd = loss_coeffs

        # --- Encoding and reconstructing
        enc_z, _, enc_Sigma, recon_x, dec_mean = self.forward_pass(inputs=inputs,
                                                is_training=is_training)

        loss_reconstruct = self.reconstruction_loss(inputs, recon_x, dec_mean)
        noise = tf.random_normal(shape=tf.shape(enc_z))
        pz_sample = tf.add(self.pz_mean, (noise * self.pz_sigma))
        divergences = lmbd*self.mmd_penalty(enc_z, pz_sample)

        return loss_reconstruct, divergences


class TCWAE_MWS(WAE):

    def __init__(self, opts):
        super().__init__(opts)

    def total_correlation(self, z, z_mean, z_logvar):
      """Estimate of total correlation and dimensionwise on a batch.
      Based on ICML paper
      """
      M = utils.get_batch_size(z)
      N = self.opts['dataset_size']
      # Compute log(q(z(x_j)|x_i)) for every sample in the batch, which is a
      # tensor of size [batch_size, batch_size, num_latents]. In the following
      # comments, [batch_size, batch_size, num_latents] are indexed by [j, i, l].
      log_qz_prob = utils.gaussian_log_density(
          tf.expand_dims(z, 1), tf.expand_dims(z_mean, 0),
          tf.expand_dims(z_logvar, 0))
      # Compute log prod_l q(z(x_j)_l) = sum_l(log(sum_i(q(z(x_j)_l|x_i)))
      # + constant) for each sample in the batch, which is a vector of size
      # [batch_size,].
      log_qz_product = tf.reduce_sum(
          tf.reduce_logsumexp(log_qz_prob, axis=1, keepdims=False) - tf.math.log(N*M),
          axis=1,
          keepdims=False)
      # Compute log(q(z(x_j))) as log(sum_i(q(z(x_j)|x_i))) + constant =
      # log(sum_i(prod_l q(z(x_j)_l|x_i))) + constant.
      log_qz = tf.reduce_logsumexp(
          tf.reduce_sum(log_qz_prob, axis=2, keepdims=False),
          axis=1,
          keepdims=False) - tf.math.log(N*M)
      # Compute log prod_l p(z_l) = sum_l(log(p(z_l)))
      # + constant) where p~N(0,1), for each sample in the batch, which is a vector of size
      # [batch_size,].
      pi = tf.constant(math.pi)
      log_pz_product = tf.reduce_sum(
          -0.5 * (tf.math.log(2*pi) + tf.square(z)),
          axis=1,
          keepdims=False)

      return tf.reduce_mean(log_qz), tf.reduce_mean(log_qz_product), tf.reduce_mean(log_pz_product)

    def loss(self, inputs, loss_coeffs, is_training):

        (lmbd1, lmbd2) = loss_coeffs

        # --- Encoding and reconstructing
        enc_z, enc_mean, enc_Sigma, recon_x, dec_mean = self.forward_pass(inputs=inputs,
                                                is_training=is_training)
        loss_reconstruct = self.reconstruction_loss(inputs, recon_x, dec_mean)

        # --- Latent regularization
        log_qz, log_qz_product, log_pz_product = self.total_correlation(enc_z, enc_mean, tf.math.log(enc_Sigma))
        # - WAE latent reg
        tc = log_qz-log_qz_product
        kl = log_qz_product-log_pz_product
        noise = tf.random_normal(shape=tf.shape(enc_z))
        pz_sample = tf.add(self.pz_mean, (noise * self.pz_sigma))
        wae_match_penalty = self.mmd_penalty(enc_z, pz_sample)
        divergences = (lmbd1*tc, lmbd2*kl, wae_match_penalty)

        return loss_reconstruct, divergences


class TCWAE_MWS_MI(TCWAE_MWS):

    def __init__(self, opts):
        super().__init__(opts)

    def loss(self, inputs, loss_coeffs, is_training):

        (lmbd1, lmbd2) = loss_coeffs

        # --- Encoding and reconstructing
        enc_z, enc_mean, enc_Sigma, recon_x, dec_mean = self.forward_pass(inputs=inputs,
                                                is_training=is_training)
        loss_reconstruct = self.reconstruction_loss(inputs, recon_x, dec_mean)

        # --- Latent regularization
        # tc
        log_qz, log_qz_product, _ = self.total_correlation(enc_z, enc_mean, tf.math.log(enc_Sigma))
        tc = log_qz-log_qz_product
        # kl
        kl = self.kl_penalty(self.pz_mean, self.pz_sigma, enc_mean, enc_Sigma)
        # mmd
        noise = tf.random_normal(shape=tf.shape(enc_z))
        pz_sample = tf.add(self.pz_mean, (noise * self.pz_sigma))
        wae_match_penalty = self.mmd_penalty(enc_z, pz_sample)

        divergences = (lmbd1*tc, lmbd2*(kl-tc), wae_match_penalty)

        return loss_reconstruct, divergences


class TCWAE_GAN(WAE):

    def __init__(self, opts):
        super().__init__(opts)

    def get_discr_pred(self, inputs, scope, reuse=False):
      """Build and get Discriminator preds.
      Based on ICML paper
      """
      with tf.variable_scope('discriminator/' + scope, reuse=reuse):
          logits, probs = discriminator(self.opts, inputs)
          """
          if scope == 'TC':
              logits, probs = discriminator(self.opts, inputs)
          elif scope == 'dimwise':
              logits, probs = dimwise_discriminator(self.opts, inputs)
          else:
              raise Exception('Unknown {} scope for Discriminator'.format(scope))
          """
          clipped = tf.clip_by_value(probs, 1e-6, 1 - 1e-6)
      return logits, clipped

    def loss(self, inputs, loss_coeffs, is_training):

        (lmbd1, lmbd2) = loss_coeffs

        # --- Encoding and reconstructing
        enc_z, enc_mean, enc_Sigma, recon_x, dec_mean = self.forward_pass(inputs=inputs,
                                                is_training=is_training)
        loss_reconstruct = self.reconstruction_loss(inputs, recon_x, dec_mean)

        # --- Latent regularization
        # TC term
        # - shuffling latent codes
        enc_z_shuffle = []
        seed = 456
        for i in range(enc_z.get_shape()[1]):
            enc_z_shuffle.append(tf.gather(enc_z[:, i], tf.random.shuffle(tf.range(tf.shape(enc_z[:, i])[0]))))
        enc_z_shuffle = tf.stack(enc_z_shuffle, axis=-1, name="tc_encoded_shuffled")
        # - Get discriminator preds
        logits_z, probs_z = self.get_discr_pred(
                                inputs=enc_z,
                                scope = 'TC',
                                reuse = False)
        _, probs_z_shuffle = self.get_discr_pred(
                                inputs=enc_z_shuffle,
                                scope = 'TC',
                                reuse=True)
        # - TC loss
        tc = tf.reduce_mean(logits_z[:,0] - logits_z[:,1])
        # - Discr loss
        discr_TC_loss = tf.math.log(probs_z[:,0]) + tf.math.log(probs_z_shuffle[:,1])
        discr_TC_loss = 0.5 * tf.reduce_mean(discr_TC_loss)

        # Dimwise term
        # - shuffling latent codes
        enc_z_shuffle = []
        seed = 892
        for d in range(enc_z.get_shape()[1]):
            enc_z_shuffle.append(tf.gather(enc_z[:, d], tf.random.shuffle(tf.range(tf.shape(enc_z[:, d])[0]))))
        enc_z_shuffle = tf.stack(enc_z_shuffle, axis=-1, name="tc_encoded_shuffled")
        # - sample from prior
        noise = tf.random_normal(shape=tf.shape(enc_z))
        pz_sample = tf.add(self.pz_mean, (noise * self.pz_sigma))
        # - Get discriminator preds
        # estimate kl(prod_d q_Z(z_d), p(z))
        logits_z, probs_z_shuffle = self.get_discr_pred(
                                inputs=enc_z_shuffle,
                                scope = 'dimwise',
                                reuse = False)
        _, probs_z_prior = self.get_discr_pred(
                                inputs=pz_sample,
                                scope = 'dimwise',
                                reuse=True)
        """
        # - dimwise loss
        dimwise = tf.reduce_mean(tf.reduce_sum(logits_z[:,:,0] - logits_z[:,:,1], axis=1))
        # - Discr loss
        discr_dimwise_loss = tf.math.log(probs_z_shuffle[:,:,0]) + tf.math.log(probs_z_prior[:,:,1])
        discr_dimwise_loss = 0.5 * tf.reduce_mean(tf.reduce_sum(discr_dimwise_loss, axis=1))
        """
        # - dimwise loss
        dimwise = tf.reduce_mean(logits_z[:,0] - logits_z[:,1])
        # - Discr loss
        discr_dimwise_loss = tf.math.log(probs_z_shuffle[:,0]) + tf.math.log(probs_z_prior[:,1])
        discr_dimwise_loss = 0.5 * tf.reduce_mean(discr_dimwise_loss)

        # - WAE latent reg
        wae_match_penalty = self.mmd_penalty(enc_z, pz_sample)
        divergences = (lmbd1*tc, lmbd2*dimwise, wae_match_penalty)

        # -- Obj
        self.discr_loss = - (lmbd1*discr_TC_loss + lmbd2*discr_dimwise_loss)

        return loss_reconstruct, divergences


class TCWAE_GAN_MI(WAE):

    def __init__(self, opts):
        super().__init__(opts)

    def get_discr_pred(self, inputs, scope, reuse=False):
      """Build and get Dsicriminator preds.
      Based on ICML paper
      """
      with tf.variable_scope('discriminator/' + scope, reuse=reuse):
          logits, probs = discriminator(self.opts, inputs)
          clipped = tf.clip_by_value(probs, 1e-6, 1 - 1e-6)
      return logits, clipped

    def loss(self, inputs, loss_coeffs, is_training):

        (lmbd1, lmbd2) = loss_coeffs

        # --- Encoding and reconstructing
        enc_z, enc_mean, enc_Sigma, recon_x, dec_mean = self.forward_pass(inputs=inputs,
                                                is_training=is_training)
        loss_reconstruct = self.reconstruction_loss(inputs, recon_x, dec_mean)

        # --- Latent regularization
        # TC term
        # - shuffling latent codes
        enc_z_shuffle = []
        seed = 456
        for i in range(enc_z.get_shape()[1]):
            enc_z_shuffle.append(tf.gather(enc_z[:, i], tf.random.shuffle(tf.range(tf.shape(enc_z[:, i])[0]))))
        enc_z_shuffle = tf.stack(enc_z_shuffle, axis=-1, name="encoded_shuffled")
        # - Get discriminator preds
        logits_z, probs_z = self.get_discr_pred(
                                inputs=enc_z,
                                scope = 'TC',
                                reuse = False)
        _, probs_z_shuffle = self.get_discr_pred(
                                inputs=enc_z_shuffle,
                                scope = 'TC',
                                reuse=True)
        # - TC loss
        tc = tf.reduce_mean(logits_z[:,0] - logits_z[:,1])
        # - Discr loss
        discr_TC_loss = tf.math.log(probs_z[:,0]) + tf.math.log(probs_z_shuffle[:,1])
        discr_TC_loss = 0.5 * tf.reduce_mean(discr_TC_loss)
        # - kl
        kl = self.kl_penalty(self.pz_mean, self.pz_sigma, enc_mean, enc_Sigma)
        # - WAE latent reg
        noise = tf.random_normal(shape=tf.shape(enc_z))
        pz_sample = tf.add(self.pz_mean, (noise * self.pz_sigma))
        wae_match_penalty = self.mmd_penalty(enc_z, pz_sample)
        divergences = (lmbd1*tc, lmbd2*(kl-tc), wae_match_penalty)

        # -- Obj
        self.discr_loss = - lmbd1*discr_TC_loss

        return loss_reconstruct, divergences
