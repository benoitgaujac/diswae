import numpy as np
import tensorflow as tf
import math

from networks import encoder, decoder, discriminator
from datahandler import datashapes
from loss_functions import l2_cost, l2sq_cost, l2sq_norm_cost, l1_cost, xentropy_cost
import utils

import pdb

class Model(object):

    def __init__(self, opts):
        self.opts = opts

        self.output_dim = datashapes[self.opts['dataset']][:-1] \
                          + [2 * datashapes[self.opts['dataset']][-1], ]

        self.pz_mean = np.zeros(opts['zdim'], dtype='float32')      # TODO don't hardcode this
        self.pz_sigma = np.ones(opts['zdim'], dtype='float32')

    def forward_pass(self, inputs, is_training, dropout_rate, reuse=False):

        enc_z, enc_mean, enc_Sigma = encoder(self.opts,
                                             input=inputs,
                                             output_dim=2 * self.opts['zdim'],
                                             scope='encoder',
                                             reuse=reuse,
                                             is_training=is_training,
                                             dropout_rate=dropout_rate)

        dec_x, dec_mean, dec_Sigma = decoder(self.opts,
                                             input=enc_z,
                                             output_dim=self.output_dim,
                                             scope='decoder',
                                             reuse=reuse,
                                             is_training=is_training,
                                             dropout_rate=dropout_rate)
        return enc_z, enc_mean, enc_Sigma, dec_x, dec_mean, dec_Sigma

    def sample_x_from_prior(self, noise):

        sample_x, sample_mean, sample_Sigma = decoder(self.opts,
                                                      input=noise,
                                                      output_dim=self.output_dim,
                                                      scope='decoder',
                                                      reuse=True,
                                                      is_training=False,
                                                      dropout_rate=1.)
        return sample_x         #, sample_mean, sample_Sigma

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


class BetaVAE(Model):

    def __init__(self, opts):
        super().__init__(opts)

    def kl_penalty(self, pz_mean, pz_sigma, encoded_mean, encoded_sigma): # To check implementation
        """
        Compute KL divergence between prior and variational distribution
        """
        kl = encoded_sigma / pz_sigma \
            + tf.square(pz_mean - encoded_mean) / pz_sigma - 1. \
            + tf.log(pz_sigma) - tf.log(encoded_sigma)
        kl = 0.5 * tf.reduce_sum(kl, axis=-1)
        return tf.reduce_mean(kl)

    def reconstruction_loss(self, labels, logits):
        """
        Compute Xentropy for bernoulli
        """
        eps = 1e-8
        labels = tf.layers.flatten(labels)
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        return tf.reduce_mean(tf.reduce_sum(cross_entropy,axis=-1))

    def loss(self, inputs, samples, loss_coeffs, is_training, dropout_rate):

        beta = loss_coeffs

        enc_z, enc_mean, enc_Sigma, recon_x, dec_mean, _ = self.forward_pass(inputs=inputs,
                                                                      is_training=is_training,
                                                                      dropout_rate=dropout_rate)

        loss_reconstruct = self.reconstruction_loss(inputs, dec_mean)
        kl = self.kl_penalty(self.pz_mean, self.pz_sigma, enc_mean, tf.log(enc_Sigma))
        matching_penalty = beta * kl
        divergences = matching_penalty
        objective = loss_reconstruct + matching_penalty

        # - Enc Sigma stats
        Sigma_tr = tf.reduce_mean(enc_Sigma, axis=-1)
        Smean, Svar = tf.nn.moments(Sigma_tr, axes=[0])
        encSigmas_stats = tf.stack([Smean, Svar], axis=-1)

        return objective, loss_reconstruct, divergences, recon_x, enc_z, encSigmas_stats


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

    def loss(self, inputs, samples, loss_coeffs, is_training, dropout_rate):

        beta = loss_coeffs

        enc_z, enc_mean, enc_Sigma, recon_x, dec_mean, _ = self.forward_pass(inputs=inputs,
                                                                      is_training=is_training,
                                                                      dropout_rate=dropout_rate)

        loss_reconstruct = self.reconstruction_loss(inputs, dec_mean)
        kl = self.kl_penalty(self.pz_mean, self.pz_sigma, enc_mean, tf.log(enc_Sigma))
        tc = self.total_correlation(enc_z, enc_mean, tf.log(enc_Sigma))

        matching_penalty = (beta - 1.) * tc + kl
        divergences = (beta*tc, kl-tc)
        objective = loss_reconstruct + matching_penalty

        # - Enc Sigma stats
        Sigma_tr = tf.reduce_mean(enc_Sigma, axis=-1)
        Smean, Svar = tf.nn.moments(Sigma_tr, axes=[0])
        encSigmas_stats = tf.stack([Smean, Svar], axis=-1)

        return objective, loss_reconstruct, divergences, recon_x, enc_z, encSigmas_stats


class FactorVAE(BetaVAE):
    def __init__(self, opts):
        super().__init__(opts)

    def get_discr_pred(self, inputs, reuse=False, is_training=False, dropout_rate=1.):
      """Build and get Dsicriminator preds.
      Based on ICML paper
      """
      with tf.variable_scope("discriminator",reuse=reuse):
          logits, probs = discriminator(self.opts, inputs, is_training, dropout_rate)
          clipped = tf.clip_by_value(probs, 1e-6, 1 - 1e-6)
      return logits, clipped


    def loss(self, inputs, samples, loss_coeffs, is_training, dropout_rate):

        (beta, gamma) = loss_coeffs

        # --- Encoding and reconstructing
        enc_z, enc_mean, enc_Sigma, recon_x, dec_mean, _ = self.forward_pass(inputs=inputs,
                                                                      is_training=is_training,
                                                                      dropout_rate=dropout_rate)
        loss_reconstruct = self.reconstruction_loss(inputs, dec_mean)

        # --- Latent regularization
        # - KL reg
        kl = self.kl_penalty(self.pz_mean, self.pz_sigma, enc_mean, tf.log(enc_Sigma))
        # - shuffling latent codes
        enc_z_shuffle = []
        seed = 456
        for i in range(enc_z.get_shape()[1]):
            enc_z_shuffle.append(tf.gather(enc_z[:, i], tf.random.shuffle(tf.range(tf.shape(enc_z[:, i])[0]))))
        enc_z_shuffle = tf.stack(enc_z_shuffle, axis=-1, name="encoded_shuffled")
        # - Get discriminator preds
        logits_z, probs_z = self.get_discr_pred(
                                inputs=enc_z,
                                is_training=is_training,
                                dropout_rate=dropout_rate)
        _, probs_z_shuffle = self.get_discr_pred(
                                inputs=enc_z_shuffle,
                                reuse=True,
                                is_training=is_training,
                                dropout_rate=dropout_rate)
        # - TC loss
        tc = tf.reduce_mean(logits_z[:, 0] - logits_z[:, 1], axis=0)
        # - Discr loss
        self.discr_loss = tf.add(
                            0.5 * tf.reduce_mean(tf.log(probs_z[:, 0])),
                            0.5 * tf.reduce_mean(tf.log(probs_z_shuffle[:, 1])))
        matching_penalty = beta*kl + gamma*tc
        divergences = (beta*kl, gamma*tc)

        # -- Obj
        objective = loss_reconstruct + matching_penalty

        # - Enc Sigma stats
        Sigma_tr = tf.reduce_mean(enc_Sigma, axis=-1)
        Smean, Svar = tf.nn.moments(Sigma_tr, axes=[0])
        encSigmas_stats = tf.stack([Smean, Svar], axis=-1)

        return objective, loss_reconstruct, divergences, recon_x, enc_z, encSigmas_stats


class WAE(Model):

    def __init__(self, opts):
        super().__init__(opts)

    def square_dist(self, sample_x, sample_y):
        """
        Wrapper to compute square distance
        """
        norms_x = tf.reduce_sum(tf.square(sample_x), axis=-1, keepdims=True)
        norms_y = tf.reduce_sum(tf.square(sample_y), axis=-1, keepdims=True)

        squared_dist = norms_x + tf.transpose(norms_y) \
                        - 2. * tf.matmul(sample_x,sample_y,transpose_b=True)
        return tf.nn.relu(squared_dist)

    def mmd_penalty(self, sample_qz, sample_pz):
        opts = self.opts
        sigma2_p = opts['pz_scale'] ** 2
        kernel = opts['mmd_kernel']
        n = utils.get_batch_size(sample_qz)
        n = tf.cast(n, tf.int32)
        nf = tf.cast(n, tf.float32)
        half_size = tf.cast((n * n - n) / 2,tf.int32)

        distances_pz = self.square_dist(sample_pz, sample_pz)
        distances_qz = self.square_dist(sample_qz, sample_qz)
        distances = self.square_dist(sample_qz, sample_pz)

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
        elif opts['cost'] == 'xentropy':
            cost = xentropy_cost(x1, logits)
        else:
            assert False, 'Unknown cost function %s' % opts['obs_cost']
        return tf.reduce_mean(cost)

    def loss(self, inputs, samples, loss_coeffs, is_training, dropout_rate):

        lmbd = loss_coeffs

        # --- Encoding and reconstructing
        enc_z, enc_mean, enc_Sigma, recon_x, dec_mean, _ = self.forward_pass(inputs=inputs,
                                                                                is_training=is_training,
                                                                                dropout_rate=dropout_rate)

        loss_reconstruct = self.reconstruction_loss(inputs, recon_x, dec_mean)
        match_penalty = lmbd*self.mmd_penalty(enc_z, samples)
        divergences = match_penalty
        objective = loss_reconstruct + match_penalty

        # - Pen Encoded Sigma
        if self.opts['pen_enc_sigma'] and self.opts['encoder'] == 'gauss':
            pen_enc_sigma = self.opts['lambda_pen_enc_sigma'] * tf.reduce_mean(
                tf.reduce_sum(tf.abs(tf.log(enc_Sigma)), axis=-1))
            objective += pen_enc_sigma
        # - Enc Sigma stats
        Sigma_tr = tf.reduce_mean(enc_Sigma, axis=-1)
        Smean, Svar = tf.nn.moments(Sigma_tr, axes=[0])
        encSigmas_stats = tf.stack([Smean, Svar], axis=-1)

        return objective, loss_reconstruct, divergences, recon_x, enc_z, encSigmas_stats


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
          -0.5 * (tf.log(2*pi) + tf.square(z)),
          axis=1,
          keepdims=False)

      return tf.reduce_mean(log_qz), tf.reduce_mean(log_qz_product), tf.reduce_mean(log_pz_product)

    def loss(self, inputs, samples, loss_coeffs, is_training, dropout_rate):

        (lmbd1, lmbd2) = loss_coeffs

        # --- Encoding and reconstructing
        enc_z, enc_mean, enc_Sigma, recon_x, dec_mean, _ = self.forward_pass(inputs=inputs,
                                                                                is_training=is_training,
                                                                                dropout_rate=dropout_rate)
        loss_reconstruct = self.reconstruction_loss(inputs, recon_x, dec_mean)

        # --- Latent regularization
        log_qz, log_qz_product, log_pz_product = self.total_correlation(enc_z, enc_mean, tf.log(enc_Sigma))
        # - WAE latent reg
        tc = log_qz-log_qz_product
        kl = log_qz_product-log_pz_product
        matching_penalty = lmbd1*tc + lmbd2*kl
        # matching_penalty = lmbd1*log_qz + (lmbd2-lmbd1)*log_qz_product + lmbd2*log_pz_product
        wae_match_penalty = self.mmd_penalty(enc_z, samples)
        divergences = (lmbd1*tc, lmbd2*kl, wae_match_penalty)

        # -- Obj
        objective = loss_reconstruct + matching_penalty

        # - Pen Encoded Sigma
        if self.opts['pen_enc_sigma'] and self.opts['encoder'] == 'gauss':
            pen_enc_sigma = self.opts['lambda_pen_enc_sigma'] * tf.reduce_mean(
                tf.reduce_sum(tf.abs(tf.log(enc_Sigma)), axis=-1))
            objective += pen_enc_sigma
        # - Enc Sigma stats
        Sigma_tr = tf.reduce_mean(enc_Sigma, axis=-1)
        Smean, Svar = tf.nn.moments(Sigma_tr, axes=[0])
        encSigmas_stats = tf.stack([Smean, Svar], axis=-1)

        return objective, loss_reconstruct, divergences, recon_x, enc_z, encSigmas_stats


class TCWAE_GAN(WAE):

    def __init__(self, opts):
        super().__init__(opts)

    def get_discr_pred(self, inputs, scope, reuse=False, is_training=False, dropout_rate=1.):
      """Build and get Dsicriminator preds.
      Based on ICML paper
      """
      with tf.variable_scope('discriminator/' + scope, reuse=reuse):
          logits, probs = discriminator(self.opts, inputs, is_training, dropout_rate)
          clipped = tf.clip_by_value(probs, 1e-6, 1 - 1e-6)
      return logits, clipped

    def loss(self, inputs, samples, loss_coeffs, is_training, dropout_rate):

        (lmbd1, lmbd2) = loss_coeffs

        # --- Encoding and reconstructing
        enc_z, enc_mean, enc_Sigma, recon_x, dec_mean, _ = self.forward_pass(inputs=inputs,
                                                                                is_training=is_training,
                                                                                dropout_rate=dropout_rate)
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
                                reuse = False,
                                is_training=is_training,
                                dropout_rate=dropout_rate)
        _, probs_z_shuffle = self.get_discr_pred(
                                inputs=enc_z_shuffle,
                                scope = 'TC',
                                reuse=True,
                                is_training=is_training,
                                dropout_rate=dropout_rate)
        # - TC loss
        tc = tf.reduce_mean(logits_z[:, 0] - logits_z[:, 1], axis=0)
        # - Discr loss
        discr_TC_loss = tf.add(
                            0.5 * tf.reduce_mean(tf.log(probs_z[:, 0])),
                            0.5 * tf.reduce_mean(tf.log(probs_z_shuffle[:, 1])))
        # Dimwise term
        # - shuffling latent codes
        enc_z_shuffle = []
        seed = 892
        for d in range(enc_z.get_shape()[1]):
            enc_z_shuffle.append(tf.gather(enc_z[:, d], tf.random.shuffle(tf.range(tf.shape(enc_z[:, d])[0]))))
        enc_z_shuffle = tf.stack(enc_z_shuffle, axis=-1, name="encoded_shuffled")
        # - Get discriminator preds
        if True:
            # estimate kl(prod_d q_Z(z_d), p(z)), no weight sharring
            logits_z, probs_z_shuffle = self.get_discr_pred(
                                    inputs=enc_z_shuffle,
                                    scope = 'dimwise',
                                    reuse = False,
                                    is_training=is_training,
                                    dropout_rate=dropout_rate)
            _, probs_z_prior = self.get_discr_pred(
                                    inputs=samples,
                                    scope = 'dimwise',
                                    reuse=True,
                                    is_training=is_training,
                                    dropout_rate=dropout_rate)
            # - dimwise loss
            dimwise = tf.reduce_mean(logits_z[:, 0] - logits_z[:, 1], axis=0)
            # - Discr loss
            discr_dimwise_loss = tf.add(
                                0.5 * tf.reduce_mean(tf.log(probs_z_shuffle[:, 0])),
                                0.5 * tf.reduce_mean(tf.log(probs_z_prior[:, 1])))
        else:
            # estimate kl(prod_d q_Z(z_d), p(Z)) = sum_d kl(q_Z(z_d), p(z_d)), weight sharring
            ldimwise, ldiscr_dimwise_loss = [], []
            reuse = False
            for d in range(enc_z.get_shape()[1]):
                logits_z, probs_z_shuffle = self.get_discr_pred(
                                        inputs=tf.expand_dims(enc_z_shuffle[:,d],axis=-1),
                                        scope = 'dimwise',
                                        reuse = reuse,
                                        is_training=is_training,
                                        dropout_rate=dropout_rate)
                reuse = True
                _, probs_z_prior = self.get_discr_pred(
                                        inputs=tf.expand_dims(samples[:,d],axis=-1),
                                        scope = 'dimwise',
                                        reuse=reuse,
                                        is_training=is_training,
                                        dropout_rate=dropout_rate)
                # - dimwise loss
                ldimwise.append(tf.reduce_mean(logits_z[:, 0] - logits_z[:, 1], axis=0))
                # - Discr loss
                ldiscr_dimwise_loss.append(tf.add(
                                    0.5 * tf.reduce_mean(tf.log(probs_z_shuffle[:, 0])),
                                    0.5 * tf.reduce_mean(tf.log(probs_z_prior[:, 1]))))
            dimwise = tf.reduce_sum(tf.stack(ldimwise))
            discr_dimwise_loss = tf.reduce_sum(tf.stack(ldiscr_dimwise_loss))
        # - WAE latent reg
        matching_penalty = lmbd1*tc + lmbd2*dimwise
        wae_match_penalty = self.mmd_penalty(enc_z, samples)
        divergences = (lmbd1*tc, lmbd2*dimwise, wae_match_penalty)

        # -- Obj
        objective = loss_reconstruct + matching_penalty
        self.discr_loss = lmbd1*discr_TC_loss + lmbd2*discr_dimwise_loss

        # - Pen Encoded Sigma
        if self.opts['pen_enc_sigma'] and self.opts['encoder'] == 'gauss':
            pen_enc_sigma = self.opts['lambda_pen_enc_sigma'] * tf.reduce_mean(
                tf.reduce_sum(tf.abs(tf.log(enc_Sigma)), axis=-1))
            objective += pen_enc_sigma
        # - Enc Sigma stats
        Sigma_tr = tf.reduce_mean(enc_Sigma, axis=-1)
        Smean, Svar = tf.nn.moments(Sigma_tr, axes=[0])
        encSigmas_stats = tf.stack([Smean, Svar], axis=-1)

        return objective, loss_reconstruct, divergences, recon_x, enc_z, encSigmas_stats


class disWAE(WAE):

    def __init__(self, opts):
        super().__init__(opts)

    def individual_square_dist(self, sample_x, sample_y):
        """
        Wrapper to compute per-component (individual) square distance
        sample_x:   [batch,zdim]
        out:        [batch,zdim,batch], out[k,j,l]=||sample_x[k,j]-sample_y[l,j]||_2^2
        """
        norms_x = tf.square(sample_x)
        norms_y = tf.square(sample_y)

        sample_x = tf.expand_dims(sample_x,-1)
        sample_y = tf.expand_dims(sample_y,-1)
        squared_dist = tf.expand_dims(norms_x,-1) + tf.transpose(norms_y) \
                        - 2. * sample_x * tf.transpose(sample_y)
        return tf.nn.relu(squared_dist)

    def dhsic_penalty(self, samples):
        """
        Compute dHSIC V-statistic estimator (Alg.1 of Pfister & Al.)
        """

        opts = self.opts
        sigma2_p = opts['pz_scale'] ** 2
        kernel = opts['mmd_kernel']
        n = utils.get_batch_size(samples)
        n = tf.cast(n, tf.int32)
        nf = tf.cast(n, tf.float32)
        half_size = tf.cast((n * n - n) / 2,tf.int32)

        distances = self.individual_square_dist(samples, samples)

        # Generating gram matrix of 1d kernel
        if opts['mmd_kernel'] == 'RBF':
            # Median heuristic for the sigma^2 of Gaussian kernel
            sigma2_k = tf.nn.top_k(
                tf.reshape(distances, [-1]), half_size).values[half_size - 1]
            # Maximal heuristic for the sigma^2 of Gaussian kernel
            # sigma2_k = tf.nn.top_k(tf.reshape(distances_qz, [-1]), 1).values[0]
            # sigma2_k += tf.nn.top_k(tf.reshape(distances, [-1]), 1).values[0]
            # sigma2_k = opts['latent_space_dim'] * sigma2_p
            if opts['verbose']:
                sigma2_k = tf.Print(sigma2_k, [sigma2_k], 'Kernel width:')
            K = tf.exp( - distances / 2. / sigma2_k)
        elif opts['mmd_kernel'] == 'IMQ':
            Cbase = 2 * opts['zdim'] * sigma2_p
            K = tf.zeros([n,opts['zdim'],n])
            for scale in [.1, .2, .5, 1., 2., 5., 10.]:
                C = Cbase * scale
                K += C / (C + distances)
        elif opts['mmd_kernel'] == 'RQ':
            K = tf.zeros([n,opts['zdim'],n])
            for scale in [.1, .2, .5, 1., 2., 5., 10.]:
                K += (1. + distances / scale / 2.) ** (-scale)
        # dHSIC V-estimator
        res1 = tf.reduce_sum(tf.reduce_prod(K,axis=1)) / nf*nf
        res2 = tf.reduce_prod(tf.reduce_sum(K,axis=[1,2]) / (nf*nf))
        res3 = tf.reduce_sum(tf.reduce_prod(tf.reduce_sum(K, axis=1) / nf, axis=0) / nf)
        return res1 + res2 - 2. * res3

    def loss(self, inputs, samples, loss_coeffs, is_training, dropout_rate):

        (lmbd1, lmbd2) = loss_coeffs

        # --- Encoding and reconstructing
        enc_z, enc_mean, enc_Sigma, recon_x, dec_mean, _ = self.forward_pass(inputs=inputs,
                                                                                is_training=is_training,
                                                                                dropout_rate=dropout_rate)
        loss_reconstruct = self.reconstruction_loss(inputs, recon_x, dec_mean)

        # --- Latent regularization
        # - shuffling latent codes
        shuffle_encoded = []
        seed = 456
        for i in range(enc_z.get_shape()[1]):
            shuffle_encoded.append(tf.gather(enc_z[:, i], tf.random.shuffle(tf.range(tf.shape(enc_z[:, i])[0]))))
        shuffled_encoded = tf.stack(shuffle_encoded, axis=-1, name="encoded_shuffled")
        # - Dimension-wise latent reg
        dimension_wise_match_penalty = self.mmd_penalty(shuffled_encoded, samples)
        # - Multidim. HSIC
        dhsic_match_penalty = self.mmd_penalty(enc_z, shuffled_encoded)
        # dhsic_match_penalty = self.dhsic_penalty(enc_z)
        # self.dhsic_mmd_penalty = lmbd1*self.mmd_penalty(enc_z, shuffled_encoded)
        # - WAE latent reg
        wae_match_penalty = self.mmd_penalty(enc_z, samples)
        matching_penalty = lmbd1*dhsic_match_penalty + lmbd2*dimension_wise_match_penalty
        divergences = (lmbd1*dhsic_match_penalty, lmbd2*dimension_wise_match_penalty, wae_match_penalty)

        # -- Obj
        objective = loss_reconstruct + matching_penalty

        # - Pen Encoded Sigma
        if self.opts['pen_enc_sigma'] and self.opts['encoder'] == 'gauss':
            pen_enc_sigma = self.opts['lambda_pen_enc_sigma'] * tf.reduce_mean(
                tf.reduce_sum(tf.abs(tf.log(enc_Sigma)), axis=-1))
            objective += pen_enc_sigma
        # - Enc Sigma stats
        Sigma_tr = tf.reduce_mean(enc_Sigma, axis=-1)
        Smean, Svar = tf.nn.moments(Sigma_tr, axes=[0])
        encSigmas_stats = tf.stack([Smean, Svar], axis=-1)

        return objective, loss_reconstruct, divergences, recon_x, enc_z, encSigmas_stats
