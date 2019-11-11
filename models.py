import numpy as np
import tensorflow as tf
from math import pi

from networks import encoder, decoder
from datahandler import datashapes
from sampling_functions import sample_gaussian
from loss_functions import l2_cost, l2sq_cost, l2sq_norm_cost, l1_cost
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



class BetaVAE(Model):

    def kl_penalty(self, pz_mean, pz_sigma, encoded_mean, encoded_sigma): # To check implementation
        """
        Compute KL divergence between prior and variational distribution
        """
        kl = encoded_sigma / pz_sigma \
            + tf.square(pz_mean - encoded_mean) / pz_sigma - 1. \
            + tf.log(pz_sigma) - tf.log(encoded_sigma)
        kl = 0.5 * tf.reduce_sum(kl, axis=-1)
        return tf.reduce_mean(kl)

    def reconstruction_loss(self, labels, logits): # To check implementation
        """
        Compute Xentropy for bernoulli
        """
        eps = 1e-8
        labels = tf.layers.flatten(labels)
        # cross_entropy = - labels*tf.log(mean_params + eps) + (1. - labels)*tf.log(1. - mean_params - eps)
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        return tf.reduce_mean(tf.reduce_sum(cross_entropy,axis=-1))

    def loss(self, inputs, samples, loss_coeffs, is_training, dropout_rate):

        beta = loss_coeffs

        enc_z, enc_mean, enc_Sigma, recon_x, dec_mean, _ = self.forward_pass(inputs=inputs,
                                                                      is_training=is_training,
                                                                      dropout_rate=dropout_rate)

        loss_reconstruct = self.reconstruction_loss(inputs, dec_mean)
        kl = self.kl_penalty(self.pz_mean, self.pz_sigma, enc_mean, enc_Sigma)
        matching_penalty = beta * kl
        divergences = matching_penalty
        objective = loss_reconstruct + matching_penalty

        # - Enc Sigma stats
        Sigma_tr = tf.reduce_mean(enc_Sigma, axis=-1)
        Smean, Svar = tf.nn.moments(Sigma_tr, axes=[0])
        encSigmas_stats = tf.stack([Smean, Svar], axis=-1)

        return objective, loss_reconstruct, divergences, recon_x, enc_z, encSigmas_stats


class TCBetaVAE(BetaVAE):

    def loss(self, inputs, samples, loss_coeffs, is_training, dropout_rate):

        beta = loss_coeffs

        enc_z, enc_mean, enc_Sigma, recon_x, dec_mean, dec_Sigma = self.forward_pass(inputs=inputs,
                                                                      is_training=is_training,
                                                                      dropout_rate=dropout_rate)

        loss_reconstruct = self.reconstruction_loss(inputs, dec_mean, dec_Sigma)

        raise NotImplementedError()
        kl_mi = self.kl_penalty()
        kl_tc = self.kl_penalty()
        kl_dimwise = self.kl_penalty()
        matching_penalty = - (beta[0]*kl_mi + beta[1]*kl_tc + beta[2]*kl_dimwise)
        divergences = (-beta[0]*kl_mi, -beta[1]*kl_tc, -beta[2]*kl_dimwise)
        objective = loss_reconstruct + matching_penalty

        # - Enc Sigma stats
        Sigma_tr = tf.reduce_mean(enc_Sigma, axis=-1)
        Smean, Svar = tf.nn.moments(Sigma_tr, axes=[0])
        encSigmas_stats = tf.stack([Smean, Svar], axis=-1)

        return objective, loss_reconstruct, divergences, recon_x, enc_z, encSigmas_stats


class FactorVAE(Model):
    def __init__(self, opts):
        super().__init__(opts)

        raise NotImplementedError()


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

    def mmd_penalty(self, opts, sample_qz, sample_pz):
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

    def reconstruction_loss(self, opts, x1, x2):
        # Flatten last dim input
        x1 = tf.layers.flatten(x1)
        x2 = tf.layers.flatten(x2)
        # - Compute chosen cost
        if self.opts['cost'] == 'l2':
            cost = l2_cost(x1, x2)
        elif self.opts['cost'] == 'l2sq':
            cost = l2sq_cost(x1, x2)
        elif self.opts['cost'] == 'l2sq_norm':
            cost = l2sq_norm_cost(x1, x2)
        elif self.opts['cost'] == 'l1':
            cost = l1_cost(x1, x2)
        else:
            assert False, 'Unknown cost function %s' % self.opts['obs_cost']

        return tf.reduce_mean(cost)

    def loss(self, inputs, samples, loss_coeffs, is_training, dropout_rate):

        lmbd = loss_coeffs

        # --- Encoding and reconstructing
        enc_z, enc_mean, enc_Sigma, recon_x, _, _ = self.forward_pass(inputs=inputs,
                                                                                is_training=is_training,
                                                                                dropout_rate=dropout_rate)

        loss_reconstruct = self.reconstruction_loss(self.opts, inputs, recon_x)
        match_penalty = lmbd*self.mmd_penalty(self.opts, enc_z, samples)
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


class disWAE(WAE):

    def __init__(self, opts):
        super().__init__(opts)


    def loss(self, inputs, samples, loss_coeffs, is_training, dropout_rate):

        (lmbd1, lmbd2) = loss_coeffs

        # --- Encoding and reconstructing
        enc_z, enc_mean, enc_Sigma, recon_x, _, _ = self.forward_pass(inputs=inputs,
                                                                                is_training=is_training,
                                                                                dropout_rate=dropout_rate)
        loss_reconstruct = self.reconstruction_loss(self.opts, inputs, recon_x)

        # --- Latent regularization
        shuffle_mask = [tf.constant(np.random.choice(np.arange(self.opts['batch_size']),
                                                     self.opts['batch_size'], False))
                        for _ in range(self.opts['zdim'])]
        shuffled_mean = []
        shuffled_Sigma = []
        for dim in range(self.opts['zdim']):
            shuffled_mean.append(tf.gather(enc_mean[:, dim], shuffle_mask[dim]))
            shuffled_Sigma.append(tf.gather(enc_Sigma[:, dim], shuffle_mask[dim]))

        shuffled_mean = tf.stack(shuffled_mean, axis=-1)
        shuffled_Sigma = tf.stack(shuffled_Sigma, axis=-1)
        p_params = tf.concat((shuffled_mean, shuffled_Sigma), axis=-1)

        shuffled_encoded = sample_gaussian(p_params, 'tensorflow')
        # - Dimension-wise latent reg
        dimension_wise_match_penalty = self.mmd_penalty(self.opts, shuffled_encoded, samples)
        # - Multidim. HSIC
        hsic_match_penalty = self.mmd_penalty(self.opts, enc_z, shuffled_encoded)
        # - WAE latent reg
        wae_match_penalty = self.mmd_penalty(self.opts, enc_z, samples)
        matching_penalty = lmbd1*hsic_match_penalty + lmbd2*dimension_wise_match_penalty
        divergences = (lmbd1*hsic_match_penalty, lmbd2*dimension_wise_match_penalty, wae_match_penalty)
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
