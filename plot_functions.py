import os
import re
import pdb

from math import sqrt
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import utils


def save_train(opts, data_train, data_test,
                     rec_train, rec_test,
                     samples,
                     loss, loss_test,
                     loss_rec, loss_rec_test,
                     betaVAE, mig, factorVAE, SAP,
                     loss_match, loss_match_test,
                     exp_dir,
                     filename):

    """ Generates and saves the plot of the following layout:
        img1 | img2 | img3
        img4 | img5 | img6

        img1    -   test reconstructions
        img2    -   train reconstructions
        img3    -   samples
        img4    -   loss curves
        img5    -   split loss curves
        img6    -   dis. metrics curves

    """
    num_pics = opts['plot_num_pics']
    num_cols = opts['plot_num_cols']
    assert num_pics % num_cols == 0
    assert num_pics % 2 == 0
    greyscale = data_train.shape[-1] == 1

    if opts['input_normalize_sym']:
        data_train = data_train / 2. + 0.5
        data_test = data_test / 2. + 0.5
        if opts['model']=='TCWAE_MWS' and opts['model']=='TCWAE_GAN' and opts['model']=='WAE':
            rec_train = rec_train / 2. + 0.5
            rec_test = rec_test / 2. + 0.5
            samples = samples / 2. + 0.5

    images = []
    ### Reconstruction plots
    for pair in [(data_train, rec_train),
                 (data_test, rec_test[:num_pics])]:
        # Arrange pics and reconstructions in a proper way
        sample, recon = pair
        assert len(sample) == num_pics
        assert len(sample) == len(recon)
        pics = []
        merged = np.vstack([recon, sample])
        r_ptr = 0
        w_ptr = 0
        for _ in range(int(num_pics / 2)):
            merged[w_ptr] = sample[r_ptr]
            merged[w_ptr + 1] = recon[r_ptr]
            r_ptr += 1
            w_ptr += 2
        for idx in range(num_pics):
            if greyscale:
                pics.append(1. - merged[idx, :, :, :])
            else:
                pics.append(merged[idx, :, :, :])
        # Figuring out a layout
        pics = np.array(pics)
        image = np.concatenate(np.split(pics, num_cols), axis=2)
        image = np.concatenate(image, axis=0)
        images.append(image)

    ### Sample plots
    assert len(samples) == num_pics
    pics = []
    for idx in range(num_pics):
        if greyscale:
            pics.append(1. - samples[idx, :, :, :])
        else:
            pics.append(samples[idx, :, :, :])
    # Figuring out a layout
    pics = np.array(pics)
    image = np.concatenate(np.split(pics, num_cols), axis=2)
    image = np.concatenate(image, axis=0)
    images.append(image)

    img1, img2, img3 = images

    # Creating a pyplot fig
    dpi = 100
    height_pic = img1.shape[0]
    width_pic = img1.shape[1]
    fig_height = 4 * 2*height_pic / float(dpi)
    fig_width = 6 * 2*width_pic / float(dpi)
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = matplotlib.gridspec.GridSpec(2, 3)

    # Filling in separate parts of the plot

    # First samples and reconstructions
    for img, (gi, gj, title) in zip([img1, img2, img3],
                             [(0, 0, 'Train reconstruction'),
                              (0, 1, 'Test reconstruction'),
                              (0, 2, 'Generated samples')]):
        plt.subplot(gs[gi, gj])
        if greyscale:
            image = img[:, :, 0]
            # in Greys higher values correspond to darker colors
            ax = plt.imshow(image, cmap='Greys',
                            interpolation='none', vmin=0., vmax=1.)
        else:
            ax = plt.imshow(img, interpolation='none', vmin=0., vmax=1.)
        ax = plt.subplot(gs[gi, gj])
        plt.text(0.47, 1., title,
                 ha="center", va="bottom", size=20, transform=ax.transAxes)
        # Removing ticks
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])
        ax.axes.set_xlim([0, width_pic])
        ax.axes.set_ylim([height_pic, 0])
        ax.axes.set_aspect(1)

    ### The loss curves
    ax = plt.subplot(gs[1, 0])
    size_filter = min(int(len(loss_test)/2),5)
    # Obj
    # y = np.convolve(loss_test, np.ones((size_filter,))/size_filter, mode='valid')
    y = loss_test
    total_num = len(y)
    x_step = max(int(total_num / 200), 1)
    x = np.arange(1, len(y) + 1, x_step)
    y = np.log(y[::x_step])
    plt.plot(x, y, linewidth=4, color='black', label='loss test')
    # y = np.convolve(loss, np.ones((size_filter,))/size_filter, mode='valid')
    y = loss
    y = np.log(y[::x_step])
    plt.plot(x, y, linewidth=2, color='black', linestyle='--',label='loss')
    plt.grid(axis='y')
    plt.legend(loc='upper right')
    plt.text(0.47, 1., 'Log Loss curves', ha="center", va="bottom",
                                size=20, transform=ax.transAxes)

    ### The loss curves
    base = plt.cm.get_cmap('tab10')
    color_list = base(np.linspace(0, 1, 6))
    ax = plt.subplot(gs[1, 1])
    losses, losses_test = [], []
    # Test
    # y = np.convolve(loss_rec_test, np.ones((size_filter,))/size_filter, mode='valid')
    y = loss_rec_test
    y = np.log(y[::x_step])
    losses_test.append(list(y))
    if opts['model'] == 'BetaVAE':
        # y = np.convolve(loss_match_test, np.ones((size_filter,))/size_filter, mode='valid')
        y = loss_match_test
        y = np.log(y[::x_step])
        losses_test.append(list(y))
        labels = ['rec',r'$\beta$KL']
    elif opts['model'] == 'BetaTCVAE':
        for l in zip(*loss_match_test):
            # y = np.convolve(l, np.ones((size_filter,))/size_filter, mode='valid')
            y = l
            y = np.log(y[::x_step])
            losses_test.append(list(y))
        labels = ['rec',r'$\beta$TC', 'KL']
    elif opts['model'] == 'FactorVAE':
        for l in zip(*loss_match_test):
            # y = np.convolve(l, np.ones((size_filter,))/size_filter, mode='valid')
            y = l
            y = np.log(y[::x_step])
            losses_test.append(list(y))
        labels = ['rec',r'$\beta$KL', r'$\gamma$TC']
    elif opts['model'] == 'WAE':
        # y = np.convolve(loss_match_test, np.ones((size_filter,))/size_filter, mode='valid')
        y = loss_match_test
        y = np.log(np.abs(y[::x_step]))
        losses_test.append(list(y))
        labels = ['rec',r'$\lambda$|mmd|']
    elif opts['model'] == 'disWAE':
        for l in zip(*loss_match_test):
            # y = np.convolve(l, np.ones((size_filter,))/size_filter, mode='valid')
            y = l
            y = np.log(np.abs(y[::x_step]))
            losses_test.append(list(y))
        labels = ['rec', r"$\lambda_1$|hsci|",r"$\lambda_2$|dimwise|",'|wae|']
    elif opts['model'] == 'TCWAE_MWS' or opts['model'] == 'TCWAE_GAN':
        for l in zip(*loss_match_test):
            # y = np.convolve(l, np.ones((size_filter,))/size_filter, mode='valid')
            y = l
            y = np.log(np.abs(y[::x_step]))
            losses_test.append(list(y))
        labels = ['rec', r"$\lambda_1$|TC|",r"$\lambda_2$|dimwise|",'|wae|']
    else:
        raise NotImplementedError('Model type not recognised')
    # Train
    # y = np.convolve(loss_rec, np.ones((size_filter,))/size_filter, mode='valid')
    y = loss_rec
    y = np.log(y[::x_step])
    losses.append(list(y))
    if opts['model'] == 'BetaVAE':
        # y = np.convolve(loss_match, np.ones((size_filter,))/size_filter, mode='valid')
        y = loss_match
        y = np.log(np.abs(y[::x_step]))
        losses.append(list(y))
    elif opts['model'] == 'BetaTCVAE':
        for l in zip(*loss_match):
            # y = np.convolve(l, np.ones((size_filter,))/size_filter, mode='valid')
            y = l
            y = np.log(y[::x_step])
            losses.append(list(y))
    elif opts['model'] == 'FactorVAE':
        for l in zip(*loss_match):
            # y = np.convolve(l, np.ones((size_filter,))/size_filter, mode='valid')
            y = l
            y = np.log(y[::x_step])
            losses.append(list(y))
        labels = ['rec',r'$\beta$KL', r'$\gamma$TC']
    elif opts['model'] == 'WAE':
        # y = np.convolve(loss_match, np.ones((size_filter,))/size_filter, mode='valid')
        y = loss_match
        y = np.log(np.abs(y[::x_step]))
        losses.append(list(y))
    else:
        for l in zip(*loss_match):
            # y = np.convolve(l, np.ones((size_filter,))/size_filter, mode='valid')
            y = l
            y = np.log(np.abs(y[::x_step]))
            losses.append(list(y))
    for i in range(len(labels)):
        plt.plot(x, losses_test[i], linewidth=4, color=color_list[i], label=labels[i]+r' test')
        plt.plot(x, losses[i], linewidth=2, color=color_list[i], linestyle='--', label=labels[i])

    plt.grid(axis='y')
    plt.legend(loc='upper right')
    plt.text(0.47, 1., 'Log split Loss curves', ha="center", va="bottom",
                                size=20, transform=ax.transAxes)

    # ### The latent reg curves
    # if opts['model'] == 'disWAE':
    #     base = plt.cm.get_cmap('tab10')
    #     color_list = base(np.linspace(0, 1, 5))
    #     ax = plt.subplot(gs[1, 2])
    #     losses = list(zip(*loss_match))
    #     labels = ['|hsci|','|dimwise|','|wae|']
    #     lmbda = opts['obj_fn_coeffs'] + [1,]
    #     for i, los, lmb, lab in zip([j for j in range(3)],
    #                             losses,
    #                             lmbda,
    #                             labels):
    #         l = np.array(los) / lmb
    #         y = np.log(np.abs(l[::x_step]))
    #         plt.plot(x, y, linewidth=2, color=color_list[i+1], label=lab)
    #     plt.grid(axis='y')
    #     plt.legend(loc='upper right')
    #     plt.text(0.47, 1., 'Latent Reg. curves', ha="center", va="bottom",
    #                                 size=20, transform=ax.transAxes)

    ### The disentangle metrics curves
    if len(mig)>0:
        ax = plt.subplot(gs[1, 2])
        # y = np.convolve(mig, np.ones((size_filter,))/size_filter, mode='valid')
        y = betaVAE
        plt.plot(x, y[::x_step], linewidth=4, color='green', label='betaVAE')
        y = mig
        plt.plot(x, y[::x_step], linewidth=4, color='red', label='MIG')
        # y = np.convolve(factorVAE, np.ones((size_filter,))/size_filter, mode='valid')
        y = factorVAE
        plt.plot(x, y[::x_step], linewidth=4, color='blue', label='factorVAE')
        y = SAP
        plt.plot(x, y[::x_step], linewidth=4, color='purple', label='SAP')
        plt.grid(axis='y')
        plt.legend(loc='upper right')
        plt.text(0.47, 1., 'Disentanglement metrics', ha="center", va="bottom",
                                    size=20, transform=ax.transAxes)


    ### Saving plots and data
    # Plot
    plots_dir = 'train_plots'
    save_path = os.path.join(exp_dir,plots_dir)
    utils.create_dir(save_path)
    fig.savefig(utils.o_gfile((save_path, filename), 'wb'),
                dpi=dpi, format='png')
    plt.close()


def plot_encSigma(opts, enc_Sigmas, exp_dir, filename):
    fig = plt.figure()
    enc_Sigmas = np.array(enc_Sigmas)
    shape = np.shape(enc_Sigmas)
    total_num = shape[0]
    x_step = max(int(total_num / 200), 1)
    x = np.arange(1, total_num + 1, x_step)
    mean, var = enc_Sigmas[::x_step,0], enc_Sigmas[::x_step,1]
    y = np.log(mean)
    plt.plot(x, y, linewidth=1, color='blue', label=r'$\Sigma$')
    plt.grid(axis='y')
    plt.legend(loc='lower left')
    plt.title(r'log norm_Tr$(\Sigma)$ curves')
    ### Saving plot
    plots_dir = 'train_plots'
    save_path = os.path.join(exp_dir,plots_dir)
    utils.create_dir(save_path)
    fig.savefig(utils.o_gfile((save_path, filename), 'wb'),cformat='png')
    plt.close()


def plot_embedded(opts, encoded, decoded, labels, exp_dir, filename, train=True):
    num_pics = np.shape(encoded[0])[0]
    embeds = []
    for i in range(len(encoded)):
        encods = np.concatenate([encoded[i],decoded[i]],axis=0)
        # encods = encoded[i]
        if np.shape(encods)[-1]==2:
            embedding = encods
        else:
            if opts['embedding']=='pca':
                embedding = PCA(n_components=2).fit_transform(encods)
            elif opts['embedding']=='umap':
                embedding = umap.UMAP(n_neighbors=15,
                                        min_dist=0.2,
                                        metric='correlation').fit_transform(encods)
            else:
                assert False, 'Unknown %s method for embedgins vizu' % opts['embedding']
        embeds.append(embedding)
    # Creating a pyplot fig
    dpi = 100
    height_pic = 300
    width_pic = 300
    fig_height = 4*height_pic / float(dpi)
    fig_width = 4*len(embeds) * height_pic  / float(dpi)
    fig = plt.figure(figsize=(fig_width, fig_height))
    #fig = plt.figure()
    gs = matplotlib.gridspec.GridSpec(1, len(embeds))
    for i in range(len(embeds)):
        ax = plt.subplot(gs[0, i])
        plt.scatter(embeds[i][:num_pics, 0], embeds[i][:num_pics, 1], alpha=0.7,
                    c=labels, s=40, label='Qz test',cmap=discrete_cmap(10, base_cmap='tab10'))
        if i==len(embeds)-1:
            plt.colorbar()
        plt.scatter(embeds[i][num_pics:, 0], embeds[i][num_pics:, 1],
                                color='black', s=80, marker='*',label='Pz')
        xmin = np.amin(embeds[i][:,0])
        xmax = np.amax(embeds[i][:,0])
        magnify = 0.01
        width = abs(xmax - xmin)
        xmin = xmin - width * magnify
        xmax = xmax + width * magnify
        ymin = np.amin(embeds[i][:,1])
        ymax = np.amax(embeds[i][:,1])
        width = abs(ymin - ymax)
        ymin = ymin - width * magnify
        ymax = ymax + width * magnify
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.legend(loc='best')
        plt.text(0.47, 1., 'UMAP latent %d' % (i+1), ha="center", va="bottom",
                                                size=20, transform=ax.transAxes)
        # Removing ticks
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])
        ax.axes.set_aspect(1)
    ### Saving plot
    if train:
        plots_dir = 'train_plots'
    else:
        plots_dir = 'test_plots'
    save_path = os.path.join(exp_dir,plots_dir)
    utils.create_dir(save_path)
    fig.savefig(utils.o_gfile((save_path, filename), 'wb'),dpi=dpi,cformat='png')
    plt.close()


def plot_interpolation(opts, interpolations, exp_dir, filename, train=True):
    ### Reshaping images
    greyscale = interpolations.shape[-1] == 1
    if opts['input_normalize_sym']:
        interpolations = interpolations / 2. + 0.5
    white_pix = 4
    num_rows = np.shape(interpolations)[1]
    num_cols = np.shape(interpolations)[2]
    images = []
    for i in range(np.shape(interpolations)[0]):
        pics = np.concatenate(np.split(interpolations[i],num_cols,axis=1),axis=3)
        pics = pics[:,0]
        pics = np.concatenate(np.split(pics,num_rows),axis=1)
        pics = pics[0]
        if greyscale:
            image = 1. - pics
        else:
            image = pics
        images.append(image)
    ### Creating plot
    dpi = 100
    height_pic = images[0].shape[0]
    width_pic = images[0].shape[1]
    fig_height = height_pic / float(dpi)
    fig_width = len(images)*width_pic / float(dpi)
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = matplotlib.gridspec.GridSpec(1, len(images))
    for (i,img) in enumerate(images):
        ax=plt.subplot(gs[0, i])
        if greyscale:
            image = img[:, :, 0]
            # in Greys higher values correspond to darker colors
            plt.imshow(image, cmap='Greys',
                            interpolation='none', vmin=0., vmax=1.)
        else:
            plt.imshow(img, interpolation='none', vmin=0., vmax=1.)
        # Removing ticks
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])
        ax.axes.set_aspect(1)
    ### Saving plot
    if train:
        plots_dir = 'train_plots'
    else:
        plots_dir = 'test_plots'
    save_path = os.path.join(exp_dir,plots_dir)
    utils.create_dir(save_path)
    fig.savefig(utils.o_gfile((save_path, filename), 'wb'),dpi=dpi,cformat='png')
    plt.close()


def save_test_smallnorb(opts, data, reconstructions, transversals, samples, exp_dir):

    """ Generates and saves rec and samples plots"""

    greyscale = data.shape[-1] == 1

    if opts['input_normalize_sym']:
        data = data / 2. + 0.5
        if opts['model']=='TCWAE_MWS' and opts['model']=='TCWAE_GAN' and opts['model']=='WAE':
            reconstructions = reconstructions / 2. + 0.5
            samples = samples / 2. + 0.5
            transversals = transversals / 2. + 0.5

    ### data
    num_cols = data.shape[0]
    obs = []
    for idx in range(data.shape[0]):
        if greyscale:
            obs.append(1. - data[idx, :, :, :])
        else:
            obs.append(data[idx, :, :, :])
    obs = np.array(obs)
    obs = np.concatenate(np.split(obs, num_cols), axis=2)
    obs = np.concatenate(obs, axis=0)
    ### Reconstruction plots
    num_cols = reconstructions.shape[0]
    rec = []
    for idx in range(reconstructions.shape[0]):
        if greyscale:
            rec.append(1. - reconstructions[idx, :, :, :])
        else:
            rec.append(reconstructions[idx, :, :, :])
    rec = np.array(rec)
    rec = np.concatenate(np.split(rec, num_cols), axis=2)
    rec = np.concatenate(rec, axis=0)
    ### Samples
    num_cols = samples.shape[0]
    gen = []
    for idx in range(samples.shape[0]):
        if greyscale:
            gen.append(1. - samples[idx, :, :, :])
        else:
            gen.append(samples[idx, :, :, :])
    gen = np.array(gen)
    gen = np.concatenate(np.split(gen, num_cols), axis=2)
    gen = np.concatenate(gen, axis=0)
    ### Latent transversal
    num_cols = transversals.shape[2]
    num_rows = transversals.shape[1]
    images = []
    names = []
    for i in range(np.shape(transversals)[0]):
        pics = np.concatenate(np.split(transversals[i],num_cols,axis=1),axis=3)
        pics = pics[:,0]
        pics = np.concatenate(np.split(pics,num_rows),axis=1)
        pics = pics[0]
        if greyscale:
            image = 1. - pics
        else:
            image = pics
        images.append(image)
        names.append(opts['model'] + '_latent_transversal_' + str(i))
    ### Creating a pyplot fig
    to_plot_list = zip([obs, rec, gen] + images,
                         [opts['model'] + '_observation',
                         opts['model'] + '_reconstruction',
                         opts['model'] + '_sample',]
                         + names)
    dpi = 100
    for img, filename in to_plot_list:
        height_pic = img.shape[0]
        width_pic = img.shape[1]
        fig_height = height_pic / 20
        fig_width = width_pic / 20
        fig = plt.figure(figsize=(fig_width, fig_height))
        if greyscale:
            image = img[:, :, 0]
            # in Greys higher values correspond to darker colors
            plt.imshow(image, cmap='Greys',
                            interpolation='none', vmin=0., vmax=1.)
        else:
            plt.imshow(img, interpolation='none', vmin=0., vmax=1.)
        # Removing axes, ticks, labels
        plt.axis('off')
        # # placing subplot
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                hspace = 0, wspace = 0)
        # Saving
        save_path = os.path.join(exp_dir,'test_plots')
        utils.create_dir(save_path)
        filename = filename + '.png'
        plt.savefig(utils.o_gfile((save_path, filename), 'wb'),
                    dpi=dpi, format='png', box_inches='tight', pad_inches=0.0)
        plt.close()


def save_test_celeba(opts, data, reconstructions, transversals, samples, exp_dir):

    """ Generates and saves rec and samples plots"""

    greyscale = data.shape[-1] == 1

    if opts['input_normalize_sym']:
        data = data / 2. + 0.5
        reconstructions = reconstructions / 2. + 0.5
        samples = samples / 2. + 0.5
        transversals = transversals / 2. + 0.5

    ### Reconstruction plots
    num_pics = 100
    num_cols = 10
    rec = []
    merged = np.vstack([reconstructions, data])
    r_ptr = 0
    w_ptr = 0
    for _ in range(int(num_pics / 2)):
        merged[w_ptr] = data[r_ptr]
        merged[w_ptr + 1] = reconstructions[r_ptr]
        r_ptr += 1
        w_ptr += 2
    for idx in range(num_pics):
        if greyscale:
            rec.append(1. - merged[idx, :, :, :])
        else:
            rec.append(merged[idx, :, :, :])
    # Figuring out a layout
    rec = np.array(rec)
    rec = np.concatenate(np.split(rec, num_cols), axis=2)
    rec = np.concatenate(rec, axis=0)
    ### Samples
    num_cols = 10
    gen = []
    for idx in range(samples.shape[0]):
        if greyscale:
            gen.append(1. - samples[idx, :, :, :])
        else:
            gen.append(samples[idx, :, :, :])
    gen = np.array(gen)
    gen = np.concatenate(np.split(gen, num_cols), axis=2)
    gen = np.concatenate(gen, axis=0)
    ### Latent transversal
    num_rows = transversals.shape[1]
    num_cols = transversals.shape[2]
    images = []
    names = []
    for i in range(np.shape(transversals)[0]):
        pics = np.concatenate(np.split(transversals[i],num_cols,axis=1),axis=3)
        pics = pics[:,0]
        pics = np.concatenate(np.split(pics,num_rows),axis=1)
        pics = pics[0]
        if greyscale:
            image = 1. - pics
        else:
            image = pics
        images.append(image)
        names.append(opts['model'] + '_latent_transversal_' + str(i))
    ### Creating a pyplot fig
    to_plot_list = zip([rec, gen] + images,
                         [opts['model'] + '_reconstruction',
                         opts['model'] + '_sample',]
                         + names)
    dpi = 100
    for img, filename in to_plot_list:
        height_pic = img.shape[0]
        width_pic = img.shape[1]
        fig_height = height_pic / 20
        fig_width = width_pic / 20
        fig = plt.figure(figsize=(fig_width, fig_height))
        if greyscale:
            image = img[:, :, 0]
            # in Greys higher values correspond to darker colors
            plt.imshow(image, cmap='Greys',
                            interpolation='none', vmin=0., vmax=1.)
        else:
            plt.imshow(img, interpolation='none', vmin=0., vmax=1.)
        # Removing axes, ticks, labels
        plt.axis('off')
        # # placing subplot
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                hspace = 0, wspace = 0)
        # Saving
        save_path = os.path.join(exp_dir,'test_plots')
        utils.create_dir(save_path)
        filename = filename + '.png'
        plt.savefig(utils.o_gfile((save_path, filename), 'wb'),
                    dpi=dpi, format='png', box_inches='tight', pad_inches=0.0)
        plt.close()


def save_dimwise_traversals(opts, transversals, exp_dir):

    """ Dimwise latent traversals"""

    assert transversals.shape[1]==opts['zdim']
    greyscale = transversals.shape[-1] == 1
    if opts['input_normalize_sym']:
        transversals = transversals / 2. + 0.5
    num_rows = transversals.shape[0]
    num_cols = transversals.shape[2]
    num_im = transversals.shape[1]
    images, names = [], []
    for i in range(num_im):
        pics = np.concatenate(np.split(transversals[:,i],num_cols,axis=1),axis=3)
        pics = pics[:,0]
        pics = np.concatenate(np.split(pics,num_rows),axis=1)
        pics = pics[0]
        if greyscale:
            image = 1. - pics
        else:
            image = pics
        images.append(image)
        names.append(opts['model'] + '_z' + str(i))
    ### Creating a pyplot fig
    to_plot_list = zip(images,names)
    dpi = 100
    for img, filename in to_plot_list:
        height_pic = img.shape[0]
        width_pic = img.shape[1]
        fig_height = height_pic / 20
        fig_width = width_pic / 20
        fig = plt.figure(figsize=(fig_width, fig_height))
        if greyscale:
            image = img[:, :, 0]
            # in Greys higher values correspond to darker colors
            plt.imshow(image, cmap='Greys',
                            interpolation='none', vmin=0., vmax=1.)
        else:
            plt.imshow(img, interpolation='none', vmin=0., vmax=1.)
        # Removing axes, ticks, labels
        plt.axis('off')
        # # placing subplot
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                hspace = 0, wspace = 0)
        # Saving
        save_path = os.path.join(exp_dir,'test_plots')
        utils.create_dir(save_path)
        save_path = os.path.join(save_path,'dimwise_traversals')
        utils.create_dir(save_path)
        filename = filename + '.png'
        plt.savefig(utils.o_gfile((save_path, filename), 'wb'),
                    dpi=dpi, format='png', box_inches='tight', pad_inches=0.0)
        plt.close()


def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""
    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:
    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return matplotlib.colors.LinearSegmentedColormap.from_list(cmap_name, color_list, N)
