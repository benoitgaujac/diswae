import os
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
                     loss,
                     loss_rec, loss_rec_test,
                     loss_match,
                     exp_dir,
                     filename):

    """ Generates and saves the plot of the following layout:
        img1 | img2 | img3
        img4 | img5 |

        img1    -   test reconstructions
        img2    -   train reconstructions
        img3    -   samples
        img4    -   loss curves
        img5    -   rec loss curves

    """
    num_pics = opts['plot_num_pics']
    num_cols = opts['plot_num_cols']
    assert num_pics % num_cols == 0
    assert num_pics % 2 == 0
    greyscale = data_train.shape[-1] == 1

    if opts['input_normalize_sym']:
        data_train = data_train / 2. + 0.5
        data_test = data_test / 2. + 0.5
        rec_train = rec_train / 2. + 0.5
        for i in range(len(rec_test)):
            rec_test[i] = rec_test[i] / 2. + 0.5
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
    total_num = len(loss)
    x_step = max(int(total_num / 200), 1)
    x = np.arange(1, len(loss) + 1, x_step)
    y = np.log(loss[::x_step])
    plt.plot(x, y, linewidth=3, color='black', label='loss')
    plt.grid(axis='y')
    plt.legend(loc='upper right')
    plt.text(0.47, 1., 'Obj curve', ha="center", va="bottom",
                                size=20, transform=ax.transAxes)

    ### The loss curves
    base = plt.cm.get_cmap('tab10')
    color_list = base(np.linspace(0, 1, 5))
    ax = plt.subplot(gs[1, 1])
    if opts['model'] == 'BetaVAE':
        losses = [loss_rec,loss_match]
        labels = ['rec',r'$\beta$KL']
    elif opts['model'] == 'WAE':
        losses = [loss_rec,loss_match]
        labels = ['rec',r'$\lambda$|mmd|']
    elif opts['model'] == 'disWAE':
        losses = [loss_rec,] + list(zip(*loss_match))
        labels = ['rec',r'$\lambda_1$|hsci|]','$\lambda_2$|dimwise|','|wae|']
    else:
        raise NotImplementedError('Model type not recognised')
    for i, los, lab in zip([j for j in range(4)],
                            losses,
                            labels):
        l = np.array(los)
        y = np.log(np.abs(l[::x_step]))
        plt.plot(x, y, linewidth=2, color=color_list[i], label=lab)
    plt.grid(axis='y')
    plt.legend(loc='upper right')
    plt.text(0.47, 1., 'Loss curves', ha="center", va="bottom",
                                size=20, transform=ax.transAxes)

    ### The latent reg curves
    if opts['model'] == 'disWAE':
        base = plt.cm.get_cmap('tab10')
        color_list = base(np.linspace(0, 1, 5))
        ax = plt.subplot(gs[1, 2])
        losses = list(zip(*loss_match))
        labels = ['|hsci|','|dimwise|','|wae|']
        lmbda = opts['obj_fn_coeffs'] + [1,]
        for i, los, lmb, lab in zip([j for j in range(3)],
                                losses,
                                lmbda,
                                labels):
            l = np.array(los) / lmb
            y = np.log(np.abs(l[::x_step]))
            plt.plot(x, y, linewidth=2, color=color_list[i+1], label=lab)
        plt.grid(axis='y')
        plt.legend(loc='upper right')
        plt.text(0.47, 1., 'Latent Reg. curves', ha="center", va="bottom",
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


def plot_embedded(opts, encoded, decoded, labels, exp_dir, filename):
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
                    # c=labels, s=40, label='Qz test',edgecolors='none',cmap=discrete_cmap(10, base_cmap='Vega10'))
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
        # ax.axes.set_xlim([0, width_pic])
        # ax.axes.set_ylim([height_pic, 0])
        ax.axes.set_aspect(1)
    ### Saving plot
    plots_dir = 'train_plots'
    save_path = os.path.join(exp_dir,plots_dir)
    utils.create_dir(save_path)
    fig.savefig(utils.o_gfile((save_path, filename), 'wb'),dpi=dpi,cformat='png')
    plt.close()


def plot_interpolation(opts, interpolations, exp_dir, filename):
    ### Reshaping images
    greyscale = interpolations.shape[-1] == 1
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
        # plt.text(0.47, 1., title,
        #          ha="center", va="bottom", size=20, transform=ax.transAxes)
        # Removing ticks
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])
        ax.axes.set_aspect(1)
    ### Saving plot
    plots_dir = 'train_plots'
    save_path = os.path.join(exp_dir,plots_dir)
    utils.create_dir(save_path)
    fig.savefig(utils.o_gfile((save_path, filename), 'wb'),dpi=dpi,cformat='png')
    plt.close()


def save_latent_interpolation(opts, data_test, label_test, # data, labels
                    encoded, reconstructed,# encoded, reconstructed points
                    full_reconstructed, sampled_reconstructed,
                    inter_anchors, inter_latent, # anchors and latents interpolation
                    samples, # samples
                    MODEL_PATH): # working directory

    # --- Create saving directory and preprocess
    plots_dir = 'test_plots'
    save_path = os.path.join(opts['exp_dir'],plots_dir)
    utils.create_dir(save_path)

    dpi = 100

    greyscale = np.shape(data_test)[-1] == 1

    if opts['input_normalize_sym']:
        full_reconstructed = full_reconstructed / 2. + 0.5
        reconstructed = reconstructed / 2. + 0.5
        sampled_reconstructed = sampled_reconstructed / 2. + 0.5
        anchors = anchors / 2. + 0.5
        inter_anchors = inter_anchors / 2. + 0.5
        inter_latent = inter_latent / 2. + 0.5
        samples = samples / 2. + 0.5
    images = []

    # --- full reconstruction plots
    num_rows = len(full_reconstructed)
    num_cols = np.shape(full_reconstructed[0])[0]
    # padding inut image
    # npad = 1
    # pad = ((0,npad),(0,0),(0,0))
    npad = 1
    pad_0 = ((npad,0),(0,0),(0,0))
    pad_1 = ((0,npad),(0,0),(0,0))
    for n in range(num_cols):
        # full_reconstructed[0][n] = np.pad(full_reconstructed[0][n,:-npad], pad, mode='constant', constant_values=1.0)
        full_reconstructed[0][n] = np.pad(full_reconstructed[0][n,npad:], pad_0, mode='constant', constant_values=1.0)
        full_reconstructed[1][n] = np.pad(full_reconstructed[1][n,:-npad], pad_1, mode='constant', constant_values=1.0)
    full_reconstructed = np.split(np.array(full_reconstructed[::-1]),num_cols,axis=1)
    pics = np.concatenate(full_reconstructed,axis=-2)
    pics = np.concatenate(np.split(pics,num_rows),axis=-3)
    pics = pics[0,0]
    if greyscale:
        image = 1. - pics
    else:
        image = pics
    images.append(image)

    # --- Sample plots
    num_pics = np.shape(samples)[0]
    num_cols = 15 #np.sqrt(num_pics)
    pics = []
    for idx in range(num_pics):
        if greyscale:
            pics.append(1. - samples[idx, :, :, :])
        else:
            pics.append(samples[idx, :, :, :])
    pics = np.array(pics)
    image = np.concatenate(np.split(pics, num_cols), axis=2)
    image = np.concatenate(image, axis=0)
    images.append(image)

    # -- Reconstruction plots
    num_cols = 14
    num_pics = num_cols**2
    # Arrange pics and reconstructions in a proper way
    # sample, recon = (data_test[:int(num_pics/2)],reconstructed[:int(num_pics/2)])
    pics = []
    for n in range(int(num_pics)):
        if n%2==0:
            # pics.append(sample[int(n/2)])
            pics.append(data_test[int(n/2)])
        else:
            # pics.append(recon[int(n/2)])
            pics.append(reconstructed[int(n/2)])
    # Figuring out a layout
    pics = np.array(pics)
    pics = np.split(pics,num_cols,axis=0)
    pics = np.concatenate(pics,axis=2)
    pics = np.concatenate(np.split(pics,num_cols),axis=1)
    pics = pics[0]
    if greyscale:
        image = 1. - pics
    else:
        image = pics
    images.append(image)

    # --- Points Interpolation plots
    white_pix = 4
    num_rows = np.shape(inter_anchors)[0]
    num_cols = np.shape(inter_anchors)[1]
    pics = np.concatenate(np.split(inter_anchors,num_cols,axis=1),axis=3)
    pics = pics[:,0]
    pics = np.concatenate(np.split(pics,num_rows),axis=1)
    pics = pics[0]
    if greyscale:
        image = 1. - pics
    else:
        image = pics
    images.append(image)

    # --- Save plots
    # img1, img2 = images
    # to_plot_list = zip([img1, img2],
    #                      ['Full Reconstructions',
    #                      'Samples'],
    #                      ['full_recon',
    #                      'prior_samples'])
    img1, img2, img3, img4 = images
    to_plot_list = zip([img1, img2, img3, img4],
                         ['Full Reconstructions',
                         'Samples',
                         'Reconstruction',
                         'Points interpolation'],
                         ['full_recon',
                         'prior_samples',
                         'reconstructed',
                         'point_inter'])

    #Settings for pyplot fig
    for img, title, filename in to_plot_list:
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
        filename = filename + '.png'
        plt.savefig(utils.o_gfile((save_path, filename), 'wb'),
                    dpi=dpi, format='png', box_inches='tight', pad_inches=0.0)
        plt.close()


    if inter_latent is not None:
        # --- Prior Interpolation plots
        white_pix = 4
        num_rows = np.shape(inter_latent)[0]
        num_cols = np.shape(inter_latent)[1]
        pics = np.concatenate(np.split(inter_latent,num_cols,axis=1),axis=3)
        pics = pics[:,0]
        pics = np.concatenate(np.split(pics,num_rows),axis=1)
        pics = pics[0]
        if greyscale:
            image = 1. - pics
        else:
            image = pics
        # --- Save plots
        to_plot_list = zip([image,],
                             ['Latent interpolation',],
                             ['latent_inter',])
        #Settings for pyplot fig
        for img, title, filename in to_plot_list:
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
            filename = filename + '.png'
            plt.savefig(utils.o_gfile((save_path, filename), 'wb'),
                        dpi=dpi, format='png', box_inches='tight', pad_inches=0.0)
            plt.close()

    # --- sampled reconstruction plots
    if opts['encoder'][0]=='gauss' and sampled_reconstructed is not None:
        num_cols = len(sampled_reconstructed)
        num_rows = np.shape(sampled_reconstructed[0])[0]
        # padding inut image
        npad = 1
        pad = ((0,0),(0,npad),(0,0))
        for n in range(num_rows):
            sampled_reconstructed[0][n] = np.pad(sampled_reconstructed[0][n,:,:-npad], pad, mode='constant', constant_values=1.)
        pics = np.concatenate(sampled_reconstructed,axis=2)
        pics = np.concatenate(np.split(pics,num_rows),axis=1)
        pics = pics[0]
        if greyscale:
            image = 1. - pics
        else:
            image = pics
        # Plotting
        height_pic = image.shape[0]
        width_pic = image.shape[1]
        fig_height = 2*height_pic / dpi
        fig_width = 2*width_pic / dpi
        fig = plt.figure(figsize=(fig_width, fig_height))

        if greyscale:
            image = image[:, :, 0]
            # in Greys higher values correspond to darker colors
            plt.imshow(image, cmap='Greys',
                            interpolation='none', vmin=0., vmax=1.)
        else:
            plt.imshow(image, interpolation='none', vmin=0., vmax=1.)
        # Removing axes, ticks, labels
        plt.axis('off')
        # # placing subplot
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                hspace = 0, wspace = 0)
        # Saving
        filename = 'sampled_recons.png'
        plt.savefig(utils.o_gfile((save_path, filename), 'wb'),
                    dpi=dpi, format='png', box_inches='tight', pad_inches=0.0)
        plt.close()

    # --- Embedings vizu
    num_pics = np.shape(encoded[0])[0]
    embeds = []
    for i in range(len(encoded)):
        encods = encoded[i]
        if np.shape(encods)[-1]==2:
            embedding = encods
        else:
            if opts['embedding']=='pca':
                embedding = PCA(n_components=2).fit_transform(encods)
            elif opts['embedding']=='umap':
                embedding = umap.UMAP(n_neighbors=30,
                                        min_dist=0.15,
                                        metric='correlation',
                                        # n_neighbors=10,
                                        # min_dist=0.1,
                                        # metric='euclidean'
                                        ).fit_transform(encods)
            elif opts['embedding']=='tsne':
                embedding = TSNE(n_components=2,
                                perplexity=40,
                                early_exaggeration=15.0,
                                init='pca').fit_transform(encods)
            else:
                assert False, 'Unknown %s method for embedgins vizu' % opts['embedding']
        embeds.append(embedding)
    # Creating a pyplot fig
    dpi = 100
    height_pic = 10
    width_pic = 10
    fig_height = height_pic
    fig_width = len(embeds) * width_pic
    # fig_width = 4*len(embeds) * width_pic  / 20
    fig = plt.figure(figsize=(fig_width, fig_height))
    # gs = matplotlib.gridspec.GridSpec(1, len(embeds))
    for i in range(len(embeds)):
        # ax = plt.subplot(gs[0, i])
        ax = fig.add_subplot(1, len(embeds), i+1)
        plt.scatter(embeds[i][:, 0], embeds[i][:, 1], alpha=0.6,
                    c=label_test, s=40, label='Qz test',cmap=discrete_cmap(10, base_cmap='tab10'))
                    # c=label_test, s=40, edgecolors='none',cmap=discrete_cmap(10, base_cmap='Vega10'))
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
        # plt.legend(loc='best')
        plt.text(0.47, 1., r'Latent space $\mathcal{Z}_{%d}$' % (i+1), ha="center", va="bottom",
                                                size=45, transform=ax.transAxes)
        # # colorbar
        # if i==len(embeds)-1:
        #     plt.colorbar()
        # Removing ticks
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])
        x0,x1 = ax.get_xlim()
        y0,y1 = ax.get_ylim()
        ax.set_aspect(abs(x1-x0)/abs(y1-y0))
    # adjust space between subplots
    plt.subplots_adjust(bottom=0.05, right=0.9, top=0.95)
    cax = plt.axes([0.91, 0.165, 0.01, 0.7])
    cbar = plt.colorbar(cax=cax)
    cbar.ax.tick_params(labelsize=35)
    # plt.tight_layout()
    # plt.subplots_adjust(left=0., right=0., top=0., bottom=0.)
    # Saving
    filename = 'embeddings.png'
    plt.savefig(utils.o_gfile((save_path, filename), 'wb'),
                dpi=dpi, format='png', bbox_inches='tight', pad_inches=0.01)
    plt.close()

def save_vlae_experiment(opts, decoded, exp_dir):
    # num_pics = opts['plot_num_pics']
    num_cols = 10
    greyscale = decoded[0].shape[-1] == 1

    if opts['input_normalize_sym']:
        for i in range(len(decoded)):
            decoded[i] = decoded[i] / 2. + 0.5

    images = []

    for n in range(len(decoded)):
        samples = decoded[n]
        num_pics = len(samples)
        num_cols = sqrt(num_pics)
        # assert len(samples) == num_pics
        pics = []
        for idx in range(num_pics):
            if greyscale:
                pics.append(1. - samples[idx, :, :, :])
            else:
                pics.append(samples[idx, :, :, :])
        # Figuring out a layout
        pics = np.array(pics)
        if n==0:
            npad = 1
            pad = ((npad,npad),(npad,npad),(0,0))
            pics[0] = np.pad(pics[0,npad:-npad,npad:-npad], pad, mode='constant', constant_values=.0)
        image = np.concatenate(np.split(pics, num_cols), axis=2)
        image = np.concatenate(image, axis=0)
        images.append(image)

    # Creating a pyplot fig
    dpi = 100
    height_pic = images[0].shape[0]
    width_pic = images[0].shape[1]
    fig_height = 1 * 2*height_pic / float(dpi)
    fig_width = opts['nlatents'] * 2*width_pic / float(dpi)
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = matplotlib.gridspec.GridSpec(1, opts['nlatents'])

    # Filling in separate parts of the plot
    for n in range(len(decoded)):
        image = images[n]
        plt.subplot(gs[0, n])
        if greyscale:
            image = image[:, :, 0]
            # in Greys higher values correspond to darker colors
            ax = plt.imshow(image, cmap='Greys',
                            interpolation='none', vmin=0., vmax=1.)
        else:
            ax = plt.imshow(img, interpolation='none', vmin=0., vmax=1.)
        ax = plt.subplot(gs[0, n])
        # title = 'sampling %d layer' % n
        # plt.text(0.47, 1., title,
        #          ha="center", va="bottom", size=20, transform=ax.transAxes)
        # Removing ticks
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])
        ax.axes.set_xlim([0, width_pic])
        ax.axes.set_ylim([height_pic, 0])
        ax.axes.set_aspect(1)
    # placing subplot
    # plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,hspace = 0, wspace = 0)
    ### Saving plots and data
    # Plot
    plots_dir = 'test_plots'
    save_path = os.path.join(exp_dir,plots_dir)
    utils.create_dir(save_path)
    filename = 'vlae_exp.png'
    fig.savefig(utils.o_gfile((save_path, filename), 'wb'),
                dpi=dpi, format='png', bbox_inches='tight', pad_inches=0.0)
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
