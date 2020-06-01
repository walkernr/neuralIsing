# -*- coding: utf-8 -*-
"""
Created on Wed Dec 4 14:15:18 2019

@author: Nicholas
"""

import argparse
import os
import numpy as np
from tqdm import tqdm, trange
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (Input, Flatten, Reshape, Lambda, Concatenate,
                                     Dense, BatchNormalization, Conv2D, Conv2DTranspose,
                                     SpatialDropout2D, AlphaDropout, Activation, LeakyReLU)
from tensorflow.keras.optimizers import SGD, RMSprop, Adam, Adamax, Nadam
from tensorflow_addons.optimizers import AdamW, LAMB, LazyAdam, NovoGrad, RectifiedAdam, SGDW, Yogi, Lookahead
from tensorflow.keras.models import Model, save_model, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.python.training.tracking.util import Checkpoint
from tensorflow.python.training.checkpoint_management import CheckpointManager
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def parse_args():
    ''' parses command line arguments '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', help='verbose output',
                        action='store_true')
    parser.add_argument('-r', '--restart', help='restart mode',
                        action='store_true')
    parser.add_argument('-pt', '--plot', help='plot results',
                        action='store_true')
    parser.add_argument('-p', '--parallel', help='parallel (cpu) mode',
                        action='store_true')
    parser.add_argument('-g', '--gpu', help='gpu mode (will default to cpu if unable)',
                        action='store_true')
    parser.add_argument('-nt', '--threads', help='number of parallel threads',
                        type=int, default=20)
    parser.add_argument('-n', '--name', help='simulation name',
                        type=str, default='run')
    parser.add_argument('-l', '--lattice_length', help='lattice size (side length)',
                        type=int, default=27)
    parser.add_argument('-si', '--sample_interval', help='interval for selecting phase points (variational autoencoder)',
                        type=int, default=1)
    parser.add_argument('-sn', '--sample_number', help='number of samples per phase point (variational autoencoder)',
                        type=int, default=1024)
    parser.add_argument('-sc', '--scale_data', help='scale data (-1, 1) -> (0, 1)',
                        action='store_true')
    parser.add_argument('-cp', '--conv_padding', help='convolutional zero-padding',
                        action='store_true')
    parser.add_argument('-cn', '--conv_number', help='convolutional layer depth',
                        type=int, default=3)
    parser.add_argument('-fbl', '--filter_base_length', help='size of filters in base hidden convolutional layer',
                        type=int, default=3)
    parser.add_argument('-fbs', '--filter_base_stride', help='size of filter stride in base hidden convolutional layer',
                        type=int, default=3)
    parser.add_argument('-fb', '--filter_base', help='base number of filters in base hidden convolutional layer',
                        type=int, default=9)
    parser.add_argument('-fl', '--filter_length', help='size of filters following base convolution',
                        type=int, default=3)
    parser.add_argument('-fs', '--filter_stride', help='size of filter strides following base convolution',
                        type=int, default=3)
    parser.add_argument('-ff', '--filter_factor', help='multiplicative factor of filters after base convolution',
                        type=int, default=9)
    parser.add_argument('-do', '--dropout', help='toggle dropout layers',
                        action='store_true')
    parser.add_argument('-zd', '--z_dimension', help='sample noise dimension',
                        type=int, default=5)
    parser.add_argument('-ka', '--kld_annealing', help='toggle kld annealing',
                        action='store_true')
    parser.add_argument('-ra', '--alpha', help='total correlation alpha',
                        type=float, default=1.0)
    parser.add_argument('-rb', '--beta', help='total correlation beta',
                        type=float, default=8.0)
    parser.add_argument('-rl', '--lamb', help='total correlation lambda',
                        type=float, default=1.0)
    parser.add_argument('-ki', '--kernel_initializer', help='kernel initializer',
                        type=str, default='lecun_normal')
    parser.add_argument('-an', '--activation', help='activation function',
                        type=str, default='selu')
    parser.add_argument('-op', '--optimizer', help='optimizer',
                        type=str, default='nadam')
    parser.add_argument('-la', '--lookahead', help='toggle lookahead optimizer',
                        action='store_true')
    parser.add_argument('-lr', '--learning_rate', help='learning rate',
                        type=float, default=1e-3)
    parser.add_argument('-bs', '--batch_size', help='size of batches',
                        type=int, default=169)
    parser.add_argument('-rs', '--random_sampling', help='random batch sampling',
                        action='store_true')
    parser.add_argument('-ep', '--epochs', help='number of training epochs',
                        type=int, default=32)
    parser.add_argument('-sd', '--random_seed', help='random seed for sample selection and learning',
                        type=int, default=128)
    args = parser.parse_args()
    return (args.verbose, args.restart, args.plot, args.parallel, args.gpu, args.threads,
            args.name, args.lattice_length, args.sample_interval, args.sample_number, args.scale_data, args.conv_padding,
            args.conv_number, args.filter_base_length, args.filter_base_stride, args.filter_base, args.filter_length, args.filter_stride, args.filter_factor,
            args.dropout, args.z_dimension, args.kld_annealing, args.alpha, args.beta, args.lamb,
            args.kernel_initializer, args.activation, args.optimizer, args.lookahead, args.learning_rate,
            args.batch_size, args.random_sampling, args.epochs, args.random_seed)


def load_thermal_params(name, lattice_length):
    ''' load thermal parameters '''
    fields = np.load(os.getcwd()+'/{}.{}.h.npy'.format(name, lattice_length))
    temps = np.load(os.getcwd()+'/{}.{}.t.npy'.format(name, lattice_length))
    return fields, temps


def load_configurations(name, lattice_length):
    ''' load configurations and thermal measurements '''
    conf = np.load(os.getcwd()+'/{}.{}.dmp.npy'.format(name, lattice_length))
    thrm = np.load(os.getcwd()+'/{}.{}.dat.npy'.format(name, lattice_length))
    return conf, thrm


def sample_gaussian(beta):
    ''' samples a point in a multivariate gaussian distribution '''
    mu, logvar = beta
    return np.random.normal(mu, np.exp(0.5*logvar))


def scale_configurations(conf):
    ''' scales input configurations '''
    # (-1, 1) -> (0, 1)
    return (conf+1)/2


def unscale_configurations(conf):
    ''' unscales input configurations '''
    # (0, 1) -> (-1, 1)
    return 2*conf-1


def binary_crossentropy(x, x_hat, scaled):
    ''' binary crossentropy error '''
    eps = 1e-4
    ns, _, _, _ = x.shape
    if not scaled:
        x = scale_configurations(x)
        x_hat = scale_configurations(x_hat)
    bc = -1*(x*np.log(x_hat+eps)+(1.-x)*np.log(1.-x_hat+eps)).reshape(ns, -1)
    return bc


def binary_crossentropy_accuracy(x, x_hat, scaled):
    ''' binary accuracy '''
    bc = binary_crossentropy(x, x_hat, scaled)
    ba = np.exp(-bc)
    return np.stack((bc.mean(1), bc.std(1)), axis=-1), np.stack((ba.mean(1), ba.std(1)), axis=-1)


def shuffle_samples(data, num_fields, num_temps, indices):
    ''' shuffles data by sample index independently for each thermal parameter combination '''
    # reorders samples independently for each (h, t) according to indices
    return np.array([[data[i, j, indices[i, j]] for j in range(num_temps)] for i in range(num_fields)])


def load_select_scale_data(name, lattice_length, interval, num_samples, scaled, seed, verbose=False):
    ''' selects random subset of data according to phase point interval and sample count '''
    # apply interval to fields, temperatures, configurations, and thermal data
    fields, temps = load_thermal_params(name, lattice_length)
    interval_fields = fields[::interval]
    interval_temps = temps[::interval]
    del fields, temps
    conf, thrm = load_configurations(name, lattice_length)
    interval_conf = conf[::interval, ::interval]
    interval_thrm = thrm[::interval, ::interval]
    del conf, thrm
    # field and temperature counts
    num_fields, num_temps = interval_fields.size, interval_temps.size
    # sample count
    total_num_samples = interval_thrm.shape[2]
    # selected sample indices
    indices = np.zeros((num_fields, num_temps, num_samples), dtype=np.uint16)
    if verbose:
        print(100*'_')
    for i in trange(num_fields, desc='Selecting Samples', disable=not verbose):
        for j in range(num_temps):
                indices[i, j] = np.random.permutation(total_num_samples)[:num_samples]
    # construct selected data subset
    select_conf = shuffle_samples(interval_conf, num_fields, num_temps, indices)
    if scaled:
        select_conf = scale_configurations(select_conf).astype(np.int8)
    select_thrm = shuffle_samples(interval_thrm, num_fields, num_temps, indices)
    # save selected data arrays
    np.save(os.getcwd()+'/{}.{}.{}.h.npy'.format(name, lattice_length, interval), interval_fields)
    np.save(os.getcwd()+'/{}.{}.{}.t.npy'.format(name, lattice_length, interval), interval_temps)
    np.save(os.getcwd()+'/{}.{}.{}.{}.{:d}.{}.conf.npy'.format(name, lattice_length,
                                                               interval, num_samples,
                                                               scaled, seed), select_conf)
    np.save(os.getcwd()+'/{}.{}.{}.{}.{:d}.{}.thrm.npy'.format(name, lattice_length,
                                                               interval, num_samples,
                                                               scaled, seed), select_thrm)
    return interval_fields, interval_temps, select_conf, select_thrm


def load_data(name, lattice_length, interval, num_samples, scaled, seed, verbose=False):
    try:
        # try loading selected data arrays
        fields = np.load(os.getcwd()+'/{}.{}.{}.h.npy'.format(name, lattice_length, interval))
        temps = np.load(os.getcwd()+'/{}.{}.{}.t.npy'.format(name, lattice_length, interval))
        conf = np.load(os.getcwd()+'/{}.{}.{}.{}.{:d}.{}.conf.npy'.format(name, lattice_length,
                                                                          interval, num_samples,
                                                                          scaled, seed))
        thrm = np.load(os.getcwd()+'/{}.{}.{}.{}.{:d}.{}.thrm.npy'.format(name, lattice_length,
                                                                          interval, num_samples,
                                                                          scaled, seed))
        if verbose:
            print(100*'_')
            print('Scaled/selected Ising configurations and thermal parameters/measurements loaded from file')
            print(100*'_')
    except:
        # generate selected data arrays
        (fields, temps,
         conf, thrm) = load_select_scale_data(name, lattice_length,
                                              interval, num_samples,
                                              scaled, seed, verbose)
        if verbose:
            print(100*'_')
            print('Ising configurations selected/scaled and thermal parameters/measurements selected')
            print(100*'_')
    return fields, temps, conf, thrm


def save_output_data(data, alias, name, lattice_length, interval, num_samples, scaled, seed, prfx):
    ''' save output data from model '''
    # file parameters
    params = (name, lattice_length, interval, num_samples, scaled, seed)
    file_name = os.getcwd()+'/{}.{}.{}.{}.{:d}.{}.'.format(*params)+prfx+'.{}.npy'.format(alias)
    np.save(file_name, data)


def plot_histogram(u, cmap, file_prfx, alias, domain_name, verbose=False):
    file_name = os.getcwd()+'/'+file_prfx+'.{}.png'.format(alias)
    # initialize figure and axes
    fig, ax = plt.subplots()
    # remove spines on top and right
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # set axis ticks to left and bottom
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.hist(u, bins=64, density=True, color=cmap(0.25))
    ax.set_xlabel(domain_name)
    ax.set_ylabel('Density')
    # save figure
    fig.savefig(file_name)
    plt.close()


def plot_bc_error_accuracy(error, accuracy, cmap,
                           name, lattice_length, interval, num_samples, scaled, seed,
                           prfx, verbose=False):
    # file name parameters
    params = (name, lattice_length, interval, num_samples, scaled, seed)
    file_prfx = '{}.{}.{}.{}.{:d}.{}.'.format(*params)+prfx
    plot_histogram(error[:, 0], cmap, file_prfx, 'bc_err_hist', 'Bernoulli Entropy (Reconstruction Loss)', verbose)
    plot_histogram(accuracy[:, 0], cmap, file_prfx, 'bc_acc_hist', 'Classification Accuracy (Reconstruction)', verbose)


def plot_batch_losses(losses, cmap, file_prfx, verbose=False):
    file_name = os.getcwd()+'/'+file_prfx+'.loss.batch.png'
    # initialize figure and axes
    fig, ax = plt.subplots()
    # remove spines on top and right
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # set axis ticks to left and bottom
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    n_epochs, n_batches = losses[0].shape[:2]
    n_iters = n_epochs*n_batches
    # plot losses
    loss_list = ['VAE Loss', 'Latent Loss', 'Reconstruction Loss']
    color_list = np.linspace(0.2, 0.8, len(losses))
    for i in trange(len(losses), desc='Plotting Batch Losses', disable=not verbose):
        ax.plot(np.arange(1, n_iters+1), losses[i].reshape(-1), color=cmap(color_list[i]), label=loss_list[i])
    ax.legend(loc='upper right')
    ax.set_xlabel('Batch')
    ax.set_ylabel('Loss')
    # save figure
    fig.savefig(file_name)
    plt.close()


def plot_epoch_losses(losses, cmap, file_prfx, verbose=False):
    file_name = os.getcwd()+'/'+file_prfx+'.loss.epoch.png'
    # initialize figure and axes
    fig, ax = plt.subplots()
    # remove spines on top and right
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # set axis ticks to left and bottom
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    # plot losses
    loss_list = ['VAE Loss', 'Latent Loss', 'Reconstruction Loss']
    color_list = np.linspace(0.2, 0.8, len(losses))
    for i in trange(len(losses), desc='Plotting Epoch Losses', disable=not verbose):
        ax.plot(np.arange(1, losses[i].shape[0]+1), losses[i].mean(1), color=cmap(color_list[i]), label=loss_list[i])
    ax.legend(loc='upper right')
    # label axes
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    # save figure
    fig.savefig(file_name)
    plt.close()


def plot_losses(losses, cmap,
                name, lattice_length, interval, num_samples, scaled, seed,
                prfx, verbose=False):
    # file name parameters
    params = (name, lattice_length, interval, num_samples, scaled, seed)
    file_prfx = '{}.{}.{}.{}.{:d}.{}.'.format(*params)+prfx
    plot_batch_losses(losses, cmap, file_prfx, verbose)
    plot_epoch_losses(losses, cmap, file_prfx, verbose)


def plot_diagram(data, fields, temps, cmap, file_prfx, alias):
    # file name parameters
    file_name = os.getcwd()+'/'+file_prfx+'.{}.png'.format(alias)
    # initialize figure and axes
    fig, ax = plt.subplots()
    # initialize colorbar
    div = make_axes_locatable(ax)
    cax = div.append_axes('top', size='5%', pad=0.8)
    # remove spines on top and right
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # set axis ticks to left and bottom
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    # plot diagram
    im = ax.imshow(data, aspect='equal', interpolation='none', origin='lower', cmap=cmap)
    # generate grid
    ax.grid(which='both', axis='both', linestyle='-', color='k', linewidth=1)
    # label ticks
    ax.set_xticks(np.arange(temps.size), minor=True)
    ax.set_yticks(np.arange(fields.size), minor=True)
    ax.set_xticks(np.arange(temps.size)[::4], minor=False)
    ax.set_yticks(np.arange(fields.size)[::4], minor=False)
    ax.set_xticklabels(np.round(temps, 2)[::4], rotation=-60)
    ax.set_yticklabels(np.round(fields, 2)[::4])
    # label axes
    ax.set_xlabel(r'$T$')
    ax.set_ylabel(r'$H$')
    # place colorbal
    fig.colorbar(im, cax=cax, orientation='horizontal', ticks=np.linspace(data.min(), data.max(), 3))
    # save figure
    fig.savefig(file_name)
    plt.close()


def plot_diagrams(m_data, s_data, fields, temps, cmap,
                  name, lattice_length, interval, num_samples, scaled, seed,
                  prfx, alias, verbose=False):
    params = (name, lattice_length, interval, num_samples, scaled, seed)
    file_prfx = '{}.{}.{}.{}.{:d}.{}.'.format(*params)+prfx
    if s_data is not None:
        mm_diag = m_data.mean(2)
        ms_diag = m_data.std(2)
        sm_diag = s_data.mean(2)
        ss_diag = s_data.std(2)
        mm_dim = mm_diag.shape[-1]
        ms_dim = ms_diag.shape[-1]
        sm_dim = sm_diag.shape[-1]
        ss_dim = ss_diag.shape[-1]
        if alias == ['m', 's'] or alias == ['m_p', 's_p']:
            d0, d1 = 'Means', 'Sigmas'
        elif alias == ['bc_err', 'bc_acc']:
            d0, d1 = 'BC Errors', 'BC Accuracies'
        for i in trange(mm_dim, desc='Plotting Mean VAE {}'.format(d0), disable=not verbose):
            plot_diagram(mm_diag[:, :, i], fields, temps, cmap, file_prfx, '{}_m_{}'.format(alias[0], i))
        for i in trange(sm_dim, desc='Plotting Mean VAE {}'.format(d1), disable=not verbose):
            plot_diagram(sm_diag[:, :, i], fields, temps, cmap, file_prfx, '{}_m_{}'.format(alias[1], i))
        for i in trange(ms_dim, desc='Plotting StDv VAE {}'.format(d0), disable=not verbose):
            plot_diagram(ms_diag[:, :, i], fields, temps, cmap, file_prfx, '{}_s_{}'.format(alias[0], i))
        for i in trange(ss_dim, desc='Plotting StDv VAE {}'.format(d1), disable=not verbose):
            plot_diagram(ss_diag[:, :, i], fields, temps, cmap, file_prfx, '{}_s_{}'.format(alias[1], i))
    else:
        zm_diag = m_data.mean(2)
        zs_diag = m_data.std(2)
        zm_dim = zm_diag.shape[-1]
        zs_dim = zs_diag.shape[-1]
        for i in trange(zm_dim, desc='Plotting Mean VAE Encodings', disable=not verbose):
            plot_diagram(zm_diag[:, :, i], fields, temps, cmap, file_prfx, '{}_m_{}'.format(alias, i))
        for i in trange(zs_dim, desc='Plotting StDv VAE Encodings', disable=not verbose):
            plot_diagram(zs_diag[:, :, i], fields, temps, cmap, file_prfx, '{}_s_{}'.format(alias, i))


def get_final_conv_shape(input_shape, conv_number,
                         filter_base_length, filter_length,
                         filter_base_stride, filter_stride,
                         filter_base, filter_factor, padded):
    ''' calculates final convolutional layer output shape '''
    if padded:
        p = 1
    else:
        p = 0
    out_filters = input_shape[2]*filter_base*filter_factor**(conv_number-1)
    out_dim = (np.array(input_shape[:2], dtype=int)-filter_base_length+p)//filter_base_stride+1
    for i in range(1, conv_number):
        out_dim = (out_dim-filter_length+p)//filter_stride+1
    return tuple(out_dim)+(out_filters,)


def get_filter_number(conv_iter, filter_base, filter_factor):
    ''' calculates the filter count for a given convolutional iteration '''
    return filter_base*filter_factor**(conv_iter)


def get_filter_length_stride(conv_iter, filter_base_length, filter_base_stride, filter_length, filter_stride):
    ''' calculates filter length and stride for a given convolutional iteration '''
    if conv_iter == 0:
        return filter_base_length, filter_base_stride
    else:
        return filter_length, filter_stride


class VAE():
    '''
    VAE Model
    Variational autoencoder modeling of the Ising spin configurations
    '''
    def __init__(self, input_shape=(27, 27, 1), scaled=False, padded=False, conv_number=3,
                 filter_base_length=3, filter_base_stride=3, filter_base=9, filter_length=3, filter_stride=3, filter_factor=9,
                 dropout=False, z_dim=5, kl_anneal=False, alpha=1.0, beta=8.0, lamb=1.0,
                 krnl_init='lecun_normal', act='selu',
                 opt='nadam', la=False, lr=1e-3, batch_size=169, dataset_size=4326400):
        self.eps = 1e-8
        ''' initialize model parameters '''
        self.scaled = scaled
        self.padded = padded
        if self.padded:
            self.padding = 'same'
        else:
            self.padding = 'valid'
        # convolutional parameters
        # number of convolutions
        self.conv_number = conv_number
        # number of filters for first convolution
        self.filter_base = filter_base
        # multiplicative factor for filters in subsequent convolutions
        self.filter_factor = filter_factor
        # filter side length
        self.filter_base_length = filter_base_length
        self.filter_length = filter_length
        # filter stride
        self.filter_base_stride = filter_base_stride
        self.filter_stride = filter_stride
        # convolutional input and output shapes
        self.input_shape = input_shape
        self.final_conv_shape = get_final_conv_shape(self.input_shape, self.conv_number,
                                                     self.filter_base_length, self.filter_length,
                                                     self.filter_base_stride, self.filter_stride,
                                                     self.filter_base, self.filter_factor, self.padded)
        self.dropout = dropout
        # latent and classification dimensions
        # latent dimension
        self.z_dim = z_dim
        # total correlation weights
        self.kl_anneal_b = kl_anneal
        self.alpha, self.beta, self.lamb = alpha, beta, lamb
        # kernel initializer and activation
        self.krnl_init = krnl_init
        self.act = act
        if self.scaled:
            self.dec_out_act = 'sigmoid'
        else:
            self.dec_out_act = 'tanh'
        self.out_init = 'glorot_uniform'
        # optimizer
        self.vae_opt_n = opt
        self.la = la
        # learning rate
        self.lr = lr
        # batch size, dataset size, and log importance weight
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self._set_log_importance_weight()
        self._set_prior_params()
        # loss history
        self.vae_loss_history = []
        self.tc_loss_history = []
        self.rc_loss_history = []
        # past epochs (changes if loading past trained model)
        self.past_epochs = 0
        # checkpoint managers
        self.vae_mngr = None
        # build full model
        self._build_model()


    def get_file_prefix(self):
        ''' gets parameter tuple and filename string prefix '''
        params = (self.conv_number,
                  self.filter_base_length, self.filter_base_stride, self.filter_base,
                  self.filter_length, self.filter_stride, self.filter_factor,
                  self.dropout, self.z_dim, self.kl_anneal_b, self.alpha, self.beta, self.lamb,
                  self.krnl_init, self.act,
                  self.vae_opt_n, self.la, self.lr,
                  self.batch_size)
        file_name = 'btcvae.{}.{}.{}.{}.{}.{}.{}.{:d}.{}.{:d}.{:.0e}.{:.0e}.{:.0e}.{}.{}.{}.{:d}.{:.0e}.{}'.format(*params)
        return file_name


    def scale_configurations(self, x):
        return (x+1)/2


    def _set_log_importance_weight(self):
        ''' logarithmic importance weights for minibatch stratified sampling '''
        n, m = self.dataset_size, self.batch_size-1
        strw = np.float32(n-m)/np.float32(n*m)
        w = np.ones((self.batch_size, self.batch_size), dtype=np.float32)/m
        w.reshape(-1)[::m+1] = 1./n
        w.reshape(-1)[1::m+1] = strw
        w[m-1, 0] = strw
        self.log_importance_weight = K.log(K.constant(w, dtype=tf.float32))
        return


    def _set_prior_params(self):
        # mu = 0, stdv = 1 => log(var) = 0
        self.mu_prior = K.constant(np.zeros(shape=(self.batch_size, self.z_dim)), dtype=tf.float32)
        self.logvar_prior = K.constant(np.zeros(shape=(self.batch_size, self.z_dim)), dtype=tf.float32)
        return


    def sample_gaussian(self, beta):
        ''' samples a point in a multivariate gaussian distribution '''
        mu, logvar = beta
        return mu+K.exp(0.5*logvar)*K.random_normal(shape=(self.batch_size, self.z_dim))


    def sample_logistic(self, shape):
        u = K.random_uniform(shape, 0.0, 1.0)
        l = K.log(u+self.eps)-K.log(1-u+self.eps)
        return l


    def sample_bernoulli(self, p):
        logp = tf.math.log_sigmoid(p)
        logq = tf.math.log_sigmoid(-p)
        l = self.sample_logistic(K.int_shape(p))
        z = logp-logq+l
        return 1./(1.+K.exp(-100*z))


    def gauss_log_density(self, z, beta=None):
        ''' logarithmic probability density for multivariate gaussian distribution given samples z and parameters beta = (mu, log(var)) '''
        if beta is None:
            mu, logvar = self.mu_prior, self.logvar_prior
        else:
            mu, logvar = beta
        norm = K.log(2*np.pi)
        zsc = (z-mu)*K.exp(-0.5*logvar)
        return -0.5*(zsc**2+logvar+norm)


    def log_sum_exp(self, z):
        ''' numerically stable logarithmic sum of exponentials '''
        m = K.max(z, axis=1, keepdims=True)
        u = z-m
        m = K.squeeze(m, 1)
        return m+K.log(K.sum(K.exp(u), axis=1, keepdims=False))


    def total_correlation_loss(self):
        # log p(z)
        logpz = K.sum(K.reshape(self.gauss_log_density(self.z), shape=(self.batch_size, -1)), axis=1)
        # log q(z|x)
        logqz_x = K.sum(K.reshape(self.gauss_log_density(self.z, (self.mu, self.logvar)), shape=(self.batch_size, -1)), axis=1)
        # log q(z) ~ log (1/MN) sum_m q(z|x_m) = -log(MN)+log(sum_m(exp(q(z|x_m))))
        _logqz = self.gauss_log_density(K.reshape(self.z, shape=(self.batch_size, 1, self.z_dim)),
                                        (K.reshape(self.mu, shape=(1, self.batch_size, self.z_dim)),
                                         K.reshape(self.logvar, shape=(1, self.batch_size, self.z_dim))))
        logqz_prodmarginals = K.sum(self.log_sum_exp(K.reshape(self.log_importance_weight, shape=(self.batch_size, self.batch_size, 1))+_logqz), axis=1)
        logqz = self.log_sum_exp(self.log_importance_weight+K.sum(_logqz, axis=2))
        # alpha controls index-code mutual information
        # beta controls total correlation
        # gamma controls dimension-wise kld
        melbo = -self.alpha*(logqz_x-logqz)-self.beta*(logqz-logqz_prodmarginals)-self.lamb*(logqz_prodmarginals-logpz)
        return -self.kl_anneal*melbo


    def kullback_leibler_divergence_loss(self):
        return -0.5*self.kl_anneal*self.beta*K.sum(1.+self.logvar-K.square(self.mu)-K.exp(self.logvar), axis=-1)


    def reconstruction_loss(self):
        if not self.scaled:
            x = self.scale_configurations(self.enc_x_input)
            x_hat = self.scale_configurations(self.x_output)
        else:
            x = self.enc_x_input
            x_hat = self.x_output
        return -K.sum(K.reshape(x*K.log(x_hat+self.eps)+(1.-x)*K.log(1.-x_hat+self.eps), shape=(self.batch_size, -1)), axis=-1)


    def _build_model(self):
        ''' builds each component of the VAE model '''
        self._build_encoder()
        self._build_decoder()
        self._build_vae()


    def _build_encoder(self):
        ''' builds encoder model '''
        # takes sample (real or fake) as input
        self.enc_x_input = Input(batch_shape=(self.batch_size,)+self.input_shape, name='enc_x_input')
        conv = self.enc_x_input
        # iterative convolutions over input
        for i in range(self.conv_number):
            filter_number = get_filter_number(i, self.filter_base, self.filter_factor)
            filter_length, filter_stride = get_filter_length_stride(i, self.filter_base_length, self.filter_base_stride, self.filter_length, self.filter_stride)
            conv = Conv2D(filters=filter_number, kernel_size=filter_length,
                          kernel_initializer=self.krnl_init,
                          padding=self.padding, strides=filter_stride,
                          name='enc_conv_{}'.format(i))(conv)
            if self.act == 'lrelu':
                conv = LeakyReLU(alpha=0.1, name='enc_conv_lrelu_{}'.format(i))(conv)
                conv = BatchNormalization(name='enc_conv_batchnorm_{}'.format(i))(conv)
                if self.dropout:
                    conv = SpatialDropout2D(rate=0.5, name='enc_conv_drop_{}'.format(i))(conv)
            elif self.act == 'selu':
                conv = Activation(activation='selu', name='enc_conv_selu_{}'.format(i))(conv)
                if self.dropout:
                    conv = AlphaDropout(rate=0.5, noise_shape=(self.batch_size, 1, 1, filter_number), name='enc_conv_drop_{}'.format(i))(conv)
        # flatten final convolutional layer
        x = Flatten(name='enc_fltn_0')(conv)
        u = 0
        if self.final_conv_shape[:2] != (1, 1):
            # dense layer
            x = Dense(units=np.prod(self.final_conv_shape),
                      kernel_initializer=self.krnl_init,
                      name='enc_dense_{}'.format(u))(x)
            if self.act == 'lrelu':
                x = LeakyReLU(alpha=0.1, name='enc_dense_lrelu_{}'.format(u))(x)
                x = BatchNormalization(name='enc_dense_batchnorm_{}'.format(u))(x)
            elif self.act == 'selu':
                x = Activation(activation='selu', name='enc_dense_selu_{}'.format(u))(x)
            u += 1
        x = Dense(units=128,
                  kernel_initializer=self.krnl_init,
                  name='enc_dense_{}'.format(u))(x)
        if self.act == 'lrelu':
            x = LeakyReLU(alpha=0.1, name='enc_dense_lrelu_{}'.format(u))(x)
            x = BatchNormalization(name='enc_dense_batchnorm_{}'.format(u))(x)
        elif self.act == 'selu':
            x = Activation(activation='selu', name='enc_dense_selu_{}'.format(u))(x)
        u += 1
        if np.any(np.array([self.alpha, self.beta, self.lamb]) > 0):
            # mean
            self.mu = Dense(units=self.z_dim,
                            kernel_initializer=self.out_init, activation='linear',
                            name='enc_mu_ouput')(x)
            # logarithmic variance
            self.logvar = Dense(units=self.z_dim,
                                kernel_initializer=self.out_init, activation='linear',
                                name='enc_logvar_ouput')(x)
            # latent space
            self.z = Lambda(self.sample_gaussian, output_shape=(self.z_dim,), name='enc_z_output')([self.mu, self.logvar])
            # build encoder
            self.encoder = Model(inputs=[self.enc_x_input], outputs=[self.mu, self.logvar, self.z],
                                 name='encoder')
        else:
            # latent space
            self.z = Dense(self.z_dim, kernel_initializer=self.out_init, activation='sigmoid',
                           name='enc_z_ouput')(x)
            # build encoder
            self.encoder = Model(inputs=[self.enc_x_input], outputs=[self.z],
                                 name='encoder')


    def _build_decoder(self):
        ''' builds decoder model '''
        # latent unit gaussian and categorical inputs
        self.dec_z_input = Input(batch_shape=(self.batch_size, self.z_dim), name='dec_z_input')
        x = self.dec_z_input
        # dense layer with same feature count as final convolution
        u = 0
        x = Dense(units=128,
                  kernel_initializer=self.krnl_init,
                  name='dec_dense_{}'.format(u))(x)
        if self.act == 'lrelu':
            x = LeakyReLU(alpha=0.1, name='dec_dense_lrelu_{}'.format(u))(x)
            x = BatchNormalization(name='dec_dense_batchnorm_{}'.format(u))(x)
        elif self.act == 'selu':
            x = Activation(activation='selu', name='dec_dense_selu_{}'.format(u))(x)
        u += 1
        if self.final_conv_shape[:2] != (1, 1):
            # repeated dense layer
            x = Dense(units=np.prod(self.final_conv_shape),
                      kernel_initializer=self.krnl_init,
                      name='dec_dense_{}'.format(u))(x)
            if self.act == 'lrelu':
                x = LeakyReLU(alpha=0.1, name='dec_dense_lrelu_{}'.format(u))(x)
                x = BatchNormalization(name='dec_dense_batchnorm_{}'.format(u))(x)
            elif self.act == 'selu':
                x = Activation(activation='selu', name='dec_dense_selu_{}'.format(u))(x)
            u += 1
        x = Dense(units=np.prod(self.final_conv_shape),
                  kernel_initializer=self.krnl_init,
                  name='dec_dense_{}'.format(u))(x)
        if self.act == 'lrelu':
            x = LeakyReLU(alpha=0.1, name='dec_dense_lrelu_{}'.format(u))(x)
            x = BatchNormalization(name='dec_dense_batchnorm_{}'.format(u))(x)
        elif self.act == 'selu':
            x = Activation(activation='selu', name='dec_dense_selu_{}'.format(u))(x)
        u += 1
        # reshape to final convolution shape
        convt = Reshape(target_shape=self.final_conv_shape, name='dec_rshp_0')(x)
        if self.dropout:
            if self.act == 'lrelu':
                convt = SpatialDropout2D(rate=0.5, name='dec_rshp_drop_0')(convt)
            elif self.act == 'selu':
                convt = AlphaDropout(rate=0.5, noise_shape=(self.batch_size, 1, 1, self.final_conv_shape[-1]), name='dec_rshp_drop_0')(convt)
        u = 0
        # transform to sample shape with transposed convolutions
        for i in range(self.conv_number-1, 0, -1):
            filter_number = get_filter_number(i-1, self.filter_base, self.filter_factor)
            convt = Conv2DTranspose(filters=filter_number, kernel_size=self.filter_length,
                                    kernel_initializer=self.krnl_init,
                                    padding=self.padding, strides=self.filter_stride,
                                    name='dec_convt_{}'.format(u))(convt)
            if self.act == 'lrelu':
                convt = LeakyReLU(alpha=0.1, name='dec_convt_lrelu_{}'.format(u))(convt)
                convt = BatchNormalization(name='dec_convt_batchnorm_{}'.format(u))(convt)
                if self.dropout:
                    convt = SpatialDropout2D(rate=0.5, name='dec_convt_drop_{}'.format(u))(convt)
            elif self.act == 'selu':
                convt = Activation(activation='selu', name='dec_convt_selu_{}'.format(u))(convt)
                if self.dropout:
                    convt = AlphaDropout(rate=0.5, noise_shape=(self.batch_size, 1, 1, filter_number), name='dec_convt_drop_{}'.format(u))(convt)
            u += 1
        self.dec_x_output = Conv2DTranspose(filters=1, kernel_size=self.filter_base_length,
                                            kernel_initializer=self.out_init, activation=self.dec_out_act,
                                            padding=self.padding, strides=self.filter_base_stride,
                                            name='dec_x_output')(convt)
        # build decoder
        self.decoder = Model(inputs=[self.dec_z_input], outputs=[self.dec_x_output],
                             name='decoder')


    def _build_vae(self):
        ''' builds variational autoencoder network '''
        self.kl_anneal = Input(batch_shape=(self.batch_size,), name='kl_anneal')
        # build VAE
        if np.all(np.array([self.alpha, self.beta, self.lamb]) == 0):
            self.x_output = self.decoder(self.encoder(self.enc_x_input))
            self.vae = Model(inputs=[self.enc_x_input], outputs=[self.x_output],
                             name='variational_autoencoder')
        elif self.alpha == self.beta == self.lamb:
            self.x_output = self.decoder(self.encoder(self.enc_x_input)[2])
            self.vae = Model(inputs=[self.enc_x_input, self.kl_anneal], outputs=[self.x_output],
                             name='variational_autoencoder')
            tc_loss = self.kl_anneal*self.kullback_leibler_divergence_loss()
            self.vae.add_loss(tc_loss)
            self.vae.add_metric(tc_loss, name='tc_loss', aggregation='mean')
        elif np.any(np.array([self.alpha, self.beta, self.lamb]) > 0):
            self.x_output = self.decoder(self.encoder(self.enc_x_input)[2])
            self.vae = Model(inputs=[self.enc_x_input, self.kl_anneal], outputs=[self.x_output],
                             name='variational_autoencoder')
            tc_loss = self.kl_anneal*self.total_correlation_loss()
            self.vae.add_loss(tc_loss)
            self.vae.add_metric(tc_loss, name='tc_loss', aggregation='mean')
        # define VAE optimizer
        if self.vae_opt_n == 'sgd':
            self.vae_opt = SGD(learning_rate=self.lr)
        elif self.vae_opt_n == 'sgdm':
            self.vae_opt = SGD(learning_rate=self.lr, momentum=0.5)
        elif self.vae_opt_n == 'nsgd':
            self.vae_opt = SGD(learning_rate=self.lr, momentum=0.5, nesterov=True)
        elif self.vae_opt_n == 'sgdw':
            self.vae_opt = SGDW(weight_decay=1e-4, learning_rate=self.lr)
        elif self.vae_opt_n == 'sgdwm':
            self.vae_opt = SGDW(weight_decay=1e-4, learning_rate=self.lr, momentum=0.5)
        elif self.vae_opt_n == 'nsgdw':
            self.vae_opt = SGDW(weight_decay=1e-4, learning_rate=self.lr, momentum=0.5, nesterov=True)
        elif self.vae_opt_n == 'rmsprop':
            self.vae_opt = RMSprop(learning_rate=self.lr)
        elif self.vae_opt_n == 'rmsprop_cent':
            self.vae_opt = RMSprop(learning_rate=self.lr, centered=True)
        elif self.vae_opt_n == 'adam':
            self.vae_opt = Adam(learning_rate=self.lr, beta_1=0.5)
        elif self.vae_opt_n == 'adam_ams':
            self.vae_opt = Adam(learning_rate=self.lr, beta_1=0.5, amsgrad=True)
        elif self.vae_opt_n == 'adamw':
            self.vae_opt = AdamW(weight_decay=1e-4, learning_rate=self.lr, beta_1=0.5)
        elif self.vae_opt_n == 'adamw_ams':
            self.vae_opt = AdamW(weight_decay=1e-4, learning_rate=self.lr, beta_1=0.5, amsgrad=True)
        elif self.vae_opt_n == 'adamax':
            self.vae_opt = Adamax(learning_rate=self.lr, beta_1=0.5)
        elif self.vae_opt_n == 'adamax_ams':
            self.vae_opt = Adamax(learning_rate=self.lr, beta_1=0.5, amsgrad=True)
        elif self.vae_opt_n == 'nadam':
            self.vae_opt = Nadam(learning_rate=self.lr, beta_1=0.5)
        elif self.vae_opt_n == 'novograd':
            self.vae_opt = NovoGrad(learning_rate=self.lr, beta_1=0.5)
        elif self.vae_opt_n == 'novograd_ams':
            self.vae_opt = NovoGrad(learning_rate=self.lr, beta_1=0.5, amsgrad=True)
        elif self.vae_opt_n == 'lazy_adam':
            self.vae_opt = LazyAdam(learning_rate=self.lr, beta_1=0.5)
        elif self.vae_opt_n == 'lazy_adam_ams':
            self.vae_opt = LazyAdam(learning_rate=self.lr, beta_1=0.5, amsgrad=True)
        elif self.vae_opt_n == 'rectified_adam':
            self.vae_opt = RectifiedAdam(learning_rate=self.lr, beta_1=0.5)
        elif self.vae_opt_n == 'rectified_adam_ams':
            self.vae_opt = RectifiedAdam(learning_rate=self.lr, beta_1=0.5, amsgrad=True)
        elif self.vae_opt_n == 'yogi':
            self.vae_opt = Yogi(learning_rate=self.lr, beta_1=0.5)
        if self.la:
            self.vae_opt = Lookahead(self.vae_opt)
        # compile VAE
        rc_loss = self.reconstruction_loss()
        self.vae.add_loss(rc_loss)
        self.vae.add_metric(rc_loss, name='rc_loss', aggregation='mean')
        self.vae.compile(optimizer=self.vae_opt)


    def encode(self, x_batch, verbose=False):
        ''' encoder input configurations '''
        return self.encoder.predict(x_batch, batch_size=self.batch_size, verbose=verbose)


    def generate(self, beta_batch, verbose=False):
        ''' generate new configurations using samples from the latent distribution '''
        # sample latent space
        if np.any(np.array([self.alpha, self.beta, self.lamb]) > 0):
            if len(beta_batch) == 2:
                z_batch = sample_gaussian(beta_batch)
            else:
                z_batch = beta_batch
        else:
            z_batch = beta_batch
        # generate configurations
        return self.decoder.predict(z_batch, batch_size=self.batch_size, verbose=verbose)


    def model_summaries(self):
        ''' print model summaries '''
        self.encoder.summary()
        self.decoder.summary()
        self.vae.summary()


    def save_weights(self, name, lattice_length, interval, num_samples, scaled, seed):
        ''' save weights to file '''
        # file parameters
        params = (name, lattice_length, interval, num_samples, scaled, seed)
        file_name = os.getcwd()+'/{}.{}.{}.{}.{:d}.{}.'.format(*params)+self.get_file_prefix()+'.weights.h5'
        # save weights
        self.vae.save_weights(file_name)


    def load_weights(self, name, lattice_length, interval, num_samples, scaled, seed):
        ''' load weights from file '''
        # file parameters
        params = (name, lattice_length, interval, num_samples, scaled, seed)
        file_name = os.getcwd()+'/{}.{}.{}.{}.{:d}.{}.'.format(*params)+self.get_file_prefix()+'.weights.h5'
        # load weights
        self.vae.load_weights(file_name, by_name=True)


    def get_losses(self):
        ''' retrieve loss histories '''
        # reshape arrays into (epochs, batches)
        vae_loss = np.array(self.vae_loss_history).reshape(-1, self.num_batches)
        tc_loss = np.array(self.tc_loss_history).reshape(-1, self.num_batches)
        rc_loss = np.array(self.rc_loss_history).reshape(-1, self.num_batches)
        return vae_loss, tc_loss, rc_loss


    def save_losses(self, name, lattice_length, interval, num_samples, scaled, seed):
        ''' save loss histories to file '''
        # retrieve losses
        losses = self.get_losses()
        # file parameters
        params = (name, lattice_length, interval, num_samples, scaled, seed)
        file_name = os.getcwd()+'/{}.{}.{}.{}.{:d}.{}.'.format(*params)+self.get_file_prefix()+'.loss.npy'
        np.save(file_name, np.stack(losses, axis=-1))


    def load_losses(self, name, lattice_length, interval, num_samples, scaled, seed):
        ''' load loss histories from file '''
        # file parameters
        params = (name, lattice_length, interval, num_samples, scaled, seed)
        file_name = os.getcwd()+'/{}.{}.{}.{}.{:d}.{}.'.format(*params)+self.get_file_prefix()+'.loss.npy'
        losses = np.load(file_name)
        # set past epochs
        self.past_epochs = losses.shape[0]
        self.num_batches = losses.shape[1]
        # change loss histories into lists
        self.vae_loss_history = list(losses[:, :, 0].reshape(-1))
        self.tc_loss_history = list(losses[:, :, 1].reshape(-1))
        self.rc_loss_history = list(losses[:, :, 2].reshape(-1))


    def initialize_checkpoint_managers(self, name, lattice_length, interval, num_samples, scaled, seed):
        ''' initialize training checkpoint managers '''
        # initialize checkpoints
        self.vae_ckpt = Checkpoint(step=tf.Variable(0), optimizer=self.vae_opt, net=self.vae)
        # file parameters
        params = (name, lattice_length, interval, num_samples, scaled, seed)
        directory = os.getcwd()+'/{}.{}.{}.{}.{:d}.{}.'.format(*params)+self.get_file_prefix()+'.ckpts'
        # initialize checkpoint managers
        self.vae_mngr = CheckpointManager(self.vae_ckpt, directory+'/vae/', max_to_keep=4)


    def load_latest_checkpoint(self, name, lattice_length, interval, num_samples, scaled, seed):
        ''' load latest training checkpoint from file '''
        # initialize checkpoint managers
        self.initialize_checkpoint_managers(name, lattice_length, interval, num_samples, scaled, seed)
        self.load_losses(name, lattice_length, interval, num_samples, scaled, seed)
        # file parameters
        params = (name, lattice_length, interval, num_samples, scaled, seed)
        directory = os.getcwd()+'/{}.{}.{}.{}.{:d}.{}.'.format(*params)+self.get_file_prefix()+'.ckpts'
        # restore checkpoints
        self.vae_ckpt.restore(self.vae_mngr.latest_checkpoint).assert_consumed()


    def get_training_indices(self):
        ''' retrieve class-balancing training indices '''
        # number of square subsectors in (h, t) space
        n_sr = np.int32(self.num_fields*self.num_temps/self.batch_size)
        # side length of (h, t) space in subsectors
        n_sr_l = np.int32(np.sqrt(n_sr))
        # side length of subsector
        sr_l = np.int32(np.sqrt(self.batch_size))
        # indices for top left subsector
        sr_indices = np.stack(np.meshgrid(np.arange(sr_l),
                                          np.arange(sr_l)), axis=-1).reshape(-1, 2)[:, ::-1]
        # all indices for subsectors
        indices = np.array([[sr_indices+sr_l*np.array([i,j]).reshape(1, -1) for i in range(n_sr_l)] for j in range(n_sr_l)])
        # flattened subsector indices
        flat_indices = np.ravel_multi_index(indices.reshape(-1, 2).T, dims=(self.num_fields, self.num_temps))
        # shuffle indices within each subsector
        for i in range(n_sr):
            flat_indices[self.batch_size*i:self.batch_size*(i+1)] = np.random.permutation(flat_indices[self.batch_size*i:self.batch_size*(i+1)])
        # shift indices to balance batches by subsector
        shift_indices = np.concatenate([flat_indices[i::self.batch_size] for i in range(self.batch_size)])
        return shift_indices


    def randomly_order_training_data(self, x_train):
        ''' reorder training data by random indices '''
        indices = np.random.permutation(self.num_fields*self.num_temps*self.num_samples)
        return x_train.reshape(self.num_fields*self.num_temps*self.num_samples, *self.input_shape)[indices]


    def reorder_training_data(self, x_train):
        ''' reorder training data by class-balancing indices '''
        x_train = x_train.reshape(self.num_fields*self.num_temps, self.num_samples, *self.input_shape)[self.get_training_indices()]
        return np.moveaxis(x_train, 0, 1).reshape(self.num_fields*self.num_temps*self.num_samples, *self.input_shape)


    def extract_unique_data(self, x_train):
        ''' extract unique samples from data '''
        x_train = np.unique(x_train.reshape(self.num_fields*self.num_temps*self.num_samples, *self.input_shape), axis=0)
        return x_train


    def draw_random_batch(self, x_train):
        ''' draws random batch from data '''
        indices = np.random.permutation(x_train.shape[0])[:self.batch_size]
        return x_train[indices]


    def draw_indexed_batch(self, x_train, j):
        ''' draws batch j '''
        return x_train[self.batch_size*j:self.batch_size*(j+1)]


    def train_vae(self, x_batch, kl_anneal):
        ''' train VAE '''
        # VAE losses
        if np.any(np.array([self.alpha, self.beta, self.lamb]) > 0):
            vae_loss, tc_loss, rc_loss = self.vae.train_on_batch([x_batch, kl_anneal])
            self.vae_loss_history.append(vae_loss)
            self.tc_loss_history.append(tc_loss)
            self.rc_loss_history.append(rc_loss)
        else:
            vae_loss, rc_loss = self.vae.train_on_batch(x_batch)
            self.vae_loss_history.append(vae_loss)
            self.tc_loss_history.append(0)
            self.rc_loss_history.append(rc_loss)


    def rolling_loss_average(self, epoch, batch):
        ''' calculate rolling loss averages over batches during training '''
        epoch = epoch+self.past_epochs
        # catch case where there are no calculated losses yet
        if batch == 0:
            vae_loss = 0
            tc_loss = 0
            rc_loss = 0
        # calculate rolling average
        else:
            # start index for current epoch
            start = self.num_batches*epoch
            # stop index for current batch (given epoch)
            stop = self.num_batches*epoch+batch+1
            # average loss histories
            vae_loss = np.mean(self.vae_loss_history[start:stop])
            tc_loss = np.mean(self.tc_loss_history[start:stop])
            rc_loss = np.mean(self.rc_loss_history[start:stop])
        return vae_loss, tc_loss, rc_loss


    def fit(self, x_train, num_epochs=4, save_step=4, random_sampling=False, verbose=False):
        ''' fit model '''
        self.num_fields, self.num_temps, self.num_samples, _, _, = x_train.shape
        self.num_batches = (self.num_fields*self.num_temps*self.num_samples)//self.batch_size
        if random_sampling:
            x_train = self.extract_unique_data(x_train)
        else:
            x_train = self.reorder_training_data(x_train)
        num_epochs += self.past_epochs
        t = np.linspace(0., 1., num_epochs*self.num_batches)
        if np.all(np.array([self.alpha, self.beta, self.lamb]) == 0):
            kl_anneal = np.zeros((num_epochs, self.num_batches))
        elif not self.kl_anneal_b:
            kl_anneal = np.ones((num_epochs, self.num_batches))
        else:
            n_cycles = 4
            linear_kl_anneal = np.linspace(0., 1., num_epochs*self.num_batches//(2*n_cycles))
            constant_kl_anneal = np.ones(num_epochs*self.num_batches//(2*n_cycles))
            cycle_kl_anneal = np.concatenate((linear_kl_anneal, constant_kl_anneal))
            kl_anneal = np.tile(cycle_kl_anneal, n_cycles).reshape(num_epochs, self.num_batches)
        if 'sgd' in self.vae_opt_n and 'n' not in self.vae_opt_n:
            lr_rampu_supconv = np.linspace(0.1, 1.0, num_epochs*self.num_batches//2)
            lr_rampd_supconv = np.linspace(1.0, 0.1, 3*num_epochs*self.num_batches//8)
            lr_min_supconv = np.linspace(0.1, 0.01, num_epochs*self.num_batches//8)
            lr_factor = np.concatenate((lr_rampu_supconv, lr_rampd_supconv, lr_min_supconv)).reshape(num_epochs, self.num_batches)
            if 'm' in self.vae_opt_n and not self.la:
                m_rampd_supconv = np.linspace(0.95, 0.85, num_epochs*self.num_batches//2)
                m_rampu_supconv = np.linspace(0.85, 0.95, 3*num_epochs*self.num_batches//8)
                m_max_supconv = 0.95*np.ones(num_epochs*self.num_batches//8)
                m_factor = np.concatenate((m_rampd_supconv, m_rampu_supconv, m_max_supconv)).reshape(num_epochs, self.num_batches)
            else:
                m_factor = np.ones((num_epochs, self.num_batches))
        else:
            # lr_constant = np.ones(7*num_epochs*self.num_batches//8)
            # lr_decay = np.linspace(1.0, 0.1, num_epochs*self.num_batches//8)
            # lr_factor = np.concatenate((lr_constant, lr_decay)).reshape(num_epochs, self.num_batches)
            lr_factor = np.ones((num_epochs, self.num_batches))
        # loop through epochs
        for i in range(self.past_epochs, num_epochs):
            # construct progress bar for current epoch
            if random_sampling:
                batch_range = trange(self.num_batches, desc='', disable=not verbose)
            else:
                b = np.arange(self.num_batches)
                np.random.shuffle(b)
                batch_range = tqdm(b, desc='', disable=not verbose)
            # loop through batches
            u = 0
            for j in batch_range:
                # set batch loss description
                batch_loss = self.rolling_loss_average(i, u)
                batch_acc = np.exp(-batch_loss[-1]/np.prod(self.input_shape))
                desc = 'Epoch: {}/{} LR Fctr: {:.4f} KL Anl: {:.4f} VAE Lss: {:.4f} TCKLD Lss: {:.4f} RCNST Lss: {:.4f} RCNST Acc: {:.4f}'.format(i+1, num_epochs, lr_factor[i, u], kl_anneal[i, u], *batch_loss, batch_acc)
                batch_range.set_description(desc)
                # fetch batch
                if random_sampling:
                    x_batch = self.draw_random_batch(x_train)
                else:
                    x_batch = self.draw_indexed_batch(x_train, j)
                # train VAE
                if self.vae_opt_n == 'sgdm' or self.vae_opt_n == 'sgdwm':
                    self.vae_opt.momentum = m_factor[i, u]
                self.vae_opt.learning_rate = lr_factor[i, u]*self.lr
                self.train_vae(x_batch=x_batch, kl_anneal=kl_anneal[i, u]*np.ones(self.batch_size))
                u += 1
            # if checkpoint managers are initialized
            if self.vae_mngr is not None:
                # increment checkpoint
                self.vae_ckpt.step.assign_add(1)
                # if save step is reached
                if np.int32(self.vae_ckpt.step) % save_step == 0:
                    # save model checkpoint
                    vae_save_path = self.vae_mngr.save()
                    print('Checkpoint DSC: {}'.format(vae_save_path))

if __name__ == '__main__':
    (VERBOSE, RSTRT, PLOT, PARALLEL, GPU, THREADS,
     NAME, N, I, NS, SC, CP,
     CN, FBL, FBS, FB, FL, FS, FF,
     DO, ZD, KA, ALPHA, BETA, LAMBDA,
     KI, AN, OPT, LA, LR,
     BS, RS, EP, SEED) = parse_args()

    plt.rc('font', family='sans-serif')
    FTSZ = 28
    FIGW = 16
    PPARAMS = {'figure.figsize': (FIGW, FIGW),
               'lines.linewidth': 4.0,
               'legend.fontsize': FTSZ,
               'axes.labelsize': FTSZ,
               'axes.titlesize': FTSZ,
               'axes.linewidth': 2.0,
               'xtick.labelsize': FTSZ,
               'xtick.major.size': 20,
               'xtick.major.width': 2.0,
               'ytick.labelsize': FTSZ,
               'ytick.major.size': 20,
               'ytick.major.width': 2.0,
               'font.size': FTSZ}
    plt.rcParams.update(PPARAMS)
    CM = plt.get_cmap('plasma')

    H, T, CONF, THRM = load_data(NAME, N, I, NS, SC, SEED, VERBOSE)
    del THRM
    NH, NT = H.size, T.size
    IS = (N, N, 1)

    if SEED == -1:
        np.random.seed(None)
    else:
        np.random.seed(SEED)
        tf.random.set_seed(SEED)
    if GPU:
        DEVICE = '/GPU:0'
    else:
        if not PARALLEL:
            THREADS = 1
        DEVICE = '/CPU:0'
        tf.config.threading.set_intra_op_parallelism_threads(THREADS)
        tf.config.threading.set_inter_op_parallelism_threads(THREADS)
    tf.device(DEVICE)

    K.clear_session()
    MDL = VAE(IS, SC, CP, CN, FBL, FBS, FB, FL, FS, FF, DO, ZD, KA, ALPHA, BETA, LAMBDA, KI, AN, OPT, LA, LR, BS, NH*NT*NS)
    PRFX = MDL.get_file_prefix()
    if RSTRT:
        MDL.load_losses(NAME, N, I, NS, SC, SEED)
        MDL.load_weights(NAME, N, I, NS, SC, SEED)
        if VERBOSE:
            MDL.model_summaries()
        # MDL.load_latest_checkpoint(NAME, N, I, NS, SC, SEED)
        MDL.fit(CONF, num_epochs=EP, save_step=EP, random_sampling=RS, verbose=VERBOSE)
        MDL.save_losses(NAME, N, I, NS, SC, SEED)
        MDL.save_weights(NAME, N, I, NS, SC, SEED)
    else:
        try:
            MDL.load_losses(NAME, N, I, NS, SC, SEED)
            MDL.load_weights(NAME, N, I, NS, SC, SEED)
            if VERBOSE:
                MDL.model_summaries()
        except:
            if VERBOSE:
                MDL.model_summaries()
            # MDL.initialize_checkpoint_managers(NAME, N, I, NS, SC, SEED)
            MDL.fit(CONF, num_epochs=EP, save_step=EP, random_sampling=RS, verbose=VERBOSE)
            MDL.save_losses(NAME, N, I, NS, SC, SEED)
            MDL.save_weights(NAME, N, I, NS, SC, SEED)
    L = MDL.get_losses()
    if np.any(np.array([ALPHA, BETA, LAMBDA]) > 0):
        MU, LOGVAR, Z = MDL.encode(CONF.reshape(-1, *IS), VERBOSE)
        SIGMA = np.exp(0.5*LOGVAR)
        PMMDL = PCA(n_components=ZD)
        PMU = PMMDL.fit_transform(MU)
        PSMDL = PCA(n_components=ZD)
        PSIGMA = PSMDL.fit_transform(SIGMA)
        PZMDL = PCA(n_components=ZD)
        PZ = PZMDL.fit_transform(Z)
        MU = MU.reshape(NH, NT, NS, ZD)
        SIGMA = SIGMA.reshape(NH, NT, NS, ZD)
        Z = Z.reshape(NH, NT, NS, ZD)
        PMU = PMU.reshape(NH, NT, NS, ZD)
        PSIGMA = PSIGMA.reshape(NH, NT, NS, ZD)
        PZ = PZ.reshape(NH, NT, NS, ZD)
        save_output_data(MU, 'vae_mean', NAME, N, I, NS, SC, SEED, PRFX)
        save_output_data(SIGMA, 'vae_sigma', NAME, N, I, NS, SC, SEED, PRFX)
        if PLOT:
            plot_diagrams(MU, SIGMA, H, T, CM, NAME, N, I, NS, SC, SEED, PRFX, ['m', 's'], VERBOSE)
            plot_diagrams(PMU, PSIGMA, H, T, CM, NAME, N, I, NS, SC, SEED, PRFX, ['m_p', 's_p'], VERBOSE)
            plot_diagrams(Z, None, H, T, CM, NAME, N, I, NS, SC, SEED, PRFX, 'z', VERBOSE)
            plot_diagrams(PZ, None, H, T, CM, NAME, N, I, NS, SC, SEED, PRFX, 'z_p', VERBOSE)
        del MU, LOGVAR, SIGMA, PMU, PSIGMA, PZ
        X = MDL.generate(Z.reshape(-1, ZD), VERBOSE).astype(np.float16)
        del Z
    else:
        Z = MDL.encode(CONF.reshape(-1, *IS), VERBOSE)
        PMDL = PCA(n_components=ZD)
        PZ = PMDL.fit_transform(Z)
        Z = Z.reshape(NH, NT, NS, ZD)
        PZ = PZ.reshape(NH, NT, NS, ZD)
        save_output_data(Z, 'vae_encoding', NAME, N, I, NS, SC, SEED, PRFX)
        if PLOT:
            plot_diagrams(Z, None, H, T, CM, NAME, N, I, NS, SC, SEED, PRFX, 'z', VERBOSE)
            plot_diagrams(PZ, None, H, T, CM, NAME, N, I, NS, SC, SEED, PRFX, 'z_p', VERBOSE)
        del PZ
        X = MDL.generate(Z.reshape(-1, ZD), VERBOSE).astype(np.float16)
        del Z
    BCERR, BCACC = binary_crossentropy_accuracy(CONF.reshape(-1, *IS), X, SC)
    del CONF, X
    save_output_data(BCERR, 'bc_err', NAME, N, I, NS, SC, SEED, PRFX)
    save_output_data(BCACC, 'bc_acc', NAME, N, I, NS, SC, SEED, PRFX)
    if VERBOSE:
        print('Mean Error: {} STD Error: {}'.format(*(BCERR.mean(0))))
        print('Mean Accuracy: {} STD Accuracy: {}'.format(*(BCACC.mean(0))))
    if PLOT:
        plot_diagrams(BCERR.reshape(NH, NT, NS, -1), BCACC.reshape(NH, NT, NS, -1), H, T, CM, NAME, N, I, NS, SC, SEED, PRFX, ['bc_err', 'bc_acc'], VERBOSE)
        plot_bc_error_accuracy(BCERR, BCACC, CM, NAME, N, I, NS, SC, SEED, PRFX, VERBOSE)
        plot_losses(L, CM, NAME, N, I, NS, SC, SEED, PRFX, VERBOSE)
