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
from tensorflow.keras.layers import (Input, Flatten, Reshape, Lambda,
                                     Dense, BatchNormalization, Conv2D, Conv2DTranspose,
                                     Activation, LeakyReLU)
from tensorflow.keras.optimizers import SGD, Adam, Adamax, Nadam
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
    parser.add_argument('-ll', '--lattice_length', help='lattice size (side length)',
                        type=int, default=27)
    parser.add_argument('-si', '--sample_interval', help='interval for selecting phase points (variational autoencoder)',
                        type=int, default=1)
    parser.add_argument('-sn', '--sample_number', help='number of samples per phase point (variational autoencoder)',
                        type=int, default=1024)
    parser.add_argument('-cn', '--conv_number', help='convolutional layer depth',
                        type=int, default=3)
    parser.add_argument('-fl', '--filter_length', help='size of filters in hidden convolutional layers',
                        type=int, default=3)
    parser.add_argument('-fb', '--filter_base', help='base number of filters in hidden convolutional layers',
                        type=int, default=9)
    parser.add_argument('-ff', '--filter_factor', help='multiplicative factor of filters in successive layers',
                        type=int, default=9)
    parser.add_argument('-zd', '--z_dimension', help='sample noise dimension',
                        type=int, default=5)
    parser.add_argument('-a', '--alpha', help='total correlation alpha',
                        type=float, default=1.0)
    parser.add_argument('-b', '--beta', help='total correlation beta',
                        type=float, default=8.0)
    parser.add_argument('-l', '--lamb', help='total correlation lambda',
                        type=float, default=1.0)
    parser.add_argument('-ki', '--kernel_initializer', help='kernel initializer',
                        type=str, default='lecun_normal')
    parser.add_argument('-an', '--activation', help='activation function',
                        type=str, default='selu')
    parser.add_argument('-lr', '--learning_rate', help='learning rate',
                        type=float, default=1e-3)
    parser.add_argument('-ep', '--epochs', help='number of training epochs',
                        type=int, default=32)
    parser.add_argument('-bs', '--batch_size', help='size of batches',
                        type=int, default=169)
    parser.add_argument('-sd', '--random_seed', help='random seed for sample selection and learning',
                        type=int, default=128)
    args = parser.parse_args()
    return (args.verbose, args.restart, args.plot, args.parallel, args.gpu, args.threads,
            args.name, args.lattice_length, args.sample_interval, args.sample_number,
            args.conv_number, args.filter_length, args.filter_base, args.filter_factor,
            args.z_dimension, args.alpha, args.beta, args.lamb,
            args.kernel_initializer, args.activation, args.learning_rate,
            args.epochs, args.batch_size, args.random_seed)


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


def scale_configurations(conf):
    ''' scales input configurations '''
    # (-1, 1) -> (0, 1)
    return (conf+1)/2


def unscale_configurations(conf):
    ''' unscales input configurations '''
    # (0, 1) -> (-1, 1)
    return 2*conf-1


def index_data_by_sample(data, num_fields, num_temps, indices):
    ''' indexes data '''
    # reorders samples independently for each (h, t) according to indices
    return np.array([[data[i, j, indices[i, j]] for j in range(num_temps)] for i in range(num_fields)])


def load_select_scale_data(name, lattice_length, interval, num_samples, seed, verbose=False):
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
    # select_conf = scale_configurations(index_data_by_sample(interval_conf, num_fields, num_temps, indices))
    select_conf = index_data_by_sample(interval_conf, num_fields, num_temps, indices)
    select_thrm = index_data_by_sample(interval_thrm, num_fields, num_temps, indices)
    # save selected data arrays
    np.save(os.getcwd()+'/{}.{}.{}.h.npy'.format(name, lattice_length, interval), interval_fields)
    np.save(os.getcwd()+'/{}.{}.{}.t.npy'.format(name, lattice_length, interval), interval_temps)
    np.save(os.getcwd()+'/{}.{}.{}.{}.{}.conf.npy'.format(name, lattice_length,
                                                          interval, num_samples, seed), select_conf)
    np.save(os.getcwd()+'/{}.{}.{}.{}.{}.thrm.npy'.format(name, lattice_length,
                                                          interval, num_samples, seed), select_thrm)
    return interval_fields, interval_temps, select_conf, select_thrm


def load_data(name, lattice_length, interval, num_samples, seed, verbose=False):
    try:
        # try loading selected data arrays
        fields = np.load(os.getcwd()+'/{}.{}.{}.h.npy'.format(name, lattice_length, interval))
        temps = np.load(os.getcwd()+'/{}.{}.{}.t.npy'.format(name, lattice_length, interval))
        conf = np.load(os.getcwd()+'/{}.{}.{}.{}.{}.conf.npy'.format(name, lattice_length,
                                                                     interval, num_samples, seed))
        thrm = np.load(os.getcwd()+'/{}.{}.{}.{}.{}.thrm.npy'.format(name, lattice_length,
                                                                     interval, num_samples, seed))
        conf = scale_configurations(conf)
        if verbose:
            print(100*'_')
            print('Scaled/selected Ising configurations and thermal parameters/measurements loaded from file')
            print(100*'_')
    except:
        # generate selected data arrays
        (fields, temps,
         conf, thrm) = load_select_scale_data(name, lattice_length,
                                              interval, num_samples, seed, verbose)
        conf = scale_configurations(conf)
        if verbose:
            print(100*'_')
            print('Ising configurations selected/scaled and thermal parameters/measurements selected')
            print(100*'_')
    return fields, temps, conf, thrm


def save_output_data(data, alias,
                     name, lattice_length, interval, num_samples,
                     conv_number, filter_length, filter_base, filter_factor,
                     z_dim, krnl_init, act, lr, batch_size, seed):
    ''' save output data from model '''
    # file parameters
    params = (name, lattice_length, interval, num_samples,
              conv_number, filter_length, filter_base, filter_factor,
              z_dim, krnl_init, act, lr, batch_size, seed, alias)
    file_name = os.getcwd()+'/{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{:.0e}.{}.{}.{}.npy'.format(*params)
    np.save(file_name, data)


def plot_batch_losses(losses, cmap, params, verbose=False):
    file_name = os.getcwd()+'/{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{:.0e}.{}.{}.loss.batch.png'.format(*params)
    # initialize figure and axes
    fig, ax = plt.subplots()
    # remove spines on top and right
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # set axis ticks to left and bottom
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    # plot losses
    loss_list = ['Reconstruction Loss', 'Latent Loss']
    color_list = np.linspace(0.2, 0.8, len(losses))
    for i in trange(len(losses), desc='Plotting Batch Losses', disable=not verbose):
        ax.plot(np.arange(losses[i].size), losses[i].reshape(-1), color=cmap(color_list[i]), label=loss_list[i])
    ax.legend(loc='upper right')
    # label axes
    ax.set_xlabel('Batch')
    ax.set_ylabel('Loss')
    # save figure
    fig.savefig(file_name)
    plt.close()


def plot_epoch_losses(losses, cmap, params, verbose=False):
    file_name = os.getcwd()+'/{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{:.0e}.{}.{}.loss.epoch.png'.format(*params)
    # initialize figure and axes
    fig, ax = plt.subplots()
    # remove spines on top and right
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # set axis ticks to left and bottom
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    # plot losses
    loss_list = ['Reconstruction Loss', 'Latent Loss']
    color_list = np.linspace(0.2, 0.8, len(losses))
    for i in trange(len(losses), desc='Plotting Epoch Losses', disable=not verbose):
        ax.plot(np.arange(losses[i].shape[0]), losses[i].mean(1), color=cmap(color_list[i]), label=loss_list[i])
    ax.legend(loc='upper right')
    # label axes
    ax.set_xlabel('Batch')
    ax.set_ylabel('Loss')
    # save figure
    fig.savefig(file_name)
    plt.close()


def plot_losses(losses, cmap,
                name, lattice_length, interval, num_samples,
                conv_number, filter_length, filter_base, filter_factor,
                z_dim, krnl_init, act, lr, batch_size, seed, verbose=False):
    # file name parameters
    params = (name, lattice_length, interval, num_samples,
              conv_number, filter_length, filter_base, filter_factor,
              z_dim, krnl_init, act, lr, batch_size, seed)
    plot_batch_losses(losses, cmap, params, verbose)
    plot_epoch_losses(losses, cmap, params, verbose)


def plot_diagram(data, fields, temps, cmap, params):
    # file name parameters
    file_name = os.getcwd()+'/{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{:.0e}.{}.{}.{}.png'.format(*params)
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


def plot_diagrams(z_data, m_data, s_data, fields, temps, cmap,
                  name, lattice_length, interval, num_samples,
                  conv_number, filter_length, filter_base, filter_factor,
                  z_dim, krnl_init, act, lr, batch_size, seed, verbose=False):
    params = (name, lattice_length, interval, num_samples,
              conv_number, filter_length, filter_base, filter_factor,
              z_dim, krnl_init, act, lr, batch_size, seed)
    z_diag = z_data.mean(2)
    for i in trange(z_dim, desc='Plotting Latent', disable=not verbose):
        plot_diagram(z_diag[:, :, i], fields, temps, cmap, params+('z_{}'.format(i),))
    if m_data is not None:
        m_diag = m_data.mean(2)
        for i in trange(z_dim, desc='Plotting Means', disable=not verbose):
            plot_diagram(m_diag[:, :, i], fields, temps, cmap, params+('m_{}'.format(i),))
    if s_data is not None:
        s_diag = s_data.mean(2)
        for i in trange(z_dim, desc='Plotting Standard Deviations', disable=not verbose):
            plot_diagram(s_diag[:, :, i], fields, temps, cmap, params+('s_{}'.format(i),))


def get_final_conv_shape(input_shape, conv_number,
                         filter_length, filter_base, filter_factor):
    ''' calculates final convolutional layer output shape '''
    return tuple(np.array(input_shape[:2])//(filter_length**conv_number))+\
           (input_shape[2]*filter_base*filter_factor**(conv_number-1),)


def get_filter_number(conv_iter, filter_base, filter_factor):
    ''' calculates the filter count for a given convolutional iteration '''
    return filter_base*filter_factor**(conv_iter-1)


class VAE():
    '''
    VAE Model
    Variational autoencoder modeling of the Ising spin configurations
    '''
    def __init__(self, input_shape=(27, 27, 1), conv_number=3,
                 filter_length=3, filter_base=9, filter_factor=9,
                 z_dim=5, alpha=1.0, beta=8.0, lamb=1.0,
                 krnl_init='lecun_normal', act='selu',
                 lr=1e-3, batch_size=169, dataset_size=4326400):
        self.eps = 1e-8
        ''' initialize model parameters '''
        # convolutional parameters
        # number of convolutions
        self.conv_number = conv_number
        # number of filters for first convolution
        self.filter_base = filter_base
        # multiplicative factor for filters in subsequent convolutions
        self.filter_factor = filter_factor
        # filter side length
        self.filter_length = filter_length
        # set stride to be same as filter size
        self.filter_stride = filter_length
        # convolutional input and output shapes
        self.input_shape = input_shape
        self.final_conv_shape = get_final_conv_shape(self.input_shape, self.conv_number,
                                                     self.filter_length, self.filter_base, self.filter_factor)
        # latent and classification dimensions
        # latent dimension
        self.z_dim = z_dim
        # total correlation weights
        self.alpha, self.beta, self.lamb = alpha, beta, lamb
        # kernel initializer and activation
        self.krnl_init = krnl_init
        self.act = act
        # learning rate
        self.lr = lr
        # batch size, dataset size, log importance weight, and callbacks
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self.log_importance_weight = self.calculate_log_importance_weight()
        self.callbacks = []
        # build full model
        self._build_model()
    

    def calculate_log_importance_weight(self):
        ''' logarithmic importance weights for minibatch stratified sampling '''
        n, m = self.dataset_size, self.batch_size-1
        strw = (n-m)/(n*m)
        w = K.concatenate((K.concatenate((1/n*K.ones((self.batch_size-2, 1)),
                                          strw*K.ones((1, 1)),
                                          1/n*K.ones((1, 1))), axis=0),
                           strw*K.ones((self.batch_size, 1)),
                           1/m*K.ones((self.batch_size, self.batch_size-2))), axis=1)
        return K.log(w)
    

    def sample_gaussian(self, beta):
        ''' samples a point in a multivariate gaussian distribution '''
        mu, logvar = beta
        return mu+K.exp(0.5*logvar)*K.random_normal(shape=(self.batch_size, self.z_dim))
    

    def gauss_log_prob(self, z, beta=None):
        ''' logarithmic probability for multivariate gaussian distribution given samples z and parameters beta = (mu, log(var)) '''
        if beta is None:
            # mu = 0, stdv = 1 => log(var) = 0
            mu, logvar = K.zeros((self.batch_size, self.z_dim)), K.zeros((self.batch_size, self.z_dim))
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
        logpz = K.sum(K.reshape(self.gauss_log_prob(self.z), (self.batch_size, -1)), 1)
        # log q(z|x)
        logqz_x = K.sum(K.reshape(self.gauss_log_prob(self.z, (self.mu, self.logvar)), (self.batch_size, -1)), 1)
        # log q(z) ~ log (1/MN) sum_m q(z|x_m) = -log(MN)+log(sum_m(exp(q(z|x_m))))
        _logqz = self.gauss_log_prob(K.reshape(self.z, (self.batch_size, 1, self.z_dim)),
                                     (K.reshape(self.mu, (1, self.batch_size, self.z_dim)),
                                     K.reshape(self.logvar, (1, self.batch_size, self.z_dim))))
        logqz_prodmarginals = K.sum(self.log_sum_exp(K.reshape(self.log_importance_weight, (self.batch_size, self.batch_size, 1))+_logqz), 1)
        logqz = self.log_sum_exp(self.log_importance_weight+K.sum(_logqz, axis=2))
        # alpha controls index-code mutual information
        # beta controls total correlation
        # gamma controls dimension-wise kld
        melbo = -self.alpha*(logqz_x-logqz)-self.beta*(logqz-logqz_prodmarginals)-self.lamb*(logqz_prodmarginals-logpz)
        return -K.mean(melbo)

    
    def kullback_leibler_loss(self):
        return 0.5*self.beta*K.mean(K.sum(K.exp(self.logvar)+K.square(self.mu)-self.logvar-1, axis=-1))


    def reconstruction_loss(self, x, x_hat):
        return -K.mean(K.sum(K.reshape(x*K.log(x_hat+self.eps)+(1-x)*K.log(1-x_hat+self.eps), (self.batch_size, -1)), 1))


    def _build_model(self):
        ''' builds each component of the VAE model '''
        self._build_encoder()
        self._build_decoder()
        self._build_vae()
    

    def _build_encoder(self):
        ''' builds encoder model '''
        # takes sample (real or fake) as input
        self.enc_input = Input(batch_shape=(self.batch_size,)+self.input_shape, name='enc_input')
        conv = self.enc_input
        # iterative convolutions over input
        for i in range(1, self.conv_number+1):
            filter_number = get_filter_number(i, self.filter_base, self.filter_factor)
            conv = Conv2D(filters=filter_number, kernel_size=self.filter_length,
                          kernel_initializer=self.krnl_init,
                          padding='valid', strides=self.filter_stride,
                          name='enc_conv_{}'.format(i))(conv)
            if self.act == 'lrelu':
                conv = BatchNormalization(name='enc_conv_batchnorm_{}'.format(i))(conv)
                conv = LeakyReLU(alpha=0.2, name='enc_conv_lrelu_{}'.format(i))(conv)
            if self.act == 'selu':
                conv = Activation(activation='selu', name='enc_conv_selu_{}'.format(i))(conv)
        # flatten final convolutional layer
        x = Flatten(name='enc_fltn_0')(conv)
        if self.final_conv_shape[:2] != (1, 1):
            # dense layer
            x = Dense(units=np.prod(self.final_conv_shape),
                      kernel_initializer=self.krnl_init,
                      name='enc_dense_0')(x)
            if self.act == 'lrelu':
                x = LeakyReLU(alpha=0.2, name='enc_dense_lrelu_0')(x)
            if self.act == 'selu':
                x = Activation(activation='selu', name='enc_dense_selu_0')(x)
        if np.any(np.array([self.alpha, self.beta, self.lamb]) > 0):
            # mean
            self.mu = Dense(units=self.z_dim,
                            kernel_initializer='glorot_uniform', activation='linear',
                            name='enc_mu_ouput')(x)
            # logarithmic variance
            self.logvar = Dense(units=self.z_dim,
                                kernel_initializer='glorot_uniform', activation='linear',
                                name='enc_logvar_ouput')(x)
            # latent space
            self.z = Lambda(self.sample_gaussian, output_shape=(self.z_dim,), name='enc_z_output')([self.mu, self.logvar])
            # build encoder
            self.encoder = Model(inputs=[self.enc_input], outputs=[self.mu, self.logvar, self.z],
                                name='encoder')
        else:
            # latent space
            self.z = Dense(self.z_dim, kernel_initializer='glorot_uniform', activation='sigmoid',
                           name='enc_z_ouput')(x)
            # build encoder
            self.encoder = Model(inputs=[self.enc_input], outputs=[self.z],
                                 name='encoder')


    def _build_decoder(self):
        ''' builds decoder model '''
        # latent unit gaussian and categorical inputs
        dec_input = Input(batch_shape=(self.batch_size, self.z_dim), name='dec_z_input')
        # dense layer with same feature count as final convolution
        x = Dense(units=np.prod(self.final_conv_shape),
                  kernel_initializer=self.krnl_init,
                  name='dec_dense_0')(dec_input)
        if self.act == 'lrelu':
            x = LeakyReLU(alpha=0.2, name='dec_dense_lrelu_0')(x)
        if self.act == 'selu':
            x = Activation(activation='selu', name='dec_dense_selu_0')(x)
        if self.final_conv_shape[:2] != (1, 1):
            # repeated dense layer
            x = Dense(units=np.prod(self.final_conv_shape),
                      kernel_initializer=self.krnl_init,
                      name='dec_dense_1')(x)
            if self.act == 'lrelu':
                x = LeakyReLU(alpha=0.2, name='dec_dense_lrelu_1')(x)
            if self.act == 'selu':
                x = Activation(activation='selu', name='dec_dense_selu_1')(x)
        # reshape to final convolution shape
        convt = Reshape(target_shape=self.final_conv_shape, name='dec_rshp_0')(x)
        u = 0
        # transform to sample shape with transposed convolutions
        for i in range(self.conv_number-1, 0, -1):
            filter_number = get_filter_number(i, self.filter_base, self.filter_factor)
            convt = Conv2DTranspose(filters=filter_number, kernel_size=self.filter_length,
                                    kernel_initializer=self.krnl_init,
                                    padding='same', strides=self.filter_stride,
                                    name='dec_convt_{}'.format(u))(convt)
            if self.act == 'lrelu':
                convt = BatchNormalization(name='dec_convt_batchnorm_{}'.format(u))(convt)
                convt = LeakyReLU(alpha=0.2, name='dec_convt_lrelu_{}'.format(u))(convt)
            if self.act == 'selu':
                convt = Activation(activation='selu', name='dec_convt_selu_{}'.format(u))(convt)
            u += 1
        self.dec_output = Conv2DTranspose(filters=1, kernel_size=self.filter_length,
                                          kernel_initializer='glorot_uniform', activation='sigmoid',
                                          padding='same', strides=self.filter_stride,
                                          name='dec_output')(convt)
        # build decoder
        self.decoder = Model(inputs=[dec_input], outputs=[self.dec_output],
                             name='decoder')


    def _build_vae(self):
        ''' builds variational autoencoder network '''
        # build VAE
        if np.any(np.array([self.alpha, self.beta, self.lamb]) > 0):
            self.vae = Model(inputs=[self.enc_input], outputs=[self.decoder(self.encoder(self.enc_input)[2])],
                             name='variational autoencoder')
            tc_loss = self.total_correlation_loss()
            self.vae.add_loss(tc_loss)
            self.vae.add_metric(tc_loss, name='tc_loss', aggregation='mean')
        else:
            self.vae = Model(inputs=[self.enc_input], outputs=[self.decoder(self.encoder(self.enc_input))],
                             name='variational autoencoder')
        # define GAN optimizer
        self.vae_opt = Nadam(lr=self.lr)
        # compile GAN
        self.vae.compile(loss=self.reconstruction_loss, optimizer=self.vae_opt)


    def sample_latent_distribution(self, batch_size=None):
        ''' draws samples from the latent gaussian and categorical distributions '''
        if batch_size == None:
            batch_size = self.batch_size
        # noise
        z = sample_gaussian(batch_size, self.z_dim)
        # categorical control variable
        c = sample_categorical(batch_size, self.c_dim)
        # uniform control variable
        u = sample_uniform(batch_size, self.u_dim)
        return z, c, u
    

    def encode(self, x_batch, verbose=False):
        ''' encoder input configurations '''
        return self.encoder.predict(x_batch, batch_size=self.batch_size, verbose=verbose)


    def generate(self, beta, verbose=False):
        ''' generate new configurations using samples from the latent distribution '''
        # sample latent space
        z = self.sample_gaussian(beta)
        # generate configurations
        return self.decoder.predict(z, batch_size=self.batch_size, verbose=verbose)


    def model_summaries(self):
        ''' print model summaries '''
        self.encoder.summary()
        self.decoder.summary()
        self.vae.summary()


    def save_weights(self, name, lattice_length, interval, num_samples, seed):
        ''' save weights to file '''
        # file parameters
        params = (name, lattice_length, interval, num_samples,
                  self.conv_number, self.filter_length, self.filter_base, self.filter_factor,
                  self.z_dim,
                  self.krnl_init, self.act, self.lr, self.batch_size, seed)
        file_name = os.getcwd()+'/{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{:.0e}.{}.{}.vae.weights.h5'.format(*params)
        # save weights
        self.vae.save_weights(file_name)


    def load_weights(self, name, lattice_length, interval, num_samples, seed):
        ''' load weights from file '''
        # file parameters
        params = (name, lattice_length, interval, num_samples,
                  self.conv_number, self.filter_length, self.filter_base, self.filter_factor,
                  self.z_dim,
                  self.krnl_init, self.act, self.lr, self.batch_size, seed)
        file_name = os.getcwd()+'/{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{:.0e}.{}.{}.vae.weights.h5'.format(*params)
        # load weights
        self.vae.load_weights(file_name, by_name=True)


class Trainer():
    '''
    VAE trainer
    '''
    def __init__(self, model, num_fields, num_temps, num_samples):
        ''' initialize trainer '''
        # model
        self.model = model
        # number of external thermal parameters
        self.num_fields = num_fields
        self.num_temps = num_temps
        # number of samples per (h, t)
        self.num_samples = num_samples
        # number of batches
        self.num_batches = (self.num_fields*self.num_temps*self.num_samples)//self.model.batch_size
        # loss histories
        self.latent_loss_history = []
        self.reconstruction_loss_history = []
        # past epochs (changes if loading past trained model)
        self.past_epochs = 0
        # checkpoint managers
        self.vae_mngr = None


    def get_losses(self):
        ''' retrieve loss histories '''
        # reshape arrays into (epochs, batches)
        latent_loss = np.array(self.latent_loss_history).reshape(-1, self.num_batches)
        reconstruction_loss = np.array(self.reconstruction_loss_history).reshape(-1, self.num_batches)
        return latent_loss, reconstruction_loss


    def save_losses(self, name, lattice_length, interval, num_samples, seed):
        ''' save loss histories to file '''
        # retrieve losses
        losses = self.get_losses()
        # file parameters
        params = (name, lattice_length, interval, num_samples,
                  self.model.conv_number, self.model.filter_length, self.model.filter_base, self.model.filter_factor,
                  self.model.z_dim,
                  self.model.krnl_init, self.model.act, self.model.lr, self.model.batch_size, seed)
        loss_file_name = os.getcwd()+'/{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{:.0e}.{}.{}.loss.npy'.format(*params)
        np.save(loss_file_name, np.stack(losses, axis=-1))


    def load_losses(self, name, lattice_length, interval, num_samples, seed):
        ''' load loss histories from file '''
        # file parameters
        params = (name, lattice_length, interval, num_samples,
                  self.model.conv_number, self.model.filter_length, self.model.filter_base, self.model.filter_factor,
                  self.model.z_dim,
                  self.model.krnl_init, self.model.act, self.model.lr, self.model.batch_size, seed)
        loss_file_name = os.getcwd()+'/{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{:.0e}.{}.{}.loss.npy'.format(*params)
        losses = np.load(loss_file_name)
        # set past epochs
        self.past_epochs = losses.shape[0]
        # change loss histories into lists
        self.latent_loss_history = list(losses[:, :, 0].reshape(-1))
        self.reconstruction_loss_history = list(losses[:, :, 1].reshape(-1))


    def initialize_checkpoint_manager(self, name, lattice_length, interval, num_samples, seed):
        ''' initialize training checkpoint managers '''
        # initialize checkpoints
        self.vae_ckpt = Checkpoint(step=tf.Variable(0), optimizer=self.model.vae_opt, net=self.model.vae)
        # file parameters
        params = (name, lattice_length, interval, num_samples,
                  self.model.conv_number, self.model.filter_length, self.model.filter_base, self.model.filter_factor,
                  self.model.z_dim,
                  self.model.krnl_init, self.model.act, self.model.lr, self.model.batch_size, seed)
        directory = os.getcwd()+'/{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{:.0e}.{}.{}.ckpts'.format(*params)
        # initialize checkpoint managers
        self.dsc_mngr = CheckpointManager(self.vae_ckpt, directory, max_to_keep=4)


    def load_latest_checkpoint(self, name, lattice_length, interval, num_samples, seed):
        ''' load latest training checkpoint from file '''
        # initialize checkpoint managers
        self.initialize_checkpoint_manager(name, lattice_length, interval, num_samples, seed)
        self.load_losses(name, lattice_length, interval, num_samples, seed)
        # file parameters
        params = (name, lattice_length, interval, num_samples,
                  self.model.conv_number, self.model.filter_length, self.model.filter_base, self.model.filter_factor,
                  self.model.z_dim,
                  self.model.krnl_init, self.model.act, self.model.lr, self.model.batch_size, seed)
        directory = os.getcwd()+'/{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{:.0e}.{}.{}.ckpts'.format(*params)
        # restore checkpoints
        self.vae_ckpt.restore(self.vae_mngr.latest_checkpoint).assert_consumed()


    def get_training_indices(self):
        ''' retrieve class-balancing training indices '''
        # number of square subsectors in (h, t) space
        n_sr = np.int32(self.num_fields*self.num_temps/self.model.batch_size)
        # side length of (h, t) space in subsectors
        n_sr_l = np.int32(np.sqrt(n_sr))
        # side length of subsector
        sr_l = np.int32(np.sqrt(self.model.batch_size))
        # indices for top left subsector
        sr_indices = np.stack(np.meshgrid(np.arange(sr_l),
                                          np.arange(sr_l)), axis=-1).reshape(-1, 2)[:, ::-1]
        # all indices for subsectors
        indices = np.array([[sr_indices+sr_l*np.array([i,j]).reshape(1, -1) for i in range(n_sr_l)] for j in range(n_sr_l)])
        # flattened subsector indices
        flat_indices = np.ravel_multi_index(indices.reshape(-1, 2).T, dims=(self.num_fields, self.num_temps))
        # shuffle indices within each subsector
        for i in range(n_sr):
            flat_indices[self.model.batch_size*i:self.model.batch_size*(i+1)] = np.random.permutation(flat_indices[self.model.batch_size*i:self.model.batch_size*(i+1)])
        # shift indices to balance batches by subsector
        shift_indices = np.concatenate([flat_indices[i::self.model.batch_size] for i in range(self.model.batch_size)])
        return shift_indices


    def reorder_training_data(self, x_train):
        ''' reorder training data by class-balancing indices '''
        x_train = x_train.reshape(self.num_fields*self.num_temps, self.num_samples, *self.model.input_shape)[self.get_training_indices()]
        return np.moveaxis(x_train, 0, 1).reshape(self.num_fields*self.num_temps*self.num_samples, *self.model.input_shape)


    def train_vae(self, x_batch):
        ''' train VAE '''
        # VAE losses
        if np.any(np.array([self.model.alpha, self.model.beta, self.model.lamb]) > 0):
            vae_loss, tc_loss = self.model.vae.train_on_batch(x_batch, x_batch)
            self.latent_loss_history.append(tc_loss)
            self.reconstruction_loss_history.append(vae_loss-tc_loss)
        else:
            vae_loss = self.model.vae.train_on_batch(x_batch, x_batch)
            self.latent_loss_history.append(0)
            self.reconstruction_loss_history.append(vae_loss)


    def rolling_loss_average(self, epoch, batch):
        ''' calculate rolling loss averages over batches during training '''
        epoch = epoch+self.past_epochs
        # catch case where there are no calculated losses yet
        if batch == 0:
            latent_loss = 0
            reconstruction_loss = 0
        # calculate rolling average
        else:
            # start index for current epoch
            start = self.num_batches*epoch
            # stop index for current batch (given epoch)
            stop = self.num_batches*epoch+batch+1
            # average loss histories
            latent_loss = np.mean(self.latent_loss_history[start:stop])
            reconstruction_loss = np.mean(self.reconstruction_loss_history[start:stop])
        return latent_loss, reconstruction_loss


    def fit(self, x_train, num_epochs=4, save_step=4, verbose=False):
        ''' fit model '''
        x_train = self.reorder_training_data(x_train)
        num_epochs += self.past_epochs
        # loop through epochs
        for i in range(self.past_epochs, num_epochs):
            # construct progress bar for current epoch
            batch_range = trange(self.num_batches, desc='', disable=not verbose)
            # loop through batches
            for j in batch_range:
                # set batch loss description
                batch_loss = self.rolling_loss_average(i, j)
                desc = 'Epoch: {}/{} TCKLD Loss: {:.4f} RCNST Loss: {:.4f}'.format(i+1, num_epochs, *batch_loss)
                batch_range.set_description(desc)
                # fetch batch
                x_batch = x_train[self.model.batch_size*j:self.model.batch_size*(j+1)]
                # train VAE
                self.train_vae(x_batch=x_batch)
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
     NAME, N, I, NS,
     CN, FL, FB, FF,
     ZD, ALPHA, BETA, LAMBDA,
     KI, AN, LR,
     EP, BS, SEED) = parse_args()

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

    H, T, CONF, THRM = load_data(NAME, N, I, NS, SEED, VERBOSE)
    NH, NT = H.size, T.size
    CONF = CONF.reshape(NH*NT*NS, N, N, 1)
    THRM = THRM.reshape(NH*NT*NS, -1)
    IS = (N, N, 1)

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

    MDL = VAE(IS, CN, FL, FB, FF, ZD, ALPHA, BETA, LAMBDA, KI, AN, LR, BS, NH*NT*NS)
    TRN = Trainer(MDL, NH, NT, NS)
    if RSTRT:
        MDL.load_weights(NAME, N, I, NS, SEED)
        if VERBOSE:
            MDL.model_summaries()
        # TRN.load_latest_checkpoint(NAME, N, I, NS, SEED)
        TRN.fit(CONF, num_epochs=EP, save_step=EP, verbose=VERBOSE)
        TRN.save_losses(NAME, N, I, NS, SEED)
        MDL.save_weights(NAME, N, I, NS, SEED)
    else:
        try:
            MDL.load_weights(NAME, N, I, NS, SEED)
            if VERBOSE:
                MDL.model_summaries()
            TRN.load_losses(NAME, N, I, NS, SEED)
        except:
            if VERBOSE:
                MDL.model_summaries()
            # TRN.initialize_checkpoint_managers(NAME, N, I, NS, SEED)
            TRN.fit(CONF, num_epochs=EP, save_step=EP, verbose=VERBOSE)
            TRN.save_losses(NAME, N, I, NS, SEED)
            MDL.save_weights(NAME, N, I, NS, SEED)
    L = TRN.get_losses()
    if PLOT:
        plot_losses(L, CM,
                    NAME, N, I, NS,
                    CN, FL, FB, FF, ZD,
                    KI, AN, LR, BS, SEED, VERBOSE)
    if np.any(np.array([ALPHA, BETA, LAMBDA]) > 0):
        M, LV, Z = MDL.encode(CONF, VERBOSE)
        Z = Z.reshape(NH, NT, NS, ZD)
        M = M.reshape(NH, NT, NS, ZD)
        S = np.exp(0.5*LV).reshape(NH, NT, NS, ZD)
        save_output_data(Z, 'latent',
                         NAME, N, I, NS,
                         CN, FL, FB, FF, ZD,
                         KI, AN, LR, BS, SEED)
        save_output_data(M, 'mean',
                         NAME, N, I, NS,
                         CN, FL, FB, FF, ZD,
                         KI, AN, LR, BS, SEED)
        save_output_data(S, 'standard_deviation',
                         NAME, N, I, NS,
                         CN, FL, FB, FF, ZD,
                         KI, AN, LR, BS, SEED)
        if PLOT:
            plot_diagrams(Z, M, S, H, T, CM,
                          NAME, N, I, NS,
                          CN, FL, FB, FF, ZD,
                          KI, AN, LR, BS, SEED, VERBOSE)
    else:
        Z = MDL.encode(CONF, VERBOSE)
        Z = Z.reshape(NH, NT, NS, ZD)
        save_output_data(Z, 'latent',
                         NAME, N, I, NS,
                         CN, FL, FB, FF, ZD,
                         KI, AN, LR, BS, SEED)
        if PLOT:
            plot_diagrams(Z, None, None, H, T, CM,
                          NAME, N, I, NS,
                          CN, FL, FB, FF, ZD,
                          KI, AN, LR, BS, SEED, VERBOSE)
