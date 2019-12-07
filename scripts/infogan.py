# -*- coding: utf-8 -*-
"""
Created on Tue Dec 3 00:13:46 2019

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
from tensorflow.keras.layers import (Input, Flatten, Reshape, Concatenate, Lambda,
                                     Dense, BatchNormalization, Conv2D, Conv2DTranspose,
                                     SpatialDropout2D, Activation, LeakyReLU)
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
    parser.add_argument('-l', '--lattice_length', help='lattice size (side length)',
                        type=int, default=27)
    parser.add_argument('-si', '--sample_interval', help='interval for selecting phase points (variational autoencoder)',
                        type=int, default=1)
    parser.add_argument('-sn', '--sample_number', help='number of samples per phase point (variational autoencoder)',
                        type=int, default=1024)
    parser.add_argument('-sc', '--scale_data', help='scale data (-1, 1) -> (0, 1)',
                        action='store_true')
    parser.add_argument('-cn', '--conv_number', help='convolutional layer depth',
                        type=int, default=3)
    parser.add_argument('-fbl', '--filter_base_length', help='size of filters in base hidden convolutional layer',
                        type=int, default=3)
    parser.add_argument('-fb', '--filter_base', help='base number of filters in base hidden convolutional layer',
                        type=int, default=9)
    parser.add_argument('-fl', '--filter_length', help='size of filters following base convolution',
                        type=int, default=3)
    parser.add_argument('-ff', '--filter_factor', help='multiplicative factor of filters after base convolution',
                        type=int, default=9)
    parser.add_argument('-gd', '--generator_dropout', help='toggle generator dropout layers',
                        action='store_true')
    parser.add_argument('-dd', '--discriminator_dropout', help='toggle discriminator dropout layers',
                        action='store_true')
    parser.add_argument('-zd', '--z_dimension', help='sample noise dimension',
                        type=int, default=100)
    parser.add_argument('-cd', '--c_dimension', help='sample classification dimension',
                        type=int, default=5)
    parser.add_argument('-ud', '--u_dimension', help='sample continuous dimension',
                        type=int, default=0)
    parser.add_argument('-ki', '--kernel_initializer', help='kernel initializer',
                        type=str, default='lecun_normal')
    parser.add_argument('-an', '--activation', help='activation function',
                        type=str, default='selu')
    parser.add_argument('-dop', '--discriminator_optimizer', help='optimizer for discriminator',
                        type=str, default='sgd')
    parser.add_argument('-gop', '--gan_optimizer', help='optimizer for gan',
                        type=str, default='adam')
    parser.add_argument('-dlr', '--discriminator_learning_rate', help='learning rate for discriminator',
                        type=float, default=1e-2)
    parser.add_argument('-glr', '--gan_learning_rate', help='learning rate for generator',
                        type=float, default=1e-3)
    parser.add_argument('-gl', '--gan_lambda', help='gan regularization lambda',
                        type=float, default=1.0)
    parser.add_argument('-ta', '--trainer_alpha', help='trainer alpha label smoothing',
                        type=float, default=0.1)
    parser.add_argument('-tb', '--trainer_beta', help='trainer beta label flipping',
                        type=float, default=0.05)
    parser.add_argument('-bs', '--batch_size', help='size of batches',
                        type=int, default=169)
    parser.add_argument('-ep', '--epochs', help='number of training epochs',
                        type=int, default=4)
    parser.add_argument('-sd', '--random_seed', help='random seed for sample selection and learning',
                        type=int, default=128)
    args = parser.parse_args()
    return (args.verbose, args.restart, args.plot, args.parallel, args.gpu, args.threads,
            args.name, args.lattice_length, args.sample_interval, args.sample_number, args.scale_data,
            args.conv_number, args.filter_base_length, args.filter_base, args.filter_length, args.filter_factor,
            args.generator_dropout, args.discriminator_dropout, args.z_dimension, args.c_dimension, args.u_dimension,
            args.kernel_initializer, args.activation,
            args.discriminator_optimizer, args.gan_optimizer, args.discriminator_learning_rate, args.gan_learning_rate,
            args.gan_lambda, args.trainer_alpha, args.trainer_beta,
            args.batch_size, args.epochs, args.random_seed)


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
    select_conf = index_data_by_sample(interval_conf, num_fields, num_temps, indices)
    if scaled:
        select_conf = scale_configurations(select_conf)
    select_thrm = index_data_by_sample(interval_thrm, num_fields, num_temps, indices)
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
    # plot losses
    loss_list = ['Discriminator (Fake) Loss', 'Discriminator (Real) Loss',
                 'Generator Loss', 'Categorical Control Loss', 'Continuous Control Loss']
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
    loss_list = ['Discriminator (Fake) Loss', 'Discriminator (Real) Loss',
                 'Generator Loss', 'Categorical Control Loss', 'Continuous Control Loss']
    color_list = np.linspace(0.2, 0.8, len(losses))
    for i in trange(len(losses), desc='Plotting Epoch Losses', disable=not verbose):
        ax.plot(np.arange(losses[i].shape[0]), losses[i].mean(1), color=cmap(color_list[i]), label=loss_list[i])
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


def plot_diagrams(c_data, u_data, fields, temps, cmap,
                  name, lattice_length, interval, num_samples, scaled, seed,
                  prfx, verbose=False):
    params = (name, lattice_length, interval, num_samples, scaled, seed)
    file_prfx = '{}.{}.{}.{}.{:d}.{}.'.format(*params)+prfx
    c_diag = c_data.mean(2)
    u_diag = u_data.mean(2)
    c_dim = c_diag.shape[-1]
    u_dim = u_diag.shape[-1]
    for i in trange(c_dim, desc='Plotting Discrete Controls', disable=not verbose):
        plot_diagram(c_diag[:, :, i], fields, temps, cmap, file_prfx, 'c_{}'.format(i))
    for i in trange(u_dim, desc='Plotting Continuous Controls', disable=not verbose):
        # plot_diagram(u_diag[:, :, i], fields, temps, cmap, prfx, 'm_{}'.format(i))
        # plot_diagram(u_diag[:, :, u_dim+i], fields, temps, cmap, prfx, 's_{}'.format(i))
        plot_diagram(u_diag[:, :, i], fields, temps, cmap, file_prfx, 'u_{}'.format(i))


def nrelu(x):
    ''' negative output relu '''
    return -K.relu(x)


def get_final_conv_shape(input_shape, conv_number,
                         filter_base_length, filter_length,
                         filter_base, filter_factor):
    ''' calculates final convolutional layer output shape '''
    return tuple(np.array(input_shape[:2])//(filter_base_length*filter_length**(conv_number-1)))+\
           (input_shape[2]*filter_base*filter_factor**(conv_number-1),)


def get_filter_number(conv_iter, filter_base, filter_factor):
    ''' calculates the filter count for a given convolutional iteration '''
    return filter_base*filter_factor**(conv_iter)


def get_filter_length_stride(conv_iter, filter_base_length, filter_length):
    ''' calculates filter length and stride for a given convolutional iteration '''
    if conv_iter == 0:
        return filter_base_length, filter_base_length
    else:
        return filter_length, filter_length


def sample_gaussian(num_rows, dimension):
    ''' unit gaussian sampling '''
    return np.random.normal(size=(num_rows, dimension))


def sample_categorical(num_rows, num_categories):
    ''' categorical sampling '''
    if num_categories > 0:
        sample = to_categorical(np.random.randint(0, num_categories, num_rows).reshape(-1, 1))
    else:
        sample = np.empty(shape=(num_rows, num_categories))
    return sample


def sample_uniform(low, high, num_rows, dimension):
    ''' uniform sampling '''
    return np.random.uniform(low=low, high=high, size=(num_rows, dimension))


def mutual_information_categorical_loss(category, prediction):
    ''' mutual information loss for categorical control variables '''
    eps = 1e-8
    entropy = -K.mean(K.sum(category*K.log(category+eps), axis=1))
    conditional_entropy = -K.mean(K.sum(category*K.log(prediction+eps), axis=1))
    return entropy+conditional_entropy


class InfoGAN():
    '''
    InfoGAN Model
    Generative adversarial modeling of the Ising spin configurations
    '''
    def __init__(self, input_shape=(27, 27, 1), conv_number=3,
                 filter_base_length=3, filter_base=9, filter_length=3, filter_factor=9,
                 gen_drop=False, dsc_drop=False,
                 z_dim=100, c_dim=5, u_dim=0,
                 krnl_init='lecun_normal', act='selu',
                 dsc_opt_n='sgd', gan_opt_n='adam', dsc_lr=1e-2, gan_lr=1e-3,
                 lamb=1.0, batch_size=169, scaled=False):
        ''' initialize model parameters '''
        self.eps = 1e-8
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
        # set stride to be same as filter size
        self.filter_base_stride = self.filter_base_length
        self.filter_stride = self.filter_length
        # convolutional input and output shapes
        self.input_shape = input_shape
        self.final_conv_shape = get_final_conv_shape(self.input_shape, self.conv_number,
                                                     self.filter_base_length, self.filter_length,
                                                     self.filter_base, self.filter_factor)
        # generator and discriminator dropout
        self.gen_drop = gen_drop
        self.dsc_drop = dsc_drop
        # latent noise dimension
        self.z_dim = z_dim
        # categorical control variable dimension
        self.c_dim = c_dim
        # continuous control variable dimension
        self.u_dim = u_dim
        # kernel initializer and activation
        self.krnl_init = krnl_init
        self.act = act
        # discriminator and gan optimizers
        self.dsc_opt_n = dsc_opt_n
        self.gan_opt_n = gan_opt_n
        # discriminator and gan learning rates
        self.dsc_lr = dsc_lr
        self.gan_lr = gan_lr
        # regularization constant
        self.lamb = lamb
        # batch size and callbacks
        self.batch_size = batch_size
        # scaling
        if scaled:
            self.gen_out_act = 'sigmoid'
        else:
            self.gen_out_act = 'tanh'
        # build full model
        self._build_model()
    

    def get_file_prefix(self):
        ''' gets parameter tuple and filename string prefix '''
        params = (self.conv_number, self.filter_base_length, self.filter_base, self.filter_length, self.filter_factor,
                  self.gen_drop, self.dsc_drop, self.z_dim, self.c_dim, self.u_dim,
                  self.krnl_init, self.act,
                  self.dsc_opt_n, self.gan_opt_n, self.dsc_lr, self.gan_lr,
                  self.lamb, self.batch_size)
        file_name = '{}.{}.{}.{}.{}.{:d}.{:d}.{}.{}.{}.{}.{}.{}.{}.{:.0e}.{:.0e}.{:.0e}.{}'.format(*params)
        return file_name


    def _build_model(self):
        ''' builds each component of the InfoGAN model '''
        self._build_generator()
        self._build_discriminator()
        self._build_auxiliary()
        self._build_gan()
    

    def sample_gaussian(self, mu):
        mu = beta[:, :self.u_dim]
        logvar = beta[:, self.u_dim:]
        return mu+K.exp(0.5*logvar)*K.random_normal(shape=(self.batch_size, self.u_dim))
    

    def kl_divergence(self, beta, beta_hat):
        ''' kl divergence for continuous control variable '''
        mu = beta[:, :self.u_dim]
        logvar = beta_hat[:, self.u_dim:]
        mu_hat = beta_hat[:, :self.u_dim]
        logvar_hat = beta_hat[:, self.u_dim:]
        return 0.5*K.mean(K.exp(logvar_hat-logvar)+K.square(mu-mu_hat)/K.exp(logvar)-1+logvar-logvar_hat)


    def _build_generator(self):
        ''' builds generator model '''
        # latent unit gaussian and categorical inputs
        self.z_input = Input(batch_shape=(self.batch_size, self.z_dim), name='z_input')
        self.c_input = Input(batch_shape=(self.batch_size, self.c_dim), name='c_input')
        self.u_input = Input(batch_shape=(self.batch_size, self.u_dim), name='u_input')
        # u_sample = Lambda(self.sample_gaussian, output_shape=(self.u_dim,), name='u_sample')(self.u_input)
        # concatenate features
        x = Concatenate(name='gen_latent_concat')([self.z_input, self.c_input, self.u_input])
        # dense layer with same feature count as final convolution
        x = Dense(units=np.prod(self.final_conv_shape),
                  kernel_initializer=self.krnl_init,
                  name='gen_dense_0')(x)
        if self.act == 'lrelu':
            x = LeakyReLU(alpha=0.2, name='gen_dense_lrelu_0')(x)
        if self.act == 'selu':
            x = Activation(activation='selu', name='gen_dense_selu_0')(x)
        if self.final_conv_shape[:2] != (1, 1):
            # repeated dense layer
            x = Dense(units=np.prod(self.final_conv_shape),
                      kernel_initializer=self.krnl_init,
                      name='gen_dense_1')(x)
            if self.act == 'lrelu':
                x = LeakyReLU(alpha=0.2, name='gen_dense_lrelu_1')(x)
            if self.act == 'selu':
                x = Activation(activation='selu', name='gen_dense_selu_1')(x)
        # reshape to final convolution shape
        convt = Reshape(target_shape=self.final_conv_shape, name='gen_rshp_0')(x)
        if self.gen_drop:
            convt = SpatialDropout2D(rate=0.5, name='gen_dense_drop_0')(convt)
        u = 0
        # transform to sample shape with transposed convolutions
        for i in range(self.conv_number-1, 0, -1):
            filter_number = get_filter_number(i-1, self.filter_base, self.filter_factor)
            convt = Conv2DTranspose(filters=filter_number, kernel_size=self.filter_length,
                                    kernel_initializer=self.krnl_init,
                                    padding='same', strides=self.filter_stride,
                                    name='gen_convt_{}'.format(u))(convt)
            if self.act == 'lrelu':
                convt = BatchNormalization(name='gen_convt_batchnorm_{}'.format(u))(convt)
                convt = LeakyReLU(alpha=0.2, name='gen_convt_lrelu_{}'.format(u))(convt)
            if self.act == 'selu':
                convt = Activation(activation='selu', name='gen_convt_selu_{}'.format(u))(convt)
            if self.gen_drop:
                convt = SpatialDropout2D(rate=0.5, name='gen_convt_drop_{}'.format(u))(convt)
            u += 1
        self.gen_output = Conv2DTranspose(filters=1, kernel_size=self.filter_base_length,
                                          kernel_initializer='glorot_uniform', activation=self.gen_out_act,
                                          padding='same', strides=self.filter_base_stride,
                                          name='gen_output')(convt)
        # build generator
        self.generator = Model(inputs=[self.z_input, self.c_input, self.u_input], outputs=[self.gen_output],
                               name='generator')


    def _build_discriminator(self):
        ''' builds discriminator model '''
        # takes sample (real or fake) as input
        self.dsc_input = Input(batch_shape=(self.batch_size,)+self.input_shape, name='dsc_input')
        conv = self.dsc_input
        # iterative convolutions over input
        for i in range(self.conv_number):
            filter_number = get_filter_number(i, self.filter_base, self.filter_factor)
            filter_length, filter_stride = get_filter_length_stride(i, self.filter_base_length, self.filter_length)
            conv = Conv2D(filters=filter_number, kernel_size=filter_length,
                          kernel_initializer=self.krnl_init,
                          padding='valid', strides=filter_stride,
                          name='dsc_conv_{}'.format(i))(conv)
            if self.act == 'lrelu':
                conv = BatchNormalization(name='dsc_conv_batchnorm_{}'.format(i))(conv)
                conv = LeakyReLU(alpha=0.2, name='dsc_conv_lrelu_{}'.format(i))(conv)
            if self.act == 'selu':
                conv = Activation(activation='selu', name='dsc_conv_selu_{}'.format(i))(conv)
            if self.dsc_drop:
                conv = SpatialDropout2D(rate=0.5, name='dsc_conv_drop_{}'.format(i))(conv)
        # flatten final convolutional layer
        x = Flatten(name='dsc_fltn_0')(conv)
        if self.final_conv_shape[:2] != (1, 1):
            # dense layer
            x = Dense(units=np.prod(self.final_conv_shape),
                      kernel_initializer=self.krnl_init,
                      name='dsc_dense_0')(x)
            if self.act == 'lrelu':
                x = LeakyReLU(alpha=0.2, name='dsc_dense_lrelu_0')(x)
            if self.act == 'selu':
                x = Activation(activation='selu', name='dsc_dense_selu_0')(x)
        # the dense layer is saved as a hidden layer
        self.dsc_hidden = x
        # dense layer
        x = Dense(units=128,
                  kernel_initializer=self.krnl_init,
                  name='dsc_dense_1')(x)
        if self.act == 'lrelu':
            x = LeakyReLU(alpha=0.2, name='dsc_dense_lrelu_1')(x)
        if self.act == 'selu':
            x = Activation(activation='selu', name='dsc_dense_selu_1')(x)
        # discriminator classification output (0, 1) -> (fake, real)
        self.dsc_output = Dense(units=1,
                                kernel_initializer='glorot_uniform', activation='sigmoid',
                                name='dsc_output')(x)
        # build discriminator
        self.discriminator = Model(inputs=[self.dsc_input], outputs=[self.dsc_output],
                                   name='discriminator')
        # define optimizer
        if self.dsc_opt_n == 'adam':
            self.dsc_opt = Adam(lr=self.dsc_lr)
        if self.dsc_opt_n == 'adamax':
            self.dsc_opt = Adamax(lr=self.dsc_lr)
        if self.dsc_opt_n == 'nadam':
            self.dsc_opt = Nadam(lr=self.dsc_lr)
        if self.dsc_opt_n == 'sgd':
            self.dsc_opt = SGD(lr=self.dsc_lr)
        # compile discriminator
        self.discriminator.compile(loss='binary_crossentropy', optimizer=self.dsc_opt)


    def _build_auxiliary(self):
        ''' builds auxiliary classification reconstruction model '''
        # initialize with dense layer taking the hidden generator layer as input
        x = Dense(units=128,
                  kernel_initializer=self.krnl_init,
                  name='aux_dense_0')(self.dsc_hidden)
        if self.act == 'lrelu':
            x = LeakyReLU(alpha=0.2, name='aux_dense_lrelu_0')(x)
        if self.act == 'selu':
            x = Activation(activation='selu', name='aux_dense_selu_0')(x)
        # auxiliary output is a reconstruction of the categorical assignments fed into the generator
        self.aux_c_output = Dense(self.c_dim,
                                  kernel_initializer='glorot_uniform', activation='softmax',
                                  name='aux_output_c')(x)
        # mu = Dense(self.u_dim,
        #            kernel_initializer='glorot_uniform', activation='tanh',
        #            name='aux_mu')(x)
        # logvar = Dense(self.u_dim,
        #                kernel_initializer='glorot_uniform', activation=nrelu,
        #                name='aux_sigma')(x)
        # self.aux_u_output = Concatenate(name='aux_u_output')([mu, logvar])
        self.aux_u_output = Dense(self.u_dim,
                                  kernel_initializer='glorot_uniform', activation='tanh',
                                  name='aux_mu')(x)
        # build auxiliary classifier
        self.auxiliary = Model(inputs=[self.dsc_input], outputs=[self.aux_c_output, self.aux_u_output],
                               name='auxiliary')


    def _build_gan(self):
        ''' builds generative adversarial network '''
        # static discriminator output
        self.discriminator.trainable = False
        gan_output = self.discriminator(self.gen_output)
        # auxiliary output
        gan_output_aux_c, gan_output_aux_u = self.auxiliary(self.gen_output)
        # build GAN
        self.gan = Model(inputs=[self.z_input, self.c_input, self.u_input], outputs=[gan_output, gan_output_aux_c, gan_output_aux_u],
                         name='infogan')
        # define GAN optimizer
        if self.gan_opt_n == 'adam':
            self.gan_opt = Adam(lr=self.dsc_lr)
        if self.gan_opt_n == 'adamax':
            self.gan_opt = Adamax(lr=self.dsc_lr)
        if self.gan_opt_n == 'nadam':
            self.gan_opt = Nadam(lr=self.dsc_lr)
        if self.gan_opt_n == 'sgd':
            self.gan_opt = SGD(lr=self.dsc_lr)
        # compile GAN
        self.gan.compile(loss={'discriminator' : 'binary_crossentropy',
                               'auxiliary' : 'categorical_crossentropy',
                               'auxiliary_1' : 'mse'},
                         loss_weights={'discriminator': 1.0,
                                       'auxiliary': self.lamb,
                                       'auxiliary_1': self.lamb},
                         optimizer=self.gan_opt)
        self.discriminator.trainable = True


    def sample_latent_distribution(self, batch_size=None):
        ''' draws samples from the latent gaussian and categorical distributions '''
        if batch_size == None:
            batch_size = self.batch_size
        # noise
        z = sample_gaussian(batch_size, self.z_dim)
        # categorical control variable
        c = sample_categorical(batch_size, self.c_dim)
        # continuous control variable
        # u = np.concatenate((sample_uniform(-1.0, 1.0, batch_size, self.u_dim),
        #                     np.log(np.square(sample_uniform(0.0, 1.0, batch_size, self.u_dim)))), axis=-1)
        u = sample_uniform(-1.0, 1.0, batch_size, self.u_dim)
        return z, c, u


    def generate(self, sample_count=None, verbose=False):
        ''' generate new configurations using samples from the latent distributions '''
        if sample_count == None:
            sample_count = self.batch_size
        # sample latent space
        z, c, u = self.sample_latent_distribution(batch_size=sample_count)
        # generate configurations
        return self.generator.predict([z, c, u], batch_size=self.batch_size, verbose=verbose)


    def generate_controlled(self, c, u, sample_count=None, verbose=False):
        ''' generate new configurations using control variables '''
        if sample_count == None:
            sample_count = self.batch_size
        # sample latent space
        c = np.tile(c, (sample_count, 1))
        # u = np.tile(np.concatenate((m, np.log(np.square(s)))), (sample_count, 1))
        u = np.tile(u, (sample_count, 1))
        z, _, _ = self.sample_latent_distribution(batch_size=sample_count)
        # generate configurations
        return self.generator.predict([z, c, u], batch_size=self.batch_size, verbose=verbose)


    def discriminate(self, x_batch, verbose=False):
        ''' discriminate input configurations '''
        return self.discriminator.predict(x_batch, batch_size=self.batch_size, verbose=verbose)


    def get_aux_dist(self, x_batch, verbose=False):
        ''' predict categorical assignments of input configurations '''
        return self.auxiliary.predict(x_batch, batch_size=self.batch_size, verbose=verbose)


    def model_summaries(self):
        ''' print model summaries '''
        self.generator.summary()
        self.discriminator.summary()
        self.auxiliary.summary()
        self.gan.summary()


    def save_weights(self, name, lattice_length, interval, num_samples, scaled, seed):
        ''' save weights to file '''
        # file parameters
        params = (name, lattice_length, interval, num_samples, scaled, seed)
        file_name = os.getcwd()+'/{}.{}.{}.{}.{:d}.{}.'.format(*params)+self.get_file_prefix()+'.gan.weights.h5'
        # save weights
        self.gan.save_weights(file_name)


    def load_weights(self, name, lattice_length, interval, num_samples, scaled, seed):
        ''' load weights from file '''
        # file parameters
        params = (name, lattice_length, interval, num_samples, scaled, seed)
        file_name = os.getcwd()+'/{}.{}.{}.{}.{:d}.{}.'.format(*params)+self.get_file_prefix()+'.gan.weights.h5'
        # load weights
        self.gan.load_weights(file_name, by_name=True)


class Trainer():
    '''
    InfoGAN trainer
    '''
    def __init__(self, model, alpha, beta):
        ''' initialize trainer '''
        # model
        self.model = model
        # smoothing constant
        self.alpha = alpha
        self.beta = beta
        # loss histories
        self.dsc_fake_loss_history = []
        self.dsc_real_loss_history = []
        self.gan_loss_history = []
        self.ent_cat_loss_history = []
        self.ent_con_loss_history = []
        # past epochs (changes if loading past trained model)
        self.past_epochs = 0
        # checkpoint managers
        self.dsc_mngr = None
        self.gan_mngr = None
    

    def get_file_prefix(self):
        ''' gets file prefix '''
        return self.model.get_file_prefix()+'.{:.0e}.{:.0e}'.format(self.alpha, self.beta)


    def get_losses(self):
        ''' retrieve loss histories '''
        # reshape arrays into (epochs, batches)
        dsc_fake_loss = np.array(self.dsc_fake_loss_history).reshape(-1, self.num_batches)
        dsc_real_loss = np.array(self.dsc_real_loss_history).reshape(-1, self.num_batches)
        gan_loss = np.array(self.gan_loss_history).reshape(-1, self.num_batches)
        ent_cat_loss = np.array(self.ent_cat_loss_history).reshape(-1, self.num_batches)
        ent_con_loss = np.array(self.ent_con_loss_history).reshape(-1, self.num_batches)
        return dsc_fake_loss, dsc_real_loss, gan_loss, ent_cat_loss, ent_con_loss


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
        self.dsc_fake_loss_history = list(losses[:, :, 0].reshape(-1))
        self.dsc_real_loss_history = list(losses[:, :, 1].reshape(-1))
        self.gan_loss_history = list(losses[:, :, 2].reshape(-1))
        self.ent_cat_loss_history = list(losses[:, :, 3].reshape(-1))
        self.ent_con_loss_history = list(losses[:, :, 4].reshape(-1))


    def initialize_checkpoint_managers(self, name, lattice_length, interval, num_samples, scaled, seed):
        ''' initialize training checkpoint managers '''
        # initialize checkpoints
        self.dsc_ckpt = Checkpoint(step=tf.Variable(0), optimizer=self.model.dsc_opt, net=self.model.discriminator)
        self.gan_ckpt = Checkpoint(step=tf.Variable(0), optimizer=self.model.gan_opt, net=self.model.gan)
        # file parameters
        params = (name, lattice_length, interval, num_samples, scaled, seed)
        directory = os.getcwd()+'/{}.{}.{}.{}.{:d}.{}.'.format(*params)+self.get_file_prefix()+'.ckpts'
        # initialize checkpoint managers
        self.dsc_mngr = CheckpointManager(self.dsc_ckpt, directory+'/discriminator/', max_to_keep=4)
        self.gan_mngr = CheckpointManager(self.gan_ckpt, directory+'/gan/', max_to_keep=4)


    def load_latest_checkpoint(self, name, lattice_length, interval, num_samples, scaled, seed):
        ''' load latest training checkpoint from file '''
        # initialize checkpoint managers
        self.initialize_checkpoint_managers(name, lattice_length, interval, num_samples, scaled, seed)
        self.load_losses(name, lattice_length, interval, num_samples, scaled, seed)
        # file parameters
        params = (name, lattice_length, interval, num_samples, scaled, seed)
        directory = os.getcwd()+'/{}.{}.{}.{}.{:d}.{}.'.format(*params)+self.get_file_prefix()+'.ckpts'
        # restore checkpoints
        self.dsc_ckpt.restore(self.dsc_mngr.latest_checkpoint).assert_consumed()
        self.gan_ckpt.restore(self.gan_mngr.latest_checkpoint).assert_consumed()


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


    def train_discriminator(self, x_batch=None):
        ''' train discriminator network '''
        # if no configurations are supplied, the generator produces the batch
        if x_batch is None:
            # generate batch
            fake_batch = self.model.generate()
            # inputs are false samples, so the discrimination targets are of null value
            target = np.zeros(self.model.batch_size).astype(int)
            # discriminator loss
            dsc_loss = self.model.discriminator.train_on_batch(fake_batch, target)
            self.dsc_fake_loss_history.append(dsc_loss)
        else:
            # inputs are true samples, so the discrimination targets are of unit value
            # target = np.ones(self.model.batch_size).astype(int)
            # label smoothing
            target = np.random.uniform(low=1.0-self.alpha, high=1.0, size=self.model.batch_size)
            # label randomizing
            flp_size = np.int32(self.beta*self.model.batch_size)
            flp_ind = np.random.choice(self.model.batch_size, size=flp_size)
            target[flp_ind] = np.random.uniform(low=0.0, high=0.1, size=flp_size)
            # discriminator loss
            dsc_loss = self.model.discriminator.train_on_batch(x_batch, target)
            self.dsc_real_loss_history.append(dsc_loss)


    def train_gan(self):
        ''' train GAN '''
        # sample latent variables (gaussian noise and categorical controls)
        z, c, u = self.model.sample_latent_distribution()
        # inputs are true samples, so the discrimination targets are of unit value
        target = np.ones(self.model.batch_size).astype(int)
        # GAN and entropy losses
        gan_loss = self.model.gan.train_on_batch([z, c, u], [target, c, u])
        self.gan_loss_history.append(gan_loss[1])
        self.ent_cat_loss_history.append(gan_loss[2])
        self.ent_con_loss_history.append(gan_loss[3])


    def rolling_loss_average(self, epoch, batch):
        ''' calculate rolling loss averages over batches during training '''
        epoch = epoch+self.past_epochs
        # catch case where there are no calculated losses yet
        if batch == 0:
            gan_loss = 0
            dscf_loss = 0
            dscr_loss = 0
            ent_cat_loss = 0
            ent_con_loss = 0
        # calculate rolling average
        else:
            # start index for current epoch
            start = self.num_batches*epoch
            # stop index for current batch (given epoch)
            stop = self.num_batches*epoch+batch+1
            # average loss histories
            gan_loss = np.mean(self.gan_loss_history[start:stop])
            dscf_loss = np.mean(self.dsc_fake_loss_history[start:stop])
            dscr_loss = np.mean(self.dsc_real_loss_history[start:stop])
            # only calculate categorical control loss if the dimension is nonzero
            if self.model.c_dim > 0:
                ent_cat_loss = np.mean(self.ent_cat_loss_history[start:stop])
            else:
                ent_cat_loss = 0
            # only calculate continuous control loss if the dimension is nonzero
            if self.model.u_dim > 0:
                ent_con_loss = np.mean(self.ent_con_loss_history[start:stop])
            else:
                ent_con_loss = 0
        return gan_loss, dscf_loss, dscr_loss, ent_cat_loss, ent_con_loss


    def fit(self, x_train, num_epochs=4, save_step=4, verbose=False):
        ''' fit model '''
        self.num_fields, self.num_temps, self.num_samples, _, _, = x_train.shape
        self.num_batches = (self.num_fields*self.num_temps*self.num_samples)//self.model.batch_size
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
                desc = 'Epoch: {}/{} GAN Loss: {:.4f} DSCF Loss: {:.4f} DSCR Loss: {:.4f} CAT Loss: {:.4f} CON Loss: {:.4f}'.format(i+1, num_epochs, *batch_loss)
                batch_range.set_description(desc)
                # fetch batch
                x_batch = x_train[self.model.batch_size*j:self.model.batch_size*(j+1)]
                # train discriminator on false samples
                self.train_discriminator(x_batch=None)
                # train discriminator on true samples
                self.train_discriminator(x_batch=x_batch)
                # train GAN
                self.train_gan()
            # if checkpoint managers are initialized
            if self.dsc_mngr is not None and self.gan_mngr is not None:
                # increment checkpoints
                self.dsc_ckpt.step.assign_add(1)
                self.gan_ckpt.step.assign_add(1)
                # if save step is reached
                if np.int32(self.dsc_ckpt.step) % save_step == 0:
                    # save model checkpoint
                    dsc_save_path = self.dsc_mngr.save()
                    gan_save_path = self.gan_mngr.save()
                    print('Checkpoint DSC: {}'.format(dsc_save_path))
                    print('Checkpoint GAN: {}'.format(gan_save_path))

if __name__ == '__main__':
    (VERBOSE, RSTRT, PLOT, PARALLEL, GPU, THREADS,
     NAME, N, I, NS, SC,
     CN, FBL, FB, FL, FF,
     GD, DD, ZD, CD, UD,
     KI, AN,
     DOPT, GOPT, DLR, GLR,
     GLAMB, TALPHA, TBETA,
     BS, EP, SEED) = parse_args()

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
    NH, NT = H.size, T.size
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

    MDL = InfoGAN(IS, CN, FBL, FB, FL, FF, GD, DD, ZD, CD, UD, KI, AN, DOPT, GOPT, DLR, GLR, GLAMB, BS, SC)
    TRN = Trainer(MDL, TALPHA, TBETA)
    PRFX = TRN.get_file_prefix()
    if RSTRT:
        MDL.load_weights(NAME, N, I, NS, SC, SEED)
        if VERBOSE:
            MDL.model_summaries()
        # TRN.load_latest_checkpoint(NAME, N, I, NS)
        TRN.fit(CONF, num_epochs=EP, save_step=EP, verbose=VERBOSE)
        TRN.save_losses(NAME, N, I, NS, SC, SEED)
        MDL.save_weights(NAME, N, I, NS, SC, SEED)
    else:
        try:
            MDL.load_weights(NAME, N, I, NS, SC, SEED)
            if VERBOSE:
                MDL.model_summaries()
            TRN.load_losses(NAME, N, I, NS, SC, SEED)
        except:
            if VERBOSE:
                MDL.model_summaries()
            # TRN.initialize_checkpoint_managers(NAME, N, I, NS)
            TRN.fit(CONF, num_epochs=EP, save_step=EP, verbose=VERBOSE)
            TRN.save_losses(NAME, N, I, NS, SC, SEED)
            MDL.save_weights(NAME, N, I, NS, SC, SEED)
    L = TRN.get_losses()
    C, U = MDL.get_aux_dist(CONF.reshape(-1, *IS), VERBOSE)
    C = C.reshape(NH, NT, NS, CD)
    # U[:, UD:] = np.exp(0.5*U[:, UD:])
    # U = U.reshape(NH, NT, NS, 2*UD)
    U = U.reshape(NH, NT, NS, UD)
    if CD > 0:
        save_output_data(C, 'categorical_control', NAME, N, I, NS, SC, SEED, PRFX)
    if UD > 0:
        save_output_data(U, 'continuous_control', NAME, N, I, NS, SC, SEED, PRFX)
    if PLOT:
        plot_losses(L, CM, NAME, N, I, NS, SC, SEED, PRFX, VERBOSE)
        plot_diagrams(C, U, H, T, CM, NAME, N, I, NS, SC, SEED, PRFX, VERBOSE)