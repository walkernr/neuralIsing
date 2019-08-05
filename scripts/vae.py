# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 20:34:06 2019

@author: Nicholas
"""

import argparse
import os
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelBinarizer
from TanhScaler import TanhScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap, LocallyLinearEmbedding
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering, DBSCAN
from sklearn.model_selection import train_test_split
from scipy.odr import ODR, Model as ODRModel, RealData


def parse_args():
    ''' parses command line arguments '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', help='verbose output',
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
                        type=str, default='ising_init')
    parser.add_argument('-ls', '--lattice_size', help='lattice size (side length)',
                        type=int, default=16)
    parser.add_argument('-si', '--super_interval', help='interval for selecting phase points (variational autoencoder)',
                        type=int, default=1)
    parser.add_argument('-sn', '--super_samples', help='number of samples per phase point (variational autoencoder)',
                        type=int, default=256)
    parser.add_argument('-sc', '--scaler', help='feature scaling method (none, global, minmax, robust, standard, tanh)',
                        type=str, default='global')
    parser.add_argument('-pd', '--prior_distribution', help='prior distribution for autoencoder',
                        type=str, default='gaussian')
    parser.add_argument('-ki', '--kernel_initializer', help='kernel initialization function',
                        type=str, default='he_uniform')
    parser.add_argument('-vgg', '--vgg', help='vgg-like structure (instead of infogan)',
                        action='store_true')
    parser.add_argument('-cd', '--convdepth', help='convolutional layer depth (final side length, power of 2)',
                        type=int, default=4)
    parser.add_argument('-cr', '--convrep', help='convolutional repetition switch',
                        action='store_true')
    parser.add_argument('-nf', '--filters', help='base number of filters in hidden convolutional layers',
                        type=int, default=64)
    parser.add_argument('-an', '--activation', help='hidden layer activations',
                        type=str, default='selu')
    parser.add_argument('-bn', '--batch_normalization', help='batch normalization layers switch',
                        action='store_true')
    parser.add_argument('-do', '--dropout', help='dropout layers switch',
                        action='store_true')
    parser.add_argument('-ld', '--latent_dimension', help='latent dimension of the autoencoder',
                        type=int, default=4)
    parser.add_argument('-opt', '--optimizer', help='neural network weight optimization function',
                        type=str, default='nadam')
    parser.add_argument('-lr', '--learning_rate', help='learning rate for neural network optimizer',
                        type=float, default=2e-3)
    parser.add_argument('-lss', '--loss', help='loss function',
                        type=str, default='mse')
    parser.add_argument('-reg', '--regularizer', help='regularizer for latent dimension',
                        type=str, default='tc')
    parser.add_argument('-a', '--alpha', help='alpha parameter for regularizer (mi term)',
                        type=float, default=1.0)
    parser.add_argument('-b', '--beta', help='beta parameter for regularizer (kld term or tc term)',
                        type=float, default=1.0)
    parser.add_argument('-l', '--lmbda', help='lambda parameter for regularizer (info term or dim-kld term)',
                        type=float, default=1.0)
    parser.add_argument('-mss', '--minibatch_stratified_sampling', help='minibatch stratified sampling mode',
                        action='store_true')
    parser.add_argument('-ep', '--epochs', help='number of epochs',
                        type=int, default=4)
    parser.add_argument('-sh', '--shuffle', help='shuffle samples',
                        action='store_true')
    parser.add_argument('-bs', '--batch_size', help='size of batches',
                        type=int, default=32)
    parser.add_argument('-sd', '--random_seed', help='random seed for sample selection and learning',
                        type=int, default=512)
    args = parser.parse_args()
    return (args.verbose, args.plot, args.parallel, args.gpu, args.threads, args.name,
            args.lattice_size, args.super_interval, args.super_samples, args.scaler,
            args.prior_distribution, args.kernel_initializer, args.vgg, args.convdepth, args.convrep, args.filters, args.activation,
            args.batch_normalization, args.dropout, args.latent_dimension, args.optimizer, args.learning_rate,
            args.loss, args.regularizer, args.alpha, args.beta, args.lmbda, args.minibatch_stratified_sampling,
            args.epochs, args.shuffle, args.batch_size, args.random_seed)


def write_specs():
    ''' writes run details to output file (and stdout if verbose is enabled) '''
    if VERBOSE:
        print(100*'-')
        print('input summary')
        print(100*'-')
        print('plot:                      %d' % PLOT)
        print('parallel:                  %d' % PARALLEL)
        print('gpu:                       %d' % GPU)
        print('threads:                   %d' % THREADS)
        print('name:                      %s' % NAME)
        print('lattice size:              %d' % N)
        print('super interval:            %d' % SNI)
        print('super samples:             %d' % SNS)
        print('scaler:                    %s' % SCLR)
        print('prior distribution:        %s' % PRIOR)
        print('vgg-like structure:        %d' % VGG)
        print('convolution depth:         %d' % CD)
        print('convolution repetition:    %d' % CR)
        print('filters:                   %d' % NF)
        print('ativation:                 %s' % ACT)
        print('batch normalization:       %d' % BN)
        print('dropout:                   %d' % DO)
        print('latent dimension:          %d' % LD)
        print('optimizer:                 %s' % OPT)
        print('learning rate:             %.2e' % LR)
        print('loss function:             %s' % LSS)
        print('regularizer:               %s' % REG)
        print('alpha:                     %.2e' % ALPHA)
        print('beta:                      %.2e' % BETA)
        print('lambda:                    %.2e' % LMBDA)
        print('mss:                       %d' % MSS)
        print('shuffle samples:           %d' % SH)
        print('batch size:                %d' % BS)
        print('epochs:                    %d' % EP)
        print('random seed:               %d' % SEED)
        print(100*'-')
    with open(OUTPREF+'.out', 'w') as out:
        out.write(100*'-' + '\n')
        out.write('input summary\n')
        out.write(100*'-' + '\n')
        out.write('plot:                      %d\n' % PLOT)
        out.write('parallel:                  %d\n' % PARALLEL)
        out.write('gpu:                       %d\n' % GPU)
        out.write('threads:                   %d\n' % THREADS)
        out.write('name:                      %s\n' % NAME)
        out.write('lattice size:              %d\n' % N)
        out.write('super interval:            %d\n' % SNI)
        out.write('super samples:             %d\n' % SNS)
        out.write('scaler:                    %s\n' % SCLR)
        out.write('prior distribution:        %s\n' % PRIOR)
        out.write('vgg-like structure:        %d\n' % VGG)
        out.write('convolution depth:         %d\n' % CD)
        out.write('convolution repetition     %d\n' % CR)
        out.write('filters:                   %d\n' % NF)
        out.write('activation:                %s\n' % ACT)
        out.write('batch_normalization:       %d\n' % BN)
        out.write('dropout:                   %d\n' % DO)
        out.write('latent dimension:          %d\n' % LD)
        out.write('optimizer:                 %s\n' % OPT)
        out.write('learning rate:             %.2e\n' % LR)
        out.write('loss function:             %s\n' % LSS)
        out.write('regularizer:               %s\n' % REG)
        out.write('alpha:                     %.2e\n' % ALPHA)
        out.write('beta:                      %.2e\n' % BETA)
        out.write('lambda:                    %.2e\n' % LMBDA)
        out.write('mss:                       %d\n' % MSS)
        out.write('shuffle samples:           %d\n' % SH)
        out.write('batch size:                %d\n' % BS)
        out.write('epochs:                    %d\n' % EP)
        out.write('random seed:               %d\n' % SEED)
        out.write(100*'-' + '\n')


def logistic(beta, t):
    ''' returns logistic sigmoid '''
    a = 0.0
    k = 1.0
    b, m = beta
    return a+np.divide(k, 1+np.exp(-b*(t-m)))


def absolute(beta, t):
    ''' returns a*|t-b|^c+d with beta = (a, b, c, d)'''
    a, b, c, d = beta
    return a*np.power(np.abs(t-b), c)+d


def odr_fit(func, dom, mrng, srng, pg):
    ''' performs orthogonal distance regression '''
    # load data
    dat = RealData(dom, mrng, EPS*np.ones(len(dom)), srng+EPS)
    # model function
    mod = ODRModel(func)
    # odr initialization
    odr = ODR(dat, mod, pg)
    # ord fit
    odr.set_job(fit_type=0)
    fit = odr.run()
    # optimal parameters and error
    popt = fit.beta
    perr = fit.sd_beta
    # new domain
    ndom = 128
    fdom = np.linspace(np.min(dom), np.max(dom), ndom)
    fval = func(popt, fdom)
    # return optimal parameters, errors, and function values
    return popt, perr, fdom, fval


def periodic_pad_input(x):
    p = 1
    tl = x[:, -p:, -p:, :]
    tc = x[:, -p:, :, :]
    tr = x[:, -p:, :p, :]
    ml = x[:, :, -p:, :]
    mc = x
    mr = x[:, :, :p, :]
    bl = x[:, :p, -p:, :]
    bc = x[:, :p, :, :]
    br = x[:, :p, :p, :]
    top = K.concatenate((tl, tc, tr), axis=2)
    middle = K.concatenate((ml, mc, mr), axis=2)
    bottom = K.concatenate((bl, bc, br), axis=2)
    return K.concatenate((top, middle, bottom), axis=1)


def periodic_pad_conv(x):
    p = 2
    mc = x
    mr = x[:, :, :p, :]
    bc = x[:, :p, :, :]
    br = x[:, :p, :p, :]
    middle = K.concatenate((mc, mr), axis=2)
    bottom = K.concatenate((bc, br), axis=2)
    return K.concatenate((middle, bottom), axis=1)


def trim_convt(x):
    p = K.int_shape(x)[1]//2
    tl = x[:, :p, :p, :]
    tr = x[:, :p, -p:, :]
    bl = x[:, -p:, :p, :]
    br = x[:, -p:, -p:, :]
    top = K.concatenate((tl, tr), axis=2)
    bottom = K.concatenate((bl, br), axis=2)
    return K.concatenate((top, bottom), axis=1)


def gauss_sampling(beta, batch_size=None):
    ''' samples a point in a multivariate gaussian distribution '''
    if batch_size is None:
        batch_size = BS
    mu, logvar = beta
    epsilon = K.random_normal(shape=(batch_size, LD), seed=SEED)
    return mu+K.exp(0.5*logvar)*epsilon


def gauss_log_prob(z, beta=None, batch_size=None):
    ''' logarithmic probability for multivariate gaussian distribution given samples z and parameters beta = (mu, log(var)) '''
    if batch_size is None:
        batch_size = BS
    if beta is None:
        # mu = 0, stdv = 1 => log(var) = 0
        mu, logvar = K.zeros((batch_size, LD)), K.zeros((batch_size, LD))
    else:
        mu, logvar = beta
    norm = K.log(2*np.pi)
    zsc = (z-mu)*K.exp(-0.5*logvar)
    return -0.5*(zsc**2+logvar+norm)


def se(x, y, batch_size=None):
    ''' squared error between samples x and predictions y '''
    if batch_size is None:
        batch_size = BS
    return K.sum(K.reshape(K.square(x-y), (batch_size, -1)), 1)


def ae(x, y, batch_size=None):
    ''' absolute error between samples x and predictions y '''
    if batch_size is None:
        batch_size = BS
    return K.sum(K.reshape(K.abs(x-y), (batch_size, -1)), 1)


def rse(x, y, batch_size=None):
    if batch_size is None:
        batch_size = BS
    return K.sqrt(K.sum(K.reshape(K.square(x-y), (batch_size, -1)), 1))


def bc(x, y, batch_size=None):
    ''' binary crossentropy between samples x and predictions y '''
    if batch_size is None:
        batch_size = BS
    return -K.sum(K.reshape(x*K.log(y+EPS)+(1-x)*K.log(1-y+EPS),
                            (batch_size, -1)), 1)


def kld(beta):
    ''' kullback-leibler divergence for gaussian parameters beta = (mu, log(var)) '''
    mu, logvar = beta
    return 0.5*K.sum(K.exp(logvar)+K.square(mu)-logvar-1, axis=-1)


def kernel_computation(x, y, batch_size=None):
    ''' kernel trick computation for maximum mean discrepancy ccalculation '''
    if batch_size is None:
        batch_size = BS
    tiled_x = K.tile(K.reshape(x, (batch_size, 1, LD)), (1, batch_size, 1))
    tiled_y = K.tile(K.reshape(y, (1, batch_size, LD)), (batch_size, 1, 1))
    return K.exp(-K.mean(K.square(tiled_x-tiled_y), axis=2)/K.cast(LD, 'float32'))


def mmd(x, y, batch_size=None):
    ''' maximum mean discrepancy for samples x and predictions y '''
    if batch_size is None:
        batch_size = BS
    x_kernel = kernel_computation(x, x, batch_size)
    y_kernel = kernel_computation(y, y, batch_size)
    xy_kernel = kernel_computation(x, y, batch_size)
    return x_kernel+y_kernel-2*xy_kernel


def sw(z, batch_size=None):
    ''' sliced wasserstein calculation for gaussian samples z'''
    if batch_size is None:
        batch_size = BS
    nrp = batch_size**2
    theta = K.random_normal(shape=(nrp, LD))
    theta = theta/K.sqrt(K.sum(K.square(theta), axis=1, keepdims=True))
    y = K.random_uniform(shape=(batch_size, LD), minval=-0.5, maxval=0.5, seed=SEED)
    px = K.dot(z, K.transpose(theta))
    py = K.dot(y, K.transpose(theta))
    w2 = (tf.nn.top_k(tf.transpose(px), k=batch_size).values-
          tf.nn.top_k(tf.transpose(py), k=batch_size).values)**2
    return w2


def log_importance_weight(batch_size=None, dataset_size=None):
    ''' logarithmic importance weights for minibatch stratified sampling '''
    if batch_size is None:
        batch_size = BS
    if dataset_size is None:
        dataset_size = SNH*SNT*SNS
    n, m = dataset_size, batch_size-1
    strw = (n-m)/(n*m)
    w = K.concatenate((K.concatenate((1/n*K.ones((batch_size-2, 1)),
                                      strw*K.ones((1, 1)),
                                      1/n*K.ones((1, 1))), axis=0),
                       strw*K.ones((batch_size, 1)),
                       1/m*K.ones((batch_size, batch_size-2))), axis=1)
    return K.log(w)


def log_sum_exp(x):
    ''' numerically stable logarithmic sum of exponentials '''
    m = K.max(x, axis=1, keepdims=True)
    u = x-m
    m = K.squeeze(m, 1)
    return m+K.log(K.sum(K.exp(u), axis=1, keepdims=False))


def tc(z, beta, batch_size=None):
    ''' modified elbo objective using tc decomposition given gaussian samples z and parameters beta = (mu, log(var)) '''
    if batch_size is None:
        batch_size = BS
    mu, logvar = beta
    # log p(z)
    logpz = K.sum(K.reshape(gauss_log_prob(z), (batch_size, -1)), 1)
    # log q(z|x)
    logqz_x = K.sum(K.reshape(gauss_log_prob(z, (mu, logvar)), (batch_size, -1)), 1)
    # log q(z) ~ log (1/MN) sum_m q(z|x_m) = -log(MN)+log(sum_m(exp(q(z|x_m))))
    _logqz = gauss_log_prob(K.reshape(z, (batch_size, 1, LD)),
                            (K.reshape(mu, (1, batch_size, LD)),
                             K.reshape(logvar, (1, batch_size, LD))))
    if MSS:
        log_iw = log_importance_weight(batch_size=batch_size)
        logqz_prodmarginals = K.sum(log_sum_exp(K.reshape(log_iw, (batch_size, batch_size, 1))+_logqz), 1)
        logqz = log_sum_exp(log_iw+K.sum(_logqz, axis=2))
    else:
        logqz_prodmarginals = K.sum(log_sum_exp(_logqz)-K.log(K.cast(BS*SNH*SNT*SNS, 'float32')), 1)
        logqz = log_sum_exp(K.sum(_logqz, axis=2))-K.log(K.cast(BS*SNH*SNT*SNS, 'float32'))
    # alpha controls index-code mutual information
    # beta controls total correlation
    # lambda controls dimension-wise kld
    melbo = -ALPHA*(logqz_x-logqz)-BETA*(logqz-logqz_prodmarginals)-LMBDA*(logqz_prodmarginals-logpz)
    return -melbo


def build_autoencoder():
    ''' builds autoencoder network '''
    if VERBOSE:
        print('building variational autoencoder network')
        print(100*'-')
    # --------------
    # initialization
    # --------------
    # output layer activation
    # sigmoid for activations on (0, 1) tanh otherwise (-1, 1)
    # caused by scaling
    if ACT == 'lrelu':
        alpha_enc = 0.25
        alpha_dec = 0.125
    if ACT == 'prelu':
        alpha_enc = Constant(value=0.25)
        alpha_dec = Constant(value=0.125)
    elif ACT == 'elu':
        alpha_enc = 1.0
        alpha_dec = 0.125
    if PRIOR == 'gaussian':
        alpha_mu = 1.0
        alpha_logsigma = 1.0
    alpha_z = 1.0
    if SCLR in ['minmax', 'tanh', 'global']:
        decact = 'sigmoid'
    else:
        decact = 'tanh'
    dp = 0.5
    # kernel initializer - customizable
    # limited tests showed he_normal performs well
    init = KIS[KI]
    # number of convolutions necessary to get down to size length 4
    # should use base 2 exponential (integer) side lengths
    nc = np.int32(np.log2(N/CD))
    cr = CR+1
    # --------------
    # encoder layers
    # --------------
    # input layer
    input = Input(shape=(N, N, NCH), name='encoder_input')
    c = Lambda(periodic_pad_conv, name='periodic_pad_input')(input)
    u = 0
    # loop through convolutions
    # filter size of (3, 3) to capture nearest neighbors from input
    for i in range(nc):
        for j in range(cr):
            k = 3
            if cr == 1:
                s = 2
            elif cr == 2:
                s = j+1
            p = 'same'
            if VGG:
                nf = 2**(j % 2)*NF
            else:
                nf = 4**i*NF
            # convolution
            c = Conv2D(filters=nf, kernel_size=k, kernel_initializer=init,
                       padding=p, strides=s, name='conv_%d' % u)(c)
            # activations
            if ACT == 'prelu':
                c = PReLU(alpha_initializer=alpha_enc, name='prelu_conv_%d' % u)(c)
            elif ACT == 'lrelu':
                c = LeakyReLU(alpha=alpha_enc, name='lrelu_conv_%d' % u)(c)
            elif ACT == 'elu':
                c = ELU(alpha=alpha_enc)(c)
            elif ACT == 'selu':
                c = Activation('selu', name='selu_conv_%d' % u)(c)
            if DO:
                c = Dropout(rate=dp, name='dropout_conv_%d' % u)(c)
            # batch normalization to scale activations
            if BN:
                c = BatchNormalization(name='batch_norm_conv_%d' % u)(c)
            u += 1
    # flatten convolutional output
    shape = K.int_shape(c)
    d0 = Flatten(name='flatten')(c)
    if PRIOR == 'gaussian':
        # gaussian parameters as dense layers
        mu = Dense(LD, name='mu', kernel_initializer=init)(d0)
        mu = Activation('linear', name='linear_mu')(mu)
        # more numerically stable to use log(var_z)
        logvar = Dense(LD, name='log_var', kernel_initializer=init)(d0)
        logvar = Activation('linear', name='linear_log_var')(logvar)
        # sample from gaussian
        z = Lambda(gauss_sampling, output_shape=(LD,), name='latent_encoding')([mu, logvar])
        # construct encoder
        encoder = Model(input, [mu, logvar, z], name='encoder')
    elif PRIOR == 'none':
        # encoding
        z = Dense(LD, name='latent_encoding', kernel_initializer=init)(d0)
        # construct encoder
        encoder = Model(input, z, name='encoder')
    if VERBOSE:
        print('encoder network summary')
        print(100*'-')
        encoder.summary()
        print(100*'-')
    # --------------
    # decoder layers
    # --------------
    # input layer (latent variables z)
    latent_input = Input(shape=(LD,), name='latent_encoding')
    # dense network of same size as convolution output from encoder
    d1 = Dense(np.prod(shape[1:]), kernel_initializer=init, name='latent_expansion')(latent_input)
    # reshape to convolution shape
    ct = Reshape(shape[1:], name='reshape_latent_expansion')(d1)
    # activations
    if ACT == 'prelu':
        ct = PReLU(alpha_initializer=alpha_dec, name='prelu_latent_expansion')(ct)
    elif ACT == 'lrelu':
        ct = LeakyReLU(alpha=alpha_dec, name='lrelu_latent_expansion')(ct)
    elif ACT == 'elu':
        ct = ELU(alpha=alpha_dec)(ct)
    elif ACT == 'selu':
        ct = Activation('selu', name='selu_latent_expansion')(ct)
    if DO:
        ct = Dropout(rate=dp, name='dropout_latent_expansion')(ct)
    # batch renormalization to scale activations
    if BN:
        ct = BatchNormalization(name='batch_norm_latent_expansion')(ct)
    u = 0
    # loop through convolution transposes
    for i in range(nc-1, -1, -1):
        for j in range(cr-1, -1, -1):
            k = 3
            p = 'same'
            if i == 0 and j == 0:
                if cr == 1:
                    s = 2
                elif cr == 2:
                    s = 1
                nf = NCH
                # output convolution transpose layer
                output = Conv2DTranspose(filters=nf, kernel_size=k, kernel_initializer=init,
                                         padding=p, strides=s, name='reconst')(ct)
                # output layer activation
                output = Activation(decact, name='activated_reconst')(output)
            else:
                if cr == 1:
                    s = 2
                elif cr == 2:
                    s = j+1
                if VGG:
                    nf = np.int32(2**((j-1) % 2)*NF)
                else:
                    nf = np.int32(4**(i+j-1)*NF)
                # transposed convolution
                ct = Conv2DTranspose(filters=nf, kernel_size=k, kernel_initializer=init,
                                    padding=p, strides=s, name='convt_%d' % u)(ct)
                # activations
                if ACT == 'prelu':
                    ct = PReLU(alpha_initializer=alpha_dec, name='prelu_convt_%d' % u)(ct)
                elif ACT == 'lrelu':
                    ct = LeakyReLU(alpha=alpha_dec, name='lrelu_convt_%d' % u)(ct)
                elif ACT == 'elu':
                    ct = ELU(alpha=alpha_dec)(ct)
                elif ACT == 'selu':
                    ct = Activation('selu', name='selu_convt_%d' % u)(ct)
                if DO:
                    ct = Dropout(rate=dp, name='dropout_convt_%d' % u)(ct)
                # batch normalization to scale activations
                if BN:
                    ct = BatchNormalization(name='batch_norm_convt_%d' % u)(ct)
                u += 1
    # construct decoder
    decoder = Model(latent_input, output, name='decoder')
    if VERBOSE:
        print('decoder network summary')
        print(100*'-')
        decoder.summary()
        print(100*'-')
    # ------------
    # construct ae
    # ------------
    # combine encoder and decoder
    if PRIOR == 'gaussian':
        output = decoder(encoder(input)[2])
    elif PRIOR == 'none':
        output = decoder(encoder(input))
    ae = Model(input, output, name='autoencoder')
    # ae loss
    # reconstruction loss
    rc = RCLS[LSS](input, output)
    if PRIOR == 'gaussian':
        # distribution divergence to regularize latent space
        if REG == 'kld':
            rg = BETA*REGS[REG]((mu, logvar))
        elif REG == 'mmd':
            rg = BETA*REGS['kld']((mu, logvar))+\
                 LMBDA*REGS[REG](K.random_normal(shape=(BS, LD)), z)
        elif REG == 'sw':
            rg = BETA*REGS['kld']((mu, logvar))+\
                 LMBDA*REGS[REG](z)
        elif REG == 'tc':
            rg = REGS['tc'](z, (mu, logvar))
    elif PRIOR == 'none' or REG == 'none':
        # no regularization
        rg = 0.0
    # combine losses
    ae_loss = K.mean(rc+rg)
    ae.add_loss(ae_loss)
    # compile ae
    ae.compile(optimizer=OPTS[OPT])
    # return ae networks
    return encoder, decoder, ae


def random_selection(dmp, dat, intrvl, ns):
    ''' selects random subset of data according to phase point interval and sample count '''
    rdmp = dmp[::intrvl, ::intrvl]
    rdat = dat[::intrvl, ::intrvl]
    nh, nt, _, _, _ = rdmp.shape
    idat = np.zeros((nh, nt, ns), dtype=np.uint16)
    if VERBOSE:
        print('selecting random classification samples from full data')
        print(100*'-')
    for i in tqdm(range(nh), disable=not VERBOSE):
        for j in tqdm(range(nt), disable=not VERBOSE):
                idat[i, j] = np.random.permutation(rdat[i, j].shape[0])[:ns]
    if VERBOSE:
        print('\n'+100*'-')
    sldmp = np.array([[rdmp[i, j, idat[i, j], :, :] for j in range(nt)] for i in range(nh)])
    sldat = np.array([[rdat[i, j, idat[i, j], :] for j in range(nt)] for i in range(nh)])
    return sldmp, sldat


def inlier_selection(dmp, dat, intrvl, ns):
    ''' selects inlier subset of data '''
    rdmp = dmp[::intrvl, ::intrvl]
    rdat = dat[::intrvl, ::intrvl]
    nh, nt, _, _, _ = rdmp.shape
    if AD:
        lof = LocalOutlierFactor(contamination='auto', n_jobs=THREADS)
    idat = np.zeros((nh, nt, ns), dtype=np.uint16)
    if VERBOSE:
        print('selecting inlier samples from classification data')
        print(100*'-')
    for i in tqdm(range(nh), disable=not VERBOSE):
        for j in tqdm(range(nt), disable=not VERBOSE):
                if AD:
                    fpred = lof.fit_predict(rdmp[i, j, :, 0])
                    try:
                        idat[i, j] = np.random.choice(np.where(fpred==1)[0], size=ns, replace=False)
                    except:
                        idat[i, j] = np.argsort(lof.negative_outlier_factor_)[:ns]
                else:
                    idat[i, j] = np.random.permutation(rdat[i, j].shape[0])[:ns]
    if VERBOSE:
        print('\n'+100*'-')
    sldmp = np.array([[rdmp[i, j, idat[i, j], :, :] for j in range(nt)] for i in range(nh)])
    sldat = np.array([[rdat[i, j, idat[i, j], :] for j in range(nt)] for i in range(nh)])
    return sldmp, sldat


if __name__ == '__main__':
    # parse command line arguments
    (VERBOSE, PLOT, PARALLEL, GPU, THREADS, NAME,
     N, SNI, SNS, SCLR,
     PRIOR, KI, VGG, CD, CR, NF, ACT, BN, DO, LD,
     OPT, LR, LSS, REG, ALPHA, BETA, LMBDA, MSS,
     EP, SH, BS, SEED) = parse_args()
    CWD = os.getcwd()
    EPS = 1e-8
    # number of phases
    NPH = 3
    # number of embedding dimensions
    if PRIOR == 'gaussian':
        ED = 2
    elif PRIOR == 'none':
        ED = 1
        ALPHA = 0
        BETA = 0
        LMBDA = 0
        MSS = 0
        REG = 'none'
    if REG != 'tc':
        ALPHA = 0
        MSS = 0
    if REG == 'kld':
        LMBDA = 0

    np.random.seed(SEED)
    # environment variables
    os.environ['KERAS_BACKEND'] = 'tensorflow'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    from tensorflow import set_random_seed
    set_random_seed(SEED)
    if PARALLEL:
        if not GPU:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        os.environ['MKL_NUM_THREADS'] = str(THREADS)
        os.environ['GOTO_NUM_THREADS'] = str(THREADS)
        os.environ['OMP_NUM_THREADS'] = str(THREADS)
        os.environ['openmp'] = 'True'
    else:
        THREADS = 1
    if GPU:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # imports
    import tensorflow as tf
    from keras.models import Model
    from keras.layers import (Input, Lambda, Dense, Conv2D, Conv2DTranspose,
                              Flatten, Reshape, BatchNormalization, Activation, Dropout)
    from keras.optimizers import SGD, Adadelta, Adam, Nadam
    from keras.initializers import (Zeros, Ones, Constant, RandomNormal, RandomUniform,
                                    TruncatedNormal, VarianceScaling, glorot_uniform, glorot_normal,
                                    lecun_uniform, lecun_normal, he_uniform, he_normal)
    from keras.activations import relu, tanh, sigmoid, linear
    from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU
    from keras.callbacks import History, CSVLogger, ReduceLROnPlateau
    from keras import backend as K
    from keras.utils import plot_model
    if PLOT:
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable
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
        SCALE = lambda a, b: (a-np.min(b))/(np.max(b)-np.min(b))
        CM = plt.get_cmap('plasma')

    # external fields and temperatures
    CH = np.load(CWD+'/%s.%d.h.npy' % (NAME, N))[::SNI]
    CT = np.load(CWD+'/%s.%d.t.npy' % (NAME, N))[::SNI]
    # external field and temperature counts
    SNH, SNT = CH.size, CT.size
    # number of data channels
    NCH = 1

    # run parameter tuple
    PRM = (NAME, N, SNI, SNS, SCLR,
           PRIOR, VGG, CD, CR, NF, ACT, BN, DO, LD, OPT, LR,
           LSS, REG, ALPHA, BETA, LMBDA, MSS,
           EP, SH, BS, SEED)
    # output file prefix
    OUTPREF = CWD+'/%s.%d.%d.%d.%s.%s.%d.%d.%d.%d.%s.%d.%d.%d.%s.%.0e.%s.%s.%.0e.%.0e.%.0e.%d.%d.%d.%d.%d' % PRM
    # write output file header
    write_specs()

    # feature range
    FMN, FMX = (0.0, 1.0)
    # scaler dictionary
    SCLRS = {'minmax':MinMaxScaler(feature_range=(FMN, FMX)),
             'standard':StandardScaler(),
             'robust':RobustScaler(),
             'tanh':TanhScaler(feature_range=(FMN, FMX))}

    # array shapes
    SHP0 = (SNH, SNT, SNS, N, N, NCH)
    SHP1 = (SNH*SNT*SNS, N, N, NCH)
    SHP2 = (SNH*SNT*SNS, N*N*NCH)
    SHP3 = (SNH, SNT, SNS, ED, LD)
    SHP4 = (SNH*SNT*SNS, ED, LD)
    SHP5 = (SNH*SNT*SNS, ED*LD)

    # scaled data dump prefix
    CPREF = CWD+'/%s.%d.%d.%d.%d' % (NAME, N, SNI, SNS, SEED)
    SCPREF = CWD+'/%s.%d.%d.%d.%s.%d' % (NAME, N, SNI, SNS, SCLR, SEED)

    try:
        # check is scaled data has already been computed
        SCDMP = np.load(SCPREF+'.dmp.sc.npy').reshape(*SHP1)
        CDAT = np.load(CPREF+'.dat.c.npy')
        if VERBOSE:
            print('scaled selected classification samples loaded from file')
            print(100*'-')
    except:
        try:
            # check if random data subset has already been selected
            CDMP = np.load(CPREF+'.dmp.c.npy')
            CDAT = np.load(CPREF+'.dat.c.npy')
            if VERBOSE:
                # print(100*'-')
                print('selected classification samples loaded from file')
                print(100*'-')
        except:
            # data selection
            DMP = np.load(CWD+'/%s.%d.dmp.npy' % (NAME, N))
            DAT = np.load(CWD+'/%s.%d.dat.npy' % (NAME, N))
            if VERBOSE:
                # print(100*'-')
                print('full dataset loaded from file')
                print(100*'-')
            CDMP, CDAT = random_selection(DMP, DAT, SNI, SNS)
            del DAT, DMP
            np.save(CPREF+'.dmp.c.npy', CDMP)
            np.save(CPREF+'.dat.c.npy', CDAT)
            if VERBOSE:
                print('selected classification samples generated')
                print(100*'-')

        # data scaling
        if SCLR == 'global':
            SCDMP = CDMP.reshape(*SHP1)
            for i in range(NCH):
                TMIN, TMAX = SCDMP[:, :, :, i].min(), SCDMP[:, :, :, i].max()
                SCDMP[:, :, :, i] = (SCDMP[:, :, :, i]-TMIN)/(TMAX-TMIN)
        elif SCLR == 'none':
            SCDMP = CDMP.reshape(*SHP1)
        else:
            SCDMP = SCLRS[SCLR].fit_transform(CDMP.reshape(*SHP2)).reshape(*SHP1)
        del CDMP
        np.save(SCPREF+'.dmp.sc.npy', SCDMP.reshape(*SHP0))
        if VERBOSE:
            print('scaled selected classification samples computed')
            print(100*'-')

    # energies
    ES = CDAT[:, :, :, 0]
    # magnetizations
    MS = CDAT[:, :, :, 1]
    # mean energies
    EM = np.mean(ES, -1)
    # specific heat capacities
    SP = np.var(np.divide(ES, CT[np.newaxis, :, np.newaxis]), 2)
    # mean magnetizations
    MM = np.mean(MS, -1)
    # magnetic susceptibilities
    SU = np.var(MS/CT[np.newaxis, :, np.newaxis], 2)

    # kernel initializers
    KIS = {'glorot_uniform': glorot_uniform(SEED),
           'lecun_uniform': lecun_uniform(SEED),
           'he_uniform': he_uniform(SEED),
           'varscale': VarianceScaling(seed=SEED),
           'glorot_normal': glorot_normal(SEED),
           'lecun_normal': lecun_normal(SEED),
           'he_normal': he_normal(SEED)}

    # optmizers
    OPTS = {'sgd': SGD(lr=LR, momentum=0.0, decay=0.0, nesterov=True),
            'adadelta': Adadelta(lr=LR, rho=0.95, epsilon=None, decay=0.0),
            'adam': Adam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True),
            'nadam': Nadam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)}

    # reconstruction losses
    RCLS = {'mse': lambda a, b: se(a, b),
            'mae': lambda a, b: ae(a, b),
            'rmse': lambda a, b: rse(a, b),
            'bc': lambda a, b: bc(a, b)}

    # regularizers
    REGS = {'kld': lambda a: kld(a),
            'mmd': lambda a, b: mmd(a, b),
            'sw': lambda a: sw(a),
            'tc': lambda a, b: tc(a, b)}

    # autencoder networks
    ENC, DEC, AE = build_autoencoder()

    try:
        # check if model already trained
        AE.load_weights(OUTPREF+'.ae.wt.h5', by_name=True)
        TLOSS = np.load(OUTPREF+'.ae.loss.trn.npy')
        # VLOSS = np.load(OUTPREF+'.ae.loss.val.npy')
        if VERBOSE:
            print('autoencoder trained weights loaded from file')
            print(100*'-')
    except:
        if VERBOSE:
            print('autoencoder training on scaled selected classification samples')
            print(100*'-')
        # output log
        CSVLG = CSVLogger(OUTPREF+'.ae.log.csv', append=True, separator=',')
        # learning rate decay on loss plateau
        LR_DECAY = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=8, verbose=VERBOSE)
        AE.fit(x=np.moveaxis(SCDMP.reshape(*SHP0), 2, 0)[np.random.shuffle(np.arange(SNS)),
                                                         np.random.shuffle(np.arange(SNH)),
                                                         np.random.shuffle(np.arange(SNT))].reshape(*SHP1),
               y=None, epochs=EP, batch_size=BS, shuffle=SH, verbose=VERBOSE, callbacks=[CSVLG, LR_DECAY, History()])
        TLOSS = AE.history.history['loss']
        # VLOSS = AE.history.history['val_loss']
        AE.save_weights(OUTPREF+'.ae.wt.h5')
        np.save(OUTPREF+'.ae.loss.trn.npy', TLOSS)
        # np.save(OUTPREF+'.ae.loss.val.npy', VLOSS)
        if VERBOSE:
            print(100*'-')
            print('autoencoder weights trained')
            print(100*'-')

    if VERBOSE:
        print('autoencoder training history information')
        print(100*'-')
        # print('| epoch | training loss | validation loss |')
        print('| epoch | training loss |')
        print(100*'-')
        for i in range(EP):
            # print('%02d %.2f %.2f' % (i, TLOSS[i], VLOSS[i]))
            print('%02d %.2f' % (i, TLOSS[i]))
        print(100*'-')

    with open(OUTPREF+'.out', 'a') as out:
        out.write('variational autoencoder training history information\n')
        out.write(100*'-' + '\n')
        # out.write('| epoch | training loss | validation loss |\n')
        out.write('| epoch | training loss |\n')
        out.write(100*'-' + '\n')
        for i in range(EP):
            # out.write('%02d %.2f %.2f\n' % (i, TLOSS[i], VLOSS[i]))
            out.write('%02d %.2f\n' % (i, TLOSS[i]))
        out.write(100*'-' + '\n')

    try:
        ZENC = np.load(OUTPREF+'.zenc.npy').reshape(*SHP4)
        ZDEC = np.load(OUTPREF+'.zdec.npy').reshape(*SHP1)
        ERR = np.load(OUTPREF+'.zerr.npy').reshape(*SHP0)
        MERR = np.load(OUTPREF+'.zerr.mean.npy')
        SERR = np.load(OUTPREF+'.zerr.stdv.npy')
        MXERR = np.load(OUTPREF+'.zerr.max.npy')
        MNERR = np.load(OUTPREF+'.zerr.min.npy')
        MAEERR = np.load(OUTPREF+'.zerr.mae.npy')
        RMSERR = np.load(OUTPREF+'.zerr.rms.npy')
        R2SCERR = np.load(OUTPREF+'.zerr.r2sc.npy')
        if PRIOR == 'gaussian':
            KLD = np.load(OUTPREF+'.zerr.kld.npy')
            MKLD = np.load(OUTPREF+'.zerr.kld.mean.npy')
            SKLD = np.load(OUTPREF+'.zerr.kld.stdv.npy')
            MXKLD = np.load(OUTPREF+'.zerr.kld.max.npy')
            MNKLD = np.load(OUTPREF+'.zerr.kld.min.npy')
        if VERBOSE:
            print('latent encodings of scaled selected classification samples loaded from file')
            print(100*'-')
    except:
        if VERBOSE:
            print('predicting latent encodings of scaled selected classification samples')
            print(100*'-')
        ZENC = np.array(ENC.predict(SCDMP, batch_size=BS, verbose=VERBOSE))
        if PRIOR == 'gaussian':
            KLD = K.eval(kld((ZENC[0, :, :], ZENC[1, :, :]))).reshape(SNH, SNT, SNS)
            ZDEC = np.array(DEC.predict(ZENC[2, :, :], batch_size=BS, verbose=VERBOSE))
            # swap latent space axes
            ZENC = np.swapaxes(ZENC, 0, 1)[:, :2, :]
            # convert log variance to standard deviation
            ZENC[:, 1, :] = np.exp(0.5*ZENC[:, 1, :])
            np.save(OUTPREF+'.zerr.kld.npy', KLD)
        elif PRIOR == 'none':
            ZDEC = np.array(DEC.predict(ZENC, verbose=VERBOSE))
            ZENC = ZENC[:, np.newaxis, :]
        # reconstruction error (signed)
        ERR = (SCDMP-ZDEC).reshape(*SHP0)
        # dump results
        np.save(OUTPREF+'.zenc.npy', ZENC.reshape(*SHP3))
        np.save(OUTPREF+'.zdec.npy', ZDEC.reshape(*SHP0))
        np.save(OUTPREF+'.zerr.npy', ERR)
        # mean and standard deviation of error
        MERR = np.mean(ERR)
        SERR = np.std(ERR)
        # minimum and maximum error
        MXERR = np.max(ERR)
        MNERR = np.min(ERR)
        MAEERR = np.mean(np.abs(ERR))
        RMSERR = np.sqrt(np.mean(np.square(ERR)))
        R2SCERR = 1-np.square(RMSERR)/0.25
        if PRIOR  == 'gaussian':
            # mean and standard deviation kullback-liebler divergence
            MKLD = np.mean(KLD)
            SKLD = np.std(KLD)
            # minimum and maximum kullback-liebler divergence
            MXKLD = np.max(KLD)
            MNKLD = np.min(KLD)
        # dump results
        np.save(OUTPREF+'.zerr.mean.npy', MERR)
        np.save(OUTPREF+'.zerr.stdv.npy', SERR)
        np.save(OUTPREF+'.zerr.max.npy', MXERR)
        np.save(OUTPREF+'.zerr.min.npy', MNERR)
        np.save(OUTPREF+'.zerr.mae.npy', MAEERR)
        np.save(OUTPREF+'.zerr.rms.npy', RMSERR)
        np.save(OUTPREF+'.zerr.r2sc.npy', R2SCERR)
        if PRIOR == 'gaussian':
            np.save(OUTPREF+'.zerr.kld.mean.npy', MKLD)
            np.save(OUTPREF+'.zerr.kld.stdv.npy', SKLD)
            np.save(OUTPREF+'.zerr.kld.max.npy', MXKLD)
            np.save(OUTPREF+'.zerr.kld.min.npy', MNKLD)

    if VERBOSE:
        print(100*'-')
        print('latent encodings of scaled selected classification samples predicted')
        print(100*'-')
        print('mean sig:        %f' % np.mean(SCDMP))
        print('stdv sig:        %f' % np.std(SCDMP))
        print('max sig          %f' % np.max(SCDMP))
        print('min sig          %f' % np.min(SCDMP))
        print('mean error:      %f' % MERR)
        print('stdv error:      %f' % SERR)
        print('max error:       %f' % MXERR)
        print('min error:       %f' % MNERR)
        print('mae error:       %f' % MAEERR)
        print('rms error:       %f' % RMSERR)
        print('r2 score:        %f' % R2SCERR)
        if PRIOR == 'gaussian':
            print('mean kld:        %f' % MKLD)
            print('stdv kld:        %f' % SKLD)
            print('max kld:         %f' % MXKLD)
            print('min kld:         %f' % MNKLD)
        print(100*'-')
    with open(OUTPREF+'.out', 'a') as out:
        out.write('fitting errors\n')
        out.write(100*'-'+'\n')
        out.write('mean sig:        %f\n' % np.mean(SCDMP))
        out.write('stdv sig:        %f\n' % np.std(SCDMP))
        out.write('max sig          %f\n' % np.max(SCDMP))
        out.write('min sig          %f\n' % np.min(SCDMP))
        out.write('mean error:      %f\n' % MERR)
        out.write('stdv error:      %f\n' % SERR)
        out.write('max error:       %f\n' % MXERR)
        out.write('min error:       %f\n' % MNERR)
        out.write('mae error:       %f\n' % MAEERR)
        out.write('rms error:       %f\n' % RMSERR)
        out.write('r2 score:        %f\n' % R2SCERR)
        if PRIOR == 'gaussian':
            out.write('mean kld:        %f\n' % MKLD)
            out.write('stdv kld:        %f\n' % SKLD)
            out.write('max kld:         %f\n' % MXKLD)
            out.write('min kld:         %f\n' % MNKLD)
        out.write(100*'-'+'\n')

    try:
        # check if pca projections already computed
        PZENC = np.load(OUTPREF+'.zenc.pca.prj.npy').reshape(*SHP4)
        CZENC = np.load(OUTPREF+'.zenc.pca.cmp.npy')
        VZENC = np.load(OUTPREF+'.zenc.pca.var.npy')
        if VERBOSE:
            print('pca projections of latent encodings loaded from file')
            print(100*'-')
    except:
        if VERBOSE:
            print('pca projecting latent encodings')
            print(100*'-')
        PCAZENC = PCA(n_components=LD)
        # pca embeddings of latent encodings
        PZENC = np.zeros((SNH*SNT*SNS, ED, LD))
        # pca projections of latent encodings
        CZENC = np.zeros((ED, LD, LD))
        # explained variance ratios
        VZENC = np.zeros((ED, LD))
        for i in range(ED):
            PZENC[:, i, :] = PCAZENC.fit_transform(ZENC[:, i, :])
            CZENC[i, :, :] = PCAZENC.components_
            VZENC[i, :] = PCAZENC.explained_variance_ratio_
        np.save(OUTPREF+'.zenc.pca.prj.npy', PZENC.reshape(*SHP3))
        np.save(OUTPREF+'.zenc.pca.cmp.npy', CZENC)
        np.save(OUTPREF+'.zenc.pca.var.npy', VZENC)

    if VERBOSE:
        print('pca fit information')
        print(100*'-')
        for i in range(ED):
            if i == 0:
                print('mean z fit')
            if i == 1:
                print('stdv z fit')
            print(100*'-')
            print('components')
            print(100*'-')
            for j in range(LD):
                print(LD*'%+f ' % tuple(CZENC[i, j, :]))
            print(100*'-')
            print('explained variances')
            print(100*'-')
            print(LD*'%f ' % tuple(VZENC[i, :]))
            print(100*'-')
    with open(OUTPREF+'.out', 'a') as out:
        out.write('pca fit information\n')
        out.write(100*'-'+'\n')
        for i in range(ED):
            if i == 0:
                out.write('mean z fit\n')
            if i == 1:
                out.write('stdv z fit\n')
            out.write(100*'-'+'\n')
            out.write('principal components\n')
            out.write(100*'-'+'\n')
            for j in range(LD):
                out.write(LD*'%+f ' % tuple(CZENC[i, j, :]) + '\n')
            out.write(100*'-'+'\n')
            out.write('explained variances\n')
            out.write(100*'-'+'\n')
            out.write(LD*'%f ' % tuple(VZENC[i, :]) + '\n')
            out.write(100*'-'+'\n')

    def ae_plots():

        # plot_model(AE, to_file=OUTPREF+'.model.ae.png')
        # plot_model(ENC, to_file=OUTPREF+'.model.enc.png')
        # plot_model(DEC, to_file=OUTPREF+'.model.dec.png')

        # plot signed reconstruction error distribution
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ex = np.linspace(np.floor(ERR.min()), np.ceil(ERR.max()), 33)
        er = np.histogram(ERR, ex)[0]/(SNH*SNT*SNS*N*N)
        dex = ex[1]-ex[0]
        ey = np.array([0.0, 0.05, 0.1, 0.25, 0.5])
        ax.bar(ex[1:]-0.5*dex, er, dex, color=CM(0.15))
        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
        ax.set_xticks(np.linspace(np.floor(ERR.min()), np.ceil(ERR.max()), 5))
        ax.set_yticks(ey)
        ax.set_xticklabels(np.linspace(np.floor(ERR.min()), np.ceil(ERR.max()), 5))
        ax.set_yticklabels(ey)
        ax.set_xlabel('ERROR')
        ax.set_ylabel('HISTOGRAM')
        fig.savefig(OUTPREF+'.ae.dist.err.me.png')
        plt.close()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ex = np.linspace(np.floor(np.abs(ERR).min()), np.ceil(np.abs(ERR).max()), 17)
        er = np.histogram(np.abs(ERR), ex)[0]/(SNH*SNT*SNS*N*N)
        dex = ex[1]-ex[0]
        ey = np.array([0.0, 0.05, 0.1, 0.25, 0.5])
        ax.bar(ex[1:]-0.5*dex, er, dex, color=CM(0.15))
        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
        ax.set_xticks(np.linspace(np.floor(np.abs(ERR).min()), np.ceil(np.abs(ERR).max()), 5))
        ax.set_yticks(ey)
        ax.set_xticklabels(np.linspace(np.floor(np.abs(ERR).min()), np.ceil(np.abs(ERR).max()), 5))
        ax.set_yticklabels(ey)
        ax.set_xlabel('ERROR')
        ax.set_ylabel('HISTOGRAM')
        fig.savefig(OUTPREF+'.ae.dist.err.mae.png')
        plt.close()

        fig, ax = plt.subplots()
        div = make_axes_locatable(ax)
        cax = div.append_axes('top', size='5%', pad=0.8)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        dat = np.sqrt(np.mean(np.square(ERR), (2, 3, 4, 5)))
        im = ax.imshow(dat, aspect='equal', interpolation='none', origin='lower', cmap=CM)
        ax.grid(which='minor', axis='both', linestyle='-', color='k', linewidth=1)
        ax.set_xticks(np.arange(CT.size), minor=True)
        ax.set_yticks(np.arange(CH.size), minor=True)
        ax.set_xticks(np.arange(CT.size)[::4], minor=False)
        ax.set_yticks(np.arange(CH.size)[::4], minor=False)
        ax.set_xticklabels(np.round(CT, 2)[::4], rotation=-60)
        ax.set_yticklabels(np.round(CH, 2)[::4])
        ax.set_xlabel('T')
        ax.set_ylabel('H')
        fig.colorbar(im, cax=cax, orientation='horizontal', ticks=np.linspace(dat.min(), dat.max(), 3))
        fig.savefig(OUTPREF+'.ae.diag.err.rms.png')
        plt.close()

        fig, ax = plt.subplots()
        div = make_axes_locatable(ax)
        cax = div.append_axes('top', size='5%', pad=0.8)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        dat = np.mean(ERR, (2, 3, 4, 5))
        im = ax.imshow(dat, aspect='equal', interpolation='none', origin='lower', cmap=CM)
        ax.grid(which='minor', axis='both', linestyle='-', color='k', linewidth=1)
        ax.set_xticks(np.arange(CT.size), minor=True)
        ax.set_yticks(np.arange(CH.size), minor=True)
        ax.set_xticks(np.arange(CT.size)[::4], minor=False)
        ax.set_yticks(np.arange(CH.size)[::4], minor=False)
        ax.set_xticklabels(np.round(CT, 2)[::4], rotation=-60)
        ax.set_yticklabels(np.round(CH, 2)[::4])
        ax.set_xlabel('T')
        ax.set_ylabel('H')
        fig.colorbar(im, cax=cax, orientation='horizontal', ticks=np.linspace(dat.min(), dat.max(), 3))
        fig.savefig(OUTPREF+'.ae.diag.err.me.png')
        plt.close()

        if PRIOR == 'gaussian':
            # plot kullback-liebler divergence distribution
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
            ex = np.linspace(0.0, np.ceil(KLD.max()), 33)
            er = np.histogram(KLD, ex)[0]/(SNH*SNT*SNS)
            dex = ex[1]-ex[0]
            ey = np.array([0.0, 0.05, 0.1, 0.25, 0.5])
            ax.bar(ex[1:]-0.5*dex, er, dex, color=CM(0.15))
            ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
            ax.set_xticks(np.linspace(0.0, np.ceil(KLD.max()), 5))
            ax.set_yticks(ey)
            ax.set_xticklabels(np.linspace(0.0, np.ceil(KLD.max()), 5))
            ax.set_yticklabels(ey)
            ax.set_xlabel('KLD')
            ax.set_ylabel('HISTOGRAM')
            fig.savefig(OUTPREF+'.ae.dist.kld.png')
            plt.close()

            fig, ax = plt.subplots()
            div = make_axes_locatable(ax)
            cax = div.append_axes('top', size='5%', pad=0.8)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
            dat = np.mean(KLD, axis=-1)
            im = ax.imshow(dat, aspect='equal', interpolation='none', origin='lower', cmap=CM)
            ax.grid(which='minor', axis='both', linestyle='-', color='k', linewidth=1)
            ax.set_xticks(np.arange(CT.size), minor=True)
            ax.set_yticks(np.arange(CH.size), minor=True)
            ax.set_xticks(np.arange(CT.size)[::4], minor=False)
            ax.set_yticks(np.arange(CH.size)[::4], minor=False)
            ax.set_xticklabels(np.round(CT, 2)[::4], rotation=-60)
            ax.set_yticklabels(np.round(CH, 2)[::4])
            ax.set_xlabel('T')
            ax.set_ylabel('H')
            fig.colorbar(im, cax=cax, orientation='horizontal', ticks=np.linspace(dat.min(), dat.max(), 3))
            fig.savefig(OUTPREF+'.ae.diag.kld.png')
            plt.close()

        shp0 = (SNH, SNT, SNS, ED*LD)
        shp1 = (SNH, SNT, ED, LD)
        shp2 = (SNH*SNT, ED*LD)
        shp3 = (SNH, SNT, SNS, ED, LD)
        ct = CT[np.newaxis, :, np.newaxis, np.newaxis]

        # diagrams for physical measurements
        MEMDIAG = np.stack((MM, EM), axis=-1).reshape(SNH, SNT, 2)
        MEVDIAG = np.stack((SU, SP), axis=-1).reshape(SNH, SNT, 2)
        # diagrams for latent variables
        ZMDIAG = np.mean(ZENC.reshape(*shp0), 2).reshape(*shp1)
        ZVDIAG = np.var(np.divide(ZENC.reshape(*shp0), ct), 2).reshape(*shp1)
        # diagrams for pca embeddings of latent variables
        PZMDIAG = np.mean(PZENC.reshape(*shp0), 2).reshape(*shp1)
        for i in range(LD):
            if PZMDIAG[0, 0, 0, i] > PZMDIAG[-1, 0, 0, i]:
                PZMDIAG[:, :, 0, i] = -PZMDIAG[:, :, 0, i]
            if PRIOR == 'gaussian':
                if PZMDIAG[0, 0, 1, i] > PZMDIAG[0, -1, 1, i]:
                    PZMDIAG[:, :, 1, i] = -PZMDIAG[:, :, 1, i]
        PZVDIAG = np.var(np.divide(PZENC.reshape(*shp0), ct), 2).reshape(*shp1)
        # plot latent variable diagrams
        for i in range(2):
            for j in range(ED):
                for k in range(LD):
                    fig, ax = plt.subplots()
                    div = make_axes_locatable(ax)
                    cax = div.append_axes('top', size='5%', pad=0.8)
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                    ax.xaxis.set_ticks_position('bottom')
                    ax.yaxis.set_ticks_position('left')
                    if i == 0:
                        dat = ZMDIAG[:, :, j, k]
                    if i == 1:
                        dat = ZVDIAG[:, :, j, k]
                    im = ax.imshow(dat, aspect='equal', interpolation='none', origin='lower', cmap=CM)
                    ax.grid(which='minor', axis='both', linestyle='-', color='k', linewidth=1)
                    ax.set_xticks(np.arange(CT.size), minor=True)
                    ax.set_yticks(np.arange(CH.size), minor=True)
                    ax.set_xticks(np.arange(CT.size)[::4], minor=False)
                    ax.set_yticks(np.arange(CH.size)[::4], minor=False)
                    ax.set_xticklabels(np.round(CT, 2)[::4], rotation=-60)
                    ax.set_yticklabels(np.round(CH, 2)[::4])
                    ax.set_xlabel('T')
                    ax.set_ylabel('H')
                    fig.colorbar(im, cax=cax, orientation='horizontal', ticks=np.linspace(dat.min(), dat.max(), 3))
                    fig.savefig(OUTPREF+'.ae.diag.ld.%d.%d.%d.png' % (i, j, k))
                    plt.close()
        # plot pca latent variable diagrams
        for i in range(2):
            for j in range(ED):
                for k in range(LD):
                    fig, ax = plt.subplots()
                    div = make_axes_locatable(ax)
                    cax = div.append_axes('top', size='5%', pad=0.8)
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                    ax.xaxis.set_ticks_position('bottom')
                    ax.yaxis.set_ticks_position('left')
                    if i == 0:
                        dat = PZMDIAG[:, :, j, k]
                    if i == 1:
                        dat = PZVDIAG[:, :, j, k]
                    im = ax.imshow(dat, aspect='equal', interpolation='none', origin='lower', cmap=CM)
                    ax.grid(which='minor', axis='both', linestyle='-', color='k', linewidth=1)
                    ax.set_xticks(np.arange(CT.size), minor=True)
                    ax.set_yticks(np.arange(CH.size), minor=True)
                    ax.set_xticks(np.arange(CT.size)[::4], minor=False)
                    ax.set_yticks(np.arange(CH.size)[::4], minor=False)
                    ax.set_xticklabels(np.round(CT, 2)[::4], rotation=-60)
                    ax.set_yticklabels(np.round(CH, 2)[::4])
                    ax.set_xlabel('T')
                    ax.set_ylabel('H')
                    fig.colorbar(im, cax=cax, orientation='horizontal', ticks=np.linspace(dat.min(), dat.max(), 3))
                    fig.savefig(OUTPREF+'.ae.diag.ld.pca.%d.%d.%d.png' % (i, j, k))
                    plt.close()
        # plot physical measurement diagrams
        for i in range(2):
            for j in range(2):
                fig, ax = plt.subplots()
                div = make_axes_locatable(ax)
                cax = div.append_axes('top', size='5%', pad=0.8)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.xaxis.set_ticks_position('bottom')
                ax.yaxis.set_ticks_position('left')
                if i == 0:
                    dat = MEMDIAG[:, :, j]
                if i == 1:
                    dat = MEVDIAG[:, :, j]
                im = ax.imshow(dat, aspect='equal', interpolation='none', origin='lower', cmap=CM)
                ax.grid(which='minor', axis='both', linestyle='-', color='k', linewidth=1)
                ax.set_xticks(np.arange(CT.size), minor=True)
                ax.set_yticks(np.arange(CH.size), minor=True)
                ax.set_xticks(np.arange(CT.size)[::4], minor=False)
                ax.set_yticks(np.arange(CH.size)[::4], minor=False)
                ax.set_xticklabels(np.round(CT, 2)[::4], rotation=-60)
                ax.set_yticklabels(np.round(CH, 2)[::4])
                ax.set_xlabel('T')
                ax.set_ylabel('H')
                fig.colorbar(im, cax=cax, orientation='horizontal', ticks=np.linspace(dat.min(), dat.max(), 3))
                fig.savefig(OUTPREF+'.ae.diag.mv.%d.%d.png' % (i, j))
                plt.close()

    if PLOT:
        # plot results
        ae_plots()
