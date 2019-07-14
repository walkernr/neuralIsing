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
                        type=str, default='he_normal')
    parser.add_argument('-pr', '--prelu', help='PReLU activation function (replaces leaky ReLU and linear activations)',
                        action='store_true')
    parser.add_argument('-bn', '--batch_normalization', help='batch normalization layers',
                        action='store_true')
    parser.add_argument('-ld', '--latent_dimension', help='latent dimension of the variational autoencoder',
                        type=int, default=4)
    parser.add_argument('-opt', '--optimizer', help='neural network weight optimization function',
                        type=str, default='nadam')
    parser.add_argument('-lr', '--learning_rate', help='learning rate for neural networ optimizer',
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
    parser.add_argument('-bs', '--batch_size', help='size of batches',
                        type=int, default=32)
    parser.add_argument('-sd', '--random_seed', help='random seed for sample selection and learning',
                        type=int, default=512)
    args = parser.parse_args()
    return (args.verbose, args.plot, args.parallel, args.gpu, args.threads, args.name,
            args.lattice_size, args.super_interval, args.super_samples, args.scaler,
            args.prior_distribution, args.kernel_initializer, args.prelu,
            args.batch_normalization, args.latent_dimension, args.optimizer, args.learning_rate,
            args.loss, args.regularizer, args.alpha, args.beta, args.lmbda, args.minibatch_stratified_sampling,
            args.epochs, args.batch_size, args.random_seed)


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
        print('prelu:                     %d' % PRELU)
        print('batch normalization:       %d' % BN)
        print('latent dimension:          %d' % LD)
        print('optimizer:                 %s' % OPT)
        print('learning rate:             %.2e' % LR)
        print('loss function:             %s' % LSS)
        print('regularizer:               %s' % REG)
        print('alpha:                     %.2e' % ALPHA)
        print('beta:                      %.2e' % BETA)
        print('lambda:                    %.2e' % LMBDA)
        print('mss:                       %d' % MSS)
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
        out.write('prelu:                     %d\n' % PRELU)
        out.write('batch_normalization:       %d\n' % BN)
        out.write('latent dimension:          %d\n' % LD)
        out.write('optimizer:                 %s\n' % OPT)
        out.write('learning rate:             %.2e\n' % LR)
        out.write('loss function:             %s\n' % LSS)
        out.write('regularizer:               %s\n' % REG)
        out.write('alpha:                     %.2e\n' % ALPHA)
        out.write('beta:                      %.2e\n' % BETA)
        out.write('lambda:                    %.2e\n' % LMBDA)
        out.write('mss:                       %d\n' % MSS)
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


def gauss_sampling(beta, batch_size=None):
    ''' samples a point in a multivariate gaussian distribution '''
    if batch_size is None:
        batch_size = BS
    z_mean, z_log_var = beta
    epsilon = K.random_normal(shape=(batch_size, LD))
    return z_mean+K.exp(0.5*z_log_var)*epsilon


def gauss_log_prob(z, beta=None, batch_size=None):
    if batch_size is None:
        batch_size = BS
    if beta is None:
        # mu = 0, stdv = 1 => log(var) = 0
        z_mean, z_log_var = K.zeros((batch_size, LD)), K.zeros((batch_size, LD))
    else:
        z_mean, z_log_var = beta
    norm = K.log(2*np.pi)
    zsc = (z-z_mean)*K.exp(-0.5*z_log_var)
    return -0.5*(zsc**2+z_log_var+norm)


def bernoulli_sampling(p, batch_size=None):
    if batch_size is None:
        batch_size = BS
    epsilon = K.random_uniform(shape=(batch_size, N, N, NCH))
    return K.sigmoid(2*(p-epsilon)/EPS)


def bernoulli_crossentropy(x, y, batch_size=None):
    if batch_size is None:
        batch_size = BS
    return binary_crossentropy(K.flatten(x), K.flatten(bernoulli_sampling(y, batch_size=batch_size)))


def kld(beta):
    z_mean, z_log_var = beta
    return 0.5*K.sum(K.exp(z_log_var)+K.square(z_mean)-z_log_var-1, axis=-1)


def kernel_computation(x, y, batch_size=None):
    if batch_size is None:
        batch_size = BS
    tiled_x = K.tile(K.reshape(x, (batch_size, 1, LD)), (1, batch_size, 1))
    tiled_y = K.tile(K.reshape(y, (1, batch_size, LD)), (batch_size, 1, 1))
    return K.exp(-K.mean(K.square(tiled_x-tiled_y), axis=2)/K.cast(LD, 'float32'))


def mmd(x, y, batch_size=None):
    if batch_size is None:
        batch_size = BS
    x_kernel = kernel_computation(x, x, batch_size)
    y_kernel = kernel_computation(y, y, batch_size)
    xy_kernel = kernel_computation(x, y, batch_size)
    return x_kernel+y_kernel-2*xy_kernel


def sw(x, batch_size=None):
    if batch_size is None:
        batch_size = BS
    nrp = batch_size**2
    theta = K.random_normal(shape=(nrp, LD))
    theta = theta/K.sqrt(K.sum(K.square(theta), axis=1, keepdims=True))
    y = K.random_uniform(shape=(batch_size, LD), minval=-0.5, maxval=0.5)
    px = K.dot(x, K.transpose(theta))
    py = K.dot(y, K.transpose(theta))
    w2 = (tf.nn.top_k(tf.transpose(px), k=batch_size).values-
          tf.nn.top_k(tf.transpose(py), k=batch_size).values)**2
    return w2


def log_sum_exp(x):
    m = K.max(x, axis=1, keepdims=True)
    u = x-m
    m = K.squeeze(m, 1)
    return m+K.log(K.sum(K.exp(u), axis=1, keepdims=False))


def log_importance_weight(batch_size=None, dataset_size=None):
    if batch_size is None:
        batch_size = BS
    if dataset_size is None:
        dataset_size = SNH*SNT*SNS
    n, m = dataset_size, batch_size-1
    strw = (n-m)/(n*m)
    w = K.concatenate((1/n*K.ones((batch_size, 1)),
                       strw*K.ones((batch_size, 1)),
                       1/m*K.ones((batch_size, batch_size-2))), axis=1)
    return K.log(w)
    # w = np.ones((batch_size,batch_size))/m
    # w.reshape(-1)[::(m+1)] = 1/n
    # w.reshape(-1)[1::(m+1)] = strw
    # w[m-1, 0] = strw
    # return K.log(K.cast(w, 'float32'))


def tc(z, beta, batch_size=None):
    if batch_size is None:
        batch_size = BS
    z_mean, z_log_var = beta
    # log p(z)
    logpz = K.sum(K.reshape(gauss_log_prob(z), (batch_size, -1)), 1)
    # log q(z|x)
    logqz_x = K.sum(K.reshape(gauss_log_prob(z, (z_mean, z_log_var)), (batch_size, -1)), 1)
    # log q(z) ~ log (1/MN) sum_m q(z|x_m) = -log(MN)+log(sum_m(exp(q(z|x_m))))
    _logqz = gauss_log_prob(K.reshape(z, (batch_size, 1, LD)),
                            (K.reshape(z_mean, (1, batch_size, LD)),
                             K.reshape(z_log_var, (1, batch_size, LD))))
    if MSS:
        log_iw = log_importance_weight()
        logqz_prodmarginals = K.sum(log_sum_exp(K.reshape(log_iw, (batch_size, batch_size, 1))+_logqz), 1)
        logqz = log_sum_exp(log_iw+K.sum(_logqz, axis=2))
    else:
        logqz_prodmarginals = K.sum(log_sum_exp(_logqz)-K.log(K.cast(BS*SNH*SNT*SNS, 'float32')), 1)
        logqz = log_sum_exp(K.sum(_logqz, axis=2))-K.log(K.cast(BS*SNH*SNT*SNS, 'float32'))
    # alpha controls mutual information
    # beta controls total correlation
    # lambda controls dimension-wise kld
    melbo = ALPHA*(logqz_x-logqz)+BETA*(logqz-logqz_prodmarginals)+LMBDA*(logqz_prodmarginals-logpz)
    return melbo


def build_autoencoder():
    if VERBOSE:
        print('building variational autoencoder network')
        print(100*'-')
    # --------------
    # initialization
    # --------------
    # output layer activation
    # sigmoid for activations on (0, 1) tanh otherwise (-1, 1)
    # caused by scaling
    if not PRELU:
        alpha_enc = 0.2
        alpha_dec = 0.0
        if PRIOR == 'gaussian':
            alpha_zm = 1.0
            alpha_zlg = 1.0
        elif PRIOR == 'none':
            alpha_z = 1.0
    if SCLR in ['minmax', 'tanh', 'global']:
        outact = 'sigmoid'
    else:
        outact = 'tanh'
    # kernel initializer - customizable
    # limited tests showed he_normal performs well
    init = KIS[KI]
    # base number of filters
    nf = 32
    # number of convolutions necessary to get down to size length 4
    # should use base 2 exponential side lengths
    nc = np.int32(np.log2(N/4.))
    # --------------
    # encoder layers
    # --------------
    # input layer
    input = Input(shape=(N, N, NCH), name='encoder_input')
    # loop through convolutions
    # filter size of (3, 3) to capture nearest neighbors from input
    for i in range(nc):
        if i == 0:
            # 0th convoluton takes input layer
            c = Conv2D(filters=2**i*nf, kernel_size=(3, 3), kernel_initializer=init,
                       padding='same', strides=2)(input)
        else:
            # (i!=0)th convolution takes prior convolution
            c = Conv2D(filters=2**i*nf, kernel_size=(3, 3), kernel_initializer=init,
                       padding='same', strides=2)(c)
        # batch normalization to scale activations
        if BN:
            c = BatchNormalization(epsilon=1e-4)(c)
        # activations
        if PRELU:
            # like leaky relu, but with trainable layer to tune alphas
            c = PReLU(alpha_initializer=init)(c)
        else:
            c = LeakyReLU(alpha=alpha_enc)(c)
    # flatten convolutional output
    shape = K.int_shape(c)
    d0 = Flatten()(c)
    # dense layer connected to flattened convolution output
    d0 = Dense(nc*32, kernel_initializer=init)(d0)
    if BN:
        d0 = BatchNormalization(epsilon=1e-4)(d0)
    if PRELU:
        # like leaky relu, but with trainable layer to tune alphas
        d0 = PReLU(alpha_initializer=init)(d0)
    else:
        d0 = LeakyReLU(alpha=alpha_enc)(d0)
    if PRIOR == 'gaussian':
        # gaussian parameters as dense layers
        z_mean = Dense(LD, name='z_mean', kernel_initializer=init)(d0)
        # activations
        # z_mean = Activation('tanh')(z_mean)
        if PRELU:
            # like leaky relu, but with trainable layer to tune alphas
            z_mean = PReLU(alpha_initializer=init)(z_mean)
        else:
            z_mean = LeakyReLU(alpha=alpha_zm)(z_mean)
        # more numerically stable to use log(var_z)
        z_log_var = Dense(LD, name='z_log_std', kernel_initializer=init)(d0)
        # activations
        # z_log_var = Activation('tanh')(z_log_var)
        if PRELU:
            # like leaky relu, but with trainable layer to tune alphas
            z_log_var = PReLU(alpha_initializer=init)(z_log_var)
        else:
            z_log_var = LeakyReLU(alpha=alpha_zlg)(z_log_var)
        # samples from the gaussians
        z = Lambda(gauss_sampling, output_shape=(LD,), name='z')([z_mean, z_log_var])
        # construct encoder
        encoder = Model(input, [z_mean, z_log_var, z], name='encoder')
    elif PRIOR == 'none':
        # encoding
        z = Dense(LD, name='z_encoding', kernel_initializer=init)(d0)
        # z = Activation('tanh')(z)
        if PRELU:
            # like leaky relu, but with trainable layer to tune alphas
            z = PReLU(alpha_initializer=init)(z)
        else:
            z = LeakyReLU(alpha=alpha_z)(z)
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
    latent_input = Input(shape=(LD,), name='z_sampling')
    # dense network of same size as convolution output from encoder
    d1 = Dense(np.prod(shape[1:]), kernel_initializer=init)(latent_input)
    # reshape to convolution shape
    d1 = Reshape(shape[1:])(d1)
    # batch renormalization to scale activations
    if BN:
        d1 = BatchNormalization(epsilon=1e-4)(d1)
    # activations
    if PRELU:
        # like leaky relu, but with trainable layer to tune alphas
        d1 = PReLU(alpha_initializer=init)(d1)
    else:
        d1 = LeakyReLU(alpha=alpha_dec)(d1)
    # loop through convolution transposes
    for i in range(nc-1, -1, -1):
        if i == nc-1:
            # (nc-1)th convoltution transpose takes reshaped dense layer
            ct = Conv2DTranspose(filters=2**i*nf, kernel_size=3, kernel_initializer=init,
                                 padding='same', strides=2)(d1)
        else:
            # (i!=(nc-1))th convolution transpose takes prior convolution transpose
            ct = Conv2DTranspose(filters=2**i*nf, kernel_size=3, kernel_initializer=init,
                                 padding='same', strides=2)(ct)
        # batch normalization to scale activations
        if BN:
            ct = BatchNormalization(epsilon=1e-4)(ct)
        # activations
        if PRELU:
            # like leaky relu, but with trainable layer to tune alphas
            ct = PReLU(alpha_initializer=init)(ct)
        else:
            ct = LeakyReLU(alpha=alpha_dec)(ct)
    # output convolution transpose layer
    output = Conv2DTranspose(filters=NCH, kernel_size=3, kernel_initializer=init,
                             padding='same', name='decoder_output')(ct)
    # output layer activation
    output = Activation(outact)(output)
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
    ae = Model(input, output, name='ae_mlp')
    # ae loss
    if PRIOR == 'gaussian':
        # distribution divergence to regularize latent space
        if REG == 'kld':
            rcsc = N*N
            reconstruction = rcsc*RCLS[LSS](K.flatten(input), K.flatten(output))
            dd = BETA*K.mean(REGS[REG]((z_mean, z_log_var)))
        elif REG == 'mmd':
            rcsc = N*N
            reconstruction = rcsc*RCLS[LSS](K.flatten(input), K.flatten(output))
            dd = BETA*K.mean(REGS['kld']((z_mean, z_log_var)))+\
                 LMBDA*K.mean(REGS['mmd'](K.random_normal(shape=(BS, LD)), z))
        elif REG == 'sw':
            rcsc = N*N
            reconstruction = rcsc*RCLS[LSS](K.flatten(input), K.flatten(output))
            dd = BETA*K.mean(REGS['kld']((z_mean, z_log_var)))+\
                 LMBDA*K.mean(REGS['sw'](z))
        elif REG == 'tc':
            rcsc = N*N
            reconstruction = rcsc*RCLS[LSS](K.flatten(input), K.flatten(output))
            dd = K.mean(tc(z, (z_mean, z_log_var)))
    elif PRIOR == 'none' or REG == 'none':
        dd = 0.0
    # combine losses
    ae_loss = K.mean(reconstruction+dd)
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
     PRIOR, KI, PRELU, BN, LD,
     OPT, LR, LSS, REG, ALPHA, BETA, LMBDA, MSS,
     EP, BS, SEED) = parse_args()
    CWD = os.getcwd()
    EPS = 1e-8
    # number of phases
    NPH = 3
    # number of embedding dimensions
    if PRIOR == 'gaussian':
        ED = 2
    elif PRIOR == 'none':
        ED = 1
    if REG != 'tc':
        ALPHA = 0
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
    import tensorflow as tf
    from keras.models import Model
    from keras.layers import (Input, Lambda, Dense, Conv2D, Conv2DTranspose,
                              Flatten, Reshape, BatchNormalization, Activation)
    from keras.losses import (mean_squared_error, mean_absolute_error, logcosh,
                              binary_crossentropy, kullback_leibler_divergence, poisson)
    from keras.optimizers import SGD, Adadelta, Adam, Nadam
    from keras.initializers import (Zeros, Ones, Constant, RandomNormal, RandomUniform,
                                    TruncatedNormal, VarianceScaling, glorot_uniform, glorot_normal,
                                    lecun_uniform, lecun_normal, he_uniform, he_normal)
    from keras.activations import relu, tanh, sigmoid, linear
    from keras.layers.advanced_activations import LeakyReLU, PReLU
    from keras.callbacks import History, CSVLogger, ReduceLROnPlateau
    from keras import backend as K
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

    # run parameter tuple
    PRM = (NAME, N, SNI, SNS, SCLR, PRIOR, PRELU, BN, LD, OPT, LR, LSS, REG, ALPHA, BETA, LMBDA, MSS, EP, BS, SEED)
    # output file prefix
    OUTPREF = CWD+'/%s.%d.%d.%d.%s.%s.%d.%d.%d.%s.%.0e.%s.%s.%.0e.%.0e.%.0e.%d.%d.%d.%d' % PRM
    # write output file header
    write_specs()

    # feature range
    FMN, FMX = (0.0, 1.0)
    # scaler dictionary
    SCLRS = {'minmax':MinMaxScaler(feature_range=(FMN, FMX)),
             'standard':StandardScaler(),
             'robust':RobustScaler(),
             'tanh':TanhScaler(feature_range=(FMN, FMX))}

    # external fields and temperatures
    CH = np.load(CWD+'/%s.%d.h.npy' % (NAME, N))[::SNI]
    CT = np.load(CWD+'/%s.%d.t.npy' % (NAME, N))[::SNI]
    # external field and temperature counts
    SNH, SNT = CH.size, CT.size
    # number of data channels
    NCH = 1

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

    # kernel
    SMN, SMX = 0.05*SCDMP.min(), 0.05*SCDMP.max()
    SM, SS = np.mean(SCDMP), 0.05*np.std(SCDMP)
    KIS = {'zeros': Zeros(),
           'ones': Ones(),
           'const': Constant(SMN+0.5*(SMX-SMN)),
           'uniform': RandomUniform(SMN, SMX, SEED),
           'glorot_uniform': glorot_uniform(SEED),
           'lecun_uniform': lecun_uniform(SEED),
           'he_uniform': he_uniform(SEED),
           'normal': RandomNormal(SM, SS, SEED),
           'truncated': TruncatedNormal(SM, SS, SEED),
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
    RCLS = {'mse': lambda a, b: mean_squared_error(a, b),
            'mae': lambda a, b: mean_absolute_error(a, b),
            'logcosh': lambda a, b: logcosh(a, b),
            'bc': lambda a, b: binary_crossentropy(a, b),
            'kld': lambda a, b: kullback_leibler_divergence(a, b),
            'poisson': lambda a, b: poisson(a, b),
            'bbc': lambda a, b: bernoulli_crossentropy(a, b)}

    # regularizers
    REGS = {'kld': lambda beta: kld(beta),
            'mmd': lambda a, b: mmd(a, b),
            'sw': lambda a: sw(a)}

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
        # LR_DECAY = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, verbose=VERBOSE)
        # split data into training and validation
        # TRN, VAL = train_test_split(SCDMP, test_size=0.125, shuffle=True)
        # # fit model
        # AE.fit(x=TRN, y=None, validation_data=(VAL, None), epochs=EP, batch_size=BS,
        #        shuffle=True, verbose=VERBOSE, callbacks=[CSVLG, LR_DECAY, History()])
        # # remove split data
        # del TRN, VAL
        LR_DECAY = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=8, verbose=VERBOSE)
        AE.fit(x=SCDMP, y=None, epochs=EP, batch_size=BS, shuffle=True,
               verbose=VERBOSE, callbacks=[CSVLG, LR_DECAY, History()])
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
        # means and standard deviation of error
        MERR = np.sqrt(np.mean(np.square(ERR)))
        SERR = np.std(ERR)
        # minimum and maximum error
        MXERR = np.max(ERR)
        MNERR = np.min(ERR)
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
        ex = np.linspace(np.floor(np.abs(ERR).min()), np.ceil(np.abs(ERR).max()), 33)
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
        ct = CT[np.newaxis, :, np.newaxis, np.newaxis]

        # diagrams for physical measurements
        # MEMDIAG = SCLRS['minmax'].fit_transform(np.stack((MM, EM), axis=-1).reshape(SNH*SNT, 2)).reshape(SNH, SNT, 2)
        # MEVDIAG = SCLRS['minmax'].fit_transform(np.stack((SU, SP), axis=-1).reshape(SNH*SNT, 2)).reshape(SNH, SNT, 2)
        # # diagrams for latent variables
        # ZMDIAG = SCLRS['minmax'].fit_transform(np.mean(ZENC.reshape(*shp0), 2).reshape(*shp2)).reshape(*shp1)
        # ZVDIAG = SCLRS['minmax'].fit_transform(np.var(np.divide(ZENC.reshape(*shp0), ct), 2).reshape(*shp2)).reshape(*shp1)
        # # diagrams for pca embeddings of latent variables
        # PZMDIAG = SCLRS['minmax'].fit_transform(np.mean(PZENC.reshape(*shp0), 2).reshape(*shp2)).reshape(*shp1)
        # for i in range(LD):
        #     if PZMDIAG[0, 0, 0, i] > PZMDIAG[-1, 0, 0, i]:
        #         PZMDIAG[:, :, 0, i] = 1-PZMDIAG[:, :, 0, i]
        #     if PRIOR == 'gaussian':
        #         if PZMDIAG[0, 0, 1, i] > PZMDIAG[0, -1, 1, i]:
        #             PZMDIAG[:, :, 1, i] = 1-PZMDIAG[:, :, 1, i]
        # PZVDIAG = SCLRS['minmax'].fit_transform(np.var(np.divide(PZENC.reshape(*shp0), ct), 2).reshape(*shp2)).reshape(*shp1)
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
                        dat = PZMDIAG[:, :, j, k]
                    dat = logistic((4/(dat.max()-dat.min()), dat.mean()), dat)
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
                    fig.savefig(OUTPREF+'.ae.diag.ld.pca.tanh.%d.%d.%d.png' % (i, j, k))
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