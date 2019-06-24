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
                        type=int, default=8)
    parser.add_argument('-si', '--super_interval', help='interval for selecting phase points (variational autoencoder)',
                        type=int, default=1)
    parser.add_argument('-sn', '--super_samples', help='number of samples per phase point (variational autoencoder)',
                        type=int, default=1024)
    parser.add_argument('-sc', '--scaler', help='feature scaling method (none, global, minmax, robust, standard, tanh)',
                        type=str, default='global')
    parser.add_argument('-bk', '--backend', help='keras backend',
                        type=str, default='tensorflow')
    parser.add_argument('-pr', '--prelu', help='PReLU activation function (replaces ReLU and linear activations)',
                        action='store_true')
    parser.add_argument('-ld', '--latent_dimension', help='latent dimension of the variational autoencoder',
                        type=int, default=8)
    parser.add_argument('-opt', '--optimizer', help='neural network weight optimization function',
                        type=str, default='nadam')
    parser.add_argument('-lr', '--learning_rate', help='learning rate for neural networ optimizer',
                        type=float, default=1e-3)
    parser.add_argument('-lss', '--loss', help='loss function',
                        type=str, default='mse')
    parser.add_argument('-ep', '--epochs', help='number of epochs',
                        type=int, default=32)
    parser.add_argument('-bs', '--batch_size', help='size of batches',
                        type=int, default=32)
    parser.add_argument('-sd', '--random_seed', help='random seed for sample selection and learning',
                        type=int, default=256)
    args = parser.parse_args()
    return (args.verbose, args.plot, args.parallel, args.gpu, args.threads, args.name,
            args.lattice_size, args.super_interval, args.super_samples,
            args.scaler, args.backend, args.prelu, args.latent_dimension,
            args.optimizer, args.learning_rate, args.loss, args.epochs, args.batch_size, args.random_seed)


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
        print('backend:                   %s' % BACKEND)
        print('prelu:                     %d' % PRELU)
        print('latent dimension:          %d' % LD)
        print('optimizer:                 %s' % OPT)
        print('learning rate:             %.2e' % LR)
        print('loss function:             %s' % LSS)
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
        out.write('backend:                   %s\n' % BACKEND)
        out.write('prelu:                     %d\n' % PRELU)
        out.write('latent dimension:          %d\n' % LD)
        out.write('optimizer:                 %s\n' % OPT)
        out.write('learning rate:             %.2e\n' % LR)
        out.write('loss function:             %s\n' % LSS)
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


def gauss_sampling(beta):
    ''' samples a point in a multivariate gaussian distribution '''
    z_mean, z_log_var = beta
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean+K.exp(0.5*z_log_var)*epsilon


def build_variational_autoencoder():
    if VERBOSE:
        print('building variational autoencoder network')
        print(100*'-')
    # --------------
    # initialization
    # --------------
    # output layer activation
    # sigmoid for activations on (0, 1) tanh otherwise (-1, 1)
    # caused by scaling
    if SCLR in ['minmax', 'tanh', 'global']:
        hidact = 'relu'
        outact = 'sigmoid'
    else:
        hidact = 'linear'
        outact = 'tanh'
    # kernel initializer - customizable
    # limited tests showed he_normal performs well
    init = 'he_normal'
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
        c = BatchNormalization(epsilon=1e-4)(c)
        # activations
        if PRELU:
            # like leaky relu, but with trainable layer to tune alphas
            c = PReLU()(c)
        else:
            c = Activation(hidact)(c)
    # flatten convolutional output
    shape = K.int_shape(c)
    d0 = Flatten()(c)
    # dense layer connected to flattened convolution output
    # d0 = Dense(np.int32(np.sqrt(np.prod(shape[1:]))), kernel_initializer=init)(d0)
    # d0 = PReLU()(d0)
    # gaussian parameters as dense layers
    z_mean = Dense(LD, name='z_mean', kernel_initializer=init)(d0)
    # activations
    if PRELU:
        # like leaky relu, but with trainable layer to tune alphas
        z_mean = PReLU()(z_mean)
    else:
        z_mean = Activation('linear')(z_mean)
    # more numerically stable to use log(var_z)
    z_log_var = Dense(LD, name='z_log_std', kernel_initializer=init)(d0)
    # activations
    if PRELU:
        # like leaky relu, but with trainable layer to tune alphas
        z_log_var = PReLU()(z_log_var)
    else:
        z_log_var = Activation('linear')(z_log_var)
    # samples from the gaussians
    z = Lambda(gauss_sampling, output_shape=(LD,), name='z')([z_mean, z_log_var])
    # construct encoder
    encoder = Model(input, [z_mean, z_log_var, z], name='encoder')
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
    d1 = BatchNormalization(epsilon=1e-4)(d1)
    # activations
    if PRELU:
        # like leaky relu, but with trainable layer to tune alphas
        d1 = PReLU()(d1)
    else:
        d1 = Activation(hidact)(d1)
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
        ct = BatchNormalization(epsilon=1e-4)(ct)
        # activations
        if PRELU:
            # like leaky relu, but with trainable layer to tune alphas
            ct = PReLU()(ct)
        else:
            ct = Activation(hidact)(ct)
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
    # -------------
    # construct vae
    # -------------
    # combine encoder and decoder
    output = decoder(encoder(input)[2])
    vae = Model(input, output, name='vae_mlp')
    # vae loss
    # scale by number of features in sample
    reconstruction = N*N*RCLS[LSS](K.flatten(input), K.flatten(output))
    # kullback-liebler divergence for gaussian distribution to regularize latent space
    kl = 0.5*K.sum(K.exp(z_log_var)+K.square(z_mean)-z_log_var-1, axis=-1)
    # combine losses
    vae_loss = K.mean(reconstruction+kl)
    vae.add_loss(vae_loss)
    # compile vae
    vae.compile(optimizer=OPTS[OPT])
    # return vae networks
    return encoder, decoder, vae


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
     N, SNI, SNS,
     SCLR, BACKEND, PRELU, LD,
     OPT, LR, LSS, EP, BS, SEED) = parse_args()
    CWD = os.getcwd()
    EPS = 0.025
    # number of phases
    NPH = 3
    # number of embedding dimensions
    ED = 2

    np.random.seed(SEED)
    # environment variables
    os.environ['KERAS_BACKEND'] = BACKEND
    if BACKEND == 'tensorflow':
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
    from keras.models import Model
    from keras.layers import (Input, Lambda, Dense, Conv2D, Conv2DTranspose,
                              Flatten, Reshape, BatchNormalization, Activation)
    from keras.losses import binary_crossentropy, mean_squared_error, mean_absolute_error, logcosh
    from keras.activations import relu, sigmoid, linear
    from keras.optimizers import SGD, Adadelta, Adam, Nadam
    from keras.activations import relu, tanh, sigmoid, linear
    from keras.layers.advanced_activations import LeakyReLU, PReLU
    from keras.callbacks import History, CSVLogger, ReduceLROnPlateau
    from keras import backend as K
    if PLOT:
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import ImageGrid
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
    PRM = (NAME, N, SNI, SNS, SCLR, PRELU, LD, OPT, LR, LSS, EP, BS, SEED)
    # output file prefix
    OUTPREF = CWD+'/%s.%d.%d.%d.%s.%d.%d.%s.%.0e.%s.%d.%d.%d' % PRM
    # write output file header
    write_specs()

    # feature range
    FRNG = (0.0, 1.0)
    # scaler dictionary
    SCLRS = {'minmax':MinMaxScaler(feature_range=FRNG),
             'standard':StandardScaler(),
             'robust':RobustScaler(),
             'tanh':TanhScaler(feature_range=FRNG)}

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
    SCPREF = CWD+'/%s.%d.%d.%d.%s.%d' % (NAME, N, SNI, SNS, SCLR, SEED)

    try:
        # check is scaled data has already been computed
        SCDMP = np.load(SCPREF+'.dmp.sc.npy').reshape(*SHP1)
        CDAT = np.load(SCPREF+'.dmp.sc.npy')
        if VERBOSE:
            print('scaled selected classification samples loaded from file')
            print(100*'-')
    except:
        try:
            # check if random data subset has already been selected
            CDMP = np.load(SCPREF+'.dmp.c.npy')
            CDAT = np.load(SCPREF+'.dat.c.npy')
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
            np.save(SCPREF+'.dmp.c.npy', CDMP)
            np.save(SCPREF+'.dat.c.npy', CDAT)
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
    SP = np.var(ES/CT[np.newaxis, :, np.newaxis], 2)
    # mean magnetizations
    MM = np.mean(MS, -1)
    # magnetic susceptibilities
    SU = np.var(MS/CT[np.newaxis, :, np.newaxis], 2)

    # reconstruction losses
    RCLS = {'bc': lambda a, b: binary_crossentropy(a, b),
            'mse': lambda a, b: mean_squared_error(a, b),
            'mea': lambda a, b: mean_absolute_error(a, b),
            'logcosh': lambda a, b: logcosh(a, b)}

    # optmizers
    OPTS = {'sgd': SGD(lr=LR, momentum=0.0, decay=0.0, nesterov=True),
            'adadelta': Adadelta(lr=LR, rho=0.95, epsilon=None, decay=0.0),
            'adam': Adam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True),
            'nadam': Nadam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)}

    # autencoder networks
    ENC, DEC, VAE = build_variational_autoencoder()

    try:
        # check if model already trained
        VAE.load_weights(OUTPREF+'.vae.wt.h5', by_name=True)
        TLOSS = np.load(OUTPREF+'.vae.loss.trn.npy')
        VLOSS = np.load(OUTPREF+'.vae.loss.val.npy')
        if VERBOSE:
            print('variational autoencoder trained weights loaded from file')
            print(100*'-')
    except:
        if VERBOSE:
            print('variational autoencoder training on scaled selected classification samples')
            print(100*'-')
        # output log
        CSVLG = CSVLogger(OUTPREF+'.vae.log.csv', append=True, separator=',')
        # learning rate decay on loss plateau
        LR_DECAY = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, verbose=VERBOSE)
        # split data into training and validation
        TRN, VAL = train_test_split(SCDMP, test_size=0.125, shuffle=True)
        # fit model
        VAE.fit(x=TRN, y=None, validation_data=(VAL, None), epochs=EP, batch_size=BS,
                shuffle=True, verbose=VERBOSE, callbacks=[CSVLG, LR_DECAY, History()])
        # remove split data
        del TRN, VAL
        TLOSS = VAE.history.history['loss']
        VLOSS = VAE.history.history['val_loss']
        VAE.save_weights(OUTPREF+'.vae.wt.h5')
        np.save(OUTPREF+'.vae.loss.trn.npy', TLOSS)
        np.save(OUTPREF+'.vae.loss.val.npy', VLOSS)
        if VERBOSE:
            print(100*'-')
            print('variational autoencoder weights trained')
            print(100*'-')

    if VERBOSE:
        print('variational autoencoder training history information')
        print(100*'-')
        print('| epoch | training loss | validation loss |')
        print(100*'-')
        for i in range(EP):
            print('%02d %.2f %.2f' % (i, TLOSS[i], VLOSS[i]))
        print(100*'-')

    with open(OUTPREF+'.out', 'a') as out:
        out.write('variational autoencoder training history information\n')
        out.write(100*'-' + '\n')
        out.write('| epoch | training loss | validation loss |\n')
        out.write(100*'-' + '\n')
        for i in range(EP):
            out.write('%02d %.2f %.2f\n' % (i, TLOSS[i], VLOSS[i]))
        out.write(100*'-' + '\n')

    try:
        ZENC = np.load(OUTPREF+'.zenc.npy').reshape(*SHP4)
        ERR = np.load(OUTPREF+'.zerr.npy')
        MERR = np.load(OUTPREF+'.zerr.mean.npy')
        SERR = np.load(OUTPREF+'.zerr.stdv.npy')
        MXERR = np.load(OUTPREF+'.zerr.max.npy')
        MNERR = np.load(OUTPREF+'.zerr.min.npy')
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
        ZENC = np.array(ENC.predict(SCDMP, verbose=VERBOSE))
        ZDEC = np.array(DEC.predict(ZENC[2, :, :], verbose=VERBOSE))
        # swap latent space axes
        ZENC = np.swapaxes(ZENC, 0, 1)[:, :2, :]
        # convert log variance to standard deviation
        ZENC[:, 1, :] = np.exp(0.5*ZENC[:, 1, :])
        # reconstruction error (signed)
        ERR = SCDMP-ZDEC
        # kullback-liebler divergence
        KLD = 0.5*np.sum(np.square(ZENC[:, 1, :])+np.square(ZENC[:, 0, :])-np.log(np.square(ZENC[:, 1, :]))-1, axis=1)
        # dump results
        np.save(OUTPREF+'.zenc.npy', ZENC.reshape(*SHP3))
        np.save(OUTPREF+'.zdec.npy', ZDEC.reshape(*SHP0))
        np.save(OUTPREF+'.zerr.npy', ERR.reshape(*SHP0))
        np.save(OUTPREF+'.zerr.kld.npy', KLD.reshape(SNH, SNT, SNS))
        # means and standard deviation of error
        MERR = np.mean(ERR)
        SERR = np.std(ERR)
        # minimum and maximum error
        MXERR = np.max(ERR)
        MNERR = np.min(ERR)
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
        print('mean kl div:     %f' % MKLD)
        print('stdv kl div:     %f' % SKLD)
        print('max kl div:      %f' % MXKLD)
        print('min kl div:      %f' % MNKLD)
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
        out.write('mean kl div:     %f\n' % MKLD)
        out.write('stdv kl div:     %f\n' % SKLD)
        out.write('max kl div:      %f\n' % MXKLD)
        out.write('min kl div:      %f\n' % MNKLD)
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
                print(LD*'%f ' % tuple(CZENC[i, j, :]))
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
                out.write(LD*'%f ' % tuple(CZENC[i, j, :]) + '\n')
            out.write(100*'-'+'\n')
            out.write('explained variances\n')
            out.write(100*'-'+'\n')
            out.write(LD*'%f ' % tuple(VZENC[i, :]) + '\n')
            out.write(100*'-'+'\n')

    def vae_plots():

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
        ax.grid(which='minor', axis='both', linestyle='-', color='k', linewidth=1)
        ax.set_xticks(np.linspace(np.floor(ERR.min()), np.ceil(ERR.max()), 5), minor=True)
        ax.set_yticks(ey, minor=True)
        plt.xticks(np.linspace(np.floor(ERR.min()), np.ceil(ERR.max()), 5))
        plt.yticks(ey)
        plt.xlabel('ERROR')
        plt.ylabel('DENSITY')
        fig.savefig(OUTPREF+'.vae.err.png')
        plt.close()

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
        ax.grid(which='minor', axis='both', linestyle='-', color='k', linewidth=1)
        ax.set_xticks(np.linspace(0.0, np.ceil(KLD.max()), 5), minor=True)
        ax.set_yticks(ey, minor=True)
        plt.xticks(np.linspace(0.0, np.ceil(KLD.max()), 5))
        plt.yticks(ey)
        plt.xlabel('KLD')
        plt.ylabel('DENSITY')
        fig.savefig(OUTPREF+'.vae.kld.png')
        plt.close()

        # diagrams for physical measurements
        MEMDIAG = SCLRS['minmax'].fit_transform(np.stack((MM, EM), axis=-1).reshape(SNH*SNT, 2)).reshape(SNH, SNT, 2)
        MEVDIAG = SCLRS['minmax'].fit_transform(np.stack((SU, SP), axis=-1).reshape(SNH*SNT, 2)).reshape(SNH, SNT, 2)
        # diagrams for latent variables
        ZMDIAG = SCLRS['minmax'].fit_transform(np.mean(ZENC.reshape(SNH, SNT, SNS, ED*LD), 2).reshape(SNH*SNT, ED*LD)).reshape(SNH, SNT, ED, LD)
        ZVDIAG = SCLRS['minmax'].fit_transform(np.var(ZENC.reshape(SNH, SNT, SNS, ED*LD)/\
                                                      CT[np.newaxis, :, np.newaxis, np.newaxis], 2).reshape(SNH*SNT, ED*LD)).reshape(SNH, SNT, ED, LD)
        # diagrams for pca embeddings of latent variables
        PZMDIAG = SCLRS['minmax'].fit_transform(np.mean(PZENC.reshape(SNH, SNT, SNS, ED*LD), 2).reshape(SNH*SNT, ED*LD)).reshape(SNH, SNT, ED, LD)
        for i in range(LD):
            if PZMDIAG[0, 0, 0, i] > PZMDIAG[-1, 0, 0, i]:
                PZMDIAG[:, :, 0, i] = 1-PZMDIAG[:, :, 0, i]
            if PZMDIAG[0, 0, 1, i] > PZMDIAG[0, -1, 1, i]:
                PZMDIAG[:, :, 1, i] = 1-PZMDIAG[:, :, 1, i]
        PZVDIAG = SCLRS['minmax'].fit_transform(np.var(PZENC.reshape(SNH, SNT, SNS, ED*LD)/\
                                                       CT[np.newaxis, :, np.newaxis, np.newaxis], 2).reshape(SNH*SNT, ED*LD)).reshape(SNH, SNT, ED, LD)

        # plot latent variable diagrams
        for i in range(1):
            for j in range(ED):
                for k in range(LD):
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                    ax.xaxis.set_ticks_position('bottom')
                    ax.yaxis.set_ticks_position('left')
                    if i == 0:
                        ax.imshow(ZMDIAG[:, :, j, k], aspect='equal', interpolation='none', origin='lower', cmap=CM)
                    if i == 1:
                        ax.imshow(ZVDIAG[:, :, j, k], aspect='equal', interpolation='none', origin='lower', cmap=CM)
                    ax.grid(which='minor', axis='both', linestyle='-', color='k', linewidth=1)
                    ax.set_xticks(np.arange(CT.size), minor=True)
                    ax.set_yticks(np.arange(CH.size), minor=True)
                    plt.xticks(np.arange(CT.size)[::4], np.round(CT, 2)[::4], rotation=-60)
                    plt.yticks(np.arange(CH.size)[::4], np.round(CH, 2)[::4])
                    plt.xlabel('T')
                    plt.ylabel('H')
                    fig.savefig(OUTPREF+'.vae.diag.ld.%d.%d.%d.png' % (i, j, k))
                    plt.close()

        # plot pca latent variable diagrams
        for i in range(1):
            for j in range(ED):
                for k in range(LD):
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                    ax.xaxis.set_ticks_position('bottom')
                    ax.yaxis.set_ticks_position('left')
                    if i == 0:
                        ax.imshow(PZMDIAG[:, :, j, k], aspect='equal', interpolation='none', origin='lower', cmap=CM)
                    if i == 1:
                        ax.imshow(PZVDIAG[:, :, j, k], aspect='equal', interpolation='none', origin='lower', cmap=CM)
                    ax.grid(which='minor', axis='both', linestyle='-', color='k', linewidth=1)
                    ax.set_xticks(np.arange(CT.size), minor=True)
                    ax.set_yticks(np.arange(CH.size), minor=True)
                    plt.xticks(np.arange(CT.size)[::4], np.round(CT, 2)[::4], rotation=-60)
                    plt.yticks(np.arange(CH.size)[::4], np.round(CH, 2)[::4])
                    plt.xlabel('T')
                    plt.ylabel('H')
                    fig.savefig(OUTPREF+'.vae.diag.ld.pca.%d.%d.%d.png' % (i, j, k))
                    plt.close()

        for i in range(2):
            for j in range(ED):
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.xaxis.set_ticks_position('bottom')
                ax.yaxis.set_ticks_position('left')
                if i == 0:
                    ax.imshow(MEMDIAG[:, :, j], aspect='equal', interpolation='none', origin='lower', cmap=CM)
                if i == 1:
                    ax.imshow(MEVDIAG[:, :, j], aspect='equal', interpolation='none', origin='lower', cmap=CM)
                ax.grid(which='minor', axis='both', linestyle='-', color='k', linewidth=1)
                ax.set_xticks(np.arange(CT.size), minor=True)
                ax.set_yticks(np.arange(CH.size), minor=True)
                plt.xticks(np.arange(CT.size)[::4], np.round(CT, 2)[::4], rotation=-60)
                plt.yticks(np.arange(CH.size)[::4], np.round(CH, 2)[::4])
                plt.xlabel('T')
                plt.ylabel('H')
                fig.savefig(OUTPREF+'.vae.diag.mv.%d.%d.png' % (i, j))
                plt.close()

    if PLOT:
        # plot results
        vae_plots()