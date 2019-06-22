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
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', help='verbose output', action='store_true')
    parser.add_argument('-pt', '--plot', help='plot results', action='store_true')
    parser.add_argument('-p', '--parallel', help='parallel run', action='store_true')
    parser.add_argument('-g', '--gpu', help='gpu run (will default to cpu if unable)', action='store_true')
    parser.add_argument('-ad', '--anomaly_detection', help='anomaly detection for embedding', action='store_true')
    parser.add_argument('-nt', '--threads', help='number of threads',
                        type=int, default=20)
    parser.add_argument('-n', '--name', help='simulation name',
                        type=str, default='ising_init')
    parser.add_argument('-ls', '--lattice_size', help='lattice size (side length)',
                        type=int, default=8)
    parser.add_argument('-si', '--super_interval', help='interval for selecting phase points (variational autoencoder)',
                        type=int, default=1)
    parser.add_argument('-sn', '--super_samples', help='number of samples per phase point (variational autoencoder)',
                        type=int, default=1024)
    parser.add_argument('-sc', '--scaler', help='feature scaler',
                        type=str, default='global')
    parser.add_argument('-ld', '--latent_dimension', help='latent dimension of the variational autoencoder',
                        type=int, default=8)
    parser.add_argument('-bk', '--backend', help='keras backend',
                        type=str, default='tensorflow')
    parser.add_argument('-opt', '--optimizer', help='optimization function',
                        type=str, default='nadam')
    parser.add_argument('-lss', '--loss', help='loss function',
                        type=str, default='mse')
    parser.add_argument('-ep', '--epochs', help='number of epochs',
                        type=int, default=32)
    parser.add_argument('-lr', '--learning_rate', help='learning rate for neural network',
                        type=float, default=1e-3)
    parser.add_argument('-sd', '--random_seed', help='random seed for sample selection and learning',
                        type=int, default=256)
    args = parser.parse_args()
    return (args.verbose, args.plot, args.parallel, args.gpu, args.threads, args.name, args.lattice_size,
            args.super_interval, args.super_samples, args.scaler, args.latent_dimension,
            args.backend, args.optimizer, args.loss, args.epochs, args.learning_rate, args.random_seed)


def write_specs():
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
        print('random seed:               %d' % SEED)
        print('super interval:            %d' % SNI)
        print('super samples:             %d' % SNS)
        print('scaler:                    %s' % SCLR)
        print('latent dimension:          %d' % LD)
        print('backend:                   %s' % BACKEND)
        print('network:                   %s' % 'cnn2d')
        print('optimizer:                 %s' % OPT)
        print('loss function:             %s' % LSS)
        print('epochs:                    %d' % EP)
        print('learning rate:             %.2e' % LR)
        print('fitting function:          %s' % 'logistic')
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
        out.write('random seed:               %d\n' % SEED)
        out.write('super interval:            %d\n' % SNI)
        out.write('super samples:             %d\n' % SNS)
        out.write('scaler:                    %s\n' % SCLR)
        out.write('latent dimension:          %d\n' % LD)
        out.write('backend:                   %s\n' % BACKEND)
        out.write('network:                   %s\n' % 'cnn2d')
        out.write('optimizer:                 %s\n' % OPT)
        out.write('loss function:             %s\n' % LSS)
        out.write('epochs:                    %d\n' % EP)
        out.write('learning rate:             %.2e\n' % LR)
        out.write('fitting function:          %s\n' % 'logistic')
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
    dat = RealData(dom, mrng, EPS*np.ones(len(dom)), srng+EPS)
    mod = ODRModel(func)
    odr = ODR(dat, mod, pg)
    odr.set_job(fit_type=0)
    fit = odr.run()
    popt = fit.beta
    perr = fit.sd_beta
    ndom = 128
    fdom = np.linspace(np.min(dom), np.max(dom), ndom)
    fval = func(popt, fdom)
    return popt, perr, fdom, fval


def gauss_sampling(beta):
    z_mean, z_log_var = beta
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean+K.exp(0.5*z_log_var)*epsilon


def build_variational_autoencoder():
    if VERBOSE:
        print('building variational autoencoder network')
        print(100*'-')
    if SCLR in ['minmax', 'tanh', 'global']:
        outact = 'sigmoid'
    else:
        outact = 'tanh'
    init = 'he_normal'
    nf = 32
    nc = np.int32(np.log2(N/4.))
    # encoder layers
    input = Input(shape=(N, N, NCH), name='encoder_input')
    for i in range(nc):
        if i == 0:
            c = Conv2D(filters=2**i*nf, kernel_size=3, kernel_initializer=init,
                       padding='same', strides=2)(input)
        else:
            c = Conv2D(filters=2**i*nf, kernel_size=3, kernel_initializer=init,
                       padding='same', strides=2)(c)
        c = BatchNormalization()(c)
        # c = Activation('relu')(c)
        # c = LeakyReLU(alpha=0.2)(c)
        c = PReLU()(c)
    shape = K.int_shape(c)
    d0 = Flatten()(c)
    # d0 = Dense(np.int32(np.sqrt(np.prod(shape[1:]))), kernel_initializer=init)(d0)
    # d0 = PReLU()(d0)
    z_mean = Dense(LD, name='z_mean', kernel_initializer=init, activation='linear')(d0)
    # more numerically stable to use log(var_z)
    z_log_var = Dense(LD, name='z_log_std', kernel_initializer=init, activation='linear')(d0)
    z = Lambda(gauss_sampling, output_shape=(LD,), name='z')([z_mean, z_log_var])
    # construct encoder
    encoder = Model(input, [z_mean, z_log_var, z], name='encoder')
    if VERBOSE:
        print('encoder network summary')
        print(100*'-')
        encoder.summary()
        print(100*'-')
    # decoder layers
    latent_input = Input(shape=(LD,), name='z_sampling')
    d1 = Dense(np.prod(shape[1:]), kernel_initializer=init)(latent_input)
    d1 = Activation('linear')(d1)
    # d1 = PReLU()(d1)
    rd1 = Reshape(shape[1:])(d1)
    for i in range(nc-1, -1, -1):
        if i == nc-1:
            ct = Conv2DTranspose(filters=2**i*nf, kernel_size=3, kernel_initializer=init,
                                 padding='same', strides=2)(rd1)
        else:
            ct = Conv2DTranspose(filters=2**i*nf, kernel_size=3, kernel_initializer=init,
                                 padding='same', strides=2)(ct)
        ct = BatchNormalization()(ct)
        # ct = Activation('relu')(ct)
        # ct = LeakyReLU(alpha=0.2)(ct)
        ct = PReLU()(ct)
    output = Conv2DTranspose(filters=NCH, kernel_size=3, activation=outact,
                             kernel_initializer=init, padding='same', name='decoder_output')(ct)
    # construct decoder
    decoder = Model(latent_input, output, name='decoder')
    if VERBOSE:
        print('decoder network summary')
        print(100*'-')
        decoder.summary()
        print(100*'-')
    # construct vae
    output = decoder(encoder(input)[2])
    vae = Model(input, output, name='vae_mlp')
    reconstruction_losses = {'bc': lambda a, b: binary_crossentropy(a, b),
                             'mse': lambda a, b: mse(a, b),
                             'hybrid': lambda a, b: 0.5*(binary_crossentropy(a, b)+mse(a, b))}
    # vae loss
    reconstruction_loss = N*N*reconstruction_losses[LSS](K.flatten(input), K.flatten(output))
    kl_loss = 0.5*K.sum(K.exp(z_log_var)+K.square(z_mean)-z_log_var-1, axis=-1)
    vae_loss = K.mean(reconstruction_loss+kl_loss)
    vae.add_loss(vae_loss)
    # compile vae
    vae.compile(optimizer=OPTS[OPT])
    # return vae networks
    return encoder, decoder, vae


def random_selection(dmp, dat, intrvl, ns):
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
    (VERBOSE, PLOT, PARALLEL, GPU, THREADS,
     NAME, N, SNI, SNS, SCLR, LD,
     BACKEND, OPT, LSS, EP, LR, SEED) = parse_args()
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
    from keras.losses import binary_crossentropy, mse
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

    PRM = (NAME, N, SNI, SNS, SCLR, OPT, LSS, LD, EP, LR, SEED)
    OUTPREF = CWD+'/%s.%d.%d.%d.%s.%s.%s.%d.%d.%.0e.%d' % PRM

    write_specs()

    LDIR = os.listdir()

    # scaler dictionary
    SCLRS = {'minmax':MinMaxScaler(feature_range=(0, 1)),
             'standard':StandardScaler(),
             'robust':RobustScaler(),
             'tanh':TanhScaler()}

    CH = np.load(CWD+'/%s.%d.h.npy' % (NAME, N))[::SNI]
    CT = np.load(CWD+'/%s.%d.t.npy' % (NAME, N))[::SNI]
    SNH, SNT = CH.size, CT.size
    NCH = 1

    SCPREF = CWD+'/%s.%d.%d.%d.%d' % (NAME, N, SNI, SNS, SEED)

    try:
        SCDMP = np.load(SCPREF+'.dmp.sc.npy').reshape(SNH*SNT*SNS, N, N, NCH)
        CDAT = np.load(SCPREF+'.dmp.sc.npy')
        if VERBOSE:
            print('scaled selected classification samples loaded from file')
            print(100*'-')
    except:
        try:
            CDMP = np.load(SCPREF+'.dmp.c.npy')
            CDAT = np.load(SCPREF+'.dat.c.npy')
            if VERBOSE:
                # print(100*'-')
                print('selected classification samples loaded from file')
                print(100*'-')
        except:
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

        if SCLR == 'global':
            SCDMP = CDMP.reshape(SNH*SNT*SNS, N, N, NCH)
            for i in range(NCH):
                TMIN, TMAX = SCDMP[:, :, :, i].min(), SCDMP[:, :, :, i].max()
                SCDMP[:, :, :, i] = (SCDMP[:, :, :, i]-TMIN)/(TMAX-TMIN)
            del TMIN, TMAX
        elif SCLR == 'none':
            SCDMP = CDMP.reshape(SNH*SNT*SNS, N, N, NCH)
        else:
            SCDMP = SCLRS[SCLR].fit_transform(CDMP.reshape(SNH*SNT*SNS, N*N*NCH)).reshape(SNH*SNT*SNS, N, N, NCH)
        del CDMP
        np.save(CWD+'/%s.%d.%d.%d.%s.%d.dmp.sc.npy' % (NAME, N, SNI, SNS, SCLR, SEED), SCDMP.reshape(SNH, SNT, SNS, N, N, NCH))
        if VERBOSE:
            print('scaled selected classification samples computed')
            print(100*'-')

    ES = CDAT[:, :, :, 0]
    MS = CDAT[:, :, :, 1]
    EM = np.mean(ES, -1)
    SP = np.var(ES/CT[np.newaxis, :, np.newaxis], 2)
    MM = np.mean(MS, -1)
    SU = np.var(MS/CT[np.newaxis, :, np.newaxis], 2)

    OPTS = {'sgd': SGD(lr=LR, momentum=0.0, decay=0.0, nesterov=True),
            'adadelta': Adadelta(lr=LR, rho=0.95, epsilon=None, decay=0.0),
            'adam': Adam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True),
            'nadam': Nadam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)}

    ENC, DEC, VAE = build_variational_autoencoder()

    try:
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
        CSVLG = CSVLogger(OUTPREF+'.vae.log.csv', append=True, separator=',')
        LR_DECAY = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, verbose=VERBOSE)
        TRN, VAL = train_test_split(SCDMP, test_size=0.125, shuffle=True)
        VAE.fit(x=TRN, y=None, validation_data=(VAL, None), epochs=EP, batch_size=SNH*SNT,
                shuffle=True, verbose=VERBOSE, callbacks=[CSVLG, LR_DECAY, History()])
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
        ZENC = np.load(OUTPREF+'.zenc.npy').reshape(SNH*SNT*SNS, ED, LD)
        ERR = np.load(OUTPREF+'.zerr.npy')
        if SCLR in ['minmax', 'global', 'none']:
            RERR = np.load(OUTPREF+'.zerr.round.npy')
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
            print('z encodings of scaled selected classification samples loaded from file')
            print(100*'-')
    except:
        if VERBOSE:
            print('predicting z encodings of scaled selected classification samples')
            print(100*'-')
        ZENC = np.array(ENC.predict(SCDMP, verbose=VERBOSE))
        ZDEC = np.array(DEC.predict(ZENC[2, :, :], verbose=VERBOSE))
        ZENC = np.swapaxes(ZENC, 0, 1)[:, :2, :]
        ZENC[:, 1, :] = np.exp(0.5*ZENC[:, 1, :])
        ERR = SCDMP-ZDEC
        if SCLR in ['minmax', 'global', 'none']:
            RERR = np.unique(SCDMP-np.round(ZDEC), return_counts=True)
        np.save(OUTPREF+'.zerr.round.npy', RERR)
        KLD = 0.5*np.sum(np.square(ZENC[:, 1, :])+np.square(ZENC[:, 0, :])-np.log(np.square(ZENC[:, 1, :]))-1, axis=1)
        np.save(OUTPREF+'.zenc.npy', ZENC.reshape(SNH, SNT, SNS, ED, LD))
        np.save(OUTPREF+'.zdec.npy', ZDEC.reshape(SNH, SNT, SNS, N, N, NCH))
        np.save(OUTPREF+'.zerr.npy', ERR.reshape(SNH, SNT, SNS, N, N, NCH))
        np.save(OUTPREF+'.zerr.kld.npy', KLD.reshape(SNH, SNT, SNS))
        MERR = np.mean(ERR)
        SERR = np.std(ERR)
        MXERR = np.max(ERR)
        MNERR = np.min(ERR)
        MKLD = np.mean(KLD)
        SKLD = np.std(KLD)
        MXKLD = np.max(KLD)
        MNKLD = np.min(KLD)
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
        print('z encodings of scaled selected classification samples predicted')
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
        if SCLR in ['minmax', 'global', 'none']:
            print('rounded output error')
            print(100*'-')
            print('error:'+RERR[0].size*' %0.2e' % tuple(RERR[0]))
            print('count:'+RERR[1].size*' %0.2e' % tuple(RERR[1]))
            print('prop: '+RERR[1].size*' %0.2e' % tuple(RERR[1]/np.sum(RERR[1])))
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
        PZENC = np.load(OUTPREF+'.zenc.pca.prj.npy').reshape(SNH*SNT*SNS, ED, LD)
        CZENC = np.load(OUTPREF+'.zenc.pca.cmp.npy')
        VZENC = np.load(OUTPREF+'.zenc.pca.var.npy')
        if VERBOSE:
            print('pca projections of z encodings  loaded from file')
            print(100*'-')
    except:
        if VERBOSE:
            print('pca projecting z encodings')
            print(100*'-')
        PCAZENC = PCA(n_components=LD)
        PZENC = np.zeros((SNH*SNT*SNS, ED, LD))
        CZENC = np.zeros((ED, LD, LD))
        VZENC = np.zeros((ED, LD))
        for i in range(ED):
            PZENC[:, i, :] = PCAZENC.fit_transform(ZENC[:, i, :])
            CZENC[i, :, :] = PCAZENC.components_
            VZENC[i, :] = PCAZENC.explained_variance_ratio_
        np.save(OUTPREF+'.zenc.pca.prj.npy', PZENC.reshape(SNH, SNT, SNS, ED, LD))
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
        plt.xlabel('ERR')
        plt.ylabel('DENSITY')
        fig.savefig(OUTPREF+'.vae.kld.png')
        plt.close()

        MEMDIAG = SCLRS['minmax'].fit_transform(np.stack((MM, EM), axis=-1).reshape(SNH*SNT, 2)).reshape(SNH, SNT, 2)
        MEVDIAG = SCLRS['minmax'].fit_transform(np.stack((SU, SP), axis=-1).reshape(SNH*SNT, 2)).reshape(SNH, SNT, 2)

        ZMDIAG = SCLRS['minmax'].fit_transform(np.mean(ZENC.reshape(SNH, SNT, SNS, ED*LD), 2).reshape(SNH*SNT, ED*LD)).reshape(SNH, SNT, ED, LD)
        ZVDIAG = SCLRS['minmax'].fit_transform(np.var(ZENC.reshape(SNH, SNT, SNS, ED*LD)/\
                                                      CT[np.newaxis, :, np.newaxis, np.newaxis], 2).reshape(SNH*SNT, ED*LD)).reshape(SNH, SNT, ED, LD)

        PZMDIAG = SCLRS['minmax'].fit_transform(np.mean(PZENC.reshape(SNH, SNT, SNS, ED*LD), 2).reshape(SNH*SNT, ED*LD)).reshape(SNH, SNT, ED, LD)
        for i in range(LD):
            if PZMDIAG[0, 0, 0, i] > PZMDIAG[-1, 0, 0, i]:
                PZMDIAG[:, :, 0, i] = 1-PZMDIAG[:, :, 0, i]
            if PZMDIAG[0, 0, 1, i] > PZMDIAG[0, -1, 1, i]:
                PZMDIAG[:, :, 1, i] = 1-PZMDIAG[:, :, 1, i]
        PZVDIAG = SCLRS['minmax'].fit_transform(np.var(PZENC.reshape(SNH, SNT, SNS, ED*LD)/\
                                                       CT[np.newaxis, :, np.newaxis, np.newaxis], 2).reshape(SNH*SNT, ED*LD)).reshape(SNH, SNT, ED, LD)

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
        vae_plots()