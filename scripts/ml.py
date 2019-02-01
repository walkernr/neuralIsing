# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 22:07:13 2019

@author: Nicholas
"""

import argparse
import os
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelBinarizer
from TanhScaler import TanhScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap, LocallyLinearEmbedding
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from scipy.odr import ODR, Model, RealData

# parse command line
PARSER = argparse.ArgumentParser()
PARSER.add_argument('-v', '--verbose', help='verbose output', action='store_true')
PARSER.add_argument('-pt', '--plot', help='plot results', action='store_true')
PARSER.add_argument('-p', '--parallel', help='parallel run', action='store_true')
PARSER.add_argument('-nt', '--threads', help='number of threads',
                    type=int, default=16)
PARSER.add_argument('-n', '--name', help='simulation name',
                        type=str, default='ising_init')
PARSER.add_argument('-ls', '--lattice_size', help='lattice size',
                    type=int, default=4)
PARSER.add_argument('-ui', '--unsuper_interval', help='interval for selecting phase points (unsupervised)',
                    type=int, default=2)
PARSER.add_argument('-un', '--unsuper_samples', help='number of samples per phase point (unsupervised)',
                    type=int, default=32)
PARSER.add_argument('-si', '--super_interval', help='interval for selecting phase points (supervised)',
                    type=int, default=1)
PARSER.add_argument('-sn', '--super_samples', help='number of samples per phase point (supervised)',
                    type=int, default=128)
PARSER.add_argument('-sc', '--scaler', help='feature scaler',
                    type=str, default='tanh')
PARSER.add_argument('-rd', '--reduction', help='supervised dimension reduction method',
                    type=str, default='tsne')
PARSER.add_argument('-np', '--projections', help='number of embedding projections',
                    type=int, default=2)
PARSER.add_argument('-cl', '--clustering', help='clustering method',
                    type=str, default='agglomerative')
PARSER.add_argument('-nc', '--clusters', help='number of clusters',
                    type=int, default=4)
PARSER.add_argument('-bk', '--backend', help='keras backend',
                    type=str, default='tensorflow')
PARSER.add_argument('-ep', '--epochs', help='number of epochs',
                    type=int, default=8)
PARSER.add_argument('-lr', '--learning_rate', help='learning rate for neural network',
                    type=float, default=1e-2)

# parse arguments
ARGS = PARSER.parse_args()
# run specifications
VERBOSE = ARGS.verbose
PLOT = ARGS.plot
PARALLEL = ARGS.parallel
THREADS = ARGS.threads
NAME = ARGS.name
N = ARGS.lattice_size
UNI = ARGS.unsuper_interval
UNS = ARGS.unsuper_samples
SNI = ARGS.super_interval
SNS = ARGS.super_samples
SCLR = ARGS.scaler
RDCN = ARGS.reduction
NP = ARGS.projections
CLST = ARGS.clustering
NC = ARGS.clusters
BACKEND = ARGS.backend
EP = ARGS.epochs
LR = ARGS.learning_rate

# random seed
SEED = 256
np.random.seed(SEED)
# number of phases
NPH = 3
# environment variables
os.environ['KERAS_BACKEND'] = BACKEND
if BACKEND == 'tensorflow':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    from tensorflow import set_random_seed
    set_random_seed(SEED)
if PARALLEL:
    os.environ['MKL_NUM_THREADS'] = str(THREADS)
    os.environ['GOTO_NUM_THREADS'] = str(THREADS)
    os.environ['OMP_NUM_THREADS'] = str(THREADS)
    os.environ['openmp'] = 'True'
from keras.models import Sequential
from keras.layers import Conv2D, AveragePooling2D, Dropout, Dense, Flatten
from keras.optimizers import Nadam
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import History

if PLOT:
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid
    plt.rc('font', family='sans-serif')
    FTSZ = 28
    PPARAMS = {'figure.figsize': (26, 20),
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

# information
if VERBOSE:
    print(66*'-')
    print('data loaded')
    print(66*'-')
    print('input summary')
    print(66*'-')
    print('plot:                      %d' % PLOT)
    print('parallel:                  %d' % PARALLEL)
    print('threads:                   %d' % THREADS)
    print('name:                      %s' % NAME)
    print('lattice size:              %s' % N)
    print('unsuper interval:          %d' % UNI)
    print('unsuper samples:           %d' % UNS)
    print('super interval:            %d' % SNI)
    print('super samples:             %d' % SNS)
    print('scaler:                    %s' % SCLR)
    print('reduction:                 %s' % RDCN)
    print('projections:               %d' % NP)
    print('clustering:                %s' % CLST)
    print('clusters:                  %d' % NC)
    print('backend:                   %s' % BACKEND)
    print('network:                   %s' % 'cnn1d')
    print('epochs:                    %d' % EP)
    print('learning rate:             %.2e' % LR)
    print('fitting function:          %s' % 'logistic')
    print(66*'-')

CWD = os.getcwd()
OUTPREF = CWD+'/%s.%d.%d.%d.%d.%d.%s.%s.%d.%s.%d.cnn2d.%d.%.0e.logistic' \
          % (NAME, N, UNI, UNS, SNI, SNS, SCLR, RDCN, NP, CLST, NC, EP, LR)
with open(OUTPREF+'.out', 'w') as out:
    out.write('# ' + 66*'-' + '\n')
    out.write('# input summary\n')
    out.write('# ' + 66*'-' + '\n')
    out.write('# plot:                      %d\n' % PLOT)
    out.write('# parallel:                  %d\n' % PARALLEL)
    out.write('# threads:                   %d\n' % THREADS)
    out.write('# name:                      %s\n' % NAME)
    out.write('# lattice size:              %s\n' % N)
    out.write('unsuper interval:            %d\n' % UNI)
    out.write('unsuper samples:             %d\n' % UNS)
    out.write('super interval:              %d\n' % SNI)
    out.write('super samples:               %d\n' % SNS)
    out.write('# scaler:                    %s\n' % SCLR)
    out.write('# reduction:                 %s\n' % RDCN)
    out.write('# projections:               %d\n' % NP)
    out.write('# clustering:                %s\n' % CLST)
    out.write('# clusters:                  %d\n' % NC)
    out.write('# backend:                   %s\n' % BACKEND)
    out.write('# network:                   %s\n' % 'cnn1d')
    out.write('# epochs:                    %d\n' % EP)
    out.write('# learning rate:             %.2e\n' % LR)
    out.write('# fitting function:          %s\n' % 'logistic')

EPS = 0.0125
# load phase point data
UH = pickle.load(open(CWD+'/%s.%d.h.pickle' % (NAME, N), 'rb'))[::UNI]
UT = pickle.load(open(CWD+'/%s.%d.t.pickle' % (NAME, N), 'rb'))[::UNI]
SH = pickle.load(open(CWD+'/%s.%d.h.pickle' % (NAME, N), 'rb'))[::SNI]
ST = pickle.load(open(CWD+'/%s.%d.t.pickle' % (NAME, N), 'rb'))[::SNI]
UNH, UNT = UH.size, UT.size
SNH, SNT = SH.size, ST.size
# load data
DAT = pickle.load(open(CWD+'/%s.%d.dat.pickle' % (NAME, N), 'rb'))
UES = DAT[::UNI, ::UNI, -UNS:, 0].reshape(UNH, UNT, UNS)
UMS = DAT[::UNI, ::UNI, -UNS:, 1].reshape(UNH, UNT, UNS)
SES = DAT[::SNI, ::SNI, -SNS:, 0].reshape(SNH, SNT, SNS)
SMS = DAT[::SNI, ::SNI, -SNS:, 1].reshape(SNH, SNT, SNS)
del DAT
UDAT = pickle.load(open(CWD+'/%s.%d.dmp.pickle' % (NAME, N), 'rb'))[::UNI, ::UNI, -UNS:, :, :].reshape(UNH, UNT, UNS, N*N)
SDAT = pickle.load(open(CWD+'/%s.%d.dmp.pickle' % (NAME, N), 'rb'))[::SNI, ::SNI, -SNS:, :, :]
# data shape
_, _, _, UNF = UDAT.shape
_, _, _, SNF0, SNF1 = SDAT.shape
# phase point multidimensional arrays
UHS = np.array([UH[i]*np.ones((UNT, UNS)) for i in range(UNH)])
UTS = np.ones((UNH, UNT, UNS))*np.array([UT[i]*np.ones(UNS) for i in range(UNT)])[np.newaxis, :, :]
SHS = np.array([SH[i]*np.ones((SNT, SNS)) for i in range(SNH)])
STS = np.ones((SNH, SNT, SNS))*np.array([ST[i]*np.ones(SNS) for i in range(SNT)])[np.newaxis, :, :]

if PLOT:
    CM = plt.get_cmap('plasma')
    SCALE = lambda a, b: (a-np.min(b))/(np.max(b)-np.min(b))
    def rgb_intens(rgb):
        gamma = 2.2
        rgblin = np.power(rgb, gamma)
        lumac = np.array([.2126, .7152, .0722])[np.newaxis, np.newaxis, :]
        luma = np.sum(np.multiply(rgblin, yfac), -1)
        return 116*np.power(luma, 1/3)-16
    if VERBOSE:
        print('colormap and scale initialized')
        print(66*'-')


# fitting function
def logistic(beta, t):
    ''' returns logistic sigmoid '''
    a = 0.0
    k = 1.0
    b, m = beta
    return a+np.divide(k, 1+np.exp(-b*(t-m)))


def absolute(beta, t):
    a, b, c = beta
    return a*np.abs(t-b)+c


# odr fitting
def odr_fit(func, dom, mrng, srng, pg):
    ''' performs orthogonal distance regression '''
    dat = RealData(dom, mrng, EPS*np.ones(len(dom)), srng+EPS)
    mod = Model(func)
    odr = ODR(dat, mod, pg)
    odr.set_job(fit_type=0)
    fit = odr.run()
    popt = fit.beta
    perr = fit.sd_beta
    ndom = 128
    fdom = np.linspace(np.min(dom), np.max(dom), ndom)
    fval = func(popt, fdom)
    return popt, perr, fdom, fval

if VERBOSE:
    print('fitting function initialized')
    print(66*'-')

# scaler dictionary
SCLRS = {'minmax':MinMaxScaler(feature_range=(0, 1)),
         'standard':StandardScaler(),
         'robust':RobustScaler(),
         'tanh':TanhScaler()}
# reduction dictionary
RDCNS = {'pca':PCA(n_components=0.99),
         'kpca':KernelPCA(n_components=NP, n_jobs=THREADS),
         'isomap':Isomap(n_components=NP, n_jobs=THREADS),
         'lle':LocallyLinearEmbedding(n_components=NP, n_jobs=THREADS),
         'tsne':TSNE(n_components=NP, perplexity=UNS,
                     early_exaggeration=12, learning_rate=200, n_iter=2000,
                     verbose=True, n_jobs=THREADS)}

if VERBOSE:
    print('scaling and reduction initialized')
    print(66*'-')

# neural network construction
def build_keras_cnn2d():
    ''' builds 2-d convolutional neural network '''
    model = Sequential([Conv2D(filters=32, kernel_size=(2, 2), activation='relu',
                               kernel_initializer='he_normal',
                               padding='valid', strides=1, input_shape=(SNF0, SNF1, 1)),
                        Dropout(rate=0.25),
                        Conv2D(filters=32, kernel_size=(2, 2), activation='relu',
                               kernel_initializer='he_normal',
                               padding='valid', strides=1),
                        # AveragePooling2D(pool_size=(2, 2)),
                        # Dropout(rate=0.25),
                        # Conv2D(filters=64, kernel_size=(2, 2), activation='relu',
                        #        kernel_initializer='he_normal',
                        #        padding='same', strides=1),
                        # Conv2D(filters=64, kernel_size=(2, 2), activation='relu',
                        #        kernel_initializer='he_normal',
                        #        padding='same', strides=1),
                        # AveragePooling2D(pool_size=(2, 2)),
                        # Dropout(rate=0.25),
                        Flatten(),
                        Dense(units=64, activation='relu'),
                        # Dropout(rate=0.5),
                        Dense(units=3, activation='sigmoid')])
    nadam = Nadam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    model.compile(loss='categorical_crossentropy', optimizer=nadam, metrics=['mae', 'acc'])
    return model

NN = KerasClassifier(build_keras_cnn2d, epochs=EP, batch_size=32,
                     shuffle=True, verbose=VERBOSE, callbacks=[History()])

# clustering dictionary
CLSTS = {'agglomerative': AgglomerativeClustering(n_clusters=NC),
         'kmeans': KMeans(n_jobs=THREADS, n_clusters=NC, init='k-means++'),
         'spectral': SpectralClustering(n_jobs=THREADS, n_clusters=NC, eigen_solver='amg')}

if VERBOSE:
    print('neural network and clustering initialized')
    print(66*'-')

# scale unsupervised data
try:
    FUDOM = pickle.load(open(CWD+'/%s.%d.%d.%d.%s.fudat.k.pickle' % (NAME, N, UNI, UNS, SCLR), 'rb'))
    FUNF = FUDOM.size
    FUDAT = pickle.load(open(CWD+'/%s.%d.%d.%d.%s.fudat.pickle' % (NAME, N, UNI, UNS, SCLR), 'rb')).reshape(UNH*UNT*UNS, FUNF)
    FUDAT = np.concatenate((np.real(FUDAT), np.imag(FUDAT)), axis=1)
    SUDAT = pickle.load(open(CWD+'/%s.%d.%d.%d.%s.sudat.pickle' % (NAME, N, UNI, UNS, SCLR), 'rb')).reshape(UNH*UNT*UNS, 2*FUNF)
    if VERBOSE:
        print('scaled unsupervised data loaded from file')
except:
    FUDOM = np.fft.rfftfreq(UNF, 1)
    FUNF = FUDOM.size
    FUDAT = np.fft.rfft(UDAT.reshape(UNH*UNT*UNS, UNF))
    pickle.dump(FUDOM, open(CWD+'/%s.%d.%d.%d.%s.fudat.k.pickle' % (NAME, N, UNI, UNS, SCLR), 'wb'))
    pickle.dump(FUDAT.reshape(UNH, UNT, UNS, FUNF), open(CWD+'/%s.%d.%d.%d.%s.fudat.pickle' % (NAME, N, UNI, UNS, SCLR), 'wb'))
    FUDAT = np.concatenate((np.real(FUDAT), np.imag(FUDAT)), axis=1)
    SUDAT = SCLRS[SCLR].fit_transform(FUDAT)
    pickle.dump(SUDAT.reshape(UNH, UNT, UNS, 2*FUNF), open(CWD+'/%s.%d.%d.%d.%s.sudat.pickle' % (NAME, N, UNI, UNS, SCLR), 'wb'))
    if VERBOSE:
        print('unsupervised data scaled')
if VERBOSE:
    print(66*'-')

# pca reduce unsupervised data
try:
    EVAR = pickle.load(open(CWD+'/%s.%d.%d.%d.%s.evar.pickle' % (NAME, N, UNI, UNS, SCLR), 'rb'))
    PUDAT = pickle.load(open(CWD+'/%s.%d.%d.%d.%s.pudat.pickle' % (NAME, N, UNI, UNS, SCLR), 'rb')).reshape(UNT*UNH*UNS, len(EVAR))
    PCOMP = pickle.load(open(CWD+'/%s.%d.%d.%d.%s.pcomp.pickle' % (NAME, N, UNI, UNS, SCLR), 'rb'))
    if VERBOSE:
        print('pca reduced unsupervised data loaded from file')
except:
    PUDAT = RDCNS['pca'].fit_transform(SUDAT)
    EVAR = RDCNS['pca'].explained_variance_ratio_
    PCOMP = RDCNS['pca'].components_
    pickle.dump(PUDAT.reshape(UNT, UNH, UNS, len(EVAR)),
                open(CWD+'/%s.%d.%d.%d.%s.pudat.pickle' % (NAME, N, UNI, UNS, SCLR), 'wb'))
    pickle.dump(EVAR, open(CWD+'/%s.%d.%d.%d.%s.evar.pickle' % (NAME, N, UNI, UNS, SCLR), 'wb'))
    pickle.dump(PCOMP, open(CWD+'/%s.%d.%d.%d.%s.pcomp.pickle' % (NAME, N, UNI, UNS, SCLR), 'wb'))
    if VERBOSE:
        print('unsupervised data pca reduced')
if VERBOSE:
    print(66*'-')
    print('principal components:     %d' % len(EVAR))
    print('explained variances:      %0.4f %0.4f %0.4f ...' % tuple(EVAR[:3]))
    print('total explained variance: %0.4f' % np.sum(EVAR))
    print(66*'-')

with open(OUTPREF+'.out', 'a') as out:
    out.write('# ' + 66*'-' + '\n')
    out.write('# pca fit\n')
    out.write('# ' + 66*'-' + '\n')
    out.write('# principal components:     %d\n' % len(EVAR))
    out.write('# explained variances:      %0.4f %0.4f %0.4f ...\n' % tuple(EVAR[:3]))
    out.write('# total explained variance: %0.4f\n' % np.sum(EVAR))

# reduction of unsupervised data
try:
    RUDAT = pickle.load(open(CWD+'/%s.%d.%d.%d.%s.%s.%d.rudat.pickle' \
                             % (NAME, N, UNI, UNS, SCLR, RDCN, NP), 'rb')).reshape(UNH*UNT*UNS, NP)
    if VERBOSE:
        print('nonlinearly reduced unsupervised data loaded from file')
except:
    if RDCN not in ('none', 'pca'):
        RUDAT = RDCNS[RDCN].fit_transform(PUDAT)
        pickle.dump(RUDAT.reshape(UNH, UNT, UNS, NP), open(CWD+'/%s.%d.%d.%d.%s.%s.%d.rudat.pickle' \
                                                           % (NAME, N, UNI, UNS, SCLR, RDCN, NP), 'wb'))
        if RDCN == 'tsne' and VERBOSE:
            print(66*'-')
        if VERBOSE:
            print('unsupervised data nonlinearly reduced')
    else:
        RUDAT = PUDAT[:, :NP]
    _, RUNF = RUDAT.shape

# clustering
if VERBOSE:
    print(np.max([66, 10+8*NC])*'-')
try:
    UPRED = pickle.load(open(CWD+'/%s.%d.%d.%d.%s.%s.%d.%s.%d.upred.pickle' \
                             % (NAME, N, UNI, UNS, SCLR, RDCN, NP, CLST, NC), 'rb')).reshape(UNH*UNT*UNS)
    if VERBOSE:
        print('clustered unsupervised data loaded from file')
except:
    UPRED = CLSTS[CLST].fit_predict(RUDAT)
    UCTM = np.array([np.mean(UTS.reshape(UNH*UNT*UNS)[UPRED == i]) for i in range(NC)])
    IUCTM = np.argsort(UCTM)
    for i in range(NC):
        UPRED[UPRED == IUCTM[i]] = i+NC
    UPRED -= NC
    pickle.dump(UPRED.reshape(UNH, UNT, UNS), open(CWD+'/%s.%d.%d.%d.%s.%s.%d.%s.%d.upred.pickle' \
                                                   % (NAME, N, UNI, UNS, SCLR, RDCN, NP, CLST, NC), 'wb'))
    if VERBOSE:
        print('unsupervised data clustered')
UCMM = np.array([np.mean(UMS.reshape(UNH*UNT*UNS)[UPRED == i]) for i in range(NC)])
UCTM = np.array([np.mean(UTS.reshape(UNH*UNT*UNS)[UPRED == i]) for i in range(NC)])
# make this better
if NC == 4:
    IUCTM = np.argsort(UCTM)
    UPRED[UPRED == np.max(IUCTM[-2:])] = np.min(IUCTM[-2:])
    UCMM = np.array([np.mean(UMS.reshape(UNH*UNT*UNS)[UPRED == i]) for i in range(NPH)])
    UCTM = np.array([np.mean(UTS.reshape(UNH*UNT*UNS)[UPRED == i]) for i in range(NPH)])
UPREDB = np.array([[np.bincount(UPRED.reshape(UNH, UNT, UNS)[i, j], minlength=NPH) for j in range(UNT)] for i in range(UNH)])/UNS
UPREDC = np.array([[np.argmax(np.bincount(UPRED.reshape(UNH, UNT, UNS)[i, j])) for j in range(UNT)] for i in range(UNH)])
UTRANS = np.array([odr_fit(logistic, UT, UPREDB[i, :, 2], EPS*np.ones(UNT), (1, 2.5))[0][1] for i in range(UNH)])
UITRANS = (UTRANS-UT[0])/(UT[-1]-UT[0])*(UNT-1)
UCPOPT, UCPERR, UCDOM, UCVAL = odr_fit(absolute, UH, UTRANS, EPS*np.ones(UNT), (1, 0, 2.5))
UICDOM = (UCDOM-UH[0])/(UH[-1]-UH[0])*(UNH-1)
UICVAL = (UCVAL-UT[0])/(UT[-1]-UT[0])*(UNT-1)

if VERBOSE:
    print(66*'-')
    print('h\tt')
    print(66*'-')
    for i in range(UNH):
        print('%.2f\t%.2f' % (UH[i], UTRANS[i]))

with open(OUTPREF+'.out', 'a') as out:
    out.write('# '+66*'-'+'\n')
    out.write('# unsupervised learning results\n')
    out.write('# h\tt\n')
    out.write('# '+66*'-'+'\n')
    for i in range(UNH):
        out.write('%.2f\t%.2f\n' % (UH[i], UTRANS[i]))

# scale supervised data
try:
    SSTDAT = pickle.load(open(CWD+'/%s.%d.%d.%d.%s.sstdat.pickle' % (NAME, N, UNI, UNS, SCLR), 'rb')).reshape(UNH*UNT*UNS, N, N)
    SSCDAT = pickle.load(open(CWD+'/%s.%d.%d.%d.%d.%d.%s.sscdat.pickle' % (NAME, N, UNI, UNS, SNI, SNS, SCLR), 'rb')).reshape(SNH*SNT*SNS, SNF0, SNF1)
    if VERBOSE:
        print(66*'-')
        print('scaled supervised data loaded from file')
except:
    SCLRS[SCLR].fit(UDAT.reshape(UNH*UNT*UNS, UNF))
    SSTDAT = SCLRS[SCLR].transform(UDAT.reshape(UNH*UNT*UNS, UNF)).reshape(UNH*UNT*UNS, N, N)
    SSCDAT = SCLRS[SCLR].transform(SDAT.reshape(SNH*SNT*SNS, SNF0*SNF1)).reshape(SNH*SNT*SNS, SNF0, SNF1)
    pickle.dump(SSTDAT.reshape(UNH, UNT, UNS, N, N), open(CWD+'/%s.%d.%d.%d.%s.sstdat.pickle' % (NAME, N, UNI, UNS, SCLR), 'wb'))
    pickle.dump(SSCDAT.reshape(SNH, SNT, SNS, SNF0, SNF1), open(CWD+'/%s.%d.%d.%d.%d.%d.%s.sscdat.pickle' % (NAME, N, UNI, UNS, SNI, SNS, SCLR), 'wb'))
    if VERBOSE:
        print(125*'-')
        print('supervised data scaled')

# fit neural network to training data and predict classification data
try:
    LOSS = pickle.load(open(CWD+'/%s.%d.%d.%d.%s.%s.%d.%s.%d.cnn1d.%d.%.0e.loss.pickle' \
                            % (NAME, N, UNI, UNS, SCLR, RDCN, NP, CLST, NC, EP, LR), 'rb'))
    MAE = pickle.load(open(CWD+'/%s.%d.%d.%d.%s.%s.%d.%s.%d.cnn1d.%d.%.0e.mae.pickle' \
                           % (NAME, N, UNI, UNS, SCLR, RDCN, NP, CLST, NC, EP, LR), 'rb'))
    ACC = pickle.load(open(CWD+'/%s.%d.%d.%d.%s.%s.%d.%s.%d.cnn1d.%d.%.0e.acc.pickle' \
                           % (NAME, N, UNI, UNS, SCLR, RDCN, NP, CLST, NC, EP, LR), 'rb'))
    SPROB = pickle.load(open(CWD+'/%s.%d.%d.%d.%d.%d.%s.%s.%d.%s.%d.cnn1d.%d.%.0e.sprob.pickle' \
                             % (NAME, N, UNI, UNS, SNI, SNS, SCLR, RDCN, NP, CLST, NC, EP, LR), 'rb'))
    if VERBOSE:
        print(66*'-')
        print('neural network fit loaded from file')
except:
    if VERBOSE:
        print(125*'-')
    # fit training data
    ENC = LabelBinarizer().fit(np.arange(NPH))
    LBLS = ENC.transform(UPRED)
    NN.fit(SSTDAT[:, :, :, np.newaxis], LBLS)
    LOSS = NN.model.history.history['loss']
    MAE = NN.model.history.history['mean_absolute_error']
    ACC = NN.model.history.history['acc']
    pickle.dump(LOSS, open(CWD+'/%s.%d.%d.%d.%s.%s.%d.%s.%d.cnn1d.%d.%.0e.loss.pickle' \
                           % (NAME, N, UNI, UNS, SCLR, RDCN, NP, CLST, NC, EP, LR), 'wb'))
    pickle.dump(MAE, open(CWD+'/%s.%d.%d.%d.%s.%s.%d.%s.%d.cnn1d.%d.%.0e.mae.pickle' \
                          % (NAME, N, UNI, UNS, SCLR, RDCN, NP, CLST, NC, EP, LR), 'wb'))
    pickle.dump(ACC, open(CWD+'/%s.%d.%d.%d.%s.%s.%d.%s.%d.cnn1d.%d.%.0e.acc.pickle' \
                          % (NAME, N, UNI, UNS, SCLR, RDCN, NP, CLST, NC, EP, LR), 'wb'))
    if VERBOSE:
        print(125*'-')
        print('neural network fitted to training data')
        print(66*'-')
    # predict classification data
    SPROB = NN.predict_proba(SSCDAT[:, :, :, np.newaxis])
    pickle.dump(SPROB, open(CWD+'/%s.%d.%d.%d.%d.%d.%s.%s.%d.%s.%d.cnn1d.%d.%.0e.sprob.pickle' \
                            % (NAME, N, UNI, UNS, SNI, SNS, SCLR, RDCN, NP, CLST, NC, EP, LR), 'wb'))

SPRED = np.argmax(SPROB, -1)
SPROBM = np.mean(SPROB.reshape(SNH, SNT, SNS, 3), 2)
SPROBS = np.std(SPROB.reshape(SNH, SNT, SNS, 3), 2)
STRANS = np.array([odr_fit(logistic, ST, SPROBM[i, :, 2], SPROBS[i, :, 2], (1, 2.5))[0][1] for i in range(SNH)])
SITRANS = (STRANS-ST[0])/(ST[-1]-ST[0])*(SNT-1)
SCPOPT, SCPERR, SCDOM, SCVAL = odr_fit(absolute, SH, STRANS, EPS*np.ones(SNT), (1, 0, 2.5))
SICDOM = (SCDOM-SH[0])/(SH[-1]-SH[0])*(SNH-1)
SICVAL = (SCVAL-ST[0])/(ST[-1]-ST[0])*(SNT-1)

if VERBOSE:
    print(66*'-')
    print('neural network predicted classification data')
    print(66*'-')
    print('h\tt')
    print(66*'-')
    for i in range(SNH):
        print('%.2f\t%.2f' % (SH[i], STRANS[i]))
    print(66*'-')

with open(OUTPREF+'.out', 'a') as out:
    out.write('# ' + 66*'-' + '\n')
    out.write('# supervised learning results\n')
    out.write('# ' + 66*'-' + '\n')
    out.write('# epoch\tloss\tmae\tacc\n')
    out.write('# ' + 66*'-' + '\n')
    for i in range(EP):
        out.write('# %02d\t' % i + 3*'%0.4f\t' % (LOSS[i], MAE[i], ACC[i]) + '\n')
    out.write('# ' + 66*'-' + '\n')
    out.write('# h\tt\n')
    out.write('# ' + 66*'-' + '\n')
    for i in range(SNH):
        out.write('%.2f\t%.2f' % (SH[i], STRANS[i]) +'\n')
    out.write('# ' + 66*'-' + '\n')

if PLOT:


    def plot_uemb():
        ''' plot of unsupervised reduced sample space '''
        fig = plt.figure()
        grid = ImageGrid(fig, 111,
                         nrows_ncols=(3, 2),
                         axes_pad=2.0,
                         share_all=True) # ,
                         # cbar_location="right",
                         # cbar_mode="single",
                         # cbar_size="4%",
                         # cbar_pad=0.4)
        for j in range(len(grid)):
            grid[j].spines['right'].set_visible(False)
            grid[j].spines['top'].set_visible(False)
            grid[j].xaxis.set_ticks_position('bottom')
            grid[j].yaxis.set_ticks_position('left')
        # cbd = grid[0].scatter(RUDAT[:, 0], RUDAT[:, 1], c=MS, cmap=CM, s=120, alpha=0.025,
        #                       edgecolors='none')
        grid[0].scatter(RUDAT[:, 0], RUDAT[:, 1], c=UES.reshape(UNH*UNT*UNS), cmap=CM, s=120, alpha=0.0125, edgecolors='none')
        grid[0].set_aspect('equal', 'datalim')
        grid[0].set_xlabel(r'$x_0$')
        grid[0].set_ylabel(r'$x_1$')
        grid[0].set_title(r'$\mathrm{(a)\enspace Sample\enspace Embedding (E)}$', y=1.02)
        grid[2].scatter(RUDAT[:, 0], RUDAT[:, 1], c=UMS.reshape(UNH*UNT*UNS), cmap=CM, s=120, alpha=0.0125, edgecolors='none')
        grid[2].set_aspect('equal', 'datalim')
        grid[2].set_xlabel(r'$x_0$')
        grid[2].set_ylabel(r'$x_1$')
        grid[2].set_title(r'$\mathrm{(c)\enspace Sample\enspace Embedding (M)}$', y=1.02)
        grid[4].scatter(RUDAT[:, 0], RUDAT[:, 1], c=UTS.reshape(UNH*UNT*UNS), cmap=CM, s=120, alpha=0.0125, edgecolors='none')
        grid[4].set_aspect('equal', 'datalim')
        grid[4].set_xlabel(r'$x_0$')
        grid[4].set_ylabel(r'$x_1$')
        grid[4].set_title(r'$\mathrm{(e)\enspace Sample\enspace Embedding (T)}$', y=1.02)
        for j in range(NPH):
            grid[1].scatter(RUDAT[UPRED == j, 0], RUDAT[UPRED == j, 1],
                            c=np.array(CM(SCALE(np.mean(UES.reshape(-1)[UPRED == j]), UES.reshape(-1))))[np.newaxis, :], s=120, alpha=0.0125,
                            edgecolors='none')
            grid[3].scatter(RUDAT[UPRED == j, 0], RUDAT[UPRED == j, 1],
                            c=np.array(CM(SCALE(np.mean(UMS.reshape(-1)[UPRED == j]), UMS.reshape(-1))))[np.newaxis, :], s=120, alpha=0.0125,
                            edgecolors='none')
            grid[5].scatter(RUDAT[UPRED == j, 0], RUDAT[UPRED == j, 1],
                            c=np.array(CM(SCALE(np.mean(UTS.reshape(-1)[UPRED == j]), UTS.reshape(-1))))[np.newaxis, :], s=120, alpha=0.0125,
                            edgecolors='none')
        grid[1].set_aspect('equal', 'datalim')
        grid[1].set_xlabel(r'$x_0$')
        grid[1].set_ylabel(r'$x_1$')
        grid[1].set_title(r'$\mathrm{(b)\enspace Cluster\enspace Embedding (E)}$', y=1.02)
        grid[3].set_aspect('equal', 'datalim')
        grid[3].set_xlabel(r'$x_0$')
        grid[3].set_ylabel(r'$x_1$')
        grid[3].set_title(r'$\mathrm{(d)\enspace Cluster\enspace Embedding (M)}$', y=1.02)
        grid[5].set_aspect('equal', 'datalim')
        grid[5].set_xlabel(r'$x_0$')
        grid[5].set_ylabel(r'$x_1$')
        grid[5].set_title(r'$\mathrm{(f)\enspace Cluster\enspace Embedding (T)}$', y=1.02)
        # cbar = grid[0].cax.colorbar(cbd)
        # cbar.solids.set(alpha=1)
        # grid[0].cax.toggle_label(True)
        fig.savefig(OUTPREF+'.uemb.png')


    def plot_uph():
        ''' plot of unsupervised phase diagram '''
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.plot(UITRANS, np.arange(UNH), color='yellow')
        ax.plot(UICVAL, UICDOM, color='yellow', linestyle='--')
        ax.imshow(rgb_intens(UPREDB), aspect='equal', interpolation='none', origin='lower', cmap=CM)
        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
        plt.xticks(np.arange(UNT), np.round(UT, 2), rotation=-60)
        plt.yticks(np.arange(UNH), np.round(UH, 2))
        plt.xlabel('T')
        plt.ylabel('H')
        plt.title('Ising Model Phase Diagram')
        fig.savefig(OUTPREF+'.uph.png')


    def plot_sph():
        ''' plot of supervised phase diagram '''
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.plot(SITRANS, np.arange(SNH), color='yellow')
        ax.plot(SICVAL, SICDOM, color='yellow', linestyle='--')
        ax.imshow(rgb_intens(SPROBM), aspect='equal', interpolation='none', origin='lower', cmap=CM)
        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
        plt.xticks(np.arange(SNT), np.round(ST, 2), rotation=-60)
        plt.yticks(np.arange(SNH), np.round(SH, 2))
        plt.xlabel('T')
        plt.ylabel('H')
        plt.title('Ising Model Phase Diagram')
        fig.savefig(OUTPREF+'.sph.png')

    plot_uemb()
    plot_uph()
    plot_sph()

    if VERBOSE:
        print('plots saved')
        print(66*'-')