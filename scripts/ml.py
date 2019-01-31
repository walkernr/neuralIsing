# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 22:07:13 2019

@author: Nicholas
"""

import argparse
import os
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
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
                    type=int, default=16)
PARSER.add_argument('-sc', '--scaler', help='feature scaler',
                    type=str, default='tanh')
PARSER.add_argument('-rd', '--reduction', help='supervised dimension reduction method',
                    type=str, default='tsne')
PARSER.add_argument('-np', '--projections', help='number of embedding projections',
                    type=int, default=2)
PARSER.add_argument('-cl', '--clustering', help='clustering method',
                    type=str, default='spectral')
PARSER.add_argument('-nc', '--clusters', help='number of clusters',
                    type=int, default=3)
PARSER.add_argument('-bk', '--backend', help='keras backend',
                    type=str, default='tensorflow')
PARSER.add_argument('-ep', '--epochs', help='number of epochs',
                    type=int, default=16)
PARSER.add_argument('-lr', '--learning_rate', help='learning rate for neural network',
                    type=float, default=1e-3)

# parse arguments
ARGS = PARSER.parse_args()
# run specifications
VERBOSE = ARGS.verbose
PLOT = ARGS.plot
PARALLEL = ARGS.parallel
THREADS = ARGS.threads
NAME = ARGS.name
N = ARGS.lattice_size
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
from keras.layers import Conv2D, AveragePooling2D, Dropout, Dense
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
OUTPREF = CWD+'/%s.%d.%s.%s.%d.%s.%d.cnn2d.%d.%.0e.logistic' \
          % (NAME, N, SCLR, RDCN, NP, CLST, NC, EP, LR)
with open(OUTPREF+'.out', 'w') as out:
    out.write('# ' + 66*'-' + '\n')
    out.write('# input summary\n')
    out.write('# ' + 66*'-' + '\n')
    out.write('# plot:                      %d\n' % PLOT)
    out.write('# parallel:                  %d\n' % PARALLEL)
    out.write('# threads:                   %d\n' % THREADS)
    out.write('# name:                      %s\n' % NAME)
    out.write('# lattice size:              %s\n' % N)
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

EPS = 0.025 # np.finfo(np.float32).eps
# load data
SMAX = 32
H = pickle.load(open(CWD+'/%s.%d.h.pickle' % (NAME, N), 'rb'))
T = pickle.load(open(CWD+'/%s.%d.t.pickle' % (NAME, N), 'rb'))
NH, NT = H.size, T.size
DAT = pickle.load(open(CWD+'/%s.%d.dat.pickle' % (NAME, N), 'rb'))
ES = DAT[:, :, -SMAX:, 0].reshape(NH, NT, -1)
MS = DAT[:, :, -SMAX:, 1].reshape(NH, NT, -1)
UDAT = pickle.load(open(CWD+'/%s.%d.dmp.pickle' % (NAME, N), 'rb'))[:, :, -SMAX:, :, :].reshape(NH, NT, SMAX, N*N)
SDAT = pickle.load(open(CWD+'/%s.%d.dmp.pickle' % (NAME, N), 'rb'))[:, :, -SMAX:, :, :]
# data shape
_, _, UNS, UNF = UDAT.shape
_, _, SNS, SNF0, SNF1 = SDAT.shape
HS = np.array([H[i]*np.ones((NT, UNS)) for i in range(NH)])
TS = np.ones((NH, NT, UNS))*np.array([T[i]*np.ones(UNS) for i in range(NT)])[np.newaxis, :, :]
del DAT

if PLOT:
    CM = plt.get_cmap('plasma')
    SCALE = lambda a, b: (a-np.min(b))/(np.max(b)-np.min(b))
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


# odr fitting
def odr_fit(mpred, spred):
    ''' performs orthogonal distance regression '''
    dat = RealData(R, mpred, EPS*np.ones(len(R)), spred+EPS)
    mod = Model(logistic)
    odr = ODR(dat, mod, FITG)
    odr.set_job(fit_type=0)
    fit = odr.run()
    popt = fit.beta
    perr = fit.sd_beta
    trans = popt[1]
    cerr = perr[1]
    ndom = 256
    fdom = np.linspace(np.min(R), np.max(R), ndom)
    fval = logistic(popt, fdom)
    return trans, cerr, fdom, fval

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
    model = Sequential([Conv2D(filters=int(SNF0)/2, kernel_size=(3, 3), activation='relu',
                               kernel_initializer='he_normal',
                               padding='valid', strides=1, input_shape=(SNF0, SNF1, 1)),
                        Conv2D(filters=int(SNF0)/2, kernel_size=(3, 3), activation='relu',
                               kernel_initializer='he_normal',
                               padding='valid', strides=1),
                        AveragePooling2D(pool_size=(2, 2)),
                        Dropout(rate=0.25),
                        Conv2D(filters=int(SNF0), kernel_size=(3, 3), activation='relu',
                               kernel_initializer='he_normal',
                               padding='valid', strides=1),
                        Conv2D(filters=int(SNF0), kernel_size=(3, 3), activation='relu',
                               kernel_initializer='he_normal',
                               padding='valid', strides=1),
                        AveragePooling2D(pool_size=(2, 2)),
                        Dropout(rate=0.25),
                        Flatten(),
                        Dense(units=SNF0, activation='relu'),
                        Dropout(rate=0.5),
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
    FUDOM = pickle.load(open(CWD+'/%s.%d.%s.fudat.k.pickle' % (NAME, N, SCLR), 'rb'))
    FUNF = FUDOM.size
    FUDAT = pickle.load(open(CWD+'/%s.%d.%s.fudat.pickle' % (NAME, N, SCLR), 'rb')).reshape(NH*NT*UNS, FUNF)
    FUDAT = np.concatenate((np.real(FUDAT), np.imag(FUDAT)), axis=1)
    SUDAT = pickle.load(open(CWD+'/%s.%d.%s.sudat.pickle' % (NAME, N, SCLR), 'rb')).reshape(NH*NT*UNS, 2*FUNF)
    if VERBOSE:
        print('scaled unsupervised data loaded from file')
except:
    FUDOM = np.fft.rfftfreq(UNF, 1)
    FUNF = FUDOM.size
    FUDAT = np.fft.rfft(UDAT.reshape(NH*NT*UNS, UNF))
    pickle.dump(FUDOM, open(CWD+'/%s.%d.%s.fudat.k.pickle' % (NAME, N, SCLR), 'wb'))
    pickle.dump(FUDAT.reshape(NH, NT, UNS, FUNF), open(CWD+'/%s.%d.%s.fudat.pickle' % (NAME, N, SCLR), 'wb'))
    FUDAT = np.concatenate((np.real(FUDAT), np.imag(FUDAT)), axis=1)
    SUDAT = SCLRS[SCLR].fit_transform(FUDAT)
    pickle.dump(SUDAT.reshape(NH, NT, UNS, 2*FUNF), open(CWD+'/%s.%d.%s.sudat.pickle' % (NAME, N, SCLR), 'wb'))
    if VERBOSE:
        print('unsupervised data scaled')
if VERBOSE:
    print(66*'-')

# pca reduce unsupervised data
try:
    EVAR = pickle.load(open(CWD+'/%s.%d.%s.evar.pickle' % (NAME, N, SCLR), 'rb'))
    PUDAT = pickle.load(open(CWD+'/%s.%d.%s.pudat.pickle' % (NAME, N, SCLR), 'rb')).reshape(NT*NH*UNS, len(EVAR))
    PCOMP = pickle.load(open(CWD+'/%s.%d.%s.pcomp.pickle' % (NAME, N, SCLR), 'rb'))
    if VERBOSE:
        print('pca reduced unsupervised data loaded from file')
except:
    PUDAT = RDCNS['pca'].fit_transform(SUDAT)
    EVAR = RDCNS['pca'].explained_variance_ratio_
    PCOMP = RDCNS['pca'].components_
    pickle.dump(PUDAT.reshape(NT, NH, UNS, len(EVAR)),
                open(CWD+'/%s.%d.%s.pudat.pickle' % (NAME, N, SCLR), 'wb'))
    pickle.dump(EVAR, open(CWD+'/%s.%d.%s.evar.pickle' % (NAME, N, SCLR), 'wb'))
    pickle.dump(PCOMP, open(CWD+'/%s.%d.%s.pcomp.pickle' % (NAME, N, SCLR), 'wb'))
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
    RUDAT = pickle.load(open(CWD+'/%s.%d.%s.%s.%d.rudat.pickle' \
                             % (NAME, N, SCLR, RDCN, NP), 'rb')).reshape(NH*NT*UNS, NP)
    if VERBOSE:
        print('nonlinearly reduced unsupervised data loaded from file')
except:
    if RDCN not in ('none', 'pca'):
        RUDAT = RDCNS[RDCN].fit_transform(PUDAT)
        pickle.dump(RUDAT.reshape(NH, NT, UNS, NP), open(CWD+'/%s.%d.%s.%s.%d.rudat.pickle' \
                                                         % (NAME, N, SCLR, RDCN, NP), 'wb'))
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
    UPRED = pickle.load(open(CWD+'/%s.%d.%s.%s.%d.%s.%d.upred.pickle' \
                             % (NAME, N, SCLR, RDCN, NP, CLST, NC), 'rb')).reshape(NH*NT*UNS)
    if VERBOSE:
        print('clustered unsupervised data loaded from file')
except:
    UPRED = CLSTS[CLST].fit_predict(RUDAT)
    UCTM = np.array([np.mean(TS.reshape(NH*NT*UNS)[UPRED == i]) for i in range(NC)])
    IUCTM = np.argsort(UCTM)
    for i in range(NC):
        UPRED[UPRED == IUCTM[i]] = i+NC
    UPRED -= NC
    pickle.dump(UPRED.reshape(NH, NT, UNS), open(CWD+'/%s.%d.%s.%s.%d.%s.%d.upred.pickle' \
                                              % (NAME, N, SCLR, RDCN, NP, CLST, NC), 'wb'))
    if VERBOSE:
        print('unsupervised data clustered')
UCMM = np.array([np.mean(MS.reshape(NH*NT*UNS)[UPRED == i]) for i in range(NC)])
UCTM = np.array([np.mean(TS.reshape(NH*NT*UNS)[UPRED == i]) for i in range(NC)])
if NC == 4:
    IUCTM = np.argsort(UCTM)
    UPRED[UPRED == np.max(IUCTM[-2:])] = np.min(IUCTM[-2:])
    UCMM = np.array([np.mean(MS.reshape(NH*NT*UNS)[UPRED == i]) for i in range(NPH)])
    UCTM = np.array([np.mean(TS.reshape(NH*NT*UNS)[UPRED == i]) for i in range(NPH)])
UPREDB = np.array([[np.bincount(UPRED.reshape(NH, NT, UNS)[i, j], minlength=NPH) for j in range(NT)] for i in range(NH)])/UNS
UPREDC = np.array([[np.argmax(np.bincount(UPRED.reshape(NH, NT, UNS)[i, j])) for j in range(NT)] for i in range(NH)])
UPREDBCS = np.cumsum(UPREDB, -1)[:, :, 1:]
UPREDBCS[:, :, 1] -= UPREDBCS[:, :, 0]
PTRANS = np.argmin(np.std(UPREDBCS, -1), -1)
TRANS = T[PTRANS]

if VERBOSE:
    print(np.max([66, 12+8*NT])*'-')
    print('\t'+NT*'\t%0.2f' % tuple(T))
    print(np.max([66, 12+8*NT])*'-')
    for i in range(-1, -NH-1, -1):
        print('%0.2f\t' % H[i] + NT*'\t%d' % tuple(UPREDC[i]))
    print(np.max([66, 12+8*NT])*'-')
    print('h\tt')
    print(66*'-')
    for i in range(NH):
        print('%.2f\t%.2f' % (H[i], TRANS[i]))
    print(66*'-')

with open(OUTPREF+'.out', 'a') as out:
    out.write('# ' + np.max([66, 10+8*NC])*'-' + '\n')
    out.write('# unsupervised learning results\n')
    out.write('# ' + np.max([66, 12+8*NT])*'-' + '\n')
    out.write('# \t'+NT*'\t%0.2f' % tuple(T)+'\n')
    out.write('# ' + np.max([66, 12+8*NT])*'-' + '\n')
    for i in range(-1, -NH-1, -1):
        out.write('# %0.2f\t' % H[i] + NT*'\t%d' % tuple(UPREDC[i])+'\n')
    out.write('# ' + np.max([66, 12+8*NT])*'-' + '\n')
    out.write('# h\tt\n')
    out.write('# '+66*'-'+'\n')
    for i in range(NH):
        out.write('  %.2f\t%.2f\n' % (H[i], TRANS[i]))

# # scale supervised data
# try:
#     SSDAT = pickle.load(open(CWD+'/sga.%d.%s.ssdat.pickle' % (MI, SCLR), 'rb')).reshape(SND*SNS, SNF)
#     if VERBOSE:
#         print(66*'-')
#         print('scaled supervised data loaded from file')
# except:
#     SCLRS[SCLR].fit(np.real(SDAT.reshape(SND*SNS, SNF)[(UPRED == 0) | (UPRED == NC-1)]))
#     SSDAT = SCLRS[SCLR].transform(SDAT.reshape(SND*SNS, SNF))
#     pickle.dump(SSDAT.reshape(SND, SNS, SNF), open(CWD+'/sga.%d.%s.ssdat.pickle' \
#                                                    % (MI, SCLR), 'wb'))
#     if VERBOSE:
#         print(125*'-')
#         print('supervised data scaled')

# # fit neural network to training data and predict classification data
# try:
#     LOSS = pickle.load(open(CWD+'/sga.%d.%s.%s.%d.%s.%d.cnn1d.%d.%.0e.loss.pickle' \
#                             % (MI, SCLR, RDCN, NP, CLST, NC, EP, LR), 'rb'))
#     MAE = pickle.load(open(CWD+'/sga.%d.%s.%s.%d.%s.%d.cnn1d.%d.%.0e.mae.pickle' \
#                            % (MI, SCLR, RDCN, NP, CLST, NC, EP, LR), 'rb'))
#     ACC = pickle.load(open(CWD+'/sga.%d.%s.%s.%d.%s.%d.cnn1d.%d.%.0e.acc.pickle' \
#                            % (MI, SCLR, RDCN, NP, CLST, NC, EP, LR), 'rb'))
#     SPROB = pickle.load(open(CWD+'/sga.%d.%s.%s.%d.%s.%d.cnn1d.%d.%.0e.sprob.pickle' \
#                              % (MI, SCLR, RDCN, NP, CLST, NC, EP, LR), 'rb'))
#     if VERBOSE:
#         print(66*'-')
#         print('neural network fit loaded from file')
# except:
#     if VERBOSE:
#         print(125*'-')
#     # fit training data
#     LBLS = np.concatenate((np.zeros(np.sum(CUPRED[:, 0]), dtype=np.uint16),
#                         np.ones(np.sum(CUPRED[:, NC-1]), dtype=np.uint16)), 0)
#     NN.fit(SSDAT[(UPRED == 0) | (UPRED == NC-1), :, np.newaxis], LBLS)
#     LOSS = NN.model.history.history['loss']
#     MAE = NN.model.history.history['mean_absolute_error']
#     ACC = NN.model.history.history['acc']
#     pickle.dump(LOSS, open(CWD+'/sga.%d.%s.%s.%d.%s.%d.cnn1d.%d.%.0e.loss.pickle' \
#                            % (MI, SCLR, RDCN, NP, CLST, NC, EP, LR), 'wb'))
#     pickle.dump(MAE, open(CWD+'/sga.%d.%s.%s.%d.%s.%d.cnn1d.%d.%.0e.mae.pickle' \
#                           % (MI, SCLR, RDCN, NP, CLST, NC, EP, LR), 'wb'))
#     pickle.dump(ACC, open(CWD+'/sga.%d.%s.%s.%d.%s.%d.cnn1d.%d.%.0e.acc.pickle' \
#                           % (MI, SCLR, RDCN, NP, CLST, NC, EP, LR), 'wb'))
#     if VERBOSE:
#         print(125*'-')
#         print('neural network fitted to training data')
#         print(66*'-')
#     # predict classification data
#     SPROB = NN.predict_proba(SSDAT[:, :, np.newaxis])[:, 1].reshape(SND, SNS)
#     pickle.dump(SPROB, open(CWD+'/sga.%d.%s.%s.%d.%s.%d.cnn1d.%d.%.0e.sprob.pickle' \
#                             % (MI, SCLR, RDCN, NP, CLST, NC, EP, LR), 'wb'))

# MSPROB = np.mean(SPROB, 1)
# SSPROB = np.std(SPROB, 1)
# SPRED = SPROB.round()
# SCM = [np.mean(RS[SPRED.reshape(-1) == i]) for i in range(2)]

# # transition prediction
# FITG = (1.0, UTRANS)
# STRANS, SERR, SDOM, SVAL = odr_fit(MSPROB, SSPROB)
# pickle.dump(np.array([STRANS, SERR], dtype=np.float32),
#             open(CWD+'/sga.%d.%s.%s.%d.%s.%d.cnn1d.%d.%.0e.strans.pickle' \
#                  % (MI, SCLR, RDCN, NP, CLST, NC, EP, LR), 'wb'))

# if VERBOSE:
#     print(66*'-')
#     print('r\tave\tstd')
#     print(66*'-')
#     for i in range(SND):
#         print('%0.2f\t' % R[i] + 2*'%0.2f\t' % (MSPROB[i], SSPROB[i]))
#     print(66*'-')
#     print('neural network predicted classification data')
#     print(66*'-')
#     print('trans\t'+2*'%0.2f\t' % (STRANS, SERR))
#     print(66*'-')

# with open(OUTPREF+'.out', 'a') as out:
#     out.write('# ' + 66*'-' + '\n')
#     out.write('# supervised learning results\n')
#     out.write('# ' + 66*'-' + '\n')
#     out.write('# epoch\tloss\tmae\tacc\n')
#     out.write('# ' + 66*'-' + '\n')
#     for i in range(EP):
#         out.write('  %02d\t' % i + 3*'%0.4f\t' % (LOSS[i], MAE[i], ACC[i]) + '\n')
#     out.write('# ' + 66*'-' + '\n')
#     out.write('# r\tave\tstd\n')
#     out.write('# ' + 66*'-' + '\n')
#     for i in range(SND):
#         out.write('  %0.4f\t' % R[i] + 2*'%0.4f\t' % (MSPROB[i], SSPROB[i]) + '\n')
#     out.write('# ' + 66*'-' + '\n')
#     out.write('# transition\n')
#     out.write('# ' + 66*'-' + '\n')
#     out.write('  '+2*'%0.4f\t' % (STRANS, SERR) + '\n')
#     out.write('# ' + 66*'-'+'\n')

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
        grid[0].scatter(RUDAT[:, 0], RUDAT[:, 1], c=ES.reshape(NH*NT*UNS), cmap=CM, s=120, alpha=0.025, edgecolors='none')
        grid[0].set_aspect('equal', 'datalim')
        grid[0].set_xlabel(r'$x_0$')
        grid[0].set_ylabel(r'$x_1$')
        grid[0].set_title(r'$\mathrm{(a)\enspace Sample\enspace Embedding (E)}$', y=1.02)
        grid[2].scatter(RUDAT[:, 0], RUDAT[:, 1], c=MS.reshape(NH*NT*UNS), cmap=CM, s=120, alpha=0.025, edgecolors='none')
        grid[2].set_aspect('equal', 'datalim')
        grid[2].set_xlabel(r'$x_0$')
        grid[2].set_ylabel(r'$x_1$')
        grid[2].set_title(r'$\mathrm{(c)\enspace Sample\enspace Embedding (M)}$', y=1.02)
        grid[4].scatter(RUDAT[:, 0], RUDAT[:, 1], c=TS.reshape(NH*NT*UNS), cmap=CM, s=120, alpha=0.025, edgecolors='none')
        grid[4].set_aspect('equal', 'datalim')
        grid[4].set_xlabel(r'$x_0$')
        grid[4].set_ylabel(r'$x_1$')
        grid[4].set_title(r'$\mathrm{(e)\enspace Sample\enspace Embedding (T)}$', y=1.02)
        for j in range(NPH):
            # SCALE(np.mean(ES.reshape(-1)[UPRED == j]), ES)
            # SCALE(np.mean(MS.reshape(-1)[UPRED == j]), MS)
            grid[1].scatter(RUDAT[UPRED == j, 0], RUDAT[UPRED == j, 1],
                            c=np.array(CM(SCALE(np.mean(ES.reshape(-1)[UPRED == j]), ES.reshape(-1))))[np.newaxis, :], s=120, alpha=0.025,
                            edgecolors='none')
            grid[3].scatter(RUDAT[UPRED == j, 0], RUDAT[UPRED == j, 1],
                            c=np.array(CM(SCALE(np.mean(MS.reshape(-1)[UPRED == j]), MS.reshape(-1))))[np.newaxis, :], s=120, alpha=0.025,
                            edgecolors='none')
            grid[5].scatter(RUDAT[UPRED == j, 0], RUDAT[UPRED == j, 1],
                            c=np.array(CM(SCALE(np.mean(TS.reshape(-1)[UPRED == j]), TS.reshape(-1))))[np.newaxis, :], s=120, alpha=0.025,
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
        ax.plot(PTRANS, np.arange(NH), color='yellow')
        ax.imshow(UPREDB, aspect='equal', interpolation='none', origin='lower', cmap=CM)
        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
        plt.xticks(np.arange(32), np.round(T, 2), rotation=-60)
        plt.yticks(np.arange(32), np.round(H, 2))
        plt.xlabel('T')
        plt.ylabel('H')
        plt.title('Ising Model Phase Diagram')
        fig.savefig(OUTPREF+'.uph.png')


    def plot_spred():
        ''' plot of prediction curves '''
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.plot(SDOM, SVAL, color=CM(SCALE(STRANS)),
                label=r'$\mathrm{Phase\enspace Probability\enspace Curve}$')
        ax.axvline(STRANS, color=CM(SCALE(STRANS)), alpha=0.50)
        for j in range(2):
            serrb = STRANS+(-1)**(j+1)*SERR
            ax.axvline(serrb, color=CM(SCALE(serrb)), alpha=0.50, linestyle='--')
        ax.scatter(R, MSPROB, color=CM(SCALE(R)), s=240, edgecolors='none', marker='*')
        ax.text(STRANS+np.diff(R)[0], .1,
                r'$r_{\mathrm{supervised}} = %.4f \pm %.4f$' % (STRANS, SERR))
        ax.set_ylim(0.0, 1.0)
        for tick in ax.get_xticklabels():
            tick.set_rotation(16)
        scitxt = ax.yaxis.get_offset_text()
        scitxt.set_x(.025)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
        ax.set_xlabel(r'$\mathrm{r}$')
        ax.set_ylabel(r'$\mathrm{Probability}$')
        fig.savefig(OUTPREF+'.spred.png')


    def plot_semb():
        ''' plot of reduced sample space '''
        fig = plt.figure()
        grid = ImageGrid(fig, 111,
                         nrows_ncols=(1, 2),
                         axes_pad=2.0,
                         share_all=True,
                         cbar_location="right",
                         cbar_mode="single",
                         cbar_size="4%",
                         cbar_pad=0.4)
        for j in range(len(grid)):
            grid[j].spines['right'].set_visible(False)
            grid[j].spines['top'].set_visible(False)
            grid[j].xaxis.set_ticks_position('bottom')
            grid[j].yaxis.set_ticks_position('left')
        cbd = grid[0].scatter(RUDAT[:, 0], RUDAT[:, 1], c=RS, cmap=CM, s=120, alpha=0.05,
                              edgecolors='none')
        grid[0].set_aspect('equal', 'datalim')
        grid[0].set_xlabel(r'$x_0$')
        grid[0].set_ylabel(r'$x_1$')
        grid[0].set_title(r'$\mathrm{(a)\enspace Sample\enspace Embedding}$', y=1.02)
        for j in range(2):
            grid[1].scatter(RUDAT[SPRED.reshape(-1) == j, 0], RUDAT[SPRED.reshape(-1) == j, 1],
                            c=np.array(CM(SCALE(SCM[j])))[np.newaxis, :], s=120, alpha=0.05,
                            edgecolors='none')
        grid[1].set_aspect('equal', 'datalim')
        grid[1].set_xlabel(r'$x_0$')
        grid[1].set_ylabel(r'$x_1$')
        grid[1].set_title(r'$\mathrm{(b)\enspace Classification\enspace Embedding}$', y=1.02)
        cbar = grid[0].cax.colorbar(cbd)
        cbar.solids.set(alpha=1)
        grid[0].cax.toggle_label(True)
        fig.savefig(OUTPREF+'.semb.png')

    plot_uemb()
    plot_uph()
    # plot_spred()
    # plot_semb()

    if VERBOSE:
        print('plots saved')
        print(66*'-')