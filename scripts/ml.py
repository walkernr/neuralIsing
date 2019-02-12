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
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from scipy.odr import ODR, Model, RealData


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', help='verbose output', action='store_true')
    parser.add_argument('-pt', '--plot', help='plot results', action='store_true')
    parser.add_argument('-p', '--parallel', help='parallel run', action='store_true')
    parser.add_argument('-nt', '--threads', help='number of threads',
                        type=int, default=16)
    parser.add_argument('-n', '--name', help='simulation name',
                            type=str, default='ising_init')
    parser.add_argument('-ls', '--lattice_size', help='lattice size',
                        type=int, default=32)
    parser.add_argument('-ui', '--unsuper_interval', help='interval for selecting phase points (unsupervised)',
                        type=int, default=1)
    parser.add_argument('-un', '--unsuper_samples', help='number of samples per phase point (unsupervised)',
                        type=int, default=32)
    parser.add_argument('-si', '--super_interval', help='interval for selecting phase points (supervised)',
                        type=int, default=1)
    parser.add_argument('-sn', '--super_samples', help='number of samples per phase point (supervised)',
                        type=int, default=256)
    parser.add_argument('-sc', '--scaler', help='feature scaler',
                        type=str, default='tanh')
    parser.add_argument('-ld', '--latent_dimension', help='latent dimension of the variational autoencoder',
                        type=int, default=4)
    parser.add_argument('-rd', '--manifold', help='manifold learning method',
                        type=str, default='tsne')
    parser.add_argument('-ed', '--embed_dimension', help='number of embedded dimensions',
                        type=int, default=2)
    parser.add_argument('-cl', '--clustering', help='clustering method',
                        type=str, default='kmeans')
    parser.add_argument('-nc', '--clusters', help='number of clusters',
                        type=int, default=3)
    parser.add_argument('-bk', '--backend', help='keras backend',
                        type=str, default='tensorflow')
    parser.add_argument('-ep', '--epochs', help='number of epochs',
                        type=int, default=4)
    parser.add_argument('-lr', '--learning_rate', help='learning rate for neural network',
                        type=float, default=1e-3)
    parser.add_argument('-sd', '--random_seed', help='random seed for sample selection and learning',
                        type=int, default=256)
    args = parser.parse_args()
    return (args.verbose, args.plot, args.parallel, args.threads, args.name, args.lattice_size,
            args.unsuper_interval, args.unsuper_samples, args.super_interval, args.super_samples,
            args.scaler, args.latent_dimension, args.manifold, args.embed_dimension, args.clustering,
            args.clusters, args.backend, args.epochs, args.learning_rate, args.random_seed)


def write_specs():
    if VERBOSE:
        print(66*'-')
        print('input summary')
        print(66*'-')
        print('plot:                      %d' % PLOT)
        print('parallel:                  %d' % PARALLEL)
        print('threads:                   %d' % THREADS)
        print('name:                      %s' % NAME)
        print('lattice size:              %s' % N)
        print('random seed:               %d' % SEED)
        print('unsuper interval:          %d' % UNI)
        print('unsuper samples:           %d' % UNS)
        print('super interval:            %d' % SNI)
        print('super samples:             %d' % SNS)
        print('scaler:                    %s' % SCLR)
        print('latent dimension:          %d' % LD)
        print('manifold learning:         %s' % MNFLD)
        print('embedded projections:      %d' % ED)
        print('clustering:                %s' % CLST)
        print('clusters:                  %d' % NC)
        print('backend:                   %s' % BACKEND)
        print('network:                   %s' % 'cnn2d')
        print('epochs:                    %d' % EP)
        print('learning rate:             %.2e' % LR)
        print('fitting function:          %s' % 'logistic')
        print(66*'-')
    with open(OUTPREF+'.out', 'w') as out:
        out.write('# ' + 66*'-' + '\n')
        out.write('# input summary\n')
        out.write('# ' + 66*'-' + '\n')
        out.write('# plot:                      %d\n' % PLOT)
        out.write('# parallel:                  %d\n' % PARALLEL)
        out.write('# threads:                   %d\n' % THREADS)
        out.write('# name:                      %s\n' % NAME)
        out.write('# lattice size:              %s\n' % N)
        out.write('# random seed:               %d\n' % SEED)
        out.write('# unsuper interval:          %d\n' % UNI)
        out.write('# unsuper samples:           %d\n' % UNS)
        out.write('# super interval:            %d\n' % SNI)
        out.write('# super samples:             %d\n' % SNS)
        out.write('# scaler:                    %s\n' % SCLR)
        out.write('# latent dimension:          %d\n' % LD)
        out.write('# manifold learning:         %s\n' % MNFLD)
        out.write('# embedded dimension:        %d\n' % ED)
        out.write('# clustering:                %s\n' % CLST)
        out.write('# clusters:                  %d\n' % NC)
        out.write('# backend:                   %s\n' % BACKEND)
        out.write('# network:                   %s\n' % 'cnn2d')
        out.write('# epochs:                    %d\n' % EP)
        out.write('# learning rate:             %.2e\n' % LR)
        out.write('# fitting function:          %s\n' % 'logistic')


def sampling(beta):
    z_mean, z_log_var = beta
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean+K.exp(0.5*z_log_var)*epsilon


def build_variational_autoencoder():
    # encoder layers
    input = Input(shape=(N, N, 1), name='encoder_input')
    conv0 = Conv2D(filters=32, kernel_size=3, activation='relu',
                   kernel_initializer='he_normal', padding='same', strides=1)(input)
    conv1 = Conv2D(filters=64, kernel_size=3, activation='relu',
                   kernel_initializer='he_normal', padding='same', strides=2)(conv0)
    shape = K.int_shape(conv1)
    fconv1 = Flatten()(conv1)
    d0 = Dense(512, activation='relu')(fconv1)
    z_mean = Dense(LD, name='z_mean')(d0)
    z_log_var = Dense(LD, name='z_log_var')(d0)
    z = Lambda(sampling, output_shape=(LD,), name='z')([z_mean, z_log_var])
    # construct encoder
    encoder = Model(input, [z_mean, z_log_var, z], name='encoder')
    if VERBOSE:
        encoder.summary()
    # decoder layers
    latent_input = Input(shape=(LD,), name='z_sampling')
    d1 = Dense(np.prod(shape[1:]), activation='relu')(latent_input)
    rd1 = Reshape(shape[1:])(d1)
    convt0 = Conv2DTranspose(filters=64, kernel_size=3, activation='relu',
                             kernel_initializer='he_normal', padding='same', strides=2)(rd1)
    convt1 = Conv2DTranspose(filters=32, kernel_size=3, activation='relu',
                             kernel_initializer='he_normal', padding='same', strides=1)(convt0)
    output = Conv2DTranspose(filters=1, kernel_size=3, activation='sigmoid',
                             kernel_initializer='he_normal', padding='same', name='decoder_output')(convt1)
    # construct decoder
    decoder = Model(latent_input, output, name='decoder')
    if VERBOSE:
        decoder.summary()
    # construct vae
    output = decoder(encoder(input)[2])
    vae = Model(input, output, name='vae_mlp')
    # vae loss
    reconstruction_loss = N*N*binary_crossentropy(K.flatten(input), K.flatten(output))
    kl_loss = -0.5*K.sum(1+z_log_var-K.square(z_mean)-K.exp(z_log_var), axis=-1)
    vae_loss = K.mean(reconstruction_loss+kl_loss)
    vae.add_loss(vae_loss)
    # compile vae
    nadam = Nadam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    vae.compile(optimizer=nadam) # , metric=['accuracy'])
    # return vae networks
    return encoder, decoder, vae


def random_selection(dmp, dat, intrvl, ns):
    rdmp = dmp[::intrvl, ::intrvl]
    rdat = dat[::intrvl, ::intrvl]
    nh, nt, _, _, _ = rdmp.shape
    idat = np.zeros((nh, nt, ns), dtype=np.uint16)
    for i in tqdm(range(nh), disable=not VERBOSE):
        for j in tqdm(range(nh), disable=not VERBOSE):
                idat[i, j] = np.random.permutation(dat[i, j].shape[0])[:ns]
    sldmp = np.array([[rdmp[i, j, idat[i, j], :] for j in range(nt)] for i in range(nh)])
    sldat = np.array([[rdat[i, j, idat[i, j], :] for j in range(nt)] for i in range(nh)])
    return sldmp, sldat


def inlier_selection(dmp, dat, intrvl, ns):
    rdmp = dmp[::intrvl, ::intrvl]
    rdat = dat[::intrvl, ::intrvl]
    nh, nt, _, _ = rdmp.shape
    lof = LocalOutlierFactor(contamination='auto', n_jobs=THREADS)
    idat = np.zeros((nh, nt, ns), dtype=np.uint16)
    for i in tqdm(range(nh), disable=not VERBOSE):
        for j in tqdm(range(nh), disable=not VERBOSE):
                fpred = lof.fit_predict(rdmp[i, j])
                try:
                    idat[i, j] = np.random.choice(np.where(fpred==1)[0], size=ns, replace=False)
                except:
                    idat[i, j] = np.argsort(lof.negative_outlier_factor_)[:ns]
    sldmp = np.array([[rdmp[i, j, idat[i, j], :] for j in range(nt)] for i in range(nh)])
    sldat = np.array([[rdat[i, j, idat[i, j], :] for j in range(nt)] for i in range(nh)])
    return sldmp, sldat


if __name__ == '__main__':
    # parse command line arguments
    (VERBOSE, PLOT, PARALLEL, THREADS, NAME, N,
     UNI, UNS, SNI, SNS,
     SCLR, LD, MNFLD, ED, CLST, NC,
     BACKEND, EP, LR, SEED) = parse_args()
    CWD = os.getcwd()
    OUTPREF = CWD+'/%s.%d.%d.%d.%s.cnn2d.%d.%d.%.0e.%d.%d.%s.%d.%s.%d.%d' % \
              (NAME, N, SNI, SNS, SCLR, LD, EP, LR, UNI, UNS, MNFLD, ED, CLST, NC, SEED)
    write_specs()

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
    else:
        THREADS = 1
    from keras.models import Model
    from keras.layers import (Input, Lambda, Dense, Conv2D, Conv2DTranspose,
                              MaxPooling2D, Dropout, Flatten, Reshape)
    from keras.losses import binary_crossentropy, categorical_crossentropy
    from keras.optimizers import Nadam
    from keras.callbacks import History
    from keras import backend as K
    if PLOT:
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import ImageGrid
        if ED == 3:
            from mpl_toolkits.mplot3d import Axes3D
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

    try:
        CDMP = np.load(CWD+'/%s.%d.%d.%d.%d.cdmp.npy' % (NAME, N, SNI, SNS, SEED))
        CDAT = np.load(CWD+'/%s.%d.%d.%d.%d.cdat.npy' % (NAME, N, SNI, SNS, SEED))
        if VERBOSE:
            print('selected classification data loaded from file')
    except:
        DAT = np.load(CWD+'/%s.%d.dat.npy' % (NAME, N))
        DMP = np.load(CWD+'/%s.%d.dmp.npy' % (NAME, N))
        if VERBOSE:
            print('data loaded from file')
        CDMP, CDAT = random_selection(DMP, DAT, SNI, SNS)
        np.save(CWD+'/%s.%d.%d.%d.%d.cdmp.npy' % (NAME, N, SNI, SNS, SEED), CDMP)
        np.save(CWD+'/%s.%d.%d.%d.%d.cdat.npy' % (NAME, N, SNI, SNS, SEED), CDAT)
        if VERBOSE:
            print('selected classification data computed')
    CH = np.load(CWD+'/%s.%d.h.npy' % (NAME, N))[::SNI]
    CT = np.load(CWD+'/%s.%d.t.npy' % (NAME, N))[::SNI]
    SNT, SNH = CH.size, CT.size

    # scaler dictionary
    SCLRS = {'minmax':MinMaxScaler(feature_range=(0, 1)),
             'standard':StandardScaler(),
             'robust':RobustScaler(),
             'tanh':TanhScaler()}

    try:
        SCDMP = np.load(CWD+'/%s.%d.%d.%d.%s.%d.scdmp.npy' \
                        % (NAME, N, SNI, SNS, SCLR, SEED)).reshape(SNH*SNT*SNS, N, N)
        if VERBOSE:
            print('scaled selected classification data loaded from file')
    except:
        SCDMP = SCLRS[SCLR].fit_transform(CDMP.reshape(SNH*SNT*SNS, N*N)).reshape(SNH*SNT*SNS, N, N)
        np.save(CWD+'/%s.%d.%d.%d.%s.%d.scdmp.npy' % (NAME, N, SNI, SNS, SCLR, SEED), SCDMP.reshape(SNH, SNT, SNS, N, N))
        if VERBOSE:
            print('scaled selected classification data computed')

    ENC, DEC, VAE = build_variational_autoencoder()

    try:
        VAE.load_weights(CWD+'/%s.%d.%d.%d.%s.cnn2d.%d.%d.%.0e.%d.vae.wt.h5' \
                         % (NAME, N, SNI, SNS, SCLR, LD, EP, LR, SEED), by_name=True)
        LOSS = np.load(CWD+'/%s.%d.%d.%d.%s.cnn2d.%d.%d.%.0e.%d.vae.loss.npy' \
                       % (NAME, N, SNI, SNS, SCLR, LD, EP, LR, SEED))
        # ACC = np.load(CWD+'/%s.%d.%d.%d.%s.cnn2d.%d.%d.%.0e.%d.vae.acc.npy' \
        #               % (NAME, N, SNI, SNS, SCLR, LD, EP, LR, SEED))
        if VERBOSE:
            print('variational autoencoder trained weights loaded from file')
    except:
        VAE.fit(SCDMP[:, :, :, np.newaxis], epochs=EP, batch_size=SNS,
                shuffle=True, verbose=VERBOSE, callbacks=[History()])
        LOSS = VAE.model.history.history['loss']
        # ACC = VAE.model.history.history['acc']
        VAE.save_weights(CWD+'/%s.%d.%d.%d.%s.cnn2d.%d.%d.%.0e.%d.vae.wt.h5' \
                         % (NAME, N, SNI, SNS, SCLR, LD, EP, LR, SEED))
        np.save(CWD+'/%s.%d.%d.%d.%s.cnn2d.%d.%d.%.0e.%d.vae.loss.npy' \
                % (NAME, N, SNI, SNS, SCLR, LD, EP, LR, SEED), LOSS)
        # np.save(CWD+'/%s.%d.%d.%d.%s.cnn2d.%d.%d.%.0e.%d.vae.acc.npy' \
        #         % (NAME, N, SNI, SNS, SCLR, LD, EP, LR, SEED), ACC)

        if VERBOSE:
            print('variational autoencoder weights trained')

    try:
        ZENC = np.load(CWD+'/%s.%d.%d.%d.%s.cnn2d.%d.%d.%.0e.%d.zenc.npy'
                       % (NAME, N, SNI, SNS, SCLR, LD, EP, LR, SEED)).reshape(SNH*SNT*SNS, LD)
        if VERBOSE:
            print('z encodings loaded from file')
    except:
        ZENC = np.swapaxes(np.array(ENC.predict(SCDMP[:, :, :, np.newaxis]))[2], 0, 1).reshape(SNH*SNT*SNS, LD)
        np.save(CWD+'/%s.%d.%d.%d.%s.cnn2d.%d.%d.%.0e.%d.zenc.npy' % (NAME, N, SNI, SNS, SCLR, LD, EP, LR, SEED),
                ZENC.reshape(SNH, SNT, SNS, LD))
        if VERBOSE:
            print('z encodings computed')

    try:
        SLZENC = np.load(CWD+'/%s.%d.%d.%d.%s.cnn2d.%d.%d.%.0e.%d.%d.%d.slzenc.npy' \
                         % (NAME, N, SNI, SNS, SCLR, LD, EP, LR, UNI, UNS, SEED))
        SLDAT = np.load(CWD+'/%s.%d.%d.%d.%s.cnn2d.%d.%d.%.0e.%d.%d.%d.sldat.npy' \
                        % (NAME, N, SNI, SNS, SCLR, LD, EP, LR, UNI, UNS, SEED))
        if VERBOSE:
            print('inlier selected z encoding loaded from file')
    except:
        pass
        SLZENC, SLDAT = inlier_selection(ZENC.reshape(SNH, SNT, SNS, LD), CDAT, UNI, UNS)
        np.save(CWD+'/%s.%d.%d.%d.%s.cnn2d.%d.%d.%.0e.%d.%d.%d.slzenc.npy' \
                % (NAME, N, SNI, SNS, SCLR, LD, EP, LR, UNI, UNS, SEED), SLZENC)
        np.save(CWD+'/%s.%d.%d.%d.%s.cnn2d.%d.%d.%.0e.%d.%d.%d.sldat.npy' \
                % (NAME, N, SNI, SNS, SCLR, LD, EP, LR, UNI, UNS, SEED), SLDAT)
        if VERBOSE:
            print('inlier selected z encoding computed')

    UH, UT = CH[::UNI], CT[::UNI]
    UNH, UNT = UH.size, UT.size
    SLES = SLDAT[:, :, :, 0]
    SLMS = SLDAT[:, :, :, 1]

    # reduction dictionary
    MNFLDS = {'pca':PCA(n_components=ED),
              'kpca':KernelPCA(n_components=ED, n_jobs=THREADS),
              'isomap':Isomap(n_components=ED, n_jobs=THREADS),
              'lle':LocallyLinearEmbedding(n_components=ED, n_jobs=THREADS),
              'tsne':TSNE(n_components=ED, perplexity=UNS,
                          early_exaggeration=24, learning_rate=200, n_iter=1000,
                          verbose=VERBOSE, n_jobs=THREADS,
                          init=PCA(n_components=ED).fit_transform(SLZENC.reshape(UNH*UNT*UNS, LD)))}

    try:
        MSLZENC = np.load(CWD+'/%s.%d.%d.%d.%s.cnn2d.%d.%d.%.0e.%d.%d.%s.%d.%d.mslzenc.npy' \
                          % (NAME, N, SNI, SNS, SCLR, LD, EP, LR, UNI, UNS, MNFLD, ED, SEED))
        if VERBOSE:
            print('inlier selected z encoding manifold loaded from file')
    except:
        MSLZENC = MNFLDS[MNFLD].fit_transform(SLZENC.reshape(UNH*UNT*UNS, LD))
        np.save(CWD+'/%s.%d.%d.%d.%s.cnn2d.%d.%d.%.0e.%d.%d.%s.%d.%d.mslzenc.npy' \
                % (NAME, N, SNI, SNS, SCLR, LD, EP, LR, UNI, UNS, MNFLD, ED, SEED), MSLZENC)
        if VERBOSE:
            print('inlier selected z encoding manifold computed')

    # clustering dictionary
    CLSTS = {'agglomerative': AgglomerativeClustering(n_clusters=NC, linkage='ward'),
             'kmeans': KMeans(n_jobs=THREADS, n_clusters=NC, init='k-means++'),
             'spectral': SpectralClustering(n_jobs=THREADS, n_clusters=NC, eigen_solver='amg')}
    try:
        CLMSLZENC = np.load(CWD+'/%s.%d.%d.%d.%s.cnn2d.%d.%d.%.0e.%d.%d.%s.%d.%s.%d.%d.clmslzenc.npy' \
                            % (NAME, N, SNI, SNS, SCLR, LD, EP, LR, UNI, UNS, MNFLD, ED, CLST, NC, SEED))
        if VERBOSE:
            print('inlier selected z encoding manifold clustering loaded from file')
    except:
        CLMSLZENC = CLSTS[CLST].fit_predict(MSLZENC)
        CLMM = np.array([np.mean(SLMS.reshape(UNH*UNT*UNS)[CLMSLZENC == i]) for i in range(NC)])
        ICLMM = np.argsort(CLMM)
        for i in range(NC):
            CLMSLZENC[CLMSLZENC == ICLMM[i]] = i+NC
        CLMSLZENC -= NC
        np.save(CWD+'/%s.%d.%d.%d.%s.cnn2d.%d.%d.%.0e.%d.%d.%s.%d.%s.%d.%d.clmslzenc.npy' \
                % (NAME, N, SNI, SNS, SCLR, LD, EP, LR, UNI, UNS, MNFLD, ED, CLST, NC, SEED), CLMSLZENC)
        if VERBOSE:
            print('inlier selected z encoding manifold clustering computed')

    CLMM = np.array([np.mean(SLMS.reshape(UNH*UNT*UNS)[CLMSLZENC == i]) for i in range(NC)])
    # make this better
    if NC > NPH:
        CLMSLZENC[(CLMSLZENC != 0) & (CLMSLZENC != NC-1)] = 1
        CLMSLZENC[CLMSLZENC == NC-1] = NPH-1
        CLMM = np.array([np.mean(SLMS.reshape(UNH*UNT*UNS)[CLMSLZENC == i]) for i in range(NPH)])
    CLBMSLZENC = np.array([[np.bincount(CLMSLZENC.reshape(UNH, UNT, UNS)[i, j], minlength=NPH) for j in range(UNT)] for i in range(UNH)])/UNS

    fig = plt.figure()
    ax = fig.add_subplot(111)
    if ED == 3:
        ax.scatter(MSLZENC[:, 0], MSLZENC[:, 1], MSLZENC[:, 2], c=SLMS.reshape(-1), cmap=plt.get_cmap('plasma'),
                   s=128, alpha=0.15, edgecolors='k')
    if ED == 2:
        ax.scatter(MSLZENC[:, 0], MSLZENC[:, 1], c=SLMS.reshape(-1), cmap=plt.get_cmap('plasma'),
                   s=128, alpha=0.15, edgecolors='k')
    fig.savefig(OUTPREF+'.vae.emb.smpl.png')

    fig = plt.figure()
    if ED == 3:
        ax = fig.add_subplot(111, projection='3d')
    elif ED == 2:
        ax = fig.add_subplot(111)
    for i in range(NPH):
        if ED == 3:
            ax.scatter(MSLZENC[CLMSLZENC == i, 0], MSLZENC[CLMSLZENC == i, 1], MSLZENC[CLMSLZENC == i, 2],
                       c=np.array(CM(SCALE(CLMM[i], SLMS.reshape(-1))))[np.newaxis, :],
                       s=128, alpha=0.15, edgecolors='k')
        if ED == 2:
            ax.scatter(MSLZENC[CLMSLZENC == i, 0], MSLZENC[CLMSLZENC == i, 1],
                       c=np.array(CM(SCALE(CLMM[i], SLMS.reshape(-1))))[np.newaxis, :],
                       s=128, alpha=0.15, edgecolors='k')
    fig.savefig(OUTPREF+'.vae.emb.clst.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.imshow(CLBMSLZENC, aspect='equal', interpolation='none', origin='lower', cmap=CM)
    ax.grid(which='minor', axis='both', linestyle='-', color='k', linewidth=1)
    ax.set_xticks(np.arange(UT.size), minor=True)
    ax.set_yticks(np.arange(UH.size), minor=True)
    plt.xticks(np.arange(CT.size)[::(2*UNI)], np.round(CT, 2)[::(2*UNI)], rotation=-60)
    plt.yticks(np.arange(CH.size)[::(2*UNI)], np.round(CH, 2)[::(2*UNI)])
    plt.xlabel('T')
    plt.ylabel('H')
    plt.title('Ising Model Phase Diagram')
    fig.savefig(OUTPREF+'.vae.diag.phase.png')
