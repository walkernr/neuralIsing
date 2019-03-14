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
    parser.add_argument('-fft', '--fft', help='use fft of spin data', action='store_true')
    parser.add_argument('-ad', '--anomaly_detection', help='anomaly detection for embedding', action='store_true')
    parser.add_argument('-nt', '--threads', help='number of threads',
                        type=int, default=20)
    parser.add_argument('-n', '--name', help='simulation name',
                            type=str, default='ising_init')
    parser.add_argument('-ls', '--lattice_size', help='lattice size (side length)',
                        type=int, default=32)
    parser.add_argument('-ui', '--unsuper_interval', help='interval for selecting phase points (manifold)',
                        type=int, default=1)
    parser.add_argument('-un', '--unsuper_samples', help='number of samples per phase point (manifold)',
                        type=int, default=64)
    parser.add_argument('-si', '--super_interval', help='interval for selecting phase points (variational autoencoder)',
                        type=int, default=1)
    parser.add_argument('-sn', '--super_samples', help='number of samples per phase point (variational autoencoder)',
                        type=int, default=1024)
    parser.add_argument('-sc', '--scaler', help='feature scaler',
                        type=str, default='minmax')
    parser.add_argument('-ld', '--latent_dimension', help='latent dimension of the variational autoencoder',
                        type=int, default=4)
    parser.add_argument('-mf', '--manifold', help='manifold learning method',
                        type=str, default='tsne')
    parser.add_argument('-ed', '--embed_dimension', help='number of embedded dimensions in manifold',
                        type=int, default=2)
    parser.add_argument('-cl', '--clustering', help='clustering method',
                        type=str, default='dbscan')
    parser.add_argument('-nc', '--clusters', help='number of clusters (neighbor criterion eps for dbscan)',
                        type=float, default=1e-3)
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
    parser.add_argument('-ev', '--embedding_variables', help='variables for learning the embedding manifold (boolean string: zm, zlv, z)',
                        type=str, default='11')
    args = parser.parse_args()
    return (args.verbose, args.plot, args.parallel, args.gpu, args.fft, args.anomaly_detection, args.threads, args.name, args.lattice_size,
            args.unsuper_interval, args.unsuper_samples, args.super_interval, args.super_samples,
            args.scaler, args.latent_dimension, args.manifold, args.embed_dimension, args.clustering,
            args.clusters, args.backend, args.optimizer, args.loss,
            args.epochs, args.learning_rate, args.random_seed, args.embedding_variables)


def write_specs():
    if VERBOSE:
        print(100*'-')
        print('input summary')
        print(100*'-')
        print('plot:                      %d' % PLOT)
        print('parallel:                  %d' % PARALLEL)
        print('gpu:                       %d' % GPU)
        print('fft:                       %d' % FFT)
        print('anomaly detection:         %d' % AD)
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
        print('embedding vars:            %s' % EV)
        print('manifold learning:         %s' % MNFLD)
        print('embedded projections:      %d' % ED)
        print('clustering:                %s' % CLST)
        if CLST == 'dbscan':
            print('neighbor eps:              %.2e' % NC)
        else:
            print('clusters:                  %d' % NC)
        print('backend:                   %s' % BACKEND)
        print('network:                   %s' % 'cnn2d')
        print('optimizer:                 %s' % OPT)
        print('loss function:             %s' % LSS)
        print('epochs:                    %d' % EP)
        print('learning rate:             %.2e' % LR)
        print('fitting function:          %s' % 'logistic')
        print(100*'-')
    with open(OUTPREF+'.out', 'w') as out:
        out.write('# ' + 100*'-' + '\n')
        out.write('# input summary\n')
        out.write('# ' + 100*'-' + '\n')
        out.write('# plot:                      %d\n' % PLOT)
        out.write('# parallel:                  %d\n' % PARALLEL)
        out.write('# gpu:                       %d\n' % GPU)
        out.write('# fft:                       %d\n' % FFT)
        out.write('# anomaly detection:         %d\n' % AD)
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
        out.write('# embedding vars:            %s\n' % EV)
        out.write('# manifold learning:         %s\n' % MNFLD)
        out.write('# embedded dimension:        %d\n' % ED)
        out.write('# clustering:                %s\n' % CLST)
        if CLST == 'dbscan':
            out.write('# neighbor eps:              %.2e\n' % NC)
        else:
            out.write('# clusters:                  %d\n' % NC)
        out.write('# backend:                   %s\n' % BACKEND)
        out.write('# network:                   %s\n' % 'cnn2d')
        out.write('# optimizer:                 %s\n' % OPT)
        out.write('# loss function:             %s\n' % LSS)
        out.write('# epochs:                    %d\n' % EP)
        out.write('# learning rate:             %.2e\n' % LR)
        out.write('# fitting function:          %s\n' % 'logistic')
        out.write('# ' + 100*'-' + '\n')


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


def sampling(beta):
    z_mean, z_log_var = beta
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean+K.exp(0.5*z_log_var)*epsilon


def build_variational_autoencoder():
    if VERBOSE:
        print('building variational autoencoder network')
        print(100*'-')
    # encoder layers
    input = Input(shape=(N, N, NCH), name='encoder_input')
    conv0 = Conv2D(filters=32, kernel_size=3, activation='relu',
                   kernel_initializer='he_normal', padding='same', strides=1)(input)
    conv1 = Conv2D(filters=64, kernel_size=3, activation='relu',
                   kernel_initializer='he_normal', padding='same', strides=2)(conv0)
    conv2 = Conv2D(filters=32, kernel_size=3, activation='relu',
                   kernel_initializer='he_normal', padding='same', strides=1)(conv1)
    conv3 = Conv2D(filters=64, kernel_size=3, activation='relu',
                   kernel_initializer='he_normal', padding='same', strides=2)(conv2)
    shape = K.int_shape(conv3)
    fconv3 = Flatten()(conv3)
    d0 = Dense(1024, activation='relu')(fconv3)
    z_mean = Dense(LD, name='z_mean')(d0)
    z_log_var = Dense(LD, name='z_log_std')(d0) # more numerically stable to use log(var_z)
    z = Lambda(sampling, output_shape=(LD,), name='z')([z_mean, z_log_var])
    # construct encoder
    encoder = Model(input, [z_mean, z_log_var, z], name='encoder')
    if VERBOSE:
        print('encoder network summary')
        print(100*'-')
        encoder.summary()
        print(100*'-')
    # decoder layers
    latent_input = Input(shape=(LD,), name='z_sampling')
    d1 = Dense(np.prod(shape[1:]), activation='relu')(latent_input)
    rd1 = Reshape(shape[1:])(d1)
    convt0 = Conv2DTranspose(filters=64, kernel_size=3, activation='relu',
                             kernel_initializer='he_normal', padding='same', strides=2)(rd1)
    convt1 = Conv2DTranspose(filters=32, kernel_size=3, activation='relu',
                             kernel_initializer='he_normal', padding='same', strides=1)(convt0)
    convt2 = Conv2DTranspose(filters=64, kernel_size=3, activation='relu',
                             kernel_initializer='he_normal', padding='same', strides=2)(convt1)
    convt3 = Conv2DTranspose(filters=32, kernel_size=3, activation='relu',
                             kernel_initializer='he_normal', padding='same', strides=1)(convt2)
    output = Conv2DTranspose(filters=NCH, kernel_size=3, activation='sigmoid',
                             kernel_initializer='he_normal', padding='same', name='decoder_output')(convt3)
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
                             'mse': lambda a, b: mse(a, b)}
    # vae loss
    reconstruction_loss = N*N*reconstruction_losses[LSS](K.flatten(input), K.flatten(output))
    kl_loss = -0.5*K.sum(1+z_log_var-K.square(z_mean)-K.exp(z_log_var), axis=-1)
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
    sldmp = np.array([[rdmp[i, j, idat[i, j], :] for j in range(nt)] for i in range(nh)])
    sldat = np.array([[rdat[i, j, idat[i, j], :] for j in range(nt)] for i in range(nh)])
    return sldmp, sldat


def inlier_selection(dmp, dat, intrvl, ns):
    rdmp = dmp[::intrvl, ::intrvl]
    rdat = dat[::intrvl, ::intrvl]
    nh, nt, _, _ = rdmp.shape
    if AD:
        lof = LocalOutlierFactor(contamination='auto', n_jobs=THREADS)
    idat = np.zeros((nh, nt, ns), dtype=np.uint16)
    if VERBOSE:
        print('selecting inlier samples from classification data')
        print(100*'-')
    for i in tqdm(range(nh), disable=not VERBOSE):
        for j in tqdm(range(nt), disable=not VERBOSE):
                if AD:
                    fpred = lof.fit_predict(rdmp[i, j])
                    try:
                        idat[i, j] = np.random.choice(np.where(fpred==1)[0], size=ns, replace=False)
                    except:
                        idat[i, j] = np.argsort(lof.negative_outlier_factor_)[:ns]
                else:
                    idat[i, j] = np.random.permutation(rdat[i, j].shape[0])[:ns]
    if VERBOSE:
        print('\n'+100*'-')
    sldmp = np.array([[rdmp[i, j, idat[i, j], :] for j in range(nt)] for i in range(nh)])
    sldat = np.array([[rdat[i, j, idat[i, j], :] for j in range(nt)] for i in range(nh)])
    return sldmp, sldat


if __name__ == '__main__':
    # parse command line arguments
    (VERBOSE, PLOT, PARALLEL, GPU, FFT, AD, THREADS, NAME, N,
     UNI, UNS, SNI, SNS,
     SCLR, LD, MNFLD, ED, CLST, NC,
     BACKEND, OPT, LSS, EP, LR, SEED, EV) = parse_args()
    if CLST == 'dbscan':
        NCS = '%.0e' % NC
    else:
        NC = int(NC)
        NCS = '%d' % NC
    CWD = os.getcwd()
    EPS = 0.025
    # number of phases
    NPH = 3

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
    from keras.models import Model
    from keras.layers import Input, Lambda, Dense, Conv2D, Conv2DTranspose, Flatten, Reshape
    from keras.losses import binary_crossentropy, mse
    from keras.optimizers import SGD, Adadelta, Adam, Nadam
    from keras.callbacks import History, CSVLogger
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

    OUTPREF = CWD+'/%s.%d.%d.%d.%s.cnn2d.%s.%s.%d.%d.%.0e.%d.%d.%s.%s.%d.%s.%s.%d.%d.%d' % \
              (NAME, N, SNI, SNS, SCLR, OPT, LSS, LD, EP, LR, UNI, UNS, EV, MNFLD, ED, CLST, NCS, AD, FFT, SEED)
    write_specs()

    try:
        CDMP = np.load(CWD+'/%s.%d.%d.%d.%d.dmp.c.npy' % (NAME, N, SNI, SNS, SEED))
        CDAT = np.load(CWD+'/%s.%d.%d.%d.%d.dat.c.npy' % (NAME, N, SNI, SNS, SEED))
        if VERBOSE:
            print(100*'-')
            print('selected classification samples loaded from file')
            print(100*'-')
    except:
        DAT = np.load(CWD+'/%s.%d.dat.npy' % (NAME, N))
        DMP = np.load(CWD+'/%s.%d.dmp.npy' % (NAME, N))
        if VERBOSE:
            print(100*'-')
            print('full dataset loaded from file')
            print(100*'-')
        CDMP, CDAT = random_selection(DMP, DAT, SNI, SNS)
        del DAT, DMP
        np.save(CWD+'/%s.%d.%d.%d.%d.dmp.c.npy' % (NAME, N, SNI, SNS, SEED), CDMP)
        np.save(CWD+'/%s.%d.%d.%d.%d.dat.c.npy' % (NAME, N, SNI, SNS, SEED), CDAT)
        if VERBOSE:
            print('selected classification samples generated')
            print(100*'-')
    CH = np.load(CWD+'/%s.%d.h.npy' % (NAME, N))[::SNI]
    CT = np.load(CWD+'/%s.%d.t.npy' % (NAME, N))[::SNI]
    SNH, SNT = CH.size, CT.size
    if FFT:
        NCH = 2
    else:
        NCH = 1

    # scaler dictionary
    SCLRS = {'minmax':MinMaxScaler(feature_range=(0, 1)),
             'standard':StandardScaler(),
             'robust':RobustScaler(),
             'tanh':TanhScaler()}

    try:
        SCDMP = np.load(CWD+'/%s.%d.%d.%d.%s.%d.%d.dmp.sc.npy' \
                        % (NAME, N, SNI, SNS, SCLR, FFT, SEED)).reshape(SNH*SNT*SNS, N, N, NCH)
        if VERBOSE:
            print('scaled selected classification samples loaded from file')
            print(100*'-')
    except:
        if FFT:
            FCDMP = np.fft.fft2(CDMP, axes=(3, 4))
            CDMP = np.concatenate((np.real(FCDMP)[:, :, :, :, :, np.newaxis],
                                   np.imag(FCDMP)[:, :, :, :, :, np.newaxis]), axis=-1)
            del FCDMP
        else:
            CDMP = CDMP[:, :, :, :, :, np.newaxis]
            NCH = 1
        if SCLR == 'glbl':
            SCDMP = CDMP.reshape(SNH*SNT*SNS, N, N, NCH)
            for i in range(NCH):
                TMIN, TMAX = SCDMP[:, :, :, i].min(), SCDMP[:, :, :, i].max()
                SCDMP[:, :, :, i] = (SCDMP[:, :, :, i]-TMIN)/(TMAX-TMIN)
            del TMIN, TMAX
        else:
            SCDMP = SCLRS[SCLR].fit_transform(CDMP.reshape(SNH*SNT*SNS, N*N*NCH)).reshape(SNH*SNT*SNS, N, N, NCH)
        np.save(CWD+'/%s.%d.%d.%d.%s.%d.%d.dmp.sc.npy' % (NAME, N, SNI, SNS, SCLR, FFT, SEED), SCDMP.reshape(SNH, SNT, SNS, N, N, NCH))
        if VERBOSE:
            print('scaled selected classification samples computed')
            print(100*'-')

    OPTS = {'sgd': SGD(lr=LR, momentum=0.0, decay=0.0, nesterov=False),
            'adadelta': Adadelta(lr=LR, rho=0.95, epsilon=None, decay=0.0),
            'adam': Adam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True),
            'nadam': Nadam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)}

    ENC, DEC, VAE = build_variational_autoencoder()

    try:
        VAE.load_weights(CWD+'/%s.%d.%d.%d.%s.cnn2d.%s.%s.%d.%d.%.0e.%d.%d.vae.wt.h5' \
                         % (NAME, N, SNI, SNS, SCLR, OPT, LSS, LD, EP, LR, FFT, SEED), by_name=True)
        TLOSS = np.load(CWD+'/%s.%d.%d.%d.%s.cnn2d.%s.%s.%d.%d.%.0e.%d.%d.vae.loss.trn.npy' \
                        % (NAME, N, SNI, SNS, SCLR, OPT, LSS, LD, EP, LR, FFT, SEED))
        VLOSS = np.load(CWD+'/%s.%d.%d.%d.%s.cnn2d.%s.%s.%d.%d.%.0e.%d.%d.vae.loss.val.npy' \
                        % (NAME, N, SNI, SNS, SCLR, OPT, LSS, LD, EP, LR, FFT, SEED))
        if VERBOSE:
            print('variational autoencoder trained weights loaded from file')
            print(100*'-')
    except:
        if VERBOSE:
            print('variational autoencoder training on scaled selected classification samples')
            print(100*'-')
        CSVLG = CSVLogger(CWD+'/%s.%d.%d.%d.%s.cnn2d.%s.%s.%d.%d.%.0e.%d.%d.vae.log.csv' \
                          % (NAME, N, SNI, SNS, SCLR, OPT, LSS, LD, EP, LR, FFT, SEED), append=True, separator=',')
        TRN, VAL = train_test_split(SCDMP, test_size=0.25, shuffle=True)
        VAE.fit(x=TRN, y=None, validation_data=(VAL, None), epochs=EP, batch_size=SNH*SNH,
                shuffle=True, verbose=VERBOSE, callbacks=[CSVLG, History()])
        del TRN, VAL
        TLOSS = VAE.history.history['loss']
        VLOSS = VAE.history.history['val_loss']
        VAE.save_weights(CWD+'/%s.%d.%d.%d.%s.cnn2d.%s.%s.%d.%d.%.0e.%d.%d.vae.wt.h5' \
                         % (NAME, N, SNI, SNS, SCLR, OPT, LSS, LD, EP, LR, FFT, SEED))
        np.save(CWD+'/%s.%d.%d.%d.%s.cnn2d.%s.%s.%d.%d.%.0e.%d.%d.vae.loss.trn.npy' \
                % (NAME, N, SNI, SNS, SCLR, OPT, LSS, LD, EP, LR, FFT, SEED), TLOSS)
        np.save(CWD+'/%s.%d.%d.%d.%s.cnn2d.%s.%s.%d.%d.%.0e.%d.%d.vae.loss.val.npy' \
                % (NAME, N, SNI, SNS, SCLR, OPT, LSS, LD, EP, LR, FFT, SEED), VLOSS)
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
        out.write('# variational autoencoder training history information\n')
        out.write('# ' + 100*'-' + '\n')
        out.write('# | epoch | training loss | validation loss |\n')
        out.write('# ' + 100*'-' + '\n')
        for i in range(EP):
            out.write('%02d %.2f %.2f\n' % (i, TLOSS[i], VLOSS[i]))
        out.write('# ' + 100*'-' + '\n')

    try:
        ZENC = np.load(CWD+'/%s.%d.%d.%d.%s.cnn2d.%s.%s.%d.%d.%.0e.%d.%d.zenc.npy'
                       % (NAME, N, SNI, SNS, SCLR, OPT, LSS, LD, EP, LR, FFT, SEED)).reshape(SNH*SNT*SNS, 2, LD)
        if VERBOSE:
            print('z encodings of scaled selected classification samples loaded from file')
            print(100*'-')
    except:
        if VERBOSE:
            print('predicting z encodings of scaled selected classification samples')
            print(100*'-')
        ZENC = np.swapaxes(np.array(ENC.predict(SCDMP, verbose=VERBOSE)), 0, 1)[:, :2, :]
        ZENC[:, 1, :] = np.exp(0.5*ZENC[:, 1, :])
        np.save(CWD+'/%s.%d.%d.%d.%s.cnn2d.%s.%s.%d.%d.%.0e.%d.%d.zenc.npy'
                % (NAME, N, SNI, SNS, SCLR, OPT, LSS, LD, EP, LR, FFT, SEED), ZENC.reshape(SNH, SNT, SNS, 2, LD))
        if VERBOSE:
            print(100*'-')
            print('z encodings of scaled selected classification samples predicted')
            print(100*'-')

    EIND = [i for i in range(len(EV)) if EV[i] == '1']
    try:
        SLZENC = np.load(CWD+'/%s.%d.%d.%d.%s.cnn2d.%s.%s.%d.%d.%.0e.%d.%d.%s.%d.%d.%d.zenc.inl.npy' \
                         % (NAME, N, SNI, SNS, SCLR, OPT, LSS, LD, EP, LR, UNI, UNS, EV, AD, FFT, SEED))
        SLDAT = np.load(CWD+'/%s.%d.%d.%d.%s.cnn2d.%s.%s.%d.%d.%.0e.%d.%d.%s.%d.%d.%d.dat.inl.npy' \
                        % (NAME, N, SNI, SNS, SCLR, OPT, LSS, LD, EP, LR, UNI, UNS, EV, AD, FFT, SEED))
        if VERBOSE:
            print('inlier selected z encodings loaded from file')
            print(100*'-')
    except:
        SLZENC, SLDAT = inlier_selection(ZENC[:, EIND, :].reshape(SNH, SNT, SNS, len(EIND)*LD), CDAT, UNI, UNS)
        np.save(CWD+'/%s.%d.%d.%d.%s.cnn2d.%s.%s.%d.%d.%.0e.%d.%d.%s.%d.%d.%d.zenc.inl.npy' \
                % (NAME, N, SNI, SNS, SCLR, OPT, LSS, LD, EP, LR, UNI, UNS, EV, AD, FFT, SEED), SLZENC)
        np.save(CWD+'/%s.%d.%d.%d.%s.cnn2d.%s.%s.%d.%d.%.0e.%d.%d.%s.%d.%d.%d.dat.inl.npy' \
                % (NAME, N, SNI, SNS, SCLR, OPT, LSS, LD, EP, LR, UNI, UNS, EV, AD, FFT, SEED), SLDAT)
        if VERBOSE:
            print('inlier selected z encodings computed')
            print(100*'-')

    UH, UT = CH[::UNI], CT[::UNI]
    UNH, UNT = UH.size, UT.size
    SLES = SLDAT[:, :, :, 0]
    SLMS = SLDAT[:, :, :, 1]

    SLEM = np.mean(SLES, -1)
    SLSP = np.divide(np.mean(np.square(SLES), -1)-np.square(SLEM), UT[np.newaxis, :])
    SLMM = np.mean(SLMS, -1)
    SLSU = np.divide(np.mean(np.square(SLMS), -1)-np.square(SLMM), UT[np.newaxis, :])

    TSNEINITPCA = PCA(n_components=ED)
    TSNEINIT = TSNEINITPCA.fit_transform(SLZENC.reshape(UNH*UNT*UNS, len(EIND)*LD))
    SLZEVAR = TSNEINITPCA.explained_variance_ratio_
    if VERBOSE:
        print('pca fit information')
        print(100*'-')
        print('explained variances: '+ED*'%f ' % tuple(SLZEVAR))
        print('total: %f' % np.sum(SLZEVAR))
        print(100*'-')
    with open(OUTPREF+'.out', 'a') as out:
        out.write('# pca fit information\n')
        out.write('# ' + 100*'-' + '\n')
        out.write('# explained variances: '+ED*'%f ' % tuple(SLZEVAR)+'\n')
        out.write('# total: %f\n' % np.sum(SLZEVAR))
        out.write('# ' + 100*'-' + '\n')

    DIAGMLV = np.mean(SCLRS['tanh'].fit_transform(SLZENC.reshape(UNH*UNT*UNS, 2*LD)).reshape(UNH, UNT, UNS, 2*LD), 2)
    if np.mean(DIAGMLV[int(UNH/2), 0, 1]) > np.mean(DIAGMLV[int(UNH/2), -1, 1]):
        DIAGMLV[:, :, 1] = 1-DIAGMLV[:, :, 1]
    DIAGSLV = SCLRS['tanh'].fit_transform(np.std(SLZENC/UT[np.newaxis, :, np.newaxis, np.newaxis], 2).reshape(UNH*UNT, 2*LD)).reshape(UNH, UNT, 2*LD)
    DIAGMMV = np.mean(SCLRS['minmax'].fit_transform(SLDAT[:, :, :, (1, 0)].reshape(UNH*UNT*UNS, 2)).reshape(UNH, UNT, UNS, 2), 2)
    DIAGSMV = SCLRS['minmax'].fit_transform(np.std(SLDAT[:, :, :, (1, 0)]/UT[np.newaxis, :, np.newaxis, np.newaxis], 2).reshape(UNH*UNT, 2)).reshape(UNH, UNT, 2)
    for i in range(2):
        for j in range(2*LD):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
            if i == 0:
                ax.imshow(DIAGMLV[:, :, j], aspect='equal', interpolation='none', origin='lower', cmap=CM)
            if i == 1:
                ax.imshow(DIAGSLV[:, :, j], aspect='equal', interpolation='none', origin='lower', cmap=CM)
            ax.grid(which='minor', axis='both', linestyle='-', color='k', linewidth=1)
            ax.set_xticks(np.arange(UT.size), minor=True)
            ax.set_yticks(np.arange(UH.size), minor=True)
            plt.xticks(np.arange(UT.size)[::4], np.round(UT, 2)[::4], rotation=-60)
            plt.yticks(np.arange(UT.size)[::4], np.round(UH, 2)[::4])
            plt.xlabel('T')
            plt.ylabel('H')
            # plt.title('Ising Model Phase Diagram')
            fig.savefig(OUTPREF+'.vae.diag.ld.%d.%d.png' % (i, j))
    for i in range(2):
        for j in range(2):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
            if i == 0:
                ax.imshow(DIAGMMV[:, :, j], aspect='equal', interpolation='none', origin='lower', cmap=CM)
            if i == 1:
                ax.imshow(DIAGSMV[:, :, j], aspect='equal', interpolation='none', origin='lower', cmap=CM)
            ax.grid(which='minor', axis='both', linestyle='-', color='k', linewidth=1)
            ax.set_xticks(np.arange(UT.size), minor=True)
            ax.set_yticks(np.arange(UH.size), minor=True)
            plt.xticks(np.arange(UT.size)[::4], np.round(UT, 2)[::4], rotation=-60)
            plt.yticks(np.arange(UT.size)[::4], np.round(UH, 2)[::4])
            plt.xlabel('T')
            plt.ylabel('H')
            # plt.title('Ising Model Phase Diagram')
            fig.savefig(OUTPREF+'.vae.diag.mv.%d.%d.png' % (i, j))
    for i in range(2):
        for j in range(2):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
            if i == 0:
                ax.imshow(CM(np.abs(DIAGMMV[:, :, j]-DIAGMLV[:, :, j])), aspect='equal', interpolation='none', origin='lower')
            if i == 1:
                ax.imshow(CM(np.abs(DIAGSMV[:, :, j]-DIAGSLV[:, :, j])), aspect='equal', interpolation='none', origin='lower')
            ax.grid(which='minor', axis='both', linestyle='-', color='k', linewidth=1)
            ax.set_xticks(np.arange(UT.size), minor=True)
            ax.set_yticks(np.arange(UH.size), minor=True)
            plt.xticks(np.arange(UT.size)[::4], np.round(UT, 2)[::4], rotation=-60)
            plt.yticks(np.arange(UT.size)[::4], np.round(UH, 2)[::4])
            plt.xlabel('T')
            plt.ylabel('H')
            # plt.title('Ising Model Phase Diagram')
            fig.savefig(OUTPREF+'.vae.diag.er.%d.%d.png' % (i, j))
            fig = plt.figure()
            ax = fig.add_subplot(111)
            if i == 0:
                ax.scatter(DIAGMMV[:, :, j].reshape(-1), DIAGMLV[:, :, j].reshape(-1),
                           c=DIAGMMV[:, :, j].reshape(-1), cmap=plt.get_cmap('plasma'),
                           s=64, alpha=0.5, edgecolors='')
            if i == 1:
                ax.scatter(DIAGSMV[:, :, j].reshape(-1), DIAGSLV[:, :, j].reshape(-1),
                           c=DIAGSMV[:, :, j].reshape(-1), cmap=plt.get_cmap('plasma'),
                           s=64, alpha=0.5, edgecolors='')
            fig.savefig(OUTPREF+'.vae.reg.%d.%d.png' % (i, j))

    # # reduction dictionary
    # MNFLDS = {'pca':PCA(n_components=ED),
    #           'kpca':KernelPCA(n_components=ED, n_jobs=THREADS),
    #           'isomap':Isomap(n_components=ED, n_jobs=THREADS),
    #           'lle':LocallyLinearEmbedding(n_components=ED, n_jobs=THREADS),
    #           'tsne':TSNE(n_components=ED, perplexity=UNS,
    #                       early_exaggeration=24, learning_rate=200, n_iter=1000,
    #                       verbose=VERBOSE, n_jobs=THREADS, init=TSNEINIT)}

    # try:
    #     MSLZENC = np.load(CWD+'/%s.%d.%d.%d.%s.cnn2d.%s.%s.%d.%d.%.0e.%d.%d.%s.%s.%d.%d.%d.%d.zenc.inl.mfld.npy' \
    #                       % (NAME, N, SNI, SNS, SCLR, OPT, LSS, LD, EP, LR, UNI, UNS, EV, MNFLD, ED, AD, FFT, SEED))
    #     if VERBOSE:
    #         print('inlier selected z encoding manifold loaded from file')
    #         print(100*'-')
    # except:
    #     MSLZENC = MNFLDS[MNFLD].fit_transform(SLZENC.reshape(UNH*UNT*UNS, len(EIND)*LD))
    #     np.save(CWD+'/%s.%d.%d.%d.%s.cnn2d.%s.%s.%d.%d.%.0e.%d.%d.%s.%s.%d.%d.%d.%d.zenc.inl.mfld.npy' \
    #             % (NAME, N, SNI, SNS, SCLR, OPT, LSS, LD, EP, LR, UNI, UNS, EV, MNFLD, ED, AD, FFT, SEED), MSLZENC)
    #     if VERBOSE:
    #         if MNFLD == 'tsne':
    #             print(100*'-')
    #         print('inlier selected z encoding manifold computed')
    #         print(100*'-')

    # # clustering dictionary
    # CLSTS = {'agglomerative': AgglomerativeClustering(n_clusters=NC, linkage='ward'),
    #          'kmeans': KMeans(n_jobs=THREADS, n_clusters=NC, init='k-means++'),
    #          'spectral': SpectralClustering(n_jobs=THREADS, n_clusters=NC, random_state=SEED),
    #          'dbscan': DBSCAN(eps=NC, min_samples=np.max([4, int(UNS/4)]), leaf_size=30)}
    # try:
    #     CLMSLZENC = np.load(CWD+'/%s.%d.%d.%d.%s.cnn2d.%s.%s.%d.%d.%.0e.%d.%d.%s.%s.%d.%s.%s.%d.%d.%d.zenc.inl.clst.npy' \
    #                         % (NAME, N, SNI, SNS, SCLR, OPT, LSS, LD, EP, LR, UNI, UNS, EV, MNFLD, ED, CLST, NCS, AD, FFT, SEED))
    #     if VERBOSE:
    #         print('inlier selected z encoding manifold clustering loaded from file')
    #         print(100*'-')
    # except:
    #     CLMSLZENC = CLSTS[CLST].fit_predict(MSLZENC)
    #     np.save(CWD+'/%s.%d.%d.%d.%s.cnn2d.%s.%s.%d.%d.%.0e.%d.%d.%s.%s.%d.%s.%s.%d.%d.%d.zenc.inl.clst.npy' \
    #             % (NAME, N, SNI, SNS, SCLR, OPT, LSS, LD, EP, LR, UNI, UNS, EV, MNFLD, ED, CLST, NCS, AD, FFT, SEED), CLMSLZENC)
    #     if VERBOSE:
    #         print('inlier selected z encoding manifold clustering computed')
    #         print(100*'-')

    # if CLST == 'dbscan':
    #     NCL = np.unique(CLMSLZENC[CLMSLZENC > -1]).size
    # else:
    #     NCL = NC

    # CLESFC = np.mean(np.array([np.mean(SLES.reshape(UNH*UNT*UNS)[CLMSLZENC == i]) for i in range(-1, NCL)])[CLMSLZENC+1].reshape(UNH, UNT, UNS), -1)
    # CLMSFC = np.mean(np.array([np.mean(SLMS.reshape(UNH*UNT*UNS)[CLMSLZENC == i]) for i in range(-1, NCL)])[CLMSLZENC+1].reshape(UNH, UNT, UNS), -1)

    # CLMEFC = np.array([np.mean(SLES.reshape(UNH*UNT*UNS)[CLMSLZENC == i]) for i in range(NCL)])
    # CLMMFC = np.array([np.mean(SLMS.reshape(UNH*UNT*UNS)[CLMSLZENC == i]) for i in range(NCL)])

    # CLCTN = np.array([np.mean(SLZENC.reshape(UNH*UNT*UNS, len(EIND)*LD)[CLMSLZENC == i], 0) for i in range(NCL)])
    # if MNFLD == 'tsne':
    #     MNFLDS[MNFLD] = TSNE(n_components=ED, perplexity=int(NCL/NPH),
    #                          early_exaggeration=24, learning_rate=200, n_iter=1000,
    #                          verbose=False, n_jobs=THREADS, init=PCA(n_components=ED).fit_transform(CLCTN))
    # # MCLCTN = MNFLDS[MNFLD].fit_transform(CLCTN)
    # MCLCTN = TSNEINITPCA.transform(CLCTN)
    # CLC = SpectralClustering(n_jobs=THREADS, n_clusters=NPH, random_state=SEED).fit_predict(MCLCTN)
    # # CLC = KMeans(n_jobs=THREADS, n_clusters=NPH, init='k-means++').fit_predict(MCLCTN)
    # # CLC = AgglomerativeClustering(n_clusters=NPH, linkage='ward').fit_predict(MCLCTN)
    # CL = np.zeros(CLMSLZENC.shape, dtype=np.int32)
    # CL[CLMSLZENC == -1] = -1
    # for i in range(NCL):
    #     CL[CLMSLZENC == i] = CLC[i]
    # CLME = np.array([np.mean(SLES.reshape(UNH*UNT*UNS)[CL == i]) for i in range(NPH)])
    # CLMM = np.array([np.mean(SLMS.reshape(UNH*UNT*UNS)[CL == i]) for i in range(NPH)])
    # ICLCM = np.argsort(CLMM)
    # for i in range(NPH):
    #     CL[CL == ICLCM[i]] = i+NPH
    # CL[CL > -1] -= NPH
    # CLMES = np.array([np.mean(SLES.reshape(UNH*UNT*UNS)[CL == i]) for i in range(NPH)])
    # CLMMS = np.array([np.mean(SLMS.reshape(UNH*UNT*UNS)[CL == i]) for i in range(NPH)])

    # CLB = np.array([[np.bincount(CL.reshape(UNH, UNT, UNS)[i, j][CL.reshape(UNH, UNT, UNS)[i, j] > -1],
    #                              minlength=NPH) for j in range(UNT)] for i in range(UNH)])/UNS
    # UTRANS = np.array([odr_fit(logistic, UT, CLB[i, :, 1], EPS*np.ones(UNT), (1, 2.5))[0][1] for i in range(UNH)])
    # UITRANS = (UTRANS-UT[0])/(UT[-1]-UT[0])*(UNT-1)
    # UCPOPT, UCPERR, UCDOM, UCVAL = odr_fit(absolute, UH, UTRANS, EPS*np.ones(UNT), (1.0, 0.0, 1.0, 2.5))
    # UICDOM = (UCDOM-UH[0])/(UH[-1]-UH[0])*(UNH-1)
    # UICVAL = (UCVAL-UT[0])/(UT[-1]-UT[0])*(UNT-1)

    # if VERBOSE:
    #     print('fit parameters calculated (t_c = a*|h-b|**c+d)')
    #     print(100*'-')
    #     print('| a | b | c | d |')
    #     print(100*'-')
    #     print(4*'%.2f ' % tuple(UCPOPT))
    #     print(4*'%.2f ' % tuple(UCPERR))
    #     print(100*'-')
    # with open(OUTPREF+'.out', 'a') as out:
    #     out.write('# fit parameters calculated (t_c = a*|h-b|**c+d)\n')
    #     out.write('# '+100*'-'+'\n')
    #     out.write('# | a | b | c | d |\n')
    #     out.write('# '+100*'-'+'\n')
    #     out.write('# '+4*'%.2f ' % tuple(UCPOPT) +'\n')
    #     out.write('# '+4*'%.2f ' % tuple(UCPERR) +'\n')
    #     out.write('# ' + 100*'-' + '\n')

    # if PLOT:
    #     fig = plt.figure()
    #     if ED == 3:
    #         ax = fig.add_subplot(111, projection='3d')
    #         ax.scatter(MSLZENC[:, 0], MSLZENC[:, 1], MSLZENC[:, 2], c=SLMS.reshape(-1), cmap=plt.get_cmap('plasma'),
    #                 s=32, alpha=0.25, edgecolors='')
    #     if ED == 2:
    #         ax = fig.add_subplot(111)
    #         ax.scatter(MSLZENC[:, 0], MSLZENC[:, 1], c=SLMS.reshape(-1), cmap=plt.get_cmap('plasma'),
    #                 s=32, alpha=0.25, edgecolors='')
    #     fig.savefig(OUTPREF+'.vae.emb.smpl.png')

    #     fig = plt.figure()
    #     if ED == 3:
    #         ax = fig.add_subplot(111, projection='3d')
    #         ax.scatter(MSLZENC[CLMSLZENC == -1, 0], MSLZENC[CLMSLZENC == -1, 1], MSLZENC[CLMSLZENC == -1, 2],
    #                 c='c', s=32, alpha=1.0, edgecolors='')
    #     elif ED == 2:
    #         ax = fig.add_subplot(111)
    #         ax.scatter(MSLZENC[CLMSLZENC == -1, 0], MSLZENC[CLMSLZENC == -1, 1],
    #                 c='c', s=32, alpha=1.0, edgecolors='')
    #     for i in range(NCL):
    #         if ED == 3:
    #             ax.scatter(MSLZENC[CLMSLZENC == i, 0], MSLZENC[CLMSLZENC == i, 1], MSLZENC[CLMSLZENC == i, 2],
    #                     c=np.array(CM(SCALE(CLMMFC[i], SLMS.reshape(-1))))[np.newaxis, :],
    #                     s=32, alpha=0.25, edgecolors='')
    #         if ED == 2:
    #             ax.scatter(MSLZENC[CLMSLZENC == i, 0], MSLZENC[CLMSLZENC == i, 1],
    #                     c=np.array(CM(SCALE(CLMMFC[i], SLMS.reshape(-1))))[np.newaxis, :],
    #                     s=32, alpha=0.25, edgecolors='')
    #     fig.savefig(OUTPREF+'.vae.emb.clst.fc.png')

    #     fig = plt.figure()
    #     if ED == 3:
    #         ax = fig.add_subplot(111, projection='3d')
    #         ax.scatter(MSLZENC[CL == -1, 0], MSLZENC[CL == -1, 1], MSLZENC[CL == -1, 2],
    #                 c='c', s=32, alpha=1.0, edgecolors='')
    #     elif ED == 2:
    #         ax = fig.add_subplot(111)
    #         ax.scatter(MSLZENC[CL == -1, 0], MSLZENC[CL == -1, 1],
    #                 c='c', s=32, alpha=1.0, edgecolors='')
    #     for i in range(NPH):
    #         if ED == 3:
    #             ax.scatter(MSLZENC[CL == i, 0], MSLZENC[CL == i, 1], MSLZENC[CL == i, 2],
    #                     c=np.array(CM(SCALE(CLMMS[i], SLMS.reshape(-1))))[np.newaxis, :],
    #                     s=32, alpha=0.25, edgecolors='')
    #         if ED == 2:
    #             ax.scatter(MSLZENC[CL == i, 0], MSLZENC[CL == i, 1],
    #                     c=np.array(CM(SCALE(CLMMS[i], SLMS.reshape(-1))))[np.newaxis, :],
    #                     s=32, alpha=0.25, edgecolors='')
    #     fig.savefig(OUTPREF+'.vae.emb.clst.rc.png')

    #     fig = plt.figure()
    #     if ED == 3:
    #         ax = fig.add_subplot(111, projection='3d')
    #         for i in range(NPH):
    #             ax.scatter(MCLCTN[CLC == i, 0], MCLCTN[CLC == i, 1], MCLCTN[CLC == i, 2],
    #                     c=np.array(CM(SCALE(CLMMFC[CLC == i], SLMS.reshape(-1)))),
    #                     s=256, alpha=1.0, edgecolors=CM(SCALE(CLMM[i], SLMS.reshape(-1))), linewidths=4.0)
    #     if ED == 2:
    #         ax = fig.add_subplot(111)
    #         for i in range(NPH):
    #             ax.scatter(MCLCTN[CLC == i, 0], MCLCTN[CLC == i, 1],
    #                     c=np.array(CM(SCALE(CLMMFC[CLC == i], SLMS.reshape(-1)))),
    #                     s=256, alpha=1.0, edgecolors=CM(SCALE(CLMM[i], SLMS.reshape(-1))), linewidths=4.0)
    #     fig.savefig(OUTPREF+'.vae.emb.ld.png')

    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #     ax.spines['right'].set_visible(False)
    #     ax.spines['top'].set_visible(False)
    #     ax.xaxis.set_ticks_position('bottom')
    #     ax.yaxis.set_ticks_position('left')
    #     ax.plot(UITRANS, np.arange(UNH), color='yellow')
    #     ax.plot(UICVAL, UICDOM, color='yellow', linestyle='--')
    #     ax.imshow(CLB, aspect='equal', interpolation='none', origin='lower', cmap=CM)
    #     ax.grid(which='minor', axis='both', linestyle='-', color='k', linewidth=1)
    #     ax.set_xticks(np.arange(UT.size), minor=True)
    #     ax.set_yticks(np.arange(UH.size), minor=True)
    #     plt.xticks(np.arange(UT.size)[::4], np.round(UT, 2)[::4], rotation=-60)
    #     plt.yticks(np.arange(UT.size)[::4], np.round(UH, 2)[::4])
    #     plt.xlabel('T')
    #     plt.ylabel('H')
    #     # plt.title('Ising Model Phase Diagram')
    #     fig.savefig(OUTPREF+'.vae.diag.phase.png')

    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #     ax.spines['right'].set_visible(False)
    #     ax.spines['top'].set_visible(False)
    #     ax.xaxis.set_ticks_position('bottom')
    #     ax.yaxis.set_ticks_position('left')
    #     ax.plot(UITRANS, np.arange(UNH), color='yellow')
    #     ax.plot(UICVAL, UICDOM, color='yellow', linestyle='--')
    #     ax.imshow(np.mean(TSNEINIT[:, 0].reshape(UNH, UNT, UNS), -1), aspect='equal', interpolation='none', origin='lower', cmap=CM)
    #     ax.grid(which='minor', axis='both', linestyle='-', color='k', linewidth=1)
    #     ax.set_xticks(np.arange(UT.size), minor=True)
    #     ax.set_yticks(np.arange(UH.size), minor=True)
    #     plt.xticks(np.arange(UT.size)[::4], np.round(UT, 2)[::4], rotation=-60)
    #     plt.yticks(np.arange(UT.size)[::4], np.round(UH, 2)[::4])
    #     plt.xlabel('T')
    #     plt.ylabel('H')
    #     # plt.title('Ising Model Phase Diagram')
    #     fig.savefig(OUTPREF+'.vae.diag.pca.0.png')

    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #     ax.spines['right'].set_visible(False)
    #     ax.spines['top'].set_visible(False)
    #     ax.xaxis.set_ticks_position('bottom')
    #     ax.yaxis.set_ticks_position('left')
    #     ax.plot(UITRANS, np.arange(UNH), color='yellow')
    #     ax.plot(UICVAL, UICDOM, color='yellow', linestyle='--')
    #     ax.imshow(np.mean(TSNEINIT[:, 1].reshape(UNH, UNT, UNS), -1), aspect='equal', interpolation='none', origin='lower', cmap=CM)
    #     ax.grid(which='minor', axis='both', linestyle='-', color='k', linewidth=1)
    #     ax.set_xticks(np.arange(UT.size), minor=True)
    #     ax.set_yticks(np.arange(UH.size), minor=True)
    #     plt.xticks(np.arange(UT.size)[::4], np.round(UT, 2)[::4], rotation=-60)
    #     plt.yticks(np.arange(UT.size)[::4], np.round(UH, 2)[::4])
    #     plt.xlabel('T')
    #     plt.ylabel('H')
    #     # plt.title('Ising Model Phase Diagram')
    #     fig.savefig(OUTPREF+'.vae.diag.pca.1.png')

    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #     ax.spines['right'].set_visible(False)
    #     ax.spines['top'].set_visible(False)
    #     ax.xaxis.set_ticks_position('bottom')
    #     ax.yaxis.set_ticks_position('left')
    #     ax.plot(UITRANS, np.arange(UNH), color='yellow')
    #     ax.plot(UICVAL, UICDOM, color='yellow', linestyle='--')
    #     ax.imshow(SLSP, aspect='equal', interpolation='none', origin='lower', cmap=CM)
    #     ax.grid(which='minor', axis='both', linestyle='-', color='k', linewidth=1)
    #     ax.set_xticks(np.arange(UT.size), minor=True)
    #     ax.set_yticks(np.arange(UH.size), minor=True)
    #     plt.xticks(np.arange(UT.size)[::4], np.round(UT, 2)[::4], rotation=-60)
    #     plt.yticks(np.arange(UT.size)[::4], np.round(UH, 2)[::4])
    #     plt.xlabel('T')
    #     plt.ylabel('H')
    #     # plt.title('Ising Model Phase Diagram')
    #     fig.savefig(OUTPREF+'.vae.diag.st.spheat.png')

    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #     ax.spines['right'].set_visible(False)
    #     ax.spines['top'].set_visible(False)
    #     ax.xaxis.set_ticks_position('bottom')
    #     ax.yaxis.set_ticks_position('left')
    #     ax.plot(UITRANS, np.arange(UNH), color='yellow')
    #     ax.plot(UICVAL, UICDOM, color='yellow', linestyle='--')
    #     ax.imshow(SLSU, aspect='equal', interpolation='none', origin='lower', cmap=CM)
    #     ax.grid(which='minor', axis='both', linestyle='-', color='k', linewidth=1)
    #     ax.set_xticks(np.arange(UT.size), minor=True)
    #     ax.set_yticks(np.arange(UH.size), minor=True)
    #     plt.xticks(np.arange(UT.size)[::4], np.round(UT, 2)[::4], rotation=-60)
    #     plt.yticks(np.arange(UT.size)[::4], np.round(UH, 2)[::4])
    #     plt.xlabel('T')
    #     plt.ylabel('H')
    #     # plt.title('Ising Model Phase Diagram')
    #     fig.savefig(OUTPREF+'.vae.diag.st.magsus.png')

    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #     ax.spines['right'].set_visible(False)
    #     ax.spines['top'].set_visible(False)
    #     ax.xaxis.set_ticks_position('bottom')
    #     ax.yaxis.set_ticks_position('left')
    #     ax.plot(UITRANS, np.arange(UNH), color='yellow')
    #     ax.plot(UICVAL, UICDOM, color='yellow', linestyle='--')
    #     ax.imshow(SLEM, aspect='equal', interpolation='none', origin='lower', cmap=CM)
    #     ax.grid(which='minor', axis='both', linestyle='-', color='k', linewidth=1)
    #     ax.set_xticks(np.arange(UT.size), minor=True)
    #     ax.set_yticks(np.arange(UH.size), minor=True)
    #     plt.xticks(np.arange(UT.size)[::4], np.round(UT, 2)[::4], rotation=-60)
    #     plt.yticks(np.arange(UT.size)[::4], np.round(UH, 2)[::4])
    #     plt.xlabel('T')
    #     plt.ylabel('H')
    #     # plt.title('Ising Model Phase Diagram')
    #     fig.savefig(OUTPREF+'.vae.diag.st.ener.png')

    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #     ax.spines['right'].set_visible(False)
    #     ax.spines['top'].set_visible(False)
    #     ax.xaxis.set_ticks_position('bottom')
    #     ax.yaxis.set_ticks_position('left')
    #     ax.plot(UITRANS, np.arange(UNH), color='yellow')
    #     ax.plot(UICVAL, UICDOM, color='yellow', linestyle='--')
    #     ax.imshow(SLMM, aspect='equal', interpolation='none', origin='lower', cmap=CM)
    #     ax.grid(which='minor', axis='both', linestyle='-', color='k', linewidth=1)
    #     ax.set_xticks(np.arange(UT.size), minor=True)
    #     ax.set_yticks(np.arange(UH.size), minor=True)
    #     plt.xticks(np.arange(UT.size)[::4], np.round(UT, 2)[::4], rotation=-60)
    #     plt.yticks(np.arange(UT.size)[::4], np.round(UH, 2)[::4])
    #     plt.xlabel('T')
    #     plt.ylabel('H')
    #     # plt.title('Ising Model Phase Diagram')
    #     fig.savefig(OUTPREF+'.vae.diag.st.mag.png')
