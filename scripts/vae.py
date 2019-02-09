import os
import pickle
import numpy as np
from TanhScaler import TanhScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

BACKEND = 'tensorflow'
SEED = 256
PARALLEL = True
THREADS = 12

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
from keras.optimizers import Nadam
from keras.models import Model
from keras.layers import Lambda, Input, Dense, Conv2D, Conv2DTranspose, Flatten, Reshape
from keras.losses import mse, binary_crossentropy
from keras import backend as K

CWD = os.getcwd()
NAME = 'ising_run'
N = 4
NI = 1
NS = 16
LR = 1e-4

H = pickle.load(open(CWD+'/%s.%d.h.pickle' % (NAME, N), 'rb'))[::NI]
T = pickle.load(open(CWD+'/%s.%d.t.pickle' % (NAME, N), 'rb'))[::NI]
NH, NT = H.size, T.size

DAT = pickle.load(open(CWD+'/%s.%d.dmp.pickle' % (NAME, N), 'rb'))[::NI, ::NI, :, :, :]
TDAT = pickle.load(open(CWD+'/%s.%d.dat.pickle' % (NAME, N), 'rb'))[::NI, ::NI, :, :]

try:
    IDAT = pickle.load(open(CWD+'/%s.%d.%d.%d.%d.aeidat.pickle' % (NAME, N, SEED, NI, NS), 'rb'))
    if VERBOSE:
        print('selected indices loaded from file')
        print(66*'-')
except:
    IDAT = np.zeros((NH, NT, NS), dtype=np.uint16)
    for i in range(NH):
        for j in range(NT):
            IDAT[i, j] = np.random.permutation(DAT[i, j].shape[0])[:NS]
    pickle.dump(IDAT, open(CWD+'/%s.%d.%d.%d.%d.aeidat.pickle' % (NAME, N, SEED, NI, NS), 'wb'))

DAT = np.array([[DAT[i, j, IDAT[i,j], :, :] for j in range(NT)] for i in range(NH)])
TDAT = np.array([[TDAT[i, j, IDAT[i,j], :] for j in range(NT)] for i in range(NH)])
del IDAT
ES, MS = TDAT[:, :, :, 0], TDAT[:, :, :, 1]
del TDAT

SCLR = TanhScaler()
SDAT = SCLR.fit_transform(DAT.reshape(NH*NT*NS, N*N)).reshape(NH*NT*NS, N, N, 1)

def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean+K.exp(0.5*z_log_var)*epsilon

input_shape = (N, N, 1)
batch_size = NS
kernel_size = 2
filters = 16
latent_dim = 3
epochs = 30

inputs = Input(shape=input_shape, name='encoder_input')
x = inputs
for i in range(2):
    filters *= 2
    x = Conv2D(filters=filters, kernel_size=kernel_size, activation='relu',
               padding='same', strides=1)(x)
shape = K.int_shape(x)
x = Flatten()(x)
x = Dense(16, activation='relu')(x)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(np.prod(shape[1:]), activation='relu')(latent_inputs)
x = Reshape(shape[1:])(x)
for i in range(2):
    x = Conv2DTranspose(filters=filters, kernel_size=kernel_size, activation='relu',
                        padding='same', strides=1)(x)
    filters //= 2
outputs = Conv2DTranspose(filters=1, kernel_size=kernel_size, activation='sigmoid',
                          padding='same', name='decoder_output')(x)

decoder = Model(latent_inputs, outputs, name='decoder')

outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')

reconstruction_loss = binary_crossentropy(K.flatten(inputs), K.flatten(outputs))
reconstruction_loss *= N*N
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5

vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
nadam = Nadam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
vae.compile(optimizer=nadam)

vae.fit(SDAT, epochs=epochs, batch_size=batch_size)
res = np.array(encoder.predict(SDAT))

figs = [plt.figure() for i in range(len(res))]
axs = [figs[i].add_subplot(111, projection='3d') for i in range(len(res))]
for i in range(len(res)):
    axs[i].scatter(res[i][:, 0], res[i][:, 1], res[i][:, 2], c=MS.reshape(-1), cmap=plt.get_cmap('plasma'),
                   s=128, alpha=0.015625, edgecolors='k')
    axs[i].set_aspect('equal', 'datalim')
plt.show()