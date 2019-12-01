# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 01:29:38 2019

@author: Nicholas
"""

import argparse
import os
import numpy as np
from tqdm import trange
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


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
    parser.add_argument('-cn', '--conv_number', help='convolutional layer depth',
                        type=int, default=3)
    parser.add_argument('-fl', '--filter_length', help='size of filters in hidden convolutional layers',
                        type=int, default=3)
    parser.add_argument('-fb', '--filter_base', help='base number of filters in hidden convolutional layers',
                        type=int, default=9)
    parser.add_argument('-ff', '--filter_factor', help='multiplicative factor of filters in successive layers',
                        type=int, default=9)
    parser.add_argument('-zd', '--z_dimension', help='sample noise dimension',
                        type=int, default=81)
    parser.add_argument('-cd', '--c_dimension', help='sample classification dimension',
                        type=int, default=3)
    parser.add_argument('-ud', '--u_dimension', help='sample uniform dimension',
                        type=int, default=3)
    parser.add_argument('-dlr', '--discriminator_learning_rate', help='learning rate for discriminator',
                        type=float, default=2e-4)
    parser.add_argument('-glr', '--gan_learning_rate', help='learning rate for generator',
                        type=float, default=1e-3)
    parser.add_argument('-ep', '--epochs', help='number of training epochs',
                        type=int, default=32)
    parser.add_argument('-bs', '--batch_size', help='size of batches',
                        type=int, default=169)
    parser.add_argument('-sd', '--random_seed', help='random seed for sample selection and learning',
                        type=int, default=128)
    args = parser.parse_args()
    return (args.verbose, args.restart, args.plot, args.parallel, args.gpu, args.threads,
            args.name, args.lattice_length, args.sample_interval, args.sample_number,
            args.conv_number, args.filter_length, args.filter_base, args.filter_factor,
            args.z_dimension, args.c_dimension, args.u_dimension,
            args.discriminator_learning_rate, args.gan_learning_rate,
            args.epochs, args.batch_size, args.random_seed)


def load_thermal_params(name, lattice_length):
    ''' load thermal parameters '''
    fields = np.load('{}.{}.h.npy'.format(name, lattice_length))
    temps = np.load('{}.{}.t.npy'.format(name, lattice_length))
    return fields, temps


def load_configurations(name, lattice_length):
    ''' load configurations and thermal measurements '''
    conf = np.load('{}.{}.dmp.npy'.format(name, lattice_length))
    thrm = np.load('{}.{}.dat.npy'.format(name, lattice_length))
    return conf, thrm


def scale_configurations(conf):
    ''' scales input configurations '''
    # (-1, 1) -> (0, 1)
    return (conf+1)/2


def index_data_by_sample(data, num_fields, num_temps, indices):
    ''' indexes data '''
    # reorders samples independently for each (h, t) according to indices
    return np.array([[data[i, j, indices[i, j]] for j in range(num_temps)] for i in range(num_fields)])


def load_select_scale_data(name, lattice_length, interval, num_samples, seed, verbose=1):
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
    for i in trange(num_fields, desc='Selecting Samples', disable=not verbose):
        for j in range(num_temps):
                indices[i, j] = np.random.permutation(total_num_samples)[:num_samples]
    # construct selected data subset
    select_conf = scale_configurations(index_data_by_sample(interval_conf, num_fields, num_temps, indices))
    select_thrm = index_data_by_sample(interval_thrm, num_fields, num_temps, indices)
    # save selected data arrays
    np.save('{}.{}.{}.h.npy'.format(name, lattice_length, interval), interval_fields)
    np.save('{}.{}.{}.t.npy'.format(name, lattice_length, interval), interval_temps)
    np.save('{}.{}.{}.{}.{}.conf.npy'.format(name, lattice_length,
                                             interval, num_samples, seed), select_conf)
    np.save('{}.{}.{}.{}.{}.thrm.npy'.format(name, lattice_length,
                                             interval, num_samples, seed), select_thrm)
    return interval_fields, interval_temps, select_conf, select_thrm


def load_data(name, lattice_length, interval, num_samples, seed, verbose=1):
    try:
        # try loading selected data arrays
        fields = np.load('{}.{}.{}.h.npy'.format(name, lattice_length, interval))
        temps = np.load('{}.{}.{}.t.npy'.format(name, lattice_length, interval))
        conf = np.load('{}.{}.{}.{}.{}.conf.npy'.format(name, lattice_length,
                                                               interval, num_samples, seed))
        thrm = np.load('{}.{}.{}.{}.{}.thrm.npy'.format(name, lattice_length,
                                                        interval, num_samples, seed))
        if verbose:
            print(100*'_')
            print('Scaled/selected Ising configurations and thermal parameters/measurements loaded from file')
            print(100*'_')
    except:
        # generate selected data arrays
        (fields, temps,
         conf, thrm) = load_select_scale_data(name, lattice_length,
                                              interval, num_samples, seed, verbose)
        if verbose:
            print(100*'_')
            print('Ising configurations selected/scaled and thermal parameters/measurements selected')
            print(100*'_')
    return fields, temps, conf, thrm


def plot_diagram(data, fields, temps, alias, cmap,
                 name, lattice_length, interval, num_samples,
                 conv_number, filter_length, filter_base, filter_factor,
                 n_dim, c_dim, z_dim, dsc_lr, gan_lr, batch_size):
    # file name parameters
    params = (name, lattice_length, interval, num_samples,
              conv_number, filter_length, filter_base, filter_factor,
              n_dim, c_dim, z_dim, dsc_lr, gan_lr, alias)
    file_name = '{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{:.0e}.{:.0e}.{}.png'.format(*params)
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


def get_final_conv_shape(input_shape, conv_number,
                         filter_length, filter_base, filter_factor):
    ''' calculates final convolutional layer output shape '''
    return tuple(np.array(input_shape[:2])//(filter_length**conv_number))+\
           (input_shape[2]*filter_base*filter_factor**(conv_number-1),)


def get_filter_number(conv_iter, filter_base, filter_factor):
    ''' calculates the filter count for a given convolutional iteration '''
    return filter_base*filter_factor**(conv_iter-1)


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


def sample_uniform(num_rows, dimension):
    ''' uniform sampling '''
    return np.random.uniform(size=(num_rows, dimension))


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
                 filter_length=3, filter_base=9, filter_factor=9,
                 z_dim=81, c_dim=3, u_dim=3,
                 dsc_lr=2e-4, gan_lr=2e-4, batch_size=169):
        ''' initialize model parameters '''
        # convolutional parameters
        # number of convolutions
        self.conv_number = conv_number
        # number of filters for first convolution
        self.filter_base = filter_base
        # multiplicative factor for filters in subsequent convolutions
        self.filter_factor = filter_factor
        # filter side length
        self.filter_length = filter_length
        # set stride to be same as filter size
        self.filter_stride = filter_length
        # convolutional input and output shapes
        self.input_shape = input_shape
        self.final_conv_shape = get_final_conv_shape(self.input_shape, self.conv_number,
                                                     self.filter_length, self.filter_base, self.filter_factor)
        # latent and classification dimensions
        # latent noise dimension
        self.z_dim = z_dim
        # categorical control variable dimension
        self.c_dim = c_dim
        # uniform control variable dimension
        self.u_dim = u_dim
        # discriminator and generator learning rates
        self.dsc_lr = dsc_lr
        self.gan_lr = gan_lr
        # batch size and callbacks
        self.batch_size = batch_size
        self.callbacks = []
        # build full model
        self._build_model()


    def _build_model(self):
        ''' builds each component of the InfoGAN model '''
        self._build_generator()
        self._build_discriminator()
        self._build_auxiliary()
        self._build_gan()


    def _build_generator(self):
        ''' builds generator model '''
        # latent unit gaussian and categorical inputs
        self.z_input = Input(batch_shape=(self.batch_size, self.z_dim), name='z_input')
        self.c_input = Input(batch_shape=(self.batch_size, self.c_dim), name='c_input')
        self.u_input = Input(batch_shape=(self.batch_size, self.u_dim), name='u_input')
        # concatenate features
        x = Concatenate()([self.z_input, self.c_input, self.u_input])
        # dense layer with same feature count as final convolution
        x = Dense(units=np.prod(self.final_conv_shape),
                  kernel_initializer='he_normal',
                  name='gen_dense_0')(x)
        x = LeakyReLU(alpha=0.2, name='gen_dense_lrelu_0')(x)
        # repeated dense layer
        x = Dense(units=np.prod(self.final_conv_shape),
                  kernel_initializer='he_normal',
                  name='gen_dense_1')(x)
        x = LeakyReLU(alpha=0.2, name='gen_dense_lrelu_1')(x)
        # reshape to final convolution shape
        convt = Reshape(target_shape=self.final_conv_shape, name='gen_rshp_0')(x)
        u = 0
        # transform to sample shape with transposed convolutions
        for i in range(self.conv_number-1, 0, -1):
            filter_number = get_filter_number(i, self.filter_base, self.filter_factor)
            convt = Conv2DTranspose(filters=filter_number, kernel_size=self.filter_length,
                                    kernel_initializer='he_normal',
                                    padding='same', strides=self.filter_stride,
                                    name='gen_convt_{}'.format(u))(convt)
            convt = BatchNormalization(name='gen_convt_batchnorm_{}'.format(u))(convt)
            convt = LeakyReLU(name='gen_convt_lrelu_{}'.format(u))(convt)
            u += 1
        self.gen_output = Conv2DTranspose(filters=1, kernel_size=self.filter_length,
                                          kernel_initializer='glorot_uniform', activation='sigmoid',
                                          padding='same', strides=self.filter_stride,
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
        for i in range(1, self.conv_number+1):
            filter_number = get_filter_number(i, self.filter_base, self.filter_factor)
            conv = Conv2D(filters=filter_number, kernel_size=self.filter_length,
                          kernel_initializer='he_normal',
                          padding='valid', strides=self.filter_stride,
                          name='dsc_conv_{}'.format(i))(conv)
            conv = BatchNormalization(name='dsc_conv_batchnorm_{}'.format(i))(conv)
            conv = LeakyReLU(alpha=0.2, name='dsc_conv_lrelu_{}'.format(i))(conv)
        # flatten final convolutional layer
        x = Flatten(name='dsc_fltn_0')(conv)
        # dense layer
        x = Dense(units=np.prod(self.final_conv_shape),
                  kernel_initializer='he_normal',
                  name='dsc_dense_0')(x)
        x = LeakyReLU(alpha=0.2, name='dsc_dense_lrelu_0')(x)
        # the dense layer is saved as a hidden layer
        self.dsc_hidden = x
        # dense layer
        x = Dense(units=128,
                  kernel_initializer='he_normal',
                  name='dsc_dense_1')(x)
        x = LeakyReLU(alpha=0.2, name='dsc_dense_lrelu_1')(x)
        # discriminator classification output (0, 1) -> (fake, real)
        self.dsc_output = Dense(units=1,
                                kernel_initializer='glorot_uniform', activation='sigmoid',
                                name='dsc_output')(x)
        # build discriminator
        self.discriminator = Model(inputs=[self.dsc_input], outputs=[self.dsc_output],
                                   name='discriminator')
        # define optimizer
        self.dsc_opt = Adam(lr=self.dsc_lr, beta_1=0.5)
        # compile discriminator
        self.discriminator.compile(loss='binary_crossentropy', optimizer=self.dsc_opt)


    def _build_auxiliary(self):
        ''' builds auxiliary classification reconstruction model '''
        # initialize with dense layer taking the hidden generator layer as input
        x = Dense(units=128,
                  kernel_initializer='he_normal',
                  name='aux_dense_0')(self.dsc_hidden)
        x = LeakyReLU(alpha=0.2, name='aux_dense_lrelu_0')(x)
        # auxiliary output is a reconstruction of the categorical assignments fed into the generator
        self.aux_c_output = Dense(self.c_dim,
                                  kernel_initializer='glorot_uniform', activation='softmax',
                                  name='aux_output_c')(x)
        self.aux_u_output = Dense(self.u_dim,
                                  kernel_initializer='glorot_uniform', activation='sigmoid',
                                  name='aux_output_u')(x)
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
        self.gan_opt = Adam(lr=self.gan_lr, beta_1=0.5)
        # compile GAN
        self.gan.compile(loss={'discriminator' : 'binary_crossentropy',
                               'auxiliary' : 'categorical_crossentropy',
                               'auxiliary_1' : 'mse'},
                         optimizer=self.gan_opt)
        self.discriminator.trainable = True


    def sample_latent_distribution(self):
        ''' draws samples from the latent gaussian and categorical distributions '''
        z = sample_gaussian(self.batch_size, self.z_dim)
        c = sample_categorical(self.batch_size, self.c_dim)
        u = sample_uniform(self.batch_size, self.u_dim)
        return z, c, u


    def generate(self, verbose=1):
        ''' generate new configurations using samples from the latent distributions '''
        z, c, u = self.sample_latent_distribution()
        return self.generator.predict([z, c, u], batch_size=self.batch_size, verbose=0)


    def discriminate(self, x_batch, verbose=1):
        ''' discriminate input configurations '''
        return self.discriminator.predict(x_batch, batch_size=self.batch_size, verbose=0)


    def get_aux_dist(self, x_batch, verbose=1):
        ''' predict categorical assignments of input configurations '''
        return self.auxiliary.predict(x_batch, batch_size=self.batch_size, verbose=0)


    def model_summaries(self):
        ''' print model summaries '''
        self.generator.summary()
        self.discriminator.summary()
        self.auxiliary.summary()
        self.gan.summary()

    def save_weights(self, name, lattice_length, interval, num_samples, seed):
        ''' save weights to file '''
        params = (name, lattice_length, interval, num_samples,
                  self.conv_number, self.filter_length, self.filter_base, self.filter_factor,
                  self.z_dim, self.c_dim, self.u_dim, self.dsc_lr, self.gan_lr)
        file_name = '{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{:.0e}.{:.0e}.gan.weights.h5'.format(*params)
        self.gan.save_weights(file_name)


    def load_weights(self, name, lattice_length, interval, num_samples, seed):
        ''' load weights from file '''
        params = (name, lattice_length, interval, num_samples,
                  self.conv_number, self.filter_length, self.filter_base, self.filter_factor,
                  self.z_dim, self.c_dim, self.u_dim, self.dsc_lr, self.gan_lr)
        file_name = '{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{:.0e}.{:.0e}.gan.weights.h5'.format(*params)
        self.gan.load_weights(file_name, by_name=True)


class Trainer():
    '''
    InfoGAN trainer
    '''
    def __init__(self, model, num_fields, num_temps, num_samples):
        ''' initialize trainer '''
        self.model = model
        self.num_fields = num_fields
        self.num_temps = num_temps
        self.num_samples = num_samples
        self.dsc_loss_history = []
        self.gan_loss_history = []
        self.ent_cat_loss_history = []
        self.ent_con_loss_history = []


    def training_indices(self):
        num_sq = np.int32(np.sqrt(np.int32(self.num_fields*self.num_temps/self.model.batch_size)))
        sq_length = np.int32(np.sqrt(self.model.batch_size))
        sq_indices = np.stack(np.meshgrid(np.arange(sq_length),
                                         np.arange(sq_length)), axis=-1).reshape(-1, 2)[:, ::-1]
        indices = np.array([[sq_indices+sq_length*np.array([i,j]).reshape(1, -1) for i in range(num_sq)] for j in range(num_sq)])
        flat_indices = np.ravel_multi_index(indices.reshape(-1, 2).T, dims=(self.num_fields, self.num_temps))
        shift_indices = np.concatenate([flat_indices[i::self.model.batch_size] for i in range(self.model.batch_size)])
        return shift_indices


    def reorder_training_data(self, x_train):
        x_train = x_train.reshape(self.num_fields*self.num_temps, self.num_samples, *self.model.input_shape)[self.training_indices()]
        return np.moveaxis(x_train, 0, 1).reshape(self.num_fields*self.num_temps*self.num_samples, *self.model.input_shape)


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
        else:
            # inputs are true samples, so the discrimination targets are of unit value
            target = np.ones(self.model.batch_size).astype(int)
            # discriminator loss
            dsc_loss = self.model.discriminator.train_on_batch(x_batch, target)
        self.dsc_loss_history.append(dsc_loss)


    def rolling_loss_average(self, epoch, num_batches, batch):
            ''' calculate rolling loss averages over batches during training '''
            # catch case where there are no calculated losses yet
            if batch == 0:
                gan_loss = 0
                dsc_loss = 0
                ent_cat_loss = 0
                ent_con_loss = 0
            # calculate rolling average
            else:
                # start index for current epoch
                start = num_batches*epoch
                # stop index for current batch (given epoch)
                stop = num_batches*epoch+batch+1
                gan_loss = np.mean(self.gan_loss_history[start:stop])
                # discriminator trains on false and true samples, so has twice as many losses
                dsc_loss = np.mean(self.dsc_loss_history[2*start:2*stop])
                if self.model.c_dim > 0:
                    ent_cat_loss = np.mean(self.ent_cat_loss_history[start:stop])
                else:
                    ent_cat_loss = 0
                if self.model.z_dim > 0:
                    ent_con_loss = np.mean(self.ent_con_loss_history[start:stop])
                else:
                    ent_con_loss = 0
            return gan_loss, dsc_loss, ent_cat_loss, ent_con_loss


    def fit(self, x_train, num_epochs=1, verbose=1):
        ''' fit model '''
        x_train = self.reorder_training_data(x_train)
        # calculate number of batches
        num_batches = x_train.shape[0]//self.model.batch_size
        # loop through epochs
        for i in range(num_epochs):
            # construct progress bar for current epoch
            batch_range = trange(num_batches, desc='', disable=not verbose)
            # loop through batches
            for j in batch_range:
                # set batch loss description
                batch_loss = self.rolling_loss_average(i, num_batches, j)
                desc = 'Epoch: {}/{} GAN Loss: {:.4f} DSC Loss: {:.4f} CAT Loss: {:.4f} CON Loss: {:.4f}'.format(i, num_epochs, *batch_loss)
                batch_range.set_description(desc)
                # fetch batch
                x_batch = x_train[self.model.batch_size*j:self.model.batch_size*(j+1)]
                # train discriminator on false samples
                self.train_discriminator()
                # train discriminator on true samples
                self.train_discriminator(x_batch)
                # train GAN
                self.train_gan()

if __name__ == '__main__':
    (VERBOSE, RSTRT, PLOT, PARALLEL, GPU, THREADS,
     NAME, N, I, NS,
     CN, FL, FB, FF,
     ZD, CD, UD,
     DLR, GLR,
     EP, BS, SEED) = parse_args()
    np.random.seed(SEED)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import tensorflow as tf
    tf.random.set_seed(SEED)
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
    from tensorflow.keras import backend as K
    from tensorflow.keras.layers import (Input, Flatten, Reshape, Concatenate,
                                         Dense, BatchNormalization, Conv2D, Conv2DTranspose, LeakyReLU)
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.models import Model
    from tensorflow.keras.utils import to_categorical
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
        CM = plt.get_cmap('plasma')

    H, T, CONF, THRM = load_data(NAME, N, I, NS, SEED)
    NH, NT = H.size, T.size
    CONF = CONF.reshape(NH*NT*NS, N, N, 1)
    THRM = THRM.reshape(NH*NT*NS, -1)
    IS = (N, N, 1)

    K.clear_session()
    MDL = InfoGAN(IS, CN, FL, FB, FF, ZD, CD, UD, DLR, GLR, BS)
    TRN = Trainer(MDL, NH, NT, NS)
    if VERBOSE:
        MDL.model_summaries()
    if RSTRT:
        MDL.load_weights(NAME, N, I, NS, SEED)
        TRN.fit(CONF, num_epochs=EP)
        MDL.save_weights(NAME, N, I, NS, SEED)
    else:
        try:
            MDL.load_weights(NAME, N, I, NS, SEED)
        except:
            TRN.fit(CONF, num_epochs=EP)
            MDL.save_weights(NAME, N, I, NS, SEED)
    C = np.concatenate(MDL.get_aux_dist(CONF, VERBOSE), axis=-1).reshape(NH, NT, NS, CD+UD)
    if PLOT:
        DGC = C.mean(2)
        for i in trange(CD+UD, desc='Plotting Diagrams', disable=not VERBOSE):
            if i < CD:
                plot_diagram(DGC[:, :, i], H, T,
                            'c_{}'.format(i), CM,
                            NAME, N, I, NS,
                            CN, FL, FB, FF,
                            ZD, CD, UD, DLR, GLR, BS)
            else:
                plot_diagram(DGC[:, :, i], H, T,
                            'u_{}'.format(i-CD), CM,
                            NAME, N, I, NS,
                            CN, FL, FB, FF,
                            ZD, CD, UD, DLR, GLR, BS)