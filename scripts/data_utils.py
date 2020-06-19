import os
import numpy as np


def load_thermal_params(name, lattice_length):
    ''' load thermal parameters '''
    fields = np.load(os.getcwd()+'/{}.{}.h.npy'.format(name, lattice_length))
    temps = np.load(os.getcwd()+'/{}.{}.t.npy'.format(name, lattice_length))
    return fields, temps


def load_configurations(name, lattice_length):
    ''' load configurations and thermal measurements '''
    conf = np.load(os.getcwd()+'/{}.{}.dmp.npy'.format(name, lattice_length))
    thrm = np.load(os.getcwd()+'/{}.{}.dat.npy'.format(name, lattice_length))
    return conf, thrm


def sample_gaussian(beta):
    ''' samples a point in a multivariate gaussian distribution '''
    mu, logvar = beta
    return np.random.normal(mu, np.exp(0.5*logvar))


def scale_configurations(conf):
    ''' scales input configurations '''
    # (-1, 1) -> (0, 1)
    return (conf+1)/2


def unscale_configurations(conf):
    ''' unscales input configurations '''
    # (0, 1) -> (-1, 1)
    return 2*conf-1


def binary_crossentropy(x, x_hat, scaled):
    ''' binary crossentropy error '''
    eps = 1e-4
    ns, _, _, _ = x.shape
    if not scaled:
        x = scale_configurations(x)
        x_hat = scale_configurations(x_hat)
    bc = -1*(x*np.log(x_hat+eps)+(1.-x)*np.log(1.-x_hat+eps)).reshape(ns, -1)
    return bc


def binary_crossentropy_accuracy(x, x_hat, scaled):
    ''' binary accuracy '''
    bc = binary_crossentropy(x, x_hat, scaled)
    ba = np.exp(-bc)
    return np.stack((bc.mean(1), bc.std(1)), axis=-1), np.stack((ba.mean(1), ba.std(1)), axis=-1)


def shuffle_samples(data, num_fields, num_temps, indices):
    ''' shuffles data by sample index independently for each thermal parameter combination '''
    # reorders samples independently for each (h, t) according to indices
    return np.array([[data[i, j, indices[i, j]] for j in range(num_temps)] for i in range(num_fields)])


def load_select_scale_data(name, lattice_length, interval, num_samples, scaled, seed, verbose=False):
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
    if verbose:
        print(100*'_')
    for i in trange(num_fields, desc='Selecting Samples', disable=not verbose):
        for j in range(num_temps):
                indices[i, j] = np.random.permutation(total_num_samples)[:num_samples]
    # construct selected data subset
    select_conf = shuffle_samples(interval_conf, num_fields, num_temps, indices)
    if scaled:
        select_conf = scale_configurations(select_conf).astype(np.int8)
    select_thrm = shuffle_samples(interval_thrm, num_fields, num_temps, indices)
    # save selected data arrays
    np.save(os.getcwd()+'/{}.{}.{}.h.npy'.format(name, lattice_length, interval), interval_fields)
    np.save(os.getcwd()+'/{}.{}.{}.t.npy'.format(name, lattice_length, interval), interval_temps)
    np.save(os.getcwd()+'/{}.{}.{}.{}.{:d}.{}.conf.npy'.format(name, lattice_length,
                                                               interval, num_samples,
                                                               scaled, seed), select_conf)
    np.save(os.getcwd()+'/{}.{}.{}.{}.{:d}.{}.thrm.npy'.format(name, lattice_length,
                                                               interval, num_samples,
                                                               scaled, seed), select_thrm)
    return interval_fields, interval_temps, select_conf, select_thrm


def load_data(name, lattice_length, interval, num_samples, scaled, seed, verbose=False):
    try:
        # try loading selected data arrays
        fields = np.load(os.getcwd()+'/{}.{}.{}.h.npy'.format(name, lattice_length, interval))
        temps = np.load(os.getcwd()+'/{}.{}.{}.t.npy'.format(name, lattice_length, interval))
        conf = np.load(os.getcwd()+'/{}.{}.{}.{}.{:d}.{}.conf.npy'.format(name, lattice_length,
                                                                          interval, num_samples,
                                                                          scaled, seed))
        thrm = np.load(os.getcwd()+'/{}.{}.{}.{}.{:d}.{}.thrm.npy'.format(name, lattice_length,
                                                                          interval, num_samples,
                                                                          scaled, seed))
        if verbose:
            print(100*'_')
            print('Scaled/selected Ising configurations and thermal parameters/measurements loaded from file')
            print(100*'_')
    except:
        # generate selected data arrays
        (fields, temps,
         conf, thrm) = load_select_scale_data(name, lattice_length,
                                              interval, num_samples,
                                              scaled, seed, verbose)
        if verbose:
            print(100*'_')
            print('Ising configurations selected/scaled and thermal parameters/measurements selected')
            print(100*'_')
    return fields, temps, conf, thrm


def save_output_data(data, alias, name, lattice_length, interval, num_samples, scaled, seed, prfx):
    ''' save output data from model '''
    # file parameters
    params = (name, lattice_length, interval, num_samples, scaled, seed)
    file_name = os.getcwd()+'/{}.{}.{}.{}.{:d}.{}.'.format(*params)+prfx+'.{}.npy'.format(alias)
    np.save(file_name, data)


def get_training_indices(num_fields, num_temps, batch_size):
    ''' retrieve class-balancing training indices '''
    # number of square subsectors in (h, t) space
    n_sr = np.int32(num_fields*num_temps/batch_size)
    # side length of (h, t) space in subsectors
    n_sr_l = np.int32(np.sqrt(n_sr))
    # side length of subsector
    sr_l = np.int32(np.sqrt(batch_size))
    # indices for top left subsector
    sr_indices = np.stack(np.meshgrid(np.arange(sr_l),
                                        np.arange(sr_l)), axis=-1).reshape(-1, 2)[:, ::-1]
    # all indices for subsectors
    indices = np.array([[sr_indices+sr_l*np.array([i,j]).reshape(1, -1) for i in range(n_sr_l)] for j in range(n_sr_l)])
    # flattened subsector indices
    flat_indices = np.ravel_multi_index(indices.reshape(-1, 2).T, dims=(num_fields, num_temps))
    # shuffle indices within each subsector
    for i in range(n_sr):
        flat_indices[batch_size*i:batch_size*(i+1)] = np.random.permutation(flat_indices[batch_size*i:batch_size*(i+1)])
    # shift indices to balance batches by subsector
    shift_indices = np.concatenate([flat_indices[i::batch_size] for i in range(batch_size)])
    return shift_indices


def randomly_order_training_data(x_train, num_fields, num_temps, num_samples, input_shape):
    ''' reorder training data by random indices '''
    indices = np.random.permutation(num_fields*num_temps*num_samples)
    return x_train.reshape(num_fields*num_temps*num_samples, *input_shape)[indices]


def reorder_training_data(x_train, num_fields, num_temps, num_samples, input_shape):
    ''' reorder training data by class-balancing indices '''
    x_train = x_train.reshape(num_fields*num_temps, num_samples, *input_shape)[get_training_indices(num_fields, num_temps, batch_size)]
    return np.moveaxis(x_train, 0, 1).reshape(num_fields*num_temps*num_samples, *input_shape)


def extract_unique_data(x_train, num_fields, num_temps, num_samples, input_shape):
    ''' extract unique samples from data '''
    x_train = np.unique(x_train.reshape(num_fields*num_temps*num_samples, *input_shape), axis=0)
    return x_train


def draw_random_batch(x_train, batch_size):
    ''' draws random batch from data '''
    indices = np.random.permutation(x_train.shape[0])[:batch_size]
    return x_train[indices]


def draw_indexed_batch(x_train, batch_size, j):
    ''' draws batch j '''
    return x_train[batch_size*j:batch_size*(j+1)]


def randomly_order_training_data_thermal(x_train, t_train, num_fields, num_temps, num_samples, input_shape, t_dim):
    ''' reorder training data by random indices '''
    indices = np.random.permutation(num_fields*num_temps*num_samples)
    return (x_train.reshape(num_fields*num_temps*num_samples, *input_shape)[indices],
            t_train.reshape(num_fields*num_temps*num_samples, t_dim)[indices])


def reorder_training_data_thermal(x_train, t_train, num_fields, num_temps, num_samples, input_shape, t_dim):
    ''' reorder training data by class-balancing indices '''
    indices = get_training_indices()
    x_train = x_train.reshape(num_fields*num_temps, num_samples, *input_shape)[indices]
    t_train = t_train.reshape(num_fields*num_temps, num_samples, t_dim)[indices]
    return (np.moveaxis(x_train, 0, 1).reshape(num_fields*num_temps*num_samples, *input_shape),
            np.moveaxis(t_train, 0, 1).reshape(num_fields*num_temps*num_samples, t_dim))


def extract_unique_data_thermal(x_train, t_train, num_fields, num_temps, num_samples, input_shape, t_dim):
    ''' extract unique samples from data '''
    x_train, indices = np.unique(x_train.reshape(num_fields*num_temps*num_samples, *input_shape), return_index=True, axis=0)
    t_train = t_train.reshape(num_fields*num_temps*num_samples, t_dim)[indices]
    return x_train, t_train


def draw_random_batch_thermal(x_train, t_train, batch_size):
    ''' draws random batch from data '''
    indices = np.random.permutation(x_train.shape[0])[:batch_size]
    return x_train[indices].astype(np.float32), t_train[indices].astype(np.float32)


def draw_indexed_batch_thermal(x_train, t_train, batch_size, j):
    ''' draws batch j '''
    ind = np.random.permutation(batch_size)
    return x_train[batch_size*j:batch_size*(j+1)].astype(np.float32)[ind], t_train[batch_size*j:batch_size*(j+1)].astype(np.float32)[ind]