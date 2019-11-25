# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 00:03:16 2019

@author: Nicholas
"""

import argparse
import os
import time
import numpy as np
import numba as nb
import itertools as it
from tqdm import tqdm

# --------------
# run parameters
# --------------


def parse_args():
    ''' parse command line arguments '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', help='verbose output mode', action='store_true')
    parser.add_argument('-r', '--restart', help='restart run mode', action='store_true')
    parser.add_argument('-p', '--parallel', help='parallel run mode', action='store_true')
    parser.add_argument('-c', '--client', help='dask client run mode', action='store_true')
    parser.add_argument('-d', '--distributed', help='distributed run mode', action='store_true')
    parser.add_argument('-rd', '--restart_dump', help='restart dump frequency',
                        type=int, default=128)
    parser.add_argument('-rn', '--restart_name', help='restart dump simulation name',
                        type=str, default='ising_init')
    parser.add_argument('-rs', '--restart_step', help='restart dump start step',
                        type=int, default=1024)
    parser.add_argument('-q', '--queue', help='job submission queue',
                        type=str, default='jobqueue')
    parser.add_argument('-a', '--allocation', help='job submission allocation',
                        type=str, default='startup')
    parser.add_argument('-nn', '--nodes', help='job node count',
                        type=int, default=1)
    parser.add_argument('-np', '--procs_per_node', help='number of processors per node',
                        type=int, default=16)
    parser.add_argument('-w', '--walltime', help='job walltime',
                        type=int, default=72)
    parser.add_argument('-m', '--memory', help='job memory (total)',
                        type=int, default=32)
    parser.add_argument('-nw', '--workers', help='job worker count (total)',
                        type=int, default=16)
    parser.add_argument('-nt', '--threads', help='threads per worker',
                        type=int, default=1)
    parser.add_argument('-mt', '--method', help='parallelization method',
                        type=str, default='fork')
    parser.add_argument('-n', '--name', help='simulation name',
                        type=str, default='init')
    parser.add_argument('-ls', '--lattice_size', help='lattice size',
                        type=int, default=8)
    parser.add_argument('-j', '--interaction', help='interaction energy',
                        type=float, default=1.0)
    parser.add_argument('-mm', '--magnetic_moment', help='magnetic moment',
                        type=float, default=1.0)
    parser.add_argument('-hn', '--field_number', help='number of external fields',
                        type=int, default=32)
    parser.add_argument('-hr', '--field_range', help='field range (low and high)',
                        type=float, nargs=2, default=[-2.0, 2.0])
    parser.add_argument('-tn', '--temperature_number', help='number of temperatures',
                        type=int, default=32)
    parser.add_argument('-tr', '--temperature_range', help='temperature range (low and high)',
                        type=float, nargs=2, default=[1.0, 5.0])
    parser.add_argument('-sc', '--sample_cutoff', help='sample recording cutoff',
                        type=int, default=0)
    parser.add_argument('-sn', '--sample_number', help='number of samples to generate',
                        type=int, default=1024)
    parser.add_argument('-rec', '--remcmc_cutoff', help='replica exchange markov chain monte carlo cutoff',
                        type=int, default=0)
    # parse arguments
    args = parser.parse_args()
    # return arguments
    return (args.verbose, args.parallel, args.client, args.distributed, args.restart,
            args.restart_dump, args.restart_name, args.restart_step,
            args.queue, args.allocation, args.nodes, args.procs_per_node,
            args.walltime, args.memory,
            args.workers, args.threads, args.method,
            args.name, args.lattice_size, args.interaction, args.magnetic_moment,
            args.field_number, *args.field_range,
            args.temperature_number, *args.temperature_range,
            args.sample_cutoff, args.sample_number, args.remcmc_cutoff)


def client_info():
    ''' print client info '''
    info = str(CLIENT.scheduler_info)
    info = info.replace('<', '').replace('>', '').split()[6:8]
    print('\n%s %s' % tuple(info))

# -----------------------------
# output file utility functions
# -----------------------------


def file_prefix(i):
    ''' returns filename prefix for simulation '''
    prefix = os.getcwd()+'/%s.%d.%02d' % (NAME, N, i)
    return prefix


def init_output(k):
    ''' initializes output filenames for a sample '''
    # extract field/temperature indices from index
    i, j = np.unravel_index(k, dims=(NH, NT), order='C')
    dat = file_prefix(i)+'.%02d.dat' % j
    dmp = file_prefix(i)+'.%02d.dmp' % j
    # clean old output files if they exist
    if os.path.isfile(dat):
        os.remove(dat)
    if os.path.isfile(dmp):
        os.remove(dmp)
    return dat, dmp


def init_outputs():
    ''' initializes output filenames for all samples '''
    if VERBOSE:
        print('initializing outputs')
        print('--------------------')
    return [init_output(k) for k in range(NS)]


def init_header(k, output):
    ''' writes header for a sample '''
    # extract pressure/temperature indices from index
    i, j = np.unravel_index(k, dims=(NH, NT), order='C')
    with open(output[0], 'w') as dat_out:
        dat_out.write('# ---------------------\n')
        dat_out.write('# simulation parameters\n')
        dat_out.write('# ---------------------\n')
        dat_out.write('# nsmpl:    %d\n' % NSMPL)
        dat_out.write('# cutoff:   %d\n' % CUTOFF)
        dat_out.write('# mod:      %d\n' % MOD)
        dat_out.write('# nswps:    %d\n' % NSWPS)
        dat_out.write('# seed:     %d\n' % SEED)
        dat_out.write('# ---------------------\n')
        dat_out.write('# material properties\n')
        dat_out.write('# ---------------------\n')
        dat_out.write('# size:     %d\n' % N)
        dat_out.write('# inter:    %f\n' % J)
        dat_out.write('# field:    %f\n' % H[i])
        dat_out.write('# temp:     %f\n' % T[j])
        dat_out.write('# --------------------\n')
        dat_out.write('# | ener | mag | acc |\n')
        dat_out.write('# --------------------\n')


def init_headers():
    ''' writes headers for all samples '''
    if DASK:
        operations = [delayed(init_header)(k, OUTPUT[k]) for k in range(NS)]
        futures = CLIENT.compute(operations)
        if VERBOSE:
            print('initializing headers')
            print('--------------------')
            progress(futures)
    elif PARALLEL:
        operations = [delayed(init_header)(k, OUTPUT[k]) for k in range(NS)]
        futures = Parallel(n_jobs=NTHREAD, backend='threading', verbose=VERBOSE)(operations)
    else:
        if VERBOSE:
            print('initializing headers')
            print('--------------------')
            for k in tqdm(range(NS)):
                init_header(k, OUTPUT[k])
        else:
            for k in range(NS):
                init_header(k, OUTPUT[k])


def write_dat(output, state):
    ''' writes properties to dat file '''
    dat = output[0]
    ener, mag, acc = state[1:4]
    with open(dat, 'a') as dat_out:
        dat_out.write('%.4E %.4E %.4E\n' % (ener/N**2, mag/N**2, acc))


def write_dmp(output, state):
    ''' writes configurations to dmp file '''
    dmp = output[1]
    config = state[0]
    with open(dmp, 'ab') as dmp_out:
        np.savetxt(dmp_out, config.reshape(1,-1))


def write_output(output, state):
    ''' writes output for a sample '''
    write_dat(output, state)
    write_dmp(output, state)


def write_outputs():
    ''' writes outputs for all samples '''
    if DASK:
        operations = [delayed(write_output)(OUTPUT[k], STATE[k]) for k in range(NS)]
        futures = CLIENT.compute(operations)
        if VERBOSE:
            print('\n---------------')
            print('writing outputs')
            print('---------------')
            progress(futures)
    elif PARALLEL:
        operations = [delayed(write_output)(OUTPUT[k], STATE[k]) for k in range(NS)]
        futures = Parallel(n_jobs=NTHREAD, backend='threading', verbose=VERBOSE)(operations)
    else:
        if VERBOSE:
            print('writing outputs')
            print('---------------')
            for k in tqdm(range(NS)):
                write_output(OUTPUT[k], STATE[k])
        else:
            for k in range(NS):
                write_output(OUTPUT[k], STATE[k])


def consolidate_outputs():
    ''' consolidates outputs across samples '''
    if VERBOSE:
        print('---------------------')
        print('consolidating outputs')
        print('---------------------')
    dat = [OUTPUT[k][0] for k in range(NS)]
    dmp = [OUTPUT[k][1] for k in range(NS)]
    for i in range(NH):
        with open(file_prefix(i)+'.dat', 'w') as dat_out:
            for j in range(NT):
                k = np.ravel_multi_index((i, j), (NH, NT), order='C')
                with open(dat[k], 'r') as dat_in:
                    for line in dat_in:
                        dat_out.write(line)
    for i in range(NH):
        with open(file_prefix(i)+'.dmp', 'w') as dmp_out:
            for j in range(NT):
                k = np.ravel_multi_index((i, j), (NH, NT), order='C')
                with open(dmp[k], 'r') as dmp_in:
                    for line in dmp_in:
                        dmp_out.write(line)
    if VERBOSE:
        print('cleaning files')
        print('--------------')
    for k in range(NS):
        os.remove(dat[k])
        os.remove(dmp[k])

# ------------------------------------------------
# sample initialization and information extraction
# ------------------------------------------------


@nb.njit
def extract(config, h):
    ''' calculates the magentization and energy of a configuration '''
    # magnetization
    mag = M*np.sum(config)
    ener = 0
    # loop through lattice
    for i in range(N):
        for j in range(N):
            s = config[i, j]
            nn = config[(i+1)%N, j]+config[i,(j+1)%N]+config[(i-1)%N, j]+config[i,(j-1)%N]
            ener -= J*s*nn
    # correction factor
    ener = 0.5*ener
    # add in magnetic contribution to energy
    ener -= h*mag
    return ener, mag


@nb.jit
def init_sample(k):
    ''' initializes sample '''
    # fetch external field strength
    i, _ = np.unravel_index(k, dims=(NH, NT), order='C')
    h = H[i]
    # generate random ising configuration
    config = np.random.choice([-1, 1], size=(N,N))
    # extract energies and magnetizations
    ener, mag = extract(config, h)
    # set acceptations
    acc = 0.0
    # return configuration
    return [config, ener, mag, acc]


def init_samples():
    ''' initializes all samples '''
    if DASK:
        operations = [delayed(init_sample)(k) for k in range(NS)]
        futures = CLIENT.compute(operations)
        if VERBOSE:
            print('initializing samples')
            print('--------------------')
            progress(futures)
            print('\n')
    elif PARALLEL:
        operations = [delayed(init_sample)(k) for k in range(NS)]
        futures = Parallel(n_jobs=NTHREAD, backend='threading', verbose=VERBOSE)(operations)
    else:
        if VERBOSE:
            print('initializing samples')
            print('--------------------')
            futures = [init_sample(k) for k in tqdm(range(NS))]
        else:
            futures = [init_sample(k) for k in range(NS)]
    return futures

# ----------------
# monte carlo move
# ----------------


@nb.njit
def spin_flip_mc(config, h, t, nts, nas):
    ''' spin flip monte carlo '''
    nts += 1
    ener, _ = extract(config, h)
    u, v = np.random.randint(0, N, size=2)
    # config[u, v] *= -1
    # nener, _ = extract(config, h)
    # de = nener-ener
    s = config[u, v]
    nn = config[(u+1)%N, v]+config[u, (v+1)%N]+config[(u-1)%N, v]+config[u, (v-1)%N]
    de = 2*s*(J*nn+h)
    if de < 0 or np.random.rand() < np.exp(-de/t):
        # update acceptations
        nas += 1
        config[u, v] *= -1
    # else:
        # revert spin
        # config[u, v] *= -1
    # return spins and tries/acceptations
    return config, nts, nas

# ---------------------
# monte carlo procedure
# ---------------------


@nb.jit
def gen_sample(k, state):
    ''' generates a monte carlo sample '''
    # initialize lammps object
    i, j = np.unravel_index(k, dims=(NH, NT), order='C')
    h, t = H[i], T[j]
    config = state[0]
    nts, nas = 0, 0
    # loop through monte carlo moves
    for _ in it.repeat(None, MOD):
        config, nts, nas = spin_flip_mc(config, h, t, nts, nas)
    # extract system properties
    ener, mag = extract(config, h)
    # acceptation ratio
    acc = nas/nts
    # return state
    return [config, ener, mag, acc]


def gen_samples():
    ''' generates all monte carlo samples '''
    if DASK:
        # list of delayed operations
        operations = [delayed(gen_sample)(k, STATE[k]) for k in range(NS)]
        # submit futures to client
        futures = CLIENT.compute(operations)
        # progress bar
        if VERBOSE:
            print('----------------------')
            print('performing monte carlo')
            print('----------------------')
            progress(futures)
    elif PARALLEL:
        operations = [delayed(gen_sample)(k, STATE[k]) for k in range(NS)]
        futures = Parallel(n_jobs=NTHREAD, backend='threading', verbose=VERBOSE)(operations)
    else:
        # loop through pressures
        if VERBOSE:
            print('----------------------')
            print('performing monte carlo')
            print('----------------------')
            futures = [gen_sample(k, STATE[k]) for k in tqdm(range(NS))]
        else:
            futures = [gen_sample(k, STATE[k]) for k in range(NS)]
    return futures


# -----------------------------------------
# replica exchange markov chain monte carlo
# -----------------------------------------

@nb.jit
def replica_exchange():
    ''' performs parallel tempering across temperature samples for each field strength '''
    # catalog swaps
    swaps = 0
    # loop through fields
    for u in range(NH):
        # loop through reference temperatures from high to low
        for v in range(NT-1, -1, -1):
            # loop through temperatures from low to current reference temperature
            for w in range(v):
                # extract index from each field/temperature index pair
                i = np.ravel_multi_index((u, v), (NH, NT), order='C')
                j = np.ravel_multi_index((u, w), (NH, NT), order='C')
                # calculate energy difference
                de = STATE[i][1]-STATE[j][1]
                # enthalpy difference
                dh = de*(1./T[v]-1./T[w])
                # metropolis criterion
                if np.random.rand() <= np.exp(dh):
                    swaps += 1
                    # swap states
                    STATE[j], STATE[i] = STATE[i], STATE[j]
    if VERBOSE:
        if PARALLEL:
            print('\n-------------------------------')
        print('%d replica exchanges performed' % swaps)
        print('-------------------------------')

# -------------
# restart files
# -------------


def load_samples_restart():
    ''' initialize samples with restart file '''
    if VERBOSE:
        if PARALLEL:
            print('\n----------------------------------')
        print('loading samples from previous dump')
        print('----------------------------------')
    return list(np.load(os.getcwd()+'/%s.%d.rstrt.%d.npy' % (RENAME, N, RESTEP), allow_pickle=True))


def dump_samples_restart():
    ''' save restart state '''
    if VERBOSE:
        if PARALLEL:
            print('\n---------------')
        print('dumping samples')
    np.save(os.getcwd()+'/%s.%d.rstrt.%d.npy' % (NAME, N, STEP+1), STATE)

# ----
# main
# ----

if __name__ == '__main__':

    (VERBOSE, PARALLEL, DASK, DISTRIBUTED, RESTART,
     REFREQ, RENAME, RESTEP,
     QUEUE, ALLOC, NODES, PPN,
     WALLTIME, MEM,
     NWORKER, NTHREAD, MTHD,
     NAME, N, J, M,
     NH, LH, HH,
     NT, LT, HT,
     CUTOFF, NSMPL, RECUTOFF) = parse_args()

    # set random seed
    SEED = 256
    np.random.seed(SEED)
    # processing or threading
    PROC = (NWORKER != 1)
    # ensure all flags are consistent
    if DISTRIBUTED and not DASK:
        DASK = 1
    if DASK and not PARALLEL:
        PARALLEL = 1

    # number of spinflips per sweep
    MOD = 8*N*N
    # number of simulations
    NS = NH*NT
    # total number of monte carlo sweeps
    NSWPS = NSMPL*MOD

    # external field
    H = np.linspace(LH, HH, NH, dtype=np.float32)
    # temperature
    T = np.linspace(LT, HT, NT, dtype=np.float32)

    # dump external fields and temperatures
    np.save('%s.%d.h.npy' % (NAME, N), H)
    np.save('%s.%d.t.npy' % (NAME, N), T)

    # -----------------
    # initialize client
    # -----------------

    if PARALLEL:
        from multiprocessing import freeze_support
    if not DASK:
        from joblib import Parallel, delayed
    if DASK:
        os.environ['DASK_ALLOWED_FAILURES'] = '64'
        os.environ['DASK_WORK_STEALING'] = 'True'
        os.environ['DASK_MULTIPROCESSING_METHOD'] = MTHD
        os.environ['DASK_LOG_FORMAT'] = '\r%(name)s - %(levelname)s - %(message)s'
        from distributed import Client, LocalCluster, progress
        from dask import delayed
    if DISTRIBUTED:
        from dask_jobqueue import PBSCluster

    if PARALLEL:
        freeze_support()
        if DASK and not DISTRIBUTED:
            # construct local cluster
            CLUSTER = LocalCluster(n_workers=NWORKER, threads_per_worker=NTHREAD, processes=PROC)
            # start client with local cluster
            CLIENT = Client(CLUSTER)
            # display client information
            if VERBOSE:
                client_info()
        if DASK and DISTRIBUTED:
            # construct distributed cluster
            CLUSTER = PBSCluster(queue=QUEUE, project=ALLOC,
                                 resource_spec='nodes=%d:ppn=%d' % (NODES, PPN),
                                 walltime='%d:00:00' % WALLTIME,
                                 processes=NWORKER, cores=NTHREAD*NWORKER, memory=str(MEM)+'GB',
                                 local_dir=os.getcwd())
            CLUSTER.start_workers(1)
            # start client with distributed cluster
            CLIENT = Client(CLUSTER)
            while 'processes=0 cores=0' in str(CLIENT.scheduler_info):
                time.sleep(5)
                if VERBOSE:
                    client_info()

    # -----------
    # monte carlo
    # -----------

    # define output file names
    OUTPUT = init_outputs()
    if CUTOFF < NSMPL:
        init_headers()
    # initialize simulation
    if RESTART:
        STATE = load_samples_restart()
        replica_exchange()
    else:
        if DASK:
            STATE = CLIENT.gather(init_samples())
        else:
            STATE = init_samples()
    # loop through to number of samples that need to be collected
    for STEP in tqdm(range(NSMPL)):
        if VERBOSE and DASK:
            client_info()
        # generate samples
        STATE[:] = gen_samples()
        # generate mc parameters
        if (STEP+1) > CUTOFF:
            # write data
            write_outputs()
        if DASK:
            # gather results from cluster
            STATE[:] = CLIENT.gather(STATE)
        if (STEP+1) % REFREQ == 0:
            # save state for restart
            dump_samples_restart()
        # replica exchange markov chain mc
        if (STEP+1) != NSMPL and (STEP+1) > RECUTOFF:
            replica_exchange()
    if DASK:
        # terminate client after completion
        CLIENT.close()
    # consolidate output files
    if CUTOFF < NSMPL:
        consolidate_outputs()
