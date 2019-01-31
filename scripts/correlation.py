# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 01:27:05 2019

@author: Nicholas
"""

import argparse
import os
import pickle
import numpy as np
import numba as nb
from tqdm import tqdm


def parse_args():
    ''' parse command line arguments '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', help='verbose output', action='store_true')
    parser.add_argument('-p', '--parallel', help='parallel run', action='store_true')
    parser.add_argument('-nw', '--workers', help='job worker count', type=int, default=16)
    parser.add_argument('-nt', '--threads', help='threads per worker', type=int, default=1)
    parser.add_argument('-n', '--name', help='simulation name',
                        type=str, default='ising_init')
    parser.add_argument('-ls', '--lattice_size', help='lattice size',
                        type=int, default=4)
    args = parser.parse_args()
    return args.verbose, args.parallel, args.workers, args.threads, args.name, args.lattice_size


def client_info():
    ''' print client info '''
    info = str(CLIENT.scheduler_info)
    info = info.replace('<', '').replace('>', '').split()[6:8]
    print('\n%s %s' % tuple(info))


@nb.njit
def correlation(dmp, corr):
    ''' calculates correlation for sample '''
    for u in range(VIND.shape[-1]):
        i, j, k = VIND[:, u]
        v = np.where(R == D[i, j, k])
        corr[v] += dmp[j]*dmp[k]/NR[v]
    return corr

# main
if __name__ == '__main__':
    VERBOSE, PARALLEL, NWORKER, NTHREAD, NAME, N = parse_args()
    # current working directory and prefix
    CWD = os.getcwd()
    PREF = CWD+'/%s.%d' % (NAME, N)
    # external fields and temperatures
    H = pickle.load(open(PREF+'.h.pickle', 'rb'))
    T = pickle.load(open(PREF+'.t.pickle', 'rb'))
    NH, NT = H.size, T.size

    # load data
    DAT = pickle.load(open(PREF+'.dat.pickle', 'rb'))
    DMP = pickle.load(open(PREF+'.dmp.pickle', 'rb')).reshape(NH, NT, -1, N**2)
    if VERBOSE:
        print('data loaded')
    # sample count
    _, _, NS, _ = DAT.shape

    # lattice indices
    IND = np.array([np.unravel_index(i, dims=(N, N), order='C') for i in range(N**2)])
    # lattice vector basis
    B = [-1, 0, 1]
    # generate lattice vectors for ever direction from base
    BR = np.array([[B[i], B[j]] for i in range(3) for j in range(3)], dtype=np.int8)
    # generate lattice distances
    D = np.zeros((BR.shape[0], N**2, N**2))
    for i in range(BR.shape[0]):
        # displacement vector matrix for sample j
        DVM = IND-(IND+N*BR[i].reshape((1, -1))).reshape((-1, 1, 2))
        # vector of displacements between atoms
        D[i] = np.sqrt(np.sum(np.square(DVM), -1))
    VIND = np.array(np.where((D <= np.max(0.5*N)) & (D > 0)))
    # correlation domain
    R, NR = np.unique(D[tuple(VIND)], return_counts=True)
    pickle.dump(R, open(PREF+'.r.pickle', 'wb'))
    if VERBOSE:
        print('correlation domain dumped')
    # correlation array
    CORR = np.zeros((NH, NT, NS, R.size))

    if PARALLEL:
        os.environ['DASK_ALLOWED_FAILURES'] = '32'
        os.environ['DASK_MULTIPROCESSING_METHOD'] = 'forkserver'
        os.environ['DASK_LOG_FORMAT'] = '\r%(name)s - %(levelname)s - %(message)s'
        from multiprocessing import freeze_support
        from distributed import Client, LocalCluster, progress
        from dask import delayed
        # local cluster
        freeze_support()
        if NWORKER == 1:
            PROC = False
        else:
            PROC = True
        CLUSTER = LocalCluster(n_workers=NWORKER, threads_per_worker=NTHREAD, processes=PROC)
        # start client with local cluster
        CLIENT = Client(CLUSTER)
        # client information
        if VERBOSE:
            client_info()
        # submit futures to client for computation
        OPERS = [delayed(correlation)(DMP[i, j, k], CORR[i, j, k]) \
                 for i in range(NH) for j in range(NT) for k in range(NS)]
        FUTURES = CLIENT.compute(OPERS)
        # progress bar
        if VERBOSE:
            progress(FUTURES)
        # gather results from workers
        RESULTS = CLIENT.gather(FUTURES)
        # assign correlations
        u = 0
        for i in range(NH):
            for j in range(NT):
                for k in range(NS):
                    CORR[i, j, k, :] = RESULTS[u]
                    u += 1
        # close client
        CLIENT.close()
    else:
        if VERBOSE:
            for i in tqdm(range(NH)):
                for j in tqdm(range(NT)):
                    for k in tqdm(range(NS)):
                        CORR[i, j, k] = correlation(DMP[i, j, k], CORR[i, j, k])
        else:
            for i in range(NH):
                for j in range(NT):
                    for k in range(NS):
                        CORR[i, j, k] = correlation(DMP[i, j, k], CORR[i, j, k])
    pickle.dump(CORR, open(PREF+'.corr.pickle', 'wb'))
    if VERBOSE:
        print('\ncorrelations dumped')