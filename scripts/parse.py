# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 17:38:15 2019

@author: Nicholas
"""

import argparse
import os
import numpy as np
from tqdm import tqdm

def parse_args():
    ''' parse command line arguments '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', help='verbose output', action='store_true')
    parser.add_argument('-n', '--name', help='simulation name',
                        type=str, default='ising_init')
    parser.add_argument('-ls', '--lattice_size', help='lattice size',
                        type=int, default=4)
    args = parser.parse_args()
    return args.verbose, args.name, args.lattice_size

VERBOSE, NAME, N = parse_args()
# current working directory and prefix
CWD = os.getcwd()
PREF = CWD+'/%s.%d' % (NAME, N)
# external fields and temperatures
H = np.load(PREF+'.h.npy')
T = np.load(PREF+'.t.npy')
NH, NT = H.size, T.size

# file listing
DATFLS = [PREF+'.%02d.dat' % i for i in range(NH)]
DMPFLS = [PREF+'.%02d.dmp' % i for i in range(NH)]
# parse data
if VERBOSE:
    DAT = np.array([np.loadtxt(DATFLS[i], dtype=np.float32).reshape(NT, -1, 3) for i in tqdm(range(NH))])
    DMP = np.array([np.loadtxt(DMPFLS[i], dtype=np.float32).reshape(NT, -1, N, N) for i in tqdm(range(NH))])
else:
    DAT = np.array([np.loadtxt(DATFLS[i], dtype=np.float32).reshape(NT, -1, 3) for i in range(NH)])
    DMP = np.array([np.loadtxt(DMPFLS[i], dtype=np.float32).reshape(NT, -1, N, N) for i in range(NH)])
np.save(PREF+'.dat.npy', DAT)
np.save(PREF+'.dmp.npy', DMP)
if VERBOSE:
    print('all data dumped')