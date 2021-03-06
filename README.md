neuralIsing
===========

The 2-dimensional square Ising model is investigated using representation learning techniques to characterize the configurations with latent codes. The intention is that the critical point as well as the crossover region can be identified without the use of a priori information grounded in traditional statistical mechanics.

https://drive.google.com/file/d/1Uw1Rjx8KsY70_0hqdfdeZwY975x8hED9/view

https://arxiv.org/abs/2005.01682

Requirements
------------

- numpy
- numba
- tqdm
- dask (or joblib)
- scikit-learn
- scipy
- tensorflow (2.2)
- tensorlow-addons
- matplotlib

ising.py
--------

This script allows for simulation of the 2-dimensional Ising model across a range of external fields and temperatures. This is done with a standard Monte Carlo (MC) simulation where the MC moves consists of an attempt at flipping a spin. with replica exchange Markov chain Monte Carlo (REMCMC) moves to aid in equilibration. All of the configurations are of the same square shape. The interaction strengths and magnetic moments are tunable. Simulations can be restarted from checkpoints in older simulations.

Use the help flag (-h) to see options.

parse.py
--------

This script simply parses the output from 'ising.py' into numpy arrays. There are flags for using verbose mode (-v), setting the name of simulation whose output will be parsed (-n), and the lattice size for the configurations in the system (-ls).

Use the help flag (-h) to see options.

beta-tcvae.py
-------------

This script uses a variational autoencoder (VAE) to produce latent encodings and reconstructions of the Ising configurations. Both the raw encodings as well as their PCA projections are included. Support for the TC-Beta VAE decomposition is provided.

Use the help flag (-h) to see options.

beta-tccvae.py
-------------

This script uses the conditional variant of the VAE conditioned on the external magnetic field and temperature.

Use the help flag (-h) to see options.

infogan.py
-------------

This script uses an information maximizing generative adversarial network (InfoGAN) to produce latent encodings and a generator for the Ising configurations. Both categorical and uniform latent codes are supported.

Use the help flag (-h) to see options.

infocgan.py
-------------

This script uses the conditional variant of the InfoGAN network conditioned on the external magnetic field and temperature.

Use the help flag (-h) to see options.

data_utils.py
-------------

This script contains utility functions for handling the simulation data.

conv_utils.py
-------------

This script contains utility functions for handling convolutions.

dist_utils.py
-------------

This script contains utility functions for sampling distributions used with InfoGAN and InfoCGAN models.
