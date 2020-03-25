from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import jax
import jax.numpy as np
# Current convention is to import original numpy as "onp"
import numpy as onp
from jax import grad, jit, vmap
from jax.experimental import optimizers


import matplotlib.pyplot as plt
import matplotlib
matplotlib.interactive(True)

from CA.CA import CA_Emissions
from CA.misc import bbvi



obj = CA_Emissions(tau = .05, Gauss_sigma = 0.01, T = 1000) 

rate = np.sin(np.arange(0,10,.01))

samp_data = obj.sample_data(rate+1)


def calc_gp_prior(x_samps, K):
	'''
	Calculates the GP log prior (time domain)
	x_samples should be nsamples by T
	K is T by T
	'''

	Kinv = np.linalg.inv(K)
	logprior -(1/2)*(np.einsum('ij,ij->i',np.matmul(x_samp,Kinv), x_samp)+ np.linalg.slogdet(K)[1]) 

  return log_prior


def log_joint(x_samples, t,  hyperparams, obj):

	ll =obj.log_likelihood(x_samples)
	log_prior = calc_gp_prior(x_samples, K)
	return log_prior + ll


logprob = lambda samples, t, hyperparams: log_joint(samples, t, hyperparams)








