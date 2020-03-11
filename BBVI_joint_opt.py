from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import matplotlib.pyplot as plt

import jax
import jax.numpy as np
# Current convention is to import original numpy as "onp"
import numpy as onp


def bbvi(logprob, N, num_samples, n_hyperparams):
    """Implements http://arxiv.org/abs/1401.0118, and uses the
    local reparameterization trick from http://arxiv.org/abs/1506.02557

    Structure of function taken from: https://github.com/HIPS/autograd/blob/master/examples/black_box_svi.py
	
	inputs: 
	logprob : joint likelihood function - Form of this function must follow a convention -
			 - first argument is variable being sampled over (x)
			 - second argument is an indicator for iteration number (t)
			 - third argument is a vector of hyperparams we wish to also optimize (hyperparams)
			
	N: number of variational parameters
	num_samples: Number of samples 
	n_hyperparams: number of hyperparams to jointly optimize 

	outputs:
	Variational_objective: the elbo
	gradient: Gradient of elbo
	unpack_params: optional function which will decompose mean, variance, and hyperparams from vector params
    """


    def unpack_params(params):
        # Variational dist is a diagonal Gaussian followed by hyperparameters
        mean, log_std, hyperparams = params[:N], params[N:-n_hyperparams], params[-n_hyperparams:]

        return mean, log_std, hyperparams

    def gaussian_entropy(log_std):
        return 0.5 * N * (1.0 + np.log(2*np.pi)) + np.sum(log_std)

    rs = npr.RandomState(0)
    def variational_objective(params,t):
        """Provides a stochastic estimate of the variational lower bound."""
        mean, log_std, hyperparams = unpack_params_E(params) 
        samples = rs.randn(num_samples, N) * np.exp(log_std) + mean #Generate samples using reparam trick
        lower_bound = gaussian_entropy(log_std) + np.mean(logprob(samples,t, hyperparms)) #return elbo and hyperparams (loadings and length scale)

        return -lower_bound

    gradient = grad(variational_objective)

    return variational_objective, gradient, unpack_params

