from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import matplotlib.pyplot as plt

import jax
import jax.numpy as np
from jax import random
from jax import grad, jit, vmap
# Current convention is to import original numpy as "onp"
import numpy as onp




def softplus(x):
	lt_34 = (x >= 34)
	gt_n37 = (x <= -30.8)
	neither_nor = np.logical_not(np.logical_or(lt_34, gt_n37))
	rval = np.where(gt_n37, 0., x)
	return np.where(neither_nor, np.log(1 + np.exp(x)), rval)



@jax.custom_transforms
def safe_logsoftplus(x, up_limit=30, low_limit = -30):
	x = np.array(x)
	x[x>up_limit] = np.log(x[x>up_limit])
	x[(x<up_limit) & (x>low_limit)] = np.log(softplus(x[(x<up_limit) & (x>low_limit)]))
	#x[x<low_limit] = x[x<low_limit]
	return x
def safe_logsoftplus_vjp(ans, x, low_limit = -30):
	x_shape = x.shape
	operator = np.ones(x.shape)
	operator[x>low_limit] =  1/ ((1+np.exp(-x[x>low_limit]))*softplus(x[x>low_limit]))
	return lambda g: np.full(x_shape,g)* operator
	#return lambda g: np.full(x_shape, g) * 1/ ((1+np.exp(-x))*safe_softplus(x))
jax.defvjp_all(safe_logsoftplus, safe_logsoftplus_vjp)



def make_cov(N, rh, len_sc):
  M1 = np.array([np.arange(N)])- np.transpose(np.array([np.arange(N)]))
  K = rh*np.exp(-(np.square(M1)/(2*np.square(len_sc))))
  return K



def bbvi(logprob, N, num_samples):
    """Implements http://arxiv.org/abs/1401.0118, and uses the
    local reparameterization trick from http://arxiv.org/abs/1506.02557

    Structure of function taken from: https://github.com/HIPS/autograd/blob/master/examples/black_box_svi.py
	
	inputs: 
	logprob : joint likelihood function - Form of this function must follow a convention -
			 - first argument is variable being sampled over (x)
			 - second argument is an indicator for iteration number (t)
			 - third argument is a vector of hyperparams we wish to also optimize (hyperparams)
			
	N: number of variational parameters divided by 2
	num_samples: Number of samples 
	--- think i don't need this: n_hyperparams: number of hyperparams to jointly optimize ------

	outputs:
	Variational_objective: the elbo
	gradient: Gradient of elbo
	unpack_params: optional function which will decompose mean, variance, and hyperparams from vector params
    """


    def unpack_params(params):
        # Variational dist is a diagonal Gaussian followed by hyperparameters
        mean, log_std, hyperparams = params[:N], params[N:2*N], params[2*N:]

        return mean, log_std, hyperparams

    def gaussian_entropy(log_std):
        return 0.5 * N * (1.0 + np.log(2*np.pi)) + np.sum(log_std)

    

    def variational_objective(params, subkey):
        """Provides a stochastic estimate of the variational lower bound."""
        mean, log_std, hyperparams = unpack_params(params) 
        samples = random.normal(subkey, shape = [num_samples, N], dtype=np.float64) * np.exp(log_std) + mean #Generate samples using reparam trick
        ### use jax vmap to calculate over samples here
        #handle for batched params
        logprob_samp  = lambda x: logprob(x, hyperparams)
        batched_logprob = vmap(logprob_samp)
        lower_bound = gaussian_entropy(log_std) + np.mean(batched_logprob(samples)) #return elbo and hyperparams (loadings and length scale)

        return -lower_bound

    #gradient = jit(grad(variational_objective))

    return variational_objective, unpack_params#, gradient, unpack_params



def calc_gp_prior(x_samps,):
	'''
	Calculates the log prior Can be used for all BBVI approaches to fourier GP (gauss, poiss, binom, negbinom)
	'''
	#set 
	#print(cdiag[i*int(samplength/n_latents):i*int(samplength/n_latents)+int(samplength/n_latents)].shape)
	logprior = -(1/2)*(np.sum(np.square(x_samp)/cdiaglat,axis=1)+(1/2)*np.sum(np.linalg.slogdet(cdiaglat)))  
	total_prior = logprior + total_prior

	return total_prior



