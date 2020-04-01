from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import jax
import jax.numpy as np
# Current convention is to import original numpy as "onp"
import numpy as onp
from jax import grad, jit, vmap
from jax.experimental import optimizers
from jax import random
from jax import jacfwd

import matplotlib.pyplot as plt
import matplotlib
matplotlib.interactive(True)

from CA.CA import CA_Emissions
#from CA.misc import bbvi, make_cov



ca_obj = CA_Emissions(AR_params = [.05], Gauss_sigma = 0.01, Tps = 1000) 

rate = np.sin(np.arange(0,10,.01))+1

samp_data = ca_obj.sample_data(rate)

ca_obj.set_data(samp_data)



# def normal_sample(key, shape):
#     """Convenience function for quasi-stateful RNG."""
#     new_key, sub_key = random.split(key)
#     return new_key, random.normal(sub_key, shape)


# normal_sample = jax.jit(normal_sample, static_argnums=(1,))



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
        mean, log_std, hyperparams = params[:N], params[N:2*N], params[2*N:]

        return mean, log_std, hyperparams

    def gaussian_entropy(log_std):
        return 0.5 * N * (1.0 + np.log(2*np.pi)) + np.sum(log_std)

    

    def variational_objective(params):
        """Provides a stochastic estimate of the variational lower bound."""
        mean, log_std, hyperparams = unpack_params(params) 
        key = random.PRNGKey(10003)
        key, subkey = random.split(key)
        samples = random.normal(subkey, [num_samples, N]) * np.exp(log_std) + mean #Generate samples using reparam trick
        ### use jax vmap to calculate over samples here
        #handle for batched params
        logprob_samp  = lambda x: logprob(x, [100,1])
        batched_logprob = vmap(logprob_samp)
        lower_bound = gaussian_entropy(log_std) + np.mean(batched_logprob(samples)) #return elbo and hyperparams (loadings and length scale)

        return -lower_bound

    gradient = grad(variational_objective)

    return variational_objective, gradient, unpack_params



def calc_gp_prior(x_samps, K):
	'''
	Calculates the GP log prior (time domain)
	x_samples should be nsamples by T
	K is T by T
	'''

	Kinv = np.linalg.inv(K)
	log_prior = -(1/2)*(np.matmul(np.matmul(x_samps,Kinv), x_samps)+ np.linalg.slogdet(K)[1]) 

	return log_prior


def log_joint(ca_params, hyperparams, ca_obj):


	ll =ca_obj.log_likelihood(ca_params)
	K = make_cov(ca_obj.Tps, hyperparams[0], hyperparams[1]) + np.eye(ca_obj.Tps)*1e-7
	log_prior = calc_gp_prior(ca_params, K)
	return log_prior + ll


##### set up optimization

num_samples = 10

rate_length = ca_obj.Tps
loc = np.ones(rate_length, np.float32)
log_scale = -10* np.ones(rate_length, np.float32)


init_ls = np.array([100], np.float32)
init_rho = np.array([3], np.float32)

full_params = np.concatenate([loc, log_scale, init_rho,init_ls])


log_joint_fullparams = lambda samples, hyperparams: log_joint(samples, hyperparams, ca_obj)
varobjective, gradient, unpack_params = bbvi(log_joint_fullparams, rate_length, num_samples)



# #testing here.....
# gradient = grad(ca_obj.log_likelihood) 
# gradient = jacfwd(ca_obj.log_likelihood) 
# rate = 2*np.ones(ca_obj.Tps)

# gradient(rate)


step_size = 1
print(varobjective(full_params))
for i in range(1000):

	full_params_grad = gradient(full_params)
	full_params -= step_size *full_params_grad
	if i % 10 == 0:
		elbo_val = varobjective(full_params)
		print('{}\t{}'.format(i, elbo_val))



# opt_init, opt_update = optimizers.adam(step_size=1e-3)
# opt_state = opt_init(net_params)


# @jit
# def step(i, opt_state, x1, y1):
#     p = optimizers.get_params(opt_state)
#     g = grad(loss)(p, x1, y1)
#     return opt_update(i, g, opt_state)

# for i in range(100):
#     opt_state = step(i, opt_state, xrange_inputs, targets)
# net_params = optimizers.get_params(opt_state)



