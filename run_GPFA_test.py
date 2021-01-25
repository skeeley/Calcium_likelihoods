from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import jax
import jax.numpy as np
# Current convention is to import original numpy as "onp"
import numpy as onp
from jax import grad, jit, vmap
from jax.experimental import optimizers
from jax.tree_util import tree_multimap  # Element-wise manipulation of 
from jax import random
from jax import jacfwd

import GP_fourier as gpf 

import matplotlib.pyplot as plt
import matplotlib
matplotlib.interactive(True)

from CA.CA_new import CA_Emissions
from CA.misc import bbvi, make_cov

from jax.config import config
config.update("jax_enable_x64", True)





timepoints = 2000
# rate = .1*np.ones(timepoints)#generate poisson rate

ca_obj = CA_Emissions(Gauss_sigma = np.array([0.005]), alpha = np.array([1]), Tps = timepoints, dt = .01, AR = 2, As = np.array([[1.81,-.82 ]]).T) #generate AR1 calcium object
ca_obj = CA_Emissions(Gauss_sigma = np.array([0.005]), alpha = np.array([1]), Tps = timepoints, dt = .01, AR = 1, As = np.array([[.99,]]).T) #generate AR1 calcium object

# plt.plot(ca_obj.sample_data(rate)[0][0:500])



# ca_obj = CA_Emissions(AR_params = [.1, .05], Gauss_sigma = 0.01, Tps = timepoints) #generate AR1 calcium object
# plt.plot(ca_obj.sample_data(rate)[0][0:500])

rate = np.exp(3*np.sin(np.arange(0,10,.005)))

samp_data, spks = ca_obj.sample_data(np.expand_dims(rate*.01, axis = 0))

#ca_obj.set_data(samp_data)





def calc_gp_prior(rate, K, Fourier = False):
	'''
	Calculates the GP log prior (time domain)
	x_samples should be nsamples by T
	K is T by T
	'''
	if Fourier:
		log_prior = -(1/2)*(np.sum(np.square(rate)/K)+ np.sum(np.log(2*np.pi*K)))  

	else:
		Kinv = np.linalg.inv(K)
		log_prior = -(1/2)*(np.shape(Kinv)[0]*np.log(2*np.pi) + np.matmul(rate,np.matmul(Kinv,rate))+ np.linalg.slogdet(K)[1]) 

	return log_prior


def log_joint(X, model_params, ca_obj, Fourier = False, nxcirc = None, wwnrm = None, Bf = None, learn_model_params  = False):

	# logjoint here can work in fourier domain or not.
	# If Fourier, need to pass in Fourier args (nxcirc, wwnrm, bf)

	# logjoint can also learn calcium hyperparams (tau, alpha, marginal variance) or not
	# if yes, please append these to they hyperparams argument AFTER the rho and length scale


	if Fourier:
		K = model_params[0] * gpf.mkcovs.mkcovdiag_ASD_wellcond(model_params[1], 1, nxcirc, wwnrm = wwnrm,addition = 1e-4)
		log_prior = calc_gp_prior(X, K, Fourier = True)
		print(log_prior)

		params = np.matmul(X, Bf)
		if learn_model_params:
			params = np.append(params, np.array([model_params[2:]]))
		

	else:
		K = make_cov(ca_obj.Tps, model_params[0], model_params[1]) + np.eye(ca_obj.Tps)*1e-2
		log_prior = calc_gp_prior(X, K)
		if learn_model_params :
			params = np.append(X, model_params[2:])
	# print(np.shape(params))

	ll =ca_obj.log_likelihood(samp_data, np.expand_dims(params, axis = 0) ,  learn_model_params  = learn_model_params )

	return log_prior + ll


##### set up optimization (time domain)

# num_samples = 5

# rate_length = ca_obj.Tps
# loc = np.zeros(rate_length, np.float32)
# log_scale = -5* np.ones(rate_length, np.float32)


# init_ls = np.array([90], np.float32)
# init_rho = np.array([1], np.float32)

# full_params = np.concatenate([loc, log_scale, init_rho,init_ls])


# log_joint_fullparams = lambda samples, hyperparams: log_joint(samples, hyperparams, ca_obj)

# varobjective, gradient, unpack_params = bbvi(log_joint_fullparams, rate_length, num_samples)








####### Fourier Domain #############
minlens = 60 #assert a minimum scale for eigenvalue thresholding
nxc_ext = 0.1

_, wwnrm, Bffts, nxcirc = gpf.comp_fourier.conv_fourier([samp_data], ca_obj.Tps, minlens,nxcirc = np.array([ca_obj.Tps+nxc_ext*ca_obj.Tps]))
N_four = Bffts[0].shape[0]


num_samples = 10

rate_length = N_four
var_mean = np.zeros(rate_length, np.float64)
log_var_scale = -5* np.ones(rate_length, np.float64)


init_ls = np.array([100], np.float64)
init_rho = np.array([1], np.float64)

init_marg_var = np.array([.2], np.float64)
init_alpha = np.array([1], np.float64)
init_tau = np.array([.1], np.float64)

full_params = np.concatenate([var_mean, log_var_scale, init_rho,init_ls,init_marg_var, init_alpha,   init_tau])
#full_params = np.concatenate([var_mean, _scale, init_rho,init_ls])

log_joint_fullparams = lambda samples, hyperparams: log_joint(samples, hyperparams, ca_obj, Fourier = True, learn_model_params  =True, Bf=Bffts[0], wwnrm = wwnrm, nxcirc = nxcirc)

varobjective, gradient, unpack_params = bbvi(log_joint_fullparams, rate_length, num_samples)




###### SGD #############

# step_size = .01*np.ones(N_four*2)
# step_size = np.append(step_size,np.array([.01,.05]))

# lenscs = []
# key = random.PRNGKey(10003)
# elbos = []
# for i in range(25000):
# 	key, subkey = random.split(key)
# 	full_params_grad = gradient(full_params, subkey)
# 	full_params -= step_size *full_params_grad
# 	if i % 100 == 0:
# 		print(full_params[0:20])
# 		elbo_val = varobjective(full_params, key)
# 		elbos.append(elbo_val)
# 		lenscs.append(full_params[-1])
# 		print('{}\t{}'.format(i, elbo_val))






###### ADAM #############

opt_init, opt_update, opt_get_params = optimizers.adam(step_size=.05)
opt_state = opt_init(full_params)

key = random.PRNGKey(10003)
# Define a compiled update step
@jit
def step(i, key, opt_state):

	objective = lambda full_params: varobjective(full_params, key)  ### pass new key to objective for sampling
	full_params = opt_get_params(opt_state)
	g = grad(objective)(full_params)
	return opt_update(i, g, opt_state)



elbos = []
for i in range(5000):
	key, subkey = random.split(key)
	opt_state = step(i, key, opt_state)
	if i % 100 == 0:
		elbo_val = varobjective(opt_get_params(opt_state), key)
		print(i, elbo_val)
		elbos.append(elbo_val)
final_params = opt_get_params(opt_state)











time_domain_params = np.matmul(final_params[0:N_four],Bffts[0]) #convert back to time domain


#
plt.subplot(3,1,1)
plt.ylabel('True rate')
plt.plot(np.arange(0,10,.01),rate[0])
plt.subplot(3,1,2)
plt.plot(np.arange(0,10,.01),spks.T)
plt.ylabel('Spikes')
plt.subplot(3,1,3)
plt.plot(np.arange(0,10,.01),samp_data[0].T)
plt.ylabel('Ca activity')
plt.xlabel('Time (sec)')

plt.figure(2)
plt.plot(np.arange(0,10,.01),np.log(rate[0]))
plt.plot(np.arange(0,10,.01),time_domain_params)
plt.legend(['True rate', 'Inferred rate'])
plt.ylabel('Rate')
plt.xlabel('Time (sec)')

plt.figure(3)
plt.plot(-np.array(elbos))
plt.ylabel('Elbo')
plt.xlabel('Iterations')




