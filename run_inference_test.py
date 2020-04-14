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

import GP_fourier as gpf 

import matplotlib.pyplot as plt
import matplotlib
matplotlib.interactive(True)

from CA.CA import CA_Emissions
from CA.misc import bbvi, make_cov



timepoints = 500
# rate = .1*np.ones(timepoints)#generate poisson rate

ca_obj = CA_Emissions(AR_params = [.05], Gauss_sigma = 0.1, Tps = timepoints) #generate AR1 calcium object
# plt.plot(ca_obj.sample_data(rate)[0][0:500])



# ca_obj = CA_Emissions(AR_params = [.1, .05], Gauss_sigma = 0.01, Tps = timepoints) #generate AR1 calcium object
# plt.plot(ca_obj.sample_data(rate)[0][0:500])

rate = 50*np.cos(np.arange(0,5,.01))+50

samp_data = ca_obj.sample_data(rate*ca_obj.dt)

ca_obj.set_data(samp_data)





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


def log_joint(ca_params, hyperparams, ca_obj, Fourier = False, nxcirc = None, wwnrm = None, Bf = None):

	if Fourier:
		K = hyperparams[0] * gpf.mkcovs.mkcovdiag_ASD_wellcond(hyperparams[1], 1, nxcirc, wwnrm = wwnrm,addition = 1e-4)
		log_prior = calc_gp_prior(ca_params, K, Fourier = True)
		ll =ca_obj.log_likelihood(np.matmul(ca_params, Bf))

	else:
		K = make_cov(ca_obj.Tps, hyperparams[0], hyperparams[1]) + np.eye(ca_obj.Tps)*1e-2
		log_prior = calc_gp_prior(ca_params, K)
		ll =ca_obj.log_likelihood(ca_params)

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
minlens = 20 #assert a minimum scale for eigenvalue thresholding
nxc_ext = 0.05

_, wwnrm, Bffts, nxcirc = gpf.comp_fourier.conv_fourier(ca_obj.data, ca_obj.Tps, minlens,nxcirc = np.array([ca_obj.Tps+nxc_ext*ca_obj.Tps]))
N_four = Bffts[0].shape[0]


num_samples = 8

rate_length = N_four
loc = np.zeros(rate_length, np.float32)
log_scale = -5* np.ones(rate_length, np.float32)


init_ls = np.array([80], np.float32)
init_rho = np.array([1], np.float32)

full_params = np.concatenate([loc, log_scale, init_rho,init_ls])


log_joint_fullparams = lambda samples, hyperparams: log_joint(samples, hyperparams, ca_obj, Fourier = True, Bf=Bffts[0], wwnrm = wwnrm, nxcirc = nxcirc)

varobjective, gradient, unpack_params = bbvi(log_joint_fullparams, rate_length, num_samples)


# #testing here.....
# gradient = grad(ca_obj.log_likelihood) 
# gradient = jacfwd(ca_obj.log_likelihood) 
# rate = 2*np.ones(ca_obj.Tps)

# gradient(rate)
#varobjective(full_params, key)

step_size = .005
key = random.PRNGKey(10003)
elbos = []
for i in range(8000):
	key, subkey = random.split(key)
	full_params_grad = gradient(full_params, subkey)
	full_params -= step_size *full_params_grad
	if i % 50 == 0:
		print(full_params[0:20])
		elbo_val = varobjective(full_params, key)
		elbos.append(elbo_val)
		print('{}\t{}'.format(i, elbo_val))


###fourier 

time_domain_params = np.matmul(full_params[0:N_four],Bffts[0])


#
plt.subplot(3,1,1)
plt.ylabel('True rate')
plt.plot(np.arange(0,5,.01),rate*ca_obj.dt)
plt.subplot(3,1,2)
plt.plot(np.arange(0,5,.01),samp_data[1])
plt.ylabel('Spikes')
plt.subplot(3,1,3)
plt.plot(np.arange(0,5,.01),samp_data[0])
plt.ylabel('Ca activity')
plt.xlabel('Time (sec)')

plt.figure(2)
plt.plot(np.arange(0,5,.01),np.exp(time_domain_params))
plt.plot(np.arange(0,5,.01),rate)
plt.legend(['Inferred rate', 'true rate'])
plt.ylabel('Rate')
plt.xlabel('Time (sec)')

plt.figure(3)
plt.plot(-np.array(elbos))
plt.ylabel('Elbo')
plt.xlabel('Iterations')


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



