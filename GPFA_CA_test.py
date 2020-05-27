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

def softplus(x):
	lt_34 = (x >= 34)
	gt_n37 = (x <= -30.8)
	neither_nor = np.logical_not(np.logical_or(lt_34, gt_n37))
	rval = np.where(gt_n37, 0., x)
	return np.where(neither_nor, np.log(1 + np.exp(x)), rval)


def generate_latents(len_sc, N):

  M1 = onp.array([range(N)])- onp.transpose(onp.array([range(N)]))
  if np.size(len_sc)>0:
    K = [np.exp(-(np.square(M1)/(2*np.square(len_sc[i])))) for i in np.arange(np.size(len_sc))]
  else:
    K = np.exp(-(np.square(M1)/(2*np.square(len_sc))))

  n_latents = np.size(len_sc)
  #draw a rate with GP stats (one or many)

  if np.size(len_sc)>0:
    latents = np.array([onp.random.multivariate_normal(onp.zeros(N), K[i]) for i in onp.arange(np.size(len_sc))])
  else:
    latents = np.array(onp.random.multivariate_normal(onp.zeros(N), K, n_latents))
  return latents

	#pass through the loadings matrix

def calc_gp_prior(X, K,n_lats, Fourier = False):
	'''
	Calculates the GP log prior (time domain)
	x_samples should be nsamples by T
	K is T by T
	'''
	total_prior = 0
	if Fourier:
		for i in np.arange(n_lats):
			x_lat = X[i]
			total_prior = total_prior -(1/2)*(np.sum(np.square(x_lat)/K[i])+ np.sum(np.log(2*np.pi*K[i])))  

	# else:
	# 	Kinv = np.linalg.inv(K)
	# 	log_prior = -(1/2)*(np.shape(Kinv)[0]*np.log(2*np.pi) + np.matmul(rate,np.matmul(Kinv,rate))+ np.linalg.slogdet(K)[1]) 

	return total_prior


def log_joint(Y, X,  model_params, ca_obj,  n_lats, Fourier = False, nxcirc = None, wwnrm = None, Bf = None, learn_model_params  = False, learn_per_neuron = False):

	# logjoint here can work in fourier domain or not.
	# If Fourier, need to pass in Fourier args (nxcirc, wwnrm, bf)

	# logjoint can also learn calcium hyperparams (tau, alpha, marginal variance) or not
	# if yes, please append these to they hyperparams argument AFTER the rho and length scale

	# X should be passed in as samples by neurons by latents
	#model params is a single vector of loadings then length scales than CA params
	X = np.reshape(X, [n_lats, -1])
	n_neurons = np.shape(Y)[0]
	loadings_hat = np.reshape(model_params[0:n_neurons*n_lats], [n_neurons,n_lats])
	ls_hat = model_params[n_neurons*n_lats:n_neurons*n_lats+n_lats]


	if Fourier:
		K = gpf.mkcovs.mkcovdiag_ASD_wellcond(ls_hat, np.ones(np.size(ls_hat)), nxcirc, wwnrm = wwnrm,addition = 1e-4).T
		if n_lats == 1:
			K = np.expand_dims(K, axis = 0)
		log_prior = calc_gp_prior(X, K,n_lats, Fourier = True)


		params = np.matmul(X, Bf)
		rates = loadings_hat@params


		if learn_model_params:
			if learn_per_neuron:
				param_butt = np.reshape(model_params[n_neurons*n_lats+n_lats:], [n_neurons,-1])
			else:
				param_butt = np.tile(model_params[n_neurons*n_lats+n_lats:], [n_neurons,1])	


			rates = np.append(rates, np.array(param_butt), axis = 1)


	else:
		K = make_cov(ca_obj.Tps, model_params[0], model_params[1]) + np.eye(ca_obj.Tps)*1e-2
		log_prior = calc_gp_prior(X, K)
		if learn_model_params :
			params = np.append(X, model_params[2:])

	ll =ca_obj.log_likelihood(Y, rates,  learn_model_params  = learn_model_params )

	return log_prior + ll

timepoints = 1000
lscs = [200,100,400]
n_lats = 3
n_neurons = 35
loadings = onp.random.randn(n_neurons,np.size(lscs))*10
latents = generate_latents(lscs, timepoints)
full_rate = loadings@latents 
x= np.array(softplus(full_rate))
#y = np.random.poisson(x)

# rate = .1*np.ones(timepoints)#generate poisson rate

ca_obj = CA_Emissions(AR_params = [.1], Gauss_sigma = np.array([0.01]), alpha = np.array([1]),link = "softplus",Tps = timepoints, dt = .01) #generate AR1 calcium object
# plt.plot(ca_obj.sample_data(rate)[0][0:500])


#rate = 40*np.sin(np.arange(0,20,.01))+50
rate = x

samp_data, spks = ca_obj.sample_data(rate*ca_obj.dt)

# ca_obj.set_data(samp_data)








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
minlens = 80 #assert a minimum scale for eigenvalue thresholding
nxc_ext = 0.1

_, wwnrm, Bffts, nxcirc = gpf.comp_fourier.conv_fourier([samp_data[0]], ca_obj.Tps, minlens,nxcirc = np.array([ca_obj.Tps+nxc_ext*ca_obj.Tps]))
N_four = Bffts[0].shape[0]


num_samples = 10

rate_length = N_four
var_mean = np.zeros(rate_length*n_lats, np.float64)
log_var_scale = -5* np.ones(rate_length*n_lats, np.float64)


init_ls = np.array([200,200,200], np.float64)
init_loadings = np.zeros(n_lats*n_neurons, np.float64)/n_neurons

init_marg_var = np.array([.01], np.float64)
init_alpha = np.array([1], np.float64)
init_tau = np.array([np.exp(-(.01/.1))], np.float64)

ca_obj = CA_Emissions(AR_params = init_tau, Gauss_sigma = init_marg_var, alpha = init_alpha ,link = "softplus",Tps = timepoints, dt = .01) #generate AR1 calcium object


full_params = np.concatenate([var_mean, log_var_scale, init_loadings,init_ls,init_marg_var, init_alpha,   init_tau])
#full_params = np.concatenate([var_mean, _scale, init_rho,init_ls])

log_joint_fullparams = lambda samples, hyperparams: log_joint(samp_data, samples, hyperparams, ca_obj, n_lats, Fourier = True, learn_model_params  =True, Bf=Bffts[0], wwnrm = wwnrm, nxcirc = nxcirc)

varobjective, gradient, unpack_params = bbvi(log_joint_fullparams, rate_length*n_lats, num_samples)




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

opt_init, opt_update, opt_get_params = optimizers.adam(step_size=.005)
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
		print(i, elbo_val, opt_get_params(opt_state)[-6:])
		elbos.append(elbo_val)
final_params = opt_get_params(opt_state)


lat_mean, var, model_params = unpack_params(final_params)
loadings_hat = np.reshape(model_params[0:n_neurons*n_lats], [n_neurons,n_lats])
ls_hat = model_params[n_neurons*n_lats:n_neurons*n_lats+n_lats]




lat_mean = np.reshape(lat_mean[:N_four*n_lats], [n_lats, N_four])
recon_latent = lat_mean@Bffts[0]

recon_traces = softplus(loadings_hat@recon_latent)


row= 2
column = 10

#neuron 1
for i in np.arange(column):

  plt.subplot(row,column,i+1)
  plt.plot(recon_traces[i+5].T,'k')
  plt.ylim([0,15])
  plt.plot(x[i+5].T,'g')
  plt.subplot(row,column,i+column+1)
  plt.plot(samp_data[i+5].T)
  plt.ylim([-0.5,5])
  #hist = sum(y_test_for_honeurs[:,neur,:])/(D-n_hos)

plt.subplot(row,column,0+1) 
plt.ylabel('firing rate')
plt.legend(['Estimated Rate', 'True rate'])  

plt.subplot(row,column,column+1) 
plt.ylabel('CA activity')

a1 = np.linalg.lstsq(recon_latent.T, latents.T)[0]
recon_lat_rot = recon_latent.T@a1


plt.figure(2)
plt.plot(recon_lat_rot,'k') 
plt.plot(latents.T,'g')


#
plt.subplot(3,1,1)
plt.ylabel('True rate')
plt.plot(np.arange(0,10,.005),rate)
plt.subplot(3,1,2)
plt.plot(np.arange(0,10,.005),samp_data[1])
plt.ylabel('Spikes')
plt.subplot(3,1,3)
plt.plot(np.arange(0,10,.005),samp_data[0])
plt.ylabel('Ca activity')
plt.xlabel('Time (sec)')

plt.figure(2)
plt.plot(np.arange(0,10,.005),rate)
plt.plot(np.arange(0,10,.005),np.exp(time_domain_params))
plt.legend(['True rate', 'Inferred rate'])
plt.ylabel('Rate')
plt.xlabel('Time (sec)')

plt.figure(3)
plt.plot(-np.array(elbos))
plt.ylabel('Elbo')
plt.xlabel('Iterations')




