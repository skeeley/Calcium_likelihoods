
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
		K = make_cov(ca_obj.Tps, model_params[0], model_params[1]) + np.eye(ca_obj.Tps)*1e-2 #need heavy regularization here. (might be differnet for different opt params)
		log_prior = calc_gp_prior(X, K)
		if learn_model_params :
			params = np.append(X, model_params[2:])

	ll =ca_obj.log_likelihood(Y, rates,  learn_model_params  = learn_model_params )

	return log_prior + ll

timepoints = 1000
lscs = [200,100]
n_lats = 2
n_neurons = 20
loadings = onp.random.randn(n_neurons,np.size(lscs))*2+1
latents = generate_latents(lscs, timepoints)
full_rate = loadings@latents 
x= np.array(softplus(full_rate)) 
#y = np.random.poisson(x)

# rate = .1*np.ones(timepoints)#generate poisson rate

ca_obj = CA_Emissions(AR = 2, As = np.array([[1.81,-.82]]).T, Gauss_sigma = np.array([0.05]), alpha = np.array([1]),link = "softplus",Tps = timepoints, dt = .01) #generate AR1 calcium object
#ca_obj = CA_Emissions(AR = 1, As = np.array([[.1]]).T, Gauss_sigma = np.array([0.01]), alpha = np.array([1]),link = "softplus",Tps = timepoints, dt = .01) #generate AR1 calcium object

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


num_samples = 25

rate_length = N_four
var_mean = np.zeros(rate_length*n_lats, np.float64)
log_var_scale = -5* np.ones(rate_length*n_lats, np.float64)


init_ls = np.array([200,100], np.float64)
init_loadings = np.zeros(n_lats*n_neurons, np.float64)/n_neurons

loginit_marg_var = np.log(np.array([.1], np.float64))
init_alpha = np.array([1], np.float64)
init_As = np.array([1.81,-.82], np.float64)

ca_obj = CA_Emissions(AR = 2, As = np.array([init_As]).T, Gauss_sigma = np.exp(loginit_marg_var), alpha = init_alpha ,link = "softplus",Tps = timepoints, dt = .01) #generate AR1 calcium object


full_params = np.concatenate([var_mean, log_var_scale, init_loadings,init_ls,np.exp(loginit_marg_var), init_alpha,   init_As])
#full_params = np.concatenate([var_mean, _scale, init_rho,init_ls])

log_joint_fullparams = lambda samples, hyperparams: log_joint(samp_data, samples, hyperparams, ca_obj, n_lats, Fourier = True, learn_model_params  =True, Bf=Bffts[0], wwnrm = wwnrm, nxcirc = nxcirc)

varobjective, gradient, unpack_params = bbvi(log_joint_fullparams, rate_length*n_lats, num_samples)


