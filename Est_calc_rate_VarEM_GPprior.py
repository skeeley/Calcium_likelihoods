from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import matplotlib.pyplot as plt
import GP_fourier as gpf 
import random as rand

import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.multivariate_normal as mvn
import autograd.scipy.stats.norm as norm
import autograd.scipy.ms as logsumexp

from autograd import grad


from autograd import value_and_grad
from autograd.misc.optimizers import adam
from scipy.optimize import minimize
from autograd.extend import primitive, defvjp

from scipy.signal import lfilter
from scipy.special import gammaln

from scipy.io import loadmat, savemat



def build_toy_ca_const(rate, Xstruct):
	#generate a toy calcium trace with constant poisson rate.
	#Outlined code here to make this more general (for Illana stuff)

def build_toy_dataset(N, D,rh, len_sc,add_noise_var, sing_GP = False, Pois_noise = False,scale_pois = 1):
  '''
  This function will generate some data with GP statistics with variance 'rh', and length scale 'len_sc'. 
  Data will be generated of length N, with batch size D. Default function will add Gaussian noise
  with marginal variance 'add_noise_var'. 
  Defaults
  sing_GP = False indicates a new GP draw with the same statistics for each Batch
  Pois_noise = False indicates Gaussian noise is added to the GP.
  '''
  #=X = tf.constant(np.tile(np.arange(N),(D,1)),dtype = tf.float32)
  #x = rbf_op(X,D,rh,len_sc)
  M1 = np.array([range(N)])- np.transpose(np.array([range(N)]))
  K = rh*np.exp(-(np.square(M1)/(2*np.square(len_sc))))
  if sing_GP is False:
    x = np.array(np.random.multivariate_normal(np.zeros(N), K, D))
  else:
    x = np.array(np.random.multivariate_normal(np.zeros(N), K))

  if Pois_noise is False:  
    y = x + [np.random.normal(0.0, np.sqrt(add_noise_var), size=N) for batch in range(D)]
  else: 
    #rate = [np.exp(x)/Pois_scale for batch in range(D)]
    x= [np.log(1 + np.exp(x)) for batch in range(D)]/np.asarray(scale_pois)
    #x = [np.exp(x)/10 for batch in range(D)]
    y = np.random.poisson(x)
    x = x[0]
  #plt.plot(x[0])
  #plt.plot(y[0],'x')
  #plt.axis('equal')
  #plt.show()
  return x, y



	#[~,ind] = datasample(fake_field,ndata) #draw samples
	#xstim =  sparse(ind,1:length(ind),1) # index the samples at 1 for every location, zeros otherwise
	#xstim = full(xstim); #convert to full matrix
	#fstim = fake_field*full(xstim) #stim dotted with field
	#sprate = np.exp(fstim) #exponential nonlinearity

	logsprate = np.log(rate)*np.ones(int(np.floor(Xstruct['T']/Xstruct['dtSp']))) #constant (for now)
	spcounts = np.random.poisson(Xstruct['dtSp']*np.exp(logsprate)) # poisson spike counts


	##### Optional Model #####
	#AR2
	if Xstruct['AR2'] is True:
	    #AR2
	    q = [exp(-par.dt_spk/par.calc_ts),exp(-par.dt_spk/par.calc_ts_b)];
	    a_true = poly(q);
	    z = filter(1,a_true,spcounts);
	    
	else:
	    #AR1
	    z= lfilter([Xstruct['a']],[1,-np.exp(-Xstruct['dtSp']/Xstruct['calc_ts'])],spcounts) #convolve with exp filter


	trace = z + np.sqrt(Xstruct['Gauss_sigma'])*np.random.randn(np.size(z))#add noise


	return trace




def neglogli_npmixGLM(params,Xstruct):
	#  Compute negative log-likelihood function for normal-Poisson mixture GLM
	#  model for calcium fluorescence data (following Ganmor et al 2016). 
	# 
	#  INPUTS:
	#       par [n x 1] - parameter vector (GLM filter parameters)
	#           Xstruct - structure with stimuli, Gaussian likelihoods, other stuff
	# 
	#  OUTPUTS:
	#    neglogli [1 x 1] - negative log-likelihood OR
	#  neglogjoint [1 x 1] - negative joint likelihood IF PRIOR


	spkvect= np.arange(Xstruct['max_spk']+1) # vector of spike conts

	#  Filter stimulus
	logrpar = 1*params+np.log(Xstruct['dtSp']) #log of Poisson rate
	rpar = np.exp(logrpar) # Poisson rate

	#  Compute Poisson log-probability for each rate, for each spike count in spkvect
	logpoisspdf_mat = (logrpar*spkvect- rpar)-gammaln(spkvect+1)

	#logpoisspdf_mat = bsxfun(@minus,bsxfun(@minus,bsxfun(@times,logrpar,spkvect),rpar),gammaln(spkvect+1));

	#  Compute joint log-probability of spike counts and Gaussian noise
	logpjoint = logpoisspdf_mat + Xstruct['log_gaus']

	# #  Original (simpler) code (but not robust to underflow)
	# pjoint = np.exp(logpjoint) # joint probability of spike count and Gaussian noise
	# livec = np.sum(pjoint,axis=1) # sum across spike counts for each observation
	# loglivec = np.log(livec) # take log 

	#  Exponentiate and sum over spike counts (robust to underflow/overflow errors)
	# maxval = np.max(logpjoint,1); # find max of each row
	# pjointnrm = np.exp(logpjoint-maxval[:,None]) # exponentiate with max of each col subtracted off
	# loglivec = np.log(np.sum(pjointnrm,axis = 1))+maxval; # sum and then add back in log-factor 

	loglivec = ms.logsumexp(logpjoint, axis = 1)

	#  sum up over time bins
	neglogli = -np.sum(loglivec) # sum up and take negative 

	if  Xstruct['prior'] is True:
	    pen = .5*params.T@Xstruct['S1_inv']@par
	    neg_post = neglogli + .5*params@Xstruct['S1_inv']@params
	    return neglogjoint #for if we got a prior

	else:
		return neglogli #just MLE



def ML_npmixGLM(Xstruct,init_params):

	# Define the loss function and set optimization to MLE only

	Xstruct['prior'] = False

	if Xstruct['Gauss_L']:
	   lossfun = lambda params: neglogli_Gauss(params,Xstruct) 
	else:
	   lossfun = lambda params: neglogli_npmixGLM(params,Xstruct)
	

	# minimize negative log likelihood (or log-posterior)
	prsML= minimize(value_and_grad(lossfun), init_params, jac=True, method='CG') # find ML estimate of params
	return prsML

def create_main_struct():

	# ---- Create struct and make stimulus design matrix ---------------------

	# Initialize parameters relating to stimulus design matrix 

	# ---- Set up filter and stim processing params ------------------- 
	# nkx = size(stim,1);  % number stim pixels (# x params)
	# nkt = 1; %size(gg.ktbas,2); % # time params per stim pixel (# t params)
	# ncols = nkx*nkt; % total number of columns in stim design matrix
	# [slen,swid] = size(stim'); % size of stimulus


	# ---- Check size of filter and width of stimulus ----------
	#assert(nkx == swid,'Mismatch between stim width and kernel width');


	# ---- Set fields of Xstruct -------------------------------------
	#Xstruct.nkx = nkx; % # stimulus spatial stimulus pixels 
	#Xstruct.nkt = nkt; % # time bins in stim filter
	Xstruct = {}
	Xstruct['dtStim'] = 1/30 # time bin size for stimulus (30 Hz for Berry data)
	Xstruct['dtSp'] = Xstruct['dtStim'] # time bins size for spike train
	Xstruct['calc_ts'] = .5# could be a hyper_parameter
	Xstruct['a'] = 2#
	Xstruct['c'] = 1#both of these are 1 for now.
	Xstruct['max_spk'] = 10 #max spikes in time bin for optimization
	Xstruct['Gauss_sigma'] = .2 #sigma of additive Gaussian Noise
	Xstruct['T'] = 24#Number of seconds per block for Michael Berry's data

	#initialize GP prior to 0.
	Xstruct['prior'] = False
	Xstruct['AR1'] = True
	Xstruct['AR2'] = False #set to AR1 or AR2
	Xstruct['Gauss_L'] = False #set to do model with Gaussian noise (no poisson mixture)

	return Xstruct

def pre_process_struct(Xstruct, stim):



	#Isolate calcium traces offset by correct index for AR1 and AR2
	if Xstruct['AR1'] is True: 

		b = np.zeros([np.size(Xstruct['calc_tr'])-1,1+Xstruct['max_spk']])
		f1 =  b + Xstruct['calc_tr'][:-1,None]
		f0 =  b  + Xstruct['calc_tr'][1:,None]

	elif Xstruct['AR2'] is True:

		b = np.zeros(length(Xstruct['calc_tr'])-2,1+Xstruct['max_spk']);
		f2 =  b + Xstruct['calc_tr'][:-2]
		f1 =  b + Xstruct['calc_tr'][2:-1]
		f0 =  b + Xstruct['calc_tr'][3:]

	else:
		raise Exception('THIS IS AN EXCEPTION!')

	#define matrix of spks
	Xstruct['spky_mat'] = np.arange(Xstruct['max_spk']+1)*np.ones(np.shape(f1));


	#Do some of the likelihood prep here:
	#set up stuff for likelihood given the hyperparams of noise and calcium
	#timescale


	#define mean for gaussian form in likelihood
	mu = (Xstruct['c']*f1*np.exp(-Xstruct['dtSp']/Xstruct['calc_ts'])+Xstruct['a']*Xstruct['spky_mat'])
	    
	#define gaussian %don't exp work in log only!!!!

	Xstruct['log_gaus'] =np.log(1/np.sqrt(np.square(Xstruct['Gauss_sigma'])*2*np.pi)) + (-(np.square(f0-mu))/(2*np.square(Xstruct['Gauss_sigma'])))
	return Xstruct


# def calc_rates_berry(data,Xstruct):
# 	nneurs, ntrials = berrydat['ABCD'].shape[:2]
# 	rates = np.zeros([nneurs,ntrials])
# 	for j in np.arange(nneurs):
# 		for i in np.arange(ntrials):
# 			Xstruct['calc_tr'] = data[j][i]
# 			Xstruct['Xstim'] =np.ones(np.size(Xstruct['calc_tr'])-1)
# 			Xstruct = pre_process_struct(Xstruct, Xstruct['Xstim'])
# 			init_params = 1
# 			prsML= ML_npmixGLM(Xstruct,init_params)
# 			rates[j][i] = np.exp(prsML['x'])
# 			print(i, np.exp(prsML['x']))
# 	return rates

## main ##
#first generate Xstruct structure that contains all necessary parameters and information about the calcium trace




#load data
# berrydat = loadmat('/Users/stephen/Google Drive/Research/Pillow/Gaussian_Process_ml/Python/J54_blocks_trials.mat')
Xstruct = create_main_struct() #use Berry's hyperpars here

rate = 20
Xstruct['calc_tr'] = build_toy_ca_const(rate, Xstruct)
Xstruct = pre_process_struct(Xstruct, Xstruct['Xstim'])
init_params = 1
prsML= ML_npmixGLM(Xstruct,init_params)
print(np.exp(prsML['x']))
# rates_stim = {}
# rates_stim['ABCD'] = calc_rates_berry(berrydat['ABCD'],Xstruct)
# rates_stim['xBCD'] = calc_rates_berry(berrydat['xBCD'],Xstruct)
# rates_stim['AxCD'] = calc_rates_berry(berrydat['AxCD'],Xstruct)
# rates_stim['ABxD'] = calc_rates_berry(berrydat['ABxD'],Xstruct)
# rates_stim['ABCx'] = calc_rates_berry(berrydat['ABCx'],Xstruct)


# savemat('rates_stim',rates_stim)












