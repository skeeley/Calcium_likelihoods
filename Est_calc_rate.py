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
import autograd.scipy.misc as ms

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

	# [S1,S2] =  GenCovM(par.x_len,par.y_len,rho,len_sc);
	# C = kron(S2,S1);
	# generate fake place field

	# %for circular boundaries!
	# opts.nxcirc = par.x_len;
	# opts.condthresh = 1e10;
	# [cdiag,U] = mkcov_ASDfactored([len_sc,rho],par.x_len,opts);
	# S1 = U*diag(cdiag)*U';

	# Just using Kron S1 with itself for the generation of the field. (square
	# grid right now)
	# fake_field = scale*mvnrnd(zeros(1,length(S1)*length(S1)),kron(S1,S1))+dc;

	# #specify lengths
	# n1 = length(S1);
	# n2 = par.y_len;


	# [~,ind] = datasample(fake_field,ndata) #draw samples
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
	# logrpar should be length T where t is the total time of the recording (total number of timebins)
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
	Xstruct['dtStim'] = 1 # time bin size for stimulus (30 Hz for Berry data)
	Xstruct['dtSp'] = Xstruct['dtStim'] # time bins size for spike train
	Xstruct['calc_ts'] = 20# could be a hyper_parameter
	Xstruct['a'] = 2#
	Xstruct['c'] = 1#both of these are 1 for now.
	Xstruct['max_spk'] = 30 #max spikes in time bin for optimization
	Xstruct['Gauss_sigma'] = .1 #sigma of additive Gaussian Noise
	Xstruct['T'] = 10000#Number of seconds per block for Michael Berry's data

	#initialize GP prior to 0.
	Xstruct['prior'] = False
	Xstruct['AR1'] = True
	Xstruct['AR2'] = False #set to AR1 or AR2
	Xstruct['Gauss_L'] = False #set to do model with Gaussian noise (no poisson mixture)

	return Xstruct

def pre_process_struct(Xstruct):



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
	mu = (Xstruct['c']*f1*np.exp(-Xstruct['dtSp']/Xstruct['calc_ts']) + Xstruct['a']*Xstruct['spky_mat'])
	    
	#define gaussian %don't exp work in log only!!!!

	Xstruct['log_gaus'] =np.log(1/np.sqrt(np.square(Xstruct['Gauss_sigma'])*2*np.pi)) + (-(np.square(f0-mu))/(2*np.square(Xstruct['Gauss_sigma'])))
	return Xstruct


## main ##



# savemat('rates_stim',rates_stim)

Xstruct = create_main_struct()
rate = .04
toyrate = build_toy_ca_const(rate, Xstruct)
Xstruct['calc_tr'] =toyrate
Xstruct = pre_process_struct(Xstruct)
init_params = 1
prsML= ML_npmixGLM(Xstruct,init_params)

print("learned rate is", np.exp(prsML.x))







