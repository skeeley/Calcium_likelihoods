from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



from scipy.signal import lfilter

import jax
import jax.scipy as sp
import jax.numpy as np
# Current convention is to import original numpy as "onp"
import numpy as onp



def log_poisson_1D(y, rate):
    return y * np.log(rate) - rate - sp.special.gammaln(y+1)

def log_gaussian_1D(y, mu, sig2):
    return -0.5 * np.log(2.0 * np.pi * sig2) -0.5 * (y - mu)**2 / sig2
def softplus(x):
    lt_34 = (x >= 34)
    gt_n37 = (x <= -30.8)
    neither_nor = np.logical_not(np.logical_or(lt_34, gt_n37))
    rval = np.where(gt_n37, 0., x)
    return np.where(neither_nor, np.log(1 + np.exp(x)), rval)



class CA_Emissions():

    # ---- Create struct and make stimulus design matrix ---------------------

    def __init__(
        self,
        dt: float = 0.01,
        As = None,
        alpha: int = 1,
        Gauss_sigma: float = 0.1,
        Tps: int = 200, #initialize to 100 timesteps
        AR: int = 2,
        link = "log"
    ):



        if dt <= 0:
            raise ValueError("dt must be positive, got {0}".format(dt))
        # if  AR_params <= 0:
        #     raise ValueError("Calcium trace timescale must be positive, got {0}".format(AR_params))
        if alpha <= 0.0:
            raise ValueError("a bound must be positive, got {0}".format(alpha))



        self.dt = dt
        self.As =  As
        self.alpha = alpha
        self.Gauss_sigma = Gauss_sigma
        self.AR = AR
        self.Tps = Tps



        # self.AR1 = AR1
        # self.AR2 = AR2


        mean_functions = dict(
            log=lambda x: np.exp(x) * self.dt,
            softplus= lambda x: softplus(x) * self.dt
            )
        self.mean = mean_functions[link]

    @property
    def params(self):
        return self.dt, self.alpha, self.Gauss_sigma, self.Tps, self.AR_params

    def sample_data(self, rate):
        '''
        Generate simulated data with default class params. Feel free to change if needed. Can be AR1 or AR2
        '''
        if np.size(np.shape(rate)) == 1:
            rate = np.expand_dims(rate,axis = 0)
        spikes = onp.random.poisson(rate)


        N,T = np.shape(rate)

        nlags, As = self.AR, self.As
        if As is None:
            As = np.abs(onp.random.randn(nlags,N))# autoregressive component must be positive!


        if onp.size(As) == nlags:
            As = As*np.ones([nlags,N])


        if np.shape(rate)[-1] != self.Tps:
            raise ValueError("rate should be the same length as T in the calcium class, got {0}".format(np.shape(data)[-1]))




        #trace = lfilter([self.alpha],coeffs,noisey_spks) #generate AR-n process with inputs spk_counts, where self.alpha is the CA increase due to a spike
        # Generate Ca data
        y = onp.zeros(np.shape(rate))
        y[:,0:nlags] = onp.sqrt(self.Gauss_sigma) * onp.random.randn()



        for t in range(nlags, T):
            mus_ar= np.zeros((N))
            np.shape(As)
            for l in range(nlags):
                #Al = As[l:(l + 1),:]
                mus_ar= mus_ar + np.squeeze(y[:,t-l-1: t-l].T*As[l:(l+1),:]) #probably can do these previous two lines in a single matmul

            #print(np.shape(spikes[:,t]), np.shape(mus_ar), N, T, np.shape(y))
                
            y[:,t] =  mus_ar + self.alpha * spikes[:,t] + self.Gauss_sigma* onp.random.randn(N) 
        return y, spikes

        # q = [exp(-self.dt/tau_samp),exp(-self.dt/tau_samp_b)]
        # a_true = onp.poly(q)
        # z = filter(1,a_true,spcounts);
        # ##### To Do: AR2 PROCESS HERE!





    def log_likelihood(self, Y, X, nlags, S = 10, learn_model_params = False):

        '''
        #Inputs
        Y: Calcium trace time series (now with batching over neurons)
        X: log-rate time series (now with batching over neurons)

        '''
        # if np.array(self.alpha).ndim == 0:
        #     self.alpha = np.expand_dims(self.alpha, axis = 0)
        # if np.array(self.Gauss_sigma).ndim == 0:
        #     self.Gauss_sigma = np.expand_dims(self.Gauss_sigma, axis = 0)

        N,T = np.shape(Y)

        #For now....On the line below all the object attributes become of size n_neurons. So I'm only taking values for the first neuron (not learning per-neuron ca params)
        #This is beacuse X is a matrix of the lograte of the data of Neurons by time, when I append the hyperparams to learn at the end
        #i'm appending for each column. If we want to end up extending each param to be learnable per-neuron, this will be  a good starting point.
        #we might want to write a different function for this...outside of log_likelihood



        if learn_model_params:
            self.Gauss_sigma, self.alpha, self.As = X[:,-(nlags + 2):-(nlags+ 1)][0].T, X[:,-(nlags+ 1):-(nlags)][0].T, X[:,-nlags:].T

        As =  self.As


        if onp.size(As) == nlags:
            As = As*np.ones([nlags,N]) #broadcast across neurons if not specified per-neuron

        try:
             np.shape(X) == np.shape(Y) #this should pre-set a number of parameters for optimization, as well.
        except AttributeError:
             print("X and Y must be same shape")


        try:
             nlags == self.AR #this should pre-set a number of parameters for optimization, as well.
        except AttributeError:
             print("nlags argument and AR order must  be the same")

        if X.ndim == 2:
            n_neurs = np.shape(X)[0]
        else:
            n_neurs = 1
            X = np.expand_dims(X, axis = 0)




      
        ##### do Poiss part 
        rate = self.mean(X[:,:(self.Tps)]) #convert to rate given dt


        #  Compute Poisson log-probability for each rate, for each spike count in spkvect
        spk_vec = np.tile(np.arange(S+1),[n_neurs, 1])
        logpoisspdf_mat = log_poisson_1D(spk_vec[:,None,:], rate[:,:,None])





        ########### general purpose AR from data_mat (to do) #############       
        padded_Y = np.pad(Y,[(0, 0), (nlags,0)], mode='constant')
        mu = np.zeros(np.shape(Y))

        # for i in np.arange(nlags):
        #     # mu += (padded_Y[:,nlags -i -1:-i -1].T*As[i]).T #having a bit of trouble here with maybe jax (?). As[i] should just be 35, 
        #                                                     #it's saying its 35,1 and not broadcasting correctly. The workaround is the transposes

        #     mu += (jax.lax.dynamic_slice(padded_Y,(:,nlags -i -1),[:,nlags]).T*As[i]).T 
        mu = (padded_Y[:,nlags  -1: -1].T*As[0]).T + (padded_Y[:,nlags-2: -2].T*As[1]).T  #THIS ONE WORKS


        mu  = mu[:,:,None] + (self.alpha[:,None]*spk_vec)[:,None,:]



        #  Compute joint log-probability of spike counts and Gaussian noise. Careful about the trimming here.....


        loglivec = sp.special.logsumexp(logpoisspdf_mat + log_gaussian_1D(Y[:,:,None], mu, self.Gauss_sigma[:,None,None]), axis = 2)
        #import ipdb; ipdb.set_trace()
        #  sum up over time bins
        logli = np.sum(loglivec) # sum up over time and neurons
        return(logli)

        # elif self.AR2:

        #     b = np.zeros(self.Tps-2,1+S);
        #     f2 =  b + self.data[:-2]
        #     f1 =  b + self.data[2:-1]
        #     f0 =  b + self.data[3:]


            ##### To Do: AR2 PROCESS HERE!




