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
        AR_params: list = [.05],
        alpha: int = 1,
        Gauss_sigma: float = 0.1,
        Tps: int = 200, #initialize to 100 timesteps
        link = "log"
    ):



        if dt <= 0:
            raise ValueError("dt must be positive, got {0}".format(dt))
        # if  AR_params <= 0:
        #     raise ValueError("Calcium trace timescale must be positive, got {0}".format(AR_params))
        if alpha <= 0.0:
            raise ValueError("a bound must be positive, got {0}".format(alpha))



        self.dt = dt
        self.AR_params =  AR_params
        self.alpha = alpha
        self.Gauss_sigma = Gauss_sigma

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

    # def set_data(self, data, S = 10):

    #     if np.shape(data)[-1] != self.Tps:
    #         raise ValueError("data should be the same length as T in the calcium class, got {0}".format(np.shape(data)[-1]))


    #     self.data = data

    #     #Isolate calcium traces offset by correct index for AR1 and AR2
    #     AR_order = np.size(self.AR_params) #right now this is only funcitonal for AR1



    #     # b = np.zeros([self.Tps-1,1+S])
    #     # for i in np.arange(AR_order)
    #     #     f1 =  b + self.data[0][:-1,None]
    #     #     f0 =  b  + self.data[0][1:,None]


    #     b = onp.zeros([AR_order+1, self.Tps+AR_order,1+S])



    #     for i in onp.arange(AR_order+1):
    #         b[i, (-i+AR_order):self.Tps+(1-i),:] = self.data[0][:,None]

    #         # ok right now I think storing a big matrix with the data being offset by the AR order is not optimal.
    #         # can probably move to something in the LL calculation with a loop over AR parameters. Not sure the 
    #         # fastest method here.



    #     self.data_mat = b

    #     self.spk_vec = np.array([np.arange(S+1)])

    #     self.spk_mat =self.spk_vec*np.ones(np.shape(self.Tps+AR_order))


    def sample_data(self, rate, AR = 1):
        '''
        Generate simulated data with default class params. Feel free to change if needed. Can be AR1 or AR2
        '''

        if np.shape(rate)[-1] != self.Tps:
            raise ValueError("rate should be the same length as T in the calcium class, got {0}".format(np.shape(data)[-1]))


        spikes = onp.random.poisson(rate)

        roots = onp.exp(-self.dt/np.array(self.AR_params)) ### here self.AR_params is an n-length vector where n is the AR process order
        coeffs = onp.poly(roots) #find coefficients with of a polynomial with roots of params
        #trace = lfilter([self.alpha],coeffs,noisey_spks) #generate AR-n process with inputs spk_counts, where self.alpha is the CA increase due to a spike
        # Generate Ca data
        Yobs = onp.zeros(np.shape(spikes))
        Yobs[:,0] = self.alpha * spikes[:,0] +  onp.sqrt(self.Gauss_sigma) * onp.random.randn()

        for t in range(1,self.Tps):
            Yobs[:,t] = self.alpha * spikes[:,t] + onp.exp(-self.dt/np.array(self.AR_params)) * Yobs[:,t-1] +  onp.sqrt(self.Gauss_sigma) * onp.random.randn()

        # q = [exp(-self.dt/tau_samp),exp(-self.dt/tau_samp_b)]
        # a_true = onp.poly(q)
        # z = filter(1,a_true,spcounts);
        # ##### To Do: AR2 PROCESS HERE!



        return Yobs, spikes


    def log_likelihood(self, Y, X, S = 10, learn_model_params = False):

        '''
        #Inputs
        Y: Calcium trace time series (now with batching over neurons!)
        X: log-rate time series (now with batching over neurons!)

        '''
        # if np.array(self.alpha).ndim == 0:
        #     self.alpha = np.expand_dims(self.alpha, axis = 0)
        # if np.array(self.Gauss_sigma).ndim == 0:
        #     self.Gauss_sigma = np.expand_dims(self.Gauss_sigma, axis = 0)

        try:
             np.shape(X) == np.shape(Y) #this should pre-set a number of parameters for optimization, as well.
        except AttributeError:
             print("X and Y must be same shape")

        if X.ndim == 2:
            n_neurs = np.shape(X)[0]
        else:
            n_neurs = 1
            X = np.expand_dims(X, axis = 0)

        #AR_order = np.size(self.AR_params)
        AR_order = 1 #FOR NOW....This is very annoying cause of casting. On the line below all the object attributes become of size n_neurons

        if learn_model_params:
            self.Gauss_sigma, self.alpha, self.AR_params = X[:,-(AR_order + 2):-(AR_order + 1)].T[0], X[:,-(AR_order + 1):-(AR_order)].T[0], X[:,-AR_order:].T[0]
            # if AR_order == 1 & n_neurs == 1:
            #     self.AR_params = [self.AR_params]


      
        




        ##### do Poiss part 
        rate = self.mean(X[:,:(self.Tps)]) #convert to rate given dt


        #  Compute Poisson log-probability for each rate, for each spike count in spkvect
        spk_vec = np.tile(np.arange(S+1),[n_neurs, 1])
        logpoisspdf_mat = log_poisson_1D(spk_vec[:,None,:], rate[:,:,None])





        ########### general purpose AR from data_mat (to do) #############       


        padded_Y = np.pad(Y,[(0, 0), (AR_order,0)], mode='constant')
        mu = np.zeros(np.shape(Y))
        for i in np.arange(AR_order):
            mu += padded_Y[:,AR_order -i -1:-i -1]*self.AR_params[i] 



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




