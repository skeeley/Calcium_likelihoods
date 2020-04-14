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




class CA_Emissions():

    # ---- Create struct and make stimulus design matrix ---------------------

    def __init__(
        self,
        dt: float = 0.01,
        AR_params: list = [.05],
        alpha: int = 1,
        max_spk: int = 10,
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
        if max_spk <= 0:
            raise ValueError("max_spk must be positive integer, got {0}".format(max_spk))


        self.dt = dt
        self.AR_params =  AR_params
        self.alpha = alpha
        self.max_spk = max_spk
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

    def set_data(self, data, S = 10):

        if np.shape(data)[-1] != self.Tps:
            raise ValueError("data should be the same length as T in the calcium class, got {0}".format(np.shape(data)[-1]))
        self.data = data

        #Isolate calcium traces offset by correct index for AR1 and AR2
        AR_order = np.size(self.AR_params) #right now this is only funcitonal for AR1



        # b = np.zeros([self.Tps-1,1+S])
        # for i in np.arange(AR_order)
        #     f1 =  b + self.data[0][:-1,None]
        #     f0 =  b  + self.data[0][1:,None]


        b = onp.zeros([AR_order+1, self.Tps+AR_order,1+S])
        for i in onp.arange(AR_order+1):
            b[i, (-i+AR_order):self.Tps+(1-i),:] = self.data[0][:,None]

            # ok right now I think storing a big matrix with the data being offset by the AR order is not optimal.
            # can probably move to something in the LL calculation with a loop over AR parameters. Not sure the 
            # fastest method here.



        self.data_mat = b

        self.spk_vec = np.array([np.arange(S+1)])

        self.spk_mat =self.spk_vec*np.ones(np.shape(self.Tps+AR_order))


    def sample_data(self, rate, AR = 1):
        '''
        Generate simulated data with default class params. Feel free to change if needed. Can be AR1 or AR2
        '''

        if np.shape(rate)[-1] != self.Tps:
            raise ValueError("rate should be the same length as T in the calcium class, got {0}".format(np.shape(data)[-1]))


        spikes = onp.random.poisson(rate)


        roots = onp.exp(-self.dt/np.array(self.AR_params)) ### here self.AR_params is an n-length vector where n is the AR process order
        coeffs = onp.poly(roots) #find coefficients with of a polynomial with roots of params
        z = lfilter([self.alpha],coeffs,spikes) #generate AR-n process with inputs spk_counts, where self.alpha is the CA increase due to a spike


        # q = [exp(-self.dt/tau_samp),exp(-self.dt/tau_samp_b)]
        # a_true = onp.poly(q)
        # z = filter(1,a_true,spcounts);
        # ##### To Do: AR2 PROCESS HERE!

        trace = z + onp.sqrt(self.Gauss_sigma)*onp.random.randn(onp.size(z))#add noise

        return trace, spikes


    def log_likelihood(self,params, learn_hyparams = False):
        try:
            self.data #this should pre-set a number of parameters for optimization, as well.
        except AttributeError:
            print("please set data first")



        if learn_hyparams:
            self.Gauss_sigma, self.alpha,self.AR_params = params[-(AR_order + 2):]

      
        AR_order = np.size(self.AR_params)

        ##### do Poiss part 
        rate = self.mean(params[:(self.Tps)]) #convert to rate given dt

        #  Compute Poisson log-probability for each rate, for each spike count in spkvect
        logpoisspdf_mat = log_poisson_1D(self.spk_vec, rate[:,None])


        ########### general purpose AR from data_mat (to do) #############       

        mu = self.data_mat[1]*np.exp(-self.dt/np.array(self.AR_params))+self.alpha*self.spk_mat

        #  Compute joint log-probability of spike counts and Gaussian noise. Careful about the trimming here.....

        loglivec = sp.special.logsumexp(logpoisspdf_mat[:-AR_order] + log_gaussian_1D(self.data_mat[0], mu, self.Gauss_sigma)[AR_order:-AR_order] , axis = 1)

        #  sum up over time bins
        logli = np.sum(loglivec) # sum up and take negative 

        return(logli)

        # elif self.AR2:

        #     b = np.zeros(self.Tps-2,1+S);
        #     f2 =  b + self.data[:-2]
        #     f1 =  b + self.data[2:-1]
        #     f0 =  b + self.data[3:]


            ##### To Do: AR2 PROCESS HERE!




