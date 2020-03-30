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
    return y.T * np.log(rate) - rate - sp.special.gammaln(y+1)

def log_gaussian_1D(y, mu, sig2):
    return -0.5 * np.log(2.0 * np.pi * sig2) -0.5 * (y - mu)**2 / sig2



class CA_Emissions():

    # ---- Create struct and make stimulus design matrix ---------------------

    def __init__(
        self,
        dt: float = 0.03333333,
        AR_params: int = 10,
        alpha: int = 1,
        max_spk: int = 10,
        Gauss_sigma: float = 0.2,
        T: int = 200, #initialize to 100 timesteps
        AR1: bool = True,
        AR2: bool = False,
        link = "log"
    ):



        if dt <= 0:
            raise ValueError("dt must be positive, got {0}".format(dt))
        if  AR_params <= 0:
            raise ValueError("Calcium trace timescale must be positive, got {0}".format AR_params))
        if alpha <= 0.0:
            raise ValueError("a bound must be positive, got {0}".format(alpha))
        if max_spk <= 0:
            raise ValueError("max_spk must be positive integer, got {0}".format(max_spk))


        self.dt = dt
        self.AR_params =  AR_params
        self.alpha = alpha
        self.max_spk = max_spk
        self.Gauss_sigma = Gauss_sigma

        self.T = T
        # self.AR1 = AR1
        # self.AR2 = AR2


        mean_functions = dict(
            log=lambda x: np.exp(x) * self.dt,
            softplus= lambda x: softplus(x) * self.dt
            )
        self.mean = mean_functions[link]

    @property
    def params(self):
        return self.dt, self.alpha, self.Gauss_sigma, self.T, self.AR_params

    def set_data(self, data):

        if np.shape(data)[-1] != self.T:
            raise ValueError("data should be the same length as T in the calcium class, got {0}".format(np.shape(data)[-1]))
        self.data = data


    def sample_data(self, rate, AR = 1):
        '''
        Generate simulated data with default class params. Feel free to change if needed. Can be AR1 or AR2
        '''

        if np.shape(rate)[-1] != self.T:
            raise ValueError("rate should be the same length as T in the calcium class, got {0}".format(np.shape(data)[-1]))


        spikes = onp.random.poisson(rate)


        roots = np.exp(-self.dt/self.AR_params) ### here self.AR_params is an n-length vector where n is the AR process order
        coeffs = np.poly(roots) #find coefficients with of a polynomial with roots of params
        z = lfilter([self.alpha],coeffs,spk_counts) #generate AR-n process with inputs spk_counts, where self.alpha is the CA increase due to a spike


        # q = [exp(-self.dt/tau_samp),exp(-self.dt/tau_samp_b)]
        # a_true = onp.poly(q)
        # z = filter(1,a_true,spcounts);
        # ##### To Do: AR2 PROCESS HERE!

        trace = z + onp.sqrt(self.Gauss_sigma)*onp.random.randn(onp.size(z))#add noise

        return trace, spikes


    def log_likelihood(self,params, S = 10, learn_hyparams = False):
        try:
            self.data
        except AttributeError:
            print("please set data first")

        #Isolate calcium traces offset by correct index for AR1 and AR2
        AR_order = len(self.AR_params) #right now this is only funcitonal for AR1

        b = np.zeros([self.T-1,1+S])
        f1 =  b + self.data[0][:-1,None]
        f0 =  b  + self.data[0][1:,None]

        spk_vec = np.array([np.arange(S+1)])

        spk_mat =spk_vec*np.ones(np.shape(f1))

        if learn_hyparams:
            self.Gauss_sigma, self.alpha,self.AR_params = params[-(AR_order + 2):]

      


        ########### general purpose AR #############         
        mu = f1*np.exp(-self.dt/self.AR_params)+self.alpha*spk_mat


        ##### do Poiss part 
        rate = self.mean(params[:(self.T)]) #convert to rate given dt

        #  Compute Poisson log-probability for each rate, for each spike count in spkvect
        logpoisspdf_mat = log_poisson_1D(spk_vec, [:,None])

        #  Compute joint log-probability of spike counts and Gaussian noise

        loglivec = np.logsumexp(logpoisspdf_mat + log_gaussian_1D(f0, mu, self.Gauss_sigma) , axis = 1)

        #  sum up over time bins
        neglogli = -np.sum(loglivec) # sum up and take negative 

        return(neglogli)

        # elif self.AR2:

        #     b = np.zeros(self.T-2,1+S);
        #     f2 =  b + self.data[:-2]
        #     f1 =  b + self.data[2:-1]
        #     f0 =  b + self.data[3:]


            ##### To Do: AR2 PROCESS HERE!




