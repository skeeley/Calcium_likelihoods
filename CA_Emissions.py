from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



from scipy.signal import lfilter


import jax
import jax.numpy as np
import jax.numpy.random as npr

# Current convention is to import original numpy as "onp"
import numpy as onp


def log_poisson_1D(y, rate):
    return y * np.log(rate) - rate - gammaln(y+1)

def log_gaussian_1D(y, mu, sig2):
    return -0.5 * np.log(2.0 * np.pi * sig2) -0.5 * (y - mu)**2 / sig2



class CA_Emissions():


    def __init__(
        self,
        dt: float = 0.03333333,
        tau: int = 1,
        alpha: int = 1,
        max_spk: int = 10,
        Gauss_sigma: float = 0.2,
        T: int = 200, #initialize to 100 timesteps
        lags: int = 1
        As = None
        link = "log"
    ):



        if dt <= 0:
            raise ValueError("dt must be positive, got {0}".format(dt))
        if tau <= 0:
            raise ValueError("Calcium trace timescale must be positive, got {0}".format(tau))
        if alpha <= 0.0:
            raise ValueError("a bound must be positive, got {0}".format(alpha))
        if max_spk <= 0:
            raise ValueError("max_spk must be positive integer, got {0}".format(max_spk))


        self.dt = dt
        self.tau = tau
        self.alpha = alpha
        self.max_spk = max_spk
        self.Gauss_sigma = Gauss_sigma
        self.nlags = nlags
        self.T = T
        if self.As is None:
            self.As = np.abs(npr.randn(lags, N)) # autoregressive component must be positive!
        #self.inv_etas = -4 + npr.randn(1, N) 


        mean_functions = dict(
            log=lambda x: np.exp(x) * self.bin_size,
            softplus= lambda x: softplus(x) * self.bin_size
            )
        self.mean = mean_functions[link]

    @property
    def params(self):
        return self.dt, self.alpha, self.Gauss_sigma, self.T, self.AR1, self.AR2  

    def set_data(self, data):

        if np.shape(data)[-1] != self.T:
            raise ValueError("data should be the same length as T in the calcium class, got {0}".format(np.shape(data)[-1]))
        self.data = data


    def sample_data(self, rate, lags = self.lags, alpha_samp = samp.alpha, sigma_samp = self.Gauss_sigma):
        '''
        Generate simulated data with default class params. Feel free to change if needed. Can be AR1 or AR2
        '''
        N, T = rate.shape[0], self.T
        if np.shape(rate)[-1] != self.T:
            raise ValueError("rate should be the same length as T in the calcium class, got {0}".format(np.shape(data)[-1]))


        spikes = npr.poisson(rate)
        mus = np.zeros_like(spikes)
        y = np.zeros((N, T))

        y[0:L] = mus[0:lags, N] + sigma_samp * npr.randn(lags, N)
        for t in range(lags, T):
            for l in range(lags):
                Al = self.As[:, l:(l + 1)]
                mus_ar= mus_ar + np.dot(y[lags-l-1:-l-1], Al.T)
                
            y[t] =  mus_ar + alpha_samp * spikes[:,t] + sigma_samp* npr.randn(N) 
        return y
        # if self.AR1: 
        #     z= lfilter(alpha_samp,[1,-np.exp(-self.dt/tau_samp)],spikes)
        #     trace = z + np.sqrt(sigma_samp)*np.random.randn(np.size(z))#add noise



        # elif self.AR2:  
        #     a = 0
        #     ##### To Do: AR2 PROCESS HERE!


    def log_likelihood(self,rate_param, S = 10):
        try:
            self.data()
        except AttributeError:
            print("please set data first")

                #Isolate calcium traces offset by correct index for AR1 and AR2
        if self.AR1: 

            b = np.zeros([self.T-1,1+S])
            f1 =  b + self.data[:-1,None]
            f0 =  b  + self.data[1:,None]


            spk_vec = np.arange(S+1) 
            spk_mat =spk_vec*np.ones(np.shape(f1))
            mu = f1*np.exp(-self.dt/self.tau)+self.alpha*spk_mat
            self.Gauss_ll = log_gaussian_1D(f0, mu, self.Gauss_sigma) #set Gaussian part


            ##### do Poiss part 
            rate = self.mean(rate_param) #convert to rate given dt

            #  Compute Poisson log-probability for each rate, for each spike count in spkvect
            logpoisspdf_mat = log_poisson_1D(spk_vec, rate)



            #  Compute joint log-probability of spike counts and Gaussian noise
            logpjoint = logpoisspdf_mat + self.Gauss_ll

            loglivec = np.logsumexp(logpjoint, axis = 1)

            #  sum up over time bins
            neglogli = -np.sum(loglivec) # sum up and take negative 

            return(neglogli)

        elif self.AR2:

            b = np.zeros(self.T-2,1+S);
            f2 =  b + self.data[:-2]
            f1 =  b + self.data[2:-1]
            f0 =  b + self.data[3:]


            ##### To Do: AR2 PROCESS HERE!




