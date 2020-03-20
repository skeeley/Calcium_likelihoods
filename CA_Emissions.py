from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import jax
import jax.numpy as np
# Current convention is to import original numpy as "onp"
import numpy as onp


def log_poisson_1D(y, rate):
    return y * np.log(rate) - rate - gammaln(y+1)

def log_gaussian_1D(y, mu, sig2):
    return -0.5 * np.log(2.0 * np.pi * sig2) -0.5 * (y - mu)**2 / sig2



class CA_Emissions():

    # ---- Create struct and make stimulus design matrix ---------------------

    def __init__(
        self,
        dt: float = 0.03333333,
        tau: int = 1,
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

        self.T = T
        self.AR1 = AR1
        self.AR2 = AR2


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


            ##### DO AR2 PROCESS HERE!




