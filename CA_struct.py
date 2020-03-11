from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import jax
import jax.numpy as np
# Current convention is to import original numpy as "onp"
import numpy as onp

class CA_struct():

    # ---- Create struct and make stimulus design matrix ---------------------

    def __init__(
        self,
        dtStim: float = 0.03333333,
        calc_ts: int = 1,
        a: int = 1,
        max_spk: int = 10,
        Gauss_sigma: float = 0.2,
        T: int = 200, #initialize to 100 timesteps
        AR1: bool = True,
        AR2: bool = False
    ):


        if dtStim <= 0:
            raise ValueError("dtStim dim must be positive, got {0}".format(dtStim))
        if calc_ts <= 0:
            raise ValueError("Calcium trace timescale must be positive, got {0}".format(calc_ts))
        if a <= 0.0:
            raise ValueError("a bound must be positive, got {0}".format(a))
        if max_spk <= 0:
            raise ValueError("max_spk must be positive integer, got {0}".format(max_spk))

        self.dtStim = dtStim
        self.calc_ts = calc_ts
        self.a = a
        self.max_spk = max_spk
        self.Gauss_sigma = Gauss_sigma

        self.T = T
        self.AR1 = AR1
        self.AR2 = AR2



    def set_data(self, data):
        if np.shape(data)[-1] =! self.T:
            raise ValueError("data should be the same length as T in the calcium class, got {0}".format(np.shape(data)[-1]))
        self.data = data

