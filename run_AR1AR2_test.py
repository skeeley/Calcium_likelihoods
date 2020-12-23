from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import jax
import jax.numpy as np
# Current convention is to import original numpy as "onp"
import numpy as onp
from jax import grad, jit, vmap
from jax.experimental import optimizers
from jax.tree_util import tree_multimap  # Element-wise manipulation of 
from jax import random
from jax import jacfwd

import GP_fourier as gpf 

import matplotlib.pyplot as plt
import matplotlib
matplotlib.interactive(True)

from CA.CA_new import CA_Emissions
from CA.misc import bbvi, make_cov

from jax.config import config
config.update("jax_enable_x64", True)