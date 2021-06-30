# Translated from Jonathan's matlab code
import autograd.numpy as np 
import autograd.numpy.random as npr
from autograd.scipy.special import gammaln, logsumexp 
from autograd import grad, hessian 
npr.seed(314)

from scipy.signal import convolve2d
from scipy.optimize import minimize
from scipy.io import loadmat

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks")
sns.set_context("talk")

color_names = ["windows blue",
               "red",
               "amber",
               "faded green",
               "dusty purple",
               "orange",
               "clay",
               "pink",
               "greyish",
               "mint",
               "light cyan",
               "steel blue",
               "forest green",
               "pastel purple",
               "salmon",
               "dark brown"]
colors = sns.xkcd_palette(color_names)

# edit Scipy's convolve2d to provide the same output as Matlab's conv2 for mode='same'
# from https://stackoverflow.com/questions/3731093/is-there-a-python-equivalent-of-matlabs-conv2-function
def conv2(x, y, mode='same'):
    return np.rot90(convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2)

def gaussian_1D(y, mu, sig2):
    return 1.0 / np.sqrt(2.0 * np.pi * sig2) * np.exp(-0.5 * (y - mu)**2 / sig2)

def log_gaussian_1D(y, mu, sig2):
    return -0.5 * np.log(2.0 * np.pi * sig2) -0.5 * (y - mu)**2 / sig2

def softplus_stable(x, bias=None, dt=1.0):

    if bias is not None:
        inp = x+bias 
    else:
        inp = x

    f = np.log1p(np.exp(inp)) * dt 
    logf = np.log(f) 
    df = np.exp(inp) / (1.0 + np.exp(inp)) * dt
    ddf = np.exp(inp) / (1.0 + np.exp(inp))**2 * dt

    return f, logf, df, ddf

def exp_stable(x, dt=1.0):

    f = np.exp(x) * dt
    logf = x + np.log(dt)
    df = f
    ddf = f

    return f, logf, df, ddf

def logistic(x):
    return 1. / (1 + np.exp(-x))

def softplus(x):
    return np.log1p(np.exp(x))

#  mdic = {"weights": weights, "spikes": spikes, "label": "2021_06_21"}
d_sim = loadmat("2021_06_21_matlab_matrix.mat", squeeze_me=True)
d_naomi = loadmat("2021_06_21_Naomi_Sim_GLM_Neural.mat", squeeze_me=True)
d_stim = np.load("2021_06_21_matlab_stim.npz")

included = d_naomi['included']
idealTraces = d_naomi['idealTraces']
soma_spikes = d_naomi['soma_spikes']

naomi_idx = 71
neuron_idx = included[naomi_idx] - 1
neuron_trace = idealTraces[naomi_idx]
naomi_spikes = soma_spikes[naomi_idx] / 7.6e-6
sim_spikes = d_sim['spikes'][neuron_idx]
sim_weights = d_sim['weights'][neuron_idx]
print(np.linalg.norm(sim_spikes - naomi_spikes))

Xmat = d_stim["Xmat"]


plt.figure()
plt.plot(neuron_trace)


# fit model
def nll_GLM_GanmorCalciumAR1(w, X, Y, hyperparams, nlfun, S=10):
    """
    Negative log-likelihood for a GLM with Ganmor AR1 mixture model for calcium imaging data.

    Input:
        w:              [D x 1]  vector of GLM regression weights
        X:              [T x D]  design matrix
        Y:              [T x 1]  calcium fluorescence observations
        hyperparams:    [3 x 1]  model hyperparameters: log tau, log alpha, log Gaussian variance
        nlfun:          [func]   function handle for nonlinearity
        S:              [scalar] number of spikes to marginalize
        return_hess:    [bool]   flag for returning Hessian

    Output:
        negative log-likelihood, gradient, and Hessian
    """

    # unpack hyperparams
    tau, sig2, center, c_max = np.exp(hyperparams)

    # compute AR(1) diffs
    taudecay = np.exp(-1.0/tau) # decay factor for one time bin
    Y = np.pad(Y, (1,0)) # pad Y by a time bin
    Ydff = (Y[1:] - taudecay * Y[:-1]) 

    # compute grid of spike counts
    # scale via sigmoid function
    ygrid = np.arange(0, S+1)
    # yinc = softplus((ygrid - center) * c_max)
    yinc = np.concatenate(([0.], softplus((ygrid[1:] - center) * c_max)))

    # import ipdb; ipdb.set_trace()

    # Gaussian log-likelihood terms
    log_gauss_grid = - 0.5 * (Ydff[:,None]-yinc[None,:])**2 / (sig2) - 0.5 * np.log(2.0 * np.pi * sig2)
    
    Xproj = X@w
    poissConst = gammaln(ygrid+1)

    # compute neglogli, gradient, and (optionally) Hessian
    f, logf, df, ddf = nlfun(Xproj)
    logPcounts = logf[:,None] * ygrid[None,:] - f[:,None] - poissConst[None,:]

    # compute log-likelihood for each time bin
    logjoint = log_gauss_grid + logPcounts 
    logli = logsumexp(logjoint, axis=1) # log likelihood for each time bin
    negL = -np.sum(logli) # negative log likelihood

    return negL

# compute df /f 
medf = np.abs(np.mean(neuron_trace))
Yobs = (neuron_trace - medf) / medf
Yobs /= 40
plt.figure()
plt.plot(Yobs)

# todo
# need to downsample Xmat2. 
# in future -> up sample Xmat when generating spikes...
# idxs = np.round(np.arange(0,50000,3.3333)).astype(int)
# idxs = np.minimum(idxs, Xmat.shape[0]-1)
# Xmat2 = Xmat[idxs, :]
# Xmat2 = Xmat2[:Yobs.shape[0], :]

dt = 1.0 / 30.0
nlfun = lambda x : softplus_stable(x, dt=dt)

# compute nll
def _obj(_params):
    w = _params[0]
    hyperparams = _params[1]
    return nll_GLM_GanmorCalciumAR1(w, Xmat, Yobs, hyperparams, nlfun)

w_init = np.zeros((Xmat.shape[1]))
w_init[0] = 2.0
hyperparams_init = [np.log(10.0), np.log(1.0), np.log(2.0), np.log(4.0)]
_params = [w_init, hyperparams_init]

from autograd.misc import flatten
from scipy.optimize import minimize

params, unflatten = flatten(_params)
obj = lambda params : _obj(unflatten(params))
grad_func = grad(obj)

params_init = np.concatenate((w_init, hyperparams_init))
res = minimize(obj, x0=params_init, jac=grad_func)
params_mle = res.x

_params_mle = unflatten(params_mle)
w_mle = _params_mle[0]
hyperparams_mle = _params_mle[1]

# print("True hyperparams: ", np.exp(hyperparams))
plt.figure()
plt.plot(np.arange(0,20,1) * 1.0 / 30.0, sim_weights, 'k', label="sim")
plt.plot(np.arange(0,20,1) * 1.0 / 30.0, w_mle, 'b', label="fit")



# Regular AR(1)
def nll_GLM_GanmorCalciumAR1(w, X, Y, hyperparams, nlfun, S=10):
    """
    Negative log-likelihood for a GLM with Ganmor AR1 mixture model for calcium imaging data.

    Input:
        w:              [D x 1]  vector of GLM regression weights
        X:              [T x D]  design matrix
        Y:              [T x 1]  calcium fluorescence observations
        hyperparams:    [3 x 1]  model hyperparameters: log tau, log alpha, log Gaussian variance
        nlfun:          [func]   function handle for nonlinearity
        S:              [scalar] number of spikes to marginalize
        return_hess:    [bool]   flag for returning Hessian

    Output:
        negative log-likelihood, gradient, and Hessian
    """

    # unpack hyperparams
    tau, alpha, sig2 = np.exp(hyperparams)

    # compute AR(1) diffs
    taudecay = np.exp(-1.0/tau) # decay factor for one time bin
    Y = np.pad(Y, (1,0)) # pad Y by a time bin
    Ydff = (Y[1:] - taudecay * Y[:-1]) / alpha

    # compute grid of spike counts
    ygrid = np.arange(0, S+1)

    # Gaussian log-likelihood terms
    log_gauss_grid = - 0.5 * (Ydff[:,None]-ygrid[None,:])**2 / (sig2 / alpha**2) - 0.5 * np.log(2.0 * np.pi * sig2)
    
    Xproj = X@w
    poissConst = gammaln(ygrid+1)

    # compute neglogli, gradient, and (optionally) Hessian
    f, logf, df, ddf = nlfun(Xproj)
    logPcounts = logf[:,None] * ygrid[None,:] - f[:,None] - poissConst[None,:]

    # compute log-likelihood for each time bin
    logjoint = log_gauss_grid + logPcounts 
    logli = logsumexp(logjoint, axis=1) # log likelihood for each time bin
    negL = -np.sum(logli) # negative log likelihood

    return negL

dt = 1.0 / 30.0
nlfun = lambda x : softplus_stable(x, dt=dt)

# compute nll
def _obj(_params):
    w = _params[0]
    hyperparams = _params[1]
    return nll_GLM_GanmorCalciumAR1(w, Xmat, Yobs, hyperparams, nlfun)

w_init = np.zeros((Xmat.shape[1]))
w_init[0] = 2.0
hyperparams_init = [np.log(2.0), np.log(1.0), np.log(1.0)]
_params = [w_init, hyperparams_init]

from autograd.misc import flatten
from scipy.optimize import minimize

params, unflatten = flatten(_params)
obj = lambda params : _obj(unflatten(params))
grad_func = grad(obj)

params_init = np.concatenate((w_init, hyperparams_init))
res = minimize(obj, x0=params_init, jac=grad_func)
params_mle = res.x

_params_mle = unflatten(params_mle)
w_mle = _params_mle[0]
hyperparams_mle = _params_mle[1]


# print("True hyperparams: ", np.exp(hyperparams))
plt.figure()
plt.plot(np.arange(0,20,1) * 1.0 / 30.0, sim_weights, 'k', label="sim")
plt.plot(np.arange(0,20,1) * 1.0 / 30.0, w_mle, 'b', label="Ganmor")

# gaussian
w_mle_gauss = np.linalg.inv(Xmat.T@Xmat)@Xmat.T@Yobs
gauss_sig2 = np.mean((Xmat@w_mle_gauss - Yobs)**2)

def normalize(w):
    return w / np.linalg.norm(w)
plt.figure()
plt.plot(np.arange(1,20,1) * 1.0 / 30.0, normalize(sim_weights[1:]), 'k', label="sim")
plt.plot(np.arange(1,20,1) * 1.0 / 30.0, normalize(w_mle[1:]), 'b', label="Ganmor")
plt.plot(np.arange(1,20,1) * 1.0 / 30.0, normalize(w_mle_gauss[1:]), 'g', label="Gaussian")
plt.xlabel("time (s)")
plt.title("Normalized Filters")
plt.legend()