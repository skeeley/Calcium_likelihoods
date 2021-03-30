# Translated from Jonathan's matlab code
import jax
import jax.numpy as np 
import numpy.random as npr
from jax.scipy.special import gammaln, logsumexp 
from jax import grad, hessian, jit
from functools import partial 

from scipy.signal import convolve2d

import matplotlib.pyplot as plt

# Wrap Scipy's convolve2d to provide the same output as Matlab's conv2 for mode='same'
# from https://stackoverflow.com/questions/3731093/is-there-a-python-equivalent-of-matlabs-conv2-function
def conv2(x, y, mode='same'):
    return np.rot90(convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2)

@jit
def gaussian_1D(y, mu, sig2):
    return 1.0 / np.sqrt(2.0 * np.pi * sig2) * np.exp(-0.5 * (y - mu)**2 / sig2)

@jit
def log_gaussian_1D(y, mu, sig2):
    return -0.5 * np.log(2.0 * np.pi * sig2) -0.5 * (y - mu)**2 / sig2

@jit
def softplus(x):

    f = np.log1p(np.exp(x))
    logf = np.log(f)
    df = np.exp(x) / (1.0 + np.exp(x))
    ddf = np.exp(x) / (1.0 + np.exp(x))**2

    return f, logf, df, ddf

@jit
def exp_stable(x):

    f = np.exp(x)
    logf = x
    df = f
    ddf = f

    return f, logf, df, ddf

# @partial(jit, static_argnums=(4))
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
    tau, alpha, sig2 = hyperparams

    # compute AR(1) diffs
    taudecay = np.exp(-1.0/tau) # decay factor for one time bin
    Ydff = (Y[1:] - taudecay * Y[:-1]) / alpha

    # compute grid of spike counts
    ygrid = np.arange(0, S+1)

    # Gaussian log-likelihood terms
    log_gauss_grid = - 0.5 * (Ydff[:,None]-ygrid[None,:])**2 / sig2 - 0.5 * np.log(2.0 * np.pi * sig2)
    
    Xproj = X[1:,:]@w
    poissConst = gammaln(ygrid+1)

    # compute neglogli, gradient, and (optionally) Hessian
    f, logf, df, ddf = nlfun(Xproj)
    logPcounts = logf[:,None] * ygrid[None,:] - f[:,None] - poissConst[None,:]

    # compute log-likelihood for each time bin
    logjoint = log_gauss_grid + logPcounts 
    logli = logsumexp(logjoint, axis=1) # log likelihood for each time bin
    negL = -np.sum(logli) # negative log likelihood

    # gradient
    dLpoiss = (df / f)[:,None] * ygrid[None,:] - df[:,None] # deriv of Poisson log likelihood
    gwts = np.sum(np.exp(logjoint-logli[:,None]) * dLpoiss, axis=1) # gradient weights 
    gradient = -X[1:,:].T@gwts

    # Hessian 
    ddLpoiss = (ddf / f - (df / f)**2)[:,None] * ygrid[None,:] - ddf[:,None]
    ddL = (ddLpoiss + dLpoiss**2)
    hwts = np.sum(np.exp(logjoint-logli[:,None]) * ddL, axis=1) - gwts**2 # hessian weights
    H = -X[1:,:].T @ (X[1:,:] * hwts[:,None])

    return negL, gradient, H

# Set calcium model hyperparams
tau = 10        # decay in one time bin
alpha = 50.0     # gain
sig = 1.5      # stdev of Gaussian noise (in spike train space)
sig2 = sig**2   # variance of noise
hyperparams = [tau,alpha,sig2] 

S = 10 # max spike count to consider
ygrid = np.arange(0, S+1)

# Set up GLM
D_in = 19 # dimension of stimulus
D = D_in + 1 # total dims with bias
T = 50000
nlfun = lambda x : softplus(x)
# nlfun = lambda x : exp_stable(x) 

# Set GLM filter
wfilt = conv2(npr.randn(D_in,1), gaussian_1D(np.arange(1, D_in+1), D_in/2, 2)[:,None,])[:,0]
wDc = np.array([-3.5])
w = np.concatenate((wDc, 2*wfilt/np.linalg.norm(wfilt)))

# Generate simulated dataset
Xmat = np.hstack((np.ones((T,1)), npr.randn(T, D_in)))

# Simulate spike response
Xproj = Xmat @ w 
R, _, _, _ = nlfun(Xproj)
Ysps = npr.poisson(R)
print("Max number of spikes: ", np.max(Ysps))

# Generate Ca data 
# Generate using regular numpy because it is faster 
import numpy as onp
Yobs = onp.zeros(T)
Yobs[0] = alpha * Ysps[0] +  onp.sqrt(sig2) * npr.randn()
for t in range(1,T):
    Yobs[t] = alpha * Ysps[t] + onp.exp(-1.0 / tau) * Yobs[t-1] + onp.sqrt(sig2) * npr.randn()
# cast the generated data to a JAX array
Yobs = np.array(Yobs)

# plot simulated data
plot_idx = np.arange(100)
plt.figure()
plt.subplot(311)
plt.plot(R[plot_idx])
plt.subplot(312)
plt.plot(Ysps[plot_idx])
plt.subplot(313)
plt.plot(Yobs[plot_idx])

# compute nll, grad, hess
negL, gradient, H = nll_GLM_GanmorCalciumAR1(w, Xmat, Yobs, hyperparams, nlfun)

jit_nll_GLM_GanmorCalciumAR1 = jit(nll_GLM_GanmorCalciumAR1, static_argnums=(4))

# autodiff
grad_w = jit(grad(lambda w : nll_GLM_GanmorCalciumAR1(w, Xmat, Yobs, hyperparams, nlfun)[0]))
gradient_autodiff = grad_w(w)
hess_w = jit(hessian(lambda w : nll_GLM_GanmorCalciumAR1(w, Xmat, Yobs, hyperparams, nlfun)[0]))
H_autodiff = hess_w(w)

# diffs
print("Gradient vs. Autodiff    : ", np.linalg.norm(gradient_autodiff - gradient))
print("Hessian  vs. Autodiff    : ", np.linalg.norm(H - H_autodiff))

# speed comparisons
# %timeit negL, gradient, H = jit_nll_GLM_GanmorCalciumAR1(w, Xmat, Yobs, hyperparams, nlfun)
# %timeit gradient_autograd = grad_w(w)
# %timeit H_autograd = hess_w(w)
