# Translated from Jonathan's matlab code
import autograd.numpy as np 
import autograd.numpy.random as npr
from autograd.scipy.special import gammaln, logsumexp 
from autograd import grad, hessian 
npr.seed(314)

from scipy.signal import convolve2d

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

# def softplus(x, dt=1.0):

#     f = np.log1p(np.exp(x)) * dt 
#     logf = np.log(f) 
#     df = np.exp(x) / (1.0 + np.exp(x)) * dt
#     ddf = np.exp(x) / (1.0 + np.exp(x))**2 * dt

#     return f, logf, df, ddf

def softplus_stable(x, bias=None, dt=1.0):

    if bias is not None:
        inp = x+bias 

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
    ar_coefs, log_alpha, log_sig2 = hyperparams
    alpha = np.exp(log_alpha)
    sig2 = np.exp(log_sig2)
    p = ar_coefs.shape[0] # AR(p)
    # compute AR(p) diffs
    Ydecay = np.zeros_like(Y)
    Y = np.pad(Y, (p,0)) # pad Y by p time bins
    for i, ai in enumerate(ar_coefs):
        Ydecay = Ydecay + ai * Y[p-1-i:-1-i]
    # Ydecay2 = ar_coefs[0] * Y[1:-1]
    # Ydecay2 = Ydecay2 + ar_coefs[1] * Y[:-2]
    # print(np.linalg.norm(Ydecay - Ydecay2))
    # import ipdb; ipdb.set_trace()
    Ydff = (Y[p:] - Ydecay) / alpha

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

    # # gradient
    # dLpoiss = (df / f)[:,None] * ygrid[None,:] - df[:,None] # deriv of Poisson log likelihood
    # gwts = np.sum(np.exp(logjoint-logli[:,None]) * dLpoiss, axis=1) # gradient weights 
    # gradient = -X.T@gwts

    # # Hessian 
    # ddLpoiss = (ddf / f - (df / f)**2)[:,None] * ygrid[None,:] - ddf[:,None]
    # ddL = (ddLpoiss + dLpoiss**2)
    # hwts = np.sum(np.exp(logjoint-logli[:,None]) * ddL, axis=1) - gwts**2 # hessian weights
    # H = -X.T @ (X * hwts[:,None])

    # return negL, gradient, H

# Set calcium model hyperparams
p = 2
ar_coefs = np.array([1.51,-0.6])        # decay in one time bin
alpha = 1.0     # gain
# sig = 0.2      # stdev of Gaussian noise (in spike train space)
sig2 = 0.001      # stdev of Gaussian noise (in spike train space)
sig = np.sqrt(sig2)   # variance of noise
hyperparams = [ar_coefs,np.log(alpha),np.log(sig2)] 

S = 10 # max spike count to consider
ygrid = np.arange(0, S+1)

# Set up GLM
D_in = 19 # dimension of stimulus
D = D_in + 1 # total dims with bias
T = 50000
dt = 0.5
bias = npr.randn(T)
nlfun = lambda x : softplus_stable(x, bias=bias, dt=dt)
# nlfun = lambda x : exp_stable(x) 

# Set GLM filter
wfilt = conv2(npr.randn(D_in,1), gaussian_1D(np.arange(1, D_in+1), D_in/2, 2)[:,None,])[:,0]
# wDc = np.array([1.0])
wDc = np.array([-5.5])
w = np.concatenate((wDc, 2*wfilt/np.linalg.norm(wfilt)))

# Generate simulated dataset
# Xmat = np.hstack((np.ones((T,1)), npr.randn(T, D_in)))

Xstim = npr.randn(T,1)
from scipy.linalg import hankel
Xmat1 = hankel(Xstim[:,0], Xstim[:,0][-19:])
Xmat = np.hstack((np.ones((T,1)), Xmat1))

# import ssm
# from ssm import LDS 
# from ssm.util import random_rotation
# true_lds = LDS(2, D_in, emissions="gaussian")
# A0 = .95 * random_rotation(D_in, theta=np.pi/40)
# S = np.arange(1, D_in+1)
# R = np.linalg.svd(npr.randn(D_in, D_in))[0] * S
# A = R.dot(A0).dot(np.linalg.inv(R))
# b =  np.zeros(D_in)
# true_lds.dynamics.As[0] = A
# true_lds.dynamics.bs[0] = b
# true_lds.dynamics.Sigmas = true_lds.dynamics.Sigmas / np.max(true_lds.dynamics.Sigmas[0]) * 0.5
# x, y = true_lds.sample(T)
# x = x / np.max(x) * 5.0

# Xmat = np.hstack((np.ones((T,1)), x))


# Simulate spike response
Xproj = Xmat @ w 
R, _, _, _ = nlfun(Xproj)
Ysps = npr.poisson(R)
print("Max number of spikes: ", np.max(Ysps))

# Generate Ca data
Yobs = np.zeros(T+p)
for t in range(p,T+p):
    ar_term = np.sum([ar_coefs[i] * Yobs[t-1-i] for i in range(p)])
    Yobs[t] = alpha * Ysps[t-p] + ar_term + np.sqrt(sig2) * npr.randn()
Yobs = Yobs[p:]


# plot simulated data
T_plot=1000
plot_idx = np.arange(T_plot)
plt.figure()
plt.subplot(411)
plt.plot(Xstim[plot_idx])
plt.xticks([])
plt.xlim([0,T_plot])
plt.title("stimulus")
plt.subplot(412)
plt.plot(R[plot_idx])
plt.xticks([])
plt.xlim([0,T_plot])
plt.title("firing rate")
plt.subplot(413)
plt.plot(Ysps[plot_idx])
plt.xticks([])
plt.xlim([0,T_plot])
plt.title("spike counts")
plt.subplot(414)
plt.plot(Yobs[plot_idx])
plt.xlim([0,T_plot])
plt.title("calcium obs")
plt.tight_layout()

# compute nll
def _obj(_params):
    w = _params[0]
    hyperparams = _params[1]
    return nll_GLM_GanmorCalciumAR1(w, Xmat, Yobs, hyperparams, nlfun)

hyperparams = [ar_coefs, np.log(alpha), np.log(sig2)]
_params = [w, hyperparams]

from autograd.misc import flatten
params, unflatten = flatten(_params)
obj = lambda params : _obj(unflatten(params))
grad_func = grad(obj)

from scipy.optimize import minimize
# params_init = 0.1 * npr.randn(params.shape[0])
# params_init = flatten
w_init = 0.1 * npr.randn(D)
ar_init = np.array([0.1,-0.1])
alpha_init = 1.0
sig2_init = 0.01
hyperparams_init = [ar_init, np.log(alpha_init), np.log(sig2_init)]
params_init, _ = flatten([w_init,hyperparams_init])
# params_init = params + 0.0
res = minimize(obj, x0=params_init, jac=grad_func)
params_mle = res.x

_params_mle = unflatten(params_mle)
w_mle_ar2 = _params_mle[0]
hyperparams_mle = _params_mle[1]

def convert_hyperparams(hyperparams):
    ar_coefs, log_alpha, log_sig2 = hyperparams 
    return ar_coefs[0], ar_coefs[1], np.exp(log_alpha), log_sig2

print("True hyperparams: ", hyperparams)
print("MLE  hyperparams : ", hyperparams_mle)

plt.figure(figsize=[8,8])
plt.subplot(321)
plt.plot(w[1:], 'k-', label="True")
plt.plot(w_mle_ar2[1:], 'g', label="Calcium AR(2)", alpha=0.7)
# plt.plot(w_mle_1[1:], 'b', label="Calcium AR(1))", alpha=0.7)
plt.legend()
plt.title("w/o noise")
plt.subplot(322)
plt.plot(convert_hyperparams(hyperparams)[:3], convert_hyperparams(hyperparams_mle)[:3],'.')
plt.xlim(plt.gca().get_ylim())
plt.gca().axis("square")
# plt.plot()
plt.tight_layout()

# add in some random measurement noise (not part of generative model)
Yobs = Yobs + 0.2 * npr.randn(*Yobs.shape)

# compute nll
def _obj(_params):
    w = _params[0]
    hyperparams = _params[1]
    return nll_GLM_GanmorCalciumAR1(w, Xmat, Yobs, hyperparams, nlfun)

hyperparams = [ar_coefs, np.log(alpha), np.log(sig2)]
_params = [w, hyperparams]

from autograd.misc import flatten
params, unflatten = flatten(_params)
obj = lambda params : _obj(unflatten(params))
grad_func = grad(obj)

res = minimize(obj, x0=params_init, jac=grad_func)
params_mle = res.x

_params_mle = unflatten(params_mle)
w_mle_ar2_noise = _params_mle[0]
hyperparams_mle = _params_mle[1]
print("MLE  hyperparams : ", hyperparams_mle)

plt.subplot(323)
plt.plot(w[1:], 'k-', label="True")
plt.plot(w_mle_ar2_noise[1:], 'g', label="Calcium AR(2)", alpha=0.7)
# plt.plot(w_mle_1[1:], 'b', label="Calcium AR(1))", alpha=0.7)
plt.legend()
plt.title("w/ noise")
plt.subplot(324)
plt.plot(convert_hyperparams(hyperparams)[:3], convert_hyperparams(hyperparams_mle)[:3],'.')
# plt.xlim(plt.gca().get_ylim())
# plt.gca().axis("square")
# plt.plot()
plt.tight_layout()
# fit with AR(1)

hyperparams = [np.array([1.0]), np.log(alpha), np.log(sig2)]
_params = [w, hyperparams]

def _obj(_params):
    w = _params[0]
    hyperparams = _params[1]
    return nll_GLM_GanmorCalciumAR1(w, Xmat, Yobs, hyperparams, nlfun)

from autograd.misc import flatten
params, unflatten = flatten(_params)
obj = lambda params : _obj(unflatten(params))
grad_func = grad(obj)

res = minimize(obj, x0=params_init, jac=grad_func)
params_mle = res.x

_params_mle = unflatten(params_mle)
w_mle_ar1 = _params_mle[0]
hyperparams_mle = _params_mle[1]
print("MLE  hyperparams : ", hyperparams_mle)

plt.subplot(325)
plt.plot(w[1:], 'k-', label="True")
plt.plot(w_mle_ar1[1:], 'g', label="Calcium AR(2)", alpha=0.7)
# plt.plot(w_mle_1[1:], 'b', label="Calcium AR(1))", alpha=0.7)
plt.legend()
plt.title("AR (1)")
# plt.subplot(326)
# plt.plot(convert_hyperparams(hyperparams)[:3], convert_hyperparams(hyperparams_mle)[:3],'.')
# # plt.xlim(plt.gca().get_ylim())
# # plt.gca().axis("square")
# # plt.plot()
plt.tight_layout()


# T_plot=1000
# plot_idx = np.arange(T_plot)
plt.figure()
plt.subplot(411)
plt.plot(Xstim[plot_idx])
plt.xticks([])
plt.xlim([0,T_plot])
plt.title("stimulus")
plt.subplot(412)
plt.plot(R[plot_idx])
plt.xticks([])
plt.xlim([0,T_plot])
plt.title("firing rate")
plt.subplot(413)
plt.plot(Ysps[plot_idx])
plt.xticks([])
plt.xlim([0,T_plot])
plt.title("spike counts")
plt.subplot(414)
plt.plot(Yobs[plot_idx])
plt.xlim([0,T_plot])
plt.title("calcium obs")
plt.tight_layout()

Ytm1 = np.concatenate(([0], Yobs[:-1]))
Ytm2 = np.concatenate((np.array([0.0,0.0]), Yobs[:-2]))
Xar = np.hstack((Xmat, Ytm1[:,None], Ytm2[:,None]))
w_mle_gauss_ar2 = np.linalg.inv(Xar.T@Xar)@Xar.T@Yobs
# Xar_zero = np.hstack((Xmat[:,:1], Ytm1[:,None]))
# w_mle_gauss_ar_zero = np.linalg.inv(Xar_zero.T@Xar_zero)@Xar_zero.T@Yobs
# def normalize(w):
    # return w / np.max(np.abs(w))
def normalize(w):
    return w / np.linalg.norm(w)
plt.figure()
plt.plot( normalize(w[1:]), 'k-', label="True")
# plt.plot( normalize(w_mle_poiss[1:]), 'r', label="Poisson", alpha=0.7)
# plt.plot( normalize(w_mle_gauss[1:]), 'b', label="Gaussian", alpha=0.7)
plt.plot( normalize(w_mle_ar2[1:]), 'r', label="Calcium AR(2), No Noise", alpha=0.7)
plt.plot( normalize(w_mle_ar2_noise[1:]), 'b', label="Calcium AR(2), Noise", alpha=0.7)
plt.plot( normalize(w_mle_ar1[1:]), 'g', label="Calcium AR(1), Noise", alpha=0.7)
plt.plot( normalize(w_mle_gauss_ar2[1:-2]), 'c', label="AR(2)", alpha=0.7)
plt.title("normalized filters")
plt.legend()
