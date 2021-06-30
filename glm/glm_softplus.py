# Translated from Jonathan's matlab code
import autograd.numpy as np 
import autograd.numpy.random as npr
from autograd.scipy.special import gammaln, logsumexp 
from autograd import grad, hessian 
npr.seed(314)

from scipy.signal import convolve2d
from scipy.optimize import minimize

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

def logistic(x):
    return 1. / (1 + np.exp(-x))

def softplus(x):
    return np.log1p(np.exp(x))

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
tau = 15.0        # decay in one time bin
# alpha = 1.25     # gain
center = 1.5
c_max = 2.75
# sig = 0.2      # stdev of Gaussian noise (in spike train space)
sig2 = 0.01      # stdev of Gaussian noise (in spike train space)
sig = np.sqrt(sig2)   # variance of noise
hyperparams = [np.log(tau),np.log(sig2),np.log(center),np.log(c_max)] 

S = 10 # max spike count to consider
ygrid = np.arange(0, S+1)

# Set up GLM
D_in = 19 # dimension of stimulus
D = D_in + 1 # total dims with bias
T = 50000
dt = 1.0 / 30.0
bias = 0.0 * npr.randn(T)
nlfun = lambda x : softplus_stable(x, bias=bias, dt=dt)
# nlfun = lambda x : exp_stable(x) 

# Set GLM filter
wfilt = conv2(npr.randn(D_in,1), gaussian_1D(np.arange(1, D_in+1), D_in/2, 2)[:,None,])[:,0]
# wDc = np.array([1.0])
wDc = np.array([-2.5])
w = np.concatenate((wDc, 6*wfilt/np.linalg.norm(wfilt)))

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
Yobs = np.zeros(T)
if Ysps[0] == 0:
    Y_inc = 0.0
else:
    Y_inc = softplus((Ysps[0] - center) * c_max)
Yobs[0] = Y_inc +  np.sqrt(sig2) * npr.randn()
for t in range(1,T):
    if Ysps[t] == 0:
        Y_inc = 0.0
    else:
        Y_inc = softplus((Ysps[t] - center) * c_max)
    Yobs[t] = Y_inc + np.exp(-1.0 / tau) * Yobs[t-1] + np.sqrt(sig2) * npr.randn()

# add in some random measurement noise (not part of generative model)
Yobs = Yobs + 0.1 * npr.randn(*Yobs.shape)

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

# import ipdb; ipdb.set_trace()

# compute nll
def _obj(_params):
    w = _params[0]
    hyperparams = _params[1]
    return nll_GLM_GanmorCalciumAR1(w, Xmat, Yobs, hyperparams, nlfun)

# hyperparams = [np.log(tau), np.log(alpha), np.log(sig2)]
hyperparams = [np.log(tau),np.log(sig2),np.log(center),np.log(c_max)] 
_params = [w, hyperparams]

from autograd.misc import flatten
params, unflatten = flatten(_params)
obj = lambda params : _obj(unflatten(params))
grad_func = grad(obj)

from scipy.optimize import minimize
# params_init = 0.1 * npr.randn(params.shape[0])
# params_init = flatten
# w_init = npr.randn(D)
w_init = np.zeros((D))
w_init[0] = -2.0
hyperparams_init = [np.log(10.0), np.log(0.1), np.log(2.0), np.log(4.0)]
# hyperparams_init = hyperparams 
# hyperparams = [tau,np.log(sig2),center,np.log(scale),np.log(c_max)] 

params_init = np.concatenate((w_init, hyperparams_init))
res = minimize(obj, x0=params_init, jac=grad_func)
params_mle = res.x

_params_mle = unflatten(params_mle)
w_mle = _params_mle[0]
hyperparams_mle = _params_mle[1]

print("True hyperparams: ", np.exp(hyperparams))
print("MLE  hyperparams : ", np.exp(hyperparams_mle))

# plot spike to increase curves 
Ygrid = np.arange(0, 11)
# Yinc = softplus((Ygrid-center)*c_max)
Yinc = np.concatenate(([0.],softplus((Ygrid[1:]-center)*c_max)))
center_mle, c_max_mle = np.exp(hyperparams_mle[2:])
# Yinc_mle = softplus((Ygrid-center_mle)*c_max_mle)
Yinc_mle = np.concatenate(([0.],softplus((Ygrid[1:]-center_mle)*c_max_mle)))
plt.figure()
plt.plot(Ygrid, Yinc)
plt.plot(Ygrid, Yinc_mle)


# gaussian
# w_mle_gauss = np.linalg.lstsq(Xmat.T@Xmat, Xmat.T@Yobs, rcond=None)
w_mle_gauss = np.linalg.inv(Xmat.T@Xmat)@Xmat.T@Yobs
gauss_sig2 = np.mean((Xmat@w_mle_gauss - Yobs)**2)

Ytm1 = np.concatenate(([0], Yobs[:-1]))
Xar = np.hstack((Xmat, Ytm1[:,None]))
w_mle_gauss_ar = np.linalg.inv(Xar.T@Xar)@Xar.T@Yobs

# poisson
def poiss_log_like(w, Xmat, Ysps, nlfun):
    rate = nlfun(Xmat@w)[0]
    ll = np.sum(Ysps * np.log(rate) - rate - gammaln(Ysps+1.0))
    return -1.0 * ll 
poiss_grad_func = grad(poiss_log_like)
res_poiss = minimize(poiss_log_like, args=(Xmat, Ysps, nlfun), x0=w_mle_gauss, jac=poiss_grad_func)
w_mle_poiss = res_poiss.x

def gauss_log_like(w, Xmat, Yobs, gauss_sig2):
    mu = Xmat @ w
    N_len = len(Yobs)
    ll = -0.5 * np.sum((Yobs - mu) **2 ) / gauss_sig2
    ll += -0.5 * N_len * np.log(2.0 * np.pi * gauss_sig2)
    return ll

mu_ar = Xar @ w_mle_gauss_ar
ar_sig2 = np.mean((mu_ar - Yobs)**2)
def ar_log_like(w, Xar, Yobs, ar_sig2):
    mu = Xar @ w
    N_len = len(Yobs)
    ll = -0.5 * np.sum((Yobs - mu) **2 ) / ar_sig2
    ll += -0.5 * N_len * np.log(2.0 * np.pi * ar_sig2)
    return ll
ar_log_like(w_mle_gauss_ar, Xar, Yobs, ar_sig2)

plt.figure()
plt.plot(w[1:], 'k-', label="True")
plt.plot(w_mle_poiss[1:], 'r', label="Poisson", alpha=0.7)
plt.plot(w_mle_gauss[1:], 'b', label="Gaussian", alpha=0.7)
plt.plot(w_mle_gauss_ar[1:-1], 'c', label="AR", alpha=0.7)
plt.plot(w_mle[1:], 'g', label="Calcium", alpha=0.7)
plt.legend()


# plt.figure()
# plt.plot(w, 'k-', label="True")
# plt.plot(w_mle_poiss, 'r', label="Poisson", alpha=0.7)
# plt.plot(w_mle_gauss, 'b', label="Gaussian", alpha=0.7)
# plt.plot(w_mle, 'g', label="Calcium", alpha=0.7)
# plt.legend()

# def normalize(w):
    # return (w - np.min(w)) / np.max(w)

def normalize(w):
    return w / np.max(np.abs(w))
def normalize(w):
    return w / np.linalg.norm(w)
plt.figure()
plt.plot( normalize(w[1:]), 'k-', label="True")
plt.plot( normalize(w_mle_poiss[1:]), 'r', label="Poisson", alpha=0.7)
plt.plot( normalize(w_mle_gauss[1:]), 'b', label="Gaussian", alpha=0.7)
plt.plot( normalize(w_mle_gauss_ar[1:-1]), 'c', label="AR", alpha=0.7)
plt.plot( normalize(w_mle[1:]), 'g', label="Calcium", alpha=0.7)
plt.legend()

plt.figure()
plt.plot(normalize(w), 'k-', label="True")
plt.plot(normalize(w_mle_poiss), 'r', label="Poisson", alpha=0.7)
plt.plot(normalize(w_mle_gauss), 'b', label="Gaussian", alpha=0.7)
plt.plot(normalize(w_mle_gauss_ar[:-1]), 'c', label="AR", alpha=0.7)
plt.plot(normalize(w_mle), 'g', label="Calcium", alpha=0.7)
plt.title("normalized w/ mean")
plt.legend()

# calcium pred
def log_poisson_1D(y, rate):
    return y * np.log(rate) - rate - gammaln(y+1)

# predicted calcium influx
def calcium_pred(w, Xmat, Yobs, hyperparams, S=10):

    ca_pred = np.zeros_like(Yobs)
    rates = nlfun(Xmat@w)[0]
    tau_mle, alpha_mle, sig2_mle = np.exp(hyperparams)
    A_mle = np.array([np.exp(-1.0/tau_mle)])
    alpha_mle = np.array([alpha_mle])
    pad = np.zeros((1, 1))
    # mus = np.concatenate((pad, A_mle[None, :] * Yobs[:-1, None]))
    mus = np.zeros_like(Yobs)[:,None]

    # marginalize over spikes
    s = np.arange(0,S,1)
    mus = mus[:,:,None] + alpha_mle[None,:,None] * s
    outp = log_poisson_1D(s[None,None,:], rates[:,None,None])
    mus = np.sum(mus * np.exp(outp), axis=2)[:,0]
    return mus

plt.figure()
plt.plot(Yobs[plot_idx],'k')
plt.ylabel("calcium obs")
gauss_pred = Xmat@w_mle_gauss
# plt.plot(gauss_pred[plot_idx],'b')
gauss_pred_ar = Xar@w_mle_gauss_ar
plt.plot(gauss_pred_ar[plot_idx],'b',alpha=0.7)

plt.figure()
plt.plot(Yobs[plot_idx],'k')
plt.ylabel("calcium obs")
ca_pred = calcium_pred(w_mle, Xmat, Yobs, hyperparams_mle)
plt.plot(ca_pred[plot_idx],'g',alpha=0.7)
plt.xlim([0, T_plot])

T_plot=1000
plot_idx = np.arange(T_plot)
plt.figure()
plt.subplot(311)
plt.plot(R[plot_idx],color=[0.3,0.3,0.3])
plt.xticks([])
plt.xlim([0,T_plot])
plt.ylabel("firing rate")
plt.subplot(312)
plt.plot(Ysps[plot_idx],color=[0.3,0.3,0.3])
plt.xticks([])
plt.xlim([0,T_plot])
plt.ylabel("spike counts")
plt.subplot(313)
plt.plot(Yobs[plot_idx],color=[0.3,0.3,0.3])
plt.xlim([0,T_plot])
plt.ylabel("calcium obs")
ca_pred = calcium_pred(w_mle, Xmat, Yobs, hyperparams_mle)
gauss_pred = Xmat@w_mle_gauss
plt.plot(gauss_pred[plot_idx],'b',alpha=0.7)
plt.plot(ca_pred[plot_idx],'g',alpha=0.7)

plt.figure()
plt.plot( (Yobs - Ytm1)[plot_idx] , color='k', label="True", alpha=0.7)
gauss_pred = Xmat[:,1:]@w_mle_gauss[1:]
plt.plot(gauss_pred[plot_idx],'b', label="Gauss", alpha=0.7)
plt.plot(ca_pred[plot_idx],'g', label="Calcium", alpha=0.7)
plt.ylabel("$y_t - y_{t-1}$")
plt.xlim([0,T_plot])
plt.legend()
