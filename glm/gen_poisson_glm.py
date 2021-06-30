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

# Set up GLM
D_in = 19 # dimension of stimulus
D = D_in + 1 # total dims with bias
T = 50000
dt = 0.01
bias = 0.0 * npr.randn(T)
nlfun = lambda x : softplus_stable(x, bias=bias, dt=dt)
# nlfun = lambda x : exp_stable(x) 

# Generate simulated dataset
# Xmat = np.hstack((np.ones((T,1)), npr.randn(T, D_in)))
T_stim = 15000
Xstim = npr.randn(T_stim,1)
from scipy.linalg import hankel
Xmat1 = hankel(Xstim[:,0], Xstim[:,0][-19:])
Xmat = np.hstack((np.ones((T_stim,1)), Xmat1))

num_neurons = 343

weights = np.zeros((num_neurons, D))
spikes = np.zeros((num_neurons, T))

expand_idxs = np.floor(np.arange(0.0,50000,1) * 0.01 / (1.0 / 30.0)).astype(int)

for n in range(num_neurons):

    # Set GLM filter
    wfilt = conv2(npr.randn(D_in,1), gaussian_1D(np.arange(1, D_in+1), D_in/2, 2)[:,None,])[:,0]
    wDc = np.array([2.0])
    w = np.concatenate((wDc, 4*wfilt/np.linalg.norm(wfilt)))

    # Simulate spike response
    Xproj = Xmat @ w 
    Xproj = Xproj[expand_idxs]
    R, _, _, _ = nlfun(Xproj)
    R_hz = R / dt
    Ysps = npr.poisson(R)

    weights[n] = w 
    spikes[n] = Ysps

from scipy.io import savemat
mdic = {"weights": weights, "spikes": spikes, "label": "2021_06_21"}
savemat("2021_06_21_matlab_matrix.mat", mdic)
# mdic_stim = {"Xmat": Xmat}
np.savez("2021_06_21_matlab_stim.npz", Xmat=Xmat)


# plot simulated data
# T_plot=1000
# plot_idx = np.arange(T_plot)
# plt.figure()
# plt.subplot(311)
# plt.plot(Xstim[plot_idx])
# plt.xticks([])
# plt.xlim([0,T_plot])
# plt.title("stimulus")
# plt.subplot(312)
# plt.plot(R[plot_idx])
# plt.xticks([])
# plt.xlim([0,T_plot])
# plt.title("firing rate (sp/bin)")
# plt.subplot(313)
# plt.plot(Ysps[plot_idx])
# plt.xticks([])
# plt.xlim([0,T_plot])
# plt.title("spike counts")

# poisson
# def poiss_log_like(w, Xmat, Ysps, nlfun):
#     rate = nlfun(Xmat@w)[0]
#     ll_all_data = Ysps * np.log(rate) - rate - gammaln(Ysps+1.0) # array with size number of data points
#     ll = np.sum(ll_all_data) # sum log likelihood of each data point
#     return -1.0 * ll 
# poiss_grad_func = grad(poiss_log_like)
# res_poiss = minimize(poiss_log_like, args=(Xmat, Ysps, nlfun), x0=np.zeros_like(w), jac=poiss_grad_func)
# w_mle_poiss = res_poiss.x

# plt.figure()
# plt.plot(w, label="true params")
# plt.plot(w_mle_poiss, label="fit params")
# plt.legend()
# plt.title("parameters")