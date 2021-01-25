import autograd.numpy as np
import autograd.numpy.random as npr
npr.seed(123)

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import seaborn as sns
sns.set_style("white")
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

from ssm import LDS
from ssm.util import random_rotation

# Set the parameters of the HMM
T = 1000   # number of time bins
D = 2      # number of latent dimensions
N = 40     # number of observed dimensions
bin_size = 0.01
link_func = "softplus"
# Make an LDS with somewhat interesting dynamics parameters
true_lds = LDS(N, D, emissions="calcium", emission_kwargs={"bin_size":bin_size,"link":link_func})
A0 = .99 * random_rotation(D, theta=np.pi/40)
# S = (1 + 3 * npr.rand(D))
S = np.arange(1, D+1)
R = np.linalg.svd(npr.randn(D, D))[0] * S
A = R.dot(A0).dot(np.linalg.inv(R))
b =  np.zeros(D)
true_lds.dynamics.As[0] = A
true_lds.dynamics.bs[0] = b
true_lds.dynamics.Sigmas = true_lds.dynamics.Sigmas / np.max(true_lds.dynamics.Sigmas[0]) * 0.5
# true_lds.dynamics.Sigmas = true_lds.dynamics.Sigmas 
# true_lds.emissions.ds[0] = np.clip(npr.randn(N), -10, 0.1)
true_lds.emissions.ds[0] = 10.0 + 3.0 * npr.randn(N)
true_lds.emissions.As[0] = np.clip(0.85 + 0.05 * npr.randn(N), 0.8, 0.95)
true_lds.emissions.betas[0] = 1.0 * np.ones(N)

# set noise on correct scale
true_lds.emissions.inv_etas[0] = np.log(1e-2 * np.ones(N))

x, y = true_lds.sample(T)
smooth_y = true_lds.smooth(x, y)

plt.ion()
plt.figure()
plt.plot(x)

plt.figure()
for n in range(N):
    plt.plot(y[:, n] + 10 * n, '-k')
    # plt.plot(smooth_y[:, n] + 4 * n, 'r--')

# n=n+1
# plt.figure()
# plt.plot(y[:,n])
u = np.zeros((T,0))
mask = np.ones_like(y)
tag = None
lls = true_lds.emissions.log_likelihoods(y, u, mask, tag, x)

lambdas = true_lds.emissions.mean(true_lds.emissions.forward(x, u, tag))[:,0,:]
# plt.figure()
# for n in range(N):
#     plt.axhline(4*n, color='k')
#     plt.plot(lambdas[:,n] / bin_size + 4 * n, '-b')

# hess = true_lds.emissions.hessian_log_emissions_prob(y, u, mask, tag, x)
# hessian_analytical = block_diag(*hess)

# hess_autograd = hessian(lambda x : true_lds.emissions.log_likelihoods(y, u, mask, tag, x))
# hessian_autograd = hess_autograd(x).reshape((T*D), (T*D))
# print("Norm of difference: ", np.linalg.norm(hessian_analytical - hessian_autograd))

# hess = true_lds.emissions.hessian_log_emissions_prob(y, u, mask, tag, x)

# test log likelihoods
# from autograd.scipy.special import gammaln, logsumexp

# def poisson_1D(y, rate):
#     return np.exp(y * np.log(rate) - rate - gammaln(y+1))

# def gaussian_1D(y, mu, sig2):
#     return 1.0 / np.sqrt(2.0 * np.pi * sig2) * np.exp(-0.5 * (y - mu)**2 / sig2)

# def log_poisson_1D(y, rate):
#     return y * np.log(rate) - rate - gammaln(y+1)

# def log_gaussian_1D(y, mu, sig2):
#     return -0.5 * np.log(2.0 * np.pi * sig2) -0.5 * (y - mu)**2 / sig2


# alphas = true_lds.emissions.As[0]
# betas = true_lds.emissions.betas[0]
# sig2s = np.exp(true_lds.emissions.inv_etas[0])
# C = true_lds.emissions.Cs[0]
# d = true_lds.emissions.ds[0]
# ll_test = 0.0
# dt = true_lds.emissions.bin_size
# S = 10
# rates = true_lds.emissions.mean(true_lds.emissions.forward(x, u, tag))[:,0,:]
# for t in range(T):
#     print(t)
#     # rates = np.log1p(np.exp(C@x[t] + d)) * dt
#     if t == 0:
#         ytm1 = np.zeros(N)
#     else:
#         ytm1 = y[t-1]

#     for n in range(N):

#         pnt = 0.0
#         s = np.arange(0, S, 1)
#         mu = alphas[n] * ytm1[n] + betas[n] * s
#         rate = rates[t,n]
#         # pnt = np.sum(gaussian_1D(y[t,n], mu, sig2s[n]) * poisson_1D(s, rates[n]))
#         pnt = logsumexp(log_gaussian_1D(y[t,n], mu, sig2s[n]) + log_poisson_1D(s, rate) )
#         # pnt = np.sum(gaussian_1D(y[t,n], mu, sig2s[n]) * poisson_1D(s, rate))

#         # for s in range(S):

#         #     mu = alphas[n] * ytm1[n] + betas[n] * s
#         #     ps = gaussian_1D(y[t,n], mu, sig2s[n]) * poisson_1D(s, rates[n])
#         #     pnt += ps

#         ll_test += pnt

test_As = np.zeros_like(true_lds.emissions.As[0])
test_As = true_lds.emissions.As[0] + 0.0
test_etas = np.zeros_like(true_lds.emissions.inv_etas[0])
for d in range(N):
    yd = y[:,d]
    # test_As[d] = 1.0 / np.sum(yd[:-1] **2 ) * np.dot(yd[:-1], yd[1:]) * 1.0 # deflate because not taking into account spikes
    sqerr = np.sum( ( yd[1:] - test_As[d] * yd[:-1] )**2 ) / (T-1)
    test_etas[d] = sqerr 

# nlfun = lambda x : softplus_stable(x, bias=0.0, dt=bin_size)

# Xmat = np.ones((T, 1))
# def _obj(_params):
#     w = _params[0]
#     hyperparams = _params[1]
#     return nll_GLM_GanmorCalciumAR1(w, Xmat, y[:, 0], hyperparams, nlfun)

# w = np.ones((1, 1))
# hyperparams = [np.log(1.0), np.log(1.0), np.log(0.1)]
# _params = [w, hyperparams]

# from autograd import grad
# from autograd.misc import flatten
# params, unflatten = flatten(_params)
# obj = lambda params : _obj(unflatten(params))
# grad_func = grad(obj)

# from scipy.optimize import minimize
# # params_init = 0.1 * npr.randn(params.shape[0])
# # params_init = flatten
# # w_init = npr.randn(D)
# w_init = np.array([20.0])
# hyperparams_init = [np.log(10.0), np.log(1.0), np.log(1.0)]
# params_init = np.concatenate((w_init, hyperparams_init))
# res = minimize(obj, x0=params_init, jac=grad_func, options={"disp": True, "maxiter": 1000})
# params_mle = res.x

# fit!
lds = LDS(N, D, emissions="calcium", emission_kwargs={"bin_size":bin_size,"link":"softplus"})
# lds.emissions.As[0] = true_lds.emissions.As[0]
# lds.emissions.betas[0] = true_lds.emissions.betas[0]
# lds.emissions.inv_etas[0] = true_lds.emissions.inv_etas[0]
lds.emissions.As[0] = test_As
# lds.emissions.betas[0] = true_lds.emissions.betas[0]
lds.emissions.inv_etas[0] = np.log(test_etas)
lds.initialize(y)
lds.emissions.ds = true_lds.emissions.ds + 3.0 * npr.randn(N)
q_elbos, q = lds.fit(y, method="laplace_em", variational_posterior="structured_meanfield", 
                            num_iters=25, initialize=False,
                            continuous_optimizer="lbfgs")

# fit gaussian
lds_gauss = LDS(N, D, emissions="gaussian")
lds_gauss.initialize(y)
q_elbos_gauss, q_gauss = lds_gauss.fit(y, method="laplace_em", variational_posterior="structured_meanfield", 
                            num_iters=25, initialize=False,
                            continuous_optimizer="newton")

# q_mf_x = q_mf.mean[0]
q_x = q.mean_continuous_states[0]
from sklearn.linear_model import LinearRegression
lr = LinearRegression(fit_intercept=False)
lr.fit(q_x, x)
q_x_trans = lr.predict(q_x)

q_x_gauss = q_gauss.mean_continuous_states[0]
from sklearn.linear_model import LinearRegression
lr_gauss = LinearRegression(fit_intercept=False)
lr_gauss.fit(q_x_gauss, x)
q_x_trans_gauss = lr.predict(q_x_gauss)



plt.figure()
plt.plot(q_elbos[1:], color=colors[0], label="calcium")
plt.plot(q_elbos_gauss[1:], color=colors[1], label="gauss")
plt.legend()

plt.figure(figsize=(8,4))
# plt.subplot(211)
# plt.plot(x + 4 * np.arange(D), '-k')
for d in range(D):
    plt.plot(x[:,d] + 4 * d, '-k', label="true" if d==0 else None)
    plt.plot(q_x_trans[:,d] + 4 * d, '-', color=colors[0], label="calcium" if d==0 else None)
    plt.plot(q_x_trans_gauss[:,d] + 4 * d, '-', color=colors[1], label="gauss" if d==0 else None)
plt.ylabel("$x$")
plt.xlim((0,1000))
plt.legend()

smooth_y = lds.smooth(q_x, y)
smooth_y_gauss = lds_gauss.smooth(q_x_gauss, y)

# plt.subplot(212)
plt.figure()
for n in range(N): 
    plt.plot(y[:, n] + 4 * n, '-k', label="true" if d==0 else None)
    plt.plot(smooth_y[:, n] + 4 * n, '--', color=colors[0], label="MF" if d==0 else None)
    plt.plot(smooth_y_gauss[:, n] + 4 * n, '--', color=colors[1], label="MF" if d==0 else None)
plt.xlim((0,T))
plt.ylabel("$y$")
plt.legend()

inf_lambdas = lds.emissions.mean(lds.emissions.forward(q_x, u, tag))[:,0,:]
plt.figure()
for n in range(N): 
    plt.plot(lambdas[:, n] + 0.5 * n, '-k', label="true" if d==0 else None)
    plt.plot(inf_lambdas[:, n] + 0.5 * n, '--', color=colors[0], label="calcium" if d==0 else None)
    plt.plot(smooth_y_gauss[:, n] + 0.5 * n, '--', color=colors[1], label="MF" if d==0 else None)
plt.xlim((0,T))
plt.ylabel("$y$")
plt.legend()

