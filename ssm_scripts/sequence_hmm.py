import autograd.numpy as np
import autograd.numpy.random as npr
# npr.seed(123)
npr.seed(523)

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
plt.ion()

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

from ssm import HMM
from ssm.util import find_permutation, one_hot

T = 2000   # number of time bins
D = 25     # number of observed neurons
K = 5      # number of states

# Make an LDS with somewhat interesting dynamics parameters
true_hmm = HMM(K, D, observations="poisson") 
P = np.eye(K) + 0.2 * np.diag(np.ones(K-1), k=1) + 1e-5 * np.ones((K,K))
P[-1,0] = 0.2
true_hmm.transitions.log_Ps = np.log(P)

log_lambdas = np.log(0.01 * np.ones((K, D)))
for k in range(K):
    log_lambdas[k,k*K:(k+1)*K] = np.log(0.2)
true_hmm.observations.log_lambdas = log_lambdas

z, y = true_hmm.sample(T)
z_test, y_test = true_hmm.sample(T)

T_plot=500
plt.figure()
plt.subplot(211)
plt.imshow(z[None,:], aspect="auto")
plt.xlim([0,T_plot])
plt.subplot(212)
# plt.plot(y)
for n in range(D):
    plt.eventplot(np.where(y[:,n]>0)[0]+1, linelengths=0.5, lineoffsets=D-n,color='k')
plt.xlim([0,T_plot])

plt.figure()
plt.imshow(true_hmm.transitions.transition_matrix, vmin=0.0, vmax=1.0, aspect="auto")

# run for multiple models
N = 10
max_ll = -np.inf
for n in range(N):
    test_hmm_temp = HMM(K, D, observations="poisson") 
    poiss_lls_temp = test_hmm_temp.fit(y, num_iters=20)
    if poiss_lls_temp[-1] > max_ll:
        max_ll = poiss_lls_temp[-1]
        poiss_lls = poiss_lls_temp 
        test_hmm = test_hmm_temp
# test_hmm = HMM(K, D, observations="poisson") 
# poiss_lls = test_hmm.fit(y, num_iters=20)
test_hmm.permute(find_permutation(z, test_hmm.most_likely_states(y)))
smoothed_z = test_hmm.most_likely_states(y)

plt.figure()
plt.subplot(211)
plt.imshow(np.row_stack((z, smoothed_z)), aspect="auto")
plt.xlim([0,T_plot])
plt.subplot(212)
# plt.plot(y)
for n in range(D):
    plt.eventplot(np.where(y[:,n]>0)[0]+1, linelengths=0.5, lineoffsets=D-n,color='k')
plt.xlim([0,T_plot])


As = np.clip(0.8 + 0.1 * npr.randn(D), 0.6, 0.95)
betas = 1.0 * np.ones(D)
inv_etas = np.log(1e-2 * np.ones(D))
etas = np.exp(inv_etas)
mus = np.zeros_like(y) # start with zero mean
y_ca = np.zeros((T, D))
y_ca_test = np.zeros_like(y_test)
y_ca[0] = mus[0, :] + np.sqrt(etas) * npr.randn(D) + betas * y[0, :]
y_ca_test[0] = mus[0, :] + np.sqrt(etas) * npr.randn(D) + betas * y_test[0, :]
for t in range(1, T):
    y_ca[t] = mus[t, :] + As * y_ca[t-1] + betas * y[t, :] + np.sqrt(etas) * npr.randn(D)
    y_ca_test[t] = mus[t, :] + As * y_ca_test[t-1] + betas * y_test[t, :] + np.sqrt(etas) * npr.randn(D)

# add per time point noise afterwards (not part of Ganmor generative model)
y_ca = y_ca + 0.2 * npr.randn(*y_ca.shape)

# T_plot
plt.figure()
plt.subplot(211)
# plt.plot(y)
for n in range(D):
    plt.eventplot(np.where(y[:,n]>0)[0]+1, linelengths=0.5, lineoffsets=D-n,color='k')
plt.xlim([0,T_plot])
plt.subplot(212)
for n in range(D):
    plt.plot(y_ca[:, n] + 4 * (D-n), '-k')
# y_dff = (y_ca - np.min(y_ca,axis=0)) / np.max(y_ca, axis=0)
# plt.imshow( y_dff.T, aspect="auto")
plt.xlim([0,T_plot])

N = 10
max_ll = -np.inf
for n in range(N):
    test_hmm_temp = HMM(K, D, observations="gaussian") 
    gauss_lls_temp = test_hmm_temp.fit(y_ca, num_iters=20)
    if gauss_lls_temp[-1] > max_ll:
        max_ll = gauss_lls_temp[-1]
        gauss_lls = gauss_lls_temp 
        test_hmm_gauss = test_hmm_temp
# test_hmm_gauss = HMM(K, D, observations="gaussian") 
# gauss_lls = test_hmm_gauss.fit(y_ca, num_iters=20)
test_hmm_gauss.permute(find_permutation(z, test_hmm_gauss.most_likely_states(y_ca)))
smoothed_z_gauss = test_hmm_gauss.most_likely_states(y_ca)
smoothed_y_gauss = test_hmm_gauss.smooth(y_ca)

test_hmm_ca = HMM(K, D, observations="calcium") 
# inv_As = np.log(As)
# inv_betas = np.log(betas)
# inv_As = np.log(np.clip(0.8 + 0.1 * npr.randn(D), 0.6, 0.95))
test_As = np.zeros_like(As)
test_etas = np.zeros_like(etas)
for d in range(D):
    yd = y_ca[:,d]
    test_As[d] = 1.0 / np.sum(yd[:-1] **2 ) * np.dot(yd[:-1], yd[1:]) * 1.0 # deflate because not taking into account spikes
    sqerr = np.sum( ( yd[1:] - test_As[d] * yd[:-1] )**2 ) / (T-1)
    test_etas[d] = sqerr 

inv_betas = np.log(1.0 * np.ones(D) + 0.2 * npr.randn(D))
inv_etas = np.log(test_etas)
inv_As = np.log(test_As)
test_hmm_ca.observations.params = \
    (test_hmm_ca.observations.log_lambdas, inv_As, inv_betas, inv_etas) 
# test_hmm_ca.observations.inv_As = inv_As
# test_hmm_ca.observations.inv_betas = inv_betas
# test_hmm_ca.observations.inv_etas = inv_etas
ca_lls = test_hmm_ca.fit(y_ca, num_iters=20, observations_mstep_kwargs={"num_iters":100})
test_hmm_ca.permute(find_permutation(z, test_hmm_ca.most_likely_states(y_ca)))
smoothed_z_ca = test_hmm_ca.most_likely_states(y_ca)
smoothed_y_ca = test_hmm_ca.smooth(y_ca)

np.mean(y_ca[1:] - y_ca[:-1], axis=0)

plt.figure()
plt.imshow(np.row_stack((z, smoothed_z, smoothed_z_gauss, smoothed_z_ca)), aspect="auto")
plt.yticks([0,1,2,3],["true","poisson","gaussian","calcium"])
plt.ylim([3.5,-0.5])
plt.xlim([0, T_plot])

plt.figure()
plt.plot(z, 'k', label="True")
plt.plot(smoothed_z, 'r', label="Poisson")
plt.plot(smoothed_z_gauss, 'b', label="Gaussian")
plt.plot(smoothed_z_ca, 'g', label="Calcium")
plt.legend()

plt.figure()
for n in range(D):
    plt.plot(y_ca[:, n] + 4 * (D-n), '-k', label="True" if n == 0 else None)
    plt.plot(smoothed_y_gauss[:, n] + 4 * (D-n), '-b', label="Gaussian" if n == 0 else None)
    plt.plot(smoothed_y_ca[:, n] + 4 * (D-n), '-g', label="Calcium" if n == 0 else None)
plt.xlim([0, T_plot])

# test data
y_ca_test = y_ca_test + 0.2 * npr.randn(*y_ca_test.shape)
poiss_test_ll = test_hmm.log_likelihood(y_test)
ca_test_ll = test_hmm_ca.log_likelihood(y_ca_test)
gauss_test_ll = test_hmm_gauss.log_likelihood(y_ca_test)

smoothed_z_test = test_hmm.most_likely_states(y_test)
smoothed_z_ca_test = test_hmm_ca.most_likely_states(y_ca_test)
smoothed_z_gauss_test = test_hmm_gauss.most_likely_states(y_ca_test)

plt.figure()
plt.imshow(np.row_stack((z_test, smoothed_z_test, smoothed_z_gauss_test, smoothed_z_ca_test)), aspect="auto")
plt.xlim([0, T_plot])
plt.yticks([0,1,2,3],["true","poisson","gaussian","calcium"])
plt.ylim([3.5,-0.5])

# posterior expectations
poiss_expectations = test_hmm.expected_states(y)[0]
gauss_expectations = test_hmm_gauss.expected_states(y_ca)[0]
ganmor_expectations = test_hmm_ca.expected_states(y_ca)[0]

plt.figure()
plt.subplot(311)
plt.imshow(one_hot(z, K=K).T, aspect="auto", vmin=0.0, vmax=1.0, cmap="Greys")
plt.title("true")
plt.xlim([0, T_plot])
plt.subplot(312)
plt.imshow(gauss_expectations.T, aspect="auto", vmin=0.0, vmax=1.0, cmap="Greys")
plt.xlim([0, T_plot])
plt.title("gaussian")
plt.subplot(313)
plt.imshow(ganmor_expectations.T, aspect="auto", vmin=0.0, vmax=1.0, cmap="Greys")
plt.xlim([0, T_plot])
plt.title("ganmor")

plt.figure()
plt.subplot(221)
plt.imshow(true_hmm.transitions.transition_matrix, aspect="auto", vmin=0.0, vmax=1.0)
plt.title("true")
plt.subplot(222)
plt.imshow(test_hmm.transitions.transition_matrix, aspect="auto", vmin=0.0, vmax=1.0)
plt.title("poisson")
plt.subplot(223)
plt.imshow(test_hmm_gauss.transitions.transition_matrix, aspect="auto", vmin=0.0, vmax=1.0)
plt.title("gaussian")
plt.subplot(224)
plt.imshow(test_hmm_ca.transitions.transition_matrix, aspect="auto", vmin=0.0, vmax=1.0)
plt.title("ganmor")

# posterior expectations
poiss_expectations = test_hmm.expected_states(y_test)[0]
gauss_expectations = test_hmm_gauss.expected_states(y_ca_test)[0]
ganmor_expectations = test_hmm_ca.expected_states(y_ca_test)[0]

plt.figure()
plt.subplot(311)
plt.imshow(one_hot(z_test, K=K).T, aspect="auto", vmin=0.0, vmax=1.0, cmap="Greys")
plt.title("true")
plt.xlim([0, T_plot])
plt.subplot(312)
plt.imshow(gauss_expectations.T, aspect="auto", vmin=0.0, vmax=1.0, cmap="Greys")
plt.xlim([0, T_plot])
plt.title("gaussian")
plt.subplot(313)
plt.imshow(ganmor_expectations.T, aspect="auto", vmin=0.0, vmax=1.0, cmap="Greys")
plt.xlim([0, T_plot])
plt.title("ganmor")