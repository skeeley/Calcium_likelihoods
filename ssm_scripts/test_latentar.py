import autograd.numpy as np
import autograd.numpy.random as npr

import ssm 
import latentar
from latentar import LatentCalcium

import matplotlib.pyplot as plt
plt.ion()

import seaborn as sns
sns.set_style("ticks")
sns.set_context("talk")
from ssm.util import softplus
from ssm.primitives import blocks_to_full, convert_lds_to_block_tridiag

D = 1
K = 8
N = 1
M = 5
bin_size = 0.05
model = LatentCalcium(N, K, D, M=M,
                        transition_kwargs={"bin_size":bin_size})


T = 10000
u = npr.randn(T, M) # input
firing_rates = np.log1p(np.exp(np.dot(u, model.transitions.Ws)))
print("Max firing rate: ", np.max(firing_rates))

# simulate data
z, x, y = model.sample(T, input=u)

x_range = [0,500]

# fit data
slds = LatentCalcium(N, K, D, M=M,
                            transition_kwargs={"bin_size":bin_size})
w_init = np.linalg.inv(u.T@u)@u.T@y 
slds.transitions.Ws = w_init.reshape((M,))

# slds.initialize(y, inputs=u)
# slds.params = model.params
q_lem_elbos, q_laplace_em = slds.fit(y, inputs=u, method="laplace_em",
                               variational_posterior="structured_meanfield",
                               initialize=False, num_iters=25)
q_lem_Ez, q_lem_x = q_laplace_em.mean[0]
q_lem_y = slds.smooth(q_lem_x, y)
q_lem_z = slds.most_likely_states(q_lem_x, y)

plt.figure()
plt.plot(model.transitions.Ws, label="true")
plt.plot(slds.transitions.Ws, label="inferred")
plt.legend()
plt.title("True and Inferred GLM parameters")

true_rates = np.log1p(np.exp(np.dot(u, model.transitions.Ws))) * bin_size
inferred_rates = np.log1p(np.exp(np.dot(u, slds.transitions.Ws))) * bin_size

plt.figure()
plt.subplot(411)
plt.plot(true_rates)
plt.plot(inferred_rates, '--')
plt.xlim(x_range)
plt.title("True and Inferred Firing Rates")
plt.subplot(412)
plt.plot(z)
plt.plot(q_lem_z, '--')
plt.xlim(x_range)
plt.title("True and Inferred Spike Counts")
plt.subplot(413)
plt.plot(x)
plt.plot(q_lem_x, '--')
plt.xlim(x_range)
plt.title("True and Posterior Mean Calcium Trace")
plt.subplot(414)
plt.plot(y, label="true")
plt.plot(q_lem_y,'--', label="inferred")
plt.xlim(x_range)
plt.legend()
plt.title("True and Smoothed Fluorescence Trace")
