import jax
import jax.numpy as jnp
import jax.random as jr
from jax import jit, grad, vmap
from jax.scipy.special import gammaln, logsumexp
from jax import value_and_grad 

import numpy as np 
import copy 

import optax 

from dynamax.hidden_markov_model import inference
import tensorflow_probability.substrates.jax.distributions as tfd

# npr.seed(523)

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
plt.ion()

import glob 
from tqdm import tqdm 

# Define Calcium LL 
# helper functions
def log_poisson_1D(y, log_rate):
    return y * log_rate - jnp.exp(log_rate) - gammaln(y+1)

def log_gaussian_1D(y, mu, sig2):
    return -0.5 * jnp.log(2.0 * jnp.pi * sig2) -0.5 * (y - mu)**2 / sig2

def calcium_log_likelihood(Y, log_rates, ca_params):
  """Compute the log likelihood under calcium observation model.
  Write this for a single time series x, then vmap across all neurons / time series. 
  """

  # unpack hyperparams
  alpha = jnp.exp(ca_params["log_alpha"]) # ar parameter
  beta = jnp.exp(ca_params["log_beta"]) # spike increase
  sig2 = jnp.exp(ca_params["log_sigma"]) # ar variance

  # compute autoregressive component of mean
  pad = jnp.zeros((1, 1, alpha.shape[0]))
  mus = jnp.concatenate((pad, alpha[None, None, :] * Y[:-1, None, :])) 

  # initialize spike count range
  # TODO: put S in function
  S = 10
  s = jnp.arange(0,S,1)

  # add spike components to autoregressive mean for each number of spikes
  mus = mus[:,:,:,None] + beta[None,None,:,None] * s

  # compute log likelihood, marginalizing out the spikes
  outg = log_gaussian_1D(Y[:,None,:,None], mus, sig2[None,None,:,None])
  outp = log_poisson_1D(s[None,None,None,:], log_rates[None,:,:,None])
  lls = logsumexp(outg + outp, axis=3) # marginalize over unseen spikes here
  
  return jnp.sum(lls, axis=2) # sum over neurons

def construct_ca_params(n, alpha, beta, sigma):
  return {'log_alpha' : jnp.log(alpha) * jnp.ones((n,)),
          'log_beta' : jnp.log(beta) * jnp.ones((n,)),
          'log_sigma' : jnp.log(sigma) * jnp.ones((n,))}

# Define HMM Marginal Likelihood
def marginal_nll(params, emissions):
    log_init_probs, log_transition_matrix, emission_params = params 
    lls = calcium_log_likelihood(emissions, 
            emission_params['log_rates'], emission_params['ca_params'])
    log_transition_matrix -= logsumexp(log_transition_matrix, -1, keepdims=True)
    transition_matrix = jnp.exp(log_transition_matrix)
    post = inference.hmm_filter(jnp.exp(log_init_probs), transition_matrix, lls)
    marg_ll = post.marginal_loglik
    return -1.0 * marg_ll
vmap_marginal_nll = vmap(marginal_nll, in_axes=(None, 0))

def expected_nll(params, emissions, state_probs):
    log_init_probs, log_transition_matrix, emission_params = params 
    lls = calcium_log_likelihood(emissions, 
            emission_params['log_rates'], emission_params['ca_params'])
    return -1.0 * jnp.sum(lls * state_probs)
vmap_expected_nll = vmap(expected_nll, in_axes=(None, 0, 0))

def init_loss_fn(params, batch_emissions, batch_state_probs):
    return jnp.mean(vmap_expected_nll(params, batch_emissions, batch_state_probs))
init_loss_grad_fn = jit(value_and_grad(init_loss_fn))

def loss_fn(params, batch_emissions):
    return jnp.mean(vmap_marginal_nll(params, batch_emissions))
loss_grad_fn = jit(value_and_grad(loss_fn))

@jit
def init_sgd_step(obs, post, params, opt_state):
    loss_val, grads = init_loss_grad_fn(params, obs, post)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_val

@jit
def sgd_step(obs, params, opt_state):
    loss_val, grads = loss_grad_fn(params, obs)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_val


from scipy.io import loadmat

# load the data and grab the neural time series
d = np.load("/Users/dzoltowski/Downloads/dandi_odor_dataset.npz", allow_pickle=True)
data = d['data']
odors = jnp.array([di['odor']-1 for di in data])
y_ca_all = jnp.array([di['dFF0'].T for di in data])
# neuron 137 appears faulty
y_ca_all = np.delete(y_ca_all, 137, 2)
Num_tr, T, D = y_ca_all.shape 

# key = jr.PRNGKey(123)
# perm_idx = jr.permutation(key, jnp.arange(B))
# train_idx = perm_idx[:]
train_idx = jnp.concatenate((jnp.arange(30), jnp.arange(50, Num_tr)))
# skip trial 30! something wrong with it
test_idx = jnp.arange(31, 50)

y_ca = y_ca_all[train_idx]
B = y_ca.shape[0]
y_ca_test = y_ca_all[test_idx]

# fit with 13 state model
K = 13
y_ca_z = jnp.hstack((jnp.zeros((60, 1, 284)), y_ca))
diffs = y_ca_z[:, 1:, :] - y_ca_z[:, :-1, :]
from sklearn.cluster import KMeans
km = KMeans(n_clusters=K)
km.fit(np.vstack(diffs))
labels = jnp.array([km.predict(diff) for diff in diffs])

def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]
post_states = jnp.array([indices_to_one_hot(_label, K) for _label in labels])

post_states_train = post_states[train_idx]
post_states_test = post_states[test_idx]

def a_loss(dif, alpha):
  S = np.arange(3)
  influx = S * alpha
  loss = np.mean(np.min((dif[:, None] - influx[None, :])**2, axis=1))
  return loss 

init_As = []
init_betas = []
init_vars = []

for d in range(D):
    if np.mod(d, 10) == 0:
        print(d)

    ca_data = y_ca[:, :, d]

    tar = ca_data[:,1:]
    X = ca_data[:,:-1]
    tar = jnp.concatenate(tar)
    X = jnp.concatenate(X)

    # grab points where tar below X -> no spikes
    tar_keep = tar[tar<X]
    X_keep = X[tar<X]

    test_A = 1.0 / jnp.sum(X_keep**2) * jnp.dot(X_keep, tar_keep)
    dif = jnp.hstack((ca_data[:,1:] - test_A * ca_data[:,:-1]))
    init_vars.append(jnp.mean(dif**2))

    beta = 1.0

    init_As.append(test_A)
    init_betas.append(beta)

init_As = np.array(init_As)
init_betas = np.array(init_betas)
init_vars = np.array(init_vars)

# initialize params
key = jr.PRNGKey(13)
log_init_probs = jnp.log(1.0 / K * jnp.ones((K,)))
transition_matrix = 0.95 * jnp.eye(K) + 0.05 / K * jnp.ones((K, K)) 
log_transition_matrix = jnp.log(transition_matrix)
log_transition_matrix -= logsumexp(log_transition_matrix, -1, keepdims=True)
key, skey = jr.split(key)
emission_params = {}
emission_params['log_rates'] = jr.normal(skey, shape=(K, D))
ca_params = construct_ca_params(D, 0.9, 1.0, 0.1)
ca_params['log_alpha'] = jnp.log(init_As)
ca_params['log_beta'] = jnp.log(init_betas)
ca_params['log_sigma'] = jnp.log(init_vars)
emission_params['ca_params'] = ca_params
params = [log_init_probs, log_transition_matrix, emission_params]

# smart initialization
# fit GLM, with lam0 = rate on first 10 time bins, lam_i = rate on final 20 time bins for each odor 
optimizer = optax.GradientTransformation=optax.adam(1e-2)
shuffle = False
opt_state = optimizer.init(params)

init_losses = []
key = jr.PRNGKey(13)
batch_size = 16

# initialize with states given by odors
for i in tqdm(range(2500)):

    # sample data
    key, skey = jr.split(key)
    rand_idx = jr.randint(skey, minval=0, maxval=B, shape=(batch_size,))
    y_in = y_ca[rand_idx]
    states_in = post_states_train[rand_idx]

    # optimize
    params, opt_state, loss_val = init_sgd_step(y_in, states_in, params, opt_state)
    init_losses.append(loss_val)

plt.figure()
plt.plot(init_losses)

params_init = copy.deepcopy(params)
opt_state_init = copy.deepcopy(opt_state)

optimizer = optax.GradientTransformation=optax.adam(1e-3)
shuffle = False
opt_state = optimizer.init(params)

losses = []
# continuoue fitting, now without odors (?)
for i in tqdm(range(2500)):

    # sample data
    key, skey = jr.split(key)
    rand_idx = jr.randint(skey, minval=0, maxval=B, shape=(batch_size,))
    y_in = y_ca[rand_idx]

    # optimize
    params, opt_state, loss_val = sgd_step(y_in, params, opt_state)
    losses.append(loss_val)

plt.figure()
plt.plot(losses)

# smooth trial
tr = 0
tr+=1
print(odors[tr])
log_init_probs, log_transition_matrix, emission_params = params 
lls = calcium_log_likelihood(y_ca[tr], 
        emission_params['log_rates'], emission_params['ca_params'])
log_transition_matrix -= logsumexp(log_transition_matrix, -1, keepdims=True)
transition_matrix = jnp.exp(log_transition_matrix)
post = inference.hmm_smoother(jnp.exp(log_init_probs), transition_matrix, lls)
smoothed_z_ca = jnp.argmax(post.smoothed_probs, axis=1)

# smoothed rates
plt.figure()
plt.plot(smoothed_z_ca)

t_trial = data[0]['t'][45:55]
plt.figure()
plt.imshow(y_ca[tr].T, aspect="auto", extent=(0, 30, 0, D), interpolation=None)
plt.title(odors[tr])
plt.xlabel("time (s)")
plt.ylabel("neuron")
plt.colorbar()
plt.axvline(10.0, color='r', linestyle='--', linewidth=0.5)
plt.axvline(11.0, color='r', linestyle='--', linewidth=0.5)

# test loss
loss_fn(params, y_ca_test)

# smooth all trials
most_likely_states_train = []
for tr in range(y_ca.shape[0]): 
    log_init_probs, log_transition_matrix, emission_params = params 
    lls = calcium_log_likelihood(y_ca[tr], 
            emission_params['log_rates'], emission_params['ca_params'])
    log_transition_matrix -= logsumexp(log_transition_matrix, -1, keepdims=True)
    transition_matrix = jnp.exp(log_transition_matrix)
    post = inference.hmm_smoother(jnp.exp(log_init_probs), transition_matrix, lls)
    smoothed_z_ca = jnp.argmax(post.smoothed_probs, axis=1)
    most_likely_states_train.append(smoothed_z_ca)
most_likely_states_train = jnp.array(most_likely_states_train)

most_likely_states_test = []
for tr in range(y_ca_test.shape[0]): 
    log_init_probs, log_transition_matrix, emission_params = params 
    lls = calcium_log_likelihood(y_ca_test[tr], 
            emission_params['log_rates'], emission_params['ca_params'])
    log_transition_matrix -= logsumexp(log_transition_matrix, -1, keepdims=True)
    transition_matrix = jnp.exp(log_transition_matrix)
    post = inference.hmm_smoother(jnp.exp(log_init_probs), transition_matrix, lls)
    smoothed_z_ca = jnp.argmax(post.smoothed_probs, axis=1)
    most_likely_states_test.append(smoothed_z_ca)
most_likely_states_test = jnp.array(most_likely_states_test)

# compute LLs on heldout trials given map states
# test_state_probs = []
# for ml_states in most_likely_states_test:
#     test_state_prob = np.zeros((T, K))
#     # test_state_prob[:,ml_states]=1.0
#     for t in range(T):
#         test_state_prob[t,ml_states[t]]=1.0
#     test_state_probs.append(test_state_prob)
# test_state_probs = jnp.array(test_state_probs)
# # ll5 = vmap_expected_nll(params, y_ca_test, test_state_probs)
# ll2 = vmap_expected_nll(params, y_ca_test, test_state_probs)

odors_train = odors[train_idx]
odors_test = odors[test_idx]

# plt.figure()
f, (a0, a1) = plt.subplots(2, 1, height_ratios=[3, 1])
a0.imshow(most_likely_states_train[jnp.argsort(odors_train)], aspect="auto", cmap="tab20", extent=(0, 30, 60, 0))
a0.set_ylabel("train trial")
a0.set_xlabel("time (s)")
a1.imshow(most_likely_states_test[jnp.argsort(odors_test)], aspect="auto", cmap="tab20", extent=(0, 30, 20, 0))
a1.set_xlabel("time (s)")
a1.set_ylabel("test trial")

vals, counts = jnp.unique(most_likely_states_test[:,46:55], return_counts=True)
odor_state = vals[jnp.argmax(counts)]
# odor_state = 4

bool_odor_state = most_likely_states_train==odor_state
mean_odor_state = jnp.mean(bool_odor_state, axis=0)
std_odor_state = jnp.std(bool_odor_state, axis=0)
t_trial = data[0]['t']

plt.figure()
plt.fill_between(t_trial, mean_odor_state-std_odor_state, mean_odor_state+std_odor_state, color=[0.3, 0.3, 1.0], alpha=0.5)
plt.plot(t_trial, mean_odor_state, color=[0.3, 0.3, 1.0])
plt.axvspan(10, 11, ymin=0.1, ymax=0.9, alpha=0.5, color='red')
plt.xlabel("time (s)")
plt.ylabel("proportion of trials in odor onset state")
plt.title("Calcium HMM")
params = (d['params0'], d['params1'], d['params2'].item())

plt.figure()
plt.axvspan(10, 11, ymin=0.0, ymax=1.0, alpha=0.5, color='red')
plt.plot(t_trial, mean_odor_state, color=[0.3, 0.3, 1.0])
plt.xlabel("time (s)")
plt.ylabel("frac. of trials in odor onset state")
plt.title("Calcium HMM")

plt.figure()
plt.axvspan(10, 11, ymin=0.0, ymax=1.0, alpha=0.5, color='green', label="odor")
plt.plot(t_trial, mean_odor_state, color=[0.3, 0.3, 1.0], label='calcium')
plt.plot(t_trial, gauss_mean_odor_state, color=[1.0, 0.3, 0.3], label='gaussian')
plt.xlabel("time (s)")
plt.ylabel("fraction of trials in odor onset state")

half_val = np.max(mean_odor_state) *0.2
peak_idx = np.argmax(mean_odor_state)
half1_idx = [i for i in range(mean_odor_state.shape[0]-1) if (mean_odor_state[i]<half_val) & (mean_odor_state[i+1]>=half_val)][0]
# half1_idx = half1_idx +0.5 # crossing happened in between these two bins, right after half1 idx
half2_idx = [i for i in range(mean_odor_state.shape[0]-1) if (mean_odor_state[i]>half_val) & (mean_odor_state[i+1]<=half_val)][0]
# half2_idx = half2_idx +0.5 # crossing happened in between two bins, right after half2 idx 

print("Peak time: ", t_trial[peak_idx])
print("1st Half: ", np.mean(t_trial[half1_idx:(half1_idx+2)]))
print("2nd Half: ", np.mean(t_trial[half2_idx:(half2_idx+2)]))
print("Half to Half: ", np.mean(t_trial[half2_idx:(half2_idx+2)]) - np.mean(t_trial[half1_idx:(half1_idx+2)]))

half_val = np.max(gauss_mean_odor_state) *0.2
peak_idx = np.argmax(gauss_mean_odor_state)
half1_idx = [i for i in range(gauss_mean_odor_state.shape[0]-1) if (gauss_mean_odor_state[i]<half_val) & (gauss_mean_odor_state[i+1]>=half_val)][0]
# half1_idx = half1_idx +0.5 # crossing happened in between these two bins, right after half1 idx
half2_idx = [i for i in range(gauss_mean_odor_state.shape[0]-1) if (gauss_mean_odor_state[i]>half_val) & (gauss_mean_odor_state[i+1]<=half_val)][0]
# half2_idx = half2_idx +0.5 # crossing happened in between two bins, right after half2 idx 

print("Peak time: ", t_trial[peak_idx])
print("1st Half: ", np.mean(t_trial[half1_idx:(half1_idx+2)]))
print("2nd Half: ", np.mean(t_trial[half2_idx:(half2_idx+2)]))
print("Half to Half: ", np.mean(t_trial[half2_idx:(half2_idx+2)]) - np.mean(t_trial[half1_idx:(half1_idx+2)]))
