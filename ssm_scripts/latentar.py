import autograd.numpy as np
import autograd.numpy.random as npr

import ssm
from ssm import hmm, lds
from ssm.hmm import HMM
from ssm.lds import SLDS
from ssm.util import ensure_args_are_lists, softplus, inv_softplus, logistic
from ssm.observations import Observations, AutoRegressiveObservations
from ssm.transitions import Transitions, InputDrivenTransitions
from ssm.init_state_distns import InitialStateDistribution
from ssm.emissions import GaussianEmissions

from autograd.scipy.special import logsumexp, gammaln

class PoissonTransitions(InputDrivenTransitions):
    """
    Hidden Markov Model whose transition probabilities are
    determined by a Poisson GLM applied to the input. 
    """
    def __init__(self, K, D, M, link="softplus", bin_size=1.0, **kwargs):
        super(PoissonTransitions, self).__init__(K, D, M)

        # uniform transitions
        Ps = np.ones((K, K))
        Ps /= Ps.sum(axis=1, keepdims=True)
        self.log_Ps = np.log(Ps)

        self.link_name = link
        self.bin_size = bin_size
        mean_functions = dict(
            log=lambda x: np.exp(x) * self.bin_size,
            softplus= lambda x: softplus(x) * self.bin_size
            )
        self.mean = mean_functions[link]

        link_functions = dict(
            log=lambda rate: np.log(rate) - np.log(self.bin_size),
            softplus=lambda rate: inv_softplus(rate / self.bin_size)
            )
        self.link = link_functions[link]

        # parameters are a vector of weights
        # no other transition parameters
        self.Ws = npr.randn(M) 

    @property
    def params(self):
        return (self.Ws,)

    @params.setter
    def params(self, value):
        self.Ws = value[0]

    def permute(self, perm):
        pass

    def initialize(self, datas, inputs=None, masks=None, tags=None):
        pass

    def log_transition_matrices(self, data, input, mask, tag):
        # assume
        T = data.shape[0]
        assert input.shape[0] == T
        # Previous state effect
        log_Ps = np.tile(self.log_Ps[None, :, :], (T-1, 1, 1))

        # Input effect
        rate = self.mean(np.dot(input[1:], self.Ws.T))[:,None,None]

        # probability of states 1 to K
        # z log lambda - lambda - gammaln(z+1)
        z = np.arange(self.K)
        poisson_log_probs = z[None,None,:] * np.log(rate) - rate - gammaln(z+1)

        """
        TODO -> poisson log probs, the final state, should be p(K>1). so normalizes properly. 
        """

        # Input effect
        log_Ps = log_Ps + poisson_log_probs

        return log_Ps - logsumexp(log_Ps, axis=2, keepdims=True)

    def log_prior(self):
        return 0.0

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        Transitions.m_step(self, expectations, datas, inputs, masks, tags, **kwargs)

    def neg_hessian_expected_log_trans_prob(self, data, input, mask, tag, expected_joints):
        # Return (T-1, D, D) array of blocks for the diagonal of the Hessian
        T, D = data.shape
        return np.zeros((T-1, D, D))

class CalciumObservations(AutoRegressiveObservations):
    def __init__(self, K, D, M, lags=1):
        super(CalciumObservations, self).__init__(K, D, M)

        # set some parameters to 0
        self.Vs *= 0.0 

        # spike count increases

        # Option 1. Directly param
        # self.bs = 4.0 * np.arange(K).reshape((K,1))

        # Option 2. Logistic function
        # self._zs = np.arange(1,K) # set spike count increase for zt = 0 to 0, others sigmoid
        # self.scale = 1.0
        # self.center = 3.0
        # self.c_max = 25.0
        # self.bs = np.concatenate(([0], logistic((self._zs-self.center)/self.scale)*self.c_max))

        # Option 3. cumulative sum of positive numbers, enforces increasing b
        # self._bs = np.log(4.0*np.ones(K-1))
        # self.bs = np.concatenate(([0], np.cumsum(np.exp(self._bs)))).reshape((K,1))

        # Option 4. single scalar of 0 to K-1
        self.scale = 5.0 
        self.bs = self.scale * np.arange(K).reshape((K,1))

        # AR parameter, shared across states
        # alpha = np.clip(0.8 + 0.1*npr.randn(), 0.7, 0.99) 
        alpha = 0.9
        self.alpha = alpha # 
        self.As = self.alpha * np.ones((K, 1, 1))

        # AR variance, shared across states (can generalize)
        inv_eta = np.log(1e-1) 
        self.inv_eta = inv_eta
        self.Sigmas = np.exp(self.inv_eta) * np.ones((K, 1, 1))

    @property
    def params(self):

        # Option 1. 
        # return (self.alpha, self.bs, self.inv_eta)

        # Option 2. 
        # return (self.alpha, self._bs, self.inv_eta, self.scale, self.center, self.c_max) 

        # Option 3
        # return (self.alpha, self._bs, self.inv_eta) 

        # Option 4
        return (self.alpha, self.inv_eta, self.scale)

    @params.setter
    def params(self, value):

        # Option 1. Directly param
        # self.alpha, self.bs, self.inv_eta = value

        # Option 2. Logistic function
        # self.alpha, self._bs, self.inv_eta, self.scale, self.center, self.c_max = value
        # self.alpha, self.inv_eta, self.scale, self.center = value
        # self.bs = np.concatenate(([0], logistic((self._zs-self.center)/self.scale)*self.c_max))
        # self.bs = np.concatenate(([0], logistic((self._bs-self.center)/self.scale)*self.c_max))

        # Option 3. cumulative sum of positive numbers, enforces increasing b
        # self.alpha, self._bs, self.inv_eta = value
        # self.bs = np.concatenate(([0], np.cumsum(np.exp(self._bs)))).reshape((self.K,1))

        # Option 4. single scalar of 0 to K-1
        self.alpha, self.inv_eta, self.scale = value
        self.bs = self.scale * np.arange(self.K).reshape((self.K,1))

        # These are the same for each option!
        self.As = self.alpha * np.ones((self.K, 1, 1))
        self.Sigmas = np.exp(self.inv_eta) * np.ones((self.K, 1, 1))

    def initialize(self, datas, inputs=None, masks=None, tags=None):
        pass

    def m_step(self, expectations, datas, inputs, masks, tags, 
                continuous_expectations=None, **kwargs):
        Observations.m_step(self, expectations, datas, inputs, masks, tags, **kwargs)


class CalciumGaussianEmissions(GaussianEmissions):
    def __init__(self, N, K, D, M=0, single_subspace=True):
        super(CalciumGaussianEmissions, self).__init__(N, K, D, M=M, single_subspace=single_subspace)
        # Make sure the input matrix Fs is set to zero and never updated
        self.Fs *= 0
        self.ds *= 0 # assume baseline subtracted?
        C = np.clip(1.5 + 0.25 * npr.randn(), 0.01, np.inf)
        self.Cs = C.reshape((1,1,1))

        self.inv_etas = np.log(1.0 + 4.0*npr.rand()).reshape((1,1))

    # Construct an emissions model
    @property
    def params(self):
        # return self.Cs, self.ds, self.inv_etas
        return self.Cs, self.inv_etas

    @params.setter
    def params(self, value):
        # self.Cs, self.ds, self.inv_etas = value
        self.Cs, self.inv_etas = value


class LatentCalcium(SLDS):
    def __init__(self, N, K, D, *, M,
            transitions="poisson",
            transition_kwargs=None,
            dynamics_kwargs=None,
            emissions="gaussian",
            emission_kwargs=None,
            single_subspace=True,
            **kwargs):

        init_state_distn = InitialStateDistribution(K, D, M=M)

        transition_kwargs = transition_kwargs or {}
        transitions = PoissonTransitions(K, D, M=M, **transition_kwargs)

        dynamics_kwargs = dynamics_kwargs or {}
        dynamics = CalciumObservations(K, D, M=M, **dynamics_kwargs)

        emission_kwargs = emission_kwargs or {}
        emissions = CalciumGaussianEmissions(N, K, D, M=M,
            single_subspace=single_subspace, **emission_kwargs)

        super().__init__(N, K=K, D=D, M=M,
                            init_state_distn=init_state_distn,
                            transitions=transitions,
                            dynamics=dynamics,
                            emissions=emissions)

