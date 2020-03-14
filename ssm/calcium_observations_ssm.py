def log_poisson_1D(y, rate):
    return y * np.log(rate) - rate - gammaln(y+1)

def log_gaussian_1D(y, mu, sig2):
    return -0.5 * np.log(2.0 * np.pi * sig2) -0.5 * (y - mu)**2 / sig2

class CalciumEmissions(_LinearEmissions):
    def __init__(self, N, K, D, M=0, single_subspace=True, link="log", bin_size=1.0, lags=1, **kwargs):
        super(CalciumEmissions, self).__init__(N, K, D, M, single_subspace=single_subspace, **kwargs)
        assert single_subspace
        # Calcium emissions combines features of Poisson emissions and autoregressive emissions

        # Define Poisson spikes
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

        # Set the bias to be small if using log link
        if link == "log":
            self.ds = -3 + .5 * npr.randn(1, N) 

        # Initialize AR component of the model
        # make these diagonal, do not allow multiple subspace for now
        self.As = np.abs(npr.randn(1, N)) # autoregressive component must be positive!
        self.inv_etas = -4 + npr.randn(1, N) 

        # # Shrink the eigenvalues of the A matrices to avoid instability.
        # # Since the As are diagonal, this is just a clip.
        self.As = np.clip(self.As, -1.0 + 1e-8, 1 - 1e-8)

        # spike part of mean
        self.betas = np.ones((1, N))

        def calcium_log_likelihood(x, y, ytm1, A, beta, inv_etas, C, d, S=10):
            s = np.arange(0,S,1)
            mus = A * ytm1[:,None] + beta[:,None] * s
            sig2s = np.exp(inv_etas)
            rate = self.mean(C@x+d)
            joint = logsumexp( log_gaussian_1D(y[:,None], mus, sig2s) + log_poisson_1D(s, rate[:,None]), axis=1)
            ll = np.sum(joint)
            return ll

        self.hess_ca = hessian(calcium_log_likelihood)

    @property
    def params(self):
        # super(CalciumEmissions, self).params are linear params C, d, F
        # return super(CalciumEmissions, self).params + (self.As, self.inv_etas, self.betas)
        return super(CalciumEmissions, self).params

    @params.setter
    def params(self, value):
        super(CalciumEmissions, self.__class__).params.fset(self, value)

    def log_likelihoods(self, data, input, mask, tag, x, S=10):
        # S is number of spikes to marginalize

        # firing rates         
        # import ipdb; ipdb.set_trace()
        lambdas = self.mean(self.forward(x, input, tag)) 

        # compute autoregressive component of mean
        pad = np.zeros((1, 1, self.N))
        mus = np.concatenate((pad, self.As[None, :, :] * data[:-1, None, :])) 

        # initialize spike count range
        s = np.arange(0,S,1)

        # add spike components to autoregressive mean for each number of spikes
        mus = mus[:,:,:,None] + self.betas[None,:,:,None] * s
        sig2s = np.exp(self.inv_etas) # get noise variances

        # compute log likelihood, marginalizing out the spikes
        outg = log_gaussian_1D(data[:,None,:,None], mus, sig2s[None,:,:,None])
        outp = log_poisson_1D(s[None,None,None,:], lambdas[:,:,:,None])
        # import ipdb; ipdb.set_trace()
        lls = logsumexp(outg + outp, axis=3) # marginalize over unseen spikes here

        # return np.sum(lls) # sum across all neurons and time points 
        return np.sum(lls * mask[:, None, :], axis=2)

    def invert(self, data, input=None, mask=None, tag=None):
        assert self.single_subspace, "Can only invert with a single emission model"
        pad = np.zeros((1, self.N))
        resid = data - np.concatenate((pad, self.As * data[:-1]))
        return self._invert(resid, input=input, mask=mask, tag=tag)

    def sample(self, z, x, input=None, tag=None):
        T, N = z.shape[0], self.N
        z = np.zeros_like(z, dtype=int)

        # firing rates
        # import ipdb; ipdb.set_trace()
        lambdas = self.mean(self.forward(x, input, tag)) 

        # sample spikes
        spikes = npr.poisson(lambdas)
        # import ipdb; ipdb.set_trace()
        # autoregressive parameters
        etas = np.exp(self.inv_etas)
        mus = np.zeros_like(lambdas) # start with zero mean
        y = np.zeros((T, N))
        y[0] = mus[0, z[0], :] + np.sqrt(etas[z[0]]) * npr.randn(N)
        for t in range(1, T):
            y[t] = mus[t, z[t], :] + self.As[z[t]] * y[t-1] + self.betas[z[t]] * spikes[t, z[t], :] + np.sqrt(etas[z[0]]) * npr.randn(N) 
        return y

    def smooth(self, expected_states, variational_mean, data, input=None, mask=None, tag=None, S=10):
        assert self.single_subspace
        # spike rate mean
        lambdas = self.mean(self.forward(variational_mean, input, tag))
        # import ipdb; ipdb.set_trace()
        # compute autoregressive component of mean
        pad = np.zeros((1, 1, self.N))
        mus = np.concatenate((pad, self.As[None, :, :] * data[:-1, None, :])) 

        # marginalize over spikes
        s = np.arange(0,S,1)
        mus = mus[:,:,:,None] + self.betas[None,:,:,None] * s
        outp = log_poisson_1D(s[None,None,None,:], lambdas[:,:,:,None])
        mus = np.sum(mus * np.exp(outp), axis=3)
        return mus[:,0,:]

    def initialize(self, datas, inputs=None, masks=None, tags=None):
        pass 

    # def hessian_log_emissions_prob(self, data, input, mask, tag, x, Ez=None):
    #     assert self.single_subspace
    #     # import ipdb; ipdb.set_trace()
    #     pad = np.zeros((1, self.N))
    #     ytm1s = np.concatenate((pad, data))
    #     hess = [self.hess_ca(xt, yt, ytm1, self.As[0], self.betas[0], self.inv_etas[0], self.Cs[0], self.ds[0]) 
    #                 for xt, yt, ytm1 in zip(x, data, ytm1s)]
    #     return hess