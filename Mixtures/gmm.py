import paddle
import numpy as np
from math import pi


class GaussianMixture(paddle.nn.Layer):
    """
    Fits a mixture of k=1,..,K Gaussians to the input data. Input tensors are expected to be flat with dimensions (n: number of samples, d: number of features).
    The model then extends them to (n, k: number of components, d).
    The model parametrization (mu, sigma) is stored as (1, k, d), and probabilities are shaped (n, k, 1) if they relate to an individual sample, or (1, k, 1) if they assign membership probabilities to one of the mixture components.
    """

    def __init__(
        self, n_components, n_features, mu_init=None, var_init=None, eps=1e-06
    ):
        """
        Initializes the model and brings all tensors into their required shape. The class expects data to be fed as a flat tensor in (n, d). The class owns:
            x:              torch.Tensor (n, k, d)
            mu:             torch.Tensor (1, k, d)
            var:            torch.Tensor (1, k, d)
            pi:             torch.Tensor (1, k, 1)
            eps:            float
            n_components:   int
            n_features:     int
            score:          float
        args:
            n_components:   int
            n_features:     int
            mu_init:        torch.Tensor (1, k, d)
            var_init:       torch.Tensor (1, k, d)
            eps:            float
        """
        super(GaussianMixture, self).__init__()
        self.eps = eps
        self.n_components = n_components
        self.n_features = n_features
        self.log_likelihood = -np.inf
        self.mu_init = mu_init
        self.var_init = var_init
        self._init_params()

    def _init_params(self):
        if self.mu_init is not None:
            assert self.mu_init.shape == (
                1,
                self.n_components,
                self.n_features,
            ), "Input mu_init does not have required tensor dimensions (1, %i, %i)" % (
                self.n_components,
                self.n_features,
            )
            out_20 = paddle.create_parameter(
                shape=self.mu_init.shape,
                dtype=self.mu_init.numpy().dtype,
                default_initializer=paddle.nn.initializer.Assign(self.mu_init),
            )
            out_20.stop_gradient = not False
            self.mu = out_20
        else:
            out_21 = paddle.create_parameter(
                shape=paddle.randn(shape=[1, self.n_components, self.n_features]).shape,
                dtype=paddle.randn(shape=[1, self.n_components, self.n_features])
                .numpy()
                .dtype,
                default_initializer=paddle.nn.initializer.Assign(
                    paddle.randn(shape=[1, self.n_components, self.n_features])
                ),
            )
            out_21.stop_gradient = not False
            self.mu = out_21
        if self.var_init is not None:
            assert self.var_init.shape == (1, self.n_components, self.n_features), (
                "Input var_init does not have required tensor dimensions (1, %i, %i)"
                % (self.n_components, self.n_features)
            )
            out_22 = paddle.create_parameter(
                shape=self.var_init.shape,
                dtype=self.var_init.numpy().dtype,
                default_initializer=paddle.nn.initializer.Assign(self.var_init),
            )
            out_22.stop_gradient = not False
            self.var = out_22
        else:
            out_23 = paddle.create_parameter(
                shape=paddle.ones(shape=[1, self.n_components, self.n_features]).shape,
                dtype=paddle.ones(shape=[1, self.n_components, self.n_features])
                .numpy()
                .dtype,
                default_initializer=paddle.nn.initializer.Assign(
                    paddle.ones(shape=[1, self.n_components, self.n_features])
                ),
            )
            out_23.stop_gradient = not False
            self.var = out_23
        out_24 = paddle.create_parameter(
            shape=paddle.empty(shape=[1, self.n_components, 1]).shape,
            dtype=paddle.empty(shape=[1, self.n_components, 1]).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(
                paddle.empty(shape=[1, self.n_components, 1])
            ),
        )
        out_24.stop_gradient = not False
        self.pi = out_24.fill_(value=1.0 / self.n_components)
        self.params_fitted = False

    def bic(self, x):
        """
        Bayesian information criterion for samples x.
        args:
            x:      torch.Tensor (n, d) or (n, k, d)
        returns:
            bic:    float
        """
        n = x.shape[0]
        if len(x.shape) == 2:
            x = x.unsqueeze(axis=1).expand(shape=[n, self.n_components, x.shape[1]])
        bic = -2.0 * self.__score(
            self.pi, self.__p_k(x, self.mu, self.var), sum_data=True
        ) * n + self.n_components * np.log(n)
        return bic

    def fit(self, x, warm_start=False, delta=1e-08, n_iter=1000):
        """
        Public method that fits data to the model.
        args:
            n_iter:     int
            delta:      float
        """
        if not warm_start and self.params_fitted:
            self._init_params()
        if len(x.shape) == 2:
            x = x.unsqueeze(axis=1).expand(
                shape=[x.shape[0], self.n_components, x.shape[1]]
            )
        i = 0
        j = np.inf
        while i <= n_iter and j >= delta:
            log_likelihood_old = self.log_likelihood
            mu_old = self.mu
            var_old = self.var
            self.__em(x)
            self.log_likelihood = self.__score(
                self.pi, self.__p_k(x, self.mu, self.var)
            )
            if self.log_likelihood.abs() == float(
                "Inf"
            ) or self.log_likelihood == float("nan"):
                self.__init__(self.n_components, self.n_features)
            i += 1
            j = self.log_likelihood - log_likelihood_old
            if j <= delta:
                self.__update_mu(mu_old)
                self.__update_var(var_old)
        self.params_fitted = True

    def predict(self, x, probs=False):
        """
        Assigns input data to one of the mixture components by evaluating the likelihood under each. If probs=True returns normalized probabilities of class membership instead.
        args:
            x:          torch.Tensor (n, d) or (n, k, d)
            probs:      bool
        returns:
            y:          torch.LongTensor (n)
        """
        if len(x.shape) == 2:
            x = x.unsqueeze(axis=1).expand(
                shape=[x.shape[0], self.n_components, x.shape[1]]
            )
        p_k = self.__p_k(x, self.mu, self.var)
        if probs:
            return p_k / (p_k.sum(axis=1, keepdim=True) + self.eps)
        else:
            _, predictions = paddle.max(x=p_k, axis=1), paddle.argmax(x=p_k, axis=1)
            return paddle.squeeze(x=predictions).astype(paddle.Tensor)

    def predict_proba(self, x):
        """
        Returns normalized probabilities of class membership.
        args:
            x:          torch.Tensor (n, d) or (n, k, d)
        returns:
            y:          torch.LongTensor (n)
        """
        return self.predict(x, probs=True)

    def score_samples(self, x):
        """
        Computes log-likelihood of data (x) under the current model.
        args:
            x:          torch.Tensor (n, d) or (n, k, d)
        returns:
            score:      torch.LongTensor (n)
        """
        if len(x.shape) == 2:
            x = x.unsqueeze(axis=1).expand(
                shape=[x.shape[0], self.n_components, x.shape[1]]
            )
        score = self.__score(self.pi, self.__p_k(x, self.mu, self.var), sum_data=False)
        return score

    def __p_k(self, x, mu, var):
        """
        Returns a tensor with dimensions (n, k, 1) indicating the likelihood of data belonging to the k-th Gaussian.
        args:
            x:      torch.Tensor (n, k, d)
            mu:     torch.Tensor (1, k, d)
            var:    torch.Tensor (1, k, d)
        returns:
            p_k:    torch.Tensor (n, k, 1)
        """
        mu = mu.expand(shape=[x.shape[0], self.n_components, self.n_features])
        var = var.expand(shape=[x.shape[0], self.n_components, self.n_features])
        exponent = paddle.exp(
            x=-0.5 * paddle.sum(x=(x - mu) * (x - mu) / var, axis=2, keepdim=True)
        )
        prefactor = paddle.rsqrt(
            x=(2.0 * pi) ** self.n_features * paddle.prod(x=var, axis=2, keepdim=True)
            + self.eps
        )
        return prefactor * exponent

    def __e_step(self, pi, p_k):
        """
        Computes weights that indicate the probabilistic belief that a data point was generated by one of the k mixture components. This is the so-called expectation step of the EM-algorithm.
        args:
            pi:         torch.Tensor (1, k, 1)
            p_k:        torch.Tensor (n, k, 1)
        returns:
            weights:    torch.Tensor (n, k, 1)
        """
        weights = pi * p_k
        return paddle.divide(
            x=weights,
            y=paddle.to_tensor(paddle.sum(x=weights, axis=1, keepdim=True) + self.eps),
        )

    def __m_step(self, x, weights):
        """
        Updates the model's parameters. This is the maximization step of the EM-algorithm.
        args:
            x:          torch.Tensor (n, k, d)
            weights:    torch.Tensor (n, k, 1)
        returns:
            pi_new:     torch.Tensor (1, k, 1)
            mu_new:     torch.Tensor (1, k, d)
            var_new:    torch.Tensor (1, k, d)
        """
        n_k = paddle.sum(x=weights, axis=0, keepdim=True)
        pi_new = paddle.divide(
            x=n_k,
            y=paddle.to_tensor(paddle.sum(x=n_k, axis=1, keepdim=True) + self.eps),
        )
        mu_new = paddle.divide(
            x=paddle.sum(x=weights * x, axis=0, keepdim=True),
            y=paddle.to_tensor(n_k + self.eps),
        )
        var_new = paddle.divide(
            x=paddle.sum(x=weights * (x - mu_new) * (x - mu_new), axis=0, keepdim=True),
            y=paddle.to_tensor(n_k + self.eps),
        )
        return pi_new, mu_new, var_new

    def __em(self, x):
        """
        Performs one iteration of the expectation-maximization algorithm by calling the respective subroutines.
        args:
            x:          torch.Tensor (n, k, d)
        """
        weights = self.__e_step(self.pi, self.__p_k(x, self.mu, self.var))
        pi_new, mu_new, var_new = self.__m_step(x, weights)
        self.__update_pi(pi_new)
        self.__update_mu(mu_new)
        self.__update_var(var_new)

    def __score(self, pi, p_k, sum_data=True):
        """
        Computes the log-likelihood of the data under the model.
        args:
            pi:         torch.Tensor (1, k, 1)
            p_k:        torch.Tensor (n, k, 1)
        """
        weights = pi * p_k
        if sum_data:
            return paddle.sum(x=paddle.log(x=paddle.sum(x=weights, axis=1) + self.eps))
        else:
            return paddle.log(x=paddle.sum(x=weights, axis=1) + self.eps)

    def __update_mu(self, mu):
        """
        Updates mean to the provided value.
        args:
            mu:         torch.FloatTensor
        """
        assert mu.shape in [
            (self.n_components, self.n_features),
            (1, self.n_components, self.n_features),
        ], (
            "Input mu does not have required tensor dimensions (%i, %i) or (1, %i, %i)"
            % (self.n_components, self.n_features, self.n_components, self.n_features)
        )
        if mu.shape == (self.n_components, self.n_features):
            self.mu = mu.unsqueeze(axis=0)
        elif mu.shape == (1, self.n_components, self.n_features):
            self.mu.data = mu

    def __update_var(self, var):
        """
        Updates variance to the provided value.
        args:
            var:        torch.FloatTensor
        """
        assert var.shape in [
            (self.n_components, self.n_features),
            (1, self.n_components, self.n_features),
        ], (
            "Input var does not have required tensor dimensions (%i, %i) or (1, %i, %i)"
            % (self.n_components, self.n_features, self.n_components, self.n_features)
        )
        if var.shape == (self.n_components, self.n_features):
            self.var = var.unsqueeze(axis=0)
        elif var.shape == (1, self.n_components, self.n_features):
            self.var.data = var

    def __update_pi(self, pi):
        """
        Updates pi to the provided value.
        args:
            pi:         torch.FloatTensor
        """
        assert pi.size() in [
            (1, self.n_components, 1)
        ], "Input pi does not have required tensor dimensions (%i, %i, %i)" % (
            1,
            self.n_components,
            1,
        )
        self.pi.data = pi
