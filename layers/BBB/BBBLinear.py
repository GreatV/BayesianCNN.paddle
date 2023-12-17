import paddle
import sys

sys.path.append("..")
from metrics import calculate_kl as KL_DIV
from ..misc import ModuleWrapper


class BBBLinear(ModuleWrapper):
    def __init__(self, in_features, out_features, bias=True, priors=None):
        super(BBBLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.device = str(
            "cuda:0" if paddle.device.cuda.device_count() >= 1 else "cpu"
        ).replace("cuda", "gpu")
        if priors is None:
            priors = {
                "prior_mu": 0,
                "prior_sigma": 0.1,
                "posterior_mu_initial": (0, 0.1),
                "posterior_rho_initial": (-3, 0.1),
            }
        self.prior_mu = priors["prior_mu"]
        self.prior_sigma = priors["prior_sigma"]
        self.posterior_mu_initial = priors["posterior_mu_initial"]
        self.posterior_rho_initial = priors["posterior_rho_initial"]
        out_8 = paddle.create_parameter(
            shape=paddle.empty(shape=(out_features, in_features)).shape,
            dtype=paddle.empty(shape=(out_features, in_features)).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(
                paddle.empty(shape=(out_features, in_features))
            ),
        )
        out_8.stop_gradient = False
        self.W_mu = out_8
        out_9 = paddle.create_parameter(
            shape=paddle.empty(shape=(out_features, in_features)).shape,
            dtype=paddle.empty(shape=(out_features, in_features)).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(
                paddle.empty(shape=(out_features, in_features))
            ),
        )
        out_9.stop_gradient = False
        self.W_rho = out_9
        if self.use_bias:
            out_10 = paddle.create_parameter(
                shape=paddle.empty(shape=out_features).shape,
                dtype=paddle.empty(shape=out_features).numpy().dtype,
                default_initializer=paddle.nn.initializer.Assign(
                    paddle.empty(shape=out_features)
                ),
            )
            out_10.stop_gradient = False
            self.bias_mu = out_10
            out_11 = paddle.create_parameter(
                shape=paddle.empty(shape=out_features).shape,
                dtype=paddle.empty(shape=out_features).numpy().dtype,
                default_initializer=paddle.nn.initializer.Assign(
                    paddle.empty(shape=out_features)
                ),
            )
            out_11.stop_gradient = False
            self.bias_rho = out_11
        else:
            self.add_parameter(name="bias_mu", parameter=None)
            self.add_parameter(name="bias_rho", parameter=None)
        self.reset_parameters()

    def reset_parameters(self):
        self.W_mu.data.normal_(*self.posterior_mu_initial)
        self.W_rho.data.normal_(*self.posterior_rho_initial)
        if self.use_bias:
            self.bias_mu.data.normal_(*self.posterior_mu_initial)
            self.bias_rho.data.normal_(*self.posterior_rho_initial)

    def forward(self, input, sample=True):
        if self.training or sample:
            W_eps = paddle.empty(shape=self.W_mu.shape).normal_(0, 1).to(self.device)
            self.W_sigma = paddle.log1p(x=paddle.exp(x=self.W_rho))
            weight = self.W_mu + W_eps * self.W_sigma
            if self.use_bias:
                bias_eps = (
                    paddle.empty(shape=self.bias_mu.shape).normal_(0, 1).to(self.device)
                )
                self.bias_sigma = paddle.log1p(x=paddle.exp(x=self.bias_rho))
                bias = self.bias_mu + bias_eps * self.bias_sigma
            else:
                bias = None
        else:
            weight = self.W_mu
            bias = self.bias_mu if self.use_bias else None
        return paddle.nn.functional.linear(weight=weight.T, bias=bias, x=input)

    def kl_loss(self):
        kl = KL_DIV(self.prior_mu, self.prior_sigma, self.W_mu, self.W_sigma)
        if self.use_bias:
            kl += KL_DIV(self.prior_mu, self.prior_sigma, self.bias_mu, self.bias_sigma)
        return kl
