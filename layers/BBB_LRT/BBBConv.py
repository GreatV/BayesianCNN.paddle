import paddle
import sys

sys.path.append("..")
from metrics import calculate_kl as KL_DIV
from ..misc import ModuleWrapper


class BBBConv2d(ModuleWrapper):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        priors=None,
    ):
        super(BBBConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size, kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = 1
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
        out_4 = paddle.create_parameter(
            shape=paddle.empty(
                shape=[out_channels, in_channels, *self.kernel_size]
            ).shape,
            dtype=paddle.empty(shape=[out_channels, in_channels, *self.kernel_size])
            .numpy()
            .dtype,
            default_initializer=paddle.nn.initializer.Assign(
                paddle.empty(shape=[out_channels, in_channels, *self.kernel_size])
            ),
        )
        out_4.stop_gradient = False
        self.W_mu = out_4
        out_5 = paddle.create_parameter(
            shape=paddle.empty(
                shape=[out_channels, in_channels, *self.kernel_size]
            ).shape,
            dtype=paddle.empty(shape=[out_channels, in_channels, *self.kernel_size])
            .numpy()
            .dtype,
            default_initializer=paddle.nn.initializer.Assign(
                paddle.empty(shape=[out_channels, in_channels, *self.kernel_size])
            ),
        )
        out_5.stop_gradient = False
        self.W_rho = out_5
        if self.use_bias:
            out_6 = paddle.create_parameter(
                shape=paddle.to_tensor(data=out_channels).shape,
                dtype="float32",
                default_initializer=paddle.nn.initializer.Assign(
                    paddle.to_tensor(data=out_channels)
                ),
            )
            out_6.stop_gradient = False
            self.bias_mu = out_6
            out_7 = paddle.create_parameter(
                shape=paddle.to_tensor(data=out_channels).shape,
                dtype="float32",
                default_initializer=paddle.nn.initializer.Assign(
                    paddle.to_tensor(data=out_channels)
                ),
            )
            out_7.stop_gradient = False
            self.bias_rho = out_7
        else:
            self.add_parameter(name="bias_mu", parameter=None)
            self.add_parameter(name="bias_rho", parameter=None)
        self.reset_parameters()

    def reset_parameters(self):
        init_normal_mu = paddle.nn.initializer.Normal(
            mean=self.posterior_mu_initial[0], std=self.posterior_mu_initial[1]
        )
        init_normal_rho = paddle.nn.initializer.Normal(
            mean=self.posterior_rho_initial[0], std=self.posterior_rho_initial[1]
        )
        init_normal_mu(self.W_mu)
        init_normal_rho(self.W_rho)
        if self.use_bias:
            init_normal_mu(self.bias_mu)
            init_normal_rho(self.bias_rho)

    def forward(self, x, sample=True):
        self.W_sigma = paddle.log1p(x=paddle.exp(x=self.W_rho))
        if self.use_bias:
            self.bias_sigma = paddle.log1p(x=paddle.exp(x=self.bias_rho))
            bias_var = self.bias_sigma**2
        else:
            self.bias_sigma = bias_var = None
        act_mu = paddle.nn.functional.conv2d(
            x=x,
            weight=self.W_mu,
            bias=self.bias_mu,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        act_var = 1e-16 + paddle.nn.functional.conv2d(
            x=x**2,
            weight=self.W_sigma**2,
            bias=bias_var,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        act_std = paddle.sqrt(x=act_var)
        if self.training or sample:
            eps = paddle.normal(mean=0, std=1, shape=act_mu.shape)
            return act_mu + act_std * eps
        else:
            return act_mu

    def kl_loss(self):
        kl = KL_DIV(self.prior_mu, self.prior_sigma, self.W_mu, self.W_sigma)
        if self.use_bias:
            kl += KL_DIV(self.prior_mu, self.prior_sigma, self.bias_mu, self.bias_sigma)
        return kl
