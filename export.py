from __future__ import print_function
import paddle
import config_bayesian as cfg
from models.BayesianModels.Bayesian3Conv3FC import BBB3Conv3FC
from models.BayesianModels.BayesianAlexNet import BBBAlexNet
from models.BayesianModels.BayesianLeNet import BBBLeNet

device = str("cuda:0" if paddle.device.cuda.device_count() >= 1 else "cpu").replace(
    "cuda", "gpu"
)


def getModel(net_type, inputs, outputs, priors, layer_type, activation_type):
    if net_type == "lenet":
        return BBBLeNet(outputs, inputs, priors, layer_type, activation_type)
    elif net_type == "alexnet":
        return BBBAlexNet(outputs, inputs, priors, layer_type, activation_type)
    elif net_type == "3conv3fc":
        return BBB3Conv3FC(outputs, inputs, priors, layer_type, activation_type)
    else:
        raise ValueError("Network should be either [LeNet / AlexNet / 3Conv3FC")


if __name__ == "__main__":
    layer_type = cfg.layer_type
    activation_type = cfg.activation_type
    priors = cfg.priors

    inputs = 1
    outputs = 10
    net_type = "lenet"
    net = getModel(net_type, inputs, outputs, priors, layer_type, activation_type)
    x = paddle.randn(shape=[256, 1, 32, 32])
    try:
        x = paddle.static.InputSpec.from_tensor(x)
        paddle.jit.save(net, input_spec=(x,), path="./model")
        print("[JIT] paddle.jit.save successed.")
        exit(0)
    except Exception as e:
        print("[JIT] paddle.jit.save failed.")
        raise e
