import paddle
import os
import numpy as np
import config_bayesian as cfg

cifar10_classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def logmeanexp(x, dim=None, keepdim=False):
    """Stable computation of log(mean(exp(x))"""
    if dim is None:
        x, dim = x.view(-1), 0
    x_max, _ = paddle.max(x=x, axis=dim, keepdim=True), paddle.argmax(
        x=x, axis=dim, keepdim=True
    )
    x = x_max + paddle.log(
        x=paddle.mean(x=paddle.exp(x=x - x_max), axis=dim, keepdim=True)
    )
    return x if keepdim else x.squeeze(axis=dim)


def adjust_learning_rate(optimizer, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def save_array_to_file(numpy_array, filename):
    file = open(filename, "a")
    shape = " ".join(map(str, numpy_array.shape))
    np.savetxt(file, numpy_array.flatten(), newline=" ", fmt="%.3f")
    file.write("\n")
    file.close()
