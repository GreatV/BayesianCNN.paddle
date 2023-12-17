import paddle
import numpy as np


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        init_Normal = paddle.nn.initializer.Normal(mean=0, std=1)
        init_Normal(m.weight)
        init_Constant = paddle.nn.initializer.Constant(value=0)
        init_Constant(m.bias)


class LeNet(paddle.nn.Layer):
    def __init__(self, num_classes, inputs=3):
        super(LeNet, self).__init__()
        self.conv1 = paddle.nn.Conv2D(in_channels=inputs, out_channels=6, kernel_size=5)
        self.conv2 = paddle.nn.Conv2D(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = paddle.nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = paddle.nn.Linear(in_features=120, out_features=84)
        self.fc3 = paddle.nn.Linear(in_features=84, out_features=num_classes)

    def forward(self, x):
        out = paddle.nn.functional.relu(x=self.conv1(x))
        out = paddle.nn.functional.max_pool2d(x=out, kernel_size=2)
        out = paddle.nn.functional.relu(x=self.conv2(out))
        out = paddle.nn.functional.max_pool2d(x=out, kernel_size=2)
        out = out.view(out.shape[0], -1)
        out = paddle.nn.functional.relu(x=self.fc1(out))
        out = paddle.nn.functional.relu(x=self.fc2(out))
        out = self.fc3(out)
        return out
