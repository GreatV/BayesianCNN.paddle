import paddle
import numpy as np


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        init_Normal = paddle.nn.initializer.Normal(mean=0, std=1)
        init_Normal(m.weight)
        init_Constant = paddle.nn.initializer.Constant(value=0)
        init_Constant(m.bias)


class AlexNet(paddle.nn.Layer):
    def __init__(self, num_classes, inputs=3):
        super(AlexNet, self).__init__()
        self.features = paddle.nn.Sequential(
            paddle.nn.Conv2D(
                in_channels=inputs, out_channels=64, kernel_size=11, stride=4, padding=5
            ),
            paddle.nn.ReLU(),
            paddle.nn.Dropout(p=0.5),
            paddle.nn.MaxPool2D(kernel_size=2, stride=2),
            paddle.nn.Conv2D(
                in_channels=64, out_channels=192, kernel_size=5, padding=2
            ),
            paddle.nn.ReLU(),
            paddle.nn.MaxPool2D(kernel_size=2, stride=2),
            paddle.nn.Conv2D(
                in_channels=192, out_channels=384, kernel_size=3, padding=1
            ),
            paddle.nn.ReLU(),
            paddle.nn.Dropout(p=0.5),
            paddle.nn.Conv2D(
                in_channels=384, out_channels=256, kernel_size=3, padding=1
            ),
            paddle.nn.ReLU(),
            paddle.nn.Conv2D(
                in_channels=256, out_channels=256, kernel_size=3, padding=1
            ),
            paddle.nn.ReLU(),
            paddle.nn.Dropout(p=0.5),
            paddle.nn.MaxPool2D(kernel_size=2, stride=2),
        )
        self.classifier = paddle.nn.Linear(in_features=256, out_features=num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x
