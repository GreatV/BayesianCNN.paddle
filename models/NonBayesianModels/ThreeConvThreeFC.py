import paddle
from layers.misc import FlattenLayer


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        init_Normal = paddle.nn.initializer.Normal(mean=0, std=1)
        init_Normal(m.weight)
        init_Constant = paddle.nn.initializer.Constant(value=0)
        init_Constant(m.bias)


class ThreeConvThreeFC(paddle.nn.Layer):
    """
    To train on CIFAR-10:
    https://arxiv.org/pdf/1207.0580.pdf
    """

    def __init__(self, outputs, inputs):
        super(ThreeConvThreeFC, self).__init__()
        self.features = paddle.nn.Sequential(
            paddle.nn.Conv2D(
                in_channels=inputs, out_channels=32, kernel_size=5, stride=1, padding=2
            ),
            paddle.nn.Softplus(),
            paddle.nn.MaxPool2D(kernel_size=3, stride=2),
            paddle.nn.Conv2D(
                in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2
            ),
            paddle.nn.Softplus(),
            paddle.nn.MaxPool2D(kernel_size=3, stride=2),
            paddle.nn.Conv2D(
                in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=1
            ),
            paddle.nn.Softplus(),
            paddle.nn.MaxPool2D(kernel_size=3, stride=2),
        )
        self.classifier = paddle.nn.Sequential(
            FlattenLayer(2 * 2 * 128),
            paddle.nn.Linear(in_features=2 * 2 * 128, out_features=1000),
            paddle.nn.Softplus(),
            paddle.nn.Linear(in_features=1000, out_features=1000),
            paddle.nn.Softplus(),
            paddle.nn.Linear(in_features=1000, out_features=outputs),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
