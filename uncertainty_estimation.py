import paddle
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt
import data
from main_bayesian import getModel
import config_bayesian as cfg

device = str("cuda:0" if paddle.device.cuda.device_count() >= 1 else "cpu").replace(
    "cuda", "gpu"
)
mnist_set = None
notmnist_set = None
transform = paddle.vision.transforms.Compose(
    [
        paddle.vision.transforms.ToPILImage(),
        paddle.vision.transforms.Resize((32, 32)),
        paddle.vision.transforms.ToTensor(),
    ]
)


def init_dataset(notmnist_dir):
    global mnist_set
    global notmnist_set
    mnist_set, _, _, _ = data.getDataset("MNIST")
    notmnist_set = paddle.vision.datasets.ImageFolder(root=notmnist_dir)


def get_uncertainty_per_image(model, input_image, T=15, normalized=False):
    input_image = input_image.unsqueeze(axis=0)
    input_images = input_image.repeat(T, 1, 1, 1)
    net_out, _ = model(input_images)
    pred = paddle.mean(x=net_out, axis=0).cpu().detach().numpy()
    if normalized:
        prediction = paddle.nn.functional.softplus(x=net_out)
        p_hat = prediction / paddle.sum(x=prediction, axis=1).unsqueeze(axis=1)
    else:
        p_hat = paddle.nn.functional.softmax(x=net_out, axis=1)
    p_hat = p_hat.detach().cpu().numpy()
    p_bar = np.mean(p_hat, axis=0)
    temp = p_hat - np.expand_dims(p_bar, 0)
    epistemic = np.dot(temp.T, temp) / T
    epistemic = np.diag(epistemic)
    aleatoric = np.diag(p_bar) - np.dot(p_hat.T, p_hat) / T
    aleatoric = np.diag(aleatoric)
    return pred, epistemic, aleatoric


def get_uncertainty_per_batch(model, batch, T=15, normalized=False):
    batch_predictions = []
    net_outs = []
    batches = batch.unsqueeze(axis=0).repeat(T, 1, 1, 1, 1)
    preds = []
    epistemics = []
    aleatorics = []
    for i in range(T):
        net_out, _ = model(batches[i])
        net_outs.append(net_out)
        if normalized:
            prediction = paddle.nn.functional.softplus(x=net_out)
            prediction = prediction / paddle.sum(x=prediction, axis=1).unsqueeze(axis=1)
        else:
            prediction = paddle.nn.functional.softmax(x=net_out, axis=1)
        batch_predictions.append(prediction)
    for sample in range(batch.shape[0]):
        pred = paddle.concat(
            x=[a_batch[sample].unsqueeze(axis=0) for a_batch in net_outs], axis=0
        )
        pred = paddle.mean(x=pred, axis=0)
        preds.append(pred)
        p_hat = (
            paddle.concat(
                x=[a_batch[sample].unsqueeze(axis=0) for a_batch in batch_predictions],
                axis=0,
            )
            .detach()
            .cpu()
            .numpy()
        )
        p_bar = np.mean(p_hat, axis=0)
        temp = p_hat - np.expand_dims(p_bar, 0)
        epistemic = np.dot(temp.T, temp) / T
        epistemic = np.diag(epistemic)
        epistemics.append(epistemic)
        aleatoric = np.diag(p_bar) - np.dot(p_hat.T, p_hat) / T
        aleatoric = np.diag(aleatoric)
        aleatorics.append(aleatoric)
    epistemic = np.vstack(epistemics)
    aleatoric = np.vstack(aleatorics)
    preds = paddle.concat(x=[i.unsqueeze(axis=0) for i in preds]).cpu().detach().numpy()
    return preds, epistemic, aleatoric


def get_sample(dataset, sample_type="mnist"):
    idx = np.random.randint(len(dataset.targets))
    if sample_type == "mnist":
        sample = dataset.data[idx]
        truth = dataset.targets[idx]
    else:
        path, truth = dataset.samples[idx]
        sample = paddle.to_tensor(data=np.array(Image.open(path)))
    sample = sample.unsqueeze(axis=0)
    sample = transform(sample)
    return sample.to(device), truth


def run(net_type, weight_path, notmnist_dir):
    init_dataset(notmnist_dir)
    layer_type = cfg.layer_type
    activation_type = cfg.activation_type
    net = getModel(
        net_type,
        1,
        10,
        priors=None,
        layer_type=layer_type,
        activation_type=activation_type,
    )
    net.set_state_dict(state_dict=paddle.load(path=weight_path))
    net.train()
    net.to(device)
    fig = plt.figure()
    fig.suptitle("Uncertainty Estimation", fontsize="x-large")
    mnist_img = fig.add_subplot(321)
    notmnist_img = fig.add_subplot(322)
    epi_stats_norm = fig.add_subplot(323)
    ale_stats_norm = fig.add_subplot(324)
    epi_stats_soft = fig.add_subplot(325)
    ale_stats_soft = fig.add_subplot(326)
    sample_mnist, truth_mnist = get_sample(mnist_set)
    pred_mnist, epi_mnist_norm, ale_mnist_norm = get_uncertainty_per_image(
        net, sample_mnist, T=25, normalized=True
    )
    pred_mnist, epi_mnist_soft, ale_mnist_soft = get_uncertainty_per_image(
        net, sample_mnist, T=25, normalized=False
    )
    mnist_img.imshow(sample_mnist.squeeze().cpu(), cmap="gray")
    mnist_img.axis("off")
    mnist_img.set_title(
        "MNIST Truth: {} Prediction: {}".format(
            int(truth_mnist), int(np.argmax(pred_mnist))
        )
    )
    sample_notmnist, truth_notmnist = get_sample(notmnist_set, sample_type="notmnist")
    pred_notmnist, epi_notmnist_norm, ale_notmnist_norm = get_uncertainty_per_image(
        net, sample_notmnist, T=25, normalized=True
    )
    pred_notmnist, epi_notmnist_soft, ale_notmnist_soft = get_uncertainty_per_image(
        net, sample_notmnist, T=25, normalized=False
    )
    notmnist_img.imshow(sample_notmnist.squeeze().cpu(), cmap="gray")
    notmnist_img.axis("off")
    notmnist_img.set_title(
        "notMNIST Truth: {}({}) Prediction: {}({})".format(
            int(truth_notmnist),
            chr(65 + truth_notmnist),
            int(np.argmax(pred_notmnist)),
            chr(65 + np.argmax(pred_notmnist)),
        )
    )
    x = list(range(10))
    data = pd.DataFrame(
        {
            "epistemic_norm": np.hstack([epi_mnist_norm, epi_notmnist_norm]),
            "aleatoric_norm": np.hstack([ale_mnist_norm, ale_notmnist_norm]),
            "epistemic_soft": np.hstack([epi_mnist_soft, epi_notmnist_soft]),
            "aleatoric_soft": np.hstack([ale_mnist_soft, ale_notmnist_soft]),
            "category": np.hstack([x, x]),
            "dataset": np.hstack([["MNIST"] * 10, ["notMNIST"] * 10]),
        }
    )
    print(data)
    sns.barplot(
        x="category", y="epistemic_norm", hue="dataset", data=data, ax=epi_stats_norm
    )
    sns.barplot(
        x="category", y="aleatoric_norm", hue="dataset", data=data, ax=ale_stats_norm
    )
    epi_stats_norm.set_title("Epistemic Uncertainty (Normalized)")
    ale_stats_norm.set_title("Aleatoric Uncertainty (Normalized)")
    sns.barplot(
        x="category", y="epistemic_soft", hue="dataset", data=data, ax=epi_stats_soft
    )
    sns.barplot(
        x="category", y="aleatoric_soft", hue="dataset", data=data, ax=ale_stats_soft
    )
    epi_stats_soft.set_title("Epistemic Uncertainty (Softmax)")
    ale_stats_soft.set_title("Aleatoric Uncertainty (Softmax)")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PyTorch Uncertainty Estimation b/w MNIST and notMNIST"
    )
    parser.add_argument("--net_type", default="lenet", type=str, help="model")
    parser.add_argument(
        "--weights_path",
        default="checkpoints/MNIST/bayesian/model_lenet.pt",
        type=str,
        help="weights for model",
    )
    parser.add_argument(
        "--notmnist_dir",
        default="data/notMNIST_small/",
        type=str,
        help="weights for model",
    )
    args = parser.parse_args()
    run(args.net_type, args.weights_path, args.notmnist_dir)
