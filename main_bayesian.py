from __future__ import print_function
import paddle
import os
import argparse
import numpy as np
import data
import utils
import metrics
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


def train_model(
    net,
    optimizer,
    criterion,
    trainloader,
    num_ens=1,
    beta_type=0.1,
    epoch=None,
    num_epochs=None,
):
    net.train()
    training_loss = 0.0
    accs = []
    kl_list = []
    for i, (inputs, labels) in enumerate(trainloader, 1):
        optimizer.clear_grad()
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = paddle.zeros(shape=[inputs.shape[0], net.num_classes, num_ens]).to(
            device
        )
        kl = 0.0
        for j in range(num_ens):
            net_out, _kl = net(inputs)
            kl += _kl
            outputs[:, :, j] = paddle.nn.functional.log_softmax(x=net_out, axis=1)
        kl = kl / num_ens
        kl_list.append(kl.item())
        log_outputs = utils.logmeanexp(outputs, dim=2)
        beta = metrics.get_beta(i - 1, len(trainloader), beta_type, epoch, num_epochs)
        loss = criterion(log_outputs, labels, kl, beta)
        loss.backward()
        optimizer.step()
        accs.append(metrics.acc(log_outputs.data, labels))
        training_loss += loss.cpu().data.numpy()
    return training_loss / len(trainloader), np.mean(accs), np.mean(kl_list)


def validate_model(
    net, criterion, validloader, num_ens=1, beta_type=0.1, epoch=None, num_epochs=None
):
    """Calculate ensemble accuracy and NLL Loss"""
    net.train()
    valid_loss = 0.0
    accs = []
    for i, (inputs, labels) in enumerate(validloader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = paddle.zeros(shape=[inputs.shape[0], net.num_classes, num_ens]).to(
            device
        )
        kl = 0.0
        for j in range(num_ens):
            net_out, _kl = net(inputs)
            kl += _kl
            outputs[:, :, j] = paddle.nn.functional.log_softmax(x=net_out, axis=1).data
        log_outputs = utils.logmeanexp(outputs, dim=2)
        beta = metrics.get_beta(i - 1, len(validloader), beta_type, epoch, num_epochs)
        valid_loss += criterion(log_outputs, labels, kl, beta).item()
        accs.append(metrics.acc(log_outputs, labels))
    return valid_loss / len(validloader), np.mean(accs)


def run(dataset, net_type):
    layer_type = cfg.layer_type
    activation_type = cfg.activation_type
    priors = cfg.priors
    train_ens = cfg.train_ens
    valid_ens = cfg.valid_ens
    n_epochs = cfg.n_epochs
    lr_start = cfg.lr_start
    num_workers = cfg.num_workers
    valid_size = cfg.valid_size
    batch_size = cfg.batch_size
    beta_type = cfg.beta_type
    trainset, testset, inputs, outputs = data.getDataset(dataset)
    train_loader, valid_loader, test_loader = data.getDataloader(
        trainset, testset, valid_size, batch_size, num_workers
    )
    net = getModel(net_type, inputs, outputs, priors, layer_type, activation_type).to(
        device
    )
    ckpt_dir = f"checkpoints/{dataset}/bayesian"
    ckpt_name = f"checkpoints/{dataset}/bayesian/model_{net_type}_{layer_type}_{activation_type}.pt"
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)
    criterion = metrics.ELBO(len(trainset)).to(device)
    optimizer = paddle.optimizer.Adam(
        parameters=net.parameters(), learning_rate=lr_start, weight_decay=0.0
    )
    tmp_lr = paddle.optimizer.lr.ReduceOnPlateau(
        patience=6, verbose=True, learning_rate=optimizer.get_lr()
    )
    optimizer.set_lr_scheduler(tmp_lr)
    lr_sched = tmp_lr
    valid_loss_max = np.Inf
    for epoch in range(n_epochs):
        train_loss, train_acc, train_kl = train_model(
            net,
            optimizer,
            criterion,
            train_loader,
            num_ens=train_ens,
            beta_type=beta_type,
            epoch=epoch,
            num_epochs=n_epochs,
        )
        valid_loss, valid_acc = validate_model(
            net,
            criterion,
            valid_loader,
            num_ens=valid_ens,
            beta_type=beta_type,
            epoch=epoch,
            num_epochs=n_epochs,
        )
        lr_sched.step(valid_loss)
        print(
            "Epoch: {} \tTraining Loss: {:.4f} \tTraining Accuracy: {:.4f} \tValidation Loss: {:.4f} \tValidation Accuracy: {:.4f} \ttrain_kl_div: {:.4f}".format(
                epoch, train_loss, train_acc, valid_loss, valid_acc, train_kl
            )
        )
        if valid_loss <= valid_loss_max:
            print(
                "Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...".format(
                    valid_loss_max, valid_loss
                )
            )
            paddle.save(obj=net.state_dict(), path=ckpt_name)
            valid_loss_max = valid_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Bayesian Model Training")
    parser.add_argument("--net_type", default="lenet", type=str, help="model")
    parser.add_argument(
        "--dataset",
        default="MNIST",
        type=str,
        help="dataset = [MNIST/CIFAR10/CIFAR100]",
    )
    args = parser.parse_args()
    run(args.dataset, args.net_type)
